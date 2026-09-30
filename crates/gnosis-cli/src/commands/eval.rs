//! `gnosis eval` — score the current configuration against a qrels file.
//!
//! Runs every judged query through the *same* retrieval path `gnosis search`
//! uses (`commands::search::run_query`) and reports nDCG@k, recall@k, MRR and
//! unjudged@k, broken out per space. Per-space reporting is the point: an
//! overall mean stays respectable while one space sits at zero, which is
//! exactly how a whole-space retrieval failure can hide.

use std::collections::BTreeMap;
use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use serde::Serialize;

use crate::commands::search::{QueryEmbedders, resolve_spaces, run_query};
use crate::qrels::Qrels;
use crate::store::{Space, SqliteStore, Store, TextQuery};
use crate::workspace::{Workspace, expand_tilde};

/// Score retrieval quality against a qrels file.
#[derive(Debug, clap::Args)]
pub struct EvalArgs {
    /// Path to the qrels file.
    #[arg(long, default_value = "gnosis-eval.toml")]
    pub qrels: PathBuf,
    /// Cutoff for the @k metrics, and how many results to retrieve.
    #[arg(long, default_value_t = 10)]
    pub limit: usize,
    /// Print a line per query, not just the aggregates.
    #[arg(long)]
    pub per_query: bool,
    /// Emit results as JSON.
    #[arg(long)]
    pub json: bool,
}

/// Scores for one query.
#[derive(Debug, Clone, Serialize)]
pub struct QueryScore {
    pub query: String,
    /// Which spaces were searched, as written in the report.
    pub spaces: String,
    pub ndcg: f64,
    pub recall: f64,
    pub mrr: f64,
    pub unjudged: usize,
    /// Judged paths that no indexed document matched. A stale annotation and
    /// a retrieval failure otherwise score identically, and conflating them is
    /// how a harness stops being trusted.
    pub missing: Vec<String>,
}

/// Aggregate over a set of queries.
#[derive(Debug, Clone, Serialize)]
pub struct Aggregate {
    pub queries: usize,
    pub ndcg: f64,
    pub recall: f64,
    pub mrr: f64,
    pub unjudged: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct Report {
    pub k: usize,
    pub text_model: String,
    pub image_model: Option<String>,
    pub overall: Aggregate,
    /// Keyed by space name; a query spanning several spaces contributes to
    /// each one it searched.
    pub by_space: BTreeMap<String, Aggregate>,
    pub per_query: Vec<QueryScore>,
}

fn mean(values: impl Iterator<Item = f64>) -> f64 {
    let mut n = 0usize;
    let mut total = 0.0;
    for v in values {
        total += v;
        n += 1;
    }
    if n == 0 { 0.0 } else { total / n as f64 }
}

fn aggregate(scores: &[QueryScore]) -> Aggregate {
    Aggregate {
        queries: scores.len(),
        ndcg: mean(scores.iter().map(|s| s.ndcg)),
        recall: mean(scores.iter().map(|s| s.recall)),
        mrr: mean(scores.iter().map(|s| s.mrr)),
        unjudged: mean(scores.iter().map(|s| s.unjudged as f64)),
    }
}

pub fn execute(ws: &Workspace, args: EvalArgs) -> Result<()> {
    if !ws.db_path.exists() {
        bail!(
            "no index found at {} — run `gnosis index`",
            ws.db_path.display()
        );
    }
    let qrels = Qrels::load(&args.qrels)?;
    if qrels.queries.is_empty() {
        bail!("{} contains no queries to evaluate", args.qrels.display());
    }

    // Judged paths are vault-relative; the index stores canonical absolute
    // paths, so resolve the vault the same way indexing did.
    let vault = std::fs::canonicalize(expand_tilde(&qrels.vault))
        .with_context(|| format!("resolving qrels vault {}", qrels.vault.display()))?;

    let store = SqliteStore::open(&ws.db_path)?;
    let indexed: std::collections::HashSet<String> = store
        .all_document_meta()?
        .into_iter()
        .map(|(path, _)| path)
        .collect();

    let available = resolve_spaces(ws, &[])?;
    let mut embedders = QueryEmbedders::new(ws);
    let filter = TextQuery {
        from: None,
        tags: None,
    };

    let mut per_query: Vec<QueryScore> = Vec::with_capacity(qrels.queries.len());
    let mut by_space: BTreeMap<String, Vec<QueryScore>> = BTreeMap::new();

    for q in &qrels.queries {
        let spaces = q.spaces(&available)?;
        let judged = q.judged(&vault);
        let missing: Vec<String> = judged
            .iter()
            .map(|j| j.path.clone())
            .filter(|p| !indexed.contains(p))
            .collect();

        let hits = run_query(&store, &mut embedders, &q.text, &spaces, args.limit, &filter)?;
        let ranked: Vec<String> = hits.into_iter().map(|h| h.path).collect();

        let score = QueryScore {
            query: q.text.clone(),
            spaces: space_label(&spaces),
            ndcg: eval::ndcg_at(&ranked, &judged, args.limit),
            recall: eval::recall_at(&ranked, &judged, args.limit),
            mrr: eval::mrr(&ranked, &judged),
            unjudged: eval::unjudged_at(&ranked, &judged, args.limit),
            missing,
        };
        for space in &spaces {
            by_space
                .entry(space.as_str().to_string())
                .or_default()
                .push(score.clone());
        }
        per_query.push(score);
    }

    let report = Report {
        k: args.limit,
        text_model: ws.config.embed.text.model.clone(),
        image_model: ws
            .config
            .embed
            .image
            .enabled
            .then(|| ws.config.embed.image.model.clone()),
        overall: aggregate(&per_query),
        by_space: by_space
            .into_iter()
            .map(|(name, scores)| (name, aggregate(&scores)))
            .collect(),
        per_query,
    };

    if args.json {
        println!("{}", serde_json::to_string(&report)?);
        return Ok(());
    }
    print_report(&report, args.per_query);
    Ok(())
}

fn space_label(spaces: &[Space]) -> String {
    spaces
        .iter()
        .map(|s| s.as_str())
        .collect::<Vec<_>>()
        .join("+")
}

fn print_report(report: &Report, per_query: bool) {
    let k = report.k;
    println!("text model:  {}", report.text_model);
    match &report.image_model {
        Some(m) => println!("image model: {m}"),
        None => println!("image model: (disabled)"),
    }
    println!();

    if per_query {
        println!(
            "{:<44} {:>7} {:>8} {:>8} {:>7} {:>9}",
            "query",
            format!("nDCG@{k}"),
            format!("recall@{k}"),
            "MRR",
            "unjdg",
            "spaces"
        );
        for s in &report.per_query {
            let q: String = s.query.chars().take(43).collect();
            println!(
                "{:<44} {:>7.3} {:>8.3} {:>8.3} {:>7} {:>9}",
                q, s.ndcg, s.recall, s.mrr, s.unjudged, s.spaces
            );
        }
        println!();
    }

    println!(
        "{:<12} {:>7} {:>7} {:>8} {:>8} {:>7}",
        "space",
        "queries",
        format!("nDCG@{k}"),
        format!("recall@{k}"),
        "MRR",
        "unjdg"
    );
    for (name, agg) in &report.by_space {
        println!(
            "{:<12} {:>7} {:>7.3} {:>8.3} {:>8.3} {:>7.1}",
            name, agg.queries, agg.ndcg, agg.recall, agg.mrr, agg.unjudged
        );
    }
    let o = &report.overall;
    println!(
        "{:<12} {:>7} {:>7.3} {:>8.3} {:>8.3} {:>7.1}",
        "overall", o.queries, o.ndcg, o.recall, o.mrr, o.unjudged
    );

    // Stale annotations must be loud: they depress scores identically to a
    // retrieval failure, and silently conflating the two makes every later
    // comparison untrustworthy.
    let missing: Vec<&String> = report
        .per_query
        .iter()
        .flat_map(|s| s.missing.iter())
        .collect();
    if !missing.is_empty() {
        println!("\nwarning: {} judged path(s) are not in the index:", missing.len());
        for path in missing.iter().take(20) {
            println!("  {path}");
        }
        if missing.len() > 20 {
            println!("  … and {} more", missing.len() - 20);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn score(ndcg: f64, recall: f64, mrr: f64, unjudged: usize) -> QueryScore {
        QueryScore {
            query: "q".into(),
            spaces: "text".into(),
            ndcg,
            recall,
            mrr,
            unjudged,
            missing: Vec::new(),
        }
    }

    #[test]
    fn aggregate_averages_each_metric() {
        let agg = aggregate(&[score(1.0, 1.0, 1.0, 0), score(0.0, 0.0, 0.0, 4)]);
        assert_eq!(agg.queries, 2);
        assert!((agg.ndcg - 0.5).abs() < 1e-9);
        assert!((agg.recall - 0.5).abs() < 1e-9);
        assert!((agg.mrr - 0.5).abs() < 1e-9);
        assert!((agg.unjudged - 2.0).abs() < 1e-9);
    }

    #[test]
    fn aggregate_of_nothing_is_zero_not_nan() {
        let agg = aggregate(&[]);
        assert_eq!(agg.queries, 0);
        assert!(agg.ndcg.is_finite(), "an empty mean must not be NaN");
        assert_eq!(agg.ndcg, 0.0);
    }

    #[test]
    fn space_label_joins_every_searched_space() {
        assert_eq!(space_label(&[Space::Text]), "text");
        assert_eq!(space_label(&[Space::Text, Space::Image]), "text+image");
    }
}
