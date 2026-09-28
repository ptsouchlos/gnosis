//! End-to-end check that `gnosis eval` scores a real index through the real
//! search path.
//!
//! The important property is not "it prints numbers" but that the numbers
//! respond to retrieval quality: a qrels file whose judgements match what
//! search actually returns must score near 1, and a deliberately wrong one
//! must score 0. A harness that cannot tell a good configuration from a
//! known-bad one proves nothing.
//!
//! Network-gated (downloads the text model on first run), like the other
//! model-backed tests. Run with:
//!   cargo test --release -p gnosis-cli --test eval_harness -- --ignored --nocapture

use std::path::{Path, PathBuf};
use std::process::Command;

fn gnosis_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_gnosis"))
}

fn run(vault: &Path, args: &[&str]) -> String {
    let output = Command::new(gnosis_bin())
        .args(args)
        .current_dir(vault)
        .output()
        .expect("failed to run gnosis");
    assert!(
        output.status.success(),
        "`gnosis {}` failed: {}",
        args.join(" "),
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).expect("stdout was not valid UTF-8")
}

/// A small vault with three clearly distinct topics, so the expected ranking
/// is unambiguous without depending on fine-grained model behaviour.
fn build_vault(name: &str) -> PathBuf {
    let vault = std::env::temp_dir().join(format!("gnosis-eval-{}-{}", std::process::id(), name));
    let _ = std::fs::remove_dir_all(&vault);
    std::fs::create_dir_all(&vault).unwrap();
    std::fs::write(vault.join("gnosis.toml"), "").unwrap();

    std::fs::write(
        vault.join("baking.md"),
        "# Baking bread\n\nSourdough starter, flour, hydration, and a long overnight proof.\n",
    )
    .unwrap();
    std::fs::write(
        vault.join("cycling.md"),
        "# Cycling\n\nGear ratios, cadence, and climbing steep mountain passes on a road bike.\n",
    )
    .unwrap();
    std::fs::write(
        vault.join("databases.md"),
        "# Databases\n\nB-tree indexes, query planners, transactions and write-ahead logging.\n",
    )
    .unwrap();

    run(&vault, &["index", "."]);
    vault
}

fn eval_json(vault: &Path, qrels: &str) -> serde_json::Value {
    eval_json_at(vault, qrels, 10)
}

fn eval_json_at(vault: &Path, qrels: &str, limit: usize) -> serde_json::Value {
    std::fs::write(vault.join("qrels.toml"), qrels).unwrap();
    let limit = limit.to_string();
    let out = run(
        vault,
        &["eval", "--qrels", "qrels.toml", "--json", "--limit", &limit],
    );
    serde_json::from_str(out.trim()).expect("eval --json output was not valid JSON")
}

#[test]
#[ignore = "downloads model and runs inference"]
fn eval_scores_a_correct_qrels_file_highly() {
    let vault = build_vault("good");
    let vault_str = vault.to_string_lossy().to_string();

    let report = eval_json(
        &vault,
        &format!(
            r#"
vault = "{vault_str}"

[[query]]
text = "sourdough starter and overnight proofing"
space = "text"
relevant = [{{ path = "baking.md", grade = 2 }}]

[[query]]
text = "b-tree indexes and query planners"
space = "text"
relevant = [{{ path = "databases.md", grade = 2 }}]
"#
        ),
    );

    let ndcg = report["overall"]["ndcg"].as_f64().unwrap();
    assert!(
        ndcg > 0.9,
        "judgements matching what search returns should score near 1, got {ndcg}"
    );
    assert_eq!(report["overall"]["queries"].as_u64().unwrap(), 2);
    assert!((report["overall"]["mrr"].as_f64().unwrap() - 1.0).abs() < 1e-9);

    let _ = std::fs::remove_dir_all(&vault);
}

/// The self-check the harness is worth nothing without: it must be able to
/// tell a correct configuration from a wrong one.
///
/// Scored at k=1 deliberately. This vault has three notes, so at k=10 every
/// document is retrieved and even a wrong judgement earns rank-based credit
/// (measured: nDCG 0.5). Discrimination only shows up at a cutoff the corpus
/// can actually exceed — itself a useful reminder that these metrics need a
/// corpus larger than the cutoff to mean anything.
#[test]
#[ignore = "downloads model and runs inference"]
fn eval_distinguishes_a_correct_qrels_file_from_a_wrong_one() {
    let vault = build_vault("bad");
    let vault_str = vault.to_string_lossy().to_string();

    let qrels = |judged: &str| {
        format!(
            r#"
vault = "{vault_str}"

[[query]]
text = "sourdough starter and overnight proofing"
space = "text"
relevant = [{{ path = "{judged}", grade = 2 }}]
"#
        )
    };

    let right = eval_json_at(&vault, &qrels("baking.md"), 1);
    let wrong = eval_json_at(&vault, &qrels("cycling.md"), 1);

    let right_ndcg = right["overall"]["ndcg"].as_f64().unwrap();
    let wrong_ndcg = wrong["overall"]["ndcg"].as_f64().unwrap();

    assert_eq!(
        right_ndcg, 1.0,
        "the baking note must be the top hit for a baking query"
    );
    assert_eq!(
        wrong_ndcg, 0.0,
        "the cycling note must not be the top hit for a baking query, got {wrong_ndcg}"
    );
    assert_eq!(wrong["overall"]["mrr"].as_f64().unwrap(), 0.0);

    let _ = std::fs::remove_dir_all(&vault);
}

/// A judged path that isn't in the index must be reported, not silently
/// scored as a miss — otherwise a stale annotation is indistinguishable from
/// a retrieval failure.
#[test]
#[ignore = "downloads model and runs inference"]
fn eval_reports_judged_paths_missing_from_the_index() {
    let vault = build_vault("missing");
    let vault_str = vault.to_string_lossy().to_string();

    let report = eval_json(
        &vault,
        &format!(
            r#"
vault = "{vault_str}"

[[query]]
text = "sourdough starter"
space = "text"
relevant = [
  {{ path = "baking.md", grade = 2 }},
  {{ path = "deleted-note.md", grade = 2 }},
]
"#
        ),
    );

    let missing = report["per_query"][0]["missing"].as_array().unwrap();
    assert_eq!(missing.len(), 1, "expected exactly one missing path");
    assert!(
        missing[0].as_str().unwrap().ends_with("deleted-note.md"),
        "got {missing:?}"
    );

    let _ = std::fs::remove_dir_all(&vault);
}

/// Per-space reporting is what makes a whole-space retrieval failure visible;
/// an overall mean alone can look healthy while one space returns nothing.
#[test]
#[ignore = "downloads model and runs inference"]
fn eval_reports_scores_per_space() {
    let vault = build_vault("spaces");
    let vault_str = vault.to_string_lossy().to_string();

    let report = eval_json(
        &vault,
        &format!(
            r#"
vault = "{vault_str}"

[[query]]
text = "gear ratios and cadence"
space = "text"
relevant = [{{ path = "cycling.md", grade = 2 }}]
"#
        ),
    );

    let by_space = report["by_space"].as_object().unwrap();
    assert!(by_space.contains_key("text"), "got {by_space:?}");
    assert_eq!(by_space["text"]["queries"].as_u64().unwrap(), 1);

    let _ = std::fs::remove_dir_all(&vault);
}
