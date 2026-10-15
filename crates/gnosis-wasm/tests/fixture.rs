//! Writes the fixture the node harness checks the built package against.
//!
//! The acceptance question for this crate is whether a ranking computed in
//! wasm, through JS values, is the ranking gnosis computes. Answering it needs
//! both sides of the comparison: this writes the Rust side — the queries and
//! the rankings `query::run_query` produces over an `ArtifactStore` on the
//! host — and `tests/node/parity.mjs` runs the same queries through the built
//! `.wasm` and compares.
//!
//! No embedding model is involved, and that is deliberate rather than a
//! shortcut: the query vectors are chunk vectors the artifact already carries.
//! A client's vectors come from transformers.js and are its own concern
//! (measured separately), while what this has to establish is that the same
//! vector produces the same ranking on the other side of the boundary.
//!
//! ```text
//! GNOSIS_WASM_ARTIFACT=/path/to/exported-artifact \
//!   cargo test -p gnosis-wasm --test fixture -- --ignored --nocapture
//! ```

#![cfg(not(target_arch = "wasm32"))]

mod support;

use std::path::{Path, PathBuf};

use serde::Serialize;
use store::artifact::{Artifact, Manifest};
use store::artifact_store::ArtifactStore;
use store::{Space, TextQuery};

/// One query and the ranking the Rust pipeline produced for it.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase", tag = "kind")]
enum Case {
    Search {
        query: String,
        space: String,
        vector: Vec<f32>,
        limit: usize,
        expected: Vec<Ranked>,
    },
    Lexical {
        query: String,
        limit: usize,
        expected: Vec<Ranked>,
    },
    Related {
        path: String,
        limit: usize,
        include_linked: bool,
        expected: Vec<Ranked>,
    },
}

#[derive(Debug, Serialize)]
struct Ranked {
    path: String,
    score: f32,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct Fixture {
    /// Where the harness should read the artifact from.
    artifact_dir: String,
    documents: usize,
    chunks: usize,
    cases: Vec<Case>,
}

fn ranked(hits: Vec<search::Hit>) -> Vec<Ranked> {
    hits.into_iter()
        .map(|h| Ranked {
            path: h.path,
            score: h.score,
        })
        .collect()
}

/// A single supplied vector, standing in for an embedder.
struct Fixed(Space, Vec<f32>);

impl query::QueryEmbedder for Fixed {
    fn embed_query(&mut self, space: Space, _query: &str) -> anyhow::Result<Vec<f32>> {
        if space != self.0 {
            anyhow::bail!("no vector for '{space}'");
        }
        Ok(self.1.clone())
    }
}

fn read_artifact(dir: &Path) -> (String, Vec<(String, Vec<u8>)>) {
    let manifest_json =
        std::fs::read_to_string(dir.join("manifest.json")).expect("reading manifest.json");
    let manifest = Manifest::from_json(&manifest_json).expect("parsing manifest.json");
    let shards = manifest
        .shards
        .iter()
        .map(|s| {
            let bytes = std::fs::read(dir.join(&s.name))
                .unwrap_or_else(|e| panic!("reading shard {}: {e}", s.name));
            (s.name.clone(), bytes)
        })
        .collect();
    (manifest_json, shards)
}

/// The fixture artifact, for a run with no exported artifact to point at. The
/// harness is then a smoke test of the boundary rather than a measurement.
fn write_fixture_artifact(dir: &Path) -> (String, Vec<(String, Vec<u8>)>) {
    let (manifest, shards) = support::artifact();
    std::fs::create_dir_all(dir).expect("creating the fixture directory");
    std::fs::write(dir.join("manifest.json"), &manifest).expect("writing manifest.json");
    for (name, bytes) in &shards {
        std::fs::write(dir.join(name), bytes).expect("writing a shard");
    }
    (manifest, shards)
}

#[test]
#[ignore = "writes a fixture for the node harness"]
fn write_the_parity_fixture() {
    let out = PathBuf::from(
        std::env::var("GNOSIS_WASM_FIXTURE").unwrap_or_else(|_| "target/wasm-fixture".into()),
    );
    std::fs::create_dir_all(&out).expect("creating the output directory");

    let (artifact_dir, manifest_json, shards) = match std::env::var("GNOSIS_WASM_ARTIFACT") {
        Ok(dir) => {
            let dir = PathBuf::from(dir);
            let (manifest, shards) = read_artifact(&dir);
            (dir, manifest, shards)
        }
        Err(_) => {
            let dir = out.join("artifact");
            let (manifest, shards) = write_fixture_artifact(&dir);
            (dir, manifest, shards)
        }
    };

    let artifact = Artifact::open(&manifest_json, &shards).expect("opening the artifact");
    let documents = artifact.documents().len();
    let total_chunks = artifact.chunks().len();
    let store = ArtifactStore::new(artifact);
    let cfg = search::bm25::SearchConfig::default();
    let filter = TextQuery::default();

    // Spread the queries across the corpus rather than taking a prefix: a
    // contiguous run of chunks is likely to be one document.
    let text_chunks: Vec<(usize, store::artifact::ArtifactChunk)> = store
        .artifact()
        .chunks()
        .iter()
        .enumerate()
        .filter(|(_, c)| c.space == Space::Text && !c.text.is_empty())
        .map(|(i, c)| (i, c.clone()))
        .collect();
    assert!(
        !text_chunks.is_empty(),
        "the artifact holds no text chunks to query with"
    );
    let wanted: usize = std::env::var("GNOSIS_WASM_CASES")
        .ok()
        .and_then(|n| n.parse().ok())
        .unwrap_or(50);
    let stride = (text_chunks.len() / wanted.max(1)).max(1);

    let mut cases: Vec<Case> = Vec::new();
    for (_, chunk) in text_chunks.iter().step_by(stride).take(wanted) {
        // The first few words, which is the shape of a real query and gives
        // the lexical channel something to match.
        let query: String = chunk
            .text
            .split_whitespace()
            .take(10)
            .collect::<Vec<_>>()
            .join(" ");
        if query.is_empty() {
            continue;
        }
        let mut embedder = Fixed(Space::Text, chunk.vector.clone());
        let hits = query::run_query(
            &store,
            &mut embedder,
            &query,
            &[Space::Text],
            10,
            &filter,
            &cfg,
        )
        .expect("running the query");
        cases.push(Case::Search {
            query: query.clone(),
            space: "text".into(),
            vector: chunk.vector.clone(),
            limit: 10,
            expected: ranked(hits),
        });

        let lexical = query::fuse_text_space(&store, &query, Vec::new(), 10, &filter, &cfg)
            .expect("running the lexical channel");
        cases.push(Case::Lexical {
            query,
            limit: 10,
            expected: ranked(lexical),
        });
    }

    // `related` over the same spread of documents: model-free, and the one
    // path where the client and the CLI do identical work.
    let paths: Vec<String> = store
        .artifact()
        .documents()
        .iter()
        .map(|d| d.path.clone())
        .collect();
    let doc_stride = (paths.len() / 10).max(1);
    let spaces: Vec<Space> = Space::ALL
        .iter()
        .copied()
        .filter(|s| store.artifact().manifest().space(*s).is_some())
        .collect();
    for path in paths.iter().step_by(doc_stride).take(10) {
        let Ok(hits) =
            query::run_related(&store, path, &spaces, 10, false, &filter, query::no_aliases)
        else {
            // A document with no chunks in any space (an unretrievable note)
            // is an error on both sides; nothing to compare.
            continue;
        };
        cases.push(Case::Related {
            path: path.clone(),
            limit: 10,
            include_linked: false,
            expected: ranked(hits),
        });
    }

    let fixture = Fixture {
        artifact_dir: artifact_dir
            .canonicalize()
            .unwrap_or(artifact_dir)
            .display()
            .to_string(),
        documents,
        chunks: total_chunks,
        cases,
    };
    let path = out.join("expected.json");
    std::fs::write(&path, serde_json::to_string(&fixture).expect("serializing")).expect("writing");
    println!(
        "wrote {} cases over {documents} documents and {total_chunks} chunks to {}",
        fixture.cases.len(),
        path.display()
    );
}
