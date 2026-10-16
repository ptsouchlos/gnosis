//! `similarities` must say what `related` says, pair by pair.
//!
//! A client draws a map of related notes from the pair scores, beside the
//! list `related` returns. If the two disagreed, the map would place a note
//! somewhere its own list contradicts. So every pair `related` lists has to
//! score the same number here, and every pair it would never list has none.

#![cfg(not(target_arch = "wasm32"))]

mod support;

use store::artifact::Artifact;
use store::artifact_store::ArtifactStore;
use store::{Space, TextQuery};

fn store() -> ArtifactStore {
    let (manifest, shards) = support::artifact();
    ArtifactStore::new(Artifact::open(&manifest, &shards).expect("opening the fixture"))
}

#[test]
fn every_pair_scores_what_related_lists_for_it() {
    let store = store();
    let paths: Vec<String> = store
        .artifact()
        .documents()
        .iter()
        .map(|d| d.path.clone())
        .collect();
    let matrix = store
        .related_similarities(&paths, Space::Text)
        .expect("similarities");
    assert_eq!(matrix.len(), paths.len());

    let mut scored = 0;
    for (i, from) in paths.iter().enumerate() {
        assert_eq!(matrix[i].len(), paths.len());
        assert_eq!(matrix[i][i], None, "{from} is never related to itself");
        // Everything, linked or not, so every pair related would score is listed.
        let listed = query::run_related(
            &store,
            from,
            &[Space::Text],
            usize::MAX,
            true,
            &TextQuery::default(),
            query::no_aliases,
        );
        let Ok(listed) = listed else {
            assert!(
                matrix[i].iter().all(Option::is_none),
                "{from} has no text chunks"
            );
            continue;
        };
        for (j, to) in paths.iter().enumerate().filter(|(j, _)| *j != i) {
            let expected = listed.iter().find(|h| &h.path == to).map(|h| h.score);
            assert_eq!(matrix[i][j], expected, "{from} -> {to}");
            scored += usize::from(expected.is_some());
        }
    }
    assert!(scored > 0, "the fixture should relate some pair");
}

#[test]
fn a_path_the_index_does_not_hold_is_named() {
    let err = store()
        .related_similarities(
            &["notes/sourdough.md".into(), "notes/nowhere.md".into()],
            Space::Text,
        )
        .expect_err("an unknown path");
    assert!(err.to_string().contains("notes/nowhere.md"), "{err}");
}
