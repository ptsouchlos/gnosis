//! The bindings, exercised through JS values in a real JS engine.
//!
//! wasm-only, and not for want of trying to share them with the host: every
//! call here takes or returns a `JsValue`, so there is nothing to run without
//! an engine. Run them with node:
//!
//! ```text
//! CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUNNER=wasm-bindgen-test-runner \
//!   cargo test -p gnosis-wasm --target wasm32-unknown-unknown
//! ```
//!
//! What they are for is the boundary itself — the conversions, the filters,
//! the error messages, and the fact that the ranking decisions reached through
//! JS are the ones the pipeline makes. The rankings themselves are measured
//! against the database elsewhere (`ArtifactStore`'s parity tests); repeating
//! that here would only re-test the same Rust.

#![cfg(target_arch = "wasm32")]

mod support;

use js_sys::{Float32Array, Map, Uint8Array};
use wasm_bindgen::JsValue;
use wasm_bindgen_test::*;

use gnosis_wasm::Index;

fn open() -> Index {
    let (manifest, shards) = support::artifact();
    Index::open(&manifest, &shard_map(&shards)).expect("opening the fixture")
}

fn shard_map(shards: &[(String, Vec<u8>)]) -> Map {
    let map = Map::new();
    for (name, bytes) in shards {
        map.set(
            &JsValue::from_str(name),
            &Uint8Array::from(bytes.as_slice()),
        );
    }
    map
}

fn vectors(entries: &[(&str, &[f32])]) -> Map {
    let map = Map::new();
    for (space, vector) in entries {
        map.set(&JsValue::from_str(space), &Float32Array::from(*vector));
    }
    map
}

/// Hit paths, in rank order, from whatever the binding returned.
fn paths(value: JsValue) -> Vec<String> {
    let hits: Vec<serde_json::Value> = serde_json::from_str(&String::from(
        js_sys::JSON::stringify(&value).expect("stringifying hits"),
    ))
    .expect("hits were not an array");
    hits.into_iter()
        .map(|h| {
            h["path"]
                .as_str()
                .expect("a hit without a path")
                .to_string()
        })
        .collect()
}

fn json(value: JsValue) -> serde_json::Value {
    serde_json::from_str(&String::from(
        js_sys::JSON::stringify(&value).expect("stringifying"),
    ))
    .expect("not valid JSON")
}

fn message(err: JsValue) -> String {
    js_sys::Reflect::get(&err, &JsValue::from_str("message"))
        .ok()
        .and_then(|m| m.as_string())
        .unwrap_or_else(|| format!("{err:?}"))
}

/// A client needs the query model and the pooling, not just the model that
/// built the index: for the image space they are different models, and a
/// guessed pooling throws away most of a vector's usable range while every
/// number still looks healthy.
#[wasm_bindgen_test]
fn info_reports_each_space_s_query_model_and_pooling() {
    let info = json(open().info().expect("info"));
    assert_eq!(info["documents"], 4);
    assert_eq!(info["chunks"], 6);
    assert_eq!(info["compression"], "deflate");
    assert_eq!(info["roots"][1], "work");

    let spaces = info["spaces"].as_array().expect("spaces");
    assert_eq!(spaces.len(), 2);
    assert_eq!(spaces[0]["space"], "text");
    assert_eq!(spaces[0]["queryModel"], "bge-small-en-v1.5");
    assert_eq!(spaces[0]["pooling"], "cls");
    assert_eq!(spaces[0]["dim"], 4);
    assert_eq!(spaces[1]["space"], "image");
    assert_eq!(
        spaces[1]["model"], "clip-vit-b-32-vision",
        "the stored vectors came from the vision tower"
    );
    assert_eq!(
        spaces[1]["queryModel"], "clip-vit-b-32-text",
        "a text query goes through the text tower instead"
    );
}

/// A shard that does not match the manifest must not open. Half an artifact
/// would answer queries from a mixture of two indexes, which is worse than
/// refusing: nothing downstream could tell.
#[wasm_bindgen_test]
fn a_tampered_shard_refuses_to_open() {
    let (manifest, mut shards) = support::artifact();
    let last = shards[0].1.len() - 1;
    shards[0].1[last] ^= 0xff;
    let Err(err) = Index::open(&manifest, &shard_map(&shards)) else {
        panic!("a shard whose hash does not match the manifest must not open");
    };
    let msg = message(err.into());
    assert!(msg.contains("partially synced"), "got {msg}");
}

/// The lexical channel reaches through the boundary: a document the dense
/// vector points nowhere near must still rank first on its text.
///
/// Presence is not the assertion to make here. Brute-force dense retrieval
/// over a four-document fixture returns everything within any sane limit, so
/// every document is always *present*; what fusion changes is the order.
#[wasm_bindgen_test]
fn fusion_puts_a_lexical_only_match_first() {
    let index = open();
    // Equidistant from all three notes, so the dense channel has no opinion
    // and anything that orders the results came from the lexical side.
    let hits = paths(
        index
            .search(
                "cardinality",
                &vectors(&[("text", &[0.5773503, 0.5773503, 0.5773503, 0.0])]),
                10,
                JsValue::UNDEFINED,
            )
            .expect("search"),
    );
    assert_eq!(
        hits.first().map(String::as_str),
        Some("work/planner.md"),
        "the lexical channel did not lift the only document that matches the query: {hits:?}"
    );
}

/// Switching the lexical channel off must actually switch it off — the same
/// query then cannot reach a document the vector does not point at.
#[wasm_bindgen_test]
fn the_search_config_crosses_the_boundary() {
    let mut index = open();
    let cfg = js_sys::JSON::parse(r#"{"lexical":false}"#).expect("config");
    index.set_search_config(cfg).expect("setting the config");

    // The config crosses as the Rust struct serializes — snake_case, no
    // renaming — so a client sends back exactly what it read.
    let roundtrip = json(index.search_config().expect("reading the config"));
    assert_eq!(roundtrip["lexical"], false);
    assert_eq!(
        roundtrip["dense_weight"].as_f64(),
        Some(0.9),
        "an override must leave the measured defaults alone: {roundtrip}"
    );

    // Aimed at the sourdough note instead, so with the lexical channel off the
    // document the query is *about* has no way to climb.
    let hits = paths(
        index
            .search(
                "cardinality",
                &vectors(&[("text", &[0.9950372, 0.0995037, 0.0, 0.0])]),
                10,
                JsValue::UNDEFINED,
            )
            .expect("search"),
    );
    assert_eq!(
        hits.last().map(String::as_str),
        Some("work/planner.md"),
        "with the lexical channel off, nothing but the vector decides, and the vector \
         points away from this document: {hits:?}"
    );
}

/// Image search must not return the title proxy, whatever the query vector
/// points at. The proxy exists so `related` can cross from an image to a
/// note; returning it as an image result is the P0 defect that made image
/// search look empty.
#[wasm_bindgen_test]
fn image_search_never_returns_a_title_proxy() {
    let index = open();
    // Pointed straight at the title proxy's own vector.
    let hits = paths(
        index
            .search(
                "loaf",
                &vectors(&[("image", &[0.2, 1.0, 0.0])]),
                10,
                JsValue::UNDEFINED,
            )
            .expect("search"),
    );
    assert_eq!(hits, vec!["notes/loaf.png".to_string()], "got {hits:?}");
}

/// A query vector of the wrong width is not a worse query, it is a
/// meaningless one, so it fails loudly and says both numbers.
#[wasm_bindgen_test]
fn a_wrong_sized_query_vector_is_refused() {
    let err = open()
        .search(
            "x",
            &vectors(&[("text", &[1.0, 0.0])]),
            10,
            JsValue::UNDEFINED,
        )
        .expect_err("a 2-dimensional query must not run against a 4-dimensional index");
    let msg = message(err.into());
    assert!(msg.contains("2 dimensions"), "got {msg}");
    assert!(msg.contains('4'), "got {msg}");
}

/// The model-free tier: no vector, no download, still a ranking.
#[wasm_bindgen_test]
fn lexical_search_needs_no_model() {
    let hits = paths(
        open()
            .lexical_search("baker's percentages", 10, JsValue::UNDEFINED)
            .expect("lexical search"),
    );
    assert_eq!(
        hits.first().map(String::as_str),
        Some("notes/hydration.md"),
        "got {hits:?}"
    );
}

/// `related` excludes the source and what it already links to — the point is
/// unlinked mentions — and `include_linked` brings them back.
#[wasm_bindgen_test]
fn related_excludes_the_source_and_its_links() {
    let index = open();
    let hits = paths(
        index
            .related("notes/sourdough.md", 10, false, JsValue::UNDEFINED)
            .expect("related"),
    );
    assert!(
        !hits.contains(&"notes/sourdough.md".to_string()),
        "got {hits:?}"
    );
    assert!(
        !hits.contains(&"notes/hydration.md".to_string()),
        "[[Hydration]] is already linked: {hits:?}"
    );

    let linked = paths(
        index
            .related("notes/sourdough.md", 10, true, JsValue::UNDEFINED)
            .expect("related"),
    );
    assert!(
        linked.contains(&"notes/hydration.md".to_string()),
        "got {linked:?}"
    );
}

/// Filters cross the boundary as a plain object, and `from` matches a root
/// *label* — an artifact records what a vault was called, not where it lived.
#[wasm_bindgen_test]
fn filters_match_a_root_label_and_a_tag() {
    let index = open();
    let from_work = js_sys::JSON::parse(r#"{"from":["work"]}"#).expect("filter");
    let hits = paths(
        index
            .lexical_search("a planner or a starter", 10, from_work)
            .expect("lexical search"),
    );
    assert_eq!(hits, vec!["work/planner.md".to_string()], "got {hits:?}");

    let baking = js_sys::JSON::parse(r#"{"tags":["baking"]}"#).expect("filter");
    let hits = paths(
        index
            .lexical_search("a planner or a starter", 10, baking)
            .expect("lexical search"),
    );
    assert!(
        !hits.contains(&"work/planner.md".to_string()),
        "got {hits:?}"
    );
}

/// Staleness is a report, not a repair: the client walks its vault, this says
/// what the artifact disagrees with, and only the machine with the models can
/// act on it.
#[wasm_bindgen_test]
fn staleness_separates_changed_added_removed_and_unknown() {
    let entries = js_sys::JSON::parse(&format!(
        r#"[
            {{"path":"notes/sourdough.md","hash":"{}"}},
            {{"path":"notes/hydration.md","hash":"{}"}},
            {{"path":"work/planner.md","mtime":1700009999}},
            {{"path":"notes/loaf.png"}},
            {{"path":"notes/brand-new.md","hash":"ff"}}
        ]"#,
        "aa".repeat(32),
        "00".repeat(32),
    ))
    .expect("entries");

    let report = json(open().staleness(entries).expect("staleness"));
    assert_eq!(report["createdUnix"], 1_700_000_400_i64);
    assert_eq!(report["unchanged"], 1, "only sourdough's hash matched");
    assert_eq!(report["changed"].as_array().unwrap().len(), 2);
    assert_eq!(report["added"][0], "notes/brand-new.md");
    assert!(
        report["removed"].as_array().unwrap().is_empty(),
        "every indexed document was listed: {report}"
    );
    assert_eq!(
        report["undetermined"][0], "notes/loaf.png",
        "an entry with neither a hash nor an mtime cannot be called fresh: {report}"
    );
}

/// A document the vault no longer has is reported as removed rather than
/// quietly kept: a client showing it would link to nothing.
#[wasm_bindgen_test]
fn staleness_reports_a_document_the_vault_no_longer_has() {
    let entries =
        js_sys::JSON::parse(r#"[{"path":"notes/sourdough.md","hash":"ff"}]"#).expect("entries");
    let report = json(open().staleness(entries).expect("staleness"));
    let removed = report["removed"].as_array().expect("removed");
    assert_eq!(removed.len(), 3, "got {report}");
    assert!(
        removed.iter().any(|p| p == "notes/loaf.png"),
        "got {report}"
    );
}
