//! End-to-end validation that an indexed image's actual pixel content (not
//! just a stub row) is retrievable via semantic search. Drives the real
//! compiled `gnosis` binary — this crate has no lib target, so integration
//! tests can't call its internals directly, and `CARGO_BIN_EXE_gnosis` is
//! the standard way Cargo exposes a package's own `[[bin]]` to its tests.
//! Network-gated (downloads the CLIP models on first run), so `#[ignore]`
//! like the model-loading unit tests in `embedder.rs`. Run with:
//!   cargo test --release -p gnosis-cli --test image_search -- --ignored --nocapture

use std::path::PathBuf;
use std::process::Command;

/// Minimal valid 2x1 PNG (red/blue pixel) — same bytes `gnosis-fs`'s and
/// `gnosis-cli`'s own unit tests embed, reused here so the test needs no
/// external fixture file.
const RED_BLUE_PNG: &[u8] = &[
    0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52,
    0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x01, 0x08, 0x02, 0x00, 0x00, 0x00, 0x7B, 0x40, 0xE8,
    0xDD, 0x00, 0x00, 0x00, 0x0F, 0x49, 0x44, 0x41, 0x54, 0x78, 0x9C, 0x63, 0xF8, 0xCF, 0xC0, 0xC0,
    0xC0, 0xF0, 0x1F, 0x00, 0x07, 0x00, 0x01, 0xFF, 0x7E, 0x08, 0xB1, 0xD0, 0x00, 0x00, 0x00, 0x00,
    0x49, 0x45, 0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82,
];

fn gnosis_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_gnosis"))
}

fn run(vault: &std::path::Path, args: &[&str]) -> String {
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

fn search_image_space(vault: &std::path::Path, query: &str) -> Vec<serde_json::Value> {
    let stdout = run(vault, &["search", query, "--in", "image", "--json"]);
    serde_json::from_str(stdout.trim()).expect("search --json output was not valid JSON")
}

#[test]
#[ignore = "downloads CLIP models and runs inference"]
fn indexed_image_content_is_found_by_semantic_search() {
    let vault = std::env::temp_dir().join(format!("gnosis-image-search-test-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&vault);
    std::fs::create_dir_all(&vault).unwrap();

    std::fs::write(vault.join("gnosis.toml"), "[embed.image]\nenabled = true\n").unwrap();
    std::fs::write(vault.join("photo.png"), RED_BLUE_PNG).unwrap();
    std::fs::write(
        vault.join("note.md"),
        "# Unrelated note\n\nA recipe for chocolate cake with cocoa powder and sugar.\n",
    )
    .unwrap();

    run(&vault, &["index", "."]);

    let matching = search_image_space(&vault, "a small red and blue image");
    assert_eq!(matching.len(), 1, "the image must be the only image-space hit");
    let hit = &matching[0];
    assert!(
        hit["path"].as_str().unwrap().ends_with("photo.png"),
        "expected photo.png, got {hit}"
    );
    assert_eq!(hit["width"], 2, "indexed image's actual pixel width must be read correctly");
    assert_eq!(hit["height"], 1, "indexed image's actual pixel height must be read correctly");
    let matching_score = hit["score"].as_f64().unwrap();

    let unrelated = search_image_space(&vault, "a recipe for chocolate cake");
    let unrelated_score = unrelated
        .iter()
        .find(|h| h["path"].as_str().unwrap().ends_with("photo.png"))
        .map(|h| h["score"].as_f64().unwrap())
        .unwrap_or(f64::NEG_INFINITY);

    assert!(
        matching_score > unrelated_score,
        "matching query score ({matching_score}) must beat the unrelated query score ({unrelated_score}) \
         — proves the embedding is semantic, not just present"
    );

    let _ = std::fs::remove_dir_all(&vault);
}
