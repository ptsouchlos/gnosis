//! A note whose content is entirely frontmatter must be retrievable.
//!
//! The failure this guards against is the sharpest one a search tool can
//! have: "I typed the note's exact name and gnosis didn't find it." It was
//! real — a property-only note produced no chunk, so it was absent from every
//! result while `status` still counted it — and 43% of the notes in a public
//! Obsidian vault have that shape.
//!
//! Network-gated (downloads the text model on first run), like the other
//! model-backed tests. Run with:
//!   cargo test --release -p gnosis-cli --test property_only_notes -- --ignored --nocapture

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

/// One property-only note among two ordinary ones, so the property-only note
/// has to be *chosen* rather than being the only thing that could come back.
fn build_vault(name: &str) -> PathBuf {
    let vault =
        std::env::temp_dir().join(format!("gnosis-properties-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&vault);
    std::fs::create_dir_all(&vault).unwrap();
    std::fs::write(vault.join("gnosis.toml"), "").unwrap();

    std::fs::write(
        vault.join("Blade Runner.md"),
        "---\n\
         categories:\n  - \"[[Movies]]\"\n\
         cover: https://example.invalid/some/very/long/cover/image/url.jpg\n\
         genre:\n  - \"[[Sci-fi]]\"\n\
         director:\n  - \"[[Ridley Scott]]\"\n\
         year: 1982\n\
         ---\n",
    )
    .unwrap();
    std::fs::write(
        vault.join("baking.md"),
        "# Baking bread\n\nSourdough starter, flour, hydration, and a long overnight proof.\n",
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

fn search(vault: &Path, query: &str) -> Vec<String> {
    let out = run(vault, &["search", query, "--json"]);
    let hits: Vec<serde_json::Value> =
        serde_json::from_str(out.trim()).expect("search --json output was not valid JSON");
    hits.into_iter()
        .map(|h| {
            Path::new(h["path"].as_str().expect("a hit without a path"))
                .file_name()
                .unwrap()
                .to_string_lossy()
                .to_string()
        })
        .collect()
}

#[test]
#[ignore = "downloads model and runs inference"]
fn a_property_only_note_is_found_by_its_own_name() {
    let vault = build_vault("name");
    let hits = search(&vault, "Blade Runner");
    assert_eq!(
        hits.first().map(String::as_str),
        Some("Blade Runner.md"),
        "searching a note's exact name must find it, got {hits:?}"
    );
    let _ = std::fs::remove_dir_all(&vault);
}

/// The property a person would actually search for. A movie note's
/// `director` is the reason to index frontmatter values rather than only the
/// title.
#[test]
#[ignore = "downloads model and runs inference"]
fn a_property_only_note_is_found_by_a_frontmatter_value() {
    let vault = build_vault("value");
    let hits = search(&vault, "Ridley Scott");
    assert_eq!(
        hits.first().map(String::as_str),
        Some("Blade Runner.md"),
        "a frontmatter value must be searchable, got {hits:?}"
    );
    let _ = std::fs::remove_dir_all(&vault);
}
