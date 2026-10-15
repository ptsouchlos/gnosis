//! A small artifact built in memory, shared by the wasm surface tests and the
//! real-corpus fixture generator.
//!
//! Built with `store::artifact::write_shard` — the same writer `gnosis export`
//! uses — rather than with a hand-written byte string, so a format change
//! reaches these tests the way it reaches the CLI.

use store::Space;
use store::artifact::{
    ArtifactChunk, Compression, DocumentEntry, Manifest, ModelInfo, Quantization, ShardInfo,
    SpaceInfo, SpaceSpec, shard_hash, write_shard,
};

pub const TEXT_DIM: usize = 4;
pub const IMAGE_DIM: usize = 3;

/// Four documents: three notes and one image, so the fixture covers both
/// spaces, a tag filter, a wikilink and a document with two chunks.
pub fn documents() -> Vec<DocumentEntry> {
    vec![
        DocumentEntry {
            root: 0,
            path: "notes/sourdough.md".into(),
            title: "Sourdough".into(),
            kind: "markdown".into(),
            content_hash: "aa".repeat(32),
            mtime: 1_700_000_000,
            links: vec!["Hydration".into()],
            tags: vec!["baking".into()],
        },
        DocumentEntry {
            root: 0,
            path: "notes/hydration.md".into(),
            title: "Hydration".into(),
            kind: "markdown".into(),
            content_hash: "bb".repeat(32),
            mtime: 1_700_000_100,
            links: Vec::new(),
            tags: vec!["baking".into()],
        },
        DocumentEntry {
            root: 1,
            path: "work/planner.md".into(),
            title: "Query planners".into(),
            kind: "markdown".into(),
            content_hash: "cc".repeat(32),
            mtime: 1_700_000_200,
            links: Vec::new(),
            tags: vec!["databases".into()],
        },
        DocumentEntry {
            root: 0,
            path: "notes/loaf.png".into(),
            title: "loaf".into(),
            kind: "image".into(),
            content_hash: "dd".repeat(32),
            mtime: 1_700_000_300,
            links: Vec::new(),
            tags: Vec::new(),
        },
    ]
}

fn chunk(
    doc_index: usize,
    ord: u32,
    space: Space,
    modality: &str,
    heading_path: &str,
    text: &str,
    vector: Vec<f32>,
) -> ArtifactChunk {
    ArtifactChunk {
        doc_index,
        ord,
        space,
        modality: modality.into(),
        heading_path: heading_path.into(),
        text: text.into(),
        vector: unit(vector),
    }
}

/// The artifact stores unit-norm vectors, and the pipeline's dense
/// normalization assumes it — a fixture with un-normalized vectors would
/// score above the ceiling the scorer normalizes against.
fn unit(mut v: Vec<f32>) -> Vec<f32> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in &mut v {
            *x /= norm;
        }
    }
    v
}

/// Text vectors are deliberately mutually orthogonal, one note per axis, so a
/// query vector can be aimed at one note (to test the dense channel alone) or
/// held equidistant from all of them (to let the lexical channel decide). A
/// fixture whose dense scores already order the corpus correctly cannot show
/// whether fusion does anything.
pub fn chunks() -> Vec<ArtifactChunk> {
    vec![
        chunk(
            0,
            0,
            Space::Text,
            "text",
            "Sourdough",
            "A wild yeast starter, a long overnight proof, and an open crumb.",
            vec![1.0, 0.0, 0.0, 0.0],
        ),
        chunk(
            0,
            1,
            Space::Text,
            "text",
            "Sourdough > Feeding",
            "Refresh the starter at a one to two to two ratio and wait for peak.",
            vec![0.95, 0.05, 0.0, 0.0],
        ),
        chunk(
            1,
            0,
            Space::Text,
            "text",
            "Hydration",
            "Baker's percentages: water as a share of flour weight in a wet dough.",
            vec![0.0, 1.0, 0.0, 0.0],
        ),
        chunk(
            2,
            0,
            Space::Text,
            "text",
            "Query planners",
            "A planner estimates cardinality to choose between an index scan and a seq scan.",
            vec![0.0, 0.0, 1.0, 0.0],
        ),
        chunk(
            3,
            0,
            Space::Image,
            "image",
            String::new().as_str(),
            "",
            vec![1.0, 0.2, 0.0],
        ),
        // A title proxy: CLIP-text over a *note's* title, which exists so
        // `related` can cross from an image to the note and back, and which
        // image *search* must never return.
        //
        // It hangs off a different document than the image on purpose. With
        // both on one document, `search` returns the same path either way —
        // it keeps the best chunk per document — so a lost modality filter
        // would be invisible. That exact trap hid a missing filter once
        // already.
        chunk(
            0,
            2,
            Space::Image,
            "text_title",
            "Sourdough",
            "Sourdough",
            vec![0.2, 1.0, 0.0],
        ),
    ]
}

fn space_info(
    space: Space,
    id: &str,
    query_id: Option<&str>,
    pooling: &str,
    dim: usize,
) -> SpaceInfo {
    SpaceInfo {
        space,
        model: ModelInfo {
            id: id.into(),
            pooling: pooling.into(),
            dim,
            query_id: query_id.map(str::to_string),
        },
    }
}

/// The fixture as a client receives it: a manifest as JSON, and shards by name.
pub fn artifact() -> (String, Vec<(String, Vec<u8>)>) {
    let documents = documents();
    let chunks = chunks();
    let specs = vec![
        SpaceSpec {
            space: Space::Text,
            dim: TEXT_DIM,
        },
        SpaceSpec {
            space: Space::Image,
            dim: IMAGE_DIM,
        },
    ];
    let bytes = write_shard(
        &documents,
        &chunks,
        &specs,
        Quantization::F32,
        Compression::Deflate,
    )
    .expect("writing the fixture shard");

    let manifest = Manifest {
        format_version: store::artifact::ARTIFACT_FORMAT_VERSION,
        created_unix: 1_700_000_400,
        spaces: vec![
            space_info(Space::Text, "bge-small-en-v1.5", None, "cls", TEXT_DIM),
            space_info(
                Space::Image,
                "clip-vit-b-32-vision",
                Some("clip-vit-b-32-text"),
                "mean",
                IMAGE_DIM,
            ),
        ],
        quantization: Quantization::F32,
        compression: Compression::Deflate,
        roots: vec!["notes".into(), "work".into()],
        shard_docs: 256,
        shards: vec![ShardInfo {
            name: "shard-0000.bin".into(),
            hash: shard_hash(&bytes),
            documents: documents.len(),
            chunks: chunks.len(),
        }],
    };
    (
        manifest.to_json().expect("serializing the manifest"),
        vec![("shard-0000.bin".to_string(), bytes)],
    )
}
