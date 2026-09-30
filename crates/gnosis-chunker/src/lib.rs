use pulldown_cmark::{Event, HeadingLevel, Parser, Tag, TagEnd};
use serde::{Deserialize, Serialize};

/// Chunking configuration: target chunk size and overlap, in ~tokens (words).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ChunkConfig {
    pub max_tokens: usize,
    pub overlap: usize,
}

impl Default for ChunkConfig {
    fn default() -> Self {
        Self {
            max_tokens: 384,
            overlap: 64,
        }
    }
}

/// A unit of text to embed, with the heading trail it came from.
#[derive(Debug, Clone)]
pub struct Chunk {
    pub ord: usize,
    /// Breadcrumb of enclosing headings, e.g. "Design > Storage".
    pub heading_path: String,
    pub text: String,
}

/// A contiguous run of body text under a single heading trail.
struct Section {
    heading_path: String,
    text: String,
}

/// Split markdown into heading-delimited sections, then window each section
/// into ~`max_tokens` chunks (token ≈ whitespace word) with `overlap`.
///
/// Token counts are approximate for now.
pub fn chunk_markdown(body: &str, max_tokens: usize, overlap: usize) -> Vec<Chunk> {
    let sections = split_sections(body);

    let mut chunks = Vec::new();
    let mut ord = 0;
    for section in sections {
        for text in window(&section.text, max_tokens, overlap) {
            chunks.push(Chunk {
                ord,
                heading_path: section.heading_path.clone(),
                text,
            });
            ord += 1;
        }
    }
    chunks
}

/// Walk markdown events, accumulating plain text per heading section.
fn split_sections(body: &str) -> Vec<Section> {
    let mut sections = Vec::new();
    let mut stack: Vec<(u8, String)> = Vec::new();
    let mut current = String::new();
    let mut heading_buf: Option<String> = None;

    let flush = |sections: &mut Vec<Section>, stack: &[(u8, String)], text: &mut String| {
        let trimmed = text.trim();
        if !trimmed.is_empty() {
            sections.push(Section {
                heading_path: heading_path(stack),
                text: trimmed.to_string(),
            });
        }
        text.clear();
    };

    for event in Parser::new(body) {
        match event {
            Event::Start(Tag::Heading { .. }) => {
                // A new heading closes the previous section.
                flush(&mut sections, &stack, &mut current);
                heading_buf = Some(String::new());
            }
            Event::End(TagEnd::Heading(level)) => {
                let title = heading_buf.take().unwrap_or_default().trim().to_string();
                let level = heading_level(level);
                // Pop same-or-deeper headings, then push this one.
                stack.retain(|(l, _)| *l < level);
                stack.push((level, title));
            }
            Event::Text(t) | Event::Code(t) => match heading_buf {
                Some(ref mut buf) => buf.push_str(&t),
                None => {
                    current.push_str(&t);
                }
            },
            Event::SoftBreak | Event::HardBreak => {
                if heading_buf.is_none() {
                    current.push(' ');
                }
            }
            Event::End(TagEnd::Paragraph)
            | Event::End(TagEnd::Item)
            | Event::End(TagEnd::CodeBlock)
                if heading_buf.is_none() =>
            {
                current.push('\n');
            }
            _ => {}
        }
    }
    flush(&mut sections, &stack, &mut current);
    sections
}

fn heading_path(stack: &[(u8, String)]) -> String {
    stack
        .iter()
        .map(|(_, t)| t.as_str())
        .filter(|t| !t.is_empty())
        .collect::<Vec<_>>()
        .join(" > ")
}

fn heading_level(level: HeadingLevel) -> u8 {
    match level {
        HeadingLevel::H1 => 1,
        HeadingLevel::H2 => 2,
        HeadingLevel::H3 => 3,
        HeadingLevel::H4 => 4,
        HeadingLevel::H5 => 5,
        HeadingLevel::H6 => 6,
    }
}

/// Split text into overlapping word windows. Returns at least one chunk for
/// non-empty input.
/// Window each page of a PDF into ~`max_tokens` chunks, labelling every chunk
/// with the page it came from.
///
/// Chunks never span a page boundary, mirroring how `chunk_markdown`'s
/// sections never span a heading. `ord` runs continuously across the whole
/// document so it stays a stable per-document ordinal, as it is for markdown.
/// A page whose text is blank — an image-only page in an otherwise digital
/// PDF — contributes nothing rather than an empty chunk.
pub fn chunk_pages(pages: &[(u32, String)], max_tokens: usize, overlap: usize) -> Vec<Chunk> {
    let mut chunks = Vec::new();
    for (page, text) in pages {
        let heading_path = format!("Page {page}");
        for piece in window(text, max_tokens, overlap) {
            chunks.push(Chunk {
                ord: chunks.len(),
                heading_path: heading_path.clone(),
                text: piece,
            });
        }
    }
    chunks
}

fn window(text: &str, max_tokens: usize, overlap: usize) -> Vec<String> {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.is_empty() {
        return Vec::new();
    }
    if words.len() <= max_tokens {
        return vec![words.join(" ")];
    }

    let step = max_tokens.saturating_sub(overlap).max(1);
    let mut chunks = Vec::new();
    let mut start = 0;
    while start < words.len() {
        let end = (start + max_tokens).min(words.len());
        chunks.push(words[start..end].join(" "));
        if end == words.len() {
            break;
        }
        start += step;
    }
    chunks
}

#[cfg(test)]
mod tests {
    #[test]
    fn chunk_config_default_matches_previous_values() {
        let cfg = super::ChunkConfig::default();
        assert_eq!(cfg.max_tokens, 384);
        assert_eq!(cfg.overlap, 64);
    }

    #[test]
    fn test_window() {
        let text = "The quick brown fox jumps over the lazy dog";
        let chunks = super::window(text, 4, 2);
        assert_eq!(
            chunks,
            vec![
                "The quick brown fox",
                "brown fox jumps over",
                "jumps over the lazy",
                "the lazy dog"
            ]
        );
    }

    #[test]
    fn test_split_sections() {
        let md = r#"# Heading 1
Some text under heading 1.
## Heading 2
Some text under heading 2.
"#;
        let sections = super::split_sections(md);
        assert_eq!(sections.len(), 2);
        assert_eq!(sections[0].heading_path, "Heading 1");
        assert_eq!(sections[0].text, "Some text under heading 1.");
        assert_eq!(sections[1].heading_path, "Heading 1 > Heading 2");
        assert_eq!(sections[1].text, "Some text under heading 2.");
    }

    #[test]
    fn test_chunk_markdown() {
        let md = r#"# Heading 1
Some text under heading 1 that is long enough to be split into multiple chunks. It has several sentences and should be divided properly.
## Heading 2
Some text under heading 2 that is also long enough to be split into multiple chunks. It has several sentences and should be divided properly.
"#;
        let chunks = super::chunk_markdown(md, 10, 2);

        assert_eq!(chunks.len(), 6);
        assert_eq!(chunks[0].heading_path, "Heading 1");
        assert_eq!(chunks[1].heading_path, "Heading 1");
        assert_eq!(chunks[2].heading_path, "Heading 1");
        assert_eq!(chunks[3].heading_path, "Heading 1 > Heading 2");
        assert_eq!(chunks[4].heading_path, "Heading 1 > Heading 2");
        assert_eq!(chunks[5].heading_path, "Heading 1 > Heading 2");
    }

    // ---- chunk_pages (PDF) ----------------------------------------------

    #[test]
    fn chunk_pages_tags_each_chunk_with_its_page() {
        let pages = vec![(1u32, "alpha beta".to_string()), (2u32, "gamma delta".to_string())];
        let chunks = super::chunk_pages(&pages, 100, 10);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].heading_path, "Page 1");
        assert_eq!(chunks[1].heading_path, "Page 2");
        assert_eq!(chunks[0].text, "alpha beta");
        assert_eq!(chunks[1].text, "gamma delta");
    }

    #[test]
    fn chunk_pages_never_merges_across_a_page_boundary() {
        // Both pages are short enough to fit in one window, so a windower
        // that concatenated first would produce a single chunk containing
        // text from both pages.
        let pages = vec![(1u32, "alpha".to_string()), (2u32, "beta".to_string())];
        let chunks = super::chunk_pages(&pages, 1000, 0);
        assert_eq!(chunks.len(), 2, "a chunk must never span two pages");
        assert!(!chunks[0].text.contains("beta"));
        assert!(!chunks[1].text.contains("alpha"));
    }

    #[test]
    fn chunk_pages_windows_a_long_page_into_several_chunks() {
        let long = (1..=50).map(|i| format!("w{i}")).collect::<Vec<_>>().join(" ");
        let chunks = super::chunk_pages(&[(7u32, long)], 10, 2);
        assert!(chunks.len() > 1, "a 50-word page must split at max_tokens 10");
        assert!(
            chunks.iter().all(|c| c.heading_path == "Page 7"),
            "every chunk of a page keeps that page's label"
        );
    }

    #[test]
    fn chunk_pages_ord_is_sequential_across_pages() {
        let long = (1..=30).map(|i| format!("w{i}")).collect::<Vec<_>>().join(" ");
        let pages = vec![(1u32, long.clone()), (2u32, long)];
        let chunks = super::chunk_pages(&pages, 10, 2);
        let ords: Vec<usize> = chunks.iter().map(|c| c.ord).collect();
        assert_eq!(
            ords,
            (0..chunks.len()).collect::<Vec<_>>(),
            "ord must run 0..n over the whole document, not restart per page"
        );
    }

    #[test]
    fn chunk_pages_skips_pages_with_no_extractable_text() {
        // An image-only page in an otherwise digital PDF extracts to nothing.
        let pages = vec![
            (1u32, "real text".to_string()),
            (2u32, "   \n  ".to_string()),
            (3u32, "more text".to_string()),
        ];
        let chunks = super::chunk_pages(&pages, 100, 10);
        assert_eq!(chunks.len(), 2, "a blank page contributes no chunks");
        assert_eq!(chunks[0].heading_path, "Page 1");
        assert_eq!(chunks[1].heading_path, "Page 3");
    }

    #[test]
    fn chunk_pages_of_nothing_is_empty() {
        assert!(super::chunk_pages(&[], 100, 10).is_empty());
    }
}
