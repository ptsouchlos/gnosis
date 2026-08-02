use pulldown_cmark::{Event, HeadingLevel, Parser, Tag, TagEnd};

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
            | Event::End(TagEnd::CodeBlock) => {
                if heading_buf.is_none() {
                    current.push('\n');
                }
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
}
