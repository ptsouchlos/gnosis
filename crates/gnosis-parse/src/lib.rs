pub mod pdf;
pub use pdf::{ParsedPdf, parse_pdf};

use std::path::Path;

/// The result of parsing a markdown document.
#[derive(Debug)]
pub struct ParsedDoc {
    /// Best-effort document title.
    pub title: String,
    /// Raw YAML frontmatter block, if present (without the `---` fences).
    pub frontmatter: Option<String>,
    /// Body markdown with the frontmatter stripped.
    pub body: String,
    /// Obsidian `[[wikilink]]` targets found in the body (alias/heading
    /// stripped) — includes embed targets too (see `embeds`), unchanged
    /// from before embeds were distinguished.
    pub links: Vec<String>,
    /// The subset of `links` that were `![[embed]]` transclusions rather
    /// than plain `[[link]]` references.
    pub embeds: Vec<String>,
    /// Tags from frontmatter `tags:` and inline `#tag` occurrences in the
    /// body (deduplicated, `#`-prefix stripped).
    pub tags: Vec<String>,
}

/// Parse markdown text: split frontmatter, derive a title, collect
/// wikilinks/embeds/tags.
pub fn parse_markdown(path: &Path, content: &str) -> ParsedDoc {
    let (frontmatter, body) = split_frontmatter(content);
    let title = derive_title(path, frontmatter, body);
    let (links, embeds) = extract_wikilinks(body);
    let tags = extract_tags(frontmatter, body);

    ParsedDoc {
        title,
        frontmatter: frontmatter.map(str::to_string),
        body: body.to_string(),
        links,
        embeds,
        tags,
    }
}

/// Split a leading `---` ... `---` YAML frontmatter block from the body.
/// Returns (frontmatter_without_fences, remaining_body).
fn split_frontmatter(content: &str) -> (Option<&str>, &str) {
    let rest = match content.strip_prefix("---\n") {
        Some(r) => r,
        None => return (None, content),
    };

    // Find the closing fence at the start of a line.
    let mut search_from = 0;
    while let Some(rel) = rest[search_from..].find("\n---") {
        let idx = search_from + rel;
        let after = &rest[idx + 4..];
        // The closing fence line must end (newline or EOF) right after `---`.
        if after.is_empty() || after.starts_with('\n') {
            let fm = &rest[..idx];
            let body = after.strip_prefix('\n').unwrap_or(after);
            return (Some(fm), body);
        }
        search_from = idx + 4;
    }

    // Unterminated frontmatter: treat the whole thing as body.
    (None, content)
}

/// Title precedence: frontmatter `title:` → first H1 → file stem.
fn derive_title(path: &Path, frontmatter: Option<&str>, body: &str) -> String {
    if let Some(fm) = frontmatter
        && let Some(title) = frontmatter_title(fm)
    {
        return title;
    }
    if let Some(h1) = first_h1(body) {
        return h1;
    }
    path.file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("untitled")
        .to_string()
}

/// Extract a `title:` value from a YAML frontmatter block (simple line scan).
fn frontmatter_title(fm: &str) -> Option<String> {
    for line in fm.lines() {
        if let Some(rest) = line.trim().strip_prefix("title:") {
            let value = rest.trim().trim_matches(['"', '\'']).trim();
            if !value.is_empty() {
                return Some(value.to_string());
            }
        }
    }
    None
}

/// First ATX H1 heading (`# Title`) in the body, outside fenced code blocks.
fn first_h1(body: &str) -> Option<String> {
    let mut in_code = false;
    for line in body.lines() {
        let trimmed = line.trim_start();
        if trimmed.starts_with("```") || trimmed.starts_with("~~~") {
            in_code = !in_code;
            continue;
        }
        if !in_code && let Some(rest) = trimmed.strip_prefix("# ") {
            let title = rest.trim();
            if !title.is_empty() {
                return Some(title.to_string());
            }
        }
    }
    None
}

/// Collect `[[wikilink]]` targets, stripping `|alias` and `#heading` parts.
/// Returns `(links, embeds)` — `links` is every target regardless of a
/// leading `!` (unchanged from before embeds were distinguished); `embeds`
/// is just the `![[...]]` subset.
fn extract_wikilinks(body: &str) -> (Vec<String>, Vec<String>) {
    let mut links = Vec::new();
    let mut embeds = Vec::new();
    let bytes = body.as_bytes();
    let mut i = 0;
    while i + 1 < bytes.len() {
        if bytes[i] == b'['
            && bytes[i + 1] == b'['
            && let Some(end) = body[i + 2..].find("]]")
        {
            let inner = &body[i + 2..i + 2 + end];
            let target = inner
                .split('|')
                .next()
                .unwrap_or(inner)
                .split('#')
                .next()
                .unwrap_or(inner)
                .trim();
            let is_embed = i > 0 && bytes[i - 1] == b'!';
            if !target.is_empty() {
                if !links.iter().any(|l| l == target) {
                    links.push(target.to_string());
                }
                if is_embed && !embeds.iter().any(|l| l == target) {
                    embeds.push(target.to_string());
                }
            }
            i = i + 2 + end + 2;
            continue;
        }
        i += 1;
    }
    (links, embeds)
}

/// Extract a frontmatter list-valued field (`aliases:`/`tags:`), handling
/// Obsidian's three real forms: inline YAML list (`[a, b]`), block YAML
/// list (`- a`/`- b`), and its bare comma-shorthand (`a, b`) — not valid
/// YAML, so handled as a fallback when the parsed value is a plain string.
fn extract_frontmatter_list(frontmatter: &str, key: &str) -> Vec<String> {
    let Ok(value) = serde_norway::from_str::<serde_norway::Value>(frontmatter) else {
        return Vec::new();
    };
    let Some(value) = value.get(key) else {
        return Vec::new();
    };
    if let Some(seq) = value.as_sequence() {
        return seq
            .iter()
            .filter_map(|v| v.as_str().map(str::trim).map(str::to_string))
            .filter(|s| !s.is_empty())
            .collect();
    }
    if let Some(s) = value.as_str() {
        return s
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_string)
            .collect();
    }
    Vec::new()
}

/// Render a note's frontmatter as indexable prose.
///
/// Exists because of what a notes vault actually looks like: on the public
/// 103-note `kepano-obsidian` vault, 44 notes have an empty body and keep
/// everything in properties, so they produce no chunk and cannot be
/// retrieved by any query — including their own exact name. A movie note's
/// `director: Ridley Scott` is exactly what someone would search for, and
/// until now it was invisible.
///
/// Keys are kept alongside their values rather than stripped. They are
/// cheap, they are what the author chose to call the field, and they let a
/// query like "director Ridley Scott" match on both halves.
///
/// Three things are deliberately left out:
///
/// - **`title`**, because the title is indexed as the chunk's heading and
///   repeating it would double its term frequency.
/// - **URLs**, because a 200-character image URL is a machine identifier
///   rather than prose; it would dominate the chunk's length (which BM25
///   normalizes against) while matching nothing a person would type.
/// - **Anything past a character budget**, so that one note with a large
///   property block cannot skew the corpus statistics every other note's
///   score is normalized against.
pub fn frontmatter_text(frontmatter: &str) -> String {
    let Ok(serde_norway::Value::Mapping(map)) =
        serde_norway::from_str::<serde_norway::Value>(frontmatter)
    else {
        return String::new();
    };

    let mut out = String::new();
    for (key, value) in &map {
        let Some(key) = key.as_str() else { continue };
        if key.eq_ignore_ascii_case("title") {
            continue;
        }
        let rendered = render_value(value);
        if rendered.is_empty() {
            continue;
        }
        if out.len() + key.len() + rendered.len() + 3 > FRONTMATTER_TEXT_BUDGET {
            break;
        }
        if !out.is_empty() {
            out.push('\n');
        }
        out.push_str(key);
        out.push_str(": ");
        out.push_str(&rendered);
    }
    out
}

/// Character budget for [`frontmatter_text`]. Chosen to sit above anything a
/// hand-written property block reaches (the largest on `kepano-obsidian` is
/// under 200 characters once URLs are dropped) and well under the embedding
/// model's own window, so the budget never silently truncates a real note.
const FRONTMATTER_TEXT_BUDGET: usize = 1024;

/// One frontmatter value as text: scalars as written, sequences joined, and
/// nested mappings flattened to their values.
fn render_value(value: &serde_norway::Value) -> String {
    match value {
        serde_norway::Value::String(s) => {
            let s = clean_scalar(s);
            if is_url(&s) { String::new() } else { s }
        }
        serde_norway::Value::Number(n) => n.to_string(),
        serde_norway::Value::Bool(b) => b.to_string(),
        serde_norway::Value::Sequence(items) => items
            .iter()
            .map(render_value)
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>()
            .join(", "),
        // A nested mapping's keys are structure rather than content at this
        // depth, so only the values are kept.
        serde_norway::Value::Mapping(map) => map
            .iter()
            .map(|(_, v)| render_value(v))
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>()
            .join(", "),
        serde_norway::Value::Null | serde_norway::Value::Tagged(_) => String::new(),
    }
}

/// Strip the `[[...]]` wrapper (and any `|alias` or `#heading`) that
/// Obsidian's property fields are full of, so a link-valued property indexes
/// as the note name it points at.
fn clean_scalar(s: &str) -> String {
    let s = s.trim();
    let inner = s
        .strip_prefix("[[")
        .and_then(|rest| rest.strip_suffix("]]"))
        .unwrap_or(s);
    inner
        .split('|')
        .next()
        .unwrap_or(inner)
        .split('#')
        .next()
        .unwrap_or(inner)
        .trim()
        .to_string()
}

fn is_url(s: &str) -> bool {
    s.starts_with("http://") || s.starts_with("https://") || s.contains("://")
}

/// A note's frontmatter `aliases:` — see `extract_frontmatter_list` for the
/// forms handled.
pub fn extract_aliases(frontmatter: &str) -> Vec<String> {
    extract_frontmatter_list(frontmatter, "aliases")
}

/// A note's authoring date from frontmatter `created:`, as Unix seconds (UTC).
///
/// Reads `YYYY-MM-DD`, optionally followed by `T` or a space and a time, which
/// is how Obsidian and its templates write the property. A time of day is
/// ignored: recency is measured in days. `None` for a missing or unreadable
/// value, so a note without a date simply gets no recency bonus.
pub fn extract_created_unix(frontmatter: &str) -> Option<i64> {
    let value = serde_norway::from_str::<serde_norway::Value>(frontmatter).ok()?;
    let raw = match value.get("created")? {
        serde_norway::Value::String(s) => s.clone(),
        _ => return None,
    };
    parse_date_unix(&raw)
}

/// A `YYYY-MM-DD` date, optionally followed by a time, as Unix seconds (UTC).
pub fn parse_date_unix(text: &str) -> Option<i64> {
    let date = text.trim().get(..10)?;
    let mut parts = date.split('-');
    let year: i64 = parts.next()?.parse().ok()?;
    let month: i64 = parts.next()?.parse().ok()?;
    let day: i64 = parts.next()?.parse().ok()?;
    if parts.next().is_some() || !(1..=12).contains(&month) || !(1..=31).contains(&day) {
        return None;
    }
    // Days from the civil calendar (Hinnant), so no date crate is needed.
    let y = if month <= 2 { year - 1 } else { year };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let doy = (153 * ((month + 9) % 12) + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    Some((era * 146_097 + doe - 719_468) * 86_400)
}

/// Tags from frontmatter `tags:` (see `extract_frontmatter_list`) plus
/// inline `#tag` occurrences in the body, skipping fenced code blocks and
/// ATX headings (avoids `# Heading`, `#include` in code, "C#" false
/// positives — not a full Obsidian-compatible tag scanner, but covers the
/// common case).
fn extract_tags(frontmatter: Option<&str>, body: &str) -> Vec<String> {
    let mut tags: Vec<String> = Vec::new();
    if let Some(fm) = frontmatter {
        for t in extract_frontmatter_list(fm, "tags") {
            let t = t.trim_start_matches('#').to_string();
            if !t.is_empty() && !tags.contains(&t) {
                tags.push(t);
            }
        }
    }
    for t in extract_inline_tags(body) {
        if !tags.contains(&t) {
            tags.push(t);
        }
    }
    tags
}

/// Inline `#tag` occurrences in the body text (not frontmatter).
fn extract_inline_tags(body: &str) -> Vec<String> {
    let mut tags = Vec::new();
    let mut in_code = false;
    for line in body.lines() {
        let trimmed = line.trim_start();
        if trimmed.starts_with("```") || trimmed.starts_with("~~~") {
            in_code = !in_code;
            continue;
        }
        if in_code {
            continue;
        }
        // Skip ATX headings entirely (avoid "# Heading" -> tag "Heading").
        if trimmed.starts_with('#') && trimmed.trim_start_matches('#').starts_with(' ') {
            continue;
        }

        let bytes = line.as_bytes();
        let mut i = 0;
        while i < bytes.len() {
            let is_tag_start = bytes[i] == b'#'
                && (i == 0 || !bytes[i - 1].is_ascii_alphanumeric())
                && i + 1 < bytes.len()
                && (bytes[i + 1].is_ascii_alphabetic()
                    || bytes[i + 1] == b'_'
                    || bytes[i + 1] == b'/');
            if is_tag_start {
                let start = i + 1;
                let mut end = start;
                while end < bytes.len()
                    && (bytes[end].is_ascii_alphanumeric()
                        || bytes[end] == b'_'
                        || bytes[end] == b'-'
                        || bytes[end] == b'/')
                {
                    end += 1;
                }
                let tag = &line[start..end];
                // Obsidian rule: a tag can't be purely numeric.
                if tag.chars().any(|c| !c.is_ascii_digit()) {
                    tags.push(tag.to_string());
                }
                i = end;
                continue;
            }
            i += 1;
        }
    }
    tags
}

#[cfg(test)]
mod obsidian_tests {
    use super::*;

    #[test]
    fn aliases_inline_list() {
        let fm = "aliases: [Foo, Bar Baz]\ntitle: Real Title";
        assert_eq!(extract_aliases(fm), vec!["Foo", "Bar Baz"]);
    }

    #[test]
    fn aliases_block_list() {
        let fm = "aliases:\n  - Foo\n  - Bar Baz\n";
        assert_eq!(extract_aliases(fm), vec!["Foo", "Bar Baz"]);
    }

    #[test]
    fn aliases_comma_shorthand() {
        let fm = "aliases: Foo, Bar Baz";
        assert_eq!(extract_aliases(fm), vec!["Foo", "Bar Baz"]);
    }

    #[test]
    fn aliases_absent() {
        assert_eq!(extract_aliases("title: X"), Vec::<String>::new());
    }

    #[test]
    fn tags_frontmatter_and_inline_merge_deduped() {
        let fm = "tags: [project, #urgent]";
        let body = "This is #project work, also #urgent and #123 (numeric, skipped).";
        let tags = extract_tags(Some(fm), body);
        assert_eq!(tags, vec!["project", "urgent"]);
    }

    #[test]
    fn inline_tags_skip_code_blocks_and_headings() {
        let body = "# Heading\n\n```\n#include <stdio.h>\n```\n\nReal #tag here, not C#.";
        let tags = extract_tags(None, body);
        assert_eq!(tags, vec!["tag"]);
    }

    /// A property-only note, in the shape `kepano-obsidian` actually uses:
    /// link-valued fields, a number, a date, an id, and a cover image URL.
    #[test]
    fn frontmatter_text_renders_a_property_only_note() {
        let fm = concat!(
            "categories:\n  - \"[[Movies]]\"\n",
            "cover: https://m.media-amazon.com/images/M/MV5BNzQzMzJhZTEt._V1_SX300.jpg\n",
            "genre:\n  - \"[[Sci-fi]]\"\n",
            "director:\n  - \"[[Ridley Scott]]\"\n",
            "cast:\n  - \"[[Harrison Ford]]\"\n  - \"[[Rutger Hauer]]\"\n",
            "rating: 7\nyear: 1982\nimdbId: tt0083658\n",
        );
        let text = frontmatter_text(fm);
        assert_eq!(
            text,
            "categories: Movies\n\
             genre: Sci-fi\n\
             director: Ridley Scott\n\
             cast: Harrison Ford, Rutger Hauer\n\
             rating: 7\n\
             year: 1982\n\
             imdbId: tt0083658"
        );
    }

    /// The URL is the one value a person would never type and that would
    /// otherwise dominate the chunk's length, which BM25 normalizes against.
    #[test]
    fn frontmatter_text_drops_urls_and_keeps_the_rest() {
        let text = frontmatter_text("url: https://obsidian.md/\ntype: Apps\n");
        assert_eq!(text, "type: Apps");
    }

    /// The title is the chunk's heading, so repeating it here would double
    /// its term frequency against every other note's.
    #[test]
    fn frontmatter_text_omits_the_title() {
        let text = frontmatter_text("title: Real Title\nauthor: Someone\n");
        assert_eq!(text, "author: Someone");
    }

    /// One note with a large property block must not skew the corpus
    /// statistics every other note is scored against.
    #[test]
    fn frontmatter_text_stops_at_its_budget() {
        let mut fm = String::new();
        for i in 0..500 {
            fm.push_str(&format!("field{i}: value{i}\n"));
        }
        let text = frontmatter_text(&fm);
        assert!(
            text.len() <= FRONTMATTER_TEXT_BUDGET,
            "got {} chars",
            text.len()
        );
        assert!(text.starts_with("field0: value0"), "got {text}");
    }

    /// Frontmatter that is not a mapping at all (or is malformed) yields
    /// nothing rather than a panic or a stringified parse tree.
    #[test]
    fn frontmatter_text_of_nothing_is_empty() {
        assert_eq!(frontmatter_text(""), "");
        assert_eq!(frontmatter_text("- just\n- a list\n"), "");
        assert_eq!(frontmatter_text("key: [unterminated\n"), "");
    }

    #[test]
    fn embeds_distinguished_from_links_but_links_unchanged() {
        let (links, embeds) = extract_wikilinks("See [[Note A]] and ![[Note B]].");
        assert_eq!(links, vec!["Note A", "Note B"]);
        assert_eq!(embeds, vec!["Note B"]);
    }

    // ---- parse_pdf -------------------------------------------------------

    /// A 1.1 KB two-page PDF using base-14 Helvetica (no embedded fonts), with
    /// "Gnosis Sample Document" as its metadata Title. Compiled in rather than
    /// read at runtime so the test needs no fixture path.
    const SAMPLE_PDF: &[u8] = include_bytes!("../../../tests/fixtures/sample.pdf");

    #[test]
    fn parse_pdf_extracts_each_page_separately() {
        let parsed = parse_pdf(SAMPLE_PDF).expect("a valid PDF must parse");
        assert_eq!(parsed.pages.len(), 2, "expected one entry per page");
        assert_eq!(parsed.pages[0].0, 1, "pages are numbered from 1");
        assert_eq!(parsed.pages[1].0, 2);
        assert!(
            parsed.pages[0].1.contains("Chunking strategies"),
            "page 1 text missing, got {:?}",
            parsed.pages[0].1
        );
        assert!(
            parsed.pages[1].1.contains("approximate nearest"),
            "page 2 text missing, got {:?}",
            parsed.pages[1].1
        );
    }

    #[test]
    fn parse_pdf_does_not_leak_text_between_pages() {
        let parsed = parse_pdf(SAMPLE_PDF).unwrap();
        assert!(
            !parsed.pages[0].1.contains("nearest"),
            "page 1 must not contain page 2's text"
        );
        assert!(
            !parsed.pages[1].1.contains("Chunking"),
            "page 2 must not contain page 1's text"
        );
    }

    #[test]
    fn parse_pdf_prefers_the_metadata_title() {
        let parsed = parse_pdf(SAMPLE_PDF).unwrap();
        assert_eq!(parsed.title, "Gnosis Sample Document");
    }

    /// `pdf-extract` indexes content-stream operands without checking their
    /// length in ~50 places, so a malformed operator *panics* rather than
    /// returning an error. A panic would unwind past the indexing pipeline's
    /// per-file error handling and abort the entire run, so one bad PDF in a
    /// vault would cost every other file. The fixture is a structurally valid
    /// PDF whose content stream holds a bare `l` (lineto) with no operands;
    /// poppler reports a syntax error and recovers.
    #[test]
    fn parse_pdf_survives_a_malformed_content_stream() {
        const MALFORMED: &[u8] = include_bytes!("../../../tests/fixtures/malformed-ops.pdf");
        assert!(
            parse_pdf(MALFORMED).is_err(),
            "a malformed content stream must return Err, never panic"
        );
    }

    #[test]
    fn parse_pdf_rejects_bytes_that_are_not_a_pdf() {
        assert!(
            parse_pdf(b"this is not a pdf at all").is_err(),
            "garbage input must fail, not yield an empty document"
        );
    }

    #[test]
    fn parse_pdf_rejects_an_empty_input() {
        assert!(parse_pdf(b"").is_err());
    }

    #[test]
    fn parse_pdf_rejects_a_pdf_with_no_extractable_text() {
        // Header and trailer only: structurally PDF-ish, no text layer. A
        // scanned page behaves this way, and it must fail rather than index
        // as an empty document.
        let empty = b"%PDF-1.4\ntrailer\n<< /Size 1 >>\n%%EOF\n";
        assert!(parse_pdf(empty).is_err());
    }

    #[test]
    fn created_reads_a_plain_date() {
        assert_eq!(extract_created_unix("created: 1970-01-02"), Some(86_400));
        assert_eq!(
            extract_created_unix("created: 2024-02-29"),
            Some(1_709_164_800)
        );
    }

    #[test]
    fn created_ignores_a_time_of_day() {
        assert_eq!(
            extract_created_unix("created: \"2024-02-29T13:45:00\""),
            extract_created_unix("created: 2024-02-29")
        );
    }

    #[test]
    fn created_is_none_when_missing_or_unreadable() {
        assert_eq!(extract_created_unix("title: x"), None);
        assert_eq!(extract_created_unix("created: yesterday"), None);
        assert_eq!(extract_created_unix("created: 2024-13-01"), None);
    }
}
