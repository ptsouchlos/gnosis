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
        if !in_code
            && let Some(rest) = trimmed.strip_prefix("# ")
        {
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
        if bytes[i] == b'[' && bytes[i + 1] == b'[' {
            if let Some(end) = body[i + 2..].find("]]") {
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

/// A note's frontmatter `aliases:` — see `extract_frontmatter_list` for the
/// forms handled.
pub fn extract_aliases(frontmatter: &str) -> Vec<String> {
    extract_frontmatter_list(frontmatter, "aliases")
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
                && (bytes[i + 1].is_ascii_alphabetic() || bytes[i + 1] == b'_' || bytes[i + 1] == b'/');
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

    #[test]
    fn embeds_distinguished_from_links_but_links_unchanged() {
        let (links, embeds) = extract_wikilinks("See [[Note A]] and ![[Note B]].");
        assert_eq!(links, vec!["Note A", "Note B"]);
        assert_eq!(embeds, vec!["Note B"]);
    }
}
