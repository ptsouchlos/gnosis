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
    /// Obsidian `[[wikilink]]` targets found in the body (alias/heading stripped).
    pub links: Vec<String>,
}

/// Parse markdown text: split frontmatter, derive a title, collect wikilinks.
pub fn parse_markdown(path: &Path, content: &str) -> ParsedDoc {
    let (frontmatter, body) = split_frontmatter(content);
    let title = derive_title(path, frontmatter.as_deref(), body);
    let links = extract_wikilinks(body);

    ParsedDoc {
        title,
        frontmatter: frontmatter.map(str::to_string),
        body: body.to_string(),
        links,
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
fn extract_wikilinks(body: &str) -> Vec<String> {
    let mut links = Vec::new();
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
                if !target.is_empty() && !links.iter().any(|l| l == target) {
                    links.push(target.to_string());
                }
                i = i + 2 + end + 2;
                continue;
            }
        }
        i += 1;
    }
    links
}
