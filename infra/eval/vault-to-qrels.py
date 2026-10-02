#!/usr/bin/env python3
"""Build a known-item qrels file from a real markdown vault.

Public IR benchmarks ask topical questions in full sentences, written by
someone who did not write the documents. Searching your own notes is a
different task: the queries are short, often name a note you know exists, and
use the vocabulary you yourself wrote. Nothing in BEIR measures that, which
leaves the signals aimed at it — title bonuses, lexical matching on a note's
name — untestable.

Known-item retrieval closes that gap without any hand-annotation: derive a
query from a note, and that note is the correct answer by construction.

Three query shapes are generated, and the third is a control:

  title_exact    the note's full title            navigational lookup
  title_partial  one or two words from the title  short keyword queries
  body_phrase    a phrase from the body that does NOT appear in the title

A title bonus will obviously flatter title-derived queries, so measuring only
those would be rigged. `body_phrase` is the check: a signal that helps the
first two and hurts the third is shape-dependent, not a general improvement,
and belongs off by default or behind query analysis.

What this does NOT measure: whether the retrieved note is the one a person
*wanted*. Known-item retrieval only asks whether a query derived from a note
finds that note back. Real queries are messier, half-remembered, and sometimes
satisfied by a different note than the one they came from. Treat the result as
a lower bound on navigational behaviour, not as a verdict on search quality.

Usage:
  infra/eval/vault-to-qrels.py ~/notes --out /tmp/vault-eval
"""

from __future__ import annotations

import argparse
import pathlib
import random
import re
import sys

# Words too common to identify a note on their own.
STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "is",
    "it", "at", "by", "from", "as", "my", "notes", "note", "template", "about",
    "this", "that", "how", "what", "new", "untitled", "index", "home", "readme",
}


def tokenize(text: str) -> list[str]:
    return [t for t in re.split(r"[^0-9A-Za-z]+", text.lower()) if t]


def strip_frontmatter(text: str) -> str:
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            return text[end + 4 :]
    return text


def title_of(path: pathlib.Path, body: str) -> str:
    """Obsidian identifies a note by its filename, so that is the title; an H1
    is only a fallback for notes that carry one instead."""
    stem = path.stem.strip()
    if stem:
        return stem
    m = re.search(r"^#\s+(.+)$", body, re.M)
    return m.group(1).strip() if m else ""


def distinctive(terms: list[str], corpus_df: dict[str, int], limit: int) -> list[str]:
    """The rarest terms, which are what someone would actually search with."""
    ranked = sorted(set(terms), key=lambda t: (corpus_df.get(t, 0), -len(t)))
    return [t for t in ranked if t not in STOPWORDS and len(t) > 2][:limit]


def toml_escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ").strip()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("vault", type=pathlib.Path)
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--seed", type=int, default=0, help="for reproducible sampling")
    ap.add_argument(
        "--per-type",
        type=int,
        default=0,
        help="cap queries per type (0 = one per note where possible)",
    )
    args = ap.parse_args()
    rng = random.Random(args.seed)

    vault = args.vault.expanduser().resolve()
    notes = [
        p for p in vault.rglob("*.md")
        if ".obsidian" not in p.parts and ".git" not in p.parts and ".gnosis" not in p.parts
    ]
    if not notes:
        print(f"no markdown notes under {vault}", file=sys.stderr)
        return 1

    parsed = []
    corpus_df: dict[str, int] = {}
    for p in notes:
        try:
            raw = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        body = strip_frontmatter(raw)
        title = title_of(p, body)
        if not title:
            continue
        parsed.append((p, title, body))
        for t in set(tokenize(body) + tokenize(title)):
            corpus_df[t] = corpus_df.get(t, 0) + 1

    queries: list[tuple[str, str, str]] = []  # (type, query text, relative path)
    for p, title, body in parsed:
        rel = p.relative_to(vault).as_posix()
        title_terms = tokenize(title)
        if not title_terms:
            continue

        queries.append(("title_exact", title, rel))

        partial = distinctive(title_terms, corpus_df, 2)
        if partial and " ".join(partial).lower() != title.lower():
            queries.append(("title_partial", " ".join(partial), rel))

        # A phrase from the body that does not lean on the title, so the control
        # cannot be answered by title matching alone. Markdown and URLs are
        # dropped: a query made of link syntax tests the tokenizer, not search.
        title_set = set(title_terms)
        sentences = [x.strip() for x in re.split(r"[.!?\n]", body) if 25 < len(x.strip()) < 160]
        usable = []
        for cand in sentences:
            if any(mark in cand for mark in ("![", "[[", "](", "http", "|", "`", "<")):
                continue
            if cand.lstrip().startswith(("#", ">", "-", "*")):
                continue
            terms = tokenize(cand)
            if len(terms) < 6:
                continue
            # Allow incidental overlap, but not a phrase carried by the title.
            if title_set and len(set(terms) & title_set) / len(title_set) > 0.5:
                continue
            usable.append(cand)
        if usable:
            queries.append(("body_phrase", rng.choice(usable), rel))

    # A query that two notes answer equally well is not a known-item query —
    # scoring it against one of them would penalise a correct result. Drop
    # collisions rather than silently accepting an unfair denominator.
    seen: dict[tuple[str, str], int] = {}
    for kind, text, _ in queries:
        key = (kind, text.strip().lower())
        seen[key] = seen.get(key, 0) + 1
    dropped = sum(n for n, in ((v,) for v in seen.values()) if n > 1)
    queries = [q for q in queries if seen[(q[0], q[1].strip().lower())] == 1]
    if dropped:
        print(f"dropped {dropped} ambiguous queries that more than one note answers")

    if args.per_type:
        by_type: dict[str, list] = {}
        for q in queries:
            by_type.setdefault(q[0], []).append(q)
        queries = []
        for kind, items in by_type.items():
            rng.shuffle(items)
            queries.extend(items[: args.per_type])

    args.out.mkdir(parents=True, exist_ok=True)
    for kind in sorted({q[0] for q in queries}):
        lines = [
            f"# Known-item queries of shape '{kind}', generated by",
            "# infra/eval/vault-to-qrels.py. Each query is derived from a note and",
            "# that note is the answer, so no hand-annotation is involved.",
            f'vault = "{vault}"',
            "",
        ]
        for _, text, rel in (q for q in queries if q[0] == kind):
            lines += [
                "[[query]]",
                f'text = "{toml_escape(text)}"',
                'space = "text"',
                f'relevant = [{{ path = "{toml_escape(rel)}", grade = 2 }}]',
                "",
            ]
        path = args.out / f"qrels-{kind}.toml"
        path.write_text("\n".join(lines), encoding="utf-8")
        n = sum(1 for q in queries if q[0] == kind)
        print(f"{path}  ({n} queries)")

    print(f"\nnotes: {len(parsed)}   queries: {len(queries)}")
    print(f"\nnext:\n  cd {vault} && gnosis index .")
    print(f"  gnosis eval --qrels {args.out}/qrels-title_exact.toml")
    return 0


if __name__ == "__main__":
    sys.exit(main())
