#!/usr/bin/env python3
"""Turn a Mr. TyDi language into a gnosis vault plus a qrels file.

Every corpus gnosis is measured on today is English: BEIR is English-only, and
the known-item generator reads whatever vault it is pointed at. So the one claim
a multilingual model exists to make — that it retrieves in languages the current
models were never trained on — has had nothing to test it.

Mr. TyDi (Zhang et al. 2021) covers eleven typologically diverse languages with
human relevance judgements over Wikipedia, which makes it the cheapest honest
answer. Three files per language, same shape as BEIR's:

  ir-format-data/topics.<split>.txt   query_id \\t query text
  ir-format-data/qrels.<split>.txt    query_id, Q0, docid, grade
  <corpus>/corpus.jsonl.gz            {"docid", "title", "text"}

Languages, by corpus download size — start small:

  swahili 11 MB · bengali 61 MB · telugu 75 MB · thai 115 MB · indonesian 173 MB
  korean 228 MB · finnish 274 MB · arabic 328 MB · japanese 1.1 GB
  russian 1.6 GB · english 5.1 GB

Note for reading the results: the lexical channel's strength varies a lot by
language, because SQLite FTS5's `unicode61` tokenizer splits on character class.
Measured with `bge-small-en-v1.5`, lexical-only nDCG@10 is 0.674 on swahili but
0.424 on korean — passable, since Korean does put spaces between phrases. Thai
and japanese are the unverified worst case: no spaces at all. A multilingual
*embedding* model cannot fix the lexical channel, so the two channels have to be
read separately per language rather than as one hybrid number.

## On --max-docs

The full Swahili corpus is 137k passages; japanese is millions. Indexing that to
compare two models is not a good use of an afternoon, so --max-docs keeps every
judged passage plus a random sample of the rest.

This makes the task easier than published Mr. TyDi, because most distractors are
gone. **Absolute scores from a sampled run are not comparable to published Mr.
TyDi numbers, or to each other across different --max-docs values.** They are
only comparable between models at the same setting, which is what a model
comparison needs. Pass --max-docs 0 for the real task.

Standard library only, matching beir-to-vault.py: this has to run on a fresh
checkout with no pip install step.

Usage:
  infra/eval/mrtydi-to-vault.py swahili --out /tmp/mrtydi-sw --max-docs 20000
  infra/eval/mrtydi-to-vault.py korean  --out /tmp/mrtydi-ko --max-docs 20000
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import pathlib
import random
import re
import sys
import urllib.request

QUERY_URL = "https://huggingface.co/datasets/castorini/mr-tydi/resolve/main/mrtydi-v1.1-{lang}/ir-format-data/{name}"
CORPUS_URL = "https://huggingface.co/datasets/castorini/mr-tydi-corpus/resolve/main/mrtydi-v1.1-{lang}/corpus.jsonl.gz"

# Corpus download size in MB, so the script can warn before a 5 GB surprise.
LANGUAGES = {
    "swahili": 11, "bengali": 61, "telugu": 75, "thai": 115, "indonesian": 173,
    "korean": 228, "finnish": 274, "arabic": 328, "japanese": 1078,
    "russian": 1588, "english": 5068,
}


def fetch(url: str, dest: pathlib.Path) -> pathlib.Path:
    """Download unless already cached. Caching matters more here than in the
    BEIR script: these corpora are large enough that re-downloading one to
    re-run an eval would be the slowest part of the loop."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"using cached {dest}")
        return dest
    print(f"downloading {url}")
    with urllib.request.urlopen(url) as response, open(dest, "wb") as out:
        out.write(response.read())
    return dest


def safe_stem(doc_id: str) -> str:
    """Mr. TyDi docids look like `2681119#1` — the `#` is harmless on disk but
    not worth relying on, and a docid is not a filename until it is made one."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", doc_id)


def toml_escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ").strip()


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("language", help=f"one of: {', '.join(LANGUAGES)}")
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--split", default="test", choices=["test", "dev", "train"])
    ap.add_argument(
        "--max-docs",
        type=int,
        default=20000,
        help="cap the vault at N passages, keeping every judged one (0 = whole corpus)",
    )
    ap.add_argument("--seed", type=int, default=0, help="for reproducible sampling")
    ap.add_argument(
        "--cache",
        type=pathlib.Path,
        default=pathlib.Path.home() / ".cache" / "gnosis" / "mrtydi",
        help="where to cache downloads",
    )
    args = ap.parse_args()

    if args.language not in LANGUAGES:
        print(f"unknown language '{args.language}'; try: {', '.join(LANGUAGES)}", file=sys.stderr)
        return 1
    rng = random.Random(args.seed)
    lang = args.language
    size_mb = LANGUAGES[lang]
    if size_mb > 400:
        print(f"note: the {lang} corpus is ~{size_mb} MB compressed", file=sys.stderr)

    topics_path = fetch(
        QUERY_URL.format(lang=lang, name=f"topics.{args.split}.txt"),
        args.cache / lang / f"topics.{args.split}.txt",
    )
    qrels_path = fetch(
        QUERY_URL.format(lang=lang, name=f"qrels.{args.split}.txt"),
        args.cache / lang / f"qrels.{args.split}.txt",
    )
    corpus_path = fetch(
        CORPUS_URL.format(lang=lang), args.cache / lang / "corpus.jsonl.gz"
    )

    query_text: dict[str, str] = {}
    for line in topics_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        qid, _, text = line.partition("\t")
        if text.strip():
            query_text[qid.strip()] = text.strip()

    judged: dict[str, list[tuple[str, int]]] = collections.defaultdict(list)
    for line in qrels_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        qid, _, docid, grade = parts[0], parts[1], parts[2], int(parts[3])
        if grade > 0:
            judged[qid].append((docid, grade))

    judged_ids = sorted(qid for qid in judged if qid in query_text)
    if not judged_ids:
        print("no judged queries found — wrong split?", file=sys.stderr)
        return 1
    relevant_docs = {docid for qid in judged_ids for docid, _ in judged[qid]}

    # One pass over the corpus. Judged passages are always kept; the rest are
    # reservoir-sampled, so the whole corpus never has to be held in memory —
    # the Japanese and Russian ones would not fit comfortably.
    budget = max(0, args.max_docs - len(relevant_docs)) if args.max_docs else None
    kept: list[tuple[str, str, str]] = []
    distractors: list[tuple[str, str, str]] = []
    seen = 0
    with gzip.open(corpus_path, "rt", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            doc = json.loads(line)
            row = (doc["docid"], doc.get("title") or "", doc.get("text") or "")
            if row[0] in relevant_docs:
                kept.append(row)
                continue
            if budget is None:
                distractors.append(row)
                continue
            seen += 1
            if len(distractors) < budget:
                distractors.append(row)
            else:
                j = rng.randrange(seen)
                if j < budget:
                    distractors[j] = row

    missing = len(relevant_docs) - len(kept)
    if missing:
        print(f"warning: {missing} judged passages are absent from the corpus", file=sys.stderr)

    vault = args.out / "vault"
    vault.mkdir(parents=True, exist_ok=True)
    for docid, title, text in kept + distractors:
        heading = title or docid
        (vault / f"{safe_stem(docid)}.md").write_text(
            f"# {heading}\n\n{text}\n", encoding="utf-8"
        )
    (vault / "gnosis.toml").write_text(
        "[embed.image]\nenabled = false\n\n[pdf]\nenabled = false\n", encoding="utf-8"
    )

    sampled = args.max_docs and len(distractors) < (seen or 0)
    lines = [
        f"# Generated by infra/eval/mrtydi-to-vault.py from Mr. TyDi {lang}"
        f" ({args.split} split).",
        "# Public dataset with human relevance judgements — no private data.",
    ]
    if sampled:
        lines.append(
            f"# SAMPLED: {len(distractors)} of {seen} distractors kept (--max-docs"
            f" {args.max_docs}, seed {args.seed}). Scores are comparable between"
            " models at this setting, NOT to published Mr. TyDi numbers."
        )
    lines += [f'vault = "{vault.resolve()}"', ""]
    for qid in judged_ids:
        rels = sorted(judged[qid], key=lambda pair: (-pair[1], pair[0]))
        lines += [
            "[[query]]",
            f'text = "{toml_escape(query_text[qid])}"',
            'space = "text"',
            "relevant = [",
        ]
        for docid, grade in rels:
            lines.append(f'  {{ path = "{safe_stem(docid)}.md", grade = {grade} }},')
        lines += ["]", ""]

    out_qrels = args.out / "qrels.toml"
    out_qrels.write_text("\n".join(lines), encoding="utf-8")

    notes = len(kept) + len(distractors)
    judgements = sum(len(judged[qid]) for qid in judged_ids)
    print(f"\nvault:  {vault}  ({notes} notes)")
    print(f"qrels:  {out_qrels}  ({len(judged_ids)} queries, {judgements} judgements)")
    if sampled:
        print(f"\nsampled {len(distractors)} of {seen} distractors — relative comparison only")
    print("\nnext:")
    print(f"  cd {vault} && gnosis index .")
    print(f"  cd {vault} && gnosis eval --qrels {out_qrels}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
