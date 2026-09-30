#!/usr/bin/env python3
"""Turn a public BEIR dataset into a gnosis vault plus a qrels file.

Gives `gnosis eval` a reproducible, public benchmark with published relevance
judgements, so retrieval quality can be measured without pointing the harness
at anyone's private notes.

Every BEIR dataset ships the same three files, so any of them works:

  corpus.jsonl   {"_id", "title", "text"}      -> one markdown note per doc
  queries.jsonl  {"_id", "text"}               -> eval queries
  qrels/test.tsv query-id, corpus-id, score    -> graded judgements

Scores map straight onto gnosis's grades (1 = relevant, 2 = ideal).

Standard library only, on purpose: this runs in CI or on a fresh checkout with
no pip install step.

Usage:
  infra/eval/beir-to-vault.py nfcorpus --out /tmp/beir-nfcorpus
  infra/eval/beir-to-vault.py scifact  --out /tmp/beir-scifact --max-queries 50

Then:
  cd /tmp/beir-nfcorpus/vault && gnosis index .
  gnosis eval --qrels ../qrels.toml
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import re
import sys
import urllib.request
import zipfile

BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"

# Datasets small enough to index quickly. Others work, but mind the size.
SUGGESTED = {
    "nfcorpus": "3.6k docs, 323 test queries, graded 1-2 (dense: ~16 relevant/query)",
    "scifact": "5.2k docs, 300 test queries, binary grades (sparse: ~1 relevant/query)",
    "arguana": "8.7k docs, 1406 test queries, binary grades",
    "scidocs": "25k docs, 1000 test queries, binary grades",
}


def fetch(name: str, cache: pathlib.Path) -> pathlib.Path:
    """Download the dataset zip unless it is already cached."""
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / f"{name}.zip"
    if archive.exists():
        print(f"using cached {archive}")
        return archive
    url = BEIR_URL.format(name=name)
    print(f"downloading {url}")
    with urllib.request.urlopen(url) as response, open(archive, "wb") as out:
        out.write(response.read())
    return archive


def read_jsonl(zf: zipfile.ZipFile, member: str) -> list[dict]:
    text = zf.read(member).decode("utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def safe_stem(doc_id: str) -> str:
    """A filename-safe stem. BEIR ids are already tame (MED-1234), but a
    dataset with slashes or spaces in an id would otherwise escape the vault."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", doc_id)


def toml_escape(s: str) -> str:
    """Escape a value for a TOML basic string."""
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ").strip()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("dataset", help=f"BEIR dataset name (try: {', '.join(SUGGESTED)})")
    parser.add_argument("--out", required=True, type=pathlib.Path, help="output directory")
    parser.add_argument(
        "--split", default="test", help="qrels split to use (default: test)"
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=0,
        help="keep only the first N judged queries, for a faster iteration loop",
    )
    parser.add_argument(
        "--cache",
        type=pathlib.Path,
        default=pathlib.Path.home() / ".cache" / "gnosis" / "beir",
        help="where to cache downloaded archives",
    )
    args = parser.parse_args()

    archive = fetch(args.dataset, args.cache)
    with zipfile.ZipFile(archive) as zf:
        root = f"{args.dataset}/"
        corpus = read_jsonl(zf, f"{root}corpus.jsonl")
        queries = read_jsonl(zf, f"{root}queries.jsonl")
        qrels_raw = zf.read(f"{root}qrels/{args.split}.tsv").decode("utf-8").splitlines()

    # qrels: skip the header, group by query.
    judged: dict[str, list[tuple[str, int]]] = collections.defaultdict(list)
    for line in qrels_raw[1:]:
        if not line.strip():
            continue
        query_id, doc_id, score = line.split("\t")[:3]
        grade = int(score)
        if grade > 0:  # BEIR records 0 rows in some datasets; they add nothing
            judged[query_id].append((doc_id, grade))

    query_text = {q["_id"]: q["text"] for q in queries}

    # Only queries that actually have judgements are worth evaluating.
    judged_ids = [qid for qid in judged if qid in query_text]
    judged_ids.sort()
    if args.max_queries:
        judged_ids = judged_ids[: args.max_queries]
        keep = {doc_id for qid in judged_ids for doc_id, _ in judged[qid]}
        print(f"limited to {len(judged_ids)} queries ({len(keep)} judged docs)")

    # Write the vault. Every corpus doc is written even when unjudged: a
    # retrieval benchmark needs the distractors, otherwise every query is
    # scored against a corpus of nothing but its own answers.
    vault = args.out / "vault"
    vault.mkdir(parents=True, exist_ok=True)
    for doc in corpus:
        title = doc.get("title") or doc["_id"]
        body = doc.get("text") or ""
        path = vault / f"{safe_stem(doc['_id'])}.md"
        path.write_text(f"# {title}\n\n{body}\n", encoding="utf-8")

    # A vault config, so `gnosis index .` works with no further setup. Image
    # and PDF handling are off: this corpus is text only.
    (vault / "gnosis.toml").write_text(
        "[embed.image]\nenabled = false\n\n[pdf]\nenabled = false\n", encoding="utf-8"
    )

    # Write the qrels file. Paths are vault-relative, which is what gnosis
    # eval expects.
    lines = [
        f"# Generated by infra/eval/beir-to-vault.py from BEIR/{args.dataset}"
        f" ({args.split} split).",
        "# Public dataset with published relevance judgements — no private data.",
        f'vault = "{vault.resolve()}"',
        "",
    ]
    for qid in judged_ids:
        rels = sorted(judged[qid], key=lambda pair: (-pair[1], pair[0]))
        lines.append("[[query]]")
        lines.append(f'text = "{toml_escape(query_text[qid])}"')
        lines.append('space = "text"')
        lines.append("relevant = [")
        for doc_id, grade in rels:
            lines.append(f'  {{ path = "{safe_stem(doc_id)}.md", grade = {grade} }},')
        lines.append("]")
        lines.append("")

    qrels_path = args.out / "qrels.toml"
    qrels_path.write_text("\n".join(lines), encoding="utf-8")

    total_judgements = sum(len(judged[qid]) for qid in judged_ids)
    print(f"\nvault:  {vault}  ({len(corpus)} notes)")
    print(f"qrels:  {qrels_path}  ({len(judged_ids)} queries, {total_judgements} judgements)")
    print("\nnext:")
    print(f"  cd {vault} && gnosis index .")
    print(f"  cd {vault} && gnosis eval --qrels {qrels_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
