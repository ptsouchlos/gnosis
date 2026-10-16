// Does the built query client reproduce gnosis's own ranking?
//
// Takes the wasm-pack output, a fixture written by `tests/fixture.rs` (the
// queries plus the rankings the Rust pipeline produced for them), and runs the
// same queries through the `.wasm` in node. Every hit must match by path and
// by score; a single mismatch fails the run and prints both rankings.
//
// It also times the queries, because latency is the other number the plugin's
// design depends on and nothing else measures it.
//
//   node tests/node/parity.mjs <pkg-dir> <fixture-dir>
//
// Both paths come in as arguments; nothing is read from this script's own
// directory, and the artifact is read as data only.

import { readFile } from "node:fs/promises";
import { resolve, join } from "node:path";
import { pathToFileURL } from "node:url";

const [pkgDir, fixtureDir] = process.argv.slice(2);
if (!pkgDir || !fixtureDir) {
  console.error("usage: node parity.mjs <pkg-dir> <fixture-dir>");
  process.exit(2);
}

const fixture = JSON.parse(await readFile(join(fixtureDir, "expected.json"), "utf8"));

// The `web` target fetches its `.wasm` by URL, which does not work for a local
// file in node — so the bytes go in explicitly. This is the same module the
// plugin loads, rather than a node-specific build of it.
const mod = await import(pathToFileURL(resolve(pkgDir, "gnosis_wasm.js")).href);
await mod.default({
  module_or_path: await readFile(resolve(pkgDir, "gnosis_wasm_bg.wasm")),
});

const manifestJson = await readFile(join(fixture.artifactDir, "manifest.json"), "utf8");
const manifest = JSON.parse(manifestJson);
const shards = new Map();
for (const shard of manifest.shards) {
  shards.set(shard.name, new Uint8Array(await readFile(join(fixture.artifactDir, shard.name))));
}

const openStart = performance.now();
const index = mod.Index.open(manifestJson, shards);
const openMs = performance.now() - openStart;

const info = index.info();
console.log(
  `opened ${info.documents} documents / ${info.chunks} chunks in ${openMs.toFixed(1)} ms ` +
    `(${info.quantization}, ${info.compression})`,
);
for (const space of info.spaces) {
  console.log(
    `  ${space.space}: ${space.chunks} chunks, ${space.dim}d, ` +
      `query model ${space.queryModel}, pooling ${space.pooling}`,
  );
}

const run = (c) => {
  switch (c.kind) {
    case "search": {
      const vectors = new Map([[c.space, new Float32Array(c.vector)]]);
      return index.search(c.query, vectors, c.limit, undefined);
    }
    case "lexical":
      return index.lexicalSearch(c.query, c.limit, undefined);
    case "related":
      return index.related(c.path, c.limit, c.includeLinked, undefined);
    case "similarities":
      // Flattened to hits, in row order, so the comparison below applies.
      return index.similarities(c.paths).flatMap((row, i) =>
        row.flatMap((score, j) =>
          score === null ? [] : [{ path: `${c.paths[i]} -> ${c.paths[j]}`, score }],
        ),
      );
    default:
      throw new Error(`unknown case kind '${c.kind}'`);
  }
};

const SCORE_TOLERANCE = 1e-6;

const byKind = new Map();
const failures = [];
let worstGap = 0;
for (const c of fixture.cases) {
  const start = performance.now();
  const hits = run(c);
  const ms = performance.now() - start;

  const stats = byKind.get(c.kind) ?? { n: 0, times: [], mismatched: 0 };
  stats.n += 1;
  stats.times.push(ms);

  const got = hits.map((h) => h.path);
  const want = c.expected.map((e) => e.path);
  const scoreGap = Math.max(
    0,
    ...hits.map((h, i) => Math.abs(h.score - (c.expected[i]?.score ?? Infinity))),
  );
  worstGap = Math.max(worstGap, scoreGap);
  // The ranking must match exactly. Scores are compared with a tolerance, and
  // the tolerance is not slack for sloppiness: IEEE-754 pins +, * and sqrt,
  // but not `ln`, and BM25's IDF takes a logarithm. wasm gets Rust's own libm
  // where x86_64 gets glibc's, and the two differ in the last bits — measured
  // at up to 1e-7 of a score in [0, 1] on scifact, which is six orders of
  // magnitude below the gap between adjacent results. A difference above this
  // is not a rounding artifact and should be read as a real one.
  if (JSON.stringify(got) !== JSON.stringify(want) || scoreGap > SCORE_TOLERANCE) {
    stats.mismatched += 1;
    if (failures.length < 3) {
      failures.push(
        `  ${c.kind} ${JSON.stringify(c.query ?? c.path)}\n` +
          `    rust: ${JSON.stringify(want)}\n` +
          `    wasm: ${JSON.stringify(got)}\n` +
          `    largest score gap: ${scoreGap}`,
      );
    }
  }
  byKind.set(c.kind, stats);
}

const percentile = (xs, p) => {
  const sorted = [...xs].sort((a, b) => a - b);
  return sorted[Math.min(sorted.length - 1, Math.floor((p / 100) * sorted.length))];
};

let mismatched = 0;
for (const [kind, s] of byKind) {
  mismatched += s.mismatched;
  console.log(
    `${kind}: ${s.n - s.mismatched}/${s.n} matched, ` +
      `median ${percentile(s.times, 50).toFixed(2)} ms, p95 ${percentile(s.times, 95).toFixed(2)} ms`,
  );
}
console.log(`largest score difference across every case: ${worstGap.toExponential(2)}`);
for (const f of failures) console.log(f);

if (mismatched > 0) {
  console.error(`${mismatched} case(s) did not reproduce the Rust ranking`);
  process.exit(1);
}
console.log("every case reproduced the Rust ranking exactly");
