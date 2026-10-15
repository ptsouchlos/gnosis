# gnosis-wasm

The query client: gnosis's retrieval pipeline compiled to WebAssembly, reading
a portable artifact. It is what lets a host without SQLite — the Obsidian
plugin — run the ranking code this project measures instead of reimplementing
it in JavaScript.

The embedder stays outside. `search` takes a query *vector*, which is what
keeps ONNX Runtime out of the build; the client embeds with transformers.js,
using the model id and the pooling that `info()` reports.

## Building

```sh
script/wasm/build.sh            # writes crates/gnosis-wasm/pkg
```

Needs `wasm-pack` and `wasm-opt`, and the `wasm-bindgen` CLI at exactly the
version `Cargo.toml` pins — the two share an unstable schema and refuse to work
across versions.

Measured on this crate as it stands:

| | bytes |
|---|---|
| `.wasm`, linked | 364,361 |
| `.wasm`, after `wasm-opt -Oz` | 326,098 |
| `.wasm`, gzipped | 141,054 |
| JS glue | 21,588 |

## API

```js
const index = Index.open(manifestJson, shards);   // shards: Map<name, Uint8Array>

index.search(query, new Map([["text", vector]]), 10);   // hybrid: vector + text
index.lexicalSearch(query, 10);                          // no model at all
index.related("notes/sourdough.md", 10, false);          // no model, full quality
index.info();                                            // models, pooling, dims, counts
index.staleness(entries);                                // what the vault disagrees with
```

Every call takes an optional last argument, a filter: `{from: ["notes"], tags:
["baking"]}`. `from` matches a root *label*, not a path — an artifact records
what a vault was called, not where it lived on the machine that built it.

## Testing

The surface tests run in node, against real JS values:

```sh
CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUNNER=wasm-bindgen-test-runner \
  cargo test -p gnosis-wasm --target wasm32-unknown-unknown
```

The acceptance harness answers the question those cannot: does a ranking
computed in wasm match the ranking gnosis computes? `tests/fixture.rs` writes
the queries and the Rust pipeline's rankings for them; `tests/node/parity.mjs`
runs the same queries through the built package and compares.

```sh
GNOSIS_WASM_ARTIFACT=/path/to/artifact GNOSIS_WASM_FIXTURE=/tmp/fix \
  cargo test --release -p gnosis-wasm --test fixture -- --ignored --nocapture
node crates/gnosis-wasm/tests/node/parity.mjs crates/gnosis-wasm/pkg /tmp/fix
```

Measured on a scifact export (5,183 documents, 6,255 chunks), 310 cases over
the f32 artifact and 210 over the i8 one: **every ranking identical**, largest
score difference 1.0e-7. That residue is not rounding slack — IEEE-754 pins
`+`, `*` and `sqrt` but not `ln`, and BM25's IDF takes a logarithm, so wasm's
libm and glibc's differ in the last bits. It is six orders of magnitude below
the gap between adjacent results.

`related` was also compared against the CLI itself on the same corpus —
model-free on both sides, so the comparison is exact rather than approximate:
**11/11 identical top-10s**.

## Latency

Median per query in node, same 6,255-chunk artifact:

| | median |
|---|---|
| `lexicalSearch` | 2.8 ms |
| dense-only `search` (lexical off) | 8.9 ms |
| `related` | 8.9 ms |
| hybrid `search` | 11 ms |
| `open` (21 shards, 13 MB f32) | 330 ms |

Every query path is now well inside the CLI's amortized 35.7 ms, and query
length no longer matters: 10.4 ms at one term against 11.0 ms at ten, where it
used to run 33 ms to 159 ms. The lexical channel costs about 2 ms on top of the
dense scan.

It did not start there. The first measurement of this crate had hybrid search
at 155 ms, almost all of it the lexical channel re-tokenizing its candidate
pool on every query — the pool has to be scored before it is truncated, and it
grows with the query. Keeping each term's frequency beside its posting fixed
it, and that turned out to need no format change at all: the index is built at
`open`, which already tokenizes every chunk to find out which terms it
contains, so counting them in the same pass is free.
