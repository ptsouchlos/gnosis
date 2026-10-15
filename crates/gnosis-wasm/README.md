# gnosis-wasm

The query client: gnosis's retrieval pipeline compiled to WebAssembly, reading
a portable artifact. It is what lets a host without SQLite — the Obsidian
plugin — run the ranking code this project measures instead of reimplementing
it in JavaScript.

Two halves: [`Index`] answers queries over an artifact, and [`Builder`] writes
one. Both draw the same line — the client walks the vault, reads files and runs
the model, because only it can; everything that decides what the index contains
or how it ranks stays in Rust, in the code the CLI runs.

The embedder stays outside. `search` takes a query *vector* and `Builder` hands
back the strings to embed, which is what keeps ONNX Runtime out of the build;
the client embeds with transformers.js, using the model id and the pooling that
`info()` reports.

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
| `.wasm`, linked | 733,773 |
| `.wasm`, after `wasm-opt -Oz` | 663,582 |
| `.wasm`, gzipped | 296,444 |
| JS glue | 29,904 |

Building an index roughly doubled that (it was 326,098 after `wasm-opt` when
this crate only queried), and the weight is the markdown parser and the YAML
parser — not, as expected, the PDF extractor, which the linker drops because
nothing here calls it. Measured by removing it: 1,005 bytes. A query-only build
for readers who never index would be a `Cargo` feature away if the size ever
matters more than the simplicity of one bundle.

## API

```js
const index = Index.open(manifestJson, shards);   // shards: Map<name, Uint8Array>

index.search(query, new Map([["text", vector]]), 10);   // hybrid: vector + text
index.lexicalSearch(query, 10);                          // no model at all
index.related("notes/sourdough.md", 10, false);          // no model, full quality
index.info();                                            // models, pooling, dims, counts
index.staleness(entries);                                // what the vault disagrees with
```

Building one:

```js
const builder = Builder.create({ model: "bge-small-en-v1.5", pooling: "cls",
                                 queryPrefix: "Represent this sentence for searching relevant passages: ",
                                 documentPrefix: "", dim: 384, root: "notes" });
// or Builder.fromArtifact(spec, manifestJson, shards) to reuse what has not changed

const { reused, texts } = builder.addDocument(path, text, file.stat.mtime);
builder.setVectors(path, flatFloat32Array);   // chunks * dim, in chunk order
const { manifest, shards, unchanged } = builder.finish(Date.now() / 1000);
```

`addDocument` parses and chunks in Rust and hands back the exact strings to
embed — document prefix and heading trail included — so the embedding input is
what `gnosis index` embeds. Both prefixes are required: `""` is a statement that
the model wants none, and leaving one out is refused rather than defaulted,
because an E5 build that forgot `"passage: "` would look healthy and rank badly.

On the query side, `info()` reports each space's `queryPrefix` as the index
recorded it. Prepend it before embedding a query; it is not derivable from the
model id, since `query_instruction = false` turns it off per index. A document whose
content hash matches the previous artifact's comes back `reused` with nothing
to embed. `unchanged` names the shards whose bytes did not change, which are
not returned — rewriting a file a syncer already has is the churn sharding
exists to avoid.

Every call takes an optional last argument, a filter: `{from: ["notes"], tags:
["baking"]}`. `from` matches a root *label*, not a path — an artifact records
what a vault was called, not where it lived on the machine that built it.

## Testing

The surface tests run in node, against real JS values:

```sh
CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUNNER=wasm-bindgen-test-runner \
  cargo test -p gnosis-wasm --target wasm32-unknown-unknown
```

`Builder` has a second, stronger check, in the CLI's own tests: an artifact it
builds from a vault's text is **byte-identical** to the one `gnosis index` and
`gnosis export` produce from the same vault, with a deterministic stand-in for
the model so that any difference in chunk boundaries or embedding inputs turns
into a byte comparison rather than hiding behind inference noise.

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
