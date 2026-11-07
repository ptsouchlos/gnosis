# gnosis-extract

The PDF half of the indexing pipeline, as its own WebAssembly bundle: one
export, [`extractPdf`], over the bytes of a PDF file.

It is a separate bundle from the query client rather than a second export on
the same `cdylib` for two reasons. A Rust crate links as one `.wasm`, and the
query client's linker would then carry `pdf-extract` — and its `lopdf` and the
font-parsing crates it drags in — into every query, for the one feature (reading
a PDF) that most queries do not touch. And `parse_pdf` is safe against a
malformed PDF *only* because it wraps `pdf-extract` in `catch_unwind`; in the
`wasm-release` profile that is `panic = "abort"`, where a panic is a trap. The
catch still works — the panic dies inside its own frame — but a hostile PDF
should not be able to share a fate with the instance the host still holds
queries in. So the extraction runs in a disposable instance the host respawns
per file, and a malformed PDF is a handled per-file boundary rather than a
crash.

## Building

```sh
script/wasm/build-extract.sh            # writes crates/gnosis-extract/pkg
```

Needs `wasm-pack` and `wasm-opt`, and the `wasm-bindgen` CLI at exactly the
version `Cargo.toml` pins — the two share an unstable schema and refuse to work
across versions.

Measured on this crate as it stands:

| | bytes |
|---|---|
| `.wasm`, linked | 1,125,270 |
| `.wasm`, after `wasm-opt -Oz` | 1,025,823 |
| `.wasm`, gzipped | 488,010 |
| JS glue | 10,089 |

The weight is `pdf-extract`'s dependency tree — `lopdf` and the font-parsing
crates behind it — which is exactly what the query client does not link.

## API

```js
const { title, pages } = extractPdf(pdfBytes);
// pages: [{ page: 1, text: "…" }, …]  1-indexed, in document order
```

`bytes` is the whole file, read as a `Uint8Array`. `title` is the document's
metadata `Title`, empty when it carries none — the host substitutes the file
stem. The call rejects bytes that are not a readable PDF, and rejects documents
that yield no text anywhere (scanned, image-only PDFs): the same two failures
the indexing pipeline treats as a per-file skip.

A malformed PDF does not reject — it **traps** the call (a `RuntimeError`),
which is what the disposable-instance design is for. `tests/node/trap.mjs` pins
that boundary.

## Testing

The native side is `parse_pdf` itself, which the workspace tests cover —
including the malformed content stream in `tests/fixtures/`:

```sh
cargo test -p parse
```

The boundary itself is pinned against the built package, in node:

```sh
node crates/gnosis-extract/tests/node/trap.mjs crates/gnosis-extract/pkg tests/fixtures
```

A valid fixture PDF extracts its title and pages; a malformed one traps the
call; a fresh instance still extracts afterward.
