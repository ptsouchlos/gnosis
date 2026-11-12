#!/usr/bin/env bash
# Build the wasm PDF-extraction bundle, and report its size.
#
# The extraction is a separate bundle from the query client because it is the
# half of the PDF pipeline the host runs in a disposable worker instance —
# under the wasm-release profile's `panic = "abort"` a panic is a trap that
# `catch_unwind` cannot catch, and the host does not want a malformed PDF to
# be able to kill the instance it still holds queries in. The
# query client's linker would otherwise carry `pdf-extract` (and its `lopdf`
# and friends) into every query, for the one feature most queries do not touch.
#
# `--target web` is what the plugin bundles; the explicit `wasm-opt` pass is not
# a nicety but the only one that works here — see the note in
# crates/gnosis-extract/Cargo.toml.
set -euo pipefail

# Derived from this script's own location, not from `git rev-parse` or the
# working directory: the plugin repo calls this script to vendor its copy of
# the extraction bundle, and `--show-toplevel` would hand back *that* repo.
root="$(cd "$(dirname "$0")/../.." && pwd)"
out="${1:-$root/crates/gnosis-extract/pkg}"

wasm-pack build "$root/crates/gnosis-extract" \
    --target web \
    --profile wasm-release \
    --out-dir "$out" \
    --no-opt

wasm="$out/gnosis_extract_bg.wasm"
before="$(stat -c %s "$wasm")"
wasm-opt -Oz \
    --enable-bulk-memory \
    --enable-sign-ext \
    --enable-nontrapping-float-to-int \
    --enable-mutable-globals \
    --enable-reference-types \
    "$wasm" -o "$wasm.opt"
mv "$wasm.opt" "$wasm"
after="$(stat -c %s "$wasm")"

printf '%s\n' "$out/gnosis_extract_bg.wasm: $before bytes linked, $after after wasm-opt"
printf '%s\n' "  gzipped: $(gzip -9 -c "$wasm" | wc -c) bytes"
printf '%s\n' "  JS glue: $(stat -c %s "$out/gnosis_extract.js") bytes"
