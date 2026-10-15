#!/usr/bin/env bash
# Build the wasm query client the Obsidian plugin consumes, and report its size.
#
# `--target web` is what the plugin bundles; the explicit `wasm-opt` pass is not
# a nicety but the only one that works here — see the note in
# crates/gnosis-wasm/Cargo.toml.
set -euo pipefail

# Derived from this script's own location, not from `git rev-parse` or the
# working directory: the plugin repo calls this script to vendor its copy of
# the query client, and `--show-toplevel` would hand back *that* repo.
root="$(cd "$(dirname "$0")/../.." && pwd)"
out="${1:-$root/crates/gnosis-wasm/pkg}"

wasm-pack build "$root/crates/gnosis-wasm" \
    --target web \
    --profile wasm-release \
    --out-dir "$out" \
    --no-opt

wasm="$out/gnosis_wasm_bg.wasm"
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

printf '%s\n' "$out/gnosis_wasm_bg.wasm: $before bytes linked, $after after wasm-opt"
printf '%s\n' "  gzipped: $(gzip -9 -c "$wasm" | wc -c) bytes"
printf '%s\n' "  JS glue: $(stat -c %s "$out/gnosis_wasm.js") bytes"
