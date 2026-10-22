#!/usr/bin/env bash
# Install the wasm toolchain CI builds the query client with: wasm-pack,
# wasm-bindgen (with its test runner) and binaryen's wasm-opt, each pinned by
# version and SHA-256, on x86_64 Linux.
#
# wasm-bindgen must equal the `wasm-bindgen` crate pin in
# crates/gnosis-wasm/Cargo.toml; the CLI refuses to work across versions.
#
# Usage: install-wasm-tools.sh <dir>
# Puts the binaries in <dir>, and on GitHub Actions adds <dir> to the PATH.
set -euo pipefail

dir="${1:?usage: install-wasm-tools.sh <dir>}"
mkdir -p "$dir"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

# fetch <url> <sha256>: download into $tmp, verify, and unpack there.
fetch() {
    local url="$1" sha="$2" file
    file="$tmp/$(basename "$url")"
    curl --fail --silent --show-error --location --retry 3 --output "$file" "$url"
    echo "$sha  $file" | sha256sum --check --quiet
    tar -xzf "$file" -C "$tmp"
}

fetch https://github.com/rustwasm/wasm-pack/releases/download/v0.15.0/wasm-pack-v0.15.0-x86_64-unknown-linux-musl.tar.gz \
    c09f971ecaed9a2efc80fdcea7a00ef6b53c7fadc8c57d1f61b53a6aa66b668a
cp "$tmp/wasm-pack-v0.15.0-x86_64-unknown-linux-musl/wasm-pack" "$dir/"

fetch https://github.com/wasm-bindgen/wasm-bindgen/releases/download/0.2.129/wasm-bindgen-0.2.129-x86_64-unknown-linux-musl.tar.gz \
    82d12bb940e2d4e72e0d5605387fc1b8ca179044e012b620f0ce4e7440e8320e
cp "$tmp/wasm-bindgen-0.2.129-x86_64-unknown-linux-musl/wasm-bindgen" \
    "$tmp/wasm-bindgen-0.2.129-x86_64-unknown-linux-musl/wasm-bindgen-test-runner" "$dir/"

fetch https://github.com/WebAssembly/binaryen/releases/download/version_116/binaryen-version_116-x86_64-linux.tar.gz \
    c55b74f3109cdae97490faf089b0286d3bba926bb6ea5ed00c8c784fc53718fd
cp "$tmp/binaryen-version_116/bin/wasm-opt" "$dir/"

if [ -n "${GITHUB_PATH:-}" ]; then
    echo "$dir" >>"$GITHUB_PATH"
fi
