#!/usr/bin/env bash
# Write the third-party notices for the wasm query client: every crate it links
# for wasm32, grouped by license, each with the license text the crate ships.
# Fails on a crate whose license is not in about.toml's accepted list, or whose
# license cannot be determined.
#
# Needs cargo-about (script/ci/install-wasm-tools.sh installs the pinned one).
#
# Usage: notices.sh <out-file>
set -euo pipefail

out="${1:?usage: notices.sh <out-file>}"
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../.." && pwd)"

cargo about generate --fail --locked \
    --config "$here/about.toml" \
    --manifest-path "$root/crates/gnosis-wasm/Cargo.toml" \
    --output-file "$out" \
    "$here/about.hbs"
