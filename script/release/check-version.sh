#!/usr/bin/env bash
# Refuse a release tag that does not name the workspace version: tag v1.2.3
# must meet version = "1.2.3" in [workspace.package], which every crate takes.
#
# Usage: check-version.sh <tag>
set -euo pipefail

tag="${1:?usage: check-version.sh <tag>}"
root="$(cd "$(dirname "$0")/../.." && pwd)"

versions="$(cargo metadata --manifest-path "$root/Cargo.toml" --no-deps --format-version 1 |
    jq -r '[.packages[].version] | unique | .[]')"

if [ "$(printf '%s\n' "$versions" | wc -l)" -ne 1 ]; then
    printf 'error: the crates disagree on their version:\n%s\n' "$versions" >&2
    exit 1
fi
if [ "$tag" != "v$versions" ]; then
    printf 'error: tag %s does not match the workspace version %s\n' "$tag" "$versions" >&2
    exit 1
fi
printf 'tag %s matches the workspace version\n' "$tag"
