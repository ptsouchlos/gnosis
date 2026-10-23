#!/usr/bin/env bash
# Package the wasm query client as a release asset: the files the Obsidian
# plugin vendors, at the archive's root, plus a GNOSIS_REV naming the commit and
# the tag, gnosis's LICENSE, and the notices for the crates it links. Writes
# gnosis-wasm-<tag>.tar.gz and its .sha256 into <out-dir>.
#
# The plugin unpacks this into vendor/gnosis-wasm/ as is, so it holds exactly
# what that directory commits: wasm-pack's npm package.json and .gitignore are
# dropped, as the plugin's own update script drops them.
#
# Usage: package-wasm.sh <tag> <out-dir>
set -euo pipefail

tag="${1:?usage: package-wasm.sh <tag> <out-dir>}"
out="${2:?usage: package-wasm.sh <tag> <out-dir>}"
root="$(cd "$(dirname "$0")/../.." && pwd)"

stage="$(mktemp -d)"
trap 'rm -rf "$stage"' EXIT

"$root/script/wasm/build.sh" "$stage"
rm -f "$stage/.gitignore" "$stage/package.json"
{
    git -C "$root" rev-parse HEAD
    printf 'release %s\n' "$tag"
} >"$stage/GNOSIS_REV"
cp "$root/LICENSE" "$stage/LICENSE"
"$root/script/release/notices.sh" "$stage/THIRD_PARTY_NOTICES.txt"

mkdir -p "$out"
name="gnosis-wasm-$tag.tar.gz"
# Sorted names, fixed owner and mtime, and gzip without a timestamp: the same
# files always give the same archive bytes. The list is sorted here because
# tar's --sort applies only to directories it walks, not to --files-from.
(
    cd "$stage"
    find . -mindepth 1 -maxdepth 1 -printf '%P\n' | LC_ALL=C sort |
        tar --owner=0 --group=0 --numeric-owner \
            --mtime=@0 -cf - --files-from=-
) | gzip -9 -n >"$out/$name"
(
    cd "$out"
    sha256sum "$name" >"$name.sha256"
)

printf '%s\n' "$out/$name" "$out/$name.sha256"
