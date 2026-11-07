#!/usr/bin/env bash
# Package the wasm bundles the Obsidian plugin vendors as release assets: the
# files it vendors, at the archive's root, plus a GNOSIS_REV naming the commit
# and the tag, gnosis's LICENSE, and the notices for the crates the bundle
# links. Writes gnosis-wasm-<tag>.tar.gz and gnosis-extract-<tag>.tar.gz, each
# with its .sha256, into <out-dir>.
#
# The plugin unpacks each into a vendor directory as is, so each archive holds
# exactly what that directory commits: wasm-pack's npm package.json and
# .gitignore are dropped, as the plugin's own update script drops them.
#
# The notices are generated once and copied into both archives: the query
# client's graph covers the extraction bundle's — see script/release/notices.sh.
#
# Usage: package-wasm.sh <tag> <out-dir>
set -euo pipefail

tag="${1:?usage: package-wasm.sh <tag> <out-dir>}"
out="${2:?usage: package-wasm.sh <tag> <out-dir>}"
root="$(cd "$(dirname "$0")/../.." && pwd)"

stage_wasm="$(mktemp -d)"
stage_extract="$(mktemp -d)"
notices="$(mktemp)"
trap 'rm -rf "$stage_wasm" "$stage_extract" "$notices"' EXIT

"$root/script/wasm/build.sh" "$stage_wasm"
"$root/script/wasm/build-extract.sh" "$stage_extract"
rm -f "$stage_wasm/.gitignore" "$stage_wasm/package.json" \
      "$stage_extract/.gitignore" "$stage_extract/package.json"
for stage in "$stage_wasm" "$stage_extract"; do
    {
        git -C "$root" rev-parse HEAD
        printf 'release %s\n' "$tag"
    } >"$stage/GNOSIS_REV"
    cp "$root/LICENSE" "$stage/LICENSE"
done
"$root/script/release/notices.sh" "$notices"
cp "$notices" "$stage_wasm/THIRD_PARTY_NOTICES.txt"
cp "$notices" "$stage_extract/THIRD_PARTY_NOTICES.txt"

# Sorted names, fixed owner and mtime, and gzip without a timestamp: the same
# files always give the same archive bytes. The list is sorted here because
# tar's --sort applies only to directories it walks, not to --files-from.
package() {
    local stage="$1" name="$2"
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
}

mkdir -p "$out"
package "$stage_wasm" "gnosis-wasm-$tag.tar.gz"
package "$stage_extract" "gnosis-extract-$tag.tar.gz"
