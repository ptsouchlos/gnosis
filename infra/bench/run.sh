#!/usr/bin/env bash
# Time indexing a synthetic vault, cold and then re-indexed unchanged (the
# incremental skip path), using the current gnosis.toml config in this repo
# (embed model, batch_size, ann.quantization) — so `just bench` gives a
# comparable number when tuning those knobs.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../.." && pwd)"

file_count="${1:-2000}"
image_count="${2:-200}"

vault_dir="$(mktemp -d -t gnosis-bench-XXXXXX)"
trap 'rm -rf "$vault_dir"' EXIT

echo "Building gnosis (release)..."
cargo build --release -q --manifest-path "$repo_root/Cargo.toml" -p gnosis-cli

echo "Generating synthetic vault: $file_count markdown files, $image_count images..."
"$script_dir/generate-vault.sh" "$vault_dir" "$file_count" "$image_count"

if [ "$image_count" -gt 0 ]; then
  cat > "$vault_dir/gnosis.toml" <<'EOF'
[embed.image]
enabled = true
EOF
fi

bin="$repo_root/target/release/gnosis"

echo
echo "=== Cold index ==="
( cd "$vault_dir" && time "$bin" index . )

echo
echo "=== Re-index (unchanged) ==="
( cd "$vault_dir" && time "$bin" index . )
