#!/usr/bin/env bash
# Generate a synthetic vault for benchmarking indexing throughput: N markdown
# files (multi-heading, multi-paragraph, so each produces several chunks —
# representative of real notes) and optionally M minimal valid PNGs.
set -euo pipefail

dir="${1:?usage: generate-vault.sh <dir> <file-count> [image-count]}"
file_count="${2:?usage: generate-vault.sh <dir> <file-count> [image-count]}"
image_count="${3:-0}"

mkdir -p "$dir"

sentences=(
  "The quick brown fox jumps over the lazy dog near the riverbank at dawn."
  "Distributed systems trade consistency for availability under network partition."
  "A well-tuned garden requires patience, water, and consistent afternoon light."
  "The old lighthouse keeper recorded every storm in a leather-bound journal."
  "Vector embeddings place semantically similar sentences close together in space."
  "Sourdough starters need daily feeding to stay active and rise properly."
  "The committee debated the proposal for three hours before reaching consensus."
  "Mountain trails above the treeline offer clear views on cloudless mornings."
  "Rust's borrow checker prevents data races at compile time, not runtime."
  "The orchestra tuned to the oboe's steady A before the conductor arrived."
)
n=${#sentences[@]}

paragraph() {
  local seed=$1
  local out=""
  for i in 0 1 2 3 4 5; do
    idx=$(( (seed + i * 7) % n ))
    out+="${sentences[$idx]} "
  done
  printf '%s\n' "$out"
}

for i in $(seq 1 "$file_count"); do
  {
    echo "# Note $i"
    echo
    paragraph "$i"
    echo
    echo "## Details"
    echo
    paragraph "$((i * 3))"
    echo
    echo "## More context"
    echo
    paragraph "$((i * 5))"
  } > "$dir/note-$i.md"
done

if [ "$image_count" -gt 0 ]; then
  mkdir -p "$dir/images"
  python3 - "$dir/images" "$image_count" <<'PY'
import sys, pathlib

out_dir, count = sys.argv[1], int(sys.argv[2])
# Minimal valid 2x1 PNG (red/blue pixel) — same bytes gnosis-fs's own tests embed.
png = bytes([
    0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00, 0x00, 0x0D,
    0x49, 0x48, 0x44, 0x52, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x01,
    0x08, 0x02, 0x00, 0x00, 0x00, 0x7B, 0x40, 0xE8, 0xDD, 0x00, 0x00, 0x00,
    0x0F, 0x49, 0x44, 0x41, 0x54, 0x78, 0x9C, 0x63, 0xF8, 0xCF, 0xC0, 0xC0,
    0xC0, 0xF0, 0x1F, 0x00, 0x07, 0x00, 0x01, 0xFF, 0x7E, 0x08, 0xB1, 0xD0,
    0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82,
])
p = pathlib.Path(out_dir)
for i in range(1, count + 1):
    (p / f"img-{i}.png").write_bytes(png)
PY
fi

echo "Generated $file_count markdown files and $image_count images in $dir"
