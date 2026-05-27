#!/usr/bin/env bash
# Create Ultralytics-style images/ + labels/ symlinks to LLVIP (no copy, no change to multispectral repo).
set -euo pipefail

ULTRA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE="$(dirname "$ULTRA_ROOT")"

if [[ -n "${LLVIP_ROOT:-}" ]]; then
  LLVIP="$LLVIP_ROOT"
elif [[ -d "$WORKSPACE/LLVIP/visible/train" ]]; then
  LLVIP="$WORKSPACE/LLVIP"
else
  LLVIP="$WORKSPACE/multispectral-object-detection/LLVIP"
fi

if [[ ! -d "$LLVIP/visible/train" ]]; then
  echo "Missing $LLVIP"
  echo "Put LLVIP at \$WORKSPACE/LLVIP or run: bash tools/setup_new_machine.sh"
  exit 1
fi

link_split() {
  local name=$1 mod=$2 split=$3
  local base="$ULTRA_ROOT/datasets/$name"
  local src="$LLVIP/$mod/$split"
  mkdir -p "$base/images" "$base/labels"
  ln -sfn "$src" "$base/images/$split"
  ln -sfn "$src" "$base/labels/$split"
  echo "OK $base -> $src"
}

link_split llvip_rgb visible train
link_split llvip_rgb visible test
link_split llvip_ir infrared train
link_split llvip_ir infrared test

echo "Done. Use data=ultralytics/cfg/datasets/llvip_rgb.yaml or llvip_ir.yaml"
