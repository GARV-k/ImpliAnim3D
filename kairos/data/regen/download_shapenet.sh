#!/bin/bash
# Login node (Internet). First step of regenerating the data from scratch:
#   1. download ShapeNetCore.v1.zip (32 GB) from HuggingFace. The dataset ShapeNet/ShapeNetCore-archive is
#      gated: run `hf auth login` once with an account that accepted the ShapeNet terms;
#   2. extract the first N models of every synset (default 200) and write their split file.
# A step is skipped when its output exists. kairos/data/link_shapenet.sh points $IA3D_DATA/raw/ShapeNetCore.v1
# at the existing download, so with those links in place this does nothing for N = 200, and a different N is
# extracted next to the existing extraction.
# Usage: kairos/data/regen/download_shapenet.sh [N]
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/../../config.sh"
N=${1:-200}
RAW=${IA3D_SHAPENET_RAW:-$IA3D_DATA/raw/ShapeNetCore.v1}
ZIP=$RAW/ShapeNetCore.v1.zip
EXT=$RAW/extracted_$N
mkdir -p "$RAW"

if [ -f "$ZIP" ]; then
  echo "### $ZIP exists, not downloading"
else
  # The default CA path of the login-node Python is unreadable, and the Xet transfer backend cannot reach
  # cas-server.xethub.hf.co from here; plain HTTPS works.
  export SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt CURL_CA_BUNDLE=/etc/pki/tls/certs/ca-bundle.crt
  export HF_HUB_DISABLE_XET=1
  HF=${IA3D_HF:-$(command -v hf || true)}
  if [ -z "$HF" ]; then  # x86 helper venv for the login node, kept with the data
    python3 -m venv "$IA3D_DATA/.tools-venv"
    "$IA3D_DATA/.tools-venv/bin/pip" install -q huggingface_hub
    HF=$IA3D_DATA/.tools-venv/bin/hf
  fi
  echo "### downloading ShapeNetCore.v1.zip (32 GB) -> $RAW ($(date))"
  "$HF" download ShapeNet/ShapeNetCore-archive ShapeNetCore.v1.zip --repo-type dataset --local-dir "$RAW"
fi

if [ -d "$EXT/ShapeNetCore.v1" ]; then
  echo "### $EXT/ShapeNetCore.v1 exists, not extracting"
else
  echo "### extracting the first $N models of every synset -> $EXT ($(date))"
  # Extracted under another name and renamed at the end, so that an interrupted extraction is not taken for a
  # complete one by the test above. A rerun extracts over what is already there.
  python3 "$HERE/extract_subset.py" "$ZIP" "$EXT.partial" "$N" "$EXT.partial/split.json"
  if [ "$N" = 200 ]; then
    cmp "$EXT.partial/split.json" "$HERE/../splits/shapenet_v1_200_per_category.json" \
      || { echo "this zip does not give the split in kairos/data/splits (extraction left in $EXT.partial)" >&2; exit 1; }
    echo "split identical to the one in the repo"
  fi
  mv "$EXT.partial" "$EXT"
fi
echo DOWNLOAD_DONE
