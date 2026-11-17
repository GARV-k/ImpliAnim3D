#!/bin/bash
# Expose the preprocessed ShapeNetCore.v1 subset (and the raw download it was made from) in the stage
# directory as symlinks. Nothing is copied. Idempotent; runs on the login node.
#
#   $IA3D_DATA/shapenet/<synset>     -> $IA3D_SHAPENET_SRC/<synset>      one link per synset of the split file
#   $IA3D_DATA/raw/ShapeNetCore.v1   -> $IA3D_SHAPENET_RAW_SRC           only needed to regenerate (kairos/data/regen)
#
# A synset that already exists as a real directory (regenerated or copied data) is left alone.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/../config.sh"
SRC=${IA3D_SHAPENET_SRC:-${WORK:?WORK is not set}/MeshSDF/data}
RAW_SRC=${IA3D_SHAPENET_RAW_SRC:-$WORK/datasets/ShapeNetCore.v1}
SPLIT=$HERE/splits/shapenet_v1_200_per_category.json
DST=$IA3D_DATA/shapenet

link() {  # link <target> <link path>: prints what happened, fails if the link could not be made
  local target=$1 path=$2
  if [ -L "$path" ]; then
    if [ "$(readlink "$path")" = "$target" ]; then echo same; else ln -sfn "$target" "$path" && echo relinked; fi
  elif [ -e "$path" ]; then echo kept
  else ln -s "$target" "$path" && echo linked
  fi
}

[ -d "$SRC" ] || { echo "source data not found: $SRC (set IA3D_SHAPENET_SRC)" >&2; exit 1; }
mkdir -p "$DST" "$IA3D_DATA/raw"
synsets=$(python3 -c 'import json, sys; print(" ".join(sorted(json.load(open(sys.argv[1])))))' "$SPLIT")
n_linked=0 n_same=0 n_kept=0 n_missing=0 n_failed=0
for syn in $synsets; do
  if [ ! -d "$SRC/$syn/meshes" ]; then echo "missing in source: $SRC/$syn/meshes" >&2; n_missing=$((n_missing + 1)); continue; fi
  # the status of link is checked here because set -e does not act inside $(...)
  state=$(link "$SRC/$syn" "$DST/$syn") || { echo "could not link $DST/$syn" >&2; n_failed=$((n_failed + 1)); continue; }
  case $state in
    linked|relinked) n_linked=$((n_linked + 1)) ;;
    same) n_same=$((n_same + 1)) ;;
    kept) n_kept=$((n_kept + 1)); echo "kept real directory $DST/$syn" ;;
  esac
done
echo "shapenet: $DST <- $SRC: $n_linked linked, $n_same already linked, $n_kept real directories kept, $n_missing missing, $n_failed failed"

if [ -d "$RAW_SRC" ]; then
  state=$(link "$RAW_SRC" "$IA3D_DATA/raw/ShapeNetCore.v1") || { n_failed=$((n_failed + 1)); state="could not be linked"; }
  echo "raw: $IA3D_DATA/raw/ShapeNetCore.v1 <- $RAW_SRC: $state"
else
  echo "raw: $RAW_SRC not found, not linked (only needed by kairos/data/regen)"
fi
[ "$n_missing" -eq 0 ] && [ "$n_failed" -eq 0 ]
