#!/bin/bash
# Download the data XMimic needs, from the Hugging Face dataset lvhaidong/LAFAN1_Retargeting_Dataset (the same source as
# BlinkMimic's and InstantTrack's download scripts):
#   data/lafan1_g1/          the 40 G1-retargeted LAFAN1 clips (LAFAN1 by Ubisoft La Forge: CC BY-NC-ND 4.0)
#   assets/g1/assets/meshes/ the G1 meshes the scene uses (Unitree G1 description: BSD-3-Clause)
# and check them against tools/data_sha256.txt (the files behind the published results).
set -euo pipefail
cd "$(dirname "$0")/.."
HF=https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset
API=https://huggingface.co/api/datasets/lvhaidong/LAFAN1_Retargeting_Dataset/tree/main
mkdir -p data/lafan1_g1 assets/g1/assets/meshes
for f in $(curl -sf "$API/g1" | python3 -c "import json, sys; [print(x['path']) for x in json.load(sys.stdin)]"); do
  curl -sfL -o "data/lafan1_g1/$(basename "$f")" "$HF/resolve/main/$f"
done
for m in $(grep -o 'file="meshes/[^"]*"' assets/g1/scene.xml | sed -e 's|file="meshes/||' -e 's|"||'); do
  curl -sfL -o "assets/g1/assets/meshes/$m" "$HF/resolve/main/robot_description/g1/meshes/$m"
done
echo "$(ls data/lafan1_g1 | wc -l | tr -d ' ') clips, $(ls assets/g1/assets/meshes | wc -l | tr -d ' ') meshes"
if command -v sha256sum >/dev/null; then sha256sum -c --quiet tools/data_sha256.txt; else shasum -a 256 -c --quiet tools/data_sha256.txt; fi
echo "checksums OK"
