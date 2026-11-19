#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
# Use Python 3.10 or 3.11 supplied by the GPU host.
"${PYTHON:-python3.10}" -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
mkdir -p vendor
if [ ! -d vendor/le-wm ]; then
  git clone https://github.com/lucas-maes/le-wm.git vendor/le-wm
  git -C vendor/le-wm checkout --detach 8edfeb336732b5f3ce7b8b210d0ba370a09e2cac
fi
if [ ! -d vendor/stable-worldmodel ]; then
  git clone https://github.com/galilai-group/stable-worldmodel.git vendor/stable-worldmodel
  git -C vendor/stable-worldmodel checkout --detach abdced49809d5eae38e24b27dc7b635c502c4812
fi
test "$(git -C vendor/le-wm rev-parse HEAD)" = 8edfeb336732b5f3ce7b8b210d0ba370a09e2cac
test "$(git -C vendor/stable-worldmodel rev-parse HEAD)" = abdced49809d5eae38e24b27dc7b635c502c4812
python -m pip install './vendor/stable-worldmodel[train,format]' \
  'stable-pretraining==0.1.6' 'transformers>=4.50,<5' 'numpy<2' 'mujoco<3.12' dm-control ogbench \
  pygame pymunk shapely opencv-python-headless scikit-learn \
  huggingface-hub zstandard matplotlib
python prepare_source.py
python -m pip freeze > environment-lock.txt
python run.py doctor
