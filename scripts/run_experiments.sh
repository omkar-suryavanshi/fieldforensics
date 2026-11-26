#!/bin/bash
# Reproduces every experiment in the paper (sequential; ~10-15 GPU hours on an RTX 3060 laptop GPU).
set -e
cd "$(dirname "$0")/../src"
PY=${PYTHON:-python}
mkdir -p ../runs
$PY build_cache.py                                   # verify pairs, de-duplicate, build splits + npy cache
train(){ n=$1; shift; [ -f ../runs/$n/train_log.json ] || $PY train.py --name $n "$@"; }
ev(){ $PY evaluate.py --name "$@"; }
MAPS="000000001,000001004,000002723,000003646"
train noaug --aug 0                                  # FieldForensics (primary, clean-trained)
ev noaug --conds none,jpeg75,social --save_maps $MAPS
train main_s0                                        # FieldForensics-R (degradation-augmented)
ev main_s0 --conds none,jpeg90,jpeg75,scale0.5,social --save_maps $MAPS
train unpaired --paired 0;  ev unpaired --conds none # ablations (augmented configuration)
train rgb --noise 0;        ev rgb --conds none
train weak --seg_w 0;       ev weak --conds none
train loso_cord --holdout cord; ev loso_cord --conds none --set src:cord
$PY gcnn.py                                          # global ResNet-50 baseline
$PY baselines.py                                     # pixel LR, forensic features + LR/GBDT
$PY feat_importance.py
$PY analyze.py                                       # metrics, bootstrap CIs, significance, figures
$PY build_paper.py                                   # paper/FieldForensics_paper.pdf + .tex
