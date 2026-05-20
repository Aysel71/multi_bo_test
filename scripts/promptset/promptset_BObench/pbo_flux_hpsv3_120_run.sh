#!/bin/bash
# Method B (MultiBO) on the 120-prompt benchmark.
#   model:  FLUX.1-schnell (4 steps — set in configs/bo_config_flux.py)
#   mode:   non-human, HPSv3-driven  (BO optimises HPSv3)
#   acqf:   manifold-dbs
# Runs one job per category over promptsets/BObench/<category>.txt
# (those 8 files are produced by convert_prompts_to_bobench.py).
#
# Output:
#   outputs/promptset/promptset_BObench/MultiBO-non-human_hpsv3/low-image/flux/
# Logs:  logs/flux-hpsv3-<category>.log
#
# Single-GPU sequential. Override GPU / CATEGORIES via env, e.g.:
#   GPU=1 CATEGORIES="color shape" bash .../pbo_flux_hpsv3_120_run.sh

set -u

GPU="${GPU:-0}"
OBJ="flux"
MODE="low-image"
ALGOS="manifold-dbs"
METRIC="hpsv3"
RESULT_PATH="./outputs/promptset/promptset_BObench/MultiBO-non-human_${METRIC}/${MODE}/${OBJ}/"
CATEGORIES="${CATEGORIES:-color shape texture spatial 3d_spatial non_spatial numeracy complex}"

export CUDA_VISIBLE_DEVICES="$GPU"
mkdir -p logs

echo "============================================================"
echo "Method B — FLUX.1-schnell / non-human / HPSv3"
echo "  GPU:        $GPU"
echo "  categories: $CATEGORIES"
echo "  output:     $RESULT_PATH"
echo "============================================================"

for category in $CATEGORIES; do
    PROMPTSET="promptsets/BObench/${category}.txt"
    LOG="logs/flux-hpsv3-${category}.log"
    if [ ! -f "$PROMPTSET" ]; then
        echo "[skip] $PROMPTSET not found — run convert_prompts_to_bobench.py first"
        continue
    fi
    echo "[$(date +%H:%M:%S)] start $category  ->  $LOG"
    python -u scripts/promptset/promptset_BObench/pbo_manifold_flux.py \
        --acf "$ALGOS" --output_path "$RESULT_PATH" \
        --q 4 --T 4 --mode "$MODE" --logit True --multi True \
        --obj_name "$OBJ" --prompts_per_category -1 \
        --promptset_path "$PROMPTSET" --use_random_seeds True \
        --non_human_score True --score_metric "$METRIC" \
        > "$LOG" 2>&1
    rc=$?
    echo "[$(date +%H:%M:%S)] done  $category  (exit $rc)"
done

echo "all categories done"
