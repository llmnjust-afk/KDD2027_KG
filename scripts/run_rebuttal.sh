#!/usr/bin/env bash
# Rebuttal experiments addressing reviewer 3.1/3.3/3.4.
set -e
cd "$(dirname "$0")/.."
export HF_HUB_DISABLE_PROGRESS_BARS=1
G7="Qwen/Qwen2.5-7B-Instruct"
G15="Qwen/Qwen2.5-1.5B-Instruct"

echo "============================================================"
echo "[$(date +%H:%M:%S)] EXP-R1: Fixed-K sweep @ 7B/2-hop (reviewer 3.1)"
echo "  Fixed K=1, K=2, K=3, and budget-matched, vs Adaptive"
echo "============================================================"
# Fixed with K=1
python3 run.py --dataset metaqa --split 2-hop --data-dir ./data/MetaQA \
    --gen-model "$G7" --limit 200 --systems fixed --max-hops 1 \
    --out-dir ./results_fixedK/k1 2>&1 | tee results_log_fixedK1.txt
# Fixed with K=2
python3 run.py --dataset metaqa --split 2-hop --data-dir ./data/MetaQA \
    --gen-model "$G7" --limit 200 --systems fixed --max-hops 2 \
    --out-dir ./results_fixedK/k2 2>&1 | tee results_log_fixedK2.txt
# Fixed with K=3 + Adaptive (reference; already have but re-run for same subset)
python3 run.py --dataset metaqa --split 2-hop --data-dir ./data/MetaQA \
    --gen-model "$G7" --limit 200 --systems fixed adaptive --max-hops 3 \
    --out-dir ./results_fixedK/k3 2>&1 | tee results_log_fixedK3.txt
# Fixed beam sweep at K=2 (budget variation): beam 2 and 6
python3 run.py --dataset metaqa --split 2-hop --data-dir ./data/MetaQA \
    --gen-model "$G7" --limit 200 --systems fixed --max-hops 2 --beam 2 \
    --out-dir ./results_fixedK/k2b2 2>&1 | tee results_log_fixedK2b2.txt
python3 run.py --dataset metaqa --split 2-hop --data-dir ./data/MetaQA \
    --gen-model "$G7" --limit 200 --systems fixed --max-hops 2 --beam 6 \
    --out-dir ./results_fixedK/k2b6 2>&1 | tee results_log_fixedK2b6.txt

echo "============================================================"
echo "[$(date +%H:%M:%S)] EXP-R2: Mixed 1/2/3-hop query stream @ 7B (reviewer 3.3)"
echo "============================================================"
python3 run.py --dataset metaqa --mixed --data-dir ./data/MetaQA \
    --gen-model "$G7" --limit 300 --systems fixed adaptive \
    --out-dir ./results_mixed 2>&1 | tee results_log_mixed_7b.txt
python3 run.py --dataset metaqa --mixed --data-dir ./data/MetaQA \
    --gen-model "$G15" --limit 300 --systems fixed adaptive \
    --out-dir ./results_mixed 2>&1 | tee results_log_mixed_15b.txt

echo "============================================================"
echo "[$(date +%H:%M:%S)] EXP-R3: 3 seeds @ 7B/2-hop (reviewer 3.4)"
echo "============================================================"
for S in 1 2 3; do
  python3 run.py --dataset metaqa --split 2-hop --data-dir ./data/MetaQA \
      --gen-model "$G7" --limit 200 --systems fixed adaptive --seed $S \
      --out-dir ./results_seeds 2>&1 | tee "results_log_seed${S}.txt"
done

echo "============================================================"
echo "[$(date +%H:%M:%S)] ALL REBUTTAL EXPERIMENTS DONE"
echo "============================================================"
