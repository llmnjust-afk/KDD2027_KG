#!/usr/bin/env bash
# Supplementary experiments: 2a validation, no-graph/vector-RAG baselines, sweep.
# Runs after the main experiments, then pushes everything to GitHub.
set -e
cd "$(dirname "$0")/.."
export HF_HUB_DISABLE_PROGRESS_BARS=1
GEN="Qwen/Qwen2.5-1.5B-Instruct"

echo "============================================================"
echo "[$(date +%H:%M:%S)] SUPP-1: 2a noisy-linking validation (3 hops)"
echo "============================================================"
for SPLIT in 1-hop 2-hop 3-hop; do
  python3 run.py --dataset metaqa --split "$SPLIT" --data-dir ./data/MetaQA \
      --gen-model "$GEN" --limit 300 --noisy-linking \
      --systems fixed adaptive abl-nograph \
      --out-dir ./results_noisy 2>&1 | tee "results_log_noisy_${SPLIT}.txt"
done

echo "============================================================"
echo "[$(date +%H:%M:%S)] SUPP-2: no-graph + vector-RAG baselines (3 hops)"
echo "============================================================"
for SPLIT in 1-hop 2-hop 3-hop; do
  python3 run.py --dataset metaqa --split "$SPLIT" --data-dir ./data/MetaQA \
      --gen-model "$GEN" --limit 200 \
      --systems nograph vector-rag fixed adaptive \
      --vector-rag-topk 8 \
      --out-dir ./results_baselines 2>&1 | tee "results_log_baselines_${SPLIT}.txt"
done

echo "============================================================"
echo "[$(date +%H:%M:%S)] SUPP-3: hyperparameter sensitivity sweep (2-hop)"
echo "============================================================"
python3 scripts/sweep.py --split 2-hop --data-dir ./data/MetaQA \
    --gen-model "$GEN" --limit 100 2>&1 | tee results_log_sweep.txt

echo "============================================================"
echo "[$(date +%H:%M:%S)] SUPP-4: significance + analysis on all splits"
echo "============================================================"
for SPLIT in 1-hop 2-hop 3-hop; do
  echo "--- significance $SPLIT ---"
  python3 scripts/significance.py --results-dir ./results --split "$SPLIT" --n-boot 10000
  echo "--- analysis $SPLIT ---"
  python3 scripts/analyze.py --results-dir ./results --split "$SPLIT"
done > results/significance_analysis_all.txt 2>&1
cat results/significance_analysis_all.txt

echo "============================================================"
echo "[$(date +%H:%M:%S)] Pushing supplementary results to GitHub"
echo "============================================================"
if [ -z "$AGR_GH_TOKEN" ]; then
  echo "AGR_GH_TOKEN not set; skipping push."
else
  git add -A results_noisy results_baselines results results_log_*.txt 2>/dev/null || true
  git -c user.name=llmnjust-afk -c user.email=llmnjust-afk@users.noreply.github.com \
      commit -m "results: supplementary experiments (2a validation, baselines, sweep, significance)" 2>/dev/null || true
  AGR_GH_TOKEN="$AGR_GH_TOKEN" python3 scripts/push_results.py \
      --repo llmnjust-afk/KDD2027_KG \
      --message "results: supplementary experiments (2a validation, baselines, sweep, significance)" 2>/dev/null || true
fi
echo "============================================================"
echo "[$(date +%H:%M:%S)] ALL SUPPLEMENTARY EXPERIMENTS DONE"
echo "============================================================"
