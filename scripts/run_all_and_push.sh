#!/usr/bin/env bash
# Orchestrate remaining experiments and auto-push results to GitHub.
# Waits for any in-flight Phase-2 job, then runs Phase 3 (3-hop) and
# Phase 4 (7B, 1-hop), then pushes everything via push_results.py.
#
# Run in background:  nohup bash scripts/run_all_and_push.sh > run_all.log 2>&1 &
set -e
cd "$(dirname "$0")/.."
export HF_HUB_DISABLE_PROGRESS_BARS=1

GEN_15B="Qwen/Qwen2.5-1.5B-Instruct"
GEN_7B="Qwen/Qwen2.5-7B-Instruct"

echo "============================================================"
echo "[$(date +%H:%M:%S)] Waiting for any in-flight 2-hop job..."
echo "============================================================"
# Wait while a run.py process is active (covers the Phase-2 job already running)
while pgrep -f "run.py.*2-hop" >/dev/null 2>&1; do
  sleep 30
done
echo "[$(date +%H:%M:%S)] Phase 2 (2-hop) finished."

echo "============================================================"
echo "[$(date +%H:%M:%S)] Phase 3: 3-hop, 5 systems, 300 questions"
echo "============================================================"
python3 run.py --dataset metaqa --split 3-hop --data-dir ./data/MetaQA \
    --gen-model "$GEN_15B" --limit 300 \
    --systems fixed adaptive abl-nograph abl-fixbeam abl-nostop \
    --out-dir ./results 2>&1 | tee results_log_3hop.txt

echo "============================================================"
echo "[$(date +%H:%M:%S)] Phase 4: 7B model, 1-hop, baseline+adaptive, 200 questions"
echo "============================================================"
python3 run.py --dataset metaqa --split 1-hop --data-dir ./data/MetaQA \
    --gen-model "$GEN_7B" --limit 200 \
    --systems fixed adaptive \
    --out-dir ./results_7b 2>&1 | tee results_log_7b.txt

echo "============================================================"
echo "[$(date +%H:%M:%S)] Summarising all Pareto tables"
echo "============================================================"
{
  echo "MetaQA 1-hop (Qwen2.5-1.5B, n=500)"
  cat results/pareto_metaqa_1-hop.txt 2>/dev/null
  echo; echo "MetaQA 2-hop (Qwen2.5-1.5B, n=300)"
  cat results/pareto_metaqa_2-hop.txt 2>/dev/null
  echo; echo "MetaQA 3-hop (Qwen2.5-1.5B, n=300)"
  cat results/pareto_metaqa_3-hop.txt 2>/dev/null
  echo; echo "MetaQA 1-hop (Qwen2.5-7B, n=200)"
  cat results_7b/pareto_metaqa_1-hop.txt 2>/dev/null
} > results/SUMMARY_all.txt
cat results/SUMMARY_all.txt

echo "============================================================"
echo "[$(date +%H:%M:%S)] Pushing results to GitHub"
echo "============================================================"
if [ -z "$AGR_GH_TOKEN" ]; then
  echo "AGR_GH_TOKEN not set; skipping push. Results are in ./results and ./results_7b"
else
  AGR_GH_TOKEN="$AGR_GH_TOKEN" python3 scripts/push_results.py \
      --repo llmnjust-afk/KDD2027_KG \
      --message "results: full MetaQA 1/2/3-hop ablations + 7B scalability (auto)"
  # also track the 7b results dir + logs
  git add -A results_7b results_log_*.txt run_all.log 2>/dev/null || true
  git -c user.name=llmnjust-afk -c user.email=llmnjust-afk@users.noreply.github.com \
      commit -m "chore: add 7B results + experiment logs" 2>/dev/null || true
  AGR_GH_TOKEN="$AGR_GH_TOKEN" python3 scripts/push_results.py \
      --repo llmnjust-afk/KDD2027_KG \
      --message "chore: add 7B results + experiment logs" 2>/dev/null || true
fi

echo "============================================================"
echo "[$(date +%H:%M:%S)] ALL DONE"
echo "============================================================"
