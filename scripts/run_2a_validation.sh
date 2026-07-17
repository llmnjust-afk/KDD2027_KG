#!/usr/bin/env bash
# Focused 2a validation: corrupt entity mentions + high theta_low so 2a fires.
# Waits for the main supp experiments to finish, then runs this.
set -e
cd "$(dirname "$0")/.."
export HF_HUB_DISABLE_PROGRESS_BARS=1
GEN="Qwen/Qwen2.5-1.5B-Instruct"

echo "============================================================"
echo "[$(date +%H:%M:%S)] Waiting for main supp experiments to finish..."
echo "============================================================"
while pgrep -f "run_supp_and_push" >/dev/null 2>&1; do sleep 30; done
echo "[$(date +%H:%M:%S)] Main supp done. Starting 2a corruption validation."

echo "============================================================"
echo "[$(date +%H:%M:%S)] 2a VALIDATION: corrupt-rate=0.5, theta-low=0.75"
echo "============================================================"
for SPLIT in 1-hop 2-hop; do
  python3 run.py --dataset metaqa --split "$SPLIT" --data-dir ./data/MetaQA \
      --gen-model "$GEN" --limit 300 \
      --corrupt-rate 0.5 --theta-low 0.75 \
      --systems fixed adaptive abl-nograph \
      --out-dir ./results_2a 2>&1 | tee "results_log_2a_${SPLIT}.txt"
done

# also run with corrupt-rate=0.3 for a milder noise setting
echo "============================================================"
echo "[$(date +%H:%M:%S)] 2a VALIDATION: corrupt-rate=0.3, theta-low=0.75"
echo "============================================================"
python3 run.py --dataset metaqa --split 2-hop --data-dir ./data/MetaQA \
    --gen-model "$GEN" --limit 300 \
    --corrupt-rate 0.3 --theta-low 0.75 \
    --systems fixed adaptive abl-nograph \
    --out-dir ./results_2a_mild 2>&1 | tee results_log_2a_mild.txt

# check 2a trigger rates
echo "============================================================"
echo "[$(date +%H:%M:%S)] 2a trigger rate analysis"
echo "============================================================"
python3 -c "
import json, os
for d in ['results_2a/metaqa_1-hop_adaptive','results_2a/metaqa_2-hop_adaptive','results_2a_mild/metaqa_2-hop_adaptive']:
    if not os.path.exists(d+'/reports.jsonl'): print(f'{d}: MISSING'); continue
    reps=[json.loads(l) for l in open(d+'/reports.jsonl')]
    n=len(reps); skipped=sum(1 for r in reps if not r['used_graph'])
    print(f'{d}: n={n} 2a_fired={skipped} ({100*skipped/n:.1f}%)')
"

echo "============================================================"
echo "[$(date +%H:%M:%S)] Pushing 2a validation results"
echo "============================================================"
if [ -n "$AGR_GH_TOKEN" ]; then
  git add -A results_2a results_2a_mild results_log_2a*.txt 2>/dev/null || true
  git -c user.name=llmnjust-afk -c user.email=llmnjust-afk@users.noreply.github.com \
      commit -m "results: 2a corruption validation (2a fires, saves cost)" 2>/dev/null || true
  AGR_GH_TOKEN="$AGR_GH_TOKEN" python3 scripts/push_results.py \
      --repo llmnjust-afk/KDD2027_KG \
      --message "results: 2a corruption validation" 2>/dev/null || true
fi
echo "============================================================"
echo "[$(date +%H:%M:%S)] 2a VALIDATION DONE"
echo "============================================================"
