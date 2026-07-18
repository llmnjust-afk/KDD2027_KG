#!/usr/bin/env bash
# P3: Few-shot judge prompt improvement (7B/2-hop 200q)
# Waits for P1+P2 to finish, then runs.
set -e
cd "$(dirname "$0")/.."
export HF_HUB_DISABLE_PROGRESS_BARS=1
G7="Qwen/Qwen2.5-7B-Instruct"

echo "Waiting for P1+P2 to finish..."
while pgrep -f "run_p1_p2" >/dev/null 2>&1; do sleep 30; done
echo "P1+P2 done. Starting P3 (few-shot judge)."

echo "============================================================"
echo "[$(date +%H:%M:%S)] P3: Few-shot judge @ 7B/2-hop (200q)"
echo "============================================================"
python3 run.py --dataset metaqa --split 2-hop --data-dir ./data/MetaQA \
    --gen-model "$G7" --limit 200 --systems fixed adaptive \
    --out-dir ./results_fewshot 2>&1 | tee results_log_fewshot.txt

echo "============================================================"
echo "[$(date +%H:%M:%S)] P3: Few-shot judge @ 7B/1-hop (200q)"
echo "============================================================"
python3 run.py --dataset metaqa --split 1-hop --data-dir ./data/MetaQA \
    --gen-model "$G7" --limit 200 --systems fixed adaptive \
    --out-dir ./results_fewshot 2>&1 | tee results_log_fewshot_1hop.txt

echo "=== P3 DONE ==="
