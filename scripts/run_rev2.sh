#!/bin/bash
# Revision-2 master runner: B(mixed Pareto) -> C(counterfactual) -> D(conformal) -> E(strong engine)
# Each stage logs to a file and continues even if one fails.
set +e
cd /data/lab/adaptive_graphrag
export HF_HUB_DISABLE_PROGRESS_BARS=1
G7="Qwen/Qwen2.5-7B-Instruct"
LOG=run_rev2.log
echo "=== REVISION-2 MASTER RUN started $(date) ===" > $LOG

# ---- B: mixed-stream Pareto sweep (the critical experiment) ----
echo ">>> [B] mixed Pareto sweep started $(date)" >> $LOG
python3 scripts/mixed_pareto.py --gen-model "$G7" --per 80 --seed 0 \
    --skip-routers --out-dir ./results_mixed_pareto >> $LOG 2>&1
echo ">>> [B] done $(date)" >> $LOG

# ---- C: counterfactual force-continue (stop-safety validation) ----
echo ">>> [C] counterfactual started $(date)" >> $LOG
python3 scripts/counterfactual.py --gen-model "$G7" --split 2-hop --limit 200 \
    --out-dir ./results_counterfactual >> $LOG 2>&1
echo ">>> [C] done $(date)" >> $LOG

# ---- D: conformal-calibrated threshold (method increment) ----
echo ">>> [D] conformal started $(date)" >> $LOG
python3 scripts/conformal.py --gen-model "$G7" --split 2-hop --cal 120 --test 200 \
    --alpha 0.10 --out-dir ./results_conformal >> $LOG 2>&1
echo ">>> [D] done $(date)" >> $LOG

# ---- E: stronger engine (Qwen3-8B) pluggability check ----
echo ">>> [E] strong engine started $(date)" >> $LOG
G3="Qwen/Qwen3-8B-Instruct"
python3 -c "from huggingface_hub import model_info; model_info('$G3')" 2>/dev/null
if [ $? -eq 0 ]; then
    python3 run.py --dataset metaqa --split 2-hop --data-dir ./data/MetaQA \
        --gen-model "$G3" --limit 150 --systems fixed adaptive \
        --out-dir ./results_qwen3 >> $LOG 2>&1
    echo ">>> [E] Qwen3-8B done $(date)" >> $LOG
else
    echo ">>> [E] Qwen3-8B not available, trying Qwen2.5-14B-Instruct" >> $LOG
    G14="Qwen/Qwen2.5-14B-Instruct"
    python3 run.py --dataset metaqa --split 2-hop --data-dir ./data/MetaQA \
        --gen-model "$G14" --limit 150 --systems fixed adaptive \
        --out-dir ./results_qwen3 >> $LOG 2>&1
    echo ">>> [E] Qwen2.5-14B done $(date)" >> $LOG
fi

echo "=== REVISION-2 MASTER RUN COMPLETE $(date) ===" >> $LOG
