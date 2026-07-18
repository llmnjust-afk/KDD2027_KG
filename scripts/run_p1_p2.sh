#!/usr/bin/env bash
# Point 1+2: WebQSP 500q full ablation + MLP trained baseline
set -e
cd "$(dirname "$0")/.."
export HF_HUB_DISABLE_PROGRESS_BARS=1
G7="Qwen/Qwen2.5-7B-Instruct"

echo "============================================================"
echo "[$(date +%H:%M:%S)] P1: WebQSP 500q full (Fixed + Adaptive)"
echo "============================================================"
python3 scripts/webqsp_pilot.py --gen-model "$G7" --limit 500 --out-dir ./results_webqsp_500 2>&1 | tee results_log_webqsp_500.txt

echo "============================================================"
echo "[$(date +%H:%M:%S)] P2: MLP trained baseline @ 7B/2-hop (200q)"
echo "============================================================"
python3 -c "
import sys, os, time, json, re, random
import numpy as np
sys.path.insert(0, '.')
from agr import (load_metaqa, build_backend, GraphRAGRetriever, generate_answer,
                 extract_answer_entities, score_query, aggregate, pareto_table,
                 save_reports, QueryReport, FixedController, AdaptiveController)
from dataclasses import dataclass

@dataclass
class TrainedKController:
    classifier: object; scaler: object; embedder: object
    beam: int = 4; max_hops: int = 3
    _predicted_k: int = 3
    def _predict_k(self, q):
        emb = np.asarray(self.embedder.embed([q])[0], dtype=np.float32).reshape(1,-1)
        emb_s = self.scaler.transform(emb)
        self._predicted_k = int(self.classifier.predict(emb_s)[0])
        return self._predicted_k
    def decide_use_graph(self, q, eh, ctx):
        self._predict_k(q); return len(eh) > 0
    def beam_for_hop(self, *a): return self.beam
    def should_stop(self, q, hi, *a): return hi + 1 >= self._predicted_k

def train_mlp(data_dir, backend):
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    questions, labels = [], []
    for hop_idx, split in enumerate(['1-hop','2-hop','3-hop'], 1):
        p = os.path.join(data_dir, split, 'train.txt')
        for line in open(p, encoding='utf-8'):
            line = line.strip()
            if not line: continue
            q = line.rsplit('\t',1)[0] if '\t' in line else line
            q = re.sub(r'\[([^\]]+)\]', r'\1', q)
            questions.append(q); labels.append(hop_idx)
    rng = np.random.default_rng(42)
    sub_q, sub_l = [], []
    for hop in [1,2,3]:
        idx = [i for i,l in enumerate(labels) if l==hop]
        rng.shuffle(idx)
        for i in idx[:min(10000,len(idx))]:
            sub_q.append(questions[i]); sub_l.append(labels[i])
    print(f'Embedding {len(sub_q)} train questions...', flush=True)
    X = np.asarray(backend.embed(sub_q), dtype=np.float32)
    y = np.array(sub_l)
    scaler = StandardScaler(); X_s = scaler.fit_transform(X)
    clf = MLPClassifier(hidden_layer_sizes=(256,128), max_iter=500, early_stopping=True, random_state=42)
    clf.fit(X_s, y)
    print(f'MLP train accuracy: {clf.score(X_s,y):.3f}', flush=True)
    return clf, scaler

def run_system(name, kg, backend, ctrl, examples, limit):
    ret = GraphRAGRetriever(kg, backend, ctrl, link_topk=5, judge_with_llm=True)
    reports = []
    for i, ex in enumerate(examples[:limit]):
        t0=time.time(); c0=backend.usage.n_calls; tk0=backend.usage.n_input_tokens
        r=ret.retrieve(ex.question); a=generate_answer(backend,ex.question,r)
        p=extract_answer_entities(a.text); sc=score_query(p,ex.answers)
        reports.append(QueryReport(qid=ex.qid,question=ex.question,gold=ex.answers,pred=p,
            hit1=sc['hit1'],f1=sc['f1'],exact=sc['exact'],used_graph=r.used_graph,
            n_hops=r.n_hops_executed,n_llm_calls=backend.usage.n_calls-c0,
            n_input_tokens=backend.usage.n_input_tokens-tk0,n_output_tokens=a.n_output_tokens))
        if (i+1)%50==0 or i==limit-1:
            agg=aggregate(reports); print(f'  [{name} {i+1}/{limit}] F1={agg.mean_f1:.3f} toks={agg.mean_n_input_tokens:.0f} t={time.time()-t0:.1f}s',flush=True)
    return reports

data_dir='./data/MetaQA'; limit=200; gen='Qwen/Qwen2.5-7B-Instruct'
kg, examples = load_metaqa(data_dir, '2-hop')
print(f'KG: {len(kg)} triples | {len(examples)} test Q (using {limit})', flush=True)
backend = build_backend({'kind':'hf','gen_model':gen,'embed_model':'sentence-transformers/all-MiniLM-L6-v2','device':'cuda','dtype':'bfloat16'})
print(f'Backend: {gen}', flush=True)

print('\\n=== Training MLP classifier ===', flush=True)
clf, scaler = train_mlp(data_dir, backend)

aggs={}
for name, ctrl in [('Fixed', FixedController(max_hops=3, beam=4)),
                    ('Adaptive', AdaptiveController(max_hops=3, base_beam=4)),
                    ('MLP-Trained-K', TrainedKController(classifier=clf, scaler=scaler, embedder=backend, beam=4, max_hops=3))]:
    print(f'\\n=== {name} ===', flush=True)
    r = run_system(name, kg, backend, ctrl, examples, limit)
    aggs[name] = aggregate(r)
    save_reports(r, f'./results_mlp/metaqa_2-hop_{name.lower().replace(\" \",\"_\")}/reports.jsonl')

print('\\n'+'='*70, flush=True)
print('MLP BASELINE COMPARISON [2-hop, n=200]', flush=True)
print('='*70, flush=True)
print(pareto_table(aggs), flush=True)

# also run on mixed stream
print('\\n=== MLP on mixed stream ===', flush=True)
per=100; rng=random.Random(0)
_,ex1=load_metaqa(data_dir,'1-hop'); _,ex2=load_metaqa(data_dir,'2-hop'); _,ex3=load_metaqa(data_dir,'3-hop')
rng.shuffle(ex1);rng.shuffle(ex2);rng.shuffle(ex3)
mixed=ex1[:per]+ex2[:per]+ex3[:per]; rng.shuffle(mixed)
aggs_m={}
for name, ctrl in [('Fixed', FixedController(max_hops=3, beam=4)),
                    ('Adaptive', AdaptiveController(max_hops=3, base_beam=4)),
                    ('MLP-Trained-K', TrainedKController(classifier=clf, scaler=scaler, embedder=backend, beam=4, max_hops=3))]:
    print(f'\\n=== {name} (mixed) ===', flush=True)
    r = run_system(name, kg, backend, ctrl, mixed, 300)
    aggs_m[name] = aggregate(r)
    save_reports(r, f'./results_mlp/metaqa_mixed_{name.lower().replace(\" \",\"_\")}/reports.jsonl')

print('\\n'+'='*70, flush=True)
print('MLP MIXED COMPARISON', flush=True)
print('='*70, flush=True)
print(pareto_table(aggs_m), flush=True)
" 2>&1 | tee results_log_mlp.txt

echo "============================================================"
echo "[$(date +%H:%M:%S)] P1+P2 DONE"
echo "============================================================"
