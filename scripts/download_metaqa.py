#!/usr/bin/env python3
"""Download MetaQA (WikiMovies KG + vanilla questions) into ./data/MetaQA.

Uses the self-contained HuggingFace mirror `camazlucas/MetaQA` (which ships the
WikiMovies KB + 1/2/3-hop vanilla splits) so no Google Drive access is needed.

Produces the layout expected by agr.kg.load_metaqa:
    data/MetaQA/kb.txt
    data/MetaQA/1-hop/questions.txt
    data/MetaQA/2-hop/questions.txt
    data/MetaQA/3-hop/questions.txt
"""
from __future__ import annotations

import argparse
import os

from huggingface_hub import hf_hub_download

REPO = "camazlucas/MetaQA"
FILES = {
    "kb.txt": "kb/kb.txt",
    "1-hop/questions.txt": "1-hop/vanilla/qa_test.txt",
    "2-hop/questions.txt": "2-hop/vanilla/qa_test.txt",
    "3-hop/questions.txt": "3-hop/vanilla/qa_test.txt",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="./data/MetaQA")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for dst_rel, src in FILES.items():
        dst = os.path.join(args.out, dst_rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            print(f"exists: {dst} ({os.path.getsize(dst)} bytes)")
            continue
        print(f"downloading {REPO}:{src} -> {dst}")
        tmp = hf_hub_download(repo_id=REPO, filename=src, repo_type="dataset")
        with open(tmp, "r", encoding="utf-8") as f, open(dst, "w", encoding="utf-8") as g:
            g.write(f.read())
        print(f"  wrote {os.path.getsize(dst)} bytes")
    print("done")


if __name__ == "__main__":
    main()
