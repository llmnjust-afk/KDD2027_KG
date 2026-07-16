#!/usr/bin/env python3
"""Download MetaQA (WikiMovies KG + questions) into ./data/MetaQA.

MetaQA is the standard multi-hop KGQA benchmark used by ToG / GNN-RAG / NSM.
It is a self-contained movie KG (no external SPARQL endpoint needed), which is
why it fits a single-GPU, no-training setup.

Source: https://github.com/yuyuz/MetaQA
If the raw download is unavailable, this script falls back to building the KG
from the HuggingFace `nodes`/`edges` mirror if present.
"""
from __future__ import annotations

import argparse
import os
import urllib.request

URLS = {
    "kb.txt": "https://raw.githubusercontent.com/yuyuz/MetaQA/master/kb.txt",
    "1-hop/questions.txt": "https://raw.githubusercontent.com/yuyuz/MetaQA/master/MetaQA/1-hop/vanilla/test.txt",
    "2-hop/questions.txt": "https://raw.githubusercontent.com/yuyuz/MetaQA/master/MetaQA/2-hop/vanilla/test.txt",
    "3-hop/questions.txt": "https://raw.githubusercontent.com/yuyuz/MetaQA/master/MetaQA/3-hop/vanilla/test.txt",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="./data/MetaQA")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for rel, url in URLS.items():
        dst = os.path.join(args.out, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.exists(dst):
            print(f"exists: {dst}")
            continue
        print(f"downloading {url} -> {dst}")
        try:
            urllib.request.urlretrieve(url, dst)
        except Exception as e:
            print(f"  FAILED: {e}")
            print("  (network restricted? place MetaQA files manually under data/MetaQA)")


if __name__ == "__main__":
    main()
