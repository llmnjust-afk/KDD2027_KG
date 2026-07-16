#!/usr/bin/env python3
"""Push experiment results to GitHub.

Run at the end of an experiment sweep to commit the results/ directory (which is
.gitignored) and push to the configured remote. The PAT is read from the
AGR_GH_TOKEN env var so it is never written to git config or any file.

Usage:
    AGR_GH_TOKEN=ghp_xxx python scripts/push_results.py \\
        --repo llmnjust-afk/KDD2027_KG --message "results: MetaQA 1/2/3-hop + ablations"

It force-adds the results/ directory (overriding .gitignore) so the Pareto
tables and per-system reports are tracked. If there is nothing new to commit it
exits cleanly with code 0.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys


def sh(cmd, check=True, capture=True):
    print(f"$ {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, check=False, capture_output=capture, text=True)
    if check and r.returncode != 0:
        sys.stderr.write(r.stdout + r.stderr)
        raise SystemExit(f"command failed (exit {r.returncode}): {' '.join(cmd)}")
    if capture:
        if r.stdout:
            print(r.stdout.rstrip())
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="llmnjust-afk/KDD2027_KG")
    ap.add_argument("--remote-name", default="origin")
    ap.add_argument("--branch", default="main")
    ap.add_argument("--message", default="chore: update experiment results")
    ap.add_argument("--results-dir", default="results")
    args = ap.parse_args()

    token = os.environ.get("AGR_GH_TOKEN")
    if not token:
        print("AGR_GH_TOKEN not set; skipping push.", flush=True)
        return 0

    sh(["git", "config", "user.name", "llmnjust-afk"])
    sh(["git", "config", "user.email", "llmnjust-afk@users.noreply.github.com"])

    # force-add results (ignored by .gitignore) + any tracked changes
    sh(["git", "add", "-A", args.results_dir])
    sh(["git", "add", "-A"])

    # nothing to commit?
    r = sh(["git", "status", "--porcelain"])
    if not r.stdout.strip():
        print("nothing to commit; working tree clean", flush=True)
        return 0

    sh(["git", "commit", "-m", args.message])

    # push using the token in the URL (not stored anywhere persistent)
    url = f"https://{token}@github.com/{args.repo}.git"
    # redact token in any echoed output
    r = sh(["git", "push", url, args.branch], check=False)
    out = (r.stdout + r.stderr).replace(token, "***")
    print(out, flush=True)
    if r.returncode != 0:
        raise SystemExit(f"push failed (exit {r.returncode})")
    print(f"\nPushed results to https://github.com/{args.repo} (branch {args.branch})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
