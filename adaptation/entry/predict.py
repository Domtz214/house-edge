#!/usr/bin/env python3
"""House Edge predictor v6 (Python standard library only).

Usage: python predict.py --data /data --out /output/submission.csv
Loads win_model.json + size_model.json (LightGBM trees exported by research/train_v6.py).
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import features as F  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test")
    args = ap.parse_args()
    with open(os.path.join(HERE, "win_model.json"), encoding="utf-8") as f:
        win = json.load(f)
    with open(os.path.join(HERE, "size_model.json"), encoding="utf-8") as f:
        size = json.load(f)

    queries, seqs = F.load_split(os.path.join(args.data, args.split))
    episodes = {}
    for q in queries:
        episodes.setdefault(q["episode_id"], []).append(q)
    rows = {}
    for qs in episodes.values():
        machines = {q["machine_id"]: (seqs[q["machine_id"]], q["machine_type"]) for q in qs}
        feats = F.episode_features(machines, {q["machine_id"]: [q["cutoff"]] for q in qs})
        for q in qs:
            fw, g = feats[(q["machine_id"], q["cutoff"])]
            pw = F.win_prob(win, fw)
            sz = F.size_probs(size, g)
            p = [1.0 - pw] + [pw * x for x in sz]
            s = sum(p)
            rows[q["query_id"]] = [x / s for x in p]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        f.write("query_id,p_0,p_1,p_2,p_5,p_20\n")
        for q in queries:
            f.write(q["query_id"] + "," + ",".join(repr(x) for x in rows[q["query_id"]]) + "\n")
    print(f"wrote {len(queries)} rows to {args.out}")


if __name__ == "__main__":
    main()
