#!/usr/bin/env python3
"""House Edge predictor v2 (standard library only).

Loads model.json (weights exported by train.py) and writes the submission CSV.
Usage: python predict.py --data /data --out /output/submission.csv
"""
import argparse
import csv
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
    args = ap.parse_args()

    with open(os.path.join(HERE, "model.json"), encoding="utf-8") as f:
        model = json.load(f)

    queries, hist, by_ep, _ = F.load_split(os.path.join(args.data, "test"), False)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["query_id", "p_0", "p_1", "p_2", "p_5", "p_20"])
        for q in queries:
            feats = F.machine_query_features(q, hist, by_ep)
            p = F.predict_probs(model, feats, q["machine_type"])
            s = sum(p)
            w.writerow([q["query_id"]] + [repr(x / s) for x in p])
    print(f"wrote {len(queries)} rows to {args.out}")


if __name__ == "__main__":
    main()
