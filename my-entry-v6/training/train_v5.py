"""Train the v5 entry (needs numpy, pandas, lightgbm). Not part of the uploaded ZIP.

    python research/train_v5.py --data public --entry my-entry-v5

1. Pseudo-queries: every observed tick t >= 24 of every machine becomes a training example,
   with features from ticks < t only and the payout at t as the label.
2. Win model: LightGBM binary classifier on the pattern features.
   Size model: LightGBM 4-class classifier (1x/2x/5x/20x) on wins only.
3. Honest check: models trained on train-split histories only, scored on validation targets.
4. Final: refit on train + validation histories, export JSON for the stdlib predictor.
"""
import argparse
import json
import os
import sys
import time

import lightgbm as lgb
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))

WIN_PARAMS = dict(objective="binary", n_estimators=400, learning_rate=0.03, num_leaves=31,
                  min_child_samples=100, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                  random_state=0, deterministic=True, force_row_wise=True, verbose=-1, n_jobs=2)
SIZE_PARAMS = dict(objective="multiclass", n_estimators=300, learning_rate=0.03, num_leaves=15,
                   min_child_samples=200, random_state=0, deterministic=True, force_row_wise=True,
                   verbose=-1, n_jobs=2)


def pseudo(F, folder, min_t=24):
    queries, seqs = F.load_split(folder)
    types = {}
    for q in queries:
        types[q["machine_id"]] = (q["machine_type"], q["cutoff"])
    Xw, yw, Xs, ys = [], [], [], []
    for m, (mt, cut) in types.items():
        s = seqs[m]
        cuts = [t for t in sorted(s) if min_t <= t < cut]
        if not cuts:
            continue
        feats = F.machine_features(s, cuts, mt)
        for t in cuts:
            f, g = feats[t]
            Xw.append(f)
            yw.append(int(s[t] > 0))
            if s[t] > 0:
                Xs.append(g)
                ys.append(s[t] - 1)
    return np.array(Xw), np.array(yw), np.array(Xs), np.array(ys)


def query_feats(F, folder):
    queries, seqs = F.load_split(folder)
    out = {}
    for q in queries:
        out[q["query_id"]] = F.machine_features(seqs[q["machine_id"]], [q["cutoff"]], q["machine_type"])[q["cutoff"]]
    return queries, out


def fit(Xw, yw, Xs, ys):
    win = lgb.LGBMClassifier(**WIN_PARAMS).fit(Xw, yw)
    size = lgb.LGBMClassifier(**SIZE_PARAMS).fit(Xs, ys)
    return win, size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="public")
    ap.add_argument("--entry", default="my-entry-v5")
    ap.add_argument("--report", default="v5-report")
    args = ap.parse_args()
    sys.path.insert(0, os.path.abspath(args.entry))
    import features as F
    t0 = time.time()
    tr = pseudo(F, os.path.join(args.data, "train"))
    print(f"train pseudo-queries: win {tr[0].shape} size {tr[2].shape}  ({time.time()-t0:.0f}s)")
    win, size = fit(*tr)

    vq, vf = query_feats(F, os.path.join(args.data, "validation"))
    os.makedirs(args.report, exist_ok=True)
    path = os.path.join(args.report, "validation_predictions.csv")
    with open(path, "w", encoding="utf-8") as f:
        f.write("query_id,p_0,p_1,p_2,p_5,p_20\n")
        for q in vq:
            fw, g = vf[q["query_id"]]
            pw = win.predict_proba(np.array([fw]))[0, 1]
            sz = size.predict_proba(np.array([g]))[0]
            p = [1 - pw] + [pw * x for x in sz]
            s = sum(p)
            f.write(q["query_id"] + "," + ",".join(repr(float(x / s)) for x in p) + "\n")
    print(f"wrote {path} (train-only model; score with check_submission.py --split validation)")

    va = pseudo(F, os.path.join(args.data, "validation"))
    allw = (np.vstack([tr[0], va[0]]), np.concatenate([tr[1], va[1]]),
            np.vstack([tr[2], va[2]]), np.concatenate([tr[3], va[3]]))
    win, size = fit(*allw)
    for name, m in (("win_model.json", win), ("size_model.json", size)):
        with open(os.path.join(args.entry, name), "w", encoding="utf-8") as f:
            json.dump(m.booster_.dump_model(), f)
    # self-check: stdlib tree inference == lightgbm
    worst = 0.0
    for q in vq[:200]:
        fw, g = vf[q["query_id"]]
        worst = max(worst, abs(F.win_prob(json.loads(json.dumps(win.booster_.dump_model())), fw)
                               - win.predict_proba(np.array([fw]))[0, 1]))
    print(f"saved models to {args.entry}; max |stdlib - lightgbm| on 200 queries = {worst:.2e}")
    print(f"total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
