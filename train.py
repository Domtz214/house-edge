#!/usr/bin/env python3
"""Local training script (needs numpy + scikit-learn). NOT part of the uploaded ZIP.

Run from the kit folder (the one containing public/ and my-entry/):
    python train.py --data public --entry my-entry

What it does
  1. Builds many training examples per machine: for a tick t, inputs are ticks < t and the
     payout at t is the label. Some examples are masked to look like "short history"
     queries (40 recent consecutive ticks + 24 sampled older ticks).
  2. Fits multinomial logistic regression (L2) on the engineered features, for several C.
  3. Scores each C on the real validation queries (squared error vs true probabilities if a
     truth file is found, else Brier vs observed outcomes), prints per-condition results.
  4. Refits the best C on train + validation, writes <entry>/model.json.
  5. Writes validation predictions (from the train-only model) to <report-dir>.
"""
import argparse
import csv
import json
import os
import random
import sys
from bisect import bisect_left

import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.join(os.getcwd(), "my-entry"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features as F  # noqa: E402

PCOLS = ["p_0", "p_1", "p_2", "p_5", "p_20"]


# ------------------------------------------------------------------ examples
def mask_rows(rows, cutoff, rng):
    recent = [r for r in rows if cutoff - 40 <= r[0] < cutoff]
    older = [r for r in rows if r[0] < cutoff - 40]
    older = sorted(rng.sample(older, min(24, len(older))))
    return older + recent


def build_examples(queries, hist, by_ep, labels, args, rng):
    meta = {}
    for q in queries:
        meta.setdefault((q["episode_id"], q["machine_id"]), (q["machine_type"], q["signal"]))
    ticks = {k: [t for t, _ in v] for k, v in hist.items()}
    X, y, T = [], [], []
    for (e, m), rows in hist.items():
        if (e, m) not in meta or meta[(e, m)][0] not in F.TYPES:
            continue
        mtype, sig = meta[(e, m)]
        cand = list(range(args.min_history, len(rows)))
        if len(cand) > args.max_per_machine:
            cand = rng.sample(cand, args.max_per_machine)
        for i in sorted(cand):
            t, yy = rows[i]
            own = rows[:bisect_left(ticks[(e, m)], t)]
            others = [hist[(e, mm)][:bisect_left(ticks[(e, mm)], t)]
                      for mm in by_ep[e] if mm != m]
            if rng.random() < args.short_frac:
                own = mask_rows(own, t, rng)
                others = [mask_rows(o, t, rng) for o in others]
            X.append(F.base_features(sig, own, others, t))
            y.append(yy)
            T.append(F.TYPES.index(mtype))
    for q in queries:  # the real labeled queries too
        if q["query_id"] in labels and q["machine_type"] in F.TYPES:
            X.append(F.machine_query_features(q, hist, by_ep))
            y.append(labels[q["query_id"]])
            T.append(F.TYPES.index(q["machine_type"]))
    return np.array(X), np.array(y), np.array(T)


# ------------------------------------------------------------------ model
def expand_np(Z, T):
    oh = np.eye(3)[T]
    return np.hstack([Z] + [Z * oh[:, [i]] for i in range(3)] + [oh])


def fit(Xb, y, T, C):
    mean = Xb.mean(0)
    std = Xb.std(0)
    std[std < 1e-9] = 1.0
    clf = LogisticRegression(C=C, max_iter=3000)
    clf.fit(expand_np((Xb - mean) / std, T), y)
    assert len(clf.classes_) >= 3, "too few payout classes in training data"
    D = clf.coef_.shape[1]
    coef, icpt = np.zeros((5, D)), np.full(5, -20.0)
    for ci, c in enumerate(clf.classes_):
        coef[c], icpt[c] = clf.coef_[ci], clf.intercept_[ci]
    return {"version": 2, "mean": mean.tolist(), "std": std.tolist(),
            "coef": coef.tolist(), "intercept": icpt.tolist()}


# ------------------------------------------------------------------ evaluation
def find_truth(folder, args):
    """Look for a file of true probabilities in the validation folder."""
    skip = {"queries.csv", "history.csv", "labels.csv", "sample_submission.csv"}
    if args.truth_file:
        files = [args.truth_file]
    else:
        files = sorted((f for f in os.listdir(folder) if f.endswith(".csv") and f not in skip),
                       key=lambda f: (not any(k in f.lower() for k in ("target", "true")), f))
    for fn in files:
        path = fn if os.path.isabs(fn) else os.path.join(folder, fn)
        with open(path, newline="", encoding="utf-8") as f:
            header = next(csv.reader(f), [])
        if "query_id" not in header:
            continue
        prefixes = {c.rsplit("_", 1)[0] for c in header if "_" in c}
        for pre in ([args.truth_prefix] if args.truth_prefix else sorted(prefixes)):
            cols = [f"{pre}_{s}" for s in ("0", "1", "2", "5", "20")]
            if all(c in header for c in cols):
                rows = F.read_csv(path)
                print(f"truth file: {path} (columns {cols})")
                return {r["query_id"]: [float(r[c]) for c in cols] for r in rows}
    print("no true-probability file found; scoring with Brier vs observed outcomes")
    return None


def score(preds, queries, labels, truth, cond):
    per = {}
    for q in queries:
        qid = q["query_id"]
        p = preds[qid]
        if truth is not None and qid in truth:
            err = sum((p[i] - truth[qid][i]) ** 2 for i in range(5))
        elif qid in labels:
            err = sum((p[i] - (1.0 if i == labels[qid] else 0.0)) ** 2 for i in range(5))
        else:
            continue
        per.setdefault(cond.get(qid, "all"), {}).setdefault(q["episode_id"], []).append(err)
    out = {}
    for c, eps in per.items():  # mean within episode, then across episodes (as in the guide)
        out[c] = float(np.mean([np.mean(v) for v in eps.values()]))
    return out


def write_preds(path, preds):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["query_id"] + PCOLS)
        for qid, p in preds.items():
            w.writerow([qid] + [repr(x) for x in p])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="public")
    ap.add_argument("--entry", default="my-entry")
    ap.add_argument("--report-dir", default="v2-report")
    ap.add_argument("--max-per-machine", type=int, default=60)
    ap.add_argument("--min-history", type=int, default=40)
    ap.add_argument("--short-frac", type=float, default=0.3)
    ap.add_argument("--c-grid", default="0.003,0.01,0.03,0.1,0.3")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--truth-file", default=None)
    ap.add_argument("--truth-prefix", default=None)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    tr_q, tr_h, tr_e, tr_y = F.load_split(os.path.join(args.data, "train"), True)
    vdir = os.path.join(args.data, "validation")
    va_q, va_h, va_e, va_y = F.load_split(vdir, True)
    truth = find_truth(vdir, args)

    cond = {}
    for q in va_q:
        if q.get("condition"):
            cond[q["query_id"]] = q["condition"]
    lp = os.path.join(vdir, "labels.csv")
    if os.path.exists(lp):
        for r in F.read_csv(lp):
            if r.get("condition"):
                cond[r["query_id"]] = r["condition"]

    print("building train examples ...")
    Xtr, ytr, Ttr = build_examples(tr_q, tr_h, tr_e, tr_y, args, rng)
    print(f"  {len(ytr)} train examples, {Xtr.shape[1]} base features")
    va_feats = {q["query_id"]: F.machine_query_features(q, va_h, va_e) for q in va_q}

    prior = np.bincount(ytr, minlength=5) / len(ytr)
    base_preds = {q["query_id"]: prior.tolist() for q in va_q}
    base = score(base_preds, va_q, va_y, truth, cond)
    print("prior-only:", {k: round(v, 5) for k, v in base.items()})

    best = (None, float("inf"), None, None)
    for C in [float(c) for c in args.c_grid.split(",")]:
        model = fit(Xtr, ytr, Ttr, C)
        preds = {q["query_id"]: F.predict_probs(model, va_feats[q["query_id"]], q["machine_type"])
                 for q in va_q}
        sc = score(preds, va_q, va_y, truth, cond)
        ratios = [sc[k] / max(base[k], 1e-9) for k in sc]
        R = float(np.mean(ratios))
        print(f"C={C:<6} err={ {k: round(v, 5) for k, v in sc.items()} }  "
              f"R vs prior-only={R:.4f}  approx pts={80 / (1 + R):.2f}")
        if R < best[1]:
            best = (C, R, model, preds)
    C, R, _, preds = best
    print(f"best C={C} (R={R:.4f})")
    write_preds(os.path.join(args.report_dir, "validation_predictions.csv"), preds)

    print("refitting on train + validation ...")
    Xva, yva, Tva = build_examples(va_q, va_h, va_e, va_y, args, rng)
    final = fit(np.vstack([Xtr, Xva]), np.concatenate([ytr, yva]), np.concatenate([Ttr, Tva]), C)
    os.makedirs(args.entry, exist_ok=True)
    with open(os.path.join(args.entry, "model.json"), "w", encoding="utf-8") as f:
        json.dump(final, f)
    print(f"saved {os.path.join(args.entry, 'model.json')}")


if __name__ == "__main__":
    main()
