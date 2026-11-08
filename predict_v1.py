#!/usr/bin/env python3
"""House Edge forecaster (standard library only, so it runs on the organizers' runner).

Usage:  python predict.py --data /path/to/public --out /path/to/submission.csv

Method
------
For each query (episode, machine, cutoff) we only look at history with tick < cutoff.
  * prior      : smoothed payout frequencies by (machine_type, signal), learned from labels
  * own        : time-decayed payout counts of the machine itself (decay by actual tick
                 distance, so missing ticks are treated as unknown, never as zero)
  * episode    : time-decayed counts of OTHER machines in the same episode (earlier ticks only)
Blend:  ep_est = (k2*prior + ep_counts) / (k2 + ep_weight)
        p      = (k*ep_est + own_counts) / (k + own_weight)
Hyperparameters (half-life, k, k2) are tuned by Brier score against observed outcomes on the
validation split (equivalent to squared error vs. true odds up to a constant), then priors
are refit on train+validation and test predictions are written. Fully deterministic.
"""
import argparse
import csv
import itertools
import math
import os
from collections import defaultdict

PAYOUTS = [0, 1, 2, 5, 20]
IDX = {p: i for i, p in enumerate(PAYOUTS)}


# ----------------------------------------------------------------- IO
def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_split(folder, with_labels):
    queries = read_csv(os.path.join(folder, "queries.csv"))
    for q in queries:
        q["cutoff"] = int(float(q["cutoff"]))
        q["signal"] = int(float(q["signal"]))
    hist = defaultdict(list)  # (episode, machine) -> [(tick, payout_idx)]
    for r in read_csv(os.path.join(folder, "history.csv")):
        hist[(r["episode_id"], r["machine_id"])].append((int(float(r["tick"])), IDX[int(float(r["payout"]))]))
    for v in hist.values():
        v.sort()
    by_episode = defaultdict(list)  # episode -> machine ids
    for (e, m) in hist:
        by_episode[e].append(m)
    labels = {}
    if with_labels:
        lp = os.path.join(folder, "labels.csv")
        if os.path.exists(lp):
            labels = {r["query_id"]: IDX[int(float(r["outcome"]))] for r in read_csv(lp)}
    return queries, hist, by_episode, labels


# ----------------------------------------------------------------- features
def decayed(rows, cutoff, half_life):
    """Decayed payout counts using ticks strictly below cutoff."""
    counts = [0.0] * 5
    total = 0.0
    for tick, y in rows:
        if tick >= cutoff:
            break
        w = 1.0 if half_life is None else 0.5 ** ((cutoff - tick) / half_life)
        counts[y] += w
        total += w
    return counts, total


def query_features(q, hist, by_episode, half_lives):
    e, m, c = q["episode_id"], q["machine_id"], q["cutoff"]
    out = {}
    others = [mm for mm in by_episode.get(e, []) if mm != m]
    for h in half_lives:
        own = decayed(hist.get((e, m), []), c, h)
        ep_counts, ep_w = [0.0] * 5, 0.0
        for mm in others:  # other machines, earlier ticks only (tick < cutoff)
            cn, w = decayed(hist[(e, mm)], c, h)
            for i in range(5):
                ep_counts[i] += cn[i]
            ep_w += w
        out[h] = (own, (ep_counts, ep_w))
    return out


# ----------------------------------------------------------------- model
def fit_prior(queries, labels, strength=20.0):
    glob = [1.0] * 5
    for q in queries:
        if q["query_id"] in labels:
            glob[labels[q["query_id"]]] += 1
    gs = sum(glob)
    glob = [g / gs for g in glob]
    by_type = defaultdict(lambda: [0.0] * 5)
    by_ts = defaultdict(lambda: [0.0] * 5)
    for q in queries:
        y = labels.get(q["query_id"])
        if y is None:
            continue
        by_type[q["machine_type"]][y] += 1
        by_ts[(q["machine_type"], q["signal"])][y] += 1

    def smooth(counts, base):
        n = sum(counts)
        return [(counts[i] + strength * base[i]) / (n + strength) for i in range(5)]

    type_p = {t: smooth(c, glob) for t, c in by_type.items()}
    ts_p = {k: smooth(c, type_p[k[0]]) for k, c in by_ts.items()}

    def prior(q):
        return ts_p.get((q["machine_type"], q["signal"])) or type_p.get(q["machine_type"]) or glob

    return prior


def blend(prior, own, ep, k, k2):
    (on, ow), (en, ew) = own, ep
    ep_est = [(k2 * prior[i] + en[i]) / (k2 + ew) for i in range(5)]
    p = [(k * ep_est[i] + on[i]) / (k + ow) for i in range(5)]
    s = sum(p)
    return [x / s for x in p]


def brier(preds, queries, labels):
    tot, n = 0.0, 0
    for q in queries:
        y = labels.get(q["query_id"])
        if y is None:
            continue
        p = preds[q["query_id"]]
        tot += sum((p[i] - (1.0 if i == y else 0.0)) ** 2 for i in range(5))
        n += 1
    return tot / max(n, 1)


def tune(val_queries, feats, prior, labels, grid):
    best, best_score = None, float("inf")
    for h, k, k2 in grid:
        preds = {}
        for q in val_queries:
            own, ep = feats[q["query_id"]][h]
            preds[q["query_id"]] = blend(prior(q), own, ep, k, k2)
        s = brier(preds, val_queries, labels)
        if s < best_score:
            best, best_score = (h, k, k2), s
    return best, best_score


def write_submission(path, preds):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["query_id", "p_0", "p_1", "p_2", "p_5", "p_20"])
        for qid, p in preds.items():
            s = sum(p)
            p = [max(0.0, x) / s for x in p]
            w.writerow([qid] + [repr(float(x)) for x in p])


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    half_lives = [8, 16, 32, 64, None]
    grid = list(itertools.product(half_lives, [2, 5, 10, 20, 50], [5, 20, 100]))

    tr_q, tr_h, tr_e, tr_y = load_split(os.path.join(args.data, "train"), True)
    va_q, va_h, va_e, va_y = load_split(os.path.join(args.data, "validation"), True)
    te_q, te_h, te_e, _ = load_split(os.path.join(args.data, "test"), False)

    # 1) tune on validation with priors learned from train only
    prior_tr = fit_prior(tr_q, tr_y)
    va_feats = {q["query_id"]: query_features(q, va_h, va_e, half_lives) for q in va_q}
    (h, k, k2), score = tune(va_q, va_feats, prior_tr, va_y, grid)
    base = brier({q["query_id"]: prior_tr(q) for q in va_q}, va_q, va_y)
    print(f"best half_life={h} k={k} k2={k2}  val Brier={score:.5f} (prior-only {base:.5f})")

    # 2) refit prior on train+validation, predict test
    prior_all = fit_prior(tr_q + va_q, {**tr_y, **va_y})
    preds = {}
    for q in te_q:
        own, ep = query_features(q, te_h, te_e, [h])[h]
        preds[q["query_id"]] = blend(prior_all(q), own, ep, k, k2)
    write_submission(args.out, preds)
    print(f"wrote {len(preds)} rows to {args.out}")


if __name__ == "__main__":
    main()
