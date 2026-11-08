"""Shared feature + inference code (standard library only).

Used by BOTH train.py (local, numpy/sklearn) and predict.py (runs on the organizers'
standard-library-only runner), so training and inference see identical features.

Temporal rules enforced here: a query with cutoff c only uses ticks < c, from its own
machine and from OTHER machines of the same episode. Lags/decays use real tick distance,
so a missing tick is "missing", never a zero payout.
"""
import csv
import math
import os
from collections import defaultdict

PAYOUTS = [0, 1, 2, 5, 20]
IDX = {p: i for i, p in enumerate(PAYOUTS)}
TYPES = ["steady", "balanced", "jackpot"]
HALF_LIVES = [4, 8, 16, 32, 64, None]   # None = all history
EP_HALF_LIVES = [8, 32]
MAX_LAG = 8
SMOOTH = 1.0


# ------------------------------------------------------------------ IO
def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_split(folder, with_labels):
    queries = read_csv(os.path.join(folder, "queries.csv"))
    for q in queries:
        q["cutoff"] = int(float(q["cutoff"]))
        q["signal"] = int(float(q["signal"]))
    hist = defaultdict(list)  # (episode, machine) -> sorted [(tick, payout_idx)]
    for r in read_csv(os.path.join(folder, "history.csv")):
        hist[(r["episode_id"], r["machine_id"])].append(
            (int(float(r["tick"])), IDX[int(float(r["payout"]))]))
    for v in hist.values():
        v.sort()
    by_episode = defaultdict(list)
    for (e, m) in hist:
        by_episode[e].append(m)
    labels = {}
    if with_labels:
        lp = os.path.join(folder, "labels.csv")
        if os.path.exists(lp):
            labels = {r["query_id"]: IDX[int(float(r["outcome"]))] for r in read_csv(lp)}
    return queries, hist, by_episode, labels


# ------------------------------------------------------------------ features
def _decayed_multi(rows, cutoff, hls):
    counts = [[0.0] * 5 for _ in hls]
    tot = [0.0] * len(hls)
    for tick, y in rows:
        if tick >= cutoff:
            break
        d = cutoff - tick
        for j, h in enumerate(hls):
            w = 1.0 if h is None else 0.5 ** (d / h)
            counts[j][y] += w
            tot[j] += w
    return counts, tot


def _smooth(counts, tot):
    return [(c + SMOOTH * 0.2) / (tot + SMOOTH) for c in counts]


def base_features(signal, own_rows, other_rows, cutoff):
    """own_rows: [(tick, payout_idx)] sorted; other_rows: list of such lists (same episode)."""
    f = []
    oc, ot = _decayed_multi(own_rows, cutoff, HALF_LIVES)
    own_s = []
    for j in range(len(HALF_LIVES)):
        s = _smooth(oc[j], ot[j])
        own_s.append(s)
        f.extend(s)
    f.extend(math.log1p(t) for t in ot)

    pooled = [[0.0] * 5 for _ in EP_HALF_LIVES]
    ptot = [0.0] * len(EP_HALF_LIVES)
    for rows in other_rows:
        c, t = _decayed_multi(rows, cutoff, EP_HALF_LIVES)
        for j in range(len(EP_HALF_LIVES)):
            ptot[j] += t[j]
            for i in range(5):
                pooled[j][i] += c[j][i]
    ep_s = []
    for j in range(len(EP_HALF_LIVES)):
        s = _smooth(pooled[j], ptot[j])
        ep_s.append(s)
        f.extend(s)
    f.extend(math.log1p(t) for t in ptot)

    seen = {t: y for t, y in own_rows if t < cutoff}
    for lag in range(1, MAX_LAG + 1):
        y = seen.get(cutoff - lag)
        oh = [0.0] * 5
        if y is None:
            f.extend(oh + [1.0])
        else:
            oh[y] = 1.0
            f.extend(oh + [0.0])

    i8, i64 = HALF_LIVES.index(8), HALF_LIVES.index(64)
    f.extend(own_s[i8][i] - own_s[i64][i] for i in range(5))   # drift (own)
    f.extend(ep_s[0][i] - ep_s[1][i] for i in range(5))        # drift (episode)
    f.extend(1.0 if signal == s else 0.0 for s in (-1, 0, 1))
    return f


def expand(z, mtype):
    """[z, z*steady, z*balanced, z*jackpot, type one-hot]: lets each machine type
    respond differently (even in opposite directions) to the same history."""
    t = [1.0 if mtype == x else 0.0 for x in TYPES]
    out = list(z)
    for ti in t:
        out.extend(v * ti for v in z)
    out.extend(t)
    return out


# ------------------------------------------------------------------ inference
def predict_probs(model, f, mtype):
    z = [(x - m) / s for x, m, s in zip(f, model["mean"], model["std"])]
    x = expand(z, mtype)
    nz = [(i, v) for i, v in enumerate(x) if v != 0.0]
    logits = []
    for k in range(5):
        w = model["coef"][k]
        logits.append(model["intercept"][k] + sum(w[i] * v for i, v in nz))
    mx = max(logits)
    ex = [math.exp(l - mx) for l in logits]
    s = sum(ex)
    return [e / s for e in ex]


def machine_query_features(q, hist, by_episode):
    e, m, c = q["episode_id"], q["machine_id"], q["cutoff"]
    own = hist.get((e, m), [])
    others = [hist[(e, mm)] for mm in by_episode.get(e, []) if mm != m]
    return base_features(q["signal"], own, others, c)
