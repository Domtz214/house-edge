"""House Edge v5 features + tree inference (Python standard library only).

Shared by research/train_v5.py (training) and predict.py (runner), so the
training features and the submitted features are produced by the same code.

Temporal rule: features for cutoff c use only ticks t < c of the same machine.
Missing ticks are unknown (they add nothing), never a zero payout.
"""
import csv
import math
import os
from collections import defaultdict

PAYOUTS = [0, 1, 2, 5, 20]
IDX = {p: i for i, p in enumerate(PAYOUTS)}
TYPES = ["steady", "balanced", "jackpot"]

# Lag pairs whose win/lose interaction drives the win chance (found in the train data).
PAIRS = [(1, 2), (1, 3), (2, 5), (4, 7)]
HLS = [6, 12, 25, 60]          # half-lives for the win-pattern tables
A = 0.1                        # Beta(A, A) prior per context cell
SHLS = [4, 10, 25, 60, 200]    # half-lives for payout-size features
# Two payout-size profiles (given a win) per machine type, estimated from validation targets.
PROF = {
    "balanced": [[0.29543605, 0.49013145, 0.18379929, 0.03063322],
                 [0.49985311, 0.34792827, 0.13047310, 0.02174552]],
    "jackpot": [[0.14496259, 0.31501378, 0.40501772, 0.13500591],
                [0.33514603, 0.24494620, 0.31493083, 0.10497694]],
    "steady": [[0.46176351, 0.43058920, 0.10549435, 0.00215295],
               [0.75398939, 0.19680849, 0.04821808, 0.00098404]],
}
LGA2 = 2 * math.lgamma(A) - math.lgamma(2 * A)


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_split(folder):
    queries = read_csv(os.path.join(folder, "queries.csv"))
    for q in queries:
        q["cutoff"] = int(float(q["cutoff"]))
    seqs = defaultdict(dict)
    for r in read_csv(os.path.join(folder, "history.csv")):
        seqs[r["machine_id"]][int(float(r["tick"]))] = IDX[int(float(r["payout"]))]
    return queries, seqs


def machine_features(seq, cutoffs, mtype):
    """Return {cutoff: (win_features, size_features)} for the requested cutoffs."""
    want = set(cutoffs)
    tmax = max(want)
    ti = TYPES.index(mtype)
    lr = [math.log(a) - math.log(b) for a, b in zip(PROF[mtype][0], PROF[mtype][1])]
    dec = [0.5 ** (1.0 / h) for h in HLS]
    sdec = [0.5 ** (1.0 / h) for h in SHLS]
    sp = [0.0] * len(HLS)
    sw = [0.0] * len(HLS)
    n1 = [[[0.0] * 4 for _ in PAIRS] for _ in HLS]
    n0 = [[[0.0] * 4 for _ in PAIRS] for _ in HLS]
    ss = [[0.0] * 4 for _ in SHLS]

    def code_at(t, pair):
        a, b = seq.get(t - pair[0]), seq.get(t - pair[1])
        if a is None or b is None:
            return None
        return 2 * (a > 0) + (b > 0)

    codes = {}
    out = {}
    for c in range(1, tmax + 1):
        # fold tick c-1 into the accumulators -> state = sum over t < c
        t = c - 1
        v = seq.get(t)
        for h, d in enumerate(dec):
            sp[h] *= d
            sw[h] *= d
            for p in range(len(PAIRS)):
                for k in range(4):
                    n1[h][p][k] *= d
                    n0[h][p][k] *= d
        for h, d in enumerate(sdec):
            for k in range(4):
                ss[h][k] *= d
        if v is not None:
            w = 1.0 if v > 0 else 0.0
            pc = [code_at(t, pair) for pair in PAIRS]
            for h in range(len(HLS)):
                sp[h] += 1.0
                sw[h] += w
                for p, k in enumerate(pc):
                    if k is not None:
                        if w:
                            n1[h][p][k] += 1.0
                        else:
                            n0[h][p][k] += 1.0
            if v > 0:
                for h in range(len(SHLS)):
                    ss[h][v - 1] += 1.0
        if c not in want:
            continue
        cur = [code_at(c, pair) for pair in PAIRS]
        f = []
        for h in range(len(HLS)):
            f.append((sw[h] + 0.5) / (sp[h] + 1.0))
            f.append(math.log1p(sp[h]))
            lls, pws, ns = [], [], []
            for p in range(len(PAIRS)):
                ll = 0.0
                for k in range(4):
                    a1, a0 = n1[h][p][k], n0[h][p][k]
                    ll += math.lgamma(a1 + A) + math.lgamma(a0 + A) - math.lgamma(a1 + a0 + 2 * A) - LGA2
                lls.append(ll)
                k = cur[p]
                if k is None:
                    pws.append(None)
                    ns.append(-1.0)
                else:
                    a1, a0 = n1[h][p][k], n0[h][p][k]
                    pws.append((a1 + A) / (a1 + a0 + 2 * A))
                    ns.append(a1 + a0)
            mx = max(lls)
            ws = [math.exp(x - mx) for x in lls]
            sw_ = sum(ws)
            ws = [x / sw_ for x in ws]
            num = sum(ws[p] * pws[p] for p in range(len(PAIRS)) if pws[p] is not None)
            den = sum(ws[p] for p in range(len(PAIRS)) if pws[p] is not None)
            f.append(num / den if den > 0 else -1.0)
            for p in range(len(PAIRS)):
                f.extend([pws[p] if pws[p] is not None else -1.0, ns[p], ws[p]])
            f.append(max(ws))
        f.append(float(ti))
        g = []
        for h in range(len(SHLS)):
            n = sum(ss[h])
            g.extend((ss[h][k] + 0.25) / (n + 1.0) for k in range(4))
            g.append(math.log1p(n))
            g.append(sum(ss[h][k] * lr[k] for k in range(4)))
        g.append(float(ti))
        out[c] = (f, g)
    return out


# ------------------------------------------------------------------ LightGBM tree inference
def _tree_value(node, x):
    # features never contain NaN (missing contexts are encoded as -1.0)
    while "leaf_value" not in node:
        if x[node["split_feature"]] <= node["threshold"]:
            node = node["left_child"]
        else:
            node = node["right_child"]
    return node["leaf_value"]


def gbm_raw(model, x):
    """Raw scores per class from a LightGBM dump_model() dict."""
    k = model["num_tree_per_iteration"]
    raw = [0.0] * k
    for i, tree in enumerate(model["tree_info"]):
        raw[i % k] += _tree_value(tree["tree_structure"], x)
    return raw


def win_prob(model, f):
    return 1.0 / (1.0 + math.exp(-gbm_raw(model, f)[0]))


def size_probs(model, g):
    raw = gbm_raw(model, g)
    mx = max(raw)
    e = [math.exp(r - mx) for r in raw]
    s = sum(e)
    return [x / s for x in e]
