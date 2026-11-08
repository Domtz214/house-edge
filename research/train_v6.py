"""v6 = v5 + own/episode surprise features. Same CLI as train_v5.py."""
import argparse, json, os, sys, time
import numpy as np, lightgbm as lgb
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_v5 import WIN_PARAMS, SIZE_PARAMS

def episodes(F, folder):
    queries, seqs = F.load_split(folder)
    eps = {}
    for q in queries:
        eps.setdefault(q["episode_id"], {})[q["machine_id"]] = (seqs[q["machine_id"]], q["machine_type"], q["cutoff"], q["query_id"])
    return queries, eps

def pseudo(F, folder, min_t=24):
    _, eps = episodes(F, folder)
    Xw, yw, Xs, ys = [], [], [], []
    for e, ms in eps.items():
        wanted = {m: [t for t in sorted(v[0]) if min_t <= t < v[2]] for m, v in ms.items()}
        wanted = {m: c for m, c in wanted.items() if c}
        if not wanted: continue
        feats = F.episode_features({m: (v[0], v[1]) for m, v in ms.items()}, wanted)
        for m, cuts in wanted.items():
            s = ms[m][0]
            for t in cuts:
                f, g = feats[(m, t)]
                Xw.append(f); yw.append(int(s[t] > 0))
                if s[t] > 0: Xs.append(g); ys.append(s[t] - 1)
    return np.array(Xw), np.array(yw), np.array(Xs), np.array(ys)

def query_feats(F, folder):
    queries, eps = episodes(F, folder)
    out = {}
    for e, ms in eps.items():
        feats = F.episode_features({m: (v[0], v[1]) for m, v in ms.items()}, {m: [v[2]] for m, v in ms.items()})
        for m, v in ms.items(): out[v[3]] = feats[(m, v[2])]
    return queries, out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="public"); ap.add_argument("--entry", default="my-entry-v6")
    ap.add_argument("--report", default="v6-report"); ap.add_argument("--final", action="store_true")
    a = ap.parse_args()
    sys.path.insert(0, os.path.abspath(a.entry)); import features as F
    t0 = time.time()
    tr = pseudo(F, os.path.join(a.data, "train")); print("train pseudo", tr[0].shape, f"{time.time()-t0:.0f}s")
    win = lgb.LGBMClassifier(**WIN_PARAMS).fit(tr[0], tr[1]); size = lgb.LGBMClassifier(**SIZE_PARAMS).fit(tr[2], tr[3])
    vq, vf = query_feats(F, os.path.join(a.data, "validation"))
    os.makedirs(a.report, exist_ok=True)
    with open(os.path.join(a.report, "validation_predictions.csv"), "w") as f:
        f.write("query_id,p_0,p_1,p_2,p_5,p_20\n")
        for q in vq:
            fw, g = vf[q["query_id"]]
            pw = win.predict_proba(np.array([fw]))[0, 1]; sz = size.predict_proba(np.array([g]))[0]
            p = [1 - pw] + [pw * x for x in sz]; s = sum(p)
            f.write(q["query_id"] + "," + ",".join(repr(float(x / s)) for x in p) + "\n")
    print("validation predictions written", f"{time.time()-t0:.0f}s")
    if a.final:
        va = pseudo(F, os.path.join(a.data, "validation"))
        win = lgb.LGBMClassifier(**WIN_PARAMS).fit(np.vstack([tr[0], va[0]]), np.concatenate([tr[1], va[1]]))
        size = lgb.LGBMClassifier(**SIZE_PARAMS).fit(np.vstack([tr[2], va[2]]), np.concatenate([tr[3], va[3]]))
        for name, m in (("win_model.json", win), ("size_model.json", size)):
            json.dump(m.booster_.dump_model(), open(os.path.join(a.entry, name), "w"))
        print("saved final models", f"{time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
