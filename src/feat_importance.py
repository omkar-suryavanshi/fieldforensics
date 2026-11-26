"""Grouped permutation importance of the forensic-feature GBDT (AUC drop on the test set), overall and per source."""
from paths import ROOT, DATA  # noqa: E402
import json, numpy as np, pandas as pd
from multiprocessing import Pool
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from baselines import extract
from dataio import load_splits

GROUPS = {"ELA": list(range(0, 6)) + [18], "Noise residual": list(range(6, 12)) + [19], "JPEG blockiness": list(range(12, 18))}

if __name__ == "__main__":
    recs = load_splits()
    tr = [r for r in recs if r["split"] in ("train", "val")]
    te = [r for r in recs if r["split"] == "testing"]
    with Pool(2) as pool:
        Ftr, _, ytr, _ = extract(tr, "none", pool)
        Fte, _, yte, meta = extract(te, "none", pool)
    np.savez(ROOT + "runs/baselines/features_none.npz", Ftr=Ftr, ytr=ytr, Fte=Fte, yte=yte)
    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0).fit(Ftr, ytr)
    rng = np.random.default_rng(0)
    out = {}
    for src in [None, "cord", "wildreceipt", "sroie", "xfund"]:
        idx = np.arange(len(yte)) if src is None else np.where(meta.source.values == src)[0]
        base = roc_auc_score(yte[idx], m.predict_proba(Fte[idx])[:, 1])
        res = {"base": base}
        for g, cols in GROUPS.items():
            drops = []
            for _ in range(10):
                X = Fte[idx].copy()
                X[:, cols] = X[rng.permutation(len(idx))][:, cols]
                drops.append(base - roc_auc_score(yte[idx], m.predict_proba(X)[:, 1]))
            res[g] = float(np.mean(drops))
        out[src or "all"] = res
        print(src or "all", {k: round(v, 3) for k, v in res.items()}, flush=True)
    json.dump(out, open(ROOT + "runs/baselines/feat_importance.json", "w"), indent=1)
