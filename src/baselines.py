"""Classical baselines: (B1) raw-pixel logistic regression (the original report's baseline) and
(B2) hand-crafted forensic statistics (ELA, SRM noise residual, JPEG 8x8 blockiness) + LR / gradient boosting."""
from paths import ROOT, DATA  # noqa: E402
import json, os, numpy as np, pandas as pd, cv2
from multiprocessing import Pool
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from dataio import load_splits, load_image, jpeg, degrade

OUT = ROOT + "runs/baselines/"


def block_stats(mp, b=32):
    h, w = mp.shape[0] // b * b, mp.shape[1] // b * b
    if h == 0 or w == 0:
        return [0] * 6
    bl = mp[:h, :w].reshape(h // b, b, w // b, b).mean((1, 3)).ravel()
    med = np.median(bl) + 1e-6
    return [bl.mean(), bl.std(), bl.max(), np.percentile(bl, 99), bl.max() / med, bl.min() / med]


def feats(img):
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    f = []
    ela = np.abs(img.astype(np.float32) - jpeg(img, 90).astype(np.float32)).mean(2)
    f += block_stats(ela)
    k = np.array([[-1, 2, -1], [2, -4, 2], [-1, 2, -1]], np.float32) / 4
    res = cv2.filter2D(g, -1, k)
    f += block_stats(np.abs(res))
    # JPEG blockiness: gradient energy on 8-pixel grid lines vs off-grid
    dx = np.abs(np.diff(g, axis=1)); dy = np.abs(np.diff(g, axis=0))
    gx = np.zeros_like(dx); gx[:, 7::8] = 1
    gy = np.zeros_like(dy); gy[7::8, :] = 1
    bmap = np.zeros_like(g)
    bmap[:, :-1] += dx * (gx * 7 - (1 - gx))
    bmap[:-1, :] += dy * (gy * 7 - (1 - gy))
    f += block_stats(bmap)
    f += [ela.mean(), np.abs(res).mean()]
    return f


def pix(img):
    return cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), (64, 64), interpolation=cv2.INTER_AREA).ravel() / 255.0


def job(a):
    r, forged, cond = a
    img, m = load_image(r, forged)
    img, m = degrade(img, m, cond)
    return feats(img), pix(img)


def extract(docs, cond, pool):
    jobs = [(r, f, cond) for r in docs for f in (0, 1)]
    res = pool.map(job, jobs, chunksize=4)
    F = np.array([x[0] for x in res], np.float32); P = np.array([x[1] for x in res], np.float32)
    y = np.array([f for _, f, _ in jobs]); meta = pd.DataFrame([dict(uid=r["uid"], source=r["source"], forged=f) for r, f, _ in jobs])
    return np.nan_to_num(F), P, y, meta


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    recs = load_splits()
    tr = [r for r in recs if r["split"] in ("train", "val")]
    te = [r for r in recs if r["split"] == "testing"]
    conds = [c for c in ["none", "jpeg90", "scale0.75", "blur1.0", "noise5"] if c == "none" or not os.path.exists(OUT + f"pred_forensic_gbdt_{c}.csv")]
    with Pool(1) as pool:
        Ftr, Ptr, ytr, _ = extract(tr, "none", pool)
        models = {"pixel_lr": (make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=3000)), "P"),
                  "forensic_lr": (make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000)), "F"),
                  "forensic_gbdt": (HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0), "F")}
        for n, (mdl, k) in models.items():
            mdl.fit(Ptr if k == "P" else Ftr, ytr)
        for cond in conds:
            Fte, Pte, yte, meta = extract(te, cond, pool)
            for n, (mdl, k) in models.items():
                d = meta.copy(); d["score"] = mdl.predict_proba(Pte if k == "P" else Fte)[:, 1]
                d.to_csv(OUT + f"pred_{n}_{cond}.csv", index=False)
            print(cond, "done", flush=True)
