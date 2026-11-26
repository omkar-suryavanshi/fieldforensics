"""Aggregate all predictions into tables (JSON) and figures for the paper."""
from paths import ROOT, DATA  # noqa: E402
import os, json, glob, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score

R = ROOT + "runs/"
FIG = ROOT + "paper/figures/"
os.makedirs(FIG, exist_ok=True)
SC = "smooth_max"
rng = np.random.default_rng(0)
NB = 2000
SRC = ["cord", "wildreceipt", "sroie", "xfund"]
SRC_LBL = {"cord": "CORD", "wildreceipt": "WildReceipt", "sroie": "SROIE", "xfund": "XFUND"}
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "figure.dpi": 200, "savefig.bbox": "tight"})


def load(path, score=SC):
    d = pd.read_csv(path, dtype={"uid": str})
    if "score" not in d:
        d["score"] = d[score]
    return d


def pair_frame(d):
    p = d.pivot_table(index="uid", columns="forged", values="score", aggfunc="first")
    return p.dropna()


def metrics(d, thr=0.5):
    y, s = d["forged"].values, d["score"].values
    p = pair_frame(d)
    pred = s >= thr
    tp, fp = (pred & (y == 1)).sum(), (pred & (y == 0)).sum()
    fn = (~pred & (y == 1)).sum()
    neg = np.sort(s[y == 0])
    t5 = neg[int(np.ceil(0.95 * len(neg))) - 1]
    out = dict(auc=roc_auc_score(y, s), ap=average_precision_score(y, s), acc=(pred == y).mean(),
               prec=tp / max(1, tp + fp), rec=tp / max(1, tp + fn), f1=2 * tp / max(1, 2 * tp + fp + fn),
               tpr_at_fpr5=(s[y == 1] > t5).mean(), pair_acc=(p[1] > p[0]).mean() + 0.5 * (p[1] == p[0]).mean(), n=len(d))
    for k in ("pix_f1", "pix_iou", "peak_hit"):
        if k in d:
            out[k] = d.loc[d.forged == 1, k].mean()
    if "fp_pix_rate" in d:
        out["auth_fp_pix"] = d.loc[d.forged == 0, "fp_pix_rate"].mean()
    return out


def boot_auc(d, B=NB):
    """Bootstrap over documents (pairs kept together)."""
    p = pair_frame(d)
    a, f = p[0].values, p[1].values
    n = len(a)
    vals = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        vals.append(roc_auc_score(np.r_[np.zeros(n), np.ones(n)], np.r_[a[i], f[i]]))
    return np.percentile(vals, [2.5, 97.5]).tolist()


def paired_boot(d1, d2, B=NB):
    """Two-sided bootstrap p-value for AUC(d1)-AUC(d2) on the same documents."""
    p1, p2 = pair_frame(d1), pair_frame(d2)
    idx = p1.index.intersection(p2.index)
    p1, p2 = p1.loc[idx], p2.loc[idx]
    n = len(idx)
    y = np.r_[np.zeros(n), np.ones(n)]
    obs = roc_auc_score(y, np.r_[p1[0], p1[1]]) - roc_auc_score(y, np.r_[p2[0], p2[1]])
    diffs = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        diffs.append(roc_auc_score(y, np.r_[p1[0].values[i], p1[1].values[i]]) - roc_auc_score(y, np.r_[p2[0].values[i], p2[1].values[i]]))
    diffs = np.array(diffs)
    pval = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return obs, max(pval, 1 / B), np.percentile(diffs, [2.5, 97.5]).tolist()


def ex(p):
    return os.path.exists(p)


T = {}
CONDS = ["none", "jpeg90", "jpeg75", "jpeg50", "scale0.75", "scale0.5", "blur1.0", "noise5", "social"]

# ---------------------------------------------------------------- late fusion (fixed equal weights; nothing tuned on test)
for base, fname in (("main_s0", "fusion"), ("noaug", "fusion_noaug")):
    os.makedirs(R + fname, exist_ok=True)
    for c in CONDS:
        pa, pb = R + f"{base}/pred_test_{c}.csv", R + f"baselines/pred_forensic_gbdt_{c}.csv"
        if ex(pa) and ex(pb):
            a = pd.read_csv(pa, dtype={"uid": str}); b = pd.read_csv(pb, dtype={"uid": str})[["uid", "forged", "score"]].rename(columns={"score": "gb"})
            f = a.merge(b, on=["uid", "forged"])
            f["score"] = 0.5 * (f["smooth_max"] + f["gb"])
            f.to_csv(R + f"{fname}/pred_test_{c}.csv", index=False)

# ---------------------------------------------------------------- main comparison table
methods = {}
for n in ["pixel_lr", "forensic_lr", "forensic_gbdt"]:
    if ex(R + f"baselines/pred_{n}_none.csv"):
        methods[n] = lambda c, n=n: R + f"baselines/pred_{n}_{c}.csv"
if ex(R + "global_cnn/pred_test_none.csv"):
    methods["global_cnn"] = lambda c: R + f"global_cnn/pred_test_{c}.csv"
for n in ["weak", "unpaired", "rgb", "noaug", "main_s0", "main_s1", "main_s2", "fusion", "fusion_noaug"]:
    if ex(R + f"{n}/pred_test_none.csv"):
        methods[n] = lambda c, n=n: R + f"{n}/pred_test_{c}.csv"

T["main"] = {}
D = {}
for n, f in methods.items():
    D[n] = load(f("none"))
    m = metrics(D[n]); m["auc_ci"] = boot_auc(D[n])
    T["main"][n] = m
    print(n, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()})

P = "noaug" if "noaug" in D else "main_s0"   # primary model: clean-trained FieldForensics
T["primary"] = P
seeds = [n for n in ["main_s0", "main_s1", "main_s2"] if n in D]
if seeds:
    T["seeds"] = {k: [float(np.mean([T["main"][s][k] for s in seeds])), float(np.std([T["main"][s][k] for s in seeds]))]
                  for k in ["auc", "acc", "f1", "pair_acc", "tpr_at_fpr5", "pix_f1", "pix_iou", "peak_hit", "ap"]}
    T["seeds"]["n"] = len(seeds)

# significance of ablations vs main_s0
if "main_s0" in D:
    T["signif"] = {}
    for n in D:
        if n != "main_s0":
            o, pv, ci = paired_boot(D["main_s0"], D[n])
            T["signif"][n] = dict(diff=o, p=pv, ci=ci)
    T["signif_primary"] = {}
    for n in D:
        if n != P:
            o, pv, ci = paired_boot(D[P], D[n])
            T["signif_primary"][n] = dict(diff=o, p=pv, ci=ci)
    if "fusion" in D and "forensic_gbdt" in D:
        T["signif_fusion"] = {}
        for n in ("forensic_gbdt", "main_s0"):
            o, pv, ci = paired_boot(D["fusion"], D[n])
            T["signif_fusion"][n] = dict(diff=o, p=pv, ci=ci)

    # aggregation-rule ablation
    raw = pd.read_csv(R + f"{P}/pred_test_none.csv", dtype={"uid": str})
    T["aggregation"] = {}
    for agg in ["max", "smooth_max", "topk", "mean"]:
        d = raw.copy(); d["score"] = d[agg]
        T["aggregation"][agg] = dict(auc=roc_auc_score(d.forged, d.score), pair_acc=metrics(d)["pair_acc"])

    # per source
    T["per_source"] = {}
    for n in D:
        T["per_source"][n] = {s: roc_auc_score(D[n][D[n].source == s].forged, D[n][D[n].source == s].score) for s in SRC}
    # per-source detailed for main
    T["per_source_main"] = {s: metrics(D[P][D[P].source == s]) for s in SRC}
    for s in SRC:
        T["per_source_main"][s]["auc_ci"] = boot_auc(D[P][D[P].source == s])
    T["per_source_R"] = {s: metrics(D["main_s0"][D["main_s0"].source == s]) for s in SRC}

    # by forged-region size quartile
    d = D[P].copy()
    q = d.loc[d.forged == 1, ["uid", "mask_frac"]].set_index("uid")["mask_frac"]
    edges = np.quantile(q, [0, .25, .5, .75, 1])
    d["sizeq"] = d.uid.map(lambda u: int(np.clip(np.searchsorted(edges, q[u], side="right") - 1, 0, 3)))
    T["size"] = {int(k): dict(auc=roc_auc_score(g.forged, g.score), pix_f1=g.loc[g.forged == 1, "pix_f1"].mean(),
                              lo=float(edges[k] * 100), hi=float(edges[k + 1] * 100),
                              n_xfund=int(((g.source == "xfund") & (g.forged == 1)).sum()), n=int((g.forged == 1).sum()),
                              auc_noxf=(roc_auc_score(g[g.source != "xfund"].forged, g[g.source != "xfund"].score) if (g.source != "xfund").sum() > 10 else None))
               for k, g in d.groupby("sizeq")}
    # score calibration per source: median document score of each class
    T["calib"] = {s: dict(auth_med=float(D[P][(D[P].source == s) & (D[P].forged == 0)].score.median()),
                          forg_med=float(D[P][(D[P].source == s) & (D[P].forged == 1)].score.median())) for s in SRC}
    T["calib_R"] = {s: dict(auth_med=float(D["main_s0"][(D["main_s0"].source == s) & (D["main_s0"].forged == 0)].score.median()),
                            forg_med=float(D["main_s0"][(D["main_s0"].source == s) & (D["main_s0"].forged == 1)].score.median())) for s in SRC}

# ---------------------------------------------------------------- robustness
T["robust"] = {}
for n, f in methods.items():
    T["robust"][n] = {}
    for c in CONDS:
        if ex(f(c)):
            dd = load(f(c)); mm = metrics(dd)
            T["robust"][n][c] = dict(auc=mm["auc"], pair_acc=mm["pair_acc"], acc=mm["acc"], pix_f1=mm.get("pix_f1"))
for key, mn in (("robust_source", P), ("robust_source_R", "main_s0")):
    if mn in methods:
        T[key] = {}
        for c in CONDS:
            if ex(methods[mn](c)):
                dd = load(methods[mn](c))
                T[key][c] = {s: roc_auc_score(dd[dd.source == s].forged, dd[dd.source == s].score) for s in SRC}

# ---------------------------------------------------------------- leave-one-source-out
T["loso"] = {}
for s in SRC:
    p = R + f"loso_{s}/pred_src_{s}_none.csv"
    if ex(p) and "main_s0" in D:
        d = load(p); m = metrics(d)
        ind = D["main_s0"][D["main_s0"].source == s]
        T["loso"][s] = dict(out_auc=m["auc"], out_ci=boot_auc(d), out_pair=m["pair_acc"], out_pixf1=m["pix_f1"],
                            in_auc=roc_auc_score(ind.forged, ind.score), in_pair=metrics(ind)["pair_acc"], in_pixf1=ind.loc[ind.forged == 1, "pix_f1"].mean())

# training logs
T["train"] = {}
for p in glob.glob(R + "*/train_log.json"):
    j = json.load(open(p)); T["train"][p.split("/")[-2].split("\\")[-1]] = dict(best_val_auc=j.get("best_val_auc"), time=j.get("train_time_s"))

if ex(R + "baselines/feat_importance.json"):
    T["featimp"] = json.load(open(R + "baselines/feat_importance.json"))
json.dump(T, open(R + "results.json", "w"), indent=1, default=float)

# ================================================================ FIGURES
NAMES = {"pixel_lr": "Pixel LR (prior baseline)", "forensic_lr": "Forensic feat. + LR", "forensic_gbdt": "Forensic feat. + GBDT",
         "global_cnn": "Global ResNet-50", "weak": "Ours-R, image labels only", "main_s0": "FieldForensics-R (ours, aug.)",
         "noaug": "FieldForensics (ours)", "rgb": "Ours-R, RGB only", "unpaired": "Ours-R, unpaired negatives", "fusion": "FieldForensics-R + forensic feat.", "fusion_noaug": "FieldForensics + forensic feat."}
COL = {"pixel_lr": "#9e9e9e", "forensic_lr": "#bcaaa4", "forensic_gbdt": "#8d6e63", "global_cnn": "#ffa726", "weak": "#ab47bc",
       "main_s0": "#64b5f6", "noaug": "#0d47a1", "rgb": "#66bb6a", "unpaired": "#ef5350", "fusion": "#26a69a", "fusion_noaug": "#004d40"}

# ROC
fig, ax = plt.subplots(figsize=(3.4, 3.0))
for n in ["pixel_lr", "global_cnn", "forensic_gbdt", "main_s0", "fusion", "noaug"]:
    if n in D:
        fpr, tpr, _ = roc_curve(D[n].forged, D[n].score)
        ax.plot(fpr, tpr, color=COL[n], lw=1.8 if n == "noaug" else 1.0, ls="--" if n == "fusion" else "-", label=f"{NAMES[n]} ({T['main'][n]['auc']:.3f})")
ax.plot([0, 1], [0, 1], "k:", lw=0.7)
ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate"); ax.legend(loc="lower right", frameon=False, fontsize=6)
ax.set_xlim(0, 1); ax.set_ylim(0, 1.01); ax.grid(alpha=.25)
fig.savefig(FIG + "roc.png"); plt.close(fig)

# robustness
CL = {"none": "Orig.", "jpeg90": "JPEG90", "jpeg75": "JPEG75", "jpeg50": "JPEG50", "scale0.75": "×0.75", "scale0.5": "×0.5",
      "blur1.0": "Blur σ1", "noise5": "Noise σ5", "social": "Re-share"}
fig, ax = plt.subplots(figsize=(7.0, 2.4))
ms = [n for n in ["noaug", "main_s0", "fusion", "forensic_gbdt", "global_cnn", "pixel_lr"] if n in T["robust"]]
w = 0.8 / max(1, len(ms))
for i, n in enumerate(ms):
    xs = [j for j, c in enumerate(CONDS) if c in T["robust"][n]]
    ax.bar(np.array(xs) + (i - len(ms) / 2 + .5) * w, [T["robust"][n][CONDS[j]]["auc"] for j in xs], w, color=COL[n], label=NAMES[n])
ax.axhline(0.5, color="k", ls=":", lw=.7)
ax.set_xticks(range(len(CONDS))); ax.set_xticklabels([CL[c] for c in CONDS]); ax.set_ylabel("Image-level AUC"); ax.set_ylim(0.4, 1.0)
ax.legend(ncol=3, frameon=False, loc="upper right", fontsize=6.5); ax.grid(axis="y", alpha=.25)
fig.savefig(FIG + "robustness.png"); plt.close(fig)

# per-source robustness of main
if "robust_source" in T:
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    cs = [c for c in CONDS if c in T["robust_source"]]
    for s, mk in zip(SRC, "osd^"):
        ax.plot(range(len(cs)), [T["robust_source"][c][s] for c in cs], marker=mk, ms=3, lw=1, label=SRC_LBL[s])
    if "robust_source_R" in T:
        for s, mk in zip(SRC, "osd^"):
            cr = [c for c in cs if c in T["robust_source_R"]]
            ax.plot([cs.index(c) for c in cr], [T["robust_source_R"][c][s] for c in cr], marker=mk, ms=2.5, lw=.8, ls=":", alpha=.6, color=ax.lines[SRC.index(s)].get_color())
    ax.set_xticks(range(len(cs))); ax.set_xticklabels([CL[c] for c in cs], rotation=45, ha="right")
    ax.axhline(0.5, color="k", ls=":", lw=.7); ax.set_ylabel("AUC (solid: FF, dotted: FF-R)"); ax.set_ylim(0.4, 1.01); ax.legend(frameon=False, ncol=2); ax.grid(alpha=.25)
    fig.savefig(FIG + "robust_source.png"); plt.close(fig)

# per-source + LOSO
if T.get("loso"):
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    ss = [s for s in SRC if s in T["loso"]]
    x = np.arange(len(ss))
    ax.bar(x - .2, [T["loso"][s]["in_auc"] for s in ss], .4, color="#1565c0", label="In-domain (all sources)")
    ax.bar(x + .2, [T["loso"][s]["out_auc"] for s in ss], .4, color="#90caf9", label="Source held out")
    for i, s in enumerate(ss):
        lo, hi = T["loso"][s]["out_ci"]; ax.errorbar(i + .2, T["loso"][s]["out_auc"], yerr=[[T["loso"][s]["out_auc"] - lo], [hi - T["loso"][s]["out_auc"]]], color="k", lw=.7, capsize=2)
    ax.set_xticks(x); ax.set_xticklabels([SRC_LBL[s] for s in ss]); ax.set_ylim(0.4, 1.02); ax.axhline(.5, color="k", ls=":", lw=.7)
    ax.set_ylabel("Image-level AUC"); ax.legend(frameon=False, loc="lower left"); ax.grid(axis="y", alpha=.25)
    fig.savefig(FIG + "loso.png"); plt.close(fig)

# size
if "size" in T:
    fig, ax = plt.subplots(figsize=(3.4, 2.0))
    ks = sorted(T["size"])
    ax.plot(ks, [T["size"][k]["auc"] for k in ks], "o-", color="#1565c0", label="Image AUC")
    ax.plot(ks, [T["size"][k]["auc_noxf"] for k in ks], "o:", color="#1565c0", alpha=.6, label="Image AUC, excl. XFUND")
    ax.plot(ks, [T["size"][k]["pix_f1"] for k in ks], "s--", color="#ef5350", label="Pixel F1")
    for k in ks:
        ax.text(k, 0.06, f"{T['size'][k]['n_xfund']}/{T['size'][k]['n']} XF", ha="center", fontsize=5.5, color="#555")
    ax.set_xticks(ks); ax.set_xticklabels([f"Q{k+1}\n{T['size'][k]['lo']:.2f}–{T['size'][k]['hi']:.2f}%" for k in ks])
    ax.set_xlabel("Forged-area quartile (% of image)"); ax.set_ylim(0, 1.02); ax.legend(frameon=False); ax.grid(alpha=.25)
    fig.savefig(FIG + "size.png"); plt.close(fig)
print("figures written")

# ---------------------------------------------------------------- qualitative heatmaps
import cv2
from dataio import load_splits, load_image
md = R + f"{T.get('primary', 'main_s0')}/maps/"
if os.path.isdir(md) and os.listdir(md):
    recs = {r["uid"]: r for r in load_splits()}
    uids = sorted({f.split("_")[0] for f in os.listdir(md)})
    chosen = []
    for s in SRC:
        c = [u for u in uids if recs[u]["source"] == s and os.path.exists(md + f"{u}_0.npy")]
        if c: chosen.append(c[0])
    fig, axes = plt.subplots(len(chosen), 4, figsize=(7.0, 1.25 * len(chosen) + 0.2))
    for i, u in enumerate(chosen):
        r = recs[u]
        fimg, m = load_image(r, 1)
        x1, y1, x2, y2 = r["mbbox"]
        fw, fh = x2 - x1, y2 - y1
        X1, X2 = max(0, x1 - fw), min(fimg.shape[1], x2 + fw)
        Y1, Y2 = max(0, y1 - 4 * fh), min(fimg.shape[0], y2 + 4 * fh)
        maps = []
        for fl in (1, 0):
            pm = np.load(md + f"{u}_{fl}.npy").astype(np.float32)
            pm = cv2.resize(pm, (fimg.shape[1], fimg.shape[0]))
            maps.append(pm)
        dfr = D[T["primary"]] if T.get("primary") in D else None
        sc = {fl: float(dfr[(dfr.uid == u) & (dfr.forged == fl)].score.iloc[0]) for fl in (0, 1)} if dfr is not None else {0: 0, 1: 0}
        axes[i, 0].imshow(fimg[Y1:Y2, X1:X2]); axes[i, 0].add_patch(plt.Rectangle((x1 - X1, y1 - Y1), fw, fh, fill=False, ec="lime", lw=0.8))
        axes[i, 1].imshow(maps[0][Y1:Y2, X1:X2], cmap="inferno", vmin=0, vmax=1)
        aimg, _ = load_image(r, 0)
        axes[i, 2].imshow(aimg[Y1:Y2, X1:X2])
        axes[i, 3].imshow(maps[1][Y1:Y2, X1:X2], cmap="inferno", vmin=0, vmax=1)
        axes[i, 0].set_ylabel(SRC_LBL[r["source"]], fontsize=7)
        axes[i, 1].set_xlabel(f"s = {sc[1]:.2f}", fontsize=6.5, labelpad=1); axes[i, 3].set_xlabel(f"s = {sc[0]:.2f}", fontsize=6.5, labelpad=1)
    for j, t in enumerate(["Forged (GT box in green)", "Predicted map (forged)", "Authentic counterpart", "Predicted map (authentic)"]):
        axes[0, j].set_title(t, fontsize=7)
    for a in axes.ravel(): a.set_xticks([]); a.set_yticks([])
    plt.tight_layout(pad=0.3); fig.savefig(FIG + "qualitative.png"); plt.close(fig)
    print("qualitative written", chosen)

# redesigned figures (overwrite the simpler versions above)
import fig_new  # noqa: E402,F401
