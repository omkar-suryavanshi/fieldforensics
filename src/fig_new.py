"""New, richer figure set for the paper."""
from paths import ROOT, DATA  # noqa: E402
import json, numpy as np, pandas as pd, cv2, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
from sklearn.metrics import roc_auc_score
from dataio import load_splits, load_image

R = ROOT + "runs/"
FIG = ROOT + "paper/figures/"
T = json.load(open(R + "results.json"))
BLUE, ORANGE, AQUA, YELLOW, VIOLET, RED, GREEN = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6", "#e34948", "#008300"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
SRC = ["cord", "wildreceipt", "sroie", "xfund"]
SL = {"cord": "CORD", "wildreceipt": "WildReceipt", "sroie": "SROIE", "xfund": "XFUND"}
plt.rcParams.update({"font.family": "serif", "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "legend.fontsize": 6.8,
                     "xtick.labelsize": 6.8, "ytick.labelsize": 6.8, "figure.dpi": 220, "savefig.bbox": "tight",
                     "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False})
recs = {r["uid"]: r for r in load_splits()}


def tidy(ax, grid_axis="y"):
    ax.grid(axis=grid_axis, color=GRID, lw=0.6); ax.set_axisbelow(True)


# ------------------------------------------------------------------ 1. Teaser: a forgery caught on the full page
def teaser():
    u = "000000001"; r = recs[u]
    f, m = load_image(r, 1); a, _ = load_image(r, 0)
    pm = np.load(R + f"noaug/maps/{u}_1.npy").astype(np.float32); pm = cv2.resize(pm, (f.shape[1], f.shape[0]))
    pa = np.load(R + f"noaug/maps/{u}_0.npy").astype(np.float32); pa = cv2.resize(pa, (a.shape[1], a.shape[0]))
    x1, y1, x2, y2 = r["mbbox"]; fw, fh = x2 - x1, y2 - y1
    X1, X2, Y1, Y2 = max(0, x1 - fw), min(f.shape[1], x2 + fw), max(0, y1 - 3 * fh), min(f.shape[0], y2 + 3 * fh)
    pred = pd.read_csv(R + "noaug/pred_test_none.csv", dtype={"uid": str})
    sf = float(pred[(pred.uid == u) & (pred.forged == 1)].smooth_max.iloc[0]); sa = float(pred[(pred.uid == u) & (pred.forged == 0)].smooth_max.iloc[0])
    fig = plt.figure(figsize=(7.0, 2.55))
    gs = fig.add_gridspec(2, 5, width_ratios=[1.05, 1.05, 1.6, 1.6, 1.2], wspace=0.08, hspace=0.12)
    ax0 = fig.add_subplot(gs[:, 0]); ax0.imshow(a); ax0.set_title("Authentic", color=INK)
    ax1 = fig.add_subplot(gs[:, 1]); ax1.imshow(f); ax1.imshow(np.ma.masked_less(pm, 0.3), cmap="inferno", alpha=0.55, vmin=0, vmax=1)
    ax1.add_patch(Rectangle((X1, Y1), X2 - X1, Y2 - Y1, fill=False, ec=BLUE, lw=1.0)); ax1.set_title("Forged + map", color=INK)
    za = fig.add_subplot(gs[0, 2]); za.imshow(a[Y1:Y2, X1:X2]); za.set_title("Original field (zoom)", color=INK)
    zf = fig.add_subplot(gs[1, 2]); zf.imshow(f[Y1:Y2, X1:X2]); zf.set_xlabel("GPT-Image-2 forged field", color=INK)
    ma = fig.add_subplot(gs[0, 3]); ma.imshow(pa[Y1:Y2, X1:X2], cmap="inferno", vmin=0, vmax=1); ma.set_title("Predicted tampering map", color=INK)
    mf = fig.add_subplot(gs[1, 3]); mf.imshow(pm[Y1:Y2, X1:X2], cmap="inferno", vmin=0, vmax=1)
    mf.add_patch(Rectangle((x1 - X1, y1 - Y1), fw, fh, fill=False, ec="#7dd87d", lw=0.9, ls="--"))
    for ax in (ax0, ax1, za, zf, ma, mf):
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_visible(True); s.set_color(GRID)
    v = fig.add_subplot(gs[:, 4]); v.axis("off")
    for yy, lab, sc, col in ((0.72, "Authentic", sa, BLUE), (0.25, "Forged", sf, RED)):
        v.text(0.05, yy + 0.12, lab, fontsize=8, color=INK, weight="bold")
        v.add_patch(FancyBboxPatch((0.05, yy - 0.02), 0.85, 0.07, boxstyle="round,pad=0.005", fc=GRID, ec="none"))
        v.add_patch(FancyBboxPatch((0.05, yy - 0.02), 0.85 * sc, 0.07, boxstyle="round,pad=0.005", fc=col, ec="none"))
        v.text(0.05, yy - 0.12, f"score s = {sc:.2f}  →  " + ("FORGED" if sc >= 0.5 else "genuine"), fontsize=7, color=INK2)
    v.set_xlim(0, 1); v.set_ylim(0, 1)
    fig.savefig(FIG + "teaser.png"); plt.close(fig)


# ------------------------------------------------------------------ 2. Architecture diagram
def architecture():
    fig, ax = plt.subplots(figsize=(7.0, 2.6)); ax.set_xlim(0, 100); ax.set_ylim(0, 40); ax.axis("off")

    def blk(x, y, w, h, fc, t=None, fs=5.8, tc=INK):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25", fc=fc, ec="none"))
        if t: ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs, color=tc)

    def arrow(x1, y1, x2, y2, c=INK2, ls="-", rad=0.0):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=6, lw=0.7, color=c, ls=ls, connectionstyle=f"arc3,rad={rad}"))
    LB, LO, LA, LG = "#d6e6f9", "#fde0d3", "#cdeee1", "#ecebe6"
    blk(0, 15, 8, 12, LG, "Input page\nH×W×3\n(native\nresolution)")
    blk(11, 23, 8, 6, LB, "RGB\nnormalise"); blk(11, 13, 8, 6, LO, "SRM\nresidual ×3")
    arrow(8, 21, 11, 26); arrow(8, 21, 11, 16)
    blk(21, 15, 4, 12, "#b9d3f3", "6-ch\nstem", 5.2); arrow(19, 26, 21, 23); arrow(19, 16, 21, 19)
    enc = [(28, 13, 16, "64\n1/4"), (34, 15, 12, "128\n1/8"), (40, 17, 8, "256\n1/16"), (46, 18.5, 5, "512\n1/32")]
    dec = [(54, 17, 8, "256"), (60, 15, 12, "128"), (66, 13, 16, "64"), (72, 11, 20, "64\n1/2"), (78, 9, 24, "32\n1/1")]
    prev = (25, 21)
    for x, y, h, t in enc:
        blk(x, y, 4.5, h, BLUE, t, 5.2, "white"); arrow(prev[0], prev[1], x, y + h / 2); prev = (x + 4.5, y + h / 2)
    for x, y, h, t in dec:
        blk(x, y, 4.5, h, AQUA, t, 5.2, "white"); arrow(prev[0], prev[1], x, y + h / 2); prev = (x + 4.5, y + h / 2)
    for (ex, ey, eh, _), (dx, dy, dh, _) in zip(enc[2::-1], dec[:3]):
        ax.add_patch(FancyArrowPatch((ex + 2.25, ey + eh), (dx + 2.25, dy + dh), arrowstyle="-|>", mutation_scale=5, lw=0.6, color=VIOLET, ls="--", connectionstyle="arc3,rad=-0.35"))
    ax.add_patch(FancyArrowPatch((23, 27), (74, 31), arrowstyle="-|>", mutation_scale=5, lw=0.6, color=VIOLET, ls="--", connectionstyle="arc3,rad=-0.12"))
    ax.add_patch(FancyArrowPatch((4, 27), (80, 33), arrowstyle="-|>", mutation_scale=5, lw=0.8, color=ORANGE, ls="--", connectionstyle="arc3,rad=-0.12"))
    ax.text(42, 37.2, "full-resolution skip (2×conv, 32 ch) — keeps 2–4 px glyph strokes", color=ORANGE, fontsize=5.8, ha="center")
    ax.text(38, 10.0, "ResNet-34 encoder (ImageNet)", color=BLUE, fontsize=6, ha="center")
    ax.text(69, 6.0, "U-Net decoder", color=AQUA, fontsize=6, ha="center")
    ax.text(42, 31.5, "skip connections", color=VIOLET, fontsize=5.6, ha="center")
    blk(85, 22, 13, 8, "#fbd6d5", "Tamper map P\n(sigmoid, H×W)"); arrow(82.5, 21, 85, 25)
    blk(84, 10, 15, 8, "#fbd6d5", "Page score\ns = max(box$_{9×9}$ * P)"); arrow(91.5, 22, 91.5, 18)
    blk(30, 0, 40, 4.2, LG, "Loss = BCE(pixels) + Dice + ½·BCE(top-256 pooled logit, image label)", 5.6)
    fig.savefig(FIG + "method.png"); plt.close(fig)


# ------------------------------------------------------------------ 3. Dataset statistics
def dataset_stats():
    rs = list(recs.values())
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 1.85), gridspec_kw=dict(width_ratios=[1.1, 1.3, 1.2], wspace=0.45))
    ax = axs[0]
    sp = ["train", "val", "testing"]; cols = [BLUE, ORANGE, AQUA]; lab = ["Train", "Val", "Test"]
    left = np.zeros(4)
    for s, c, l in zip(sp, cols, lab):
        v = np.array([sum(1 for r in rs if r["source"] == k and r["split"] == s) for k in SRC])
        ax.barh(range(4), v, left=left, color=c, label=l, height=0.6, edgecolor="#fcfcfb", lw=1)
        left += v
    for i, t in enumerate(left): ax.text(t + 15, i, f"{int(t)}", va="center", fontsize=6.3, color=INK2)
    ax.set_yticks(range(4)); ax.set_yticklabels([SL[k] for k in SRC]); ax.invert_yaxis(); ax.set_xlabel("Documents (pairs)")
    ax.legend(frameon=False, loc="lower right", ncol=1); tidy(ax, "x"); ax.set_title("(a) Split composition", loc="left")
    ax = axs[1]
    fr = np.array([r["mask_frac"] * 100 for r in rs])
    bins = np.logspace(np.log10(fr.min()), np.log10(fr.max()), 30)
    for k, c in zip(SRC, [BLUE, ORANGE, AQUA, YELLOW]):
        ax.hist([r["mask_frac"] * 100 for r in rs if r["source"] == k], bins=bins, color=c, alpha=0.85, label=SL[k], histtype="stepfilled", lw=0)
    ax.set_xscale("log"); ax.axvline(np.median(fr), color=INK, lw=0.8, ls="--"); ax.text(np.median(fr) * 1.1, ax.get_ylim()[1] * 0.92, f"median {np.median(fr):.2f}%", fontsize=6.2, color=INK)
    ax.set_xlabel("Forged area (% of page, log scale)"); ax.set_ylabel("Documents"); ax.legend(frameon=False, fontsize=6, loc="upper left"); tidy(ax)
    ax.set_title("(b) The edit is tiny", loc="left")
    ax = axs[2]
    def ftype(fn):
        fn = fn.lower()
        if "tel" in fn or "phone" in fn: return "Phone"
        if "date" in fn or "time" in fn: return "Date / time"
        if "price" in fn or "total" in fn or "tax" in fn or "cnt" in fn: return "Price / total"
        if "addr" in fn: return "Address"
        if "answer" in fn: return "Form answer"
        return "Other key"
    ct = pd.Series([ftype(r["field"]) for r in rs]).value_counts()
    ax.barh(range(len(ct)), ct.values, color=BLUE, height=0.6)
    for i, v in enumerate(ct.values): ax.text(v + 15, i, str(v), va="center", fontsize=6.3, color=INK2)
    ax.set_yticks(range(len(ct))); ax.set_yticklabels(ct.index); ax.invert_yaxis(); ax.set_xlabel("Forged fields"); tidy(ax, "x")
    ax.set_title("(c) What was forged", loc="left")
    fig.savefig(FIG + "dataset_stats.png"); plt.close(fig)


# ------------------------------------------------------------------ 4. Per-source grouped bars
def per_source_bars():
    ps = T["per_source"]
    ms = [("noaug", "FieldForensics", BLUE), ("main_s0", "FieldForensics-R", ORANGE), ("forensic_gbdt", "Forensic GBDT", AQUA), ("fusion", "FF-R + GBDT", YELLOW)]
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    w = 0.2
    for i, (k, l, c) in enumerate(ms):
        xs = np.arange(4) + (i - 1.5) * w
        vals = [ps[k][s] for s in SRC]
        ax.bar(xs, vals, w - 0.03, color=c, label=l, edgecolor="#fcfcfb", lw=0.6)
        for x, v in zip(xs, vals): ax.text(x, v + 0.008, f"{v:.2f}", ha="center", fontsize=4.6, color=INK2, rotation=90, va="bottom")
    ax.axhline(0.5, color=INK2, ls=":", lw=0.7); ax.text(3.55, 0.51, "chance", fontsize=5.5, color=INK2, ha="right")
    ax.set_xticks(range(4)); ax.set_xticklabels([SL[s] + ("\n(JPEG)" if s in ("wildreceipt", "sroie") else "\n(PNG)") for s in SRC])
    ax.set_ylim(0.45, 1.09); ax.set_ylabel("Image-level AUC"); ax.legend(frameon=False, ncol=2, fontsize=6, loc="upper center", bbox_to_anchor=(0.5, 1.2)); tidy(ax)
    fig.savefig(FIG + "per_source.png"); plt.close(fig)


# ------------------------------------------------------------------ 5. Robustness heatmap
def robustness_heatmap():
    rb = T["robust"]
    rows = [("noaug", "FieldForensics"), ("main_s0", "FieldForensics-R"), ("fusion", "FF-R + GBDT"), ("forensic_gbdt", "Forensic GBDT"),
            ("forensic_lr", "Forensic LR"), ("global_cnn", "Global ResNet-50"), ("pixel_lr", "Pixel LR (prior)")]
    cols = [("none", "Clean"), ("jpeg90", "JPEG 90"), ("jpeg75", "JPEG 75"), ("jpeg50", "JPEG 50"), ("scale0.75", "×0.75"), ("scale0.5", "×0.5"),
            ("blur1.0", "Blur σ1"), ("noise5", "Noise σ5"), ("social", "Re-share")]
    M = np.full((len(rows), len(cols)), np.nan)
    for i, (k, _) in enumerate(rows):
        for j, (c, _) in enumerate(cols):
            v = rb.get(k, {}).get(c, {}).get("auc")
            if v is not None: M[i, j] = v
    fig, ax = plt.subplots(figsize=(7.0, 2.05))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seq", ["#f3f2ee", "#b9d3f3", "#6aa3e6", "#2a78d6", "#123f78"])
    im = ax.imshow(np.ma.masked_invalid(M), cmap=cmap, vmin=0.5, vmax=1.0, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isnan(M[i, j]):
                ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, fc="white", ec="#d8d7d2", hatch="///", lw=0)); ax.text(j, i, "n/e", ha="center", va="center", fontsize=5.5, color="#a3a29c")
            else:
                ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center", fontsize=6.3, color="white" if M[i, j] > 0.78 else INK)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels([c[1] for c in cols]); ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[1] for r in rows])
    ax.tick_params(length=0); [s.set_visible(False) for s in ax.spines.values()]
    ax.set_xticks(np.arange(-.5, len(cols)), minor=True); ax.set_yticks(np.arange(-.5, len(rows)), minor=True)
    ax.grid(which="minor", color="white", lw=1.5); ax.tick_params(which="minor", length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01); cb.set_label("AUC", fontsize=6.5); cb.outline.set_visible(False); cb.ax.tick_params(labelsize=6)
    fig.savefig(FIG + "robustness.png"); plt.close(fig)


# ------------------------------------------------------------------ 6. Score distributions
def score_dists():
    ms = [("baselines/pred_pixel_lr_none.csv", "score", "Pixel LR (prior report)"), ("baselines/pred_forensic_gbdt_none.csv", "score", "Forensic GBDT"),
          ("main_s0/pred_test_none.csv", "smooth_max", "FieldForensics-R"), ("noaug/pred_test_none.csv", "smooth_max", "FieldForensics")]
    fig, axs = plt.subplots(1, 4, figsize=(7.0, 1.6), sharey=False, gridspec_kw=dict(wspace=0.3))
    bins = np.linspace(0, 1, 26)
    for ax, (p, col, name) in zip(axs, ms):
        d = pd.read_csv(R + p)
        ax.hist(d[d.forged == 0][col], bins=bins, color=BLUE, alpha=0.75, label="Authentic", histtype="stepfilled", lw=0)
        ax.hist(d[d.forged == 1][col], bins=bins, color=RED, alpha=0.6, label="Forged", histtype="stepfilled", lw=0)
        auc = roc_auc_score(d.forged, d[col]); ax.set_title(f"{name}\nAUC {auc:.3f}", fontsize=7)
        ax.set_xlabel("Document score"); tidy(ax); ax.set_xlim(0, 1)
    axs[0].set_ylabel("Documents"); axs[0].legend(frameon=False, fontsize=6, loc="upper left")
    fig.savefig(FIG + "score_dist.png"); plt.close(fig)


# ------------------------------------------------------------------ 7. Accuracy–robustness trade-off
def tradeoff():
    rb = T["robust"]
    pts = [("noaug", "FieldForensics", BLUE), ("main_s0", "FieldForensics-R", BLUE), ("fusion", "FF-R + GBDT", INK2), ("fusion_noaug", "FF + GBDT", INK2),
           ("forensic_gbdt", "Forensic GBDT", INK2), ("forensic_lr", "Forensic LR", INK2), ("global_cnn", "Global ResNet-50", INK2), ("pixel_lr", "Pixel LR", INK2)]
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    ax.plot([0.5, 1], [0.5, 1], color=GRID, lw=0.8); ax.text(0.665, 0.69, "no loss", fontsize=5.5, color="#a3a29c", rotation=50)
    off = {"noaug": (0.01, 0.002, "left"), "main_s0": (-0.01, 0.004, "right"), "fusion": (-0.01, 0.002, "right"), "fusion_noaug": (0.01, -0.006, "left"),
           "forensic_gbdt": (0, 0.01, "center"), "forensic_lr": (-0.006, 0.01, "right"), "global_cnn": (0.008, 0.01, "left"), "pixel_lr": (0.008, -0.018, "left")}
    for k, l, c in pts:
        cl, j = rb.get(k, {}).get("none", {}).get("auc"), rb.get(k, {}).get("jpeg75", {}).get("auc")
        if cl is None or j is None: continue
        ax.scatter(cl, j, s=34 if c == BLUE else 20, color=c, edgecolor="#fcfcfb", lw=1, zorder=3)
        dx, dy, ha = off[k]; ax.text(cl + dx, j + dy, l, fontsize=6, ha=ha, color=INK if c == BLUE else INK2)
    if "noaug" in rb and "main_s0" in rb:
        ax.annotate("", (rb["main_s0"]["none"]["auc"], rb["main_s0"]["jpeg75"]["auc"]), (rb["noaug"]["none"]["auc"], rb["noaug"]["jpeg75"]["auc"]),
                    arrowprops=dict(arrowstyle="->", color=ORANGE, lw=0.9, ls="--"))
        ax.text(0.9, 0.652, "degradation\naugmentation", fontsize=5.6, color=ORANGE, ha="left")
    ax.set_xlim(0.48, 1.06); ax.set_ylim(0.48, 0.71); ax.set_xlabel("AUC on clean pages"); ax.set_ylabel("AUC after JPEG-75 re-compression"); tidy(ax, "both")
    fig.savefig(FIG + "tradeoff.png"); plt.close(fig)


# ------------------------------------------------------------------ 8. ROC restyled
def roc():
    from sklearn.metrics import roc_curve
    ms = [("baselines/pred_pixel_lr_none.csv", "score", "Pixel LR (prior)", "#a3a29c", ":"), ("global_cnn/pred_test_none.csv", "score", "Global ResNet-50", "#a3a29c", "--"),
          ("baselines/pred_forensic_gbdt_none.csv", "score", "Forensic GBDT", AQUA, "-"), ("main_s0/pred_test_none.csv", "smooth_max", "FieldForensics-R", ORANGE, "-"),
          ("noaug/pred_test_none.csv", "smooth_max", "FieldForensics", BLUE, "-")]
    fig, ax = plt.subplots(figsize=(3.3, 2.7))
    for p, col, name, c, ls in ms:
        d = pd.read_csv(R + p); fpr, tpr, _ = roc_curve(d.forged, d[col])
        ax.plot(fpr, tpr, color=c, ls=ls, lw=2 if name == "FieldForensics" else 1.3, label=f"{name} ({roc_auc_score(d.forged, d[col]):.3f})")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate"); ax.legend(frameon=False, loc="lower right", fontsize=6.2)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.01); tidy(ax, "both")
    fig.savefig(FIG + "roc.png"); plt.close(fig)


for fn in (teaser, architecture, dataset_stats, per_source_bars, robustness_heatmap, score_dists, tradeoff, roc):
    fn(); print("ok", fn.__name__, flush=True)
