from paths import ROOT, DATA  # noqa: E402
import argparse, json, os, time
import numpy as np, pandas as pd, torch
from common import *

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--conds", default="none")
ap.add_argument("--set", default="test", help="test | holdout:<source> (all docs of a source)")
ap.add_argument("--save_maps", default="")
args = ap.parse_args()
OUT = ROOT + f"runs/{args.name}/"
cfg = json.load(open(OUT + "train_log.json"))["args"]
model = ForensicUNet(noise=bool(cfg["noise"]), pretrained=False).cuda()
model.load_state_dict(torch.load(OUT + "best.pt", map_location="cuda"))
model.eval()
recs = load_splits()
if args.set == "test":
    docs = [r for r in recs if r["split"] == "testing"]
else:  # src:<source> -> test-split documents of one source (used for leave-one-source-out)
    src = args.set.split(":")[1]
    docs = [r for r in recs if r["split"] == "testing" and r["source"] == src]
save = set(args.save_maps.split(",")) if args.save_maps else set()
os.makedirs(OUT + "maps", exist_ok=True)


def pix_metrics(pm, m, thr=0.5):
    p = pm > thr
    tp = float((p & (m > 0)).sum()); fp = float((p & (m == 0)).sum()); fn = float((~p & (m > 0)).sum())
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 1.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 1.0
    return f1, iou, fp / p.size


for cond in args.conds.split(","):
    rows, t0 = [], time.time()
    for r in docs:
        for forged in (0, 1):
            img, m = load_image(r, forged)
            img, m = degrade(img, m, cond)
            pm = predict_map(model, img)
            s = image_scores(pm)
            f1, iou, fpr = pix_metrics(pm, m)
            # oracle-threshold-free localisation: is the argmax inside the forged field?
            hit = float(m.ravel()[cv2.blur(pm, (9, 9)).argmax()] > 0) if forged else np.nan
            rows.append(dict(uid=r["uid"], source=r["source"], field=r["field"], mask_frac=r["mask_frac"], forged=forged, **s,
                             pix_f1=f1 if forged else np.nan, pix_iou=iou if forged else np.nan, fp_pix_rate=fpr, peak_hit=hit))
            if cond == "none" and r["uid"] in save:
                small = cv2.resize(pm, (pm.shape[1] // 2, pm.shape[0] // 2)) if max(pm.shape) > 1500 else pm
                np.save(OUT + f"maps/{r['uid']}_{forged}.npy", small.astype(np.float16))
    tag = args.set.replace(":", "_")
    pd.DataFrame(rows).to_csv(OUT + f"pred_{tag}_{cond}.csv", index=False)
    print(cond, len(rows), f"{time.time() - t0:.0f}s", flush=True)
