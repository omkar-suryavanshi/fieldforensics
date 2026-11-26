from paths import ROOT, DATA  # noqa: E402
import argparse, json, os, time, math, random
import numpy as np, torch, torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from common import *

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--iters", type=int, default=1500)
ap.add_argument("--bs", type=int, default=8)
ap.add_argument("--crop", type=int, default=384)
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--noise", type=int, default=1)
ap.add_argument("--paired", type=int, default=1)
ap.add_argument("--aug", type=int, default=1)
ap.add_argument("--seg_w", type=float, default=1.0, help="0 = weakly supervised (image labels only)")
ap.add_argument("--holdout", default="", help="source excluded from train/val (leave-one-source-out)")
ap.add_argument("--evals", type=int, default=1)
ap.add_argument("--workers", type=int, default=2)
args = ap.parse_args()

OUT = ROOT + f"runs/{args.name}/"
os.makedirs(OUT, exist_ok=True)
torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
recs = load_splits()
train = [r for r in recs if r["split"] == "train" and r["source"] != args.holdout]
val = [r for r in recs if r["split"] == "val" and r["source"] != args.holdout][::4]  # 25% subset (val is for monitoring only; final checkpoint is used)


def validate(model):
    model.eval()
    ys, ss = [], []
    for r in val:
        for forged in (0, 1):
            img, _ = load_image(r, forged)
            ss.append(image_scores(predict_map(model, img))["smooth_max"]); ys.append(forged)
    model.train()
    return roc_auc_score(ys, ss)


if __name__ == "__main__":
    ds = CropDataset(train, crop=args.crop, n=args.iters * args.bs, paired=bool(args.paired), aug=bool(args.aug), seed=args.seed)
    dl = torch.utils.data.DataLoader(ds, batch_size=args.bs, num_workers=args.workers, pin_memory=True, persistent_workers=True,
                                     prefetch_factor=4)
    model = ForensicUNet(noise=bool(args.noise)).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(1, (i + 1) / 300) * 0.5 * (1 + math.cos(math.pi * min(i, args.iters) / args.iters)))
    scaler = torch.amp.GradScaler()
    log, best, t0 = [], -1, time.time()
    eval_at = {round(args.iters * (k + 1) / args.evals) for k in range(args.evals)}
    run = 0.0
    for it, (x, m, v, y) in enumerate(dl, 1):
        x, m, v, y = x.cuda(non_blocking=True), m.cuda(non_blocking=True), v.cuda(non_blocking=True), y.cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16):
            lg = model(x)
        lg = lg.float()
        img_l = F.binary_cross_entropy_with_logits(topk_pool(lg, v), y)
        loss = 0.5 * img_l
        if args.seg_w > 0:
            bce = (F.binary_cross_entropy_with_logits(lg, m, reduction="none", pos_weight=torch.tensor(5.0, device="cuda")) * v).sum() / v.sum()
            pr = torch.sigmoid(lg) * v
            fm = y > 0.5
            dice = 1 - (2 * (pr[fm] * m[fm]).sum() + 1) / (pr[fm].sum() + m[fm].sum() + 1) if fm.any() else lg.sum() * 0
            loss = loss + args.seg_w * (bce + dice)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(opt); scaler.update(); sched.step()
        run = 0.98 * run + 0.02 * loss.item() if it > 1 else loss.item()
        if it % 100 == 0:
            print(f"it {it} loss {run:.4f} img {img_l.item():.3f} {time.time() - t0:.0f}s", flush=True)
        if it in eval_at:
            auc = validate(model)
            log.append({"iter": it, "loss": run, "val_auc": auc, "time": time.time() - t0})
            print("VAL", log[-1], flush=True)
            if auc > best:
                best = auc
                torch.save(model.state_dict(), OUT + "best.pt")
        if it >= args.iters:
            break
    json.dump({"args": vars(args), "log": log, "best_val_auc": best, "train_docs": len(train), "val_docs": len(val),
               "train_time_s": time.time() - t0}, open(OUT + "train_log.json", "w"), indent=1)
    print("DONE best val auc", best)
