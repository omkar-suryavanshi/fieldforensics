"""Baseline B3: conventional whole-image CNN classifier (ImageNet ResNet-50, document resized to 512x512)."""
from paths import ROOT, DATA  # noqa: E402
import os, json, random, time, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F, torchvision, cv2
from sklearn.metrics import roc_auc_score
from common import load_splits, load_image, degrade, jpeg

OUT = ROOT + "runs/global_cnn/"
S = 512


class DS(torch.utils.data.Dataset):
    def __init__(self, docs, aug, cond="none"):
        self.items = [(r, f) for r in docs for f in (0, 1)]; self.aug, self.cond = aug, cond

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        r, f = self.items[i]
        img, m = load_image(r, f)
        img, _ = degrade(img, m, self.cond)
        if self.aug and random.random() < 0.5:
            img = jpeg(img, random.randint(50, 95))
        img = cv2.resize(img, (S, S), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(img).permute(2, 0, 1).float() / 255
        x = (x - torch.tensor([0.485, 0.456, 0.406])[:, None, None]) / torch.tensor([0.229, 0.224, 0.225])[:, None, None]
        return x, torch.tensor(float(f))


def run(model, docs, cond="none"):
    dl = torch.utils.data.DataLoader(DS(docs, False, cond), batch_size=16, num_workers=2)
    model.eval(); s = []
    with torch.no_grad():
        for x, _ in dl:
            with torch.autocast("cuda", dtype=torch.float16):
                s.append(torch.sigmoid(model(x.cuda()).float()[:, 0]).cpu())
    model.train()
    return torch.cat(s).numpy()


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    torch.manual_seed(0); random.seed(0)
    recs = load_splits()
    tr = [r for r in recs if r["split"] == "train"]; va = [r for r in recs if r["split"] == "val"]; te = [r for r in recs if r["split"] == "testing"]
    model = torchvision.models.resnet50(weights="IMAGENET1K_V2"); model.fc = nn.Linear(2048, 1); model = model.cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler()
    dl = torch.utils.data.DataLoader(DS(tr, True), batch_size=16, shuffle=True, num_workers=2, drop_last=True, persistent_workers=True)
    best, log, t0 = -1, [], time.time()
    for ep in range(4):
        for x, y in dl:
            with torch.autocast("cuda", dtype=torch.float16):
                l = F.binary_cross_entropy_with_logits(model(x.cuda()).float()[:, 0], y.cuda())
            opt.zero_grad(); scaler.scale(l).backward(); scaler.step(opt); scaler.update()
        yv = np.array([f for r in va for f in (0, 1)]); auc = roc_auc_score(yv, run(model, va))
        log.append(dict(epoch=ep, val_auc=auc, t=time.time() - t0)); print(log[-1], flush=True)
        if auc > best:
            best = auc; torch.save(model.state_dict(), OUT + "best.pt")
    model.load_state_dict(torch.load(OUT + "best.pt"))
    json.dump({"log": log, "best_val_auc": best}, open(OUT + "train_log.json", "w"), indent=1)
    for cond in ["none", "jpeg75", "jpeg50", "scale0.5", "social"]:
        s = run(model, te, cond)
        pd.DataFrame([dict(uid=r["uid"], source=r["source"], forged=f) for r in te for f in (0, 1)]).assign(score=s).to_csv(OUT + f"pred_test_{cond}.csv", index=False)
        print(cond, "done", flush=True)
