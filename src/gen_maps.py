"""Save probability maps of a trained model for a few test documents (qualitative figure)."""
from paths import ROOT, DATA  # noqa: E402
import sys, json, os, numpy as np, torch, cv2
from common import ForensicUNet, load_splits, load_image, predict_map

name, uids = sys.argv[1], sys.argv[2].split(",")
OUT = ROOT + f"runs/{name}/"
cfg = json.load(open(OUT + "train_log.json"))["args"]
model = ForensicUNet(noise=bool(cfg["noise"]), pretrained=False).cuda()
model.load_state_dict(torch.load(OUT + "best.pt", map_location="cuda")); model.eval()
recs = {r["uid"]: r for r in load_splits()}
os.makedirs(OUT + "maps", exist_ok=True)
for u in uids:
    for f in (0, 1):
        img, _ = load_image(recs[u], f)
        pm = predict_map(model, img)
        small = cv2.resize(pm, (pm.shape[1] // 2, pm.shape[0] // 2)) if max(pm.shape) > 1500 else pm
        np.save(OUT + f"maps/{u}_{f}.npy", small.astype(np.float16))
        print(u, f, float(pm.max()), flush=True)
