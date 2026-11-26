"""Torch-free data access + degradations (used by CPU baselines)."""
from paths import ROOT, DATA  # noqa: E402
import json, numpy as np, cv2
CACHE = ROOT + "cache/"
SOURCES = ["cord", "wildreceipt", "sroie", "xfund"]


def load_splits():
    return json.load(open(CACHE + "splits.json"))


def load_pair(r, mmap=True):
    """Return authentic (H,W,3) uint8 array (memmap) and forged patch + bbox."""
    a = np.load(CACHE + f"img/{r['uid']}_auth.npy", mmap_mode="r" if mmap else None)
    p = np.load(CACHE + f"img/{r['uid']}_patch.npy")
    return a, p, r["mbbox"]


def load_image(r, forged):
    a, p, (x1, y1, x2, y2) = load_pair(r, mmap=False)
    img = np.array(a)
    m = np.zeros(img.shape[:2], np.uint8)
    if forged:
        img[y1:y2 + 1, x1:x2 + 1] = p
        m[y1:y2 + 1, x1:x2 + 1] = 1
    return img, m


# ---------------------------------------------------------------- degradations
def jpeg(img, q):
    ok, enc = cv2.imencode(".jpg", img[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, int(q)])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)[:, :, ::-1].copy()


def degrade(img, mask, kind):
    """Test-time perturbation applied to the full image (both classes alike)."""
    if kind == "none":
        return img, mask
    if kind.startswith("jpeg"):
        return jpeg(img, int(kind[4:])), mask
    if kind.startswith("scale"):
        s = float(kind[5:])
        h, w = img.shape[:2]
        nh, nw = max(32, round(h * s)), max(32, round(w * s))
        return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA), cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
    if kind.startswith("blur"):
        s = float(kind[4:])
        return cv2.GaussianBlur(img, (0, 0), s), mask
    if kind.startswith("noise"):
        s = float(kind[5:])
        rng = np.random.default_rng(0)
        return np.clip(img.astype(np.float32) + rng.normal(0, s, img.shape), 0, 255).astype(np.uint8), mask
    if kind == "social":  # downscale 0.75 + JPEG 80: a messaging-app style re-share
        i, m = degrade(img, mask, "scale0.75")
        return jpeg(i, 80), m
    raise ValueError(kind)
