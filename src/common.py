"""Shared data, model and inference code for the AIForge-Doc v2 field-forgery study."""
from paths import ROOT, DATA  # noqa: E402
import json, random, math, io
import numpy as np, cv2, torch, torch.nn as nn, torch.nn.functional as F
import torchvision

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


# ---------------------------------------------------------------- training data
class CropDataset(torch.utils.data.Dataset):
    """Random full-resolution crops. Authentic negatives are sampled at the *same* field
    location as the forgery (paired hard negatives) unless paired=False."""

    def __init__(self, recs, crop=512, n=10 ** 9, paired=True, p_field=0.7, aug=True, seed=0):
        self.recs, self.crop, self.n, self.paired, self.p_field, self.aug = recs, crop, n, paired, p_field, aug
        self.seed = seed

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rng = random.Random(self.seed * 1000003 + i)
        r = rng.choice(self.recs)
        forged = rng.random() < 0.5
        a, p, (x1, y1, x2, y2) = load_pair(r)
        H, W = a.shape[:2]
        c = self.crop
        # optional down-scaling augmentation: crop a larger window then resize to c
        s = 1.0
        if self.aug and rng.random() < 0.2:
            s = rng.uniform(0.5, 1.0)
        cw = min(W, int(round(c / s)))
        ch = min(H, int(round(c / s)))
        use_field = rng.random() < self.p_field and (forged or self.paired)
        if use_field:
            lo_x, hi_x = max(0, x2 + 1 - cw), min(x1, W - cw)
            lo_y, hi_y = max(0, y2 + 1 - ch), min(y1, H - ch)
            if lo_x > hi_x:  # field wider than crop: centre on field
                lo_x = hi_x = int(np.clip((x1 + x2) // 2 - cw // 2, 0, W - cw))
            if lo_y > hi_y:
                lo_y = hi_y = int(np.clip((y1 + y2) // 2 - ch // 2, 0, H - ch))
            cx, cy = rng.randint(lo_x, hi_x), rng.randint(lo_y, hi_y)
        else:
            cx, cy = rng.randint(0, W - cw), rng.randint(0, H - ch)
        # JPEG aug must respect the image's own 8x8 grid: expand crop to grid-aligned origin
        do_jpeg = self.aug and rng.random() < 0.5
        gx, gy = (cx - cx % 8, cy - cy % 8) if do_jpeg else (cx, cy)
        img = np.array(a[gy:cy + ch, gx:cx + cw])
        m = np.zeros(img.shape[:2], np.float32)
        if forged:
            ix1, iy1 = max(x1, gx), max(y1, gy)
            ix2, iy2 = min(x2 + 1, cx + cw), min(y2 + 1, cy + ch)
            if ix2 > ix1 and iy2 > iy1:
                img[iy1 - gy:iy2 - gy, ix1 - gx:ix2 - gx] = p[iy1 - y1:iy2 - y1, ix1 - x1:ix2 - x1]
                m[iy1 - gy:iy2 - gy, ix1 - gx:ix2 - gx] = 1
        if do_jpeg:
            img = jpeg(img, rng.randint(50, 95))
        img, m = img[cy - gy:, cx - gx:], m[cy - gy:, cx - gx:]
        if s != 1.0:
            nw, nh = max(1, round(img.shape[1] * s)), max(1, round(img.shape[0] * s))
            img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
            m = cv2.resize(m, (nw, nh), interpolation=cv2.INTER_NEAREST)
        if self.aug and rng.random() < 0.1:
            img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.3, 1.0))
        if self.aug and rng.random() < 0.1:
            img = np.clip(img + np.random.default_rng(i).normal(0, rng.uniform(1, 5), img.shape), 0, 255).astype(np.uint8)
        # pad to crop size (white page padding); valid mask marks real pixels
        out = np.full((c, c, 3), 255, np.uint8)
        om = np.zeros((c, c), np.float32)
        valid = np.zeros((c, c), np.float32)
        h, w = min(c, img.shape[0]), min(c, img.shape[1])
        out[:h, :w], om[:h, :w], valid[:h, :w] = img[:h, :w], m[:h, :w], 1
        lab = float(om.max() > 0)
        return (torch.from_numpy(out).permute(2, 0, 1), torch.from_numpy(om)[None], torch.from_numpy(valid)[None],
                torch.tensor(lab))


# ---------------------------------------------------------------- model
def srm_kernels():
    k1 = np.array([[0, 0, 0, 0, 0], [0, -1, 2, -1, 0], [0, 2, -4, 2, 0], [0, -1, 2, -1, 0], [0, 0, 0, 0, 0]], np.float32) / 4
    k2 = np.array([[-1, 2, -2, 2, -1], [2, -6, 8, -6, 2], [-2, 8, -12, 8, -2], [2, -6, 8, -6, 2], [-1, 2, -2, 2, -1]], np.float32) / 12
    k3 = np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, -2, 1, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], np.float32) / 2
    return torch.from_numpy(np.stack([k1, k2, k3])[:, None])


class ConvBNReLU(nn.Sequential):
    def __init__(self, i, o):
        super().__init__(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class Up(nn.Module):
    def __init__(self, i, skip, o):
        super().__init__()
        self.c = nn.Sequential(ConvBNReLU(i + skip, o), ConvBNReLU(o, o))

    def forward(self, x, s):
        x = F.interpolate(x, size=s.shape[-2:], mode="nearest")
        return self.c(torch.cat([x, s], 1))


class ForensicUNet(nn.Module):
    """U-Net with ResNet-34 encoder. Optional noise-residual (SRM) stream fused at the stem;
    a full-resolution skip keeps stroke-level detail for thin text fields."""

    def __init__(self, noise=True, pretrained=True):
        super().__init__()
        self.noise = noise
        enc = torchvision.models.resnet34(weights="IMAGENET1K_V1" if pretrained else None)
        cin = 6 if noise else 3
        conv1 = nn.Conv2d(cin, 64, 7, 2, 3, bias=False)
        with torch.no_grad():
            conv1.weight[:, :3] = enc.conv1.weight
            if noise:
                conv1.weight[:, 3:] = enc.conv1.weight.mean(1, keepdim=True).repeat(1, 3, 1, 1) * 0.5
        self.stem = nn.Sequential(conv1, enc.bn1, enc.relu)
        self.pool = enc.maxpool
        self.l1, self.l2, self.l3, self.l4 = enc.layer1, enc.layer2, enc.layer3, enc.layer4
        self.register_buffer("srm", srm_kernels())
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.full = nn.Sequential(ConvBNReLU(cin, 32), ConvBNReLU(32, 32))
        self.u4 = Up(512, 256, 256)
        self.u3 = Up(256, 128, 128)
        self.u2 = Up(128, 64, 64)
        self.u1 = Up(64, 64, 64)
        self.u0 = Up(64, 32, 32)
        self.head = nn.Conv2d(32, 1, 1)

    def prep(self, x):  # x uint8 B,3,H,W
        x = x.float() / 255.0
        rgb = (x - self.mean) / self.std
        if not self.noise:
            return rgb
        g = (0.299 * x[:, :1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]) * 255
        n = torch.clamp(F.conv2d(g, self.srm, padding=2), -3, 3) / 3
        return torch.cat([rgb, n], 1)

    def forward(self, x):
        x = self.prep(x)
        f0 = self.full(x)
        s1 = self.stem(x)
        e1 = self.l1(self.pool(s1))
        e2 = self.l2(e1)
        e3 = self.l3(e2)
        e4 = self.l4(e3)
        d = self.u4(e4, e3)
        d = self.u3(d, e2)
        d = self.u2(d, e1)
        d = self.u1(d, s1)
        d = self.u0(d, f0)
        return self.head(d)


def topk_pool(logits, valid, k=256):
    """Image-level logit = mean of the k highest valid pixel logits."""
    B = logits.shape[0]
    l = logits.float().masked_fill(valid < 0.5, -1e4).view(B, -1)
    return l.topk(k, dim=1).values.mean(1)


# ---------------------------------------------------------------- inference
@torch.no_grad()
def predict_map(model, img, tile=1024, overlap=128, dev="cuda"):
    """Fully-convolutional tiled inference at native resolution. Returns HxW probability map."""
    H, W = img.shape[:2]
    out = np.zeros((H, W), np.float32)
    cnt = np.zeros((H, W), np.float32)
    step = tile - overlap
    ys = [0] if H <= tile else list(range(0, H - tile, step)) + [H - tile]
    xs = [0] if W <= tile else list(range(0, W - tile, step)) + [W - tile]
    for y in ys:
        for x in xs:
            t = img[y:y + tile, x:x + tile]
            h, w = t.shape[:2]
            ph, pw = (32 - h % 32) % 32, (32 - w % 32) % 32
            t = np.pad(t, ((0, ph), (0, pw), (0, 0)), constant_values=255)
            xt = torch.from_numpy(np.ascontiguousarray(t)).permute(2, 0, 1)[None].to(dev)
            with torch.autocast("cuda", dtype=torch.float16):
                p = torch.sigmoid(model(xt).float())[0, 0, :h, :w].cpu().numpy()
            out[y:y + h, x:x + w] += p
            cnt[y:y + h, x:x + w] += 1
    return out / cnt


def image_scores(pm):
    """Aggregate a probability map into image-level scores (several rules, for the ablation)."""
    sm = cv2.blur(pm, (9, 9))
    flat = pm.ravel()
    k = min(256, flat.size)
    return {"max": float(pm.max()), "smooth_max": float(sm.max()),
            "topk": float(np.partition(flat, -k)[-k:].mean()), "mean": float(pm.mean())}
