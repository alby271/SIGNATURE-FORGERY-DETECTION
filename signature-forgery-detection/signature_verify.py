"""Signature forgery detection: three pair-wise verifiers (reference vs questioned).

Dataset layout (any real dataset can be arranged/symlinked into this):
    <root>/<writer>/genuine/*.png|jpg
    <root>/<writer>/forged/*.png|jpg
Run:  python signature_verify.py --demo            (synthetic data smoke test)
      python signature_verify.py --data <root>     (real data, writer-disjoint split)
"""
import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

H, W = 96, 192
IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


# ---------- preprocessing (shared) ----------
def preprocess(img):
    """Gray -> Otsu (ink=255) -> crop to ink bbox -> fit into HxW canvas keeping aspect."""
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    _, bw = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ys, xs = np.nonzero(bw)
    if len(xs) == 0:
        return np.zeros((H, W), np.uint8)
    bw = bw[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    s = min((W - 4) / bw.shape[1], (H - 4) / bw.shape[0])
    bw = cv2.resize(bw, (max(1, int(bw.shape[1] * s)), max(1, int(bw.shape[0] * s))),
                    interpolation=cv2.INTER_AREA)
    out = np.zeros((H, W), np.uint8)
    y0, x0 = (H - bw.shape[0]) // 2, (W - bw.shape[1]) // 2
    out[y0:y0 + bw.shape[0], x0:x0 + bw.shape[1]] = bw
    return out


# ---------- solution 1: contour + geometric features ----------
def geometric_features(bw):
    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = [c for c in cnts if cv2.contourArea(c) > 2] or cnts
    ys, xs = np.nonzero(bw)
    pts = np.vstack(cnts) if cnts else np.zeros((1, 1, 2), np.int32)
    x, y, w, h = cv2.boundingRect(pts)
    hull_area = max(cv2.contourArea(cv2.convexHull(pts)), 1)
    ink = len(xs) / (H * W)
    hu = cv2.HuMoments(cv2.moments(bw, binaryImage=True)).ravel()
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-12)
    col, row = bw.sum(0) / 255.0, bw.sum(1) / 255.0
    return np.concatenate([
        [w / max(h, 1), ink, len(cnts), sum(cv2.arcLength(c, True) for c in cnts) / (H + W),
         len(xs) / hull_area, xs.mean() / W if len(xs) else 0, ys.mean() / H if len(xs) else 0],
        hu,
        cv2.resize(col[None], (8, 1), interpolation=cv2.INTER_AREA).ravel() / H,  # horizontal profile
        cv2.resize(row[None], (4, 1), interpolation=cv2.INTER_AREA).ravel() / W,  # vertical profile
    ])


class GeometricVerifier:
    """Score = -distance between standardized feature vectors."""
    def fit(self, pairs):
        feats = np.array([geometric_features(a) for a, _, _ in pairs] +
                         [geometric_features(b) for _, b, _ in pairs])
        self.mu, self.sd = feats.mean(0), feats.std(0) + 1e-6
        return self

    def _dist(self, a, b):
        fa = (geometric_features(a) - self.mu) / self.sd
        fb = (geometric_features(b) - self.mu) / self.sd
        return np.linalg.norm(fa - fb) / np.sqrt(len(fa))

    def score(self, pairs):
        return -np.array([self._dist(a, b) for a, b, _ in pairs])


# ---------- solution 2: HOG + SVM on reference/questioned difference ----------
_hog = cv2.HOGDescriptor((W, H), (32, 32), (16, 16), (16, 16), 9)


def hog(bw):
    return _hog.compute(bw).ravel()


class HogSvmVerifier:
    """RBF-SVM on |HOG(ref)-HOG(q)|: writer-independent, so it generalises to unseen writers."""
    def _x(self, pairs):
        return np.array([np.abs(hog(a) - hog(b)) for a, b, _ in pairs])

    def fit(self, pairs):
        self.sc = StandardScaler().fit(self._x(pairs))
        self.svm = SVC(kernel="rbf", C=10, class_weight="balanced").fit(
            self.sc.transform(self._x(pairs)), [l for *_, l in pairs])
        return self

    def score(self, pairs):
        return self.svm.decision_function(self.sc.transform(self._x(pairs)))


# ---------- solution 3: Siamese CNN ----------
class SiameseNet(nn.Module):
    def __init__(self):
        super().__init__()
        blk = lambda i, o: nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))
        self.enc = nn.Sequential(blk(1, 16), blk(16, 32), blk(32, 64), blk(64, 96),
                                 nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.head = nn.Linear(96, 1)

    def forward(self, a, b):
        return self.head(torch.abs(self.enc(a) - self.enc(b))).squeeze(1)


def _tensor(imgs):
    return torch.from_numpy(np.stack(imgs)).float().div(255).unsqueeze(1)


class CnnVerifier:
    def __init__(self, epochs=25, seed=0):
        self.epochs, self.seed = epochs, seed

    def fit(self, pairs):
        torch.manual_seed(self.seed)
        self.net = SiameseNet()
        A, B = _tensor([a for a, _, _ in pairs]), _tensor([b for _, b, _ in pairs])
        y = torch.tensor([float(l) for *_, l in pairs])
        opt = torch.optim.Adam(self.net.parameters(), 1e-3)
        lossf = nn.BCEWithLogitsLoss(pos_weight=(len(y) - y.sum()) / y.sum().clamp(min=1))
        for _ in range(self.epochs):
            self.net.train()
            for idx in torch.randperm(len(y)).split(32):
                # augment: small random shift keeps natural-variation tolerance
                sh = lambda t: torch.roll(t, (random.randint(-2, 2), random.randint(-4, 4)), (2, 3))
                opt.zero_grad()
                lossf(self.net(sh(A[idx]), sh(B[idx])), y[idx]).backward()
                opt.step()
        return self

    @torch.no_grad()
    def score(self, pairs):
        self.net.eval()
        return self.net(_tensor([a for a, _, _ in pairs]), _tensor([b for _, b, _ in pairs])).numpy()


# ---------- data ----------
def load_dataset(root):
    """{writer: {'genuine': [bw...], 'forged': [bw...]}}"""
    data = {}
    for wdir in sorted(p for p in Path(root).iterdir() if p.is_dir()):
        d = {}
        for kind, flat in (("genuine", "original_*"), ("forged", "forgeries_*")):
            # either <writer>/<kind>/* or Kaggle CEDAR style <writer>/original_*, <writer>/forgeries_*
            files = [f for f in sorted(list((wdir / kind).glob("*")) + list(wdir.glob(flat)))
                     if f.suffix.lower() in IMG_EXT]
            d[kind] = [preprocess(im) for im in (cv2.imread(str(f)) for f in files) if im is not None]
        if len(d["genuine"]) >= 2 and d["forged"]:
            data[wdir.name] = d
    return data


def synthetic_dataset(n_writers=40, n_gen=12, n_forg=8, seed=0):
    """Splines per writer; genuine = small jitter, skilled forgery = traced with larger deviation."""
    rng = np.random.default_rng(seed)

    def draw(ctrl):
        t = np.linspace(0, 1, len(ctrl))
        tt = np.linspace(0, 1, 400)
        xy = np.stack([np.interp(tt, t, ctrl[:, 0]), np.interp(tt, t, ctrl[:, 1])], 1)
        xy = np.stack([np.convolve(np.pad(xy[:, k], 12, mode="edge"), np.ones(25) / 25, "valid")[:400]
                       for k in (0, 1)], 1)
        img = np.full((160, 320), 255, np.uint8)
        cv2.polylines(img, [(xy * [300, 140] + [10, 10]).astype(np.int32)], False, 0, int(rng.integers(2, 4)))
        return preprocess(img)

    data = {}
    for w in range(n_writers):
        base = np.column_stack([np.linspace(0, 1, 9), rng.random(9)])
        data[f"w{w}"] = {
            "genuine": [draw(base + rng.normal(0, 0.012, base.shape) * [0.5, 1]) for _ in range(n_gen)],
            "forged": [draw(base + rng.normal(0, 0.05, base.shape) * [0.5, 1]) for _ in range(n_forg)],
        }
    return data


def make_pairs(data, rng, n_refs=1):
    """(reference genuine, questioned, label): 1 genuine, 0 forged. n_refs references per questioned sample."""
    pairs = []
    for d in data.values():
        g, f = d["genuine"], d["forged"]
        for i, q in enumerate(g):
            for j in rng.choice([j for j in range(len(g)) if j != i], min(n_refs, len(g) - 1), replace=False):
                pairs.append((g[j], q, 1))
        for q in f:
            for j in rng.choice(len(g), min(n_refs, len(g)), replace=False):
                pairs.append((g[j], q, 0))
    return pairs


def eer_threshold(score, y):
    """Threshold where FAR ~= FRR (accept if score >= t)."""
    cands = np.unique(score)
    gap = [abs((score[~y] >= t).mean() - (score[y] < t).mean()) for t in cands]
    return cands[int(np.argmin(gap))]


def report(name, model, val, test):
    yv = np.array([l for *_, l in val], bool)
    t = eer_threshold(model.score(val), yv)
    y = np.array([l for *_, l in test], bool)
    sc = model.score(test)
    p = sc >= t
    far = (p & ~y).sum() / max((~y).sum(), 1)   # forgeries accepted
    frr = (~p & y).sum() / max(y.sum(), 1)      # genuine rejected
    print(f"{name:<22} acc={(p == y).mean():.3f}  FAR={far:.3f}  FRR={frr:.3f}")
    return dict(name=name, score=sc, y=y, acc=(p == y).mean(), far=far, frr=frr)


def plot_results(results, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5, 5))  # ROC: FAR vs genuine-accept rate over all thresholds
    for r in results:
        ts = np.unique(r["score"])
        far = [(r["score"][~r["y"]] >= t).mean() for t in ts]
        gar = [(r["score"][r["y"]] >= t).mean() for t in ts]
        ax.plot(far, gar, label=r["name"])
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="chance")
    ax.set(xlabel="FAR (forgeries accepted)", ylabel="Genuine accepted (1 - FRR)", title="ROC on unseen writers")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "roc.png", dpi=150)

    fig, ax = plt.subplots(figsize=(7, 4))  # accuracy / FAR / FRR at the validation-tuned threshold
    x = np.arange(len(results))
    for i, k in enumerate(("acc", "far", "frr")):
        ax.bar(x + (i - 1) * 0.27, [r[k] for r in results], 0.27, label=k.upper())
    ax.set_xticks(x, [r["name"] for r in results])
    ax.set(ylim=(0, 1), title="Accuracy, FAR, FRR on unseen writers")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "metrics.png", dpi=150)

    fig, axs = plt.subplots(1, len(results), figsize=(4 * len(results), 3.2))  # score separation
    for ax, r in zip(np.atleast_1d(axs), results):
        ax.hist(r["score"][r["y"]], bins=30, alpha=0.6, label="genuine")
        ax.hist(r["score"][~r["y"]], bins=30, alpha=0.6, label="forged")
        ax.set(title=r["name"], xlabel="score (higher = more genuine)")
        ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "scores.png", dpi=150)
    print(f"plots saved to {outdir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--plots", metavar="DIR", help="save ROC, metric and score-distribution graphs here")
    args = ap.parse_args()
    if not (args.data or args.demo):
        ap.error("pass --data <root> or --demo")
    random.seed(0)
    rng = np.random.default_rng(0)
    data = synthetic_dataset() if args.demo else load_dataset(args.data)
    writers = list(data)
    rng.shuffle(writers)
    # writer-disjoint 60/15/25: threshold is tuned on val writers, scored on unseen test writers
    a, b = int(0.6 * len(writers)), int(0.75 * len(writers))
    train, val, test = (make_pairs({w: data[w] for w in ws}, rng, n)
                        for ws, n in ((writers[:a], 4), (writers[a:b], 2), (writers[b:], 2)))
    print(f"writers {a}/{b - a}/{len(writers) - b}, pairs {len(train)}/{len(val)}/{len(test)}")
    results = [report(name, m.fit(train), val, test)
               for name, m in [("1 Geometric", GeometricVerifier()), ("2 HOG+SVM", HogSvmVerifier()),
                               ("3 Siamese CNN", CnnVerifier(args.epochs))]]
    if args.plots:
        plot_results(results, args.plots)


if __name__ == "__main__":
    main()
