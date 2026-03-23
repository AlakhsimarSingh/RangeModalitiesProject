import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Set

import cv2
import numpy as np
from sklearn.decomposition import PCA


@dataclass
class FBPCAConfig:
    root: Path
    label_root: Path
    out_root: Path
    modalities: List[str]
    splits: List[str] = ("train", "valid", "test")
    img_exts: Set[str] = (".png", ".jpg", ".jpeg")
    fg_samples_per_image: int = 800
    max_pixels: int = 300_000
    bbox_dilate: int = 3
    random_seed: int = 42


def read_channel(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 3:
        img = img[:, :, 0]
    return img.astype(np.float32)


def load_stack(img_name: str, split: str, config: FBPCAConfig) -> np.ndarray:
    """Returns H x W x 4."""
    return np.stack(
        [read_channel(config.root / mod / split / img_name) for mod in config.modalities],
        axis=-1
    )


def parse_labels(label_path: Path, H: int, W: int, bbox_dilate: int = 3):
    """Parse YOLO format labels → list of (x1,y1,x2,y2) pixel boxes."""
    boxes = []
    if not label_path.exists():
        return boxes
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            _, cx, cy, bw, bh = map(float, parts[:5])
            x1 = max(0, int((cx - bw / 2) * W) - bbox_dilate)
            y1 = max(0, int((cy - bh / 2) * H) - bbox_dilate)
            x2 = min(W, int((cx + bw / 2) * W) + bbox_dilate)
            y2 = min(H, int((cy + bh / 2) * H) + bbox_dilate)
            if x2 > x1 and y2 > y1:
                boxes.append((x1, y1, x2, y2))
    return boxes


def fg_mask(boxes, H: int, W: int) -> np.ndarray:
    """Build binary foreground mask from bounding boxes."""
    mask = np.zeros((H, W), dtype=bool)
    for x1, y1, x2, y2 in boxes:
        mask[y1:y2, x1:x2] = True
    return mask


def per_channel_normalize(stack: np.ndarray) -> np.ndarray:
    """Normalize each channel to [0,1] using per-image min/max."""
    normed = np.zeros_like(stack)
    for c in range(stack.shape[-1]):
        ch = stack[:, :, c]
        mn, mx = ch.min(), ch.max()
        normed[:, :, c] = (ch - mn) / (mx - mn + 1e-6)
    return normed


def gather_fg_pixels(config: FBPCAConfig) -> np.ndarray:
    np.random.seed(config.random_seed)
    train_ref_dir = config.root / config.modalities[0] / "train"
    train_imgs = [p for p in train_ref_dir.iterdir() if p.suffix.lower() in config.img_exts]

    fg_pixels = []
    n_skipped = 0

    for img_path in train_imgs:
        stack = load_stack(img_path.name, "train", config)
        normed = per_channel_normalize(stack)
        H, W = stack.shape[:2]

        label_path = config.label_root / "train" / (img_path.stem + ".txt")
        boxes = parse_labels(label_path, H, W, config.bbox_dilate)

        if not boxes:
            n_skipped += 1
            continue

        mask = fg_mask(boxes, H, W)
        pixels = normed[mask]  # N_fg x 4

        if len(pixels) == 0:
            continue

        # Oversample foreground
        n = min(len(pixels), config.fg_samples_per_image)
        idx = np.random.choice(len(pixels), n, replace=len(pixels) < n)
        fg_pixels.append(pixels[idx])

    fg_pixels = np.concatenate(fg_pixels)
    if len(fg_pixels) > config.max_pixels:
        idx = np.random.choice(len(fg_pixels), config.max_pixels, replace=False)
        fg_pixels = fg_pixels[idx]

    return fg_pixels


def fit_fbpca(samples: np.ndarray, n_components: int = 3) -> PCA:
    pca = PCA(n_components=n_components)
    pca.fit(samples)
    return pca


def compute_global_norms(config: FBPCAConfig, pca_model: PCA) -> tuple[np.ndarray, np.ndarray]:
    train_ref_dir = config.root / config.modalities[0] / "train"
    train_imgs = [p for p in train_ref_dir.iterdir() if p.suffix.lower() in config.img_exts]

    global_lo = np.full(3, np.inf)
    global_hi = np.full(3, -np.inf)

    for img_path in train_imgs:
        stack = load_stack(img_path.name, "train", config)
        normed = per_channel_normalize(stack)
        proj = pca_model.transform(normed.reshape(-1, 4)).reshape(
            normed.shape[0], normed.shape[1], 3
        )
        for c in range(3):
            global_lo[c] = min(global_lo[c], np.percentile(proj[:, :, c], 1))
            global_hi[c] = max(global_hi[c], np.percentile(proj[:, :, c], 99))

    return global_lo, global_hi


def fuse_image(img_name: str, split: str, config: FBPCAConfig, pca_model: PCA, global_lo: np.ndarray, global_hi: np.ndarray) -> np.ndarray:
    """4-ch stack → 3-ch BGR uint8 via FB-PCA."""
    stack = load_stack(img_name, split, config)
    normed = per_channel_normalize(stack)
    proj = pca_model.transform(normed.reshape(-1, 4)).reshape(
        normed.shape[0], normed.shape[1], 3
    )

    def norm_ch(ch, lo, hi):
        return ((np.clip(ch, lo, hi) - lo) / (hi - lo + 1e-6) * 255).astype(np.uint8)

    r = norm_ch(proj[:, :, 0], global_lo[0], global_hi[0])
    g = norm_ch(proj[:, :, 1], global_lo[1], global_hi[1])
    b = norm_ch(proj[:, :, 2], global_lo[2], global_hi[2])
    return cv2.merge([b, g, r])  # OpenCV BGR


def create_fbpca_dataset(config: FBPCAConfig, pca_model: PCA, global_lo: np.ndarray, global_hi: np.ndarray) -> None:
    for split in config.splits:
        out_dir = config.out_root / split
        out_dir.mkdir(parents=True, exist_ok=True)

        ref_dir = config.root / config.modalities[0] / split
        split_imgs = [p for p in ref_dir.iterdir() if p.suffix.lower() in config.img_exts]

        for img_path in split_imgs:
            fused = fuse_image(img_path.name, split, config, pca_model, global_lo, global_hi)
            cv2.imwrite(str(out_dir / (img_path.stem + ".png")), fused)


def run_fbpca_pipeline(config: FBPCAConfig, n_components: int = 3) -> tuple[PCA, np.ndarray, np.ndarray]:
    fg_pixels = gather_fg_pixels(config)
    if fg_pixels.size == 0:
        raise ValueError("No foreground pixels found. Check ROOT and LABEL_ROOT paths")

    pca_model = fit_fbpca(fg_pixels, n_components=n_components)
    global_lo, global_hi = compute_global_norms(config, pca_model)
    create_fbpca_dataset(config, pca_model, global_lo, global_hi)
    return pca_model, global_lo, global_hi


def save_fbpca_params(pca_model: PCA, global_lo: np.ndarray, global_hi: np.ndarray, prefix: str = "fbpca") -> None:
    np.save(f"{prefix}_components.npy", pca_model.components_)
    np.save(f"{prefix}_mean.npy", pca_model.mean_)
    np.save(f"{prefix}_global_lo.npy", global_lo)
    np.save(f"{prefix}_global_hi.npy", global_hi)


if __name__ == "__main__":
    config = FBPCAConfig(
        root=Path("SnowPole Detection A Comprehensive Dataset for Detection and Localization Using LiDAR Imaging in Nordic Winter Conditions") / "SnowPole_Detection_Dataset",
        label_root=Path("SnowPole Detection A Comprehensive Dataset for Detection and Localization Using LiDAR Imaging in Nordic Winter Conditions") / "SnowPole_Detection_Dataset" / "labels",
        out_root=Path("dataset_fbpca/images"),
        modalities=["reflec", "signal", "nearir", "range"],
        splits=["train", "valid", "test"],
    )

    pca, glo, ghi = run_fbpca_pipeline(config, n_components=3)
    print("FB-PCA explained variance ratio:", pca.explained_variance_ratio_)
    save_fbpca_params(pca, glo, ghi)