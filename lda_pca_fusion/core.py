import cv2
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import List, Set

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from tqdm import tqdm


@dataclass
class LDAPCAConfig:
    root: Path
    label_root: Path
    out_root: Path
    modalities: List[str]
    splits: List[str] = ("train", "valid", "test")
    img_exts: Set[str] = (".png", ".jpg", ".jpeg")
    fg_samples_per_image: int = 500
    bg_samples_per_image: int = 200
    max_pixels: int = 300_000
    random_seed: int = 42


def read_first_channel(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {path}")
    if img.ndim == 3:
        img = img[:, :, 0]
    return img.astype(np.float32)


def load_stack(img_name: str, split: str, config: LDAPCAConfig) -> np.ndarray:
    """Returns H x W x 4 stack of all modalities."""
    return np.stack(
        [read_first_channel(config.root / mod / split / img_name) for mod in config.modalities],
        axis=-1
    )


def parse_yolo_labels(label_path: Path, H: int, W: int):
    boxes = []
    if not label_path.exists():
        return boxes
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            _, cx, cy, bw, bh = map(float, parts[:5])
            x1 = max(0, int((cx - bw / 2) * W))
            y1 = max(0, int((cy - bh / 2) * H))
            x2 = min(W, int((cx + bw / 2) * W))
            y2 = min(H, int((cy + bh / 2) * H))
            if x2 > x1 and y2 > y1:
                boxes.append((x1, y1, x2, y2))
    return boxes


def remove_lda_component(X: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Remove LDA direction from X via Gram-Schmidt."""
    return X - (X @ direction)[:, None] * direction[None, :]


def gather_fg_bg_pixels(config: LDAPCAConfig) -> tuple[np.ndarray, np.ndarray]:
    np.random.seed(config.random_seed)
    train_ref_dir = config.root / config.modalities[0] / "train"
    train_images = [p for p in train_ref_dir.iterdir() if p.suffix.lower() in config.img_exts]

    fg_pixels, bg_pixels = [], []
    images_with_labels = 0

    for img_path in tqdm(train_images, desc="Sampling"):
        stack = load_stack(img_path.name, "train", config)
        H, W, _ = stack.shape
        flat = stack.reshape(-1, 4)

        label_path = config.label_root / "train" / (img_path.stem + ".txt")
        boxes = parse_yolo_labels(label_path, H, W)

        if boxes:
            images_with_labels += 1
            fg_mask = np.zeros((H, W), dtype=bool)
            for x1, y1, x2, y2 in boxes:
                fg_mask[y1:y2, x1:x2] = True

            fg_flat = flat[fg_mask.reshape(-1)]
            bg_flat = flat[~fg_mask.reshape(-1)]

            if len(fg_flat) > 0:
                n = min(len(fg_flat), config.fg_samples_per_image)
                fg_pixels.append(fg_flat[np.random.choice(len(fg_flat), n, replace=False)])

            n = min(len(bg_flat), config.bg_samples_per_image)
            bg_pixels.append(bg_flat[np.random.choice(len(bg_flat), n, replace=False)])
        else:
            # No label: treat all pixels as background
            n = min(len(flat), config.bg_samples_per_image)
            bg_pixels.append(flat[np.random.choice(len(flat), n, replace=False)])

    fg_pixels = np.concatenate(fg_pixels) if fg_pixels else np.array([])
    bg_pixels = np.concatenate(bg_pixels)

    # Cap to MAX_PIXELS
    if len(fg_pixels) > config.max_pixels // 2:
        fg_pixels = fg_pixels[np.random.choice(len(fg_pixels), config.max_pixels // 2, replace=False)]
    if len(bg_pixels) > config.max_pixels // 2:
        bg_pixels = bg_pixels[np.random.choice(len(bg_pixels), config.max_pixels // 2, replace=False)]

    return fg_pixels, bg_pixels


def fit_lda(X_fg: np.ndarray, X_bg: np.ndarray) -> np.ndarray:
    X_all = np.concatenate([X_fg, X_bg])
    y_all = np.array([1] * len(X_fg) + [0] * len(X_bg), dtype=np.int32)

    lda = LinearDiscriminantAnalysis(n_components=1)
    lda.fit(X_all, y_all)

    lda_dir = lda.scalings_[:, 0]
    lda_dir /= (np.linalg.norm(lda_dir) + 1e-10)
    return lda_dir


def fit_pca_residual(X_all: np.ndarray, lda_dir: np.ndarray, n_components: int = 2) -> PCA:
    X_res = remove_lda_component(X_all, lda_dir)
    pca = PCA(n_components=n_components)
    pca.fit(X_res)
    return pca


def compute_global_norms_lda_pca(config: LDAPCAConfig, lda_dir: np.ndarray, pca_model: PCA) -> tuple[np.ndarray, np.ndarray]:
    train_ref_dir = config.root / config.modalities[0] / "train"
    train_images = [p for p in train_ref_dir.iterdir() if p.suffix.lower() in config.img_exts]

    global_lo = np.full(3, np.inf)
    global_hi = np.full(3, -np.inf)

    for img_path in tqdm(train_images, desc="Stats"):
        stack = load_stack(img_path.name, "train", config)
        H, W, _ = stack.shape
        flat = stack.reshape(-1, 4)

        ch0 = (flat @ lda_dir).reshape(H, W)
        res = remove_lda_component(flat, lda_dir)
        pca_out = pca_model.transform(res)
        ch1 = pca_out[:, 0].reshape(H, W)
        ch2 = pca_out[:, 1].reshape(H, W)

        for i, ch in enumerate([ch0, ch1, ch2]):
            global_lo[i] = min(global_lo[i], np.percentile(ch, 1))
            global_hi[i] = max(global_hi[i], np.percentile(ch, 99))

    return global_lo, global_hi


def fuse_lda_pca(stack: np.ndarray, lda_dir: np.ndarray, pca_model: PCA, global_lo: np.ndarray, global_hi: np.ndarray) -> np.ndarray:
    """
    H x W x 4  →  H x W x 3  uint8
      R = LDA channel          (pole vs background discriminant)
      G = PCA residual comp 1  (next most informative direction)
      B = PCA residual comp 2
    """
    H, W, _ = stack.shape
    flat = stack.reshape(-1, 4)

    ch0 = (flat @ lda_dir).reshape(H, W)
    res = remove_lda_component(flat, lda_dir)
    pca_out = pca_model.transform(res)
    ch1 = pca_out[:, 0].reshape(H, W)
    ch2 = pca_out[:, 1].reshape(H, W)

    def norm(ch, lo, hi):
        return ((np.clip(ch, lo, hi) - lo) / (hi - lo + 1e-6) * 255).astype(np.uint8)

    r = norm(ch0, global_lo[0], global_hi[0])
    g = norm(ch1, global_lo[1], global_hi[1])
    b = norm(ch2, global_lo[2], global_hi[2])

    return cv2.merge([b, g, r])  # OpenCV uses BGR internally


def create_lda_pca_dataset(config: LDAPCAConfig, lda_dir: np.ndarray, pca_model: PCA, global_lo: np.ndarray, global_hi: np.ndarray) -> None:
    for split in config.splits:
        out_dir = config.out_root / split
        out_dir.mkdir(parents=True, exist_ok=True)

        ref_dir = config.root / config.modalities[0] / split
        split_images = [p for p in ref_dir.iterdir() if p.suffix.lower() in config.img_exts]

        for img_path in tqdm(split_images, desc=split):
            stack = load_stack(img_path.name, split, config)
            fused = fuse_lda_pca(stack, lda_dir, pca_model, global_lo, global_hi)
            cv2.imwrite(str(out_dir / img_path.name), fused)


def run_lda_pca_pipeline(config: LDAPCAConfig) -> tuple[np.ndarray, PCA, np.ndarray, np.ndarray]:
    fg_pixels, bg_pixels = gather_fg_bg_pixels(config)
    if len(fg_pixels) == 0 or len(bg_pixels) == 0:
        raise ValueError("No fg/bg pixels found. Check ROOT and LABEL_ROOT paths")

    lda_dir = fit_lda(fg_pixels, bg_pixels)
    X_all = np.concatenate([fg_pixels, bg_pixels])
    pca_model = fit_pca_residual(X_all, lda_dir, n_components=2)
    global_lo, global_hi = compute_global_norms_lda_pca(config, lda_dir, pca_model)
    create_lda_pca_dataset(config, lda_dir, pca_model, global_lo, global_hi)
    return lda_dir, pca_model, global_lo, global_hi


def save_lda_pca_params(lda_dir: np.ndarray, pca_model: PCA, global_lo: np.ndarray, global_hi: np.ndarray, prefix: str = "lda_pca") -> None:
    np.save(f"{prefix}_lda_dir.npy", lda_dir)
    np.save(f"{prefix}_pca_components.npy", pca_model.components_)
    np.save(f"{prefix}_pca_mean.npy", pca_model.mean_)
    np.save(f"{prefix}_global_lo.npy", global_lo)
    np.save(f"{prefix}_global_hi.npy", global_hi)


def train_yolo_lda_pca(data_yaml: str, model_yaml: str, project: str, name: str, **kwargs) -> None:
    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise ImportError("ultralytics package is required for YOLO training") from e

    model = YOLO(model_yaml)
    model.train(data=data_yaml, project=project, name=name, **kwargs)


if __name__ == "__main__":
    config = LDAPCAConfig(
        root=Path("SnowPole Detection A Comprehensive Dataset for Detection and Localization Using LiDAR Imaging in Nordic Winter Conditions") / "SnowPole_Detection_Dataset",
        label_root=Path("SnowPole Detection A Comprehensive Dataset for Detection and Localization Using LiDAR Imaging in Nordic Winter Conditions") / "SnowPole_Detection_Dataset" / "labels",
        out_root=Path("dataset_lda_pca/images"),
        modalities=["reflec", "signal", "nearir", "range"],
        splits=["train", "valid", "test"],
    )

    lda_dir, pca, glo, ghi = run_lda_pca_pipeline(config)
    print("LDA direction:", lda_dir)
    print("PCA explained variance:", pca.explained_variance_ratio_)
    save_lda_pca_params(lda_dir, pca, glo, ghi)

    # Example training
    # train_yolo_lda_pca(
    #     data_yaml="weighted_fusion.yaml",
    #     model_yaml="yolo11n.yaml",
    #     project="weighted_3pca",
    #     name="weighted_yolo11n_pca3",
    #     imgsz=1280,
    #     epochs=500,
    #     patience=80,
    #     batch=8,
    #     device=0,
    #     amp=False,
    #     augment=False,
    #     workers=0,
    # )