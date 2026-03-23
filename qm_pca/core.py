import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from sklearn.decomposition import PCA


@dataclass
class QMPCAConfig:
    root: Path
    out_root: Path
    modalities: List[str]
    splits: List[str] = ("train", "valid", "test")
    img_exts: List[str] = (".png", ".jpg", ".jpeg")
    max_pixels: int = 200_000
    sample_per_image: int = 1000
    random_seed: int = 42


def read_first_channel(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 3:
        img = img[:, :, 0]
    return img.astype(np.float32)


def gather_pca_training_samples(config: QMPCAConfig) -> np.ndarray:
    np.random.seed(config.random_seed)
    train_dir = config.root / config.modalities[0] / "train"
    image_files = [p for p in train_dir.iterdir() if p.suffix.lower() in config.img_exts]

    samples = []
    for img_path in image_files:
        stacked = []
        for mod in config.modalities:
            mod_path = config.root / mod / "train" / img_path.name
            img = read_first_channel(mod_path)
            stacked.append(img)

        stacked = np.stack(stacked, axis=-1)
        pixels = stacked.reshape(-1, len(config.modalities))

        if len(pixels) > config.sample_per_image:
            idx = np.random.choice(len(pixels), config.sample_per_image, replace=False)
            pixels = pixels[idx]

        samples.append(pixels)

    samples = np.concatenate(samples, axis=0)
    if len(samples) > config.max_pixels:
        idx = np.random.choice(len(samples), config.max_pixels, replace=False)
        samples = samples[idx]

    return samples


def fit_pca(samples: np.ndarray, n_components: int = 3) -> PCA:
    pca = PCA(n_components=n_components)
    pca.fit(samples)
    return pca


def project_pca(img_stack: np.ndarray, pca_model: PCA) -> np.ndarray:
    h, w, c = img_stack.shape
    flat = img_stack.reshape(-1, c)
    projected = pca_model.transform(flat)
    return projected.reshape(h, w, pca_model.n_components_)


def normalize_channels(img: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    out = img.astype(np.float32).copy()
    for c in range(out.shape[-1]):
        channel = out[:, :, c]
        mn = channel.min()
        mx = channel.max()
        if mx - mn > 0:
            out[:, :, c] = (channel - mn) / (mx - mn)
        else:
            out[:, :, c] = 0.0
    return out


def create_pca_dataset(config: QMPCAConfig, pca_model: PCA, normalize: bool = True) -> None:
    for split in config.splits:
        out_dir = config.out_root / split
        out_dir.mkdir(parents=True, exist_ok=True)

        ref_dir = config.root / config.modalities[0] / split
        image_files = [p for p in ref_dir.iterdir() if p.suffix.lower() in config.img_exts]

        for img_path in image_files:
            stacked = []
            for mod in config.modalities:
                mod_path = config.root / mod / split / img_path.name
                img = read_first_channel(mod_path)
                stacked.append(img)

            stacked_arr = np.stack(stacked, axis=-1)
            projected = project_pca(stacked_arr, pca_model)

            if normalize:
                projected = normalize_channels(projected)

            output_img = (projected * 255).astype(np.uint8)
            cv2.imwrite(str(out_dir / img_path.name), output_img)


def run_pca_pipeline(config: QMPCAConfig, n_components: int = 3) -> PCA:
    samples = gather_pca_training_samples(config)
    if samples.size == 0:
        raise ValueError("No PCA samples found. Check ROOT path and modalities")

    pca_model = fit_pca(samples, n_components=n_components)
    create_pca_dataset(config, pca_model, normalize=True)
    return pca_model


def train_yolo(data_yaml: str, model_yaml: str, project: str, name: str, **kwargs) -> None:
    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise ImportError("ultralytics package is required for YOLO training") from e

    model = YOLO(model_yaml)
    model.train(data=data_yaml, project=project, name=name, **kwargs)


if __name__ == "__main__":
    config = QMPCAConfig(
        root=Path("SnowPole Detection A Comprehensive Dataset for Detection and Localization Using LiDAR Imaging in Nordic Winter Conditions") / "SnowPole_Detection_Dataset",
        out_root=Path("dataset_pca3/images"),
        modalities=["reflec", "signal", "nearir", "range"],
        splits=["train", "valid", "test"],
    )

    pca = run_pca_pipeline(config, n_components=3)
    print("PCA explained variance ratio:", pca.explained_variance_ratio_)

    # Example YOLO training (uncomment to run)
    # train_yolo(
    #     data_yaml="weighted_fusion.yaml",
    #     model_yaml="yolo11n.yaml",
    #     project="weighted_3pca",
    #     name="weighted_yolo11n_pca3",
    #     imgsz=1024,
    #     epochs=500,
    #     patience=80,
    #     batch=8,
    #     device=0,
    #     amp=False,
    #     augment=False,
    #     workers=0,
    # )
