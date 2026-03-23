from pathlib import Path
from . import QMPCAConfig, run_pca_pipeline, train_yolo


def run_default():
    config = QMPCAConfig(
        root=Path("SnowPole Detection A Comprehensive Dataset for Detection and Localization Using LiDAR Imaging in Nordic Winter Conditions") / "SnowPole_Detection_Dataset",
        out_root=Path("dataset_pca3/images"),
        modalities=["reflec", "signal", "nearir", "range"],
        splits=["train", "valid", "test"],
    )
    pca = run_pca_pipeline(config, n_components=3)
    print("PCA explained variance ratio:", pca.explained_variance_ratio_)


if __name__ == "__main__":
    run_default()
