from pathlib import Path
from . import FBPCAConfig, run_fbpca_pipeline, save_fbpca_params


def run_default():
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


if __name__ == "__main__":
    run_default()