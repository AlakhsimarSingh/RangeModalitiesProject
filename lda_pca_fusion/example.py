from pathlib import Path
from . import LDAPCAConfig, run_lda_pca_pipeline, save_lda_pca_params


def run_default():
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


if __name__ == "__main__":
    run_default()