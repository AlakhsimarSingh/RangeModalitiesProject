import argparse
import sys
from pathlib import Path

from .core import LDAPCAConfig, run_lda_pca_pipeline, save_lda_pca_params, train_yolo_lda_pca


def add_lda_pca_run_subparser(subparsers):
    parser = subparsers.add_parser("lda-pca-run", help="Run LDA+PCA fusion pipeline")
    parser.add_argument("--root", type=str, required=True, help="Root dataset path")
    parser.add_argument("--label-root", type=str, required=True, help="Label root path")
    parser.add_argument("--out-root", type=str, required=True, help="Output root path")
    parser.add_argument("--modalities", nargs="+", default=["reflec", "signal", "nearir", "range"], help="List of modalities")
    parser.add_argument("--splits", nargs="+", default=["train", "valid", "test"], help="Dataset splits")
    parser.add_argument("--fg-samples-per-image", type=int, default=500, help="FG samples per image")
    parser.add_argument("--bg-samples-per-image", type=int, default=200, help="BG samples per image")
    parser.add_argument("--max-pixels", type=int, default=300000, help="Max pixels for fitting")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed")


def add_train_subparser(subparsers):
    parser = subparsers.add_parser("train", help="Train YOLO model")
    parser.add_argument("--data-yaml", type=str, required=True, help="Data YAML file")
    parser.add_argument("--model-yaml", type=str, required=True, help="Model YAML file")
    parser.add_argument("--project", type=str, required=True, help="Project name")
    parser.add_argument("--name", type=str, required=True, help="Run name")
    parser.add_argument("--imgsz", type=int, default=1280, help="Image size")
    parser.add_argument("--epochs", type=int, default=500, help="Number of epochs")
    parser.add_argument("--patience", type=int, default=80, help="Early stopping patience")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--device", type=str, default="0", help="Device (e.g., 0 for GPU)")
    parser.add_argument("--amp", action="store_true", help="Use AMP")
    parser.add_argument("--augment", action="store_true", help="Use augmentation")
    parser.add_argument("--workers", type=int, default=0, help="Number of workers")


def add_export_subparser(subparsers):
    parser = subparsers.add_parser("export", help="Export LDA+PCA components")
    parser.add_argument("--lda-dir-path", type=str, required=True, help="Path to saved LDA direction")
    parser.add_argument("--pca-components-path", type=str, required=True, help="Path to saved PCA components")
    parser.add_argument("--pca-mean-path", type=str, required=True, help="Path to saved PCA mean")
    parser.add_argument("--glo-path", type=str, required=True, help="Path to global_lo")
    parser.add_argument("--ghi-path", type=str, required=True, help="Path to global_hi")
    parser.add_argument("--output", type=str, required=True, help="Output file path")


def main():
    parser = argparse.ArgumentParser(description="LDA+PCA Fusion CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    add_lda_pca_run_subparser(subparsers)
    add_train_subparser(subparsers)
    add_export_subparser(subparsers)

    args = parser.parse_args()

    if args.command == "lda-pca-run":
        config = LDAPCAConfig(
            root=Path(args.root),
            label_root=Path(args.label_root),
            out_root=Path(args.out_root),
            modalities=args.modalities,
            splits=args.splits,
            fg_samples_per_image=args.fg_samples_per_image,
            bg_samples_per_image=args.bg_samples_per_image,
            max_pixels=args.max_pixels,
            random_seed=args.random_seed,
        )
        lda_dir, pca, glo, ghi = run_lda_pca_pipeline(config)
        print("LDA direction:", lda_dir)
        print("PCA explained variance:", pca.explained_variance_ratio_)
        save_lda_pca_params(lda_dir, pca, glo, ghi, prefix="lda_pca")
        print("LDA+PCA params saved")

    elif args.command == "train":
        train_yolo_lda_pca(
            data_yaml=args.data_yaml,
            model_yaml=args.model_yaml,
            project=args.project,
            name=args.name,
            imgsz=args.imgsz,
            epochs=args.epochs,
            patience=args.patience,
            batch=args.batch,
            device=args.device,
            amp=args.amp,
            augment=args.augment,
            workers=args.workers,
        )

    elif args.command == "export":
        import numpy as np
        lda_dir = np.load(args.lda_dir_path)
        pca_components = np.load(args.pca_components_path)
        pca_mean = np.load(args.pca_mean_path)
        glo = np.load(args.glo_path)
        ghi = np.load(args.ghi_path)
        with open(args.output, "w") as f:
            f.write(f"LDA direction:\n{lda_dir}\n")
            f.write(f"PCA components:\n{pca_components}\n")
            f.write(f"PCA mean:\n{pca_mean}\n")
            f.write(f"Global lo:\n{glo}\n")
            f.write(f"Global hi:\n{ghi}\n")
        print(f"LDA+PCA details exported to {args.output}")

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()