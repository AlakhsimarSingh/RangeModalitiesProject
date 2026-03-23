import argparse
import sys
from pathlib import Path

from .core import QMPCAConfig, run_pca_pipeline, train_yolo


def add_pca_run_subparser(subparsers):
    parser = subparsers.add_parser("pca-run", help="Run PCA fusion pipeline")
    parser.add_argument("--root", type=str, required=True, help="Root dataset path")
    parser.add_argument("--out-root", type=str, required=True, help="Output root path")
    parser.add_argument("--modalities", nargs="+", default=["reflec", "signal", "nearir", "range"], help="List of modalities")
    parser.add_argument("--splits", nargs="+", default=["train", "valid", "test"], help="Dataset splits")
    parser.add_argument("--max-pixels", type=int, default=200000, help="Max pixels for PCA training")
    parser.add_argument("--sample-per-image", type=int, default=1000, help="Samples per image")
    parser.add_argument("--n-components", type=int, default=3, help="Number of PCA components")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed")


def add_pca_train_subparser(subparsers):
    parser = subparsers.add_parser("pca-train", help="Train YOLO model")
    parser.add_argument("--data-yaml", type=str, required=True, help="Data YAML file")
    parser.add_argument("--model-yaml", type=str, required=True, help="Model YAML file")
    parser.add_argument("--project", type=str, required=True, help="Project name")
    parser.add_argument("--name", type=str, required=True, help="Run name")
    parser.add_argument("--imgsz", type=int, default=1024, help="Image size")
    parser.add_argument("--epochs", type=int, default=500, help="Number of epochs")
    parser.add_argument("--patience", type=int, default=80, help="Early stopping patience")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--device", type=str, default="0", help="Device (e.g., 0 for GPU)")
    parser.add_argument("--amp", action="store_true", help="Use AMP")
    parser.add_argument("--augment", action="store_true", help="Use augmentation")
    parser.add_argument("--workers", type=int, default=0, help="Number of workers")


def add_export_subparser(subparsers):
    parser = subparsers.add_parser("export", help="Export PCA components")
    parser.add_argument("--pca-path", type=str, required=True, help="Path to saved PCA model (numpy file)")
    parser.add_argument("--output", type=str, required=True, help="Output file path")


def main():
    parser = argparse.ArgumentParser(description="QM-PCA CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    add_pca_run_subparser(subparsers)
    add_pca_train_subparser(subparsers)
    add_export_subparser(subparsers)

    args = parser.parse_args()

    if args.command == "pca-run":
        config = QMPCAConfig(
            root=Path(args.root),
            out_root=Path(args.out_root),
            modalities=args.modalities,
            splits=args.splits,
            max_pixels=args.max_pixels,
            sample_per_image=args.sample_per_image,
            random_seed=args.random_seed,
        )
        pca = run_pca_pipeline(config, n_components=args.n_components)
        print("PCA explained variance ratio:", pca.explained_variance_ratio_)
        # Optionally save PCA
        np.save("pca_model.npy", pca)
        print("PCA model saved to pca_model.npy")

    elif args.command == "pca-train":
        train_yolo(
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
        pca = np.load(args.pca_path, allow_pickle=True).item()
        with open(args.output, "w") as f:
            f.write(f"Components:\n{pca.components_}\n")
            f.write(f"Explained variance:\n{pca.explained_variance_ratio_}\n")
        print(f"PCA details exported to {args.output}")

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()