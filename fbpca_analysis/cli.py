import argparse
import sys
from pathlib import Path

from .core import FBPCAConfig, run_fbpca_pipeline, save_fbpca_params


def add_fbpca_run_subparser(subparsers):
    parser = subparsers.add_parser("fbpca-run", help="Run FB-PCA fusion pipeline")
    parser.add_argument("--root", type=str, required=True, help="Root dataset path")
    parser.add_argument("--label-root", type=str, required=True, help="Label root path")
    parser.add_argument("--out-root", type=str, required=True, help="Output root path")
    parser.add_argument("--modalities", nargs="+", default=["reflec", "signal", "nearir", "range"], help="List of modalities")
    parser.add_argument("--splits", nargs="+", default=["train", "valid", "test"], help="Dataset splits")
    parser.add_argument("--fg-samples-per-image", type=int, default=800, help="FG samples per image")
    parser.add_argument("--max-pixels", type=int, default=300000, help="Max pixels for PCA training")
    parser.add_argument("--bbox-dilate", type=int, default=3, help="Bbox dilation")
    parser.add_argument("--n-components", type=int, default=3, help="Number of PCA components")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed")


def add_export_subparser(subparsers):
    parser = subparsers.add_parser("export", help="Export FB-PCA components")
    parser.add_argument("--components-path", type=str, required=True, help="Path to saved PCA components (numpy file)")
    parser.add_argument("--mean-path", type=str, required=True, help="Path to saved PCA mean")
    parser.add_argument("--glo-path", type=str, required=True, help="Path to global_lo")
    parser.add_argument("--ghi-path", type=str, required=True, help="Path to global_hi")
    parser.add_argument("--output", type=str, required=True, help="Output file path")


def main():
    parser = argparse.ArgumentParser(description="FB-PCA Analysis CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    add_fbpca_run_subparser(subparsers)
    add_export_subparser(subparsers)

    args = parser.parse_args()

    if args.command == "fbpca-run":
        config = FBPCAConfig(
            root=Path(args.root),
            label_root=Path(args.label_root),
            out_root=Path(args.out_root),
            modalities=args.modalities,
            splits=args.splits,
            fg_samples_per_image=args.fg_samples_per_image,
            max_pixels=args.max_pixels,
            bbox_dilate=args.bbox_dilate,
            random_seed=args.random_seed,
        )
        pca, glo, ghi = run_fbpca_pipeline(config, n_components=args.n_components)
        print("FB-PCA explained variance ratio:", pca.explained_variance_ratio_)
        save_fbpca_params(pca, glo, ghi, prefix="fbpca")
        print("FB-PCA params saved")

    elif args.command == "export":
        import numpy as np
        components = np.load(args.components_path)
        mean = np.load(args.mean_path)
        glo = np.load(args.glo_path)
        ghi = np.load(args.ghi_path)
        with open(args.output, "w") as f:
            f.write(f"Components:\n{components}\n")
            f.write(f"Mean:\n{mean}\n")
            f.write(f"Global lo:\n{glo}\n")
            f.write(f"Global hi:\n{ghi}\n")
        print(f"FB-PCA details exported to {args.output}")

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()