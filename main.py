#!/usr/bin/env python3
"""
Unified CLI for Snowpole Detection Fusion Experiments

Run different fusion methods:
- fbpca: Foreground-Biased PCA
- qm-pca: Quad Modal PCA
- lda-pca: LDA + PCA Fusion
"""

import argparse
import sys
from pathlib import Path

# Import the CLI mains from each package
try:
    from fbpca_analysis.cli import main as fbpca_main
    from qm_pca.cli import main as qm_pca_main
    from lda_pca_fusion.cli import main as lda_pca_main
except ImportError as e:
    print(f"Error importing packages: {e}")
    print("Make sure all packages are installed or in PYTHONPATH")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Unified CLI for Snowpole Detection Fusion Experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py fbpca fbpca-run --root "..." --label-root "..." --out-root "..."
  python main.py qm-pca pca-run --root "..." --out-root "..."
  python main.py lda-pca lda-pca-run --root "..." --label-root "..." --out-root "..."
        """
    )

    subparsers = parser.add_subparsers(dest="package", help="Fusion package to run")

    # FB-PCA subparser
    fbpca_parser = subparsers.add_parser("fbpca", help="Foreground-Biased PCA")
    fbpca_subparsers = fbpca_parser.add_subparsers(dest="command")
    # Add the subcommands from fbpca cli
    from fbpca_analysis.cli import add_fbpca_run_subparser, add_export_subparser as add_fbpca_export
    add_fbpca_run_subparser(fbpca_subparsers)
    add_fbpca_export(fbpca_subparsers)

    # QM-PCA subparser
    qm_pca_parser = subparsers.add_parser("qm-pca", help="Quad Modal PCA")
    qm_pca_subparsers = qm_pca_parser.add_subparsers(dest="command")
    from qm_pca.cli import add_pca_run_subparser, add_export_subparser as add_qm_pca_export
    add_pca_run_subparser(qm_pca_subparsers)
    add_qm_pca_export(qm_pca_subparsers)

    # LDA-PCA subparser
    lda_pca_parser = subparsers.add_parser("lda-pca", help="LDA + PCA Fusion")
    lda_pca_subparsers = lda_pca_parser.add_subparsers(dest="command")
    from lda_pca_fusion.cli import add_lda_pca_run_subparser, add_train_subparser, add_export_subparser as add_lda_pca_export
    add_lda_pca_run_subparser(lda_pca_subparsers)
    add_train_subparser(lda_pca_subparsers)
    add_lda_pca_export(lda_pca_subparsers)

    args = parser.parse_args()

    if not args.package:
        parser.print_help()
        return

    # Simulate sys.argv for the sub-cli
    if args.package == "fbpca":
        # Replace sys.argv with the fbpca args
        original_argv = sys.argv[:]
        sys.argv = ["fbpca_cli"] + sys.argv[2:]  # Remove "main.py fbpca"
        try:
            fbpca_main()
        finally:
            sys.argv = original_argv

    elif args.package == "qm-pca":
        original_argv = sys.argv[:]
        sys.argv = ["qm_pca_cli"] + sys.argv[2:]
        try:
            qm_pca_main()
        finally:
            sys.argv = original_argv

    elif args.package == "lda-pca":
        original_argv = sys.argv[:]
        sys.argv = ["lda_pca_cli"] + sys.argv[2:]
        try:
            lda_pca_main()
        finally:
            sys.argv = original_argv

    else:
        parser.print_help()


if __name__ == "__main__":
    main()