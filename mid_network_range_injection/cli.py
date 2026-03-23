import argparse
import sys

from .core import train_mid_network_range_injection


def main():
    parser = argparse.ArgumentParser(description="Mid-Network Range Injection CLI")
    parser.add_argument("--data-yaml", type=str, required=True, help="Data YAML file")
    parser.add_argument("--model-yaml", type=str, required=True, help="Model YAML file")
    parser.add_argument("--range-train-dir", type=str, required=True, help="Range train images directory")
    parser.add_argument("--range-val-dir", type=str, required=True, help="Range val images directory")
    parser.add_argument("--imgsz", type=int, default=1024, help="Image size")
    parser.add_argument("--epochs", type=int, default=500, help="Number of epochs")
    parser.add_argument("--patience", type=int, default=80, help="Early stopping patience")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--device", type=str, default="0", help="Device (e.g., 0 for GPU)")
    parser.add_argument("--amp", action="store_true", help="Use AMP")
    parser.add_argument("--augment", action="store_true", help="Use augmentation")
    parser.add_argument("--workers", type=int, default=0, help="Number of workers")

    args = parser.parse_args()

    train_mid_network_range_injection(
        data_yaml=args.data_yaml,
        model_yaml=args.model_yaml,
        range_train_dir=args.range_train_dir,
        range_val_dir=args.range_val_dir,
        imgsz=args.imgsz,
        epochs=args.epochs,
        patience=args.patience,
        batch=args.batch,
        device=args.device,
        amp=args.amp,
        augment=args.augment,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()