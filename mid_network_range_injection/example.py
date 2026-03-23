from pathlib import Path
from . import train_mid_network_range_injection


def run_default():
    # Example usage
    train_mid_network_range_injection(
        data_yaml="dual_stream.yaml",
        model_yaml="mid_network_range_injection/yolo11_rangebranch.yaml",
        range_train_dir="dataset_dual/range/train",
        range_val_dir="dataset_dual/range/valid",
        imgsz=1024,
        epochs=500,
        patience=80,
        batch=8,
        device=0,
        amp=False,
        augment=False,
        workers=0,
    )


if __name__ == "__main__":
    run_default()