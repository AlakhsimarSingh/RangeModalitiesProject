import cv2
import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
from pathlib import Path

from ultralytics.nn.tasks import DetectionModel
from ultralytics.nn.modules import Conv
from ultralytics.utils import LOGGER
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.data.dataset import YOLODataset

try:
    from ultralytics.utils.torch_utils import de_parallel
except ImportError:
    def de_parallel(model):
        return model.module if hasattr(model, "module") else model


class RangeEncoder(nn.Module):
    """
    Lightweight range encoder.
    Input : (B, 1, H, W)
    Output: (B, RANGE_CH, H/16, W/16)
    """
    def __init__(self, out_ch: int = 128):
        super().__init__()
        self.encode = nn.Sequential(
            Conv(1,      16,     3, 2),   # → P1  /2
            Conv(16,     32,     3, 2),   # → P2  /4
            Conv(32,     64,     3, 2),   # → P3  /8
            Conv(64,     out_ch, 3, 2),   # → P4  /16
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode(x)


class MidNetworkRangeInjectionModel(DetectionModel):
    """
    Standard YOLOv11n (parsed from yaml as-is, ch=3) plus a lightweight
    range encoder injected at neck P4 (after layer INJECT_AFTER).
    """

    INJECT_AFTER = 13
    RANGE_CH     = 128

    def __init__(self, cfg="yolo11n.yaml", ch=3, nc=None, verbose=True):
        # Build standard 3-ch model from yaml
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

        # Range branch — attached AFTER super().__init__() completes
        self.range_encoder = RangeEncoder(out_ch=self.RANGE_CH)
        p4_ch = self._p4_channels()
        self.range_proj = nn.Sequential(
            nn.Conv2d(p4_ch + self.RANGE_CH, p4_ch, 1, bias=False),
            nn.BatchNorm2d(p4_ch),
            nn.SiLU(),
        )
        LOGGER.info(
            f"MidNetworkRangeInjectionModel: range injected after layer {self.INJECT_AFTER}, "
            f"P4 ch={p4_ch}, range_ch={self.RANGE_CH}"
        )

    def _p4_channels(self) -> int:
        m = self.model[self.INJECT_AFTER]
        try:
            return m.cv2.conv.out_channels
        except AttributeError:
            return 128  # yolo11n-n width=0.25 → 512*0.25=128

    def _run_once(self, x_comb4: torch.Tensor, range_feat=None) -> torch.Tensor:
        """Layer-by-layer forward. Injects range_feat after INJECT_AFTER."""
        y   = []
        inp = x_comb4
        for i, m in enumerate(self.model):
            if m.f != -1:
                inp = (
                    y[m.f]
                    if isinstance(m.f, int)
                    else [inp if j == -1 else y[j] for j in m.f]
                )
            inp = m(inp)

            if i == self.INJECT_AFTER and range_feat is not None:
                inp = self.range_proj(
                    torch.cat([inp, range_feat], dim=1)
                )

            y.append(inp if m.i in self.save else None)

        return inp

    def _extract_and_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Split 4-ch tensor, run both streams."""
        if not hasattr(self, "_range_debug_done"):
            print("\n[MODEL DEBUG]")
            print("Input shape:", x.shape)
            print("Range channel stats:",
                x[:, 3].min().item(),
                x[:, 3].max().item())
            self._range_debug_done = True
        x_comb4 = x[:, :3]
        # If input has 4 channels use the range channel, else use zeros
        if x.shape[1] >= 4:
            x_range = x[:, 3:4]
        else:
            x_range = torch.zeros(
                x.shape[0], 1, x.shape[2], x.shape[3],
                dtype=x.dtype, device=x.device
            )
        range_feat = self.range_encoder(x_range)
        return self._run_once(x_comb4, range_feat)

    def forward(self, x, augment=False, profile=False, visualize=False, **kwargs):
        # During super().__init__(), ultralytics calls forward() with a 3-ch
        # tensor to compute strides — range_encoder doesn't exist yet, skip it
        if not hasattr(self, "range_encoder"):
            if isinstance(x, dict):
                return self.criterion(self._run_once(x["img"][:, :3]), x)
            return self._run_once(x[:, :3])

        # Normal training call: ultralytics passes full batch dict via model(batch)
        if isinstance(x, dict):
            preds = self._extract_and_forward(x["img"])
            return self.criterion(preds, x)

        # Inference / validation call: plain tensor
        return self._extract_and_forward(x)

    def _predict_once(self, x, profile=False, visualize=False, embed=None, **kwargs):
        """Val/predict path."""
        if not hasattr(self, "range_encoder"):
            return self._run_once(x[:, :3])
        return self._extract_and_forward(x)

    def loss(self, batch, preds=None):
        """AMP / DDP path: unwrap_model(model).loss(batch, preds)."""
        if preds is None:
            preds = self._extract_and_forward(batch["img"])
        return self.criterion(preds, batch)


class MidNetworkRangeInjectionDataset(YOLODataset):
    """
    Overrides load_image() to stack Comb4 (3-ch) + Range (1-ch) → 4-ch array.
    """

    def __init__(self, *args, range_dir: str, **kwargs):
        self.range_dir = Path(range_dir)
        super().__init__(*args, **kwargs)

    def load_image(self, i):
        """Return (img_HWC, (orig_h, orig_w), (resized_h, resized_w))."""
        img_path  = Path(self.im_files[i])

        # Comb4 (3-ch BGR)
        img_comb4 = cv2.imread(str(img_path))
        if img_comb4 is None:
            raise FileNotFoundError(f"Comb4 image not found: {img_path}")

        # Range (1-ch)
        range_path = self.range_dir / img_path.name
        try:
            img_range = cv2.imread(str(range_path), cv2.IMREAD_UNCHANGED)
            if img_range is None:
                raise FileNotFoundError
            if img_range.ndim == 3:
                img_range = img_range[:, :, 0]
            img_range = img_range.astype(np.float32) / 255.0
        except Exception:
            LOGGER.warning(f"Range image not found: {range_path} — using zeros")
            img_range = np.zeros(img_comb4.shape[:2], dtype=np.uint8)
        img_4ch = np.concatenate(
            [img_comb4, img_range[:, :, None]], axis=-1
        ).astype(np.uint8)

        orig_h, orig_w = img_4ch.shape[:2]
        r = self.imgsz / max(orig_h, orig_w)
        if r != 1:
            interp = cv2.INTER_LINEAR if r > 1 else cv2.INTER_AREA
            img_4ch = cv2.resize(
                img_4ch,
                (int(orig_w * r), int(orig_h * r)),
                interpolation=interp,
            )

        return img_4ch, (orig_h, orig_w), img_4ch.shape[:2]

    @staticmethod
    def collate_fn(batch):
        # Use ultralytics default collate for all keys except img
        from ultralytics.data.dataset import YOLODataset as _YDS
        new_batch = _YDS.collate_fn(batch)
        return new_batch


class MidNetworkRangeInjectionTrainer(DetectionTrainer):

    def __init__(self, range_train_dir: str, range_val_dir: str, **kwargs):
        self.range_train_dir = range_train_dir
        self.range_val_dir   = range_val_dir
        super().__init__(**kwargs)

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = MidNetworkRangeInjectionModel(
            cfg=cfg or self.args.model,
            ch=3,
            nc=self.data["nc"],
            verbose=verbose,
        )
        if weights:
            model.load(weights)
        return model

    def set_model_attributes(self):
        """Set standard attributes then ensure criterion is initialised."""
        super().set_model_attributes()
        # super() calls model.init_criterion() internally in newer ultralytics;
        # force it here for older versions that don't
        if not hasattr(self.model, "criterion") or self.model.criterion is None:
            self.model.criterion = self.model.init_criterion()

    def build_dataset(self, img_path, mode="train", batch=None):
        gs        = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        range_dir = self.range_train_dir if mode == "train" else self.range_val_dir
        # Mosaic is incompatible with custom load_image — force off
        self.args.mosaic = 0.0
        self.args.mixup  = 0.0
        return MidNetworkRangeInjectionDataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=mode == "train",
            hyp=self.args,
            rect=self.args.rect,
            cache=self.args.cache or None,
            single_cls=self.args.single_cls or False,
            stride=int(gs),
            pad=0.0 if mode == "train" else 0.5,
            prefix=f"{mode}: ",
            task=self.args.task,
            classes=self.args.classes,
            data=self.data,
            fraction=self.args.fraction if mode == "train" else 1.0,
            range_dir=range_dir,
        )


def train_mid_network_range_injection(data_yaml: str, model_yaml: str, range_train_dir: str, range_val_dir: str, **kwargs) -> None:
    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise ImportError("ultralytics package is required for YOLO training") from e

    # Use custom trainer
    trainer = MidNetworkRangeInjectionTrainer(
        range_train_dir=range_train_dir,
        range_val_dir=range_val_dir,
        model=model_yaml,
        data=data_yaml,
        **kwargs
    )
    trainer.train()