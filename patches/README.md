# 4-Channel Ultralytics Modifications

This directory contains patches and setup scripts for enabling 4-channel (RGBD) input support in ultralytics YOLO models.

## Overview

The modifications enable YOLOv11 and other YOLO versions to accept 4-channel images (RGB + Depth) for training and inference, without dropping the 4th channel during data loading.

## Files

- `patches/ultralytics_4ch.patch` - Git patch file containing all modifications
- `scripts/setup_4ch_ultralytics.py` - Automated setup script

## Modified Files

The patch modifies 3 files in ultralytics:

1. `ultralytics/data/base.py` - Training data loading
2. `ultralytics/data/loaders.py` - Inference data loading (3 locations)
3. `ultralytics/nn/modules/block.py` - Added Silence class for compatibility

## Changes Made

### Data Loading (base.py, loaders.py)
- Modified `cv2_flag` logic to use `cv2.IMREAD_UNCHANGED` for 4-channel images
- Ensures 4th channel (depth) is preserved during loading

### Module Compatibility (block.py)
- Added `Silence` class implementation
- Updated `__all__` exports

## Usage

### Automatic Setup
```bash
# Install ultralytics 8.4.6 first
pip install ultralytics==8.4.6

# Run setup script
python scripts/setup_4ch_ultralytics.py
```

### Manual Application
```bash
# Install ultralytics 8.4.6
pip install ultralytics==8.4.6

# Find ultralytics location
python -c "import ultralytics; print(ultralytics.__file__)"

# Apply patch
cd /path/to/ultralytics
git apply /path/to/project/patches/ultralytics_4ch.patch
```

## Verification

After applying modifications, verify with:
```python
from ultralytics import YOLO
model = YOLO("yolo11n.yaml", task="detect")
# Should load successfully with 4-channel support
```

## Data Configuration

Use `channels: 4` in your data YAML:
```yaml
path: ./dataset
train: images/train
val: images/valid
nc: 1
names: ["snowpole"]
channels: 4  # Enable 4-channel input
```

## Compatibility

- **ultralytics**: 8.4.6 (tested)
- **YOLO versions**: v8, v9, v10, v11
- **Python**: 3.8+
- **OpenCV**: Any version with cv2.IMREAD_UNCHANGED

## Troubleshooting

### Patch Application Fails
- Ensure ultralytics 8.4.6 is installed
- Check that no other modifications exist
- Try `git apply --whitespace=fix`

### Import Errors
- Restart Python session after applying patch
- Verify Silence class is in block.py

### Data Loading Issues
- Ensure images are saved with 4 channels (RGBA PNG)
- Check data YAML has `channels: 4`

## Development

To modify the patches:
1. Make changes to ultralytics source
2. Generate new patch: `git diff > patches/ultralytics_4ch.patch`
3. Test setup script
4. Update this README

## Related Documentation

- `PATCH.md` - Detailed technical documentation of changes
- `main.py` - Unified CLI for fusion experiments