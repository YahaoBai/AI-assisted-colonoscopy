# V3 CNN + ConvLSTM

Independent V3 implementation (no legacy code modification).

## Architecture
- Shared CNN: ResNet18 front layers (`conv1 -> layer2`) to extract feature maps.
- Temporal: ConvLSTM over 3-frame feature maps.
- Head: GAP + MLP + `tanh` to output normalized `[yaw, pitch]`.

## Normalization Modes
- `robust_quantile`: fit on train split using `ActionNormConfig(quantile=0.995)`.
- `fixed_degree`: fixed scale using `1 degree` by default.

Saved checkpoint metadata always includes:
- `norm_mode`
- `normalizer_config`
- `scale_yaw`
- `scale_pitch`

## Train
```bash
python v3/train_convlstm3f.py --dataset-dir <orig_3f_npz_dir> --norm-mode robust_quantile --norm-quantile 0.995
python v3/train_convlstm3f.py --dataset-dir <orig_3f_npz_dir> --norm-mode fixed_degree --fixed-degree 1.0
```

## Inference
```bash
python v3/infer_convlstm3f.py --model-dir checkpoints_v3/convlstm3f --model-name convlstm3f_best.pth --input-path <npz_or_dir>
```

## API
```python
from v3.api_convlstm3f import predict
action = predict(img_stack)  # img_stack shape: (3, H, W)
```
