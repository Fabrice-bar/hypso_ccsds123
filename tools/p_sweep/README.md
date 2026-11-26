# P / local-sum / prediction-mode sweep

Sweeps P = 1..10 × 4 local sums × full/reduced prediction (80 configurations per image) on HYPSO-1 L1a
images. Every other parameter stays at the Vorhaug (2024) flight configuration, Table 3.1. The baseline
(P=3, narrow column, reduced) is one of the 80 configurations.

## Files

- `p_sweep.py`: P / local-sum / mode sweep, plus the shared code (image readers, header, engines, verification).
- `vmax_sweep.py`: ν_max sweep (weight adaptation) at several A* values. It needs `p_sweep.py` and `fast_engine.py` in this folder.
- `fast_engine.py`: the model's predictor and hybrid encoder as one compiled (Numba) loop.
  `--verify` checks it bit for bit against the unmodified model in `ccsds123_i2_hlm/`.

## Setup (from the repository root)

```
pip install -r requirements.txt
pip install matplotlib netCDF4 numba psutil
```

## Run (from the repository root)

```
python tools/p_sweep/p_sweep.py --model-dir . --images data --verify --dry-run
python tools/p_sweep/p_sweep.py --model-dir . --images data --verify --out results/p_sweep
```

- `--images data` reads every `.nc` file in `data/`. It uses `products/dn` (raw uint16 counts, lines × samples × bands).
- `--center-crop 256x256` or `--crop Y0:Y1,X0:X1` runs on a crop instead of the full image.
- `--lossless` switches both the sweep and the baseline to lossless.
- `--plot-only --out results/p_sweep` redraws the plots from existing results.
- An interrupted run continues where it stopped when the same command is run again.

## Outputs (in `--out`)

- `results.csv`: one row per run.
- `rate_vs_P_*.png/.pdf`: bits/sample against P per image and the mean, with the baseline marked.
- `delta_vs_baseline_*.png/.pdf`: mean % change against the baseline.
- `summary_*.md`: best configuration per image, and the best single configuration across all images.

## ν_max sweep

```
python tools/p_sweep/vmax_sweep.py --model-dir . --images data --verify --out results/vmax_sweep
```

Defaults:
- ν_max = -1..9 at A* = 0 (lossless), 2, 4 and 8, with no relative limit.
- ν_min = -1, t_inc = 2^11.
- P=3, wide neighbor, full; everything else as in Vorhaug 2024.

Options: `--vmax`, `--abs-error`, `--rel-error R`, `--vmin`, `--tinc`, `--omega`, `--P`, `--local-sum`, `--mode`, and the crop options.

Plots: rate change against ν_max relative to ν_max = 4 (`vmax_delta_*`), bits/sample (`vmax_rate_*`) and PSNR (`vmax_psnr_*`).
