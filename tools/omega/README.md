# HYPSO weight-resolution study: Green Book Figures 4-17 and 4-18

This script sweeps the predictor weight-component resolution, Omega, using
CCSDS 123 issue 2 and your five HYPSO captures. The study design follows
CCSDS 120.2-G-2, section 4.2.4, printed pages 4-17 and 4-18. It produces PNG
and PDF plots plus CSV results. It does not recreate the original corpus's
numerical results or claim to average several captures from one instrument.
Each HYPSO curve is the measured rate for one selected capture/crop.

## Install

Extract this ZIP into the root of your existing verification-model repository.
It adds `tools/omega/`. Keep all five Python files together. Existing sweep
folders and the upstream codec are not modified. Use the same environment as
for your vmin study: NumPy, bitarray, netCDF4, matplotlib and psutil.
On Windows without an activated environment, replace `python` in the commands
with `.\.venv\Scripts\python.exe`. Commands below are each a single line.

## First study: Figure 4-17, 256 x 256

```bash
python tools/omega/omega_sweep.py --figure 417 --input-dir data --crop 256 256 --omega-min 4 --omega-max 19 --a-values 0 --nu-min -1 --nu-max 3 --t-inc 2048 --phi 3 --workers 2 --output results/omega_fig417_256
```

This is **80 configurations**: 5 images x 16 resolutions x 1 error limit.
`fig417_lossless.png` / `.pdf` contains two panels:

1. Compressed bits/sample versus Omega, one curve per HYPSO image.
2. Percentage rate increase versus Omega, again one curve per image.

The reference is the same image at Omega=19, with all other settings identical.
The percentage is `100 * (rate_at_omega / rate_at_19 - 1)`.
It is not the difference in bits/sample, a percentage of the raw file size,
or the difference from the lowest rate anywhere in the sweep.

## Figure 4-18: error-limit comparison

```bash
python tools/omega/omega_sweep.py --figure 418 --input-dir data --crop 256 256 --omega-min 4 --omega-max 19 --a-values 0 1 3 7 15 --nu-min -1 --nu-max 3 --t-inc 2048 --phi 3 --workers 2 --output results/omega_fig418_256
```

This is **400 configurations**. `fig418_error_limits.png` / `.pdf` has one
panel per image and one curve per A. Each image/A pair has its own Omega=19
reference. The default list stops at your usual A=15. The original Figure
4-18 also includes A=31; use `--a-values 0 1 3 7 15 31` to include it
(480 configurations for five images and all 16 resolutions).

To create both figure layouts from a single simulation grid, use
`--figure both`. With the 400-point grid above, that costs no extra encodes:
the A=0 subset supplies Figure 4-17. You can also replot existing Figure 4-18
results containing A=0 without any new simulations:

```bash
python tools/omega/omega_sweep.py --plot-only --figure both --output results/omega_fig418_256
```

A quick initial near-lossless screen with only A=3 and A=15 needs 160
configurations: `--figure 418 --a-values 3 15`. To reduce the resolution grid,
use e.g. `--omega-values 4 8 11 14 16 19` instead of min/max options.

## Full images and other crops

Replace `--crop 256 256` with `--full-image`. Do not pass both options.
The script reads each NetCDF's dimensions separately; it does not assume
that all scenes are 956 x 684 or that Adelaide has the same dimensions.
The full image always starts at row=0, col=0 and uses every native band.

Example with fewer full-image candidates (30 configurations):

```bash
python tools/omega/omega_sweep.py --figure 417 --input-dir data --full-image --omega-values 4 8 11 14 16 19 --a-values 0 --nu-min -1 --nu-max 3 --t-inc 2048 --phi 3 --workers 2 --max-memory-gb 90 --total-memory-gb 90 --output results/omega_fig417_full
```

For a centre 128 x 128 crop, use `--crop 128 128`; centre is the default.
Use `--anchor top-left` for a crop starting at (0,0), or `--row R --col C`
for an explicit zero-based origin. All configurations for one scene reuse
the same crop; origin and SHA-256 are saved. Oversized/out-of-bounds crops
are reported as skipped; they are never resized silently.

`--images lisbon` selects one scene. The default scenes are Lisbon,
Bangladesh, Santarem, Doha and Finnmark. For Adelaide or another NetCDF:

```bash
python tools/omega/omega_sweep.py --figure 417 --files data/adelaide_2025-09-09T01-12-29Z-l1a.nc --full-image --omega-values 11 14 19 --workers 2 --output results/omega_adelaide_full
```

Append `--dry-run` to any command to inspect paths, crop sizes, configuration
count and estimated RAM without encoding or creating result files.
A very small workflow check is:

```bash
python tools/omega/omega_sweep.py --figure both --images lisbon --input-dir data --crop 8 8 --omega-values 4 19 --a-values 0 3 --workers 2 --output results/omega_smoke
```

## Fixed HYPSO profile and relation to the book

| Parameter | Default | Command option |
|---|---:|---|
| Prediction bands P | 3 | `--p` |
| Prediction mode | reduced | `--prediction-mode` |
| Local sum | narrow-neighbor | `--local-sum` |
| Initial adaptation nu_min | -1 | `--nu-min` |
| Final adaptation nu_max | 3 | `--nu-max` |
| Update interval t_inc | 2048 | `--t-inc` |
| Sample-representative damping phi | 3 | `--phi` |
| Offset psi | 7 near-lossless; 0 lossless | `--psi` |
| Register size R | 64 | fixed |
| Representative resolution Theta | 3 | fixed |
| Dynamic range D | 16 unsigned | fixed |
| Entropy coder | hybrid | fixed |
| Weight initialization / exponent offsets | default / all zero | fixed |
| Weight resolution Omega | every integer 4..19 | `--omega-min`, `--omega-max`, or `--omega-values` |

The Figure 4-17 caption explicitly specifies nu_max=3. We use that common
value for both HYPSO plot modes so changes across A are evaluated with the
same predictor settings. The original Figure 4-18 caption gives no such
explicit override; Annex C's general defaults include nu_max=7 and t_inc=64,
plus instrument-dependent local sums, prediction modes and damping. Your
HYPSO defaults here use t_inc=2048, phi=3, reduced mode and narrow-neighbour
sums, following the profile you have been studying. These differences are
intentional and are recorded in every output. `--t-inc 64 --nu-max 7` selects
those two Annex C defaults if you want a separate comparison; it does not
make a HYPSO crop identical to one of the book's instrument experiments.

Only Omega varies along each curve. In the model header it is encoded as
`weight_component_resolution = Omega - 4`. The model changes initialization,
weight bounds, prediction scaling, representative calculations and update
scaling accordingly. Its applied exponent is checked against
`rho = nu(t) + D - Omega`; the observer is not hard-coded to Omega=19.
The error limit A changes between Figure 4-18 curves, with every other fixed
profile parameter held constant. Damping remains fixed even when A=0.

All native `products/dn` bands are used in their original order, consistent
with the earlier predictor sweeps. No wavelength selection, calibration,
binning, resampling or artificial noise is applied. The A limits are in raw
DN, not calibrated radiance or reflectance units. Choosing fewer wavelength
bands would be a separate experiment with a different predictor input.

## References, partial results and plotting

Omega=19 is always included, even if it is absent from your requested range
or list. These references run first within each image/A group and appear in
the plots/CSV. A restricted list such as `--omega-values 8 11 14` therefore
encodes four resolutions: 8,11,14 and the reference 19.

Missing/failed references leave the corresponding percentage values blank
and curves absent; another image or A is never substituted. Negative rate
changes and non-monotonic curves are retained. The vertical scale is automatic
and does not hide rates above the book's plotted 10% range. Partial results
show completed-point counts and missing points as gaps.

Outputs:

- `results.csv`: actual parameters, bit counts, bits/sample, compressed payload
  SHA-256, max reconstruction error, RMSE, timings and memory diagnostics.
- `rate_comparison.csv`: per-point matched reference rate and percentage change.
- `fig417_lossless.png/.pdf` and/or `fig418_error_limits.png/.pdf`.
- `adaptation_progression.csv`: actual checked update schedule for each Omega.
- `metadata.json`, `plan.csv`, per-job JSON/log files and failure/skip tables.

Rate counts the CCSDS header, entropy payload including termination, and
output-word padding per scalar sample. External initialization is excluded
from the plotted rate; an inclusive rate is also saved. Lossless A=0 uses the
lossless header mode, rather than signalling absolute-only A=0. These choices
are documented for reproducibility and can matter particularly for tiny crops.
Error checks use encoder-side reconstruction; there is no independent decoder
validation. Runtime and memory are recorded but are never plotted.

## Workers and resuming

Each worker handles one whole image/crop configuration, with its own original
serial predictor and hybrid encoder. `--workers 2` means up to two independent
configurations at once. A finished process releases its memory before its
slot starts another configuration. There is no nested predictor pool. Native
library threads are limited to one per worker; physical cores are not pinned.

Concurrency is limited by CPU allowance, available RAM and the total budget.
`--max-memory-gb` is a per-configuration planning ceiling, while
`--total-memory-gb` includes all active configurations and coordinator headroom.
These are estimates, not hard operating-system limits. The existing estimate
for P=3, low-memory mode and 956x684x120 samples is about 30.9 GiB per job;
a 90 GiB total budget admits at most two such configurations simultaneously.
The Python reference model still uses int64 arrays at every Omega, so reducing
Omega should not be interpreted as reducing host RAM or measuring FPGA savings.

Completed points are checkpointed individually. Repeat the same command to
resume; changing worker count or RAM budgets is allowed. Use a new output
folder when changing the simulation grid, crop, input or fixed settings.
`--plot-only --figure ...` can change the plot layout for an existing grid.
Interrupting a run stops active workers, preserves completed points, and
regenerates partial plots. See `VALIDATION.md` for checks and limitations.
