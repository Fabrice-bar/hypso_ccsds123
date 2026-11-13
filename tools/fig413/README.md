# HYPSO joint offset / damping / noise study (Figure 4-13 style)

This script searches the full selected grid of sample-representative damping
phi and offset psi. It creates heatmaps like CCSDS 120.2-G-2 Figure 4-13:

`expansion (%) = 100 * (compressed size / best tested compressed size - 1)`

The minimum is found **separately for each image, A and added-noise level**.
A value of 0% is a best tested pair; 20% means that configuration needs 20%
more compressed bits than the best pair in that panel. These percentages
are not expansion relative to the uncompressed image.

The horizontal axis is normalized offset `psi / 2**Theta`; the vertical axis
is normalized damping `phi / 2**Theta`. At Theta = 3, integers 0..7 map to
0, 0.125, ..., 0.875. They are **not** divided by 7.

## Relationship to the Green Book and preceding HYPSO studies

The book's experiment starts with a spatially aggregated, flat-field-corrected,
high-SNR AVIRIS-NG image. It uses P = 2, A = 40, Theta = 4, Omega = 19 and
added Gaussian sigma values 0, 25, 50, 100, 200, 400. The first spectral band
has phi = psi = 0; the chosen pair is used in all remaining bands.

This package applies the same **grid-search and relative-size visualization**
to your HYPSO crops, retaining A <= 15 and your predictor/coder choices.
It does not recreate the book's source image or preprocessing. HYPSO raw
DN already contains instrument noise: **sigma = 0 means no additional noise,
not a noise-free image**. Noise is added independently to scalar samples as
a controlled sensitivity experiment, not a calibrated model of HYPSO noise.

The predictor's final adaptation parameter is held at **nu_max = 4** across
the whole heatmap. This is a candidate from your earlier adaptation study
and keeps the heatmap focused on the (phi, psi) interaction at a fixed
predictor profile. The previous offset package varied nu_max with phi;
its rates are therefore not generally interchangeable with these rates.
Use `--nu-max` to examine another fixed profile in a separate output folder.

The first band is zeroed by default to match Figure 4-13. Both per-band
representative tables are embedded in the header at every grid point, so
their overhead is consistent across the grid. `--first-band uniform` applies
the chosen pair to all bands for comparison with your preceding scripts.

## Default configuration

| Parameter | Default |
| --- | --- |
| Captures | Lisbon, Bangladesh, Santarem, Doha, Finnmark |
| Crop | Centre 128 x 128; adjustable |
| Bands / input | All native bands; raw integer `products/dn` |
| P | 3 |
| Prediction | Reduced |
| Local sum | Narrow neighbour-oriented |
| Encoder | Hybrid |
| D / R / Omega | 16 / 64 / 19 |
| Theta | 3 |
| phi grid | Every legal value 0..7 |
| psi grid | Every legal value 0..7 |
| First spectral band | phi = psi = 0 |
| nu_min / nu_max | -1 / **4, fixed** |
| t_inc | 512 |
| A | 3 and 15 DN; lossless omitted |
| Added sigma/A | 0, 0.5, 1, 2, 5, 10 |
| Noise seed / repeats | 2026 / 1 |

The full legal grid includes phi = 6 and 7, which were not in your preceding
six-curve sweep. It is necessary to test these too before calling the result
the optimum over all legal pairs at Theta = 3. The optimum remains conditional
on the crop, noise realization, A, predictor profile and encoder.

The chosen default sigma/A levels compare added noise with the reconstruction
error limit. They are exploratory HYPSO settings, not the book's exact levels:

| A | Added Gaussian sigma values (DN) |
| --- | --- |
| 3 | 0, 1.5, 3, 6, 15, 30 |
| 15 | 0, 7.5, 15, 30, 75, 150 |

`--sigmas` accepts absolute DN values instead, e.g.
`--sigmas 0 25 50 100 200 400`. Do not combine this with `--noise-ratios`.

## Installation

Extract the ZIP into the verification-model repository root, creating
`tools/fig413/`. Keep all Python helpers together and run **fig413_sweep.py**.
Existing scripts and upstream model files are not replaced.

Use the same environment as your existing sweeps. It needs NumPy, bitarray,
Matplotlib, netCDF4 and psutil. If the extra packages are missing:

```bash
python -m pip install matplotlib netCDF4 psutil
```

The commands below assume the .venv is activated and the current directory
is the repository root. On Windows without activation, replace `python`
with `.\.venv\Scripts\python.exe`. All examples are single terminal lines.
Use `--repo PATH` if the script is outside the verification-model repository.

## Start with one image and one A

This runs 384 configurations: 6 noise levels x 8 phi x 8 psi.

```bash
python tools/fig413/fig413_sweep.py --input-dir data --images lisbon --a-values 15 --crop 128 128 --workers 12 --output results/fig413_lisbon_A15_128
```

It creates a six-panel PNG/PDF heatmap for Lisbon at A = 15, marks a best
pair in each panel, and lists all exact ties in `best_pairs.csv`.

Append `--dry-run` to check the plan and estimated memory without encoding.

For a quick functional check before the scientific run (eight jobs):

```bash
python tools/fig413/fig413_sweep.py --input-dir data --images lisbon --a-values 15 --crop 16 16 --phi-values 0 7 --psi-values 0 7 --sigmas 0 15 --workers 2 --output results/fig413_smoke
```

Only four pairs are tested in that smoke run, and its crop is tiny. It is
not an optimization over the complete legal grid and is labelled accordingly
in the summary. Untested cells appear gray, without interpolation.

## No added noise: optimize the original crops

```bash
python tools/fig413/fig413_sweep.py --input-dir data --crop 128 128 --sigmas 0 --workers 12 --output results/fig413_original_128
```

This has **640 configurations**: five images x two A values x 64 pairs.
It produces individual maps and a combined original-image overview.

## Full five-image noise study

```bash
python tools/fig413/fig413_sweep.py --input-dir data --crop 128 128 --workers 12 --output results/fig413_all_128
```

This has **3,840 configurations**: five images x two A values x six noise
levels x 64 pairs. Each image/A receives its own Figure 4-13-style figure.
Start with the smaller run above to check local runtime before committing
to the full grid. No full 3,840-point study has been precomputed here.

For 256 x 256, change `--crop` and use a new output folder:

```bash
python tools/fig413/fig413_sweep.py --input-dir data --images lisbon --a-values 15 --crop 256 256 --workers 12 --output results/fig413_lisbon_A15_256
```

All settings within an image use the identical centre crop and native bands.
Crop origins, input hashes and noisy-input hashes are stored. There is no
spatial binning, flat-field correction, rescaling, spectral subsampling or
denoising. An oversized crop is explicitly skipped, never silently resized.

## Noise generation and fair comparisons

For each source/crop/realization, the script derives a deterministic seed
from the input-file hash, crop coordinates, master seed and repetition index.
It generates a standard-normal field using NumPy PCG64. The field is shared
by **all parameter pairs** and scaled by sigma:

`compression_input = clip(round(original_DN + sigma * Z), 0, 65535)`

Each sigma starts from the original crop; noise is not added cumulatively.
The same underlying field is used across A and sigma for paired comparisons.
Worker count and completion order do not change the noisy input. Sigma = 0
uses the original integer samples exactly.

Rounding and clipping can change the achieved noise distribution. The CSV
records the measured perturbation mean, standard deviation, RMS, clipped
sample count and clipped fraction. Inspect clipping when using large sigmas;
the displayed sigma is the requested Gaussian standard deviation before
rounding/clipping. There is no automatic input rescaling to hide clipping.

The A-bound check compares reconstructed samples against the **noisy input
that was compressed**. It is not an A-bound on the total difference from the
original, unperturbed crop. Compression is not being treated as denoising.

## Repeated realizations and refinement

`--repeats 3` tests three independently seeded realizations at each nonzero
noise level. Sigma = 0 is evaluated only once because there is no randomness.
Each pair is evaluated on the same realizations as all other pairs.

For repeated realizations, the script first calculates each pair's mean
compressed size, then finds the minimum mean and calculates expansion.
It does not average independently selected per-realization minima.
Per-cell rate standard deviations are saved; they are blank for a single
nonzero-noise realization, where a sample standard deviation cannot be estimated.

With all default images/parameters, three realizations give 10,240 jobs.
It can be more practical to repeat a shortlist of promising pairs first,
using `--phi-values` and `--psi-values`. This tests their Cartesian product.

`--theta 4` gives 16 phi values and 16 psi values (0..15), normalized by 16,
as in the book. It quadruples the grid size: the full default five-image run
would have 15,360 configurations at one realization. Other settings remain
unchanged, including A <= 15. Use a new output folder when changing the grid,
resolution, seed, repeats or any scientific setting.

## Workers, memory and resume

One worker runs one complete image/A/noise-realization/phi/psi configuration:
load crop, add paired noise, predict sequentially, encode, check and save.
`--workers 12` allows up to 12 independent configurations concurrently.
There are no nested predictor workers and no independently compressed tiles.
Each worker is configured for one native numerical-library thread; the OS
schedules processes on the available CPUs.

CPU affinity/Slurm allocation and available RAM can reduce concurrency.
Low-memory hybrid encoding is enabled by default. The original predictor
and numeric encoding operations are retained, while large text trace arrays
are omitted. A source-compatibility guard protects the compact encoder.

Memory increases roughly with concurrent jobs. Conservative low-memory
planning estimates at P = 3 and 120 bands are approximately 1.26 GiB per
128 x 128 job and 3.55 GiB per 256 x 256 job, plus coordinator overhead.
The default total budget is 90 GiB, further capped by currently available
system/container RAM. `--total-memory-gb 60` lowers the planning budget.
These are estimates rather than hard OS-enforced limits.

Completed configurations are checkpointed. Repeat the same command/output
directory to resume; worker count and RAM budgets may change. Changing the
grid, source, crop, profile, noise settings, storage mode, model or code
requires a new output folder. Ctrl+C preserves completed checkpoints and
stops active jobs. Incomplete runs return a nonzero exit status.

## Saved files and interpretation

- `<image>_A<value>_fig413.png` / `.pdf`: noise-level heatmaps.
- `all_images_no_added_noise.png` / `.pdf`: overview if sigma = 0 is selected.
- `best_pairs.csv`: best tested pair(s), rate, A, sigma, profile and completeness.
- `heatmap_cells.csv`: mean rates, expansion, variability and completion per cell.
- `results.csv`: every individual encoder run, hashes, errors and noise statistics.
- `metadata.json`, `plan.csv`, `adaptation_progression.csv`, `jobs/`: reproducible
  settings, verified schedules, exact headers, checkpoints and per-job logs.
- `skipped.csv`, `failures.csv`, `last_execution.json`: execution diagnostics.

The star marks one best tested pair. All exact ties are listed as `phi:psi`
in the CSV; a tie is not hidden by calling one pair uniquely optimal.
The `full_legal_grid` flag distinguishes a complete legal search from a
user-selected subset. All figures use one common color scale, but each
image/A/sigma panel has its own size minimum.

Only cells with all requested realizations contribute to the heatmap.
Incomplete cells are gray. If any tested cell is missing, the panel and its
minimum are explicitly marked provisional. Gray cells are never filled by
interpolation or mistaken for 0% expansion.

The metric includes the header and embedded representative tables, entropy
payload/tail and output-word padding. External initialization is excluded
from the plotted rate; an inclusive rate is also saved. Error checking uses
encoder-side reconstruction, not an independent decoder.

Runtime and memory are recorded but **not plotted**. Python resource use is
not an FPGA resource/throughput measurement. This noise study is a controlled
sensitivity test; its best pairs should subsequently be checked on other
crops and real captures before choosing an operational compression profile.

To regenerate figures without loading images or running compression:

```bash
python tools/fig413/fig413_sweep.py --plot-only --output results/fig413_lisbon_A15_128
```

The same five original HYPSO filenames as the preceding packages are used.
`--images lisbon doha` selects a subset; `--files PATH1.nc PATH2.nc` supports
explicit alternatives. `--input-dir` searches direct matches first, then
unambiguous subdirectories. See `VALIDATION.md` for the checks performed.
