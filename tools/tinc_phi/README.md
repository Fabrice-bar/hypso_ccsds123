# HYPSO update-interval / damping study

This is a Figure 4-11-style study from CCSDS 120.2-G-2, using the five HYPSO
captures and **A = 0, 3, 15 DN**. It plots compressed bits/sample against
**log2(t_inc)**. Each image has one panel per A, with one curve per damping
value phi. The combined figure has five image rows and three A columns.

The crop defaults to **128 x 128** and is adjustable. Every configuration
uses the same centre crop of a given image and retains all native bands
(120 for these captures). Raw, unscaled integer `products/dn` is used;
there is no calibration, binning, or wavelength filtering in this study.

## Install

Extract the ZIP into the verification-model repository root. This adds
`tools/tinc_phi/` alongside your existing tools. Keep all five Python files
together; run **tinc_phi_sweep.py**. The other files are private helpers.
No existing sweep or upstream codec file is replaced.

Use the Python environment that already runs the verification model. If the
plotting/NetCDF packages are missing, install them into that same environment:

```powershell
.\.venv\Scripts\python.exe -m pip install matplotlib netCDF4 psutil
```

The model also needs its existing NumPy and bitarray dependencies. This
package was tested with Python 3.12, NumPy 1.26.4 and bitarray 2.8.2; it does
not require a Python upgrade. On Linux, replace the Windows interpreter path
in the commands below with `python` in your activated environment.

Run commands from the repository root, which contains `ccsds123_i2_hlm/`.
If the tool is elsewhere, provide `--repo PATH_TO_REPOSITORY`.

## Run 128 x 128 first

```powershell
.\.venv\Scripts\python.exe tools/tinc_phi/tinc_phi_sweep.py --input-dir data --crop 128 128 --workers 12 --output results/tinc_phi_128
```

All five images are selected by default: Lisbon, Bangladesh, Santarem,
Doha and Finnmark. This produces **5 x 3 x 6 x 8 = 720 configurations**.
Append `--dry-run` to check input files, the grid and estimated memory without
encoding. To use a subset, append e.g. `--images lisbon bangladesh doha`.

For a quick functional test before the 720-point run:

```powershell
.\.venv\Scripts\python.exe tools/tinc_phi/tinc_phi_sweep.py --input-dir data --images lisbon --crop 16 16 --phi-values 0 5 --a-values 0 15 --t-inc-values 16 128 --workers 2 --output results/tinc_phi_smoke
```

This tests eight configurations. Its tiny-crop rates are not a substitute for
the intended study.

## Then run 256 x 256

```powershell
.\.venv\Scripts\python.exe tools/tinc_phi/tinc_phi_sweep.py --input-dir data --crop 256 256 --workers 12 --output results/tinc_phi_256
```

Use a different output folder for a different crop. Centre crops of different
sizes are nested within each original image. `--anchor top-left` is also
available. An oversized crop is reported and skipped, never silently resized.
The native source dimensions, crop origin and crop hash are saved.

If you omit `--output`, the default includes the size:
`results/tinc_phi_128x128` or `results/tinc_phi_256x256`.

## Configuration

| Parameter | Setting |
| --- | --- |
| P | 3 |
| Prediction | Reduced |
| Local sum | Narrow neighbour-oriented |
| Entropy encoder | Hybrid |
| Input dynamic range D | 16 bits |
| Register R | 64 bits |
| Weight resolution Omega | 19 |
| Sample-representative resolution Theta | 3 |
| Offset psi | 7 for A > 0; 0 for lossless, as required by the model |
| Initial adaptation nu_min | -1 |
| Final adaptation nu_max | **7 - phi for each configuration** |
| Absolute error A | **0, 3, 15 DN** |
| Damping phi | 0, 1, 2, 3, 4, 5 |
| Update interval t_inc | 16, 32, 64, 128, 256, 512, 1024, 2048 |

| phi | 0 | 1 | 2 | 3 | 4 | 5 |
| --- | --- | --- | --- | --- | --- | --- |
| nu_max | 7 | 6 | 5 | 4 | 3 | 2 |

The coupling follows the Green Book choice
`nu_max = 7 - 2**(3 - Theta) * phi`, which is `7 - phi` at Theta = 3.
The script deliberately has no independent `--nu-max` option. This is a
coupled phi/nu_max experiment: do not interpret differences between curves
as the isolated effect of phi while every other parameter stays fixed.

All other settings stay fixed across the sweep, including the hybrid
encoder's reference defaults, default predictor weight initialization and
zero weight exponent offsets. Full header snapshots are saved for audit.
The lossless fidelity mode and its required zero psi are the documented
exception between A = 0 and A > 0.

Custom grids use literal values, for example:

```powershell
.\.venv\Scripts\python.exe tools/tinc_phi/tinc_phi_sweep.py --input-dir data --crop 128 128 --t-inc-values 32 128 512 2048 --phi-values 0 1 3 5 --a-values 0 3 15 --workers 4 --output results/tinc_phi_custom
```

Every A must be between **0 and 15**, including custom values. A = 20 is
rejected. Update intervals must be powers of two between 16 and 2048.
Phi may be 0..7 for Theta = 3; its default grid remains 0..5. `--nu-min`
can be changed, provided it does not exceed any selected `7 - phi`.
The x-axis uses exponents 4..11; the terminal accepts intervals, not exponents.

## What the workers do

One job is **one image crop + one A + one phi + one t_inc**. Its nu_max is
derived from phi. A worker process loads that crop, runs the **original
sequential predictor**, runs the hybrid encoder, checks reconstruction, and
returns the measured rate. It then exits; the coordinator launches another
configuration when a slot is available.

`--workers 12` allows up to 12 complete configurations at once. Each worker
uses one native numerical-library thread. The operating system schedules
these processes on the available CPUs; this is not physical-core pinning.
There is **no parallel predictor inside a configuration**, no nested worker
pool, and no partitioning of the image into independent compression tiles.
Increasing workers accelerates the overall sweep, not one isolated
configuration. `--workers 1` gives the same scientific results serially.

Actual concurrency is capped by CPU affinity/Slurm allocation and available
RAM. Each active worker has its own codec arrays, so memory grows roughly
with the number of concurrent jobs. The coordinator keeps only metadata.

Low-memory mode is on by default. It omits the hybrid encoder's large text
trace arrays while retaining its numerical coding operations and payload.
The original predictor is unchanged. Compatibility with the reference
encoder source is checked before using this adapter. If that check fails on
a different model version, do not bypass it; use a compatible model or
`--standard-memory` with an appropriate RAM budget and a new output folder.

For P = 3 and 120 bands, conservative planning estimates are approximately:

| Crop | Per active configuration | Twelve active configurations |
| --- | ---: | ---: |
| 128 x 128 | 1.26 GiB | 15.1 GiB |
| 256 x 256 | 3.55 GiB | 42.6 GiB |

These are estimates, not measured peaks or guaranteed OS limits. There is
also coordinator/plotting overhead. The default total budget is 90 GiB,
further limited by currently available system/container memory. You can
lower it, e.g. `--total-memory-gb 60`. When active jobs reserve too much RAM
for another job, the remaining jobs wait and run later. A job that cannot
fit even on its own is listed in `skipped.csv`. An incomplete run returns
a nonzero exit status. Increasing the RAM budget and repeating the same
command retries unfinished jobs.

## Reading the plots

Lower bits/sample means better compression for the chosen A. Compare
intervals within a curve, and inspect whether the preferred interval
changes across A or phi. Use the same crop and rate definition throughout.

**Hollow markers** mean the programmed adaptation parameter never reached
its requested nu_max before the crop ended. The point is still a valid
whole-crop compression measurement, but it is not evidence of sustained
operation at that final parameter. Reaching the cap also does not prove
that the learned predictor weights have converged.

The adaptation clock counts spatial samples **per band**, not all 120 bands
combined. Actual weight updates at spatial index `t >= 2` use `rho[t-1]`.
For a 128 x 128 crop, phi = 0, nu_min = -1 and t_inc = 2048, the first
possible update using nu_max = 7 would be at spatial index 16513; the crop
ends at index 16383. This point is hollow. A 256 x 256 crop reaches that cap.
The script verifies the applied schedule against the actual predictor arrays.

Bits/sample includes the CCSDS header, entropy payload and tail, and
output-word padding, divided by the number of scalar samples. External
initialization information is excluded from the plotted rate and included
in a separate CSV rate. Compare like-for-like definitions between studies.

## Saved files and resume

- `all_images_tinc_phi.png` and `.pdf`: combined image-by-A plot.
- `<image>_tinc_phi.png` and `.pdf`: one figure per input image.
- `results.csv`: rate, error, applied parameters, adaptation status,
  crop hash, per-job runtime and process peak memory.
- `best_tinc_per_image_a_phi.csv`: lowest observed rate per image/A/phi,
  with completed/expected grid counts. These are crop-specific minima.
- `adaptation_progression.csv`: verified adaptation segments for each job.
- `plan.csv`, `metadata.json`: configuration grid, source/model/code hashes.
- `jobs/`: completed checkpoints, exact headers and per-configuration logs.
- `last_execution.json`: concurrency and resource measurements; `skipped.csv`
  and `failures.csv` explain unfinished jobs.

**Runtime and memory are not plotted.** No full trace CSVs or temporary
shared-array files are generated. Completed runs are saved atomically.

Repeat exactly the same scientific command/output directory to resume. You
may change `--workers` and RAM budgets without rerunning finished points.
Changing crop, grid, fixed parameters, code, model, storage mode or inputs
requires a new output directory. The script never silently overwrites a
different experiment. Ctrl+C stops active processes and preserves completed
configuration checkpoints; interrupted configurations restart on resume.

To regenerate figures without loading the imagery or running compression:

```powershell
.\.venv\Scripts\python.exe tools/tinc_phi/tinc_phi_sweep.py --plot-only --output results/tinc_phi_128
```

## Input filenames

The default scene selector looks for these filenames directly under `data/`,
then in subdirectories if no direct match exists:

- `lisbon_2022-09-22T10-55-19Z-l1a.nc`
- `bangladesh_2022-10-26T04-02-56Z-l1a.nc`
- `santarem_2022-09-24T13-52-04Z-l1a.nc`
- `doha_2022-09-12T06-38-31Z-l1a.nc`
- `finnmark_2022-07-06T19-40-03Z-l1a.nc`

Ambiguous copies in subdirectories require a more specific `--input-dir`.
For other filenames, use `--files PATH1.nc PATH2.nc` instead of `--images`.
All selected captures retain their native spectral bands.

Reconstruction checks are encoder-side, not an independent decoder test.
Python timings and RAM do not measure FPGA throughput or hardware storage.
See `VALIDATION.md` for the checks performed on this release.
