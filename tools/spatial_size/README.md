# HYPSO spatial-size sensitivity study

This package measures how a fixed predictor profile behaves on nested spatial
crops and on an entire capture. It records compressed bits/sample, runtime,
measured peak resident memory, and the actual predictor update-exponent schedule.

## Install

Extract the ZIP into your verification repository root. The entry point is:

`tools/spatial_size/spatial_size_sweep.py`

Keep its three helper files in the same folder. They are unchanged copies from
your uploaded `tools.zip`. Existing scripts and old result folders are not edited.
Use the Python environment that already runs your sweeps (NumPy, bitarray,
netCDF4, matplotlib, psutil, and the NTNU model). Validation used Python 3.12,
NumPy 1.26.4, bitarray 2.8.2, and psutil 7.2.2 on Linux.

Invoke it with **python**, rather than attempting to execute the .py file directly.
From the repository root on Linux/SSH, if the environment is named `.venv`:

```bash
source .venv/bin/activate
```

If your environment has another name, activate that environment instead. On
Windows PowerShell replace `python` below with `.\.venv\Scripts\python.exe`.

## Fixed profile and variable sizes

| Parameter | Default |
|---|---|
| Images | Lisbon, Bangladesh, Santarem, Doha, Finnmark |
| Input | Native integer `products/dn`, all bands, unsigned D=16 |
| Crop sizes | 128x128, 256x256, 512x512, full native image |
| Crop placement | Centre, nested within the same scene |
| Prediction bands P | 3 |
| Mode | Reduced |
| Sum | Narrow neighbour-oriented |
| Initial parameter nu_min | -6 |
| Final parameter nu_max | 4 |
| Change interval t_inc | 128, selected from the terminal |
| Absolute error limit A | 0 (lossless) |
| Damping phi | 0 (explicit choice from your recent shortlist; change with --phi) |
| Sample-representative resolution Theta | 3 |
| Offset psi | 0 for lossless; defaults to 7 if A>0 |
| Weight resolution / register | Omega=19 / R=64 |
| Encoder | Hybrid, same model defaults as your previous runner |
| Storage | Low-memory hybrid adapter enabled |
| Workers | 1 |

Apart from dimensions and any requested t_inc comparisons, all codec settings
remain fixed. Each configuration starts with fresh predictor and encoder state.
`full` means the entire native Y x X x bands array **as one image**. No tiling,
band removal, downsampling, calibration or binning is introduced. An invalid
square crop is skipped explicitly; it is never resized to fit.

## Run one interval on the five images

Check input paths, dimensions and memory estimates first:

```bash
python tools/spatial_size/spatial_size_sweep.py --input-dir data --sizes 128 256 512 full --t-inc 128 --workers 1 --total-memory-gb 90 --output results/spatial_size_t128 --dry-run
```

Then execute the same command without `--dry-run`:

```bash
python tools/spatial_size/spatial_size_sweep.py --input-dir data --sizes 128 256 512 full --t-inc 128 --workers 1 --total-memory-gb 90 --output results/spatial_size_t128
```

This requests **20 jobs**, assuming all four sizes fit all five captures.

For a first run on only Lisbon, add `--images lisbon` and use its own output
folder. For only the full capture:

```bash
python tools/spatial_size/spatial_size_sweep.py --input-dir data --images lisbon --sizes full --t-inc 128 --workers 1 --output results/lisbon_full_t128
```

## Compare intervals automatically

```bash
python tools/spatial_size/spatial_size_sweep.py --input-dir data --sizes 128 256 512 full --t-inc-values 128 512 2048 --workers 1 --total-memory-gb 90 --output results/spatial_size_tinc_comparison
```

This requests **60 jobs**. Each interval becomes a curve in the size-performance
plots. Use either `--t-inc VALUE` or `--t-inc-values VALUES`, not both. Intervals
must be powers of two from 16 to 2048. They are literal sample intervals, not
the encoded header exponents. There is no automatic scaling of t_inc with size.

If you instead run separate commands with `--t-inc 512` and `--t-inc 2048`, use
separate output folders for those experiments. A changed scientific grid cannot
resume into an earlier grid's output folder.

## Small local workflow test

```bash
python tools/spatial_size/spatial_size_sweep.py --input-dir data --images lisbon --sizes 8 16 --t-inc-values 16 128 --workers 1 --output results/spatial_size_smoke
```

This is four configurations, with no full-image job. Its purpose is to check the
workflow; these small crops often cannot reach the requested final adaptation
parameter. Use full-size results to address the actual scientific question.

Explicit NetCDF paths are also supported:

```bash
python tools/spatial_size/spatial_size_sweep.py --files data/my_capture.nc --sizes 128 full --t-inc 128 --output results/my_capture_size
```

The variable must be a 3-D integer DN array. `--variable` and `--axis-order` can
override its path and dimension interpretation. The default five-scene loader
uses original capture filenames, with direct matches preferred over recursive
matches. Duplicate ambiguous recursive matches are rejected.

## Outputs

- `results.csv`: one successful run per row, including crop coordinates/hash,
  exact dimensions, t_inc, rate, header/payload/padding lengths, reconstruction
  errors, timing fields, measured memory, and adaptation statistics.
- `<scene>_size_sensitivity.png/.pdf`: four panels: bits/sample, predictor plus
  encoder seconds, measured peak worker RAM, and percentage of updates made at
  nu_max. The horizontal positions use spatial pixel count; tick labels show
  dimensions. Stars identify full images. One curve per t_inc.
- `<scene>_adaptation_tinc<T>.png/.pdf`: applied adaptation parameter versus
  spatial raster index (left) and percentage through the raster (right). One
  curve per size. Hollow endpoints indicate nu_max was never reached.
- `adaptation_progression.csv`: compact constant-value intervals for the
  actual verified schedule; includes nu and rho, start/end indices and counts.
- `metadata.json`: input hashes/shapes, fixed parameters, experiment grid,
  model/script hashes and software/platform provenance.
- `plan.csv`: all requested jobs and estimated memory, including invalid sizes.
- `skipped.csv`: sizes that did not fit the capture or current RAM budget.
- `failures.csv`: failed worker jobs with locations of their logs.
- `jobs/*.json`: successful per-run records and complete header snapshots.
  `jobs/*.log` contains model output, and `*.spec.json` records child inputs.
- `last_execution.json`: invocation counts and sampled aggregate resident RAM.

The script stores summary measurements and compact schedule intervals, not
per-sample trace CSVs, full reconstruction arrays, or compressed binary files.

## Runtime and memory measurements

Each job executes in a **new Python process**, including when workers=1. Thus
the memory high-water mark belongs to that job instead of a previous, larger job.
The coordinator does not retain image cubes; a worker loads only its own crop.

- `predictor_seconds` and `encoder_seconds`: model stage timings. The added
  exponent-verification scan is subtracted from the predictor timing.
- `model_seconds`: predictor + encoder time, used in the overview plot.
- `data_load_seconds`: reading and hashing the selected DN array.
- `codec_and_checks_seconds`: encoding plus bound checks, hashing and cleanup.
- `wall_seconds`: parent-observed elapsed time from child launch to completion;
  includes interpreter/import startup, data loading, codec, checks and polling
  latency. It excludes initial parent planning and final plotting.
- `peak_rss_gib`: measured maximum **resident memory of that fresh worker**,
  including input, imports, model arrays and correctness checks. Linux/macOS use
  OS `ru_maxrss`; Windows uses its available peak-working-set/peak-RSS field.
  If an OS high-water API is unavailable, the field is labelled as a sampled
  lower bound instead. NumPy allocations are included.
- `sampled_peak_rss_bytes`: independent parent RSS sampling, default every 50ms.
- `last_execution.json` also records the sampled sum of parent and active worker
  RSS. Shared pages may be counted more than once; this is not total machine RAM.

Use **workers=1** for the cleanest comparison. `--workers N` supports concurrent
fresh processes, subject to CPU and memory estimates, but bandwidth/CPU contention
affects timing. Requested and peak concurrent worker counts are recorded per row.
Avoid interpreting Python timings or memory as FPGA throughput/resources.

`--repeats 3` runs each combination three times in separate processes. Overview
plots show medians with min/max whiskers. Repeats describe timing variability;
they are not independent scenes. A single run is usually sufficient for bitrate.

Memory API references:
- https://docs.python.org/3/library/resource.html
- https://psutil.readthedocs.io/stable/

## Full-image feasibility and 120 GB RAM

The default per-job and total estimate ceilings are **90 GiB**, leaving headroom
on your 120 GB machine. Both can be changed with `--max-memory-gb` and
`--total-memory-gb`. The scheduler also checks currently available memory and
Linux cgroup limits. These are conservative planning checks, **not enforced OS
memory limits**; other workloads can change availability during a job.

For the locally available Adelaide capture, 598x1092x120, the estimate with P=3
and low-memory storage is approximately **30.86 GiB per full-image job**. Estimates
for 128, 256 and 512 square crops are approximately 1.26, 3.55 and 12.69 GiB.
These are estimates, not measured full-image peaks. Your files are inspected
individually and may have different shapes.

The compact hybrid adapter removes eight Unicode diagnostic arrays, while
retaining numerical state and bitstream generation. Predictor arrays remain.
The original encoder would have a much larger memory footprint. To compare that
storage implementation, use `--standard-memory` and a new output folder; the
planning guard will still apply. Keep storage mode fixed within a size study.

If the adapter's source-identity check rejects your upstream encoder version,
do not bypass it. Validate an updated adapter or use standard mode for sizes
that fit memory.

Oversized or memory-blocked jobs are recorded and skipped, so feasible smaller
cases can finish. Running the same command later reconsiders skipped/failed jobs.
Raising a RAM budget or changing worker count does not invalidate completed rows.

## Adaptation progression and interpretation

The model uses, for the weight update at spatial index t:

```
rho[t-1] = clip(nu_min + floor(((t-1) - Nx) / t_inc), nu_min, nu_max) + D - Omega
```

The index is spatial **per band**, not multiplied by the band count. Updates
occur at t=2..Nx*Ny-1; the first two spatial positions are initialization.
The script checks every used exponent in the model array against the saved
piecewise progression. It records when nu_max is first used and the fraction of
updates at that value. This is a verified **schedule**, not a direct measurement
that prediction weights or compression performance have converged.

For your default nu_min=-6, nu_max=4, at 128x128:

| t_inc | Final applied nu | Fraction of updates at nu_max |
|---|---:|---:|
| 128 | 4 | about 91.4% |
| 512 | 4 | about 68.0% |
| 2048 | 1 | 0% |

Larger crops allow more time at the final setting, but also alter the image
content, boundary conditions and header overhead. Their width Nx also appears
in the schedule. Therefore rate changes cannot be attributed solely to adaptation.
The centre crops are nested and their exact coordinates are recorded. You may
choose `--anchor top-left` as a separate study with a common starting origin.

Compare rate and adaptation coverage together. A poor small-crop rate with
t_inc=2048 does not establish poor steady-state performance at nu_max=4 because
the crop never reaches that final value. Conversely, reaching nu_max does not
prove that prediction weights have converged.

Primary bitrate matches your previous sweeps: header + payload/tail + word padding
per scalar sample. External initialization is excluded, with an inclusive column
also saved. Errors use encoder-side reconstruction. Lossless runs must have
maximum error zero. An independent decoder remains a separate validation step.

## Resume and regenerate plots

Repeat the same scientific command to resume. Every completed job is saved
atomically. Ctrl+C terminates active children; completed jobs remain and in-flight
jobs are retried. One coordinator owns each output folder. Changed input data,
settings, grid, storage mode or code require a new folder; no overwrite is implicit.

```bash
python tools/spatial_size/spatial_size_sweep.py --plot-only --output results/spatial_size_t128
```

This rebuilds CSVs/plots from successful job records without the NetCDF files or
codec repository. Do not delete `jobs/*.json` if you need to regenerate results.

See `VALIDATION.md` for the tests and the limits of local execution.
