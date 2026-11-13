# HYPSO sample-representative offset / damping study

This is a **Figure 4-12-style** study from CCSDS 120.2-G-2. It replaces the
previous t_inc sweep with a sweep of the sample-representative offset **psi**.
The x-axis is psi = 0..7, the y-axis is compressed bits/sample, each curve is
a damping value phi, and each panel represents an image and absolute error A.
The combined default figure has five rows and two columns: **A = 3 and 15 DN**.

Lossless A = 0 is omitted because psi has no effect on lossless compression.
This driver accepts A = 1..15 only. It never runs A = 20.

## Why these settings

The supplied 128 x 128 t_inc/phi plots suggest **t_inc = 512** as a useful
common compromise for the next controlled experiment:

- Lisbon, Bangladesh and Santarem generally benefit from larger intervals.
- Doha is near its best region around 512--1024.
- Finnmark is generally best near 256 and worsens at larger intervals,
  particularly at 1024--2048.
- Phi = 3 and 4 are strong candidates across the near-lossless panels.

This is a visual interpretation of the supplied figure, not a numerical
optimization from its CSV. There is no demonstrated universal optimum.
The script holds one common interval fixed across all images and offsets,
so differences along each curve isolate the effect of psi for that profile.
`--t-inc` remains adjustable for a later sensitivity check.

Keep **all six phi curves (0..5)** initially. The preceding damping ranking
was measured at psi = 7; changing psi can change which damping value works
best. A faster focused study can use phi = 3 and 4, but it does not replace
the full interaction comparison.

The Green Book discussion immediately before Figure 4-12 states that offset
has no effect in lossless mode and recommends psi = 2^Theta - 1 as a reasonable
default when damping is appropriately chosen. With Theta = 3 this is **7**.
Here 7 is the comparison baseline, not an assumed measured optimum for HYPSO.

Larger t_inc means more spatial samples between increases in the weight-update
scaling exponent. It delays the reduction in weight-update magnitude and
keeps the faster initial adaptation active longer. It does not mean that
predictor weights are updated only once every t_inc samples.

## Default profile

| Parameter | Setting |
| --- | --- |
| Images | Lisbon, Bangladesh, Santarem, Doha, Finnmark |
| Crop | 128 x 128, centred; adjustable |
| Bands | All native bands, normally 120 |
| Input | Raw integer `products/dn`; no calibration or wavelength filtering |
| P | 3 |
| Prediction mode | Reduced |
| Local sum | Narrow neighbour-oriented |
| Encoder | Hybrid |
| D | 16 bits |
| R | 64 bits |
| Omega | 19 |
| Theta | 3 |
| nu_min | -1 |
| nu_max | **7 - phi**, as in the preceding study |
| t_inc | **512**, fixed throughout a run |
| A | **3, 15 DN** |
| phi curves | **0, 1, 2, 3, 4, 5** |
| psi sweep | **0, 1, 2, 3, 4, 5, 6, 7** |

| phi | 0 | 1 | 2 | 3 | 4 | 5 |
| --- | --- | --- | --- | --- | --- | --- |
| nu_max | 7 | 6 | 5 | 4 | 3 | 2 |

Default job count: **5 images x 2 A values x 6 phi values x 8 psi values = 480**.
All other predictor and hybrid-encoder settings are held fixed at the same
values as the previous package; default weight initialization and zero
weight exponent offsets are retained. Exact header snapshots are saved.

Within each psi curve, phi and its corresponding nu_max are fixed. Across
phi curves, nu_max changes with phi. These are comparisons of paired
(phi, nu_max) settings, not a claim that every non-phi setting is constant.

Psi is a sample-representative parameter, not a radiometric correction or
a constant DN value added to the image. The selected A remains the allowed
absolute reconstruction-error bound for every offset value.

## Install

Extract the ZIP into the verification-model repository root. It creates
`tools/offset_phi/`. Keep all five Python files together. The executable is
**offset_phi_sweep.py**; the other files are private helpers. Existing sweep
files and upstream model files are not replaced.

Use the Python environment that already runs your model. If necessary:

```powershell
.\.venv\Scripts\python.exe -m pip install matplotlib netCDF4 psutil
```

The existing model also requires NumPy and bitarray. The supplied code was
tested with Python 3.12, NumPy 1.26.4 and bitarray 2.8.2; an environment
upgrade is not required. On Linux/SSH, use `python` from the activated .venv
instead of the Windows interpreter path below.

## First run: 128 x 128

From the repository root:

```powershell
.\.venv\Scripts\python.exe tools/offset_phi/offset_phi_sweep.py --input-dir data --crop 128 128 --t-inc 512 --workers 12 --output results/offset_phi_128_tinc512
```

Append `--dry-run` to verify input files, grid size, crop positions and
estimated memory without running compression or writing results.

For a quick functional check (eight configurations):

```powershell
.\.venv\Scripts\python.exe tools/offset_phi/offset_phi_sweep.py --input-dir data --images lisbon --crop 16 16 --phi-values 3 4 --psi-values 0 7 --a-values 3 15 --workers 2 --output results/offset_phi_smoke
```

Small-crop results only test the workflow. They are not substitutes for the
128/256 scientific comparison.

## Larger crop: 256 x 256

```powershell
.\.venv\Scripts\python.exe tools/offset_phi/offset_phi_sweep.py --input-dir data --crop 256 256 --t-inc 512 --workers 12 --output results/offset_phi_256_tinc512
```

Use a new output folder for a changed crop. All parameter combinations use
the same centre crop of each image. Crop origins and hashes are recorded;
larger centre crops are nested around the same centre but include more
spatial content. Crops larger than their sources are skipped explicitly,
never silently resized. `--anchor top-left` is available if needed.

## Focused and sensitivity runs

For a faster comparison of the strongest damping candidates, add
`--phi-values 3 4` and choose another output folder. This gives **160 jobs**
for the five images, two A values and all eight offsets.

```powershell
.\.venv\Scripts\python.exe tools/offset_phi/offset_phi_sweep.py --input-dir data --crop 128 128 --t-inc 512 --phi-values 3 4 --workers 12 --output results/offset_phi_128_focused
```

To study whether the preferred psi changes with adaptation, repeat selected
phi/psi candidates at `--t-inc 256` and/or `--t-inc 1024` in separate folders.
There is no need to rerun every configuration before inspecting this first
offset study. Custom offsets use `--psi-values 0 3 5 7`; each value must be
0..7 at Theta = 3. All A values must be 1..15; custom intervals must be powers
of two from 16 through 2048. `--nu-min` must not exceed any selected `7 - phi`.

## Workers and memory

A job is **one image crop + A + phi + psi**, with fixed t_inc and nu_min and
derived nu_max. Each worker loads the crop, runs the original sequential
predictor, runs the hybrid encoder, checks reconstruction and saves metrics.
When that process finishes, its slot becomes available for another job.

`--workers 12` allows up to twelve complete configurations to run in
parallel. Each has one native numerical-library thread. The OS schedules
the processes on CPUs; this is not physical-core pinning. There are no
nested predictor workers and no independent tiling of a single crop.
`--workers 1` produces the same scientific results serially.

Actual concurrency is limited by CPU affinity/Slurm allocation and RAM.
Each active job has its own codec arrays, so memory grows roughly with
the number of simultaneous jobs. The default total planning budget is
90 GiB and is further capped by currently available system/container RAM.
Use `--total-memory-gb 60`, for example, to impose a lower planning budget.
It is an estimate, not an enforced operating-system memory limit.

With P = 3 and 120 bands, conservative low-memory estimates are about
1.26 GiB per 128 x 128 job and 3.55 GiB per 256 x 256 job, plus coordinator
overhead. Twelve concurrent jobs therefore have estimated totals of about
15.1 and 42.6 GiB respectively. These are not measured peaks or guarantees.

Low-memory mode is enabled by default. It omits large hybrid-encoder text
trace arrays while keeping its numeric operations and compressed payload.
The original predictor is unchanged. A source-compatibility check protects
this adapter. If your model differs, use a compatible revision or explicitly
select `--standard-memory` with an appropriate budget and a new output
folder; do not bypass the compatibility check.

Jobs wait when active configurations occupy the available budget. A job
that cannot fit on its own is recorded in `skipped.csv` and can be retried
after increasing the budget. Incomplete runs return a nonzero exit status.

## Outputs and interpretation

- `all_images_offset_phi.png` / `.pdf`: combined image-by-A figure.
- `<image>_offset_phi.png` / `.pdf`: one figure per image.
- `results.csv`: all rates, applied parameters, error metrics and resource
  measurements. **No runtime or memory graphs are generated.**
- `best_psi_per_image_a_phi.csv`: lowest observed rate for each image/A/phi,
  with the gain against psi = 7 at that same phi.
- `best_phi_psi_per_image_a.csv`: lowest observed joint (phi, psi) choice for
  each image/A, at the fixed t_inc and prescribed nu_max coupling.
- `metadata.json`, `plan.csv`, `adaptation_progression.csv`, and `jobs/`:
  reproducible settings, crop/source/model/code hashes, verified adaptation
  schedules, completed checkpoints and logs.
- `last_execution.json`, `skipped.csv`, `failures.csv`: execution diagnostics.

Lower bits/sample is better at a fixed A. A positive
`gain_vs_psi7_bits_per_sample` means fewer bits/sample than psi = 7 at the
same image, A and phi; its percentage uses that psi = 7 rate as denominator.
For the joint summary, the reference also uses the winning phi, so it is an
offset gain within that phi, not a gain against the best old phi overall.
The baseline fields are blank if psi = 7 was not completed in that group.

Summaries include completed/expected counts. An incomplete group's minimum
is provisional. Exact rate ties are listed; the closest offset to the prior
psi = 7 baseline is selected for reporting, then the lower phi if needed.
Small differences on one crop do not establish a universal configuration.

**Hollow markers** mean the programmed nu_max was not reached before the
crop ended. At the default 128 x 128, t_inc = 512 and phi = 0..5, all caps
are reached. The check remains useful for smaller crops or custom intervals.
Reaching the cap does not demonstrate convergence of the predictor weights.

The plotted rate includes the CCSDS header, entropy payload/tail and
output-word padding per scalar sample. External initialization information
is excluded from that rate and included in an additional CSV rate. The
definition matches the previous t_inc study. Psi = 7 points should reproduce
its rates when the source, crop, model and other parameters match exactly.

No full trace CSVs or temporary shared-array files are generated.
Reconstruction checks are encoder-side, not an independent decoder.
Python memory/runtime measurements do not measure FPGA resource usage.

## Resume and replot

Repeat the same scientific command and output folder to resume. Changing
workers or memory budgets is allowed; finished jobs are not rerun. Changing
crop, grid, interval, profile, code, model, inputs or storage mode requires
a new output directory. Prior experiments are never silently overwritten.

Ctrl+C stops active processes and preserves completed configuration
checkpoints. Interrupted configurations restart on resume.

```powershell
.\.venv\Scripts\python.exe tools/offset_phi/offset_phi_sweep.py --plot-only --output results/offset_phi_128_tinc512
```

Plot-only needs the saved output and plotting environment, not input
imagery or model execution.

## Default input names

- `lisbon_2022-09-22T10-55-19Z-l1a.nc`
- `bangladesh_2022-10-26T04-02-56Z-l1a.nc`
- `santarem_2022-09-24T13-52-04Z-l1a.nc`
- `doha_2022-09-12T06-38-31Z-l1a.nc`
- `finnmark_2022-07-06T19-40-03Z-l1a.nc`

Use `--images lisbon doha` for a subset. Direct filename matches under
`--input-dir` take priority over recursive matches. Ambiguous subdirectory
copies require a more specific input directory. Other filenames can be
supplied with `--files PATH1.nc PATH2.nc` instead of `--images`.

See `VALIDATION.md` for release-specific checks and limitations.
