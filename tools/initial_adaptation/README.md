# HYPSO initial adaptation study

This follows the damping-versus-final-adaptation study. The next question is:
**Do the promising phi=0/1 settings stay good when the predictor's initial
adaptation rate and transition interval change?**

The supplied plot supports shortlisting phi=0 and phi=1, with a broad good region
around nu_max=2..4 for several scenes. Lisbon also warrants retaining nu_max=0.
Exact winners cannot be read reliably from almost overlapping plotted curves;
this package calculates summaries from its CSV instead.

The design follows CCSDS 120.2-G-2 section 4.2.2, particularly Figure 4-8 and the
damping/initial-adaptation interactions in Figures 4-10 and 4-11. It is a targeted
HYPSO follow-up, not a reproduction of the book's numerical results.

## Installation

Extract the ZIP **into the verification repository root**. The resulting entry
point is `tools/initial_adaptation/initial_adaptation_sweep.py`. Keep all four
Python files together. `_damping_core.py` is the unchanged engine from the earlier
Figure 4-9 package; `sweep_workers.py` and `hybrid_no_traces.py` are also unchanged.
Do not run the private engine as the entry point.

Use the Python environment that already runs your previous sweeps. Requirements
are the verification model dependencies plus NumPy, netCDF4, matplotlib and
psutil. The package does not replace the model, alter previous scripts, or install
dependencies. Python 3.11/3.12 with the model's pinned NumPy/bitarray versions is
the intended environment; validation used Python 3.12.

## Default experiment

| Property | Value |
|---|---|
| Images | Lisbon, Bangladesh, Santarem, Doha, Finnmark |
| Crop | Centre 128 x 128; all native bands (120 for these captures) |
| Input | Unscaled integer `products/dn`, unsigned D=16 |
| P | 3 |
| Local sum / prediction mode | Narrow neighbour / reduced |
| Encoder | Hybrid |
| A | 0, 3, 15 DN |
| Damping phi | 0, 1 |
| Final parameter nu_max | 0, 2, 4 |
| Initial parameter nu_min | -6, 0 |
| Change interval t_inc | 16, 128, 1024 samples per band |
| Representative resolution Theta | 3 |
| Representative offset psi | 7 for A>0; 0 for A=0 |
| Weight resolution / register | Omega=19 / R=64 |
| Weight initialization / exponent offsets | Model default / all zero |
| Other encoder settings | Same model defaults as the earlier Figure 4-9 runner; recorded in metadata |
| Workers | 2 |

This is 5 x 3 x 2 x 3 x 2 x 3 = **540 full configurations**, 108 per image.
No band removal, calibration, binning, spatial tiling or warm starts are applied.
The baseline schedule is nu_min=-6, t_inc=128 at each SAME phi, nu_max, A and crop.

nu_min controls the initial update scaling; t_inc controls how many spatial
samples pass before the scaling increases by one, until nu_max is reached.
These are predictor parameters, not CPU speed settings. With nu_min=nu_max the
schedule is constant, so changing t_inc has no effect. Some grid entries are
therefore intentional repeated schedules, useful for seeing this invariance.

## PowerShell: small local test first

Run from the repository root. Each command is a single line.

```powershell
.\.venv\Scripts\python.exe tools/initial_adaptation/initial_adaptation_sweep.py --input-dir data --images lisbon --crop 16 16 --a-values 0 3 --vmax-values 2 --t-inc-values 16 128 --workers 2 --low-memory --output results/initial_adaptation_smoke
```

This executes **16 configurations**. It tests the workflow; it cannot select the
best adaptation profile because several schedules do not reach nu_max within a
16x16 crop. Such results have hollow markers and flags in the CSV.

## PowerShell: full default study

First validate paths, configurations and estimated memory without encoding:

```powershell
.\.venv\Scripts\python.exe tools/initial_adaptation/initial_adaptation_sweep.py --input-dir data --workers 2 --low-memory --output results/initial_adaptation_128 --dry-run
```

Then run:

```powershell
.\.venv\Scripts\python.exe tools/initial_adaptation/initial_adaptation_sweep.py --input-dir data --workers 2 --low-memory --output results/initial_adaptation_128
```

On Linux/SSH, replace `.\.venv\Scripts\python.exe` with the Python executable in
your Linux environment, for example `python` after activating that environment.
An optional `--total-memory-gb 90` can cap the **estimated aggregate budget** on
your 120 GB machine; it is not an operating-system RAM reservation or limit.
`--workers 4` is supported if the estimated memory and allocated CPU count allow it.

## Preserve the exact previous crop

The default repeats the centre-128 convention used by the prior damping script.
For stronger verification, add `--crop-from` pointing to the **metadata.json from
your actual damping run**, instead of setting `--crop`, `--row` or `--col`.
For example, only if that was your previous output folder:

```powershell
.\.venv\Scripts\python.exe tools/initial_adaptation/initial_adaptation_sweep.py --input-dir data --crop-from results/damping_fig49_128/metadata.json --workers 2 --low-memory --output results/initial_adaptation_128
```

This reuses each scene's origin, size, variable and axis order, then verifies its
DN hash. It also checks recorded fixed profile/header settings and model hashes
when available. A mismatch stops the run rather than silently changing the study.
It does not read the old numerical results or reuse old completed jobs.

## Change the grid

All range options accept space-separated lists, not a maximum. For example:

```powershell
.\.venv\Scripts\python.exe tools/initial_adaptation/initial_adaptation_sweep.py --input-dir data --nu-min-values -6 -3 0 --t-inc-values 32 128 512 --vmax-values 2 3 4 --phi-values 0 1 --a-values 0 3 15 --workers 2 --low-memory --output results/initial_adaptation_refined
```

Every nu_min must be <= every nu_max in a requested rectangular grid. Every
t_inc must be a power of two in 16..2048. Invalid combinations are rejected.
If the baseline pair (-6,128) is omitted, only absolute-rate plots are generated.
Use a new output folder for each changed grid, crop, fixed setting or model.

## Outputs and interpretation

- `results.csv`: one row per configuration, saved atomically after every job.
  Includes rates, header/payload/padding sizes, reconstruction error, payload
  hash, timings and adaptation-schedule reachability.
- `metadata.json`: exact crops/hashes, all fixed header fields, grid, software
  hashes and worker settings. Header fields can use the standard's encoded
  representations; the `profile` and CSV expose physical parameter values.
- `all_images_initial_phi0.png/.pdf` and `...phi1...`: rows are scenes; columns
  are A; x is nu_max; the six curves are nu_min/t_inc pairs.
- `all_images_change_phi0.png/.pdf` and `...phi1...`: same layout, but y is
  percentage rate change from (-6,128) at the SAME scene, A, phi and nu_max.
  Negative means fewer compressed bits. Missing baselines produce gaps.
- `<scene>_initial_phi0.png/.pdf` and `...phi1...`: individual scene figures.
- `best_per_scene.csv`: the minimum observed rate for each scene/A combination.
  Exact ties are resolved deterministically by the parameter tuple; an apparent
  tiny advantage does not establish practical superiority.
- `profile_ranking.csv`: common profiles ranked **separately for each A** by
  worst-case and then mean percentage rate penalty relative to each scene's
  best tested configuration. Includes unweighted mean rate, sample-weighted
  pooled rate, and completeness flags.

Do not compare partial rankings as final evidence. Require `all_images_present=1`
and `full_grid_complete=1`. Keep two or three consistently good profiles rather
than selecting a configuration solely because it wins one crop. Look for broad
good regions. A later refinement can add nu_max=1/3 and intermediate schedules.

The primary bitrate matches the previous script: header + entropy payload/tail +
output-word padding, divided by scalar sample count. External initialization is
excluded; a separate inclusive-rate column is recorded. Rates are deterministic;
timings vary with load. This is not a model of measured FPGA performance.

The maximum-absolute-error and RMSE checks use encoder-side quantizer-bin-centre
reconstruction, including correction of the model's first spatial sample entry.
They verify the configured bound but do not replace an independent decoder test
or downstream scientific-quality validation.

## Resume, plotting and memory

Repeat the **same command** to resume; completed rows are retained. Changing only
worker count or RAM budgets is allowed. Ctrl+C preserves completed rows; an
in-flight configuration must run again. The coordinator is the only CSV writer,
and one process runs each complete configuration. Concurrent runs cannot share
an output folder. Plots are generated at completion, on Ctrl+C, or with:

```powershell
.\.venv\Scripts\python.exe tools/initial_adaptation/initial_adaptation_sweep.py --plot-only --output results/initial_adaptation_128
```

`--overwrite` explicitly restarts an existing output from **this tool**; it refuses
to overwrite a different study. Old damping/P-sweep output folders are preserved.

`--low-memory` discards eight hybrid Unicode diagnostic arrays while keeping
numeric state and the bitstream. It does not remove predictor arrays or change
the predictor equations. The adapter checks the supported upstream class source
and refuses mismatched code. If that check fails, use the standard mode with an
adequate memory budget, or validate an updated adapter; do not bypass the guard.
The script stores summary CSVs, not per-sample trace dumps or compressed files.

At default 128x128x120/P=3, its conservative estimate is about 1.26 GiB per
configuration in low-memory mode, plus coordinator/input overhead. RAM grows
roughly with the number of simultaneous configurations. Estimates are not measured
peak RAM and should retain headroom.

See `VALIDATION.md` for the tests performed and their limits.
