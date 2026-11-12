# HYPSO nu_min / damping study

Plot **compressed bits/sample versus the initial adaptation parameter nu_min**,
with one curve per sample-representative damping value phi. This follows the
type of comparison in CCSDS 120.2-G-2 Figure 4-10, using your HYPSO scenes and
chosen fixed profile. Here `phi` means the damping parameter, written phi or
the Greek letter φ (the “ø” in your request).

## Install and run

Extract the ZIP into the verification-model repository root. It adds a separate
`tools/vmin_phi/` folder. Keep its six Python files together; it includes its
own helpers and does not replace the spatial-size or earlier sweep scripts.
Use the environment that already runs your studies: NumPy, bitarray, netCDF4,
matplotlib and psutil. No new third-party dependency is required.

From the repository root, with that environment active:

```bash
python tools/vmin_phi/vmin_phi_sweep.py --input-dir data --crop 256 256 --workers 12 --output results/vmin_phi_256
```

On Windows PowerShell, replace `python` with `.\.venv\Scripts\python.exe` if
your environment is not activated. Keep the command on one line.

## Default experiment

| Parameter | Value |
|---|---|
| Images | Lisbon, Bangladesh, Santarem, Doha, Finnmark |
| Crop | 256 x 256, centred, all native bands |
| nu_min sweep | -6, -5, -4, -3, -2, -1, 0, 1, 2 |
| phi curves | 0, 1, 2, 3, 4, 5 |
| A panels | 0, 3, 15 DN |
| Fixed nu_max | 4 |
| Fixed t_inc | 128 |
| P | 3 |
| Prediction / local sum | Reduced / narrow neighbour-oriented |
| Entropy encoder | Hybrid |
| Theta / Omega / register size | 3 / 19 / 64 |
| Input / dynamic range | Unscaled integer products/dn / unsigned D=16 |
| psi | 0 for A=0; fixed at 7 for A>0 |
| Storage | Low-memory hybrid adapter on by default |

The default is **5 x 9 x 6 x 3 = 810 separate compression runs**. Each scene
uses exactly the same centre crop for every parameter combination; coordinates
and a crop hash are saved. No binning, calibration, wavelength selection, or
resizing occurs. A crop that does not fit is skipped explicitly.

nu_max and t_inc stay fixed while nu_min, phi and A vary. nu_max does not change
automatically with phi. This isolates the requested interaction at the chosen
nu_max; it does not claim to reproduce every Green Book experiment setting.
At 256x256 and t_inc=128, much of the raster may use the final adaptation
parameter. Effects of nu_min primarily concern startup and the path to that
state. Saved adaptation coverage helps interpret small differences.

## Select values from the terminal

Example with only phi=0 and phi=1, and the same A values:

```bash
python tools/vmin_phi/vmin_phi_sweep.py --input-dir data --crop 256 256 --vmin-min -6 --vmin-max 2 --phi-values 0 1 --a-values 0 3 15 --nu-max 4 --t-inc 128 --workers 12 --output results/vmin_phi01_256
```

That is 270 runs. Range endpoints are **inclusive**. For selected nu_min points,
use `--vmin-values -6 -3 0 2` instead of the range options. The script enforces
`-6 <= nu_min <= nu_max <= 9`. With fixed Theta=3, valid phi values are 0..7.
Use `--images lisbon` for one scene, or name several scenes after `--images`.

`--t-inc 512` or `--t-inc 2048` changes the fixed interval for a separate study.
Allowed intervals are powers of two from 16 to 2048, expressed as literal sample
intervals rather than encoded header exponents. Use a new output folder when
changing the scientific grid, crop, or fixed parameters.

To inspect paths, shape, run count and estimated resources without encoding,
append `--dry-run` to your command. This does not create a results folder.

Small workflow test, eight configurations:

```bash
python tools/vmin_phi/vmin_phi_sweep.py --input-dir data --images lisbon --crop 16 16 --vmin-values -6 2 --phi-values 0 1 --a-values 0 3 --workers 12 --output results/vmin_phi_smoke
```

Small crops test setup and correctness, not representative compression rates
or parallel speed. Startup overhead can make them slower with many workers.

## Plots and CSV files

- `all_images_vmin_phi.png` and `.pdf`: one row per image, one column per A,
  nu_min on the horizontal axis, compressed bits/sample on the vertical axis,
  and one coloured curve per phi. The default layout is 5 rows by 3 columns.
- `<scene>_vmin_phi.png` and `.pdf`: the same comparison for one image.
- `results.csv`: every completed configuration, rate and exact bit counts,
  payload hash, reconstruction errors, crop coordinates/hash, adaptation
  statistics, timings and memory diagnostics. **Time and memory are not plotted.**
- `best_per_image_a.csv`: the lowest measured rate in each image/A group, plus
  completed/expected point counts. These are minima for the tested crop/grid,
  not general HYPSO recommendations. Exact ties list lower phi, then lower nu_min.
- `adaptation_progression.csv`: verified intervals of the applied update
  exponent; includes nu_min, phi, A and the job ID.
- `metadata.json`, `plan.csv`, `skipped.csv`, `failures.csv`, `jobs/`: settings,
  provenance, plans, checkpointed results and per-run logs.

Rates count header, entropy payload/tail and output-word padding per scalar
sample. External initialization is excluded from the plotted rate; the CSV
also records an inclusive rate. Error checks use encoder-side reconstruction,
not an independent decoder. A=0 must reconstruct exactly; A>0 must respect the
absolute bound in DN. All positive-A runs in a study use the same field width
for signalling the error limit.

Incomplete plots identify the number of completed points and leave gaps where
configurations are missing. To regenerate them without recompressing data:

```bash
python tools/vmin_phi/vmin_phi_sweep.py --plot-only --output results/vmin_phi_256
```

## Workers, memory and resuming

The previous dependency-aware predictor is retained: one configuration runs
at a time, and up to `--workers N` processes cooperate on its row/band tasks.
Tasks wait for spatial and spectral dependencies. The predictor state is not
reset at task boundaries. Hybrid encoding remains serial. `--workers 1` uses
the original serial predictor. CPU affinity, available tasks and scheduler
allocation can limit the process count. Twelve workers do not imply a
twelvefold speedup.

Shared numerical arrays are not copied in full per worker. A 256x256x120 crop
at P=3 in reduced mode needs about **1.23 GiB of temporary shared files**.
Default location: OUTPUT/_work; use `--work-dir /path/to/local/scratch` to put
them on a local SSD if your output folder is on a network drive. They are
removed when a run completes, fails normally or is interrupted with Ctrl+C.

Default RAM planning ceilings remain 90 GiB, suitable as headroom on your
120 GB machine. `--max-memory-gb` and `--total-memory-gb` adjust them; they are
estimates, not operating-system limits. The legacy peak_rss fields measure
only the crop coordinator. Added tree-RSS fields include children but count
shared pages repeatedly; Linux tree-PSS apportions shared pages. Sampling can
miss peaks. These columns do not estimate FPGA resources.

Completed configurations are saved individually. Repeat the same command to
resume without rerunning them. Worker count, temporary location, row-task size
and memory budgets can change on resume; scientific settings and source code
must match. Ctrl+C stops the active process tree, keeps completed points and
regenerates partial plots. The interrupted point restarts on the next run.

The predictor and low-memory encoder have source checks against the validated
NTNU model at commit `9fabb0371ebf93d72a0f83ca053e9cdb2c63e3c6`.
Changed upstream implementations require revalidation; do not bypass the checks.

See VALIDATION.md for the tests performed and their limitations.
