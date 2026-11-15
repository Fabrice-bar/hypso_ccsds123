# HYPSO nu_min / damping study — configuration workers (v2)

Plot **compressed bits/sample versus the initial adaptation parameter nu_min**,
with one curve per sample-representative damping value phi. This follows the
type of comparison in CCSDS 120.2-G-2 Figure 4-10, using your HYPSO scenes and
chosen fixed profile. Here `phi` means the damping parameter, written phi or
the Greek letter φ (the “ø” in your request).

## Install and run

Extract the ZIP into the verification-model repository root. It adds a separate
`tools/vmin_phi/` folder. Keep its six Python files together; it includes its
own helpers. Overwrite the existing files in `tools/vmin_phi/` when upgrading.
Other sweep folders are not changed. `wavefront_predictor.py` is retained for
archive compatibility but is no longer imported or used by this sweep.
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

`--workers N` now means **up to N complete configurations running concurrently**.
For example, with `--workers 2`, one process can run Lisbon / A=3 / phi=0 /
nu_min=-3 while a second runs Lisbon / A=3 / phi=0 / nu_min=-2. Each process
loads its own crop, runs the original serial predictor, performs reconstruction
and adaptation checks, and runs the hybrid entropy encoder. As soon as a
process finishes, its slot can start the next pending configuration. A fresh
process is used for each configuration so its memory is released on exit.

There is no nested predictor process pool, no row/band task splitting, and no
shared predictor workspace. Native numeric-library thread counts are set to
one in each worker. The operating system schedules workers; physical cores
are not pinned. CPU affinity and SLURM allocation can reduce concurrency.
`--workers 1` evaluates one complete configuration at a time.

Every active configuration needs its own input and codec arrays, so memory
roughly scales with the number of simultaneous configurations. The scheduler
reserves each job's estimated RAM before launching it and waits when another
job would exceed the available/total memory budget. For P=3 and low-memory
storage, a 956x684x120 image is estimated at about 30.9 GiB per configuration.
A 90 GiB total budget therefore admits at most two such jobs simultaneously,
even if `--workers 12` is requested. Actual peak memory can differ.

The default `--max-memory-gb 90` is a **per-configuration** estimate ceiling;
`--total-memory-gb 90` is an **aggregate** planning budget including a
coordinator allowance. These are admission estimates, not operating-system
hard limits. The scheduler also checks currently available RAM (including
Linux container limits). Lower concurrency or raise a budget only when the
machine has enough available memory. Start with two workers for full images.

The console shows each job's PID and active worker count. `last_execution.json`
records `execution_mode`, `peak_concurrent_workers`, and
`predictor_workers_per_configuration` (always 1). `results.csv` retains its
existing columns; predictor pool/process counts are 1, shared-workspace bytes
and rows-per-task are 0, and peak concurrent worker counts now reflect other
complete configurations running at the same time. Runtime and memory remain
CSV diagnostics only; they are not plotted.

`--rows-per-task` and `--work-dir` are accepted for command compatibility, but
are ignored and print an explanatory message when supplied with a non-default
value. No new dependency is required. The low-memory hybrid encoder source
check against the validated NTNU implementation is retained.

Completed configurations are checkpointed by the parent process, which is the
only writer of the combined CSV and plots. Repeat the same command to resume;
worker count and memory budgets may change. Ctrl+C stops active configuration
processes, preserves completed points and writes partial plots. Incomplete
configurations restart on the next run.

**Use a new output folder when upgrading from the old parallel-predictor
version.** Its metadata includes the script source digest, so this release
intentionally refuses to append to old results. Old result folders remain
available for comparison or `--plot-only`.

Full-size run for the five existing 956x684 study scenes (50 configurations):

```bash
python tools/vmin_phi/vmin_phi_sweep.py --input-dir data --images lisbon bangladesh santarem doha finnmark --crop 956 684 --anchor top-left --vmin-min -3 --vmin-max 1 --nu-max 4 --t-inc 2048 --a-values 3 --phi-values 0 3 --p 3 --local-sum narrow-neighbor --prediction-mode reduced --psi 7 --workers 2 --low-memory --max-memory-gb 90 --total-memory-gb 90 --output results/vmin_full_A3_phi0_3_config_workers
```

This version retains the existing crop interface. For another source size,
specify its actual height and width in `--crop`; dimensions are validated,
never silently resized. Add `--dry-run` to see the plan without encoding.

See VALIDATION.md for the tests performed and their limitations.
