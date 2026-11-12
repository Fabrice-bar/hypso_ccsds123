# HYPSO spatial-size study: parallel prediction inside one crop

Version 2 changes `--workers`: **one crop/configuration runs at a time**. Its
predictor uses up to N processes; hybrid encoding remains sequential. The next
crop starts after the previous job releases its arrays. `--workers 1` uses the
original predictor. Plots show bitrate and adaptation only; time and memory
remain in CSV/JSON, without plots.

## Install and run

Extract the ZIP into the verification repository root, replacing the previous
`tools/spatial_size` files together. Keep all four helper Python files beside
`spatial_size_sweep.py`. Upstream `ccsds123_i2_hlm` files are not modified.
Use the environment that already runs your studies: NumPy, bitarray, netCDF4,
matplotlib and psutil. No additional third-party package is required.

With the environment active, from the repository root:

```bash
python tools/spatial_size/spatial_size_sweep.py --input-dir data --images lisbon --sizes 128 256 full --t-inc 128 --workers 12 --output results/lisbon_size_parallel12
```

This runs 128x128, then 256x256, then full Lisbon. Each job requests twelve
predictor processes. CPU affinity, available bands/row blocks and
`SLURM_CPUS_PER_TASK` can lower that limit. Processes are not pinned to physical
cores. Small jobs can finish before all processes perform work.

On Windows PowerShell, replace `python` with `.\.venv\Scripts\python.exe` and
keep the command on one line. On Linux an example activation command is
`source .venv/bin/activate`; use your actual environment name.

**Use a new output folder when upgrading from version 1.** Code identity and
execution measurements differ. Finish or stop an active sweep before replacing
its script files: newly spawned processes must load the same version as their
controller. This update takes effect when you start the new command.

Small functional test (startup overhead can outweigh any speed gain):

```bash
python tools/spatial_size/spatial_size_sweep.py --input-dir data --images lisbon --sizes 16 32 --t-inc 128 --workers 12 --output results/lisbon_parallel_smoke
```

Omit `--images lisbon` for all five scenes. Use `--t-inc-values 128 512 2048`
instead of `--t-inc 128` to compare intervals. Allowed intervals are powers of
two from 16 to 2048. `--files PATH.nc` accepts explicit captures. All requested
configurations still run sequentially, ordered by size, scene, interval, repeat.

## How the parallel predictor works

1. Load one crop and initialize its predictor once. Retain all native bands.
2. Create tasks containing a few complete spatial rows in one spectral band.
   Each task calls the original ten per-sample predictor methods.
3. A task waits for the preceding row block in its band AND the same block in
   the preceding band. Within a task, rows/columns follow raster order.
4. A process pool runs ready tasks. Completed tasks release their dependants,
   forming a pipeline through rows and bands.
5. After all prediction is finished, verify reconstruction and the adaptation
   schedule. Release prediction traces, retaining mapped indices, and run the
   unchanged hybrid encoder. Save the result, clean up, then start the next crop.

The preceding-band dependency covers earlier prediction bands transitively:
the reference obtains their local differences from the preceding band's
vector. The preceding-row dependency preserves spatial neighbours and learned
weight history. Narrow first-row cross-band references are covered too.
**There are no tile boundaries, weight resets, or extra headers.** Task
boundaries are only scheduling boundaries in one continuous compression run.

Processes can execute Python predictor loops on different cores. Numerical
arrays live in shared memory-mapped files, rather than twelve private copies.
NumPy ndarray views provide normal indexing into those same shared buffers.

Hybrid entropy coding retains its original sequential state/order. Loading,
initialization, checks and plotting also include serial work. Pipeline startup,
drain and memory traffic limit speedup; twelve workers do not mean a twelvefold
whole-run gain. More processes can be slower when overhead dominates.

`--rows-per-task 0` automatically chooses 1–8 rows per task. An explicit override
is available: larger blocks reduce communication but may reduce concurrency.
This setting never changes predictor state or the adaptation interval.

## Scientific defaults

| Parameter | Default |
|---|---|
| Captures | Lisbon, Bangladesh, Santarem, Doha, Finnmark |
| Input | Native unscaled integer products/dn, all bands, unsigned D=16 |
| Sizes / placement | 128, 256, 512, full; nested centre crops |
| P / mode / local sum | 3 / reduced / narrow neighbour-oriented |
| nu_min / nu_max / t_inc | -6 / 4 / 128 |
| A / damping phi | 0 (lossless) / 0 |
| Theta / Omega / register | 3 / 19 / 64 |
| Offset psi | 0 for lossless; default 7 when A>0 |
| Encoder / storage | Hybrid / low-memory diagnostic storage |

Existing CLI overrides remain. `full` uses every spatial sample as one image;
no tiling, band selection, binning, calibration or resizing is introduced.
Oversized crops are skipped explicitly. `--anchor top-left` is also available.
Changing crop size changes scene content and boundaries as well as the fraction
of initial adaptation. The plotted schedule is verified against the model;
it does not measure convergence of learned weights.

## RAM and temporary disk

Shared arrays are allocated once per crop, plus private worker imports and small
constants. RAM does not grow as N full copies of the crop. Planning adds a
0.10 GiB allowance per predictor process to the codec estimate. Default per-job
and total ceilings remain 90 GiB (`--max-memory-gb`, `--total-memory-gb`). The
runner also checks currently available memory. These are estimates, not OS limits.
Use `--dry-run` to see the plan without encoding or writing results.

Shared files use a unique folder under OUTPUT/_work by default. Prefer a local
SSD. If output is on a slow/network disk, append `--work-dir /path/to/local/scratch`.
Reduced-mode temporary file size is `samples * 8 * (15 + 2*P)` bytes. At P=3,
598x1092x120 needs about **12.26 GiB temporary disk**, shared once, not per worker.
The runner checks free disk space plus 0.25 GiB headroom. Full prediction mode
adds three vector components. `--workers 1` needs no shared-array files.

Normal completion, caught failures and Ctrl+C clean up temporary arrays. An OS
crash or forced kill of the controller can leave a folder behind. Remove that
run's leftover _work subfolder only after all its processes have stopped.

## Outputs and resuming

- `results.csv`: bitrate, exact bit counts, payload SHA-256, crop hash, errors,
  adaptation coverage, time, memory and execution counts. `predictor_pool_size`
  is the pool limit; `predictor_processes_used` counts processes that completed
  tasks; `concurrent_crops` is always 1.
- `<scene>_size_sensitivity.png/.pdf`: bits/sample and fraction of updates at
  nu_max versus spatial size. Stars mark full images. One curve per t_inc;
  repeats use median/min/max. No time or memory panels.
- `<scene>_adaptation_tinc<T>.png/.pdf`: schedule versus raster index and fraction
  of the raster. One curve per size. `adaptation_progression.csv` stores intervals.
- `metadata.json`, `plan.csv`, `skipped.csv`, `failures.csv`: provenance and outcomes.
  `jobs/*.json` includes worker PIDs; `jobs/*.log` reports task progress and encoding.

A source check guards the validated Predictor at NTNU commit
`9fabb0371ebf93d72a0f83ca053e9cdb2c63e3c6`. Changed predictor code requires
revalidation of this scheduler. The low-memory encoder has a separate source
check. Do not bypass either guard.

Completed crops are checkpointed. Repeat the same scientific command to resume;
worker count, row-block size, temporary location and budgets may change. Changed
inputs/settings/grid/code/storage require a new output folder. Ctrl+C stops the
whole active process tree; completed crops remain, while an interrupted crop
starts over next time.

Regenerate revised plots from version 1 or version 2 results without encoding:

```bash
python tools/spatial_size/spatial_size_sweep.py --plot-only --output results/YOUR_EXISTING_FOLDER
```

## Measurement details

Predictor time includes pool startup and shared-array setup. The added schedule
verification scan is excluded. Encoder time is separate. Wall time includes the
whole crop subprocess, loading/checks/cleanup, excluding initial planning and plots.

Legacy `peak_rss_*` fields describe only the crop coordinator. Added process-tree
fields cover its children: summed RSS double-counts shared pages; Linux PSS
apportions them and is more useful. `sampled_tree_peak_pss_gib` is blank when
unavailable. Samples can miss peaks and are not full-machine RAM measurements.
PSS is sampled every 0.5 seconds; RSS uses --memory-poll-ms (default 50).

Reconstruction checks are encoder-side, not independent-decoder validation.
Python timings/memory do not measure FPGA resources. See VALIDATION.md.
