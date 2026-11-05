# HYPSO compression sweeps: two workers

Use these tools inside your own CCSDS-123 verification repository. The five entry points keep their original compression settings, metrics and plot formats. `sweep_workers.py` is a required sixth file: keep it beside all the entry points. No new third-party packages are needed beyond your existing model and plotting environment.

## Start on aulus7

1. Upload `HYPSO_sweeps_multicore.zip` into the root of your remote repository (drag it there in VS Code Remote SSH). Open a terminal in that repository with the same Python environment you used for the serial run.
2. Unpack the updated tools into a separate folder to preserve any edits in your own `tools/` checkout:

   ```bash
   cd ~/master_H26/hypso_ccsds123
   python -m zipfile -e HYPSO_sweeps_multicore.zip parallel_sweeps
   ```

3. If you want to continue the serial Lisbon run in its existing output directory, first stop the old command with Ctrl+C in its terminal. Wait until saving/plotting finishes and the shell prompt returns. An already running interpreter cannot gain the new workers.
4. Run your previous experiment with the new entry point:

   ```bash
   python parallel_sweeps/tools/predictor_sweep.py --input-dir data --images lisbon --p-max 7 --crop 256 256 --max-memory-gb 30 --total-memory-gb 40 --workers 2 --output results/p7_sweep_256_lisbon
   ```

   Use a machine or job allocation with two CPUs and sufficient memory for this example. The parent prints how many configurations are already complete and how many remain. Do not add `--overwrite` to a resume command. For a fresh experiment, change the output folder, e.g. to `results/p7_256_lisbon_two_workers`.

   The four NetCDF tools accept `--dry-run` to validate the input, settings and estimated fresh-run memory without compression. Existing result compatibility is checked on the actual run. On Windows use the same commands, substituting `.\.venv\Scripts\python.exe` for `python` from the repository root.

## What the two workers do

- One job is one complete codec configuration on an entire selected crop. For Lisbon with P=0 through 7 there are 8 P values × 4 sums × 2 modes = 64 jobs. The first two normally compare full and reduced mode at P=0 using the same local sum. When a worker finishes, the parent checkpoints its result and assigns the next outstanding job.
- Workers are separate Python processes. The predictor and entropy coder still scan each cube in their original order. Splitting the image into independent halves would change predictor history and potentially the bitstream. Workers are not pinned to a specific physical core; the operating system schedules them.
- A worker has its own weights, encoder state and temporary model arrays. Both workers read the same input values: the parent writes each validated crop once as a temporary .npy file; the workers open it as a read-only memory map. No live NetCDF handles or full cubes are sent per task. Allow temporary disk space for the selected int64 crops (approximately 60 MiB for one 256 × 256 × 120 crop). On Linux, set TMPDIR to node-local scratch for these temporary files if appropriate.
- Only the parent writes the combined CSV, metadata and plots. CSV rows are replaced atomically in a fixed configuration order even when jobs finish in a different order. The raw-file runner also lets each worker write codec outputs in its own configuration directory.
- Ctrl+C stops the new worker processes. Completed, checkpointed rows remain usable; unfinished configurations are rerun on resume. The parent reports a worker exception or an unexpected exit with its configuration.

## Select a runner

| Runner | Swept parameters | Resume behavior |
|---|---|---|
| predictor_sweep.py | Number of preceding bands, local sum, prediction mode | Compatible previous P sweeps; increasing P max also supported |
| predictor_A_sweep.py | Absolute error limit, local sum, prediction mode; fixed psi=0 | Compatible A sweeps from this script |
| absolute_error_sweep.py | Absolute error limit, local sum, prediction mode; original near-lossless psi=7 | Compatible A sweeps from this script |
| vmax_sweep.py | Final adaptation exponent and absolute error limit | Compatible runs with the same ranges and settings |
| run_study.py | Raw-file P or error sweeps, plus a single baseline | As before, requires a fresh output directory. A one-configuration baseline uses one process. |

The default is `--workers 2`. Use `--workers 1` to run without spawning children. The new and matching stock serial releases of the four NetCDF runners can share a checkpoint. They check the recorded script identity, input paths/crop hashes, codec source, Python/NumPy versions and study settings as applicable. Worker count is not part of compression identity, so you can switch from 1 to 2 or back. If your private script/codec differs or the identity does not match, keep that folder and use a fresh output path. Do not run an old and new program into the same directory simultaneously.

## Memory and timing

`--max-memory-gb` on the NetCDF scripts still checks a **single configuration**. The new `--total-memory-gb` applies to the combined estimated working set; it does not allocate RAM or enforce an operating-system limit. Use your actual job memory allocation for that budget. For the 256 × 256 Lisbon crop, 120 bands and P max 7, two workers plus the coordinator and input estimate about **35.2 GiB**. Estimates are conservative and actual peaks depend on the configuration. If the allocation is too small, use one worker or reduce the crop size.

The scripts log worker PIDs, per-configuration model time and total wall time for newly computed configurations. Compare the same arguments and data with **two fresh output folders**, once with `--workers 1` and once with `--workers 2`. A resumed run skips completed work, so it is unsuitable for a speed comparison. Dividing the serial wall time by the parallel wall time gives the speedup. A worker count of two cannot accelerate one isolated compression; startup, plotting, storage and memory bandwidth also affect total time.

## Measured check

Using the NTNU verification model at commit `9fabb0371ebf93d72a0f83ca053e9cdb2c63e3c6`, the real Lisbon centre 32 × 32 × 120 crop, P=0…1, all four local sums and both modes (16 lossless configurations):

- One worker: 67.2 s.
- Two workers: 32.3 s.

These are whole-command times for one comparison on the test host, including plots. All non-timing CSV fields matched. Use these times as evidence for this workload, not a forecast for a 256 × 256 run on a shared machine.
