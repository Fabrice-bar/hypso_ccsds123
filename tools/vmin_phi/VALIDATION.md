# Validation — configuration workers (v2)

Tested on Linux with Python 3.12, NumPy 1.26.0, bitarray 2.8.2,
psutil 5.9.5, netCDF4 1.7.5.1.post1 and matplotlib 3.10.8.

## What changed

Only `vmin_phi_sweep.py` and `_sweep_engine.py` changed among the Python files.
The uploaded `_damping_core.py`, `sweep_workers.py`, `hybrid_no_traces.py` and
legacy `wavefront_predictor.py` retain their exact original bytes. This runner
no longer imports or invokes the wavefront predictor. No upstream codec file
was modified.

The scheduler launches independent fresh processes, each performing one whole
configuration with the original serial predictor and the existing hybrid
encoder/low-memory adapter. Only the coordinator writes combined results.
Each job retains a distinct log, specification and checkpoint.

## Compression equivalence

A real Adelaide DN patch at source row 100, column 100, with shape 8x8x120,
was evaluated at nu_min={-3,1}, phi={0,3}, A={0,3,15}, nu_max=4,
t_inc=16, P=3, reduced prediction and narrow-neighbour local sums.

All 12 configurations were run using the uploaded original version with one
worker, then this version with one worker, then with two concurrent workers.
Payload SHA-256, bit counts, rates, crop hashes, reconstruction errors and
adaptation statistics matched exactly. Error bounds held for every point.
The same two-worker run demonstrably had two configurations in flight;
each configuration recorded one predictor process and zero shared-workspace
bytes. This checks scheduling equivalence, not independent decoder validity.

## Scheduling and workflow

- Four configurations at t_inc=2048 completed with four concurrent workers.
- The same four points with a 1.2 GiB test budget were automatically limited to
  one worker; all completed with matching payload hashes and bit counts.
- Runtime process inspection observed two independent worker processes and
  zero nested worker processes.
- Repeating a completed run with a different worker count started no new
  configurations and left completed checkpoint bytes/timestamps unchanged.
- Removing one checkpoint reran only that point.
- Changed scientific settings and old-version metadata were rejected.
- Dry-run wrote no outputs; plot-only regenerated plots without compression.
- Interrupting a run terminated active workers and kept completed checkpoints;
  resuming finished the remaining points without changing completed ones.
- A deliberately invalid DN input failed in one worker while a valid input in
  another worker completed and remained saved. The sweep reported failure.
- Plots retain bits/sample versus nu_min; runtime and memory are CSV-only.

## Limits

These are small-crop correctness and scheduling tests. Full-image simulations
and native Windows execution were not performed for this update. No full-image
speedup or actual peak-memory reduction is claimed. Memory admission uses the
existing conservative estimates; it is not a hard operating-system limit.
Native library thread settings and fresh subprocess execution are used on
both Linux and Windows; cores are not pinned.

The earlier validation record below documents the uploaded v1 package only.
Its within-configuration worker and shared-array descriptions do not apply to v2.

---

## Historical validation of the uploaded v1 package

# Validation

Test environment: Linux, Python 3.12, NumPy 1.26.4, bitarray 2.8.2, psutil 7.2.2.
Reference model: NTNU commit `9fabb0371ebf93d72a0f83ca053e9cdb2c63e3c6`.

The four codec helpers (_damping_core.py, sweep_workers.py,
hybrid_no_traces.py, wavefront_predictor.py) are byte-for-byte unchanged from
the spatial-size version 2 package. The new driver constructs the nu_min/phi/A
grid and plots it. Its execution engine is extracted from that package with
per-job nu_min, phi and A settings, parameter-aware logging, a shared error
limit field width, and a plotting callback. Upstream codec files are unchanged.

## Parameter application and equivalence

Two explicit NetCDF fixtures contained different patches of real Adelaide DN
data, with 120 native bands. A 12x12 centre crop from each was tested with:

- nu_min = -6, -1, 2;
- phi = 0, 5;
- A = 0, 3, 15 DN;
- fixed nu_max = 4, t_inc = 16, P = 3, reduced, narrow-neighbour, hybrid.

All **36 configurations** ran with one worker and again with four predictor
processes. They matched in payload SHA-256, header/payload/padding bit counts,
bits/sample, reconstruction error, crop hash and adaptation statistics.

Each of the same 36 points was also recomputed directly using the original
serial Predictor and standard-memory HybridEncoder. Bit counts and payload
hashes matched. The nu_min-to-header offset, fixed nu_max and damping fields
were checked explicitly. Lossless reconstruction had zero error; near-lossless
reconstruction respected the selected bound. Every scene retained the same
crop hash across its parameter combinations.

## Requested crop size

One actual 256x256x120 centre crop of
`adelaide_2025-09-09T01-12-29Z-l1a.nc` was encoded, at source origin (171,418).
Settings: nu_min=-3, phi=1, A=3, nu_max=4, t_inc=128 and the default P=3,
reduced/narrow-neighbour/hybrid profile. Twelve processes were requested;
CPU affinity allowed nine. The run completed at 5.5017659505 bits/sample,
with maximum encoder-side reconstruction error 3 DN. This is a functional
test result for that Adelaide crop, not a result for one of the five study scenes.

## Workflow checks

- Default planning creates 810 distinct configurations: five images, nine
  nu_min values, six phi values and three A values; crop defaults to 256x256.
- Resume retained completed CSV bytes and job timestamps, including after
  changing worker count. Changed fixed parameters were rejected.
- Plot-only regeneration completed without re-encoding images.
- Illegal nu_min/nu_max combinations, phi values, t_inc values and conflicting
  range/list options were rejected.
- A requested 256x256 crop of a small fixture was skipped explicitly before
  launching a worker; it was never silently resized.
- Temporary shared arrays were removed after completed runs.
- The positive-A header width stays common when a custom grid includes A=64.
- Per-image and combined plots were generated. The full five-row, three-column,
  six-curve layout was checked using clearly labelled synthetic layout data;
  these are not scientific results and are not included in this package.
- Runtime and memory are absent from plots and remain available in CSV/JSON.

## Limits

The actual Lisbon, Bangladesh, Santarem, Doha and Finnmark files were not
available locally, so their complete 810-point study was not executed here.
The user runs that study against those files with the supplied command.

Checks use encoder-side reconstruction, not an independent decoder. Native
Windows execution was not tested; multiprocessing spawn was tested on Linux.
The established process-tree cleanup and dependency-aware scheduling are
retained from the previous validated engine. Python timings and memory use do
not measure FPGA throughput or hardware storage requirements.
