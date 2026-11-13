# Validation of this release

Validated on Linux with Python 3.12, NumPy 1.26.4, bitarray 2.8.2,
NetCDF4, Matplotlib and psutil 7.2.2. The NTNU verification-model reference
commit was `9fabb0371ebf93d72a0f83ca053e9cdb2c63e3c6`.

The three imported helpers `_damping_core.py`, `sweep_workers.py` and
`hybrid_no_traces.py` are byte-for-byte unchanged from the previous vmin/phi
package. This release adds a new driver and a configuration-parallel
execution engine. It does **not** include or invoke the previous wavefront
predictor. Upstream verification-model files are unchanged.

## Configuration and process checks

- The default grid has 720 distinct jobs: five scenes x three A values x six
  phi values x eight update intervals. A three-scene grid has 432 jobs.
- The default crop is 128 x 128; the same planner accepts 256 x 256.
- Every job has nu_min = -1 and nu_max = 7 - phi.
- A outside 0..15 is rejected, including A = 20. Invalid update intervals,
  phi values and nu_min/nu_max combinations are rejected.
- For all selected phi values, the reported adaptation segments were
  compared against the explicit per-sample exponent calculation at every
  default interval. Both the 128 and 256 boundary cases were checked.

## Serial/parallel and original-codec equivalence

A NetCDF fixture contained a patch of actual Adelaide raw DN data with 120
native bands. Using an 8 x 8 centre crop, the following 36 configurations
ran with **one worker**, then again with **four concurrent complete jobs**:

- phi = 0, 1, 2, 3, 4, 5;
- A = 0, 3, 15 DN;
- t_inc = 16 and 2048;
- nu_min = -1, nu_max = 7 - phi, P = 3, reduced/narrow-neighbour/hybrid.

All 36 matched in payload SHA-256, header/payload/padding bit counts,
bits/sample, reconstruction error, crop hash and adaptation statistics.
The four-worker run reached four concurrent configuration processes. Each
configuration used one predictor process and one native library thread.

Twelve of those configurations (phi = 0 and 5, all three A values and both
intervals) also ran with the unmodified standard-memory hybrid encoder.
Their payload hashes, bit counts and reconstruction metrics matched the
compact encoder. Every run uses the original sequential Predictor.

Header snapshots were checked for P, reduced mode, narrow-neighbour sums,
hybrid coder, R = 64, Omega = 19, Theta = 3, damping, psi, nu_min/nu_max
encoding offsets and interval encoding. Lossless psi was zero; near-lossless
psi was seven. Reconstruction error was zero for lossless and within A for
near-lossless runs.

An additional sixteen actual configurations exercised all eight default
intervals at phi = 0 and 5, A = 3. Their interval header fields matched
log2(t_inc) - 4.

## Real 128 and 256 crops

One configuration was run at each requested crop size using the available
file `adelaide_2025-09-09T01-12-29Z-l1a.nc`, whose native raw shape is
598 x 1092 x 120. These were functional checks, not results for the five
named study scenes.

Both used A = 0, phi = 0, t_inc = 2048, nu_min = -1, nu_max = 7 and the
default P = 3 / reduced / narrow-neighbour / hybrid profile.

| Crop | Centre origin (row, col) | Bits/sample | Maximum reconstruction error | nu_max reached |
| --- | --- | ---: | ---: | --- |
| 128 x 128 x 120 | (235, 482) | 9.723046875 | 0 DN | No |
| 256 x 256 x 120 | (171, 418) | 7.690576171875 | 0 DN | Yes, first applied at spatial index 16641 |

The 128 crop ended before the cap could be used and its marker was hollow.
Both checked the actual applied predictor schedule against the reported
schedule. Different-sized centre crops also contain different surrounding
spatial content; these two rates alone do not isolate adaptation effects.

## Workflow and plotting checks

- Resume preserved finished job timestamps and scientific CSV data after
  changing worker count. A changed crop was rejected in the same directory.
- Plot-only regeneration worked without input imagery/model loading.
- With four workers requested and a restrictive memory budget, concurrency
  dropped to one and all four queued configurations still completed.
- A job skipped for insufficient memory completed when retried with a larger
  budget. A partially completed study returned a nonzero exit status.
- Two distinct real-image fixtures produced distinct crop hashes. A crop
  larger than its source was explicitly skipped, never resized.
- Ctrl+C during two active jobs terminated both configuration processes.
  Neither process had nested worker children.
- PNG and PDF outputs were generated. The actual small-crop results and
  the complete five-row/three-column/six-curve layout were visually checked.
  The full layout check used clearly labelled synthetic data, which are not
  scientific results and are not included in the release.
- No runtime or memory plots are produced. Those measurements are retained
  only in CSV/JSON diagnostics.

## Limits

The Lisbon, Bangladesh, Santarem, Doha and Finnmark data files were not
available in the test environment. The full 720-point scientific sweep must
be run locally against those captures; it has not been precomputed here.

Native Windows execution was not tested. The runner launches fresh Python
processes with explicit argument lists and includes the inherited Windows
output-directory lock and Unicode-path fallback.

Reconstruction checks use the encoder's reconstructed samples, not an
independent decoder. Reaching nu_max means the programmed exponent reached
its cap, not that learned weights converged. Python memory/runtime numbers
are not FPGA resource/throughput measurements.
