# Validation of the offset / damping release

Environment: Linux, Python 3.12, NumPy 1.26.4, bitarray 2.8.2, NetCDF4,
Matplotlib and psutil 7.2.2. Reference NTNU verification-model commit:
`9fabb0371ebf93d72a0f83ca053e9cdb2c63e3c6`.

The imported `_damping_core.py`, `sweep_workers.py` and
`hybrid_no_traces.py` are byte-for-byte unchanged from the preceding
HYPSO_tinc_phi_study package. Upstream model files are unchanged.
The new driver constructs the psi grid, plots it and computes summaries.
The existing configuration-parallel engine now passes psi per job and
includes it in result identity checks, progress records and logging.

## Parameter application and serial/parallel equivalence

A NetCDF fixture used a patch of actual Adelaide raw DN data, retaining all
120 native bands. An 8 x 8 centre crop was run with:

- A = 3, 15;
- phi = 0, 3, 5;
- every psi from 0 through 7;
- t_inc = 512, nu_min = -1, nu_max = 7 - phi;
- P = 3, reduced, narrow-neighbour, hybrid, R = 64, Omega = 19, Theta = 3.

These **48 configurations** ran with one worker and again with four concurrent
configuration processes. All matched in payload SHA-256, header/payload/
padding bit counts, rates, reconstruction error, crop hashes and adaptation
statistics. Four concurrent configurations were observed in the parallel run.

Actual header snapshots were checked for the offset field AND every entry
of the offset table, not just a plot label. They were also checked for the
phi value, nu_min/nu_max encoding, t_inc encoding and fixed predictor/coder
profile. Different offsets produced different payloads in the fixture.
Each reported reconstruction error remained within its selected A.

Twelve cases (the same A/phi combinations with psi = 0 and 7) were also
run with the unmodified standard-memory HybridEncoder. They matched the
compact encoder's payload hash, bit counts and scientific metrics.
All runs use the original sequential Predictor.

## Compatibility with the preceding t_inc study

The previous release was run on the same fixture/crop, with t_inc = 512,
psi = 7, A = 3/15 and phi = 0/3/5. All six configurations matched this
new release's corresponding psi = 7 results exactly in scientific metrics,
payload hashes and crop hashes. The rate definition is unchanged.

## Real 128 x 128 crop

An actual 128 x 128 x 120 centre crop of
`adelaide_2025-09-09T01-12-29Z-l1a.nc` was encoded at origin (235,482).
The native source shape is 598 x 1092 x 120.

Settings: psi = 3, phi = 3, A = 3, t_inc = 512, nu_min = -1, nu_max = 4,
with the default P = 3 / reduced / narrow-neighbour / hybrid profile.
The run completed at **6.623111979166667 bits/sample**, with maximum
encoder-side reconstruction error **3 DN**. The programmed nu_max was
first applied at spatial index 2689. The full applied schedule was verified.

This is a functional check on Adelaide, not a measured offset optimum for
any of the five study captures. The new offset release was not run at
256 x 256 in this validation; crop planning for both 128 and 256 was checked,
and the existing crop-loading and original codec implementation are reused.

## Workflow checks

- Default planning produces 480 distinct jobs; the focused phi = 3/4 grid
  produces 160. No job has A = 0. All default 128 x 128 configurations reach
  their programmed final exponent at t_inc = 512.
- Illegal psi values, A = 0, A > 15, invalid intervals and invalid
  nu_min/nu_max combinations were rejected.
- Resume preserved completed job timestamps and CSV content when the worker
  count changed. A different t_inc in the same output folder was rejected.
- Plot-only regenerated outputs without loading imagery or executing the
  model.
- A restrictive RAM budget reduced four requested workers to one active job
  while all three queued jobs still completed.
- A t_inc = 256 override was applied and recorded correctly.
- Two distinct real-image fixtures produced distinct crop hashes.
- Summary minima and gains versus psi = 7 were checked against the raw
  results. A missing psi = 7 baseline was reported as blank. Exact rate ties
  were listed and retained psi = 7 when it shared the minimum.
- PNG/PDF figures were generated. The real small-crop plot and a full
  five-row/two-column/six-curve layout were visually inspected. The complete
  layout check used clearly labelled synthetic data; no such data are
  included in the deliverable or represented as scientific results.
- Runtime and memory are not plotted; only CSV/JSON measurements are saved.

## Limits and interpretation

The actual Lisbon, Bangladesh, Santarem, Doha and Finnmark files were not
available in the validation environment. Their full 480-point experiment
has not been executed here and should be run locally.

The suggested t_inc = 512 and emphasis on phi = 3/4 come from visual
interpretation of the user's supplied 128 x 128 t_inc results. They are
practical common candidates, not exact optima computed from a source CSV.

Reconstruction checks use encoder-side reconstructed samples, not an
independent decoder. Native Windows execution was not tested. The existing
fresh-process launcher, Windows output lock, Unicode NetCDF path fallback
and process cleanup are retained. Python resource measurements do not
measure FPGA throughput or storage requirements.
