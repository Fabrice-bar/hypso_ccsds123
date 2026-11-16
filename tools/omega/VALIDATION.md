# Validation

Environment: Linux, Python 3.12, NumPy 1.26.0, bitarray 2.8.2,
psutil 5.9.5, netCDF4 1.7.5.1.post1 and matplotlib 3.10.8.
The source-checked low-memory hybrid adapter uses the NTNU reference class
from commit 9fabb0371ebf93d72a0f83ca053e9cdb2c63e3c6.

## Source and parameter checks

The supplied Green Book (CCSDS 120.2-G-2) was inspected at section 4.2.4,
printed pages 4-17 and 4-18, and Annex C. Both pages were rendered to verify
plot axes, reference, legend and error-limit values. The local model's header
and predictor confirm the Omega field encoding (Omega-4) and legal range 4..19.

The existing input/header/encode helper and low-memory adapter are reused
without changing codec equations. The new worker changes the header's actual
weight-component resolution, rechecks legality, and verifies every segment
of the model's applied update exponent using the current Omega.

## Compression and reference matching

An 8x8x120 crop of real Adelaide integer DN data was run at every integer
Omega from 4 through 19, with A=0,3,15, P=3, reduced/narrow-neighbour,
phi=3, psi=7 (0 lossless), nu_min=-1, nu_max=3 and t_inc=16.
All 48 configurations completed with two independent configuration workers.

- All recorded header resolutions match the requested physical Omega.
- The actual applied adaptation exponent equals nu(t)+16-Omega.
- Encoder-side maximum error is zero for A=0 and never exceeds A otherwise.
- Nine points (Omega=4,10,19 crossed with A=0,3,15) were recomputed directly
  with the original serial predictor and standard-memory hybrid encoder.
  Payload SHA-256, bit counts, bits/sample, RMSE, maximum error and adaptation
  statistics matched exactly.
- The same nine points also matched between one-worker and two-worker runs.
- Each percentage was checked against 100*(rate/rate_at_19-1), using the
  reference from the same image and A. Every Omega=19 reference yields 0%.
- Removing one A's reference leaves only that A's percentages unavailable;
  other A references are not substituted.
- The original book's A list 0,1,3,7,15,31 was also exercised at Omega=4 and 19.

## Image size, scheduling and workflow

- Two real-data fixture files with different full dimensions (10x12x120 and
  12x10x120) completed using --full-image, with origins exactly (0,0).
- A requested Omega list containing only 10 automatically added and encoded
  Omega=19 for both files and each A.
- Five differently sized Adelaide test-patch fixtures produced both plot
  layouts. These are test patches, not the five study captures. The plots
  were visually checked; all labels and legends are legible.
- Four requested workers under a deliberately small 1.2 GiB budget were
  limited to one active configuration; all planned points completed.
- Completed checkpoints resumed with a changed worker count without being
  modified. Changing a fixed scientific setting was rejected.
- Plot-only regenerated both plot types without encoding. Dry-run wrote no
  output directory. Invalid Omega and conflicting crop/full-image options
  were rejected.

## Actual 256 x 256 x 120 smoke test

One centre crop of the available real Adelaide capture was encoded at source
origin (171,418), with Omega=19, A=3, P=3, reduced/narrow-neighbour, hybrid,
phi=3, psi=7, nu_min=-1, nu_max=3 and t_inc=2048. It completed at
5.9897460938 bits/sample with maximum reconstruction error
3 DN. The model reached nu_max, and the applied adaptation
schedule checks passed. The recorded predictor+encoder time on this test
machine was 178.74 seconds; this is not a runtime estimate
for the user's server. Plot spacing was refined after this run; the codec and
parameter-application code were unchanged. Final plot-only regeneration was
visually checked on the five-panel test layout.

## Limits

The complete five-scene production sweeps and full-size image compression
were not run here. Full-image selection was tested with different dimensions
on small real-data fixtures. Native Windows execution was not tested.
Small-crop correctness checks do not predict full-image throughput or memory.
Checks use encoder-side reconstruction, not an independent decoder.
Host Python memory and runtime are not FPGA resource or throughput estimates.
