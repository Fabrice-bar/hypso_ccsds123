# Validation of version 2: parallel prediction within one crop

Tested on Linux, Python 3.12, NumPy 1.26.4, bitarray 2.8.2 and psutil 7.2.2,
with the NTNU reference model at commit
`9fabb0371ebf93d72a0f83ca053e9cdb2c63e3c6`. Its source files were not changed.
All multiprocessing tests used the spawn start method.

## Numerical and bitstream equivalence

Thirty direct predictor comparisons passed. Every scalar trace array, both
vector arrays, and the weight-update scaling schedule matched the unmodified
serial predictor exactly. The same hybrid encoder then produced identical
payload bytes and lengths from both predictors. Encoder-side reconstructed
samples obeyed the selected error bound; lossless cases had zero error.

- 24 combinations: all four local-sum types, reduced/full prediction, and
  A=0/3/15. Damping values 0/1/5 were included; row blocks of 1/2/3 were used.
- Four boundary configurations: 2x2 and other small rectangular images, one
  spectral band, P=0 and P=15, high damping, and t_inc=2048. Zero-length reduced
  prediction vectors were included.
- Two real-data cases: an Adelaide 16x48x120 crop at A=0 and A=15. Both used
  twelve distinct predictor processes, with exact array/bitstream agreement.

Shared-array cleanup and release of traces before encoding were checked.
The source-identity gate rejects predictors whose validated class AST differs.

## Complete runner

A complete 20x24x120 NetCDF fixture contained real Adelaide raw DN samples.
The runner encoded 8x8, 16x16 and its entire 20x24 extent at t_inc=16 and 128,
using one process and then four predictor processes.

- All six serial/parallel results matched: rates, payload SHA-256, exact bit
  counts, reconstruction errors, crop hashes and adaptation statistics.
- Every result reported one concurrent crop. Full mode retained the complete
  fixture; it did not divide it into separately compressed tiles.
- Oversized crops were skipped, and all temporary shared files were removed.
- Resume preserved completed CSV bytes and job timestamps, including after a
  worker-count change. Changed scientific settings were rejected.
- Plot-only regeneration worked for both version 1 and version 2 records.
  The revised overview was visually checked: no runtime or memory panels.
- A memory-blocked job succeeded when resumed with a sufficient budget.
- The actual 598x1092x120 Adelaide file was planned with a deliberately small
  memory budget; full-image encoding was correctly skipped before launch.
- Ctrl+C during an active nested process pool returned an interrupted status,
  stopped its workers and removed its temporary shared workspace. No running
  orphan worker remained.

## One local timing comparison

The same real Adelaide 128x128x120 centre crop used P=3, reduced prediction,
narrow neighbour sums, nu_min=-6, nu_max=4, t_inc=128, A=0, phi=0 and hybrid.
The two runs were sequential. Twelve workers were requested for the parallel
run, but CPU affinity allowed nine. Both produced the identical payload SHA-256:

`7f59fe78626d6a3ccadd4f98e7b866e0f95966eef78f77776c4fd38d72d39512`

| Measurement | Original serial predictor | Parallel predictor (9 processes) |
|---|---:|---:|
| Predictor stage, including setup | 19.75 s | 6.06 s |
| Hybrid encoder | 5.26 s | 5.18 s |
| Predictor + encoder | 25.01 s | 11.25 s |
| Crop job wall time | 25.53 s | 11.78 s |

This is one local comparison, not a statistical benchmark or a prediction for
other machines. Small crops can be slower because of process startup and
scheduling. The sequential hybrid encoder limits whole-run speedup.

## Limits

The five named study captures were not available locally. Actual compression
tests used Adelaide data and explicit test fixtures; none is presented as a
scientific result for Lisbon or the other named scenes. The largest real crop
encoded here was 128x128x120. Full-size 598x1092x120 compression was not run in
this approximately 9 GiB environment. Full-mode behavior was tested on the
complete small fixture, and the full-size planning guard was tested separately.

Native Windows execution and independent-decoder validation were not performed.
Spawn-based execution was tested on Linux. The original encoder calculations,
bitstream ordering and header generation remain unchanged. Runtime/memory
measurements do not estimate FPGA throughput or hardware resources.
