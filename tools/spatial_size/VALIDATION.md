# Validation

Tests used Linux/Python 3.12 and NTNU verification-model commit
`9fabb0371ebf93d72a0f83ca053e9cdb2c63e3c6`.

- All three helper files are unchanged from your current uploaded `tools.zip`.
- Verified the compact schedule representation against the previous vector
  formula for **80** combinations of shape, nu_min/nu_max and t_inc. These include
  2x2, 8x12, 128x128, 256x256 and 598x1092 shapes; constant schedules; and both
  reached and unreached final settings.
- Created a complete 12x20x120 NetCDF test file containing real Adelaide DN data.
  Encoded its 8x8 and 12x12 centre crops and its full 12x20 extent at t_inc=16/128,
  nu_min=0, nu_max=4, P=3, reduced narrow-neighbour, A=0, phi=0, hybrid.
- **Six serial and six two-worker runs** matched in scientific fields, including
  payload hash, bit counts, rates, errors, crop hashes and adaptation coverage.
- All six matched an independently constructed call to the original runner with
  the original standard-memory hybrid encoder.
- Full mode retained all 12x20x120 samples, with origin (0,0) and the exact full
  input hash. No spatial tiling or spectral subset was used.
- Every tested lossless case reconstructed with maximum error zero.
- Fresh workers reported positive OS peak-RSS values and valid stage/wall times.
- Oversized 64x64 requests were recorded as skipped, not silently resized.
- Resume preserved completed CSV bytes and ran no completed jobs again.
  Changed fixed settings were rejected. Plot-only regeneration passed.
- A memory-blocked job succeeded on resume after increasing the budget.
- Two timing repeats used separate job records/processes and identical payloads.
- PNG overview and adaptation plots were visually inspected.

The actual 598x1092x120 Adelaide capture was inspected and planned. Its full-image
low-memory estimate was approximately 30.86 GiB; this workspace had about 9 GiB
available. A full-image execution request with a restrictive budget exercised
the skip path without launching an oversized worker. **The full-sized capture
was not compressed here.** Full-mode correctness was exercised on the smaller
complete NetCDF test file, not claimed as a full-scene benchmark.

The actual five HYPSO captures were not available locally for execution. No test
fixtures or scientific-looking test plots are shipped as their results. Native
Windows execution and independent-decoder validation were not performed.

Peak RSS is measured; resource planning numbers are estimates. Python timings
and memory use do not measure FPGA throughput or FPGA storage requirements.
