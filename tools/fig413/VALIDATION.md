# Validation record

Validated on Linux with Python 3.12, NumPy 1.26.4, bitarray 2.8.2,
netCDF4, Matplotlib and psutil 7.2.2. The verification model was
NTNU-SmallSat-Lab/ccsds123_issue_2_verification_model at commit
`9fabb0371ebf93d72a0f83ca053e9cdb2c63e3c6`.

The supplied CCSDS 120.2-G-2 PDF was inspected, including the discussion
and Figure 4-13 on printed page 4-13. The adapted settings and differences
from that experiment are documented in README.md and output metadata.

## Checks performed

- Validated job counts: 3,840 for the default grid, 640 with no added noise,
  15,360 at Theta = 4, and 10,240 with three noise realizations at Theta = 3.
- Checked argument bounds, crop planning, repeat handling and deterministic
  noise generation. Sigma = 0 preserves the original integer input exactly.
  Boundary-value tests verified rounding/clipping behavior at 0 and 65535.
- Serialized and read back actual model headers at Theta = 3 and 4. Both
  representative tables preserve phi = psi = 0 in the first band and the
  selected values in all remaining bands, including maximum legal values.
- Ran 24 configurations on an 8 x 8 x 120 patch of real Adelaide HYPSO DN
  with one worker, then four workers. This covered A = 3/15, phi = 0/7,
  psi = 0/7, no added noise and two realizations at sigma = 3 DN.
  Payload hashes, bit counts, reconstruction checks and noise statistics
  agreed exactly. Four concurrent workers were observed; each ran a complete
  configuration with one native numerical-library thread.
- Compared eight noisy cases with the original, trace-producing hybrid
  encoder. Compressed payload hashes and numerical results matched the
  compact encoder used by default.
- Encoded the complete 64-pair legal grid at Theta = 3, A = 15, sigma = 0
  on the small real-data patch. Checked the selected grid minimum and the
  percentage-expansion calculations.
- Encoded four Theta = 4 boundary cases with phi/psi in {0, 15}; verified
  their error bounds and normalization by 16.
- Checked paired-input identities across parameter pairs and A values.
  Deliberately inconsistent input hashes were rejected. Checked incomplete
  cells, provisional panels, ties and aggregation across realizations.
- Confirmed that repeating a completed command with a different worker
  count reuses its checkpoints, while changed scientific settings are
  rejected in the same output directory. Plot-only regeneration succeeded.
- With `--first-band uniform`, four no-added-noise cases matched the
  preceding offset-sweep package exactly at identical settings.
- Completed one real 128 x 128 x 120 Adelaide configuration with A = 15,
  sigma = 15 DN, phi = 3, psi = 7, Theta = 3, nu_min = -1, nu_max = 4,
  t_inc = 512 and first-band parameters zero. It produced
  4.681022135416667 bits/sample, maximum encoder-side reconstruction error
  15 DN and no clipped input samples. This is a functional test, not a
  claimed optimum for Adelaide or any of the five study scenes.
- Inspected the rendered heatmap layout, including a complete real-data
  grid and a six-panel layout using explicitly labelled synthetic values.
  Synthetic layout values are not distributed as scientific results.

## Limits

The five selected study captures were unavailable in this test workspace;
the full 3,840-configuration HYPSO study has not been run here. No 256 x 256
encoding run was performed for this release. Native Windows execution was
not tested, although the runner uses separate Python subprocesses rather
than relying on fork-only behavior.

Error checks use the encoder's reconstructed samples, not an independent
decoder. With added noise, A bounds error against the noisy input being
compressed; it does not bound total error against the original crop.
Resource estimates are planning aids, not guaranteed memory limits or
predictions of FPGA resource use.
