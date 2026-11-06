# Validation of damping_sweep.py

Checked on 6 November 2026 against NTNU verification-model commit
`9fabb0371ebf93d72a0f83ca053e9cdb2c63e3c6`, using Linux, Python 3.12 and the
repository-compatible NumPy 1.26.4 / bitarray 2.8.2 environment.

## Numerical and process checks

Real raw DN values from the centre of
`adelaide_2025-09-09T01-12-29Z-l1a.nc` were used in an 8 x 8 x 120 test crop.
Test fixtures used the five expected scene filenames solely to exercise the
input interface. They are not measurements of the Lisbon/Bangladesh/etc. scenes.
No fixture data or simulated measurement plots are included in this package.

The 72-case test grid was:

- A = 0, 3, 15;
- phi = 0, 1, 2, 3, 4, 5;
- nu_max = 0, 1, 2, 3;
- nu_min = 0, t_inc = 16 (chosen to exercise the schedule within the tiny crop);
- P = 3, narrow-neighbour sums, reduced prediction, hybrid encoder.

All 72 cases produced identical non-timing CSV fields between:

1. New script, one worker, original hybrid trace storage.
2. New script, two spawned workers, compact hybrid trace storage.

Every case also matched the previous `vmax_sweep.py` encoder call at the same
phi and nu_max, including payload SHA256, payload length, header length,
padding, bits/sample, compression ratio, maximum error and RMSE.

All lossless reconstruction checks returned zero maximum error. Every
near-lossless case met its specified absolute error bound. Varying phi changed
the encoded payload on the test data, confirming that the sweep is active.

## Integration checks

- Automatic resume scheduled no new jobs and preserved CSV bytes.
- Changing psi while reusing the output directory was rejected and preserved
  the existing results.
- `--crop-from` reused and verified the saved crop shape, origin and hash.
- `--plot-only` recreated plots and summaries without loading source images.
- Out-of-range damping and invalid t_inc values were rejected.
- A dry run checked the default 900-configuration grid and 128 x 128 x 120
  input shape using fixture data, without executing that full study.
- Both individual three-panel figures and the five-row/three-column figure
  were rendered and visually inspected. The full-layout check used synthetic
  rates solely to inspect labels, legends and spacing.

The included `sweep_workers.py` and `hybrid_no_traces.py` are unchanged copies
of the previously supplied helpers. No predictor, quantizer or entropy-coding
equations were modified.

## Scope

The full 900-configuration study on the user's five 128 x 128 HYPSO crops has
not been run here. Windows-native execution has not been tested; the process
helper uses the same spawn contract on Linux and Windows. Reconstruction
checks remain encoder-side, not an independent compressed-bitstream decoder.
