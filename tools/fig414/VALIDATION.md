# Validation

Validated against NTNU verification-model commit
`9fabb0371ebf93d72a0f83ca053e9cdb2c63e3c6` using real HYPSO DN data.

- Compared original and compact hybrid encoders: exact header and payload bits
  agreed for zero, uniform, and band-dependent representative profiles. Cases
  covered A=3 and 15 and P=1,3,6. Band-dependent phi/psi header tables were parsed
  back through the model and matched the requested profiles exactly.
- Ran 96 small Lisbon encodes covering two A values, two P values, two spectral
  strides, baseline/uniform profiles and all resulting band-profile proposals.
  Every encoder-side reconstruction satisfied its absolute error bound.
- Ran a centered 256 x 256 Lisbon case with one 24-band group, P=3, A=3,
  nu_min=-1, nu_max=3, t_inc=2048: baseline, uniform (3,7), and a band-dependent
  proposal all completed. The cached DN matched the NetCDF samples exactly.
  This was a functionality check on one group, not the full scientific sweep.
- Compared one versus two worker processes on an identical small study: encoded
  digests, sizes and summary CSVs agreed. Resuming skipped completed encodes;
  resuming with a different nu_max was rejected.
- Checked balanced spectral grouping: every selected native band appears once
  at each stride, with four 24-band groups for the default 96-band window.
- Verified full-spatial dimension selection on actual Lisbon (956 x 684) and
  Adelaide (598 x 1092) files; the latter was a reader check, not an added default
  study scene. No image size is hard-coded.
- Inspected generated PNG plots and checked CSV/PDF generation.

The complete five-scene 256/full-image scientific sweep has NOT been run here.
Error validation uses encoder-side reconstruction, not an independent decoder.
The profile search is a documented heuristic and is not proven globally optimal.
Memory figures are conservative estimates, not measured peak-RSS guarantees.
