# Validation

Validated against NTNU verification-model commit
`9fabb0371ebf93d72a0f83ca053e9cdb2c63e3c6`, Python 3.12 on Linux.

The engine, multiprocessing helper and compact-hybrid adapter are byte-for-byte
copies of those in the previously supplied `HYPSO_figure49_damping.zip`.
Only the new study orchestration, plots, summaries and documentation were added.

## Executed checks

- A real Adelaide L1A centre crop, 8 x 8 x 120, was encoded for 48 configurations:
  A=0/3/15, phi=0/1, nu_max=0/2, nu_min=-6/0, t_inc=16/128.
- All 48 serial runs using the original hybrid encoder matched all 48 two-worker
  runs using the compact adapter in every non-timing CSV field, including payload
  SHA256, payload/header/padding lengths, rates and reconstruction errors.
- All 48 also matched the prior Figure-4.9 runner when its header was independently
  rebuilt for each individual schedule. This checks the propagation of swept
  nu_min and t_inc into the actual codec configuration.
- Lossless cases had maximum error zero; every near-lossless case passed the
  configured absolute-error bound check. Tested schedules included both reached
  and unreached final adaptation parameters.
- Resume with a different worker count preserved the CSV byte-for-byte and ran
  no completed jobs again. Changing psi in the same output folder was rejected.
- Exact crop reuse passed. A changed crop hash and a different fixed profile were
  rejected. Invalid t_inc and nu_min/nu_max combinations were rejected.
- The default five-image/128x128/540-job plan passed a dry run. An insufficient
  total-memory budget was rejected without writing outputs.
- Plot-only execution and summary rate-penalty arithmetic passed. Partial-result
  summaries were labelled incomplete.
- PNGs were inspected for the five-row layout and relative-change layout.

## Scope and limits

Only Adelaide data were available locally for these tests. Test fixtures used
the expected filenames to exercise the five-scene loader and layout. They were
explicitly test aliases, not scientific results for Lisbon or the other scenes.
No such fixture data or plots are included in this package.

The full 540-configuration study on your actual five 128x128 captures was **not**
executed. Native Windows execution was not tested; the multiprocessing tests used
the same `spawn` process method used on Windows. An independent decoder was not
tested. Memory values are conservative estimates, not measured peak RSS.

The source identity check on the low-memory adapter remains enabled. It must not
be disabled to accommodate a different upstream implementation.
