# HYPSO damping/adaptation study: Green Book Figure 4-9

This script studies the interaction between sample-representative damping `phi`
and the final predictor adaptation parameter `nu_max`. Each figure has one
panel per absolute error limit, with compressed bits/sample versus `nu_max`
and one curve per `phi`. The combined figure has one row per HYPSO scene.

It follows CCSDS 120.2-G-2 (December 2022), Section 4.2.2, Figure 4-9 (printed
page 4-9). It uses your HYPSO data/profile rather than reproducing that book's
numerical results. In particular, this study uses A=15 rather than the book's
A=20, narrow-neighbour/reduced prediction and your previous initial adaptation
settings. `nu_max` is swept again; fixing it to 1 would miss the interaction
that Figure 4-9 investigates.

## Installation

Extract the ZIP into the root of your verification-model repository. It creates
`tools/fig49/` with:

- `damping_sweep.py`: the new sweep, plots and summary tables.
- `sweep_workers.py`: the existing process helper, included unchanged.
- `hybrid_no_traces.py`: the existing optional compact hybrid adapter, unchanged.
- This README and `VALIDATION.md`.

Existing sweep scripts are not overwritten. Use the Python environment already
used for your sweeps. Dependencies are NumPy, bitarray, netCDF4, matplotlib and
psutil. Keep your repository-compatible Python/package versions; Python 3.11
with the repository requirements is suitable for the existing setup.

Run from the repository root, which contains `ccsds123_i2_hlm/` and `data/`.

## Default scientific configuration

| Parameter | Setting |
|---|---|
| Scenes | Lisbon, Bangladesh, Santarem, Doha, Finnmark |
| Input | `products/dn`, original integer DN; all native spectral bands |
| Spatial crop | Centre 128 x 128; same crop is used for every configuration |
| P | 3 |
| Local sum | Narrow-neighbour-oriented |
| Prediction mode | Reduced |
| Entropy encoder | Hybrid |
| nu_max sweep | 0, 1, ..., 9, inclusive |
| phi sweep | 0, 1, 2, 3, 4, 5 |
| Absolute error A | 0, 3, 15 DN; relative error control disabled |
| nu_min | -6 |
| t_inc | 128 spatial samples per band |
| Sample-representative resolution Theta | 3 |
| Sample-representative offset psi | 7 for A>0; required value 0 for A=0 |
| Dynamic range D / weight resolution Omega / register size R | 16 / 19 / 64 |
| Weight initialization / exponent offsets | Default / all zero |

Other hybrid encoder settings come from local `header.py`. The script records
complete base headers, swept values, crop hashes and model source hashes in
`metadata.json`. There is no calibration, denoising, band removal or extra binning.

## Run the full study

Linux/SSH, with the environment activated, from the repository root:

```bash
python tools/fig49/damping_sweep.py --input-dir data --crop 128 128 --phi-values 0 1 2 3 4 5 --vmax-min 0 --vmax-max 9 --a-values 0 3 15 --p 3 --local-sum narrow-neighbor --prediction-mode reduced --nu-min -6 --t-inc 128 --workers 2 --low-memory --max-memory-gb 8 --total-memory-gb 90 --output results/fig49_damping_128
```

Windows PowerShell, from the repository root:

```powershell
.\.venv\Scripts\python.exe tools/fig49/damping_sweep.py --input-dir data --crop 128 128 --phi-values 0 1 2 3 4 5 --vmax-min 0 --vmax-max 9 --a-values 0 3 15 --p 3 --local-sum narrow-neighbor --prediction-mode reduced --nu-min -6 --t-inc 128 --workers 2 --low-memory --max-memory-gb 8 --total-memory-gb 90 --output results/fig49_damping_128
```

This is 5 scenes x 3 error limits x 6 damping values x 10 nu_max values = **900
configurations**, two at a time. Do not launch a separate two-worker process
for every damping value; the script schedules the whole grid itself.

`--low-memory` removes large hybrid text-trace arrays without changing coding
methods. The adapter checks that the installed HybridEncoder matches the
validated upstream implementation. If it reports a changed encoder, use the
original storage mode by omitting this flag; do not disable the source check.
For 128 x 128 x 120, estimates are about 1.26 GiB per job in compact mode versus
4.35 GiB in original mode, plus coordinator/input memory. These are conservative
estimates, not measurements or enforced limits. Keeping two workers does not
guarantee a twofold speed-up. The predictor is unchanged.

Append `--dry-run` to check files, dimensions, grid size and memory before running.
Remove that flag for compression. A dry run does not create result files.

## Small local smoke test

```powershell
.\.venv\Scripts\python.exe tools/fig49/damping_sweep.py --input-dir data --images lisbon --crop 8 8 --nu-min 0 --t-inc 16 --vmax-min 0 --vmax-max 2 --phi-values 0 3 5 --a-values 0 3 --workers 2 --low-memory --output results/fig49_smoke
```

This tests 18 configurations and plot generation. It deliberately uses a faster
adaptation schedule for the tiny crop; do not use it as scientific evidence for
the 128 x 128 study. On Linux replace the interpreter with `python`.

## Reuse exact earlier crop locations

Instead of `--crop 128 128`, use:

```text
--crop-from results/YOUR_PREVIOUS_VMAX_FOLDER/metadata.json
```

Substitute the real metadata path. Do not combine `--crop-from` with `--crop`,
`--row` or `--col`. The crop dimensions/origins, variable/axis order and data
hash are reused/checked; predictor and encoder settings still come from the
new command/defaults. Omitting this option uses the default centre crops.

## Results and how to use them

- `all_images_damping.png` and `.pdf`: five rows (scenes), three columns (A),
  with six curves (phi) per panel, in the style of Figure 4-9.
- `<scene>_damping.png` and `.pdf`: one three-panel figure per scene.
- `results.csv`: each configuration, rate, compression ratio, encoder-side
  maximum error and RMSE, effective adaptation schedule, payload hash and times.
- `best_by_phi.csv`: lowest rate/nu_max for each scene/A/phi.
- `best_overall.csv`: lowest rate/phi/nu_max for each scene/A.
- `metadata.json`: reproducibility information and fixed/swept settings.

Summary tables explicitly mark incomplete groups. Exact rate ties choose the
smaller nu_max, then smaller phi; this is a tie-breaking convention, not evidence
that this pair is uniquely best. Read the full curves before selecting a common
profile for all scenes. Lower bits/sample is better compression. Different
near-lossless settings can produce different reconstructions within the same A.
Rate alone does not establish suitability for HYPSO scientific applications.

The primary rate includes the CCSDS header, entropy payload/termination and word
padding, divided by the number of scalar samples. Separate hybrid initialization
and optional-table files are excluded from that rate; their sizes and an inclusive
rate are also in the CSV. The NetCDF file's compressed size is not the denominator.

The error check uses the encoder-side quantizer-bin reconstruction, including the
special first spatial sample. It is not an independent bitstream decoder check.
No per-run debug CSVs or encoded output files are written. Timing measures the
Python verification model, not FPGA throughput or hardware memory requirements.

## Resume and redraw

Completed results are checkpointed after every configuration, even with multiple
workers. Repeat the same command/output to resume. Changing only worker count or
memory budgets is allowed. Changed ranges, crops, settings, code, Python/NumPy
version or storage mode require a new output directory (or explicit `--overwrite`).
Do not use the previous vmax sweep's output directory for this new study.

```bash
python tools/fig49/damping_sweep.py --output results/fig49_damping_128 --plot-only
```

This redraws all plots and summary CSVs from saved results without recompressing
or requiring input NetCDF files. Use the PowerShell interpreter equivalent there.

## Short-crop adaptation

The model uses spatial index t independently per band, with the ramp proportional
to `nu_min + floor((t - columns)/t_inc)`, clipped to `[nu_min, nu_max]`. The actual
model's final exponent is checked against the plotted schedule. The CSV records
whether each requested final value was reached and the fraction of updates spent
there. Hollow markers/shading identify unreachable values. For the default
128 x 128 / nu_min=-6 / t_inc=128 study, all requested nu_max values 0..9 are
reached. Never count the 120 bands as 120 times more adaptation steps per band.
