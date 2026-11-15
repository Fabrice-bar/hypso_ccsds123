# HYPSO Figure 4-14 study

Extract this ZIP into the root of your existing CCSDS verification-model repository.
It adds `tools/fig414/` and leaves the model and previous sweep scripts unchanged.
Use the Python environment that already runs your model (Python 3.11 or 3.12).

This recreates the TYPE of comparison in CCSDS 120.2-G-2 Figure 4-14 using HYPSO
data, rather than claiming a numerical reproduction of the AVIRIS-NG experiment.

## Commands

Install extra dependencies in your existing environment:

```text
python -m pip install -r tools/fig414/requirements-extra.txt
```

Run all five images with centered 256 x 256 crops:

```text
python tools/fig414/fig414_sweep.py --input-dir data --crop 256 256 --a-values 3 --workers 2 --memory-budget-gb 90 --output results/fig414_256_A3
```

Run each image's full spatial extent:

```text
python tools/fig414/fig414_sweep.py --input-dir data --full-image --a-values 3 --workers 2 --memory-budget-gb 90 --output results/fig414_full_A3
```

`--full-image` reads EACH file's own dimensions. Do not combine it with `--crop`.
Spectral grouping remains: full image means full spatial extent, not all 120 bands
encoded together in one stream in this experiment.

Preview the plan without simulations or reading the DN crop:

```text
python tools/fig414/fig414_sweep.py --input-dir data --crop 256 256 --dry-run
```

Quick functionality test (not a scientifically representative crop):

```text
python tools/fig414/fig414_sweep.py --input-dir data --images lisbon --crop 8 8 --p-values 1 3 --strides 1 4 --phi-values 3 --workers 2 --output results/fig414_smoke
```

Commands are single lines, suitable for PowerShell or Linux. In PowerShell, if
the virtual environment is not activated, replace `python` with
`.\.venv\Scripts\python.exe`.

Change crop size with `--crop 128 128`, `--crop 256 256`, etc. Crops are centered
by default; use `--row 0 --col 0` for the upper-left corner. Origins are zero-based.
Select scenes with `--images lisbon finnmark` or any subset of the five names.
The script finds the original filenames recursively and rejects ambiguous duplicates.
All five supplied study files are 956 x 684 x 120; the centered 256 crop begins
at row 350, column 214. The program does not hard-code these dimensions.

Use `--a-values 3 15` in a new output directory to include both near-lossless
limits. A=1..15 is supported; lossless is omitted. For A=15, your earlier
Bangladesh grid favoured psi=6, so consider `--psi-values 6 7` when running that
case. This expands the search and run count.

## Configuration

| Parameter | Default |
|---|---|
| Scenes | Lisbon, Bangladesh, Santarem, Doha, Finnmark |
| Spatial crop | Centered 256 x 256; adjustable or full spatial image |
| Input | Integer `products/dn`, D=16 |
| Wavelengths | `metadata/corrections/wavelengths` |
| P | 1, 2, 3, 4, 5, 6, forming the x axis |
| Prediction / local sum | Reduced / narrow neighbor-oriented |
| Encoder / sample ordering | Hybrid / BIL |
| R / Omega / Theta | 64 / 19 / 3 |
| nu_min / nu_max | -1 / 3 |
| t_inc | 2048; adjustable with `--t-inc` |
| A | 3 DN; optional 15 |
| Uniform profile seeds | (phi,psi) = (2,7), (3,7), (4,7) |
| Baseline | phi_z=psi_z=0 at the SAME A |
| Added noise | None; sensor noise is retained |

nu_max is fixed at 3 for all phi values, as in Figure 4-14; it does NOT use
nu_max=7-phi. This is also consistent with the good nu_max=3–4 region in your
full-Lisbon experiment. t_inc=2048 is a working HYPSO choice from your full-scene
tests, not a claimed original Figure 4-14 setting or universal optimum. Keep it
unchanged between crop and full-image comparisons. With the defaults, the first
update at nu_max occurs at spatial index 8449 in a 256 x 256 crop, so the final
setting is reached. Adaptation reach is recorded for every encoding.

## Wavelength-aware grouping

The five inspected HYPSO files have 120 increasing band centers, approximately
387.85–802.52 nm. Native bands 4..118 (zero-based) lie within 400–800 nm. The
script reads these measured coordinates, checks their ordering, and records the
actual wavelengths and gap ranges instead of inventing uniformly spaced bands.

For a balanced comparison it selects **96 consecutive bands centered within the
valid 400–800 nm window**: native indices 13..108, about **433.99–764.91 nm**.
This is a subset of the 400–800 nm range, not every band in that interval.

| Native stride | Groups | Bands/group | Median separation |
|---|---:|---:|---:|
| 1 | 4 | 24 | 3.483 nm |
| 2 | 4 | 24 | 6.967 nm |
| 4 | 4 | 24 | 13.934 nm |

At stride k, divide the selected sequence into blocks of 24*k bands. Within
each block, take every phase r=0..k-1 at stride k. Every group contains 24 bands
in increasing spectral order. At each stride, the union of groups is EXACTLY
the same 96 native bands, each used once. Each group is encoded independently
with fresh predictor and encoder state. Average rate = total bits / total samples
over the four groups. This avoids changing the wavelength population or number
of bands per stream when changing spacing.

This changes band-center separation, not the sensor's optical spectral resolution.
There is no interpolation, spectral averaging, normalization, re-binning, added
noise, atmospheric correction, or application of radiometric/smile/destriping
matrices. Input samples remain original integer L1A DN. The wavelength vector is
the nominal one-dimensional calibration, not a per-pixel smile correction.

`--band-start` selects a different native zero-based start index.
`--wavelength-min` and `--wavelength-max` change the validity window deliberately.
For broader coverage, `--band-count 112 --group-bands 28 --strides 1 2 4` retains
more native bands but changes the book's 24-band group size. band-count must be
divisible by group-bands times EVERY stride. HYPSO cannot supply the book's five
spacings with equal 24-band groups from only 120 bands.

## Profile search: best tested, not a proven global optimum

The Green Book names an ad hoc algorithm for finding good band-dependent profiles
but does not specify it. This script uses a bounded, reproducible alternative:

1. For each scene/A/stride/group/P case, encode the zero baseline and uniform seeds.
2. Compute a zero-order Shannon entropy of mapped quantizer indices for each band
   in each seed encoding. This is a proposal score, NOT the plotted compressed rate.
3. Select the lowest-score seed pair for each band and assemble a band-dependent
   phi_z/psi_z profile. Re-encode it: feedback couples bands, so combining individual
   scores cannot be assumed to give the predicted total size.
4. Keep the lowest ACTUAL compressed size among baseline, seeds, and the measured
   proposal. If the proposal duplicates a seed, reuse its exact measurement.

The solid line is **best tested profile**. The dashed line is the zero baseline
at the same near-lossless A; zero phi/psi does not make A=3 lossless. A uniform
profile or the baseline may win, and overlapping curves mean the search found
no improvement, not proof that no better profile exists. The default proposal
can vary phi and psi across bands, but psi candidates are limited to 0 and 7.
Broaden `--phi-values` and `--psi-values` to investigate other candidates.

Selections are made on the same image being plotted, as an image-specific study.
A reusable onboard profile should later be evaluated on other scenes or regions.

## Cost, workers and restarting

The default has 5 images * 3 strides * 4 groups * 6 P values = **360 cases**.
Each has four seed encodes and at most one distinct proposal: **up to 1800 encodes**.
Adding A=15 doubles this; adding psi values expands the seed grid. `--dry-run`
prints the bound first. `--p-values 1 3 6` halves the default P sweep for a
smaller initial investigation.

One child process handles one complete configuration and then exits, releasing
its memory. Workers do not split a crop or share prediction state. The OS schedules
the processes; they are not pinned to physical cores. Numeric-library threads
are limited to one per process to avoid nested parallelism.

The compact hybrid adapter drops diagnostic text arrays but retains numeric
state, coding methods, binary output and termination. It checks the upstream
encoder source against the validated implementation at NTNU commit
`9fabb0371ebf93d72a0f83ca053e9cdb2c63e3c6`. It refuses changed encoder code;
do not bypass that check without validation. The predictor still keeps full
arrays, but each trial has 24 spectral bands rather than all 120.

Conservative allowances are roughly 1.18 GiB/job for 256 x 256 x 24 and P<=6,
and 7.28 GiB/job for 956 x 684 x 24. These are estimates, not measured peaks or
hard OS limits. The coordinator caps workers by the requested total budget,
available RAM, job count and CPU count. Start with two workers and inspect actual
memory use before increasing it. Runtime and memory are not plotted.

Completed encodes are saved atomically. Repeat the SAME command with `--resume`
to skip them. Worker count and memory budget may change; codec/input/crop/model
settings must match. A different experiment needs a new output folder. Do not
run two coordinators in the same output folder or edit cached/job files.
Source size/mtime and hashed cached DN provide input provenance.

```text
python tools/fig414/fig414_sweep.py --input-dir data --crop 256 256 --workers 4 --output results/fig414_256_A3 --resume
python tools/fig414/fig414_sweep.py --output results/fig414_256_A3 --plot-only
```

## Output

- `all_images_fig414.png/.pdf` and individual `<scene>_fig414.png/.pdf` plots.
- `summary.csv`: averaged baseline/best bits/sample and savings by P/spacing.
- `group_results.csv`: group rates and selected profiles.
- `best_profiles.csv`: phi_z, psi_z, normalized values, native indices and wavelengths.
- `evaluations.csv`: all measured configurations, coded size, error metrics and timing.
- `band_groups.csv`: the exact wavelength grouping.
- `manifest.json`, `cache/`, `jobs/`: provenance, cached DN, checkpoints and logs.

Only fully searched groups contribute to a point, and a point is withheld until
all groups for its P/spacing are complete. Partial figures are labelled. All
reported rates use actual header bits (including profile tables), payload,
entropy termination and output-word padding. The common external hybrid
initialization (accumulator=8 per band) is recorded but excluded from the primary
rate, consistent with your previous plots. Its full-array transmission cost is
also recorded. NetCDF file size and search computation are not counted.

Every encode checks encoder-side reconstruction against A. This is not an
independent decoder validation. The compact/original bitstreams and profile-header
round-trip were also checked; see VALIDATION.md.

## Sources and differences from the original experiment

Green Book section 4.2.2, printed page 4-14: processed AVIRIS-NG, 384 bands,
24-band groups, five spacings, P=1..6, nu_min=-1, nu_max=3, A=50 and added Gaussian
noise sigma=100. This HYPSO adaptation uses measured band spacing, four groups,
native DN, A<=15 and no added noise. The book's text is inconsistent about the
group count (16 groups, later “24 images”); this script uses explicit indices and
unambiguous sample-weighted aggregation.

- Green Book: https://ccsds.org/Pubs/120x2g2.pdf
- Blue Book, sample representatives and table syntax, sections 4.9 and 5.3.3.5:
  https://ccsds.org/Pubs/123x0b2e2c3.pdf
- Model: https://github.com/NTNU-SmallSat-Lab/ccsds123_issue_2_verification_model

Do not expect the original book's numerical curves, or assume P=3 from your
120-band lossless study must remain optimal in these near-lossless 24-band groups.
