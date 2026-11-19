# Validation — 18 November 2026

Reference model: NTNU `ccsds123_issue_2_verification_model`, commit
`9fabb0371ebf93d72a0f83ca053e9cdb2c63e3c6`.
Executed with Python 3.12, NumPy 1.26.0, bitarray 2.8.2 and psutil 5.9.5.

## Checks performed

1. **Signed weight quantization:** 63 cases across Omega=6,12,19 and Q values
   including 3 and Omega+3. Compared the implementation with an independent
   two's-complement bit-string construction of the Q MSBs and prescribed suffix.

2. **Real DN execution:** Adelaide HYPSO-2 L1A, all 120 bands, eight columns,
   two 16-row training strips at source rows 100 and 200, each immediately followed
   by test heights 1,4,8. All 14 training/test runs completed; reconstruction was
   exactly lossless. These small checks are not the five-image scientific study.

3. **Final training weights:** Ran the unmodified upstream predictor on each
   training strip plus one additional row. The exported post-final-sample weights
   exactly equalled the weights used at the first pixel of the additional row.
   This catches exporting the final trace entry before its last update.

4. **Direct reference comparison:** Loaded the saved serialized headers for
   four real-DN test cases (default/custom, both locations), then ran the original
   predictor and sample-adaptive encoder directly, without the study observer.
   Payload SHA-256, padded bit count, bits/sample and lossless reconstruction
   exactly matched. The sample-adaptive accumulator initialization was K=3.

5. **Custom table accounting:** The 120-band full-mode P=3/Q=9 cases encoded
   714 signed coefficients. Verified 6426 table data bits, 6432 bits after byte
   padding, header round-trip and the actual initial weights inside the predictor.
   Separate executable checks also passed at Q=3 and Q=22.

6. **Full pilot workflow:** Executed all 60 tests plus 10 training jobs on five
   explicitly tagged software fixtures. The fixtures contain different narrow
   Adelaide subsets, use four bands, have different native widths (4 through 8),
   and retain heights sufficient for the default 16/64/256 sweep. Their filenames
   exercise the scene selection interface; they are NOT actual Lisbon, Bangladesh,
   Santarem, Doha or Finnmark measurements.

7. **Serial/parallel equivalence:** Repeated that entire pilot with one worker
   and two workers. All 60 test payload hashes, header hashes, padded sizes,
   bits/sample, crop hashes and maximum reconstruction errors matched exactly.
   Recorded peak concurrency was one and two, respectively.

8. **Operational checks:** Resume skipped all completed jobs, including when
   changing worker count. Plot-only succeeded. Changing the experiment was
   rejected for an existing output directory. Dry-run wrote no results. Invalid
   crop coordinates were rejected. A 1.1 GiB estimate budget admitted one worker
   when four were requested on tiny test inputs. All 30 paired comparisons and
   15 location averages matched independent arithmetic.

9. **Outputs:** PNG/PDF plots were generated; the primary plot was visually
   inspected. Python syntax compilation passed for all three bundled modules.

## Scope

No full-width five-scene scientific run was executed here. The user will generate
those results using the supplied command. The code counts the Issue-2 model's
actual header, so numerical overhead need not exactly match the software used
for the published Green Book figure. Two locations and the mean of paired
percentage reductions are explicit pilot choices; the book does not give its
exact location count or averaging convention.

The study checks encoder-side reconstruction and consistency with the reference
model. It does not include an independent decoder or claim formal CCSDS conformance.
It does not benchmark FPGA throughput. Memory budgets are conservative estimates,
not measured guarantees or OS-enforced limits.
