"""Run the unchanged NTNU model with explicit settings and generate plots.

Run from the repository root (the script may be in tools/), or pass --repo.
For p-sweep, --p-values MAX_P tests every P from 0 through MAX_P, inclusive.
The reconstruction checks use the encoder's clipped quantizer-bin centres.
They are NOT an independent decode of the compressed bitstream.
"""
import argparse
import contextlib
import csv
import gc
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


def parse_raw(path):
    m = re.fullmatch(r"(.+)-u16be-(\d+)x(\d+)x(\d+)\.raw", path.name)
    if not m or "x" in m.group(1):
        raise ValueError("Use the BSQ u16be file produced by prepare_hypso.py.")
    nz, ny, nx = map(int, m.groups()[1:])
    if path.stat().st_size != nz * ny * nx * 2:
        raise ValueError("Input byte count does not agree with its filename dimensions.")
    original = np.fromfile(path, dtype=">u2").reshape(nz, ny, nx).transpose(1, 2, 0)
    return original.astype(np.int64)


def make_header(hd, raw, p, a, coder, mode, local_sum):
    h = hd.Header(raw.name)
    h.set_dynamic_range(16)
    h.set_encoding_order_bip()
    h.set_output_word_size(1)
    h.set_entropy_coder_type(getattr(hd.EntropyCoderType, coder.upper()))
    h.set_prediction_bands_num(p)
    h.set_prediction_mode(getattr(hd.PredictionMode, mode.upper()))
    h.set_local_sum_type(getattr(hd.LocalSumType, local_sum.upper()))
    h.set_register_size(64)
    h.set_weight_component_resolution(19)
    h.set_weight_update_change_interval(64)
    h.set_weight_update_initial_parameter(-1)
    h.set_weight_update_final_parameter(7)
    h.set_weight_init_method(hd.WeightInitMethod.DEFAULT)
    h.set_weight_exponent_offset_flag(hd.WeightExponentOffsetFlag.ALL_ZERO)
    # A=0 is lossless, with the same absolute-error header structure as A>0.
    h.set_quantizer_fidelity_control_method(hd.QuantizerFidelityControlMethod.ABSOLUTE_ONLY)
    h.set_periodic_error_updating_flag(hd.PeriodicErrorUpdatingFlag.NOT_USED)
    h.set_absolute_error_limit_assignment_method(hd.ErrorLimitAssignmentMethod.BAND_INDEPENDENT)
    h.set_absolute_error_limit_bit_depth(15)
    h.set_absolute_error_limit_value(a)
    h.set_relative_error_limit_value(0)
    h.set_sample_representative_flag(hd.SampleRepresentativeFlag.INCLUDED)
    h.set_sample_representative_resolution(3)
    h.set_fixed_damping_value(0)
    h.set_fixed_offset_value(7)
    h.set_damping_table_array_to_default()
    h.set_damping_offset_table_array_to_default()
    h.set_unary_length_limit(18)
    h.set_rescaling_counter_size(6)
    h.set_initial_count_exponent(1)
    h.set_accumulator_init_constant(3)
    h.set_accumulator_init_table_to_default()
    h.check_legal_config()
    return h


def run_one(args, hd, ccsds123, original, p, a, run_dir, common):
    run_dir.mkdir()
    config = dict(common, prediction_bands=p, absolute_error_limit=a)
    (run_dir / "configuration.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    h = make_header(hd, args.raw, p, a, args.coder, args.prediction_mode, args.local_sum)
    header, optional = h.get_header_bitstreams()
    if len(optional):
        raise RuntimeError("Unexpected optional tables in this starter configuration.")
    header_path = run_dir / "input_header.bin"
    header_path.write_bytes(header.tobytes())
    accumulator_path = run_dir / "input_accumulator.bin"
    if args.coder == "hybrid":
        from bitarray import bitarray
        # Explicit, fixed value of 8 for every band; same as this repo's default.
        accumulator = bitarray(f"{8:017b}" * original.shape[2])
        accumulator_path.write_bytes(accumulator.tobytes())
    old_cwd = Path.cwd()
    start = time.perf_counter()
    try:
        # Passing a basename avoids upstream parsing problems with directory names.
        os.chdir(args.raw.parent)
        with (run_dir / "model.log").open("w", encoding="utf-8") as log:
            with contextlib.redirect_stdout(log):
                model = ccsds123.CCSDS123(args.raw.name)
                model.output_folder = run_dir.as_posix()
                model.set_header_file(header_path.as_posix())
                if args.coder == "hybrid":
                    model.set_hybrid_accu_init_file(accumulator_path.as_posix())
                model.compress_image()
    finally:
        os.chdir(old_cwd)
    seconds = time.perf_counter() - start
    if not np.array_equal(model.image_sample, original):
        raise RuntimeError("The model loaded a different sample ordering from the source crop.")
    # A sample representative is predictor state, not the reconstructed output.
    reconstruction = model.predictor.clipped_quantizer_bin_center.copy()
    # The first spatial sample of each band is encoded without quantization.
    # Upstream leaves its bin-centre debug entry at the sentinel value -1.
    reconstruction[0, 0, :] = (model.predictor.predicted_sample_value[0, 0, :]
                                + model.predictor.quantizer_index[0, 0, :])
    error = reconstruction.astype(np.int64) - original
    max_error = int(np.max(np.abs(error)))
    if max_error > a:
        raise RuntimeError(f"Reconstruction error {max_error} exceeded A={a}.")
    reconstruction.transpose(2, 0, 1).astype(">u2").tofile(run_dir / "model_reconstruction.raw")
    compressed = run_dir / "z-output-bitstream.bin"
    compressed_bytes = compressed.stat().st_size
    side_bytes = sum((run_dir / name).stat().st_size for name in
                     ("hybrid_initial_accumulator.bin", "optional_tables.bin", "error_limits.bin"))
    row = {
        "P": p, "A": a, "sample_count": original.size,
        "raw_bytes": args.raw.stat().st_size,
        "compressed_bytes_including_header": compressed_bytes,
        "side_information_bytes": side_bytes,
        "bits_per_sample": 8 * compressed_bytes / original.size,
        "bits_per_sample_with_side_information": 8 * (compressed_bytes + side_bytes) / original.size,
        "compression_ratio": args.raw.stat().st_size / compressed_bytes,
        "max_abs_error_model": max_error,
        "rmse_model": float(np.sqrt(np.mean(error.astype(np.float64) ** 2))),
        "model_seconds_including_debug_write": seconds,
        "independent_bitstream_decode_verified": False,
    }
    (run_dir / "metrics.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
    if not args.keep_debug:
        # Only remove trace CSVs that this invocation generated in its new folder.
        for trace in run_dir.glob("*.csv"):
            trace.unlink()
    print(f"P={p:2d}, A={a:2d}: {row['bits_per_sample']:.4f} bits/sample, "
          f"CR={row['compression_ratio']:.3f}, max error={max_error}, {seconds:.2f} s", flush=True)
    del model, reconstruction, error
    gc.collect()
    return row


def make_plots(args, rows):
    if args.mode == "baseline":
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    key = "P" if args.mode == "p-sweep" else "A"
    xlabel = "Number of prediction bands, P" if key == "P" else "Absolute error limit, A (DN)"
    fig, ax = plt.subplots(figsize=(7.3, 4.7), layout="constrained")
    x = [r[key] for r in rows]
    ax.plot(x, [r["bits_per_sample"] for r in rows], "o-", color="#1b6588", lw=2)
    ax.set(xlabel=xlabel, ylabel="Compressed rate (bits/sample)", xticks=x)
    ax.set_title("HYPSO Adelaide: exploratory parameter sweep\n" + args.raw.name, fontsize=11)
    ax.grid(alpha=0.25)
    fig.savefig(args.output / "compressed_rate.png", dpi=180)
    plt.close(fig)
    if args.mode == "error-sweep":
        fig, ax = plt.subplots(figsize=(7.3, 4.7), layout="constrained")
        ax.plot(x, [r["max_abs_error_model"] for r in rows], "o-", label="Maximum absolute error")
        ax.plot(x, [r["rmse_model"] for r in rows], "s-", label="RMSE")
        ax.set(xlabel=xlabel, ylabel="Reconstruction error (DN)", xticks=x)
        ax.set_title("Encoder-side reconstruction check\nIndependent bitstream decoding remains to be verified", fontsize=11)
        ax.grid(alpha=0.25)
        ax.legend()
        fig.savefig(args.output / "reconstruction_error.png", dpi=180)
        plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("baseline", "p-sweep", "error-sweep"))
    p.add_argument("raw", type=Path)
    p.add_argument("--repo", type=Path, default=Path("."))
    p.add_argument("--output", type=Path, required=True, help="New results directory")
    p.add_argument("--p", type=int, default=3)
    p.add_argument("--p-values", dest="p_max", type=int, default=4, metavar="MAX_P",
                   help="Maximum P for p-sweep: test 0 through MAX_P inclusive (0-15; default: 4)")
    p.add_argument("--a-values", nargs="+", type=int, default=[0, 1, 2, 4, 8])
    p.add_argument("--coder", choices=("hybrid", "sample_adaptive"), default="hybrid")
    p.add_argument("--prediction-mode", choices=("reduced", "full"), default="reduced")
    p.add_argument("--local-sum", choices=("wide_column_oriented", "narrow_column_oriented",
                    "wide_neighbor_oriented", "narrow_neighbor_oriented"), default="wide_column_oriented")
    p.add_argument("--keep-debug", action="store_true", help="Keep all generated intermediate CSVs")
    args = p.parse_args()
    args.raw = args.raw.resolve()
    args.repo = args.repo.resolve()
    args.output = args.output.resolve()
    if not (args.repo / "ccsds123_i2_hlm" / "ccsds123.py").is_file():
        p.error("Run from the repository root, or provide --repo /path/to/repository.")
    if args.output.exists():
        p.error("The output directory already exists. Choose a fresh --output directory.")
    if any(v not in range(16) for v in (args.p, args.p_max)):
        p.error("Prediction-band values must be between 0 and 15.")
    if any(v not in range(32768) for v in args.a_values):
        p.error("This starter supports absolute error limits from 0 to 32767.")
    original = parse_raw(args.raw)
    sys.path.insert(0, str(args.repo))
    from ccsds123_i2_hlm import header as hd, ccsds123
    try:
        commit = subprocess.check_output(["git", "-C", str(args.repo), "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown (record the downloaded repository version manually)"
    common = {
        "repository_commit": commit, "python_version": platform.python_version(),
        "numpy_version": np.__version__, "input_filename": args.raw.name,
        "input_sha256": hashlib.sha256(args.raw.read_bytes()).hexdigest(),
        "input_shape_y_x_z": list(original.shape), "input_disk_order": "BSQ",
        "input_type": "u16be", "dynamic_range": 16, "encoding_order": "BIP",
        "entropy_coder": args.coder, "prediction_mode": args.prediction_mode,
        "local_sum": args.local_sum, "register_size": 64,
        "weight_resolution": 19, "weight_update_interval": 64,
        "weight_update_initial": -1, "weight_update_final": 7,
        "weight_initialization": "default", "weight_exponent_offsets": 0,
        "fidelity_control": "absolute_only", "relative_error_control": False,
        "periodic_error_updates": False, "absolute_limit_bit_depth": 15,
        "sample_representative_resolution": 3, "damping": 0, "offset": 7,
        "unary_limit": 18, "rescale_counter": 6, "initial_count_exponent": 1,
        "hybrid_initial_accumulator": 8, "sample_adaptive_K": 3,
        "output_word_bytes": 1,
        "rate_definition": "8 * complete compressed bitstream file bytes / original sample count",
        "reconstruction_check": "encoder clipped quantizer-bin centres; NOT independent bitstream decoding",
        "experimental_scope": "exploratory crop; Annex-C-inspired settings, not an exact figure reproduction",
    }
    args.output.mkdir(parents=True)
    (args.output / "study_configuration.json").write_text(json.dumps(common, indent=2), encoding="utf-8")
    pairs = ([(args.p, 0)] if args.mode == "baseline" else
             [(v, 0) for v in range(args.p_max + 1)] if args.mode == "p-sweep" else
             [(args.p, v) for v in args.a_values])
    rows = []
    for pred_bands, error_limit in pairs:
        row = run_one(args, hd, ccsds123, original, pred_bands, error_limit,
                      args.output / f"P{pred_bands:02d}_A{error_limit:05d}", common)
        rows.append(row)
        with (args.output / "results.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    make_plots(args, rows)
    print(f"Results: {args.output}")
    print("Error checks use encoder-side reconstruction, not an independent decoder.")


if __name__ == "__main__":
    main()