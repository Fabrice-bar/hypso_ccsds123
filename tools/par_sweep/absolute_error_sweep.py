#!/usr/bin/env python3
r"""CCSDS 123.0-B-2 ABSOLUTE ERROR LIMIT (A) sweep on HYPSO-1 L1A crops.

Based on predictor_sweep.py, retaining the original five HYPSO scenes and
exact integer-DN loading. P is held fixed (default P=3), while A is varied.

Usage, from the verification model repository root:
  .\.venv\Scripts\python.exe tools\absolute_error_sweep.py --images lisbon --a-values 0 1 2 3 4 7 15 --sums WIDE_NEIGHBOR_ORIENTED --modes FULL
  .\.venv\Scripts\python.exe tools\absolute_error_sweep.py --a-values 0 1 2 3 4 7 15
  .\.venv\Scripts\python.exe tools\absolute_error_sweep.py --a-max 15 --p 3

A=0 uses genuine LOSSLESS control and psi=0; A>0 uses ABSOLUTE_ONLY with
band-independent limit A and repo default sample representative offset psi=7.
All settings apart from the fidelity mode/error limit and required offset are fixed.
These differences mean A=0 is a LOSSLESS baseline, rather than an ABSOLUTE_ONLY
header with A=0. The fixed damping parameter phi remains 3.

The model computes compression rate and encoder-side reconstructed DN errors.
Its reconstruction is NOT a test of the output of an independent decoder.
For scientific fidelity, validate on decompressed full-size images later.

Saves results.csv, metadata.json, per-image graphs, and all_images.png/pdf.
Same command resumes interrupted runs. Never mix this output with a P-sweep.

Requires sweep_workers.py beside this file. --workers 2 (default) runs two
complete configurations concurrently; use --workers 1 for serial execution.
"""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback
from contextlib import redirect_stdout
from enum import Enum

import numpy as np

import sweep_workers as sw


SCENES = {
    "lisbon": "lisbon_2022-09-22T10-55-19Z-l1a.nc",
    "bangladesh": "bangladesh_2022-10-26T04-02-56Z-l1a.nc",
    "santarem": "santarem_2022-09-24T13-52-04Z-l1a.nc",
    "doha": "doha_2022-09-12T06-38-31Z-l1a.nc",
    "finnmark": "finnmark_2022-07-06T19-40-03Z-l1a.nc",
}
SUM_TYPES = {
    "WIDE_NEIGHBOR_ORIENTED": "Wide neighbour",
    "NARROW_NEIGHBOR_ORIENTED": "Narrow neighbour",
    "WIDE_COLUMN_ORIENTED": "Wide column",
    "NARROW_COLUMN_ORIENTED": "Narrow column",
}
MODES = ("FULL", "REDUCED")
FIELDS = [
    "image", "capture", "a", "p", "local_sum", "prediction_mode", "rows", "columns",
    "bands", "sample_count", "encoder", "header_bits", "payload_bits",
    "padding_bits", "stream_bits", "external_info_bits", "bits_per_sample",
    "bits_per_sample_with_external_info", "compression_ratio", "max_error",
    "mae", "rmse", "psnr_db", "mean_sam_deg", "p95_sam_deg",
    "predictor_seconds", "encoder_seconds",
]
VERSION = "a-sweep-1.0-mp1"
SERIAL_VERSION = "a-sweep-1.0"
SERIAL_DIGESTS = (
    "37d83a1c4d10343e8ea9df74142992ead0ba4b8d6c624daca627625ee43d1b04",
    "c367cde9f651f108d3b7b7a47cf1f020d8845714320b264dfa507023f842f896",
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", type=Path, default=Path("data/hypso"))
    p.add_argument("--images", nargs="+", choices=list(SCENES), default=list(SCENES))
    p.add_argument("--p", type=int, default=3, help="Fixed number of prediction bands (0..15).")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--a-values", type=int, nargs="+", metavar="A",
                       help="Explicit A values, e.g. 0 1 2 3 4 7 15 31.")
    group.add_argument("--a-max", type=int, metavar="A_MAX",
                       help="Test ALL A values from 0 through A_MAX inclusive.")
    p.add_argument("--sums", nargs="+", choices=list(SUM_TYPES), default=list(SUM_TYPES),
                   help="Local sum types to test; default all four.")
    p.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES),
                   help="Prediction modes to test; default FULL and REDUCED.")
    p.add_argument("--crop", type=int, nargs=2, metavar=("ROWS", "COLS"), default=(32, 32))
    p.add_argument("--row", type=int, help="Zero-based first row; default: centred.")
    p.add_argument("--col", type=int, help="Zero-based first column; default: centred.")
    p.add_argument("--full-image", action="store_true", help="Use every spatial sample (large RAM use).")
    p.add_argument("--variable", default="products/dn", help="NetCDF raw-DN variable path.")
    p.add_argument("--axis-order", choices=("yxz", "yzx", "xyz", "xzy", "zxy", "zyx"),
                   help="Storage-axis order, only needed for unrecognised NetCDF dimension names.")
    p.add_argument("--repo", type=Path, help="Verification repo root; normally detected automatically.")
    p.add_argument("--output", type=Path, default=Path("results/absolute_error_sweep"))
    p.add_argument("--max-memory-gb", type=float, default=4.0,
                   help="Conservative per-run RAM budget in GiB; default 4.0.")
    p.add_argument("--overwrite", action="store_true", help="Replace this script's previous results.")
    p.add_argument("--plot-only", action="store_true", help="Plot saved CSV without rerunning the model.")
    p.add_argument("--dry-run", action="store_true", help="Validate inputs/settings and print the plan only.")
    sw.add_options(p)
    a = p.parse_args()
    if not 0 <= a.p <= 15:
        p.error("--p must be between 0 and 15.")
    if a.a_max is not None and not 0 <= a.a_max <= 31:
        p.error("--a-max must be 0..31 (the model's default DA=5 allows A<=31).")
    a.a_values = sorted(set(a.a_values if a.a_values is not None else
                            (range(a.a_max + 1) if a.a_max is not None else [0, 1, 2, 3, 4, 7, 15, 31])))
    if not a.a_values or a.a_values[0] < 0 or a.a_values[-1] > 31:
        p.error("--a-values must all be in 0..31 with default DA=5.")
    a.sums = list(dict.fromkeys(a.sums))
    a.modes = list(dict.fromkeys(a.modes))
    if min(a.crop) < 2 or a.max_memory_gb <= 0:
        p.error("Crop dimensions must be at least 2 and memory budget must be positive.")
    if any(v is not None and v < 0 for v in (a.row, a.col)):
        p.error("--row and --col must be nonnegative.")
    if a.full_image and (a.row is not None or a.col is not None):
        p.error("--full-image cannot be combined with --row or --col.")
    a.images = list(dict.fromkeys(a.images))
    return a


def find_repo(explicit):
    candidates = [explicit] if explicit else [Path.cwd(), Path(__file__).resolve().parent.parent]
    for root in candidates:
        if root and (root / "ccsds123_i2_hlm" / "header.py").is_file():
            return root.resolve()
    raise ValueError("Cannot find ccsds123_i2_hlm. Run from the repo root, or supply --repo PATH.")


def file_digest(paths, root):
    h = hashlib.sha256()
    for path in sorted(paths):
        h.update(path.relative_to(root).as_posix().encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def json_value(value):
    if isinstance(value, Enum):
        return value.name
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"Cannot record configuration value of type {type(value).__name__}")


def header_snapshot(header):
    keys = set(vars(type(header))) | set(vars(header))
    return {k: json_value(getattr(header, k)) for k in sorted(keys)
            if not k.startswith("_") and not callable(getattr(header, k))
            and k not in ("header_bitstream", "optional_tables_bitstream")}


def find_capture(folder, name):
    matches = sorted(folder.rglob(name))
    if len(matches) != 1:
        raise ValueError(f"Expected one {name} under {folder}; found {len(matches)}. "
                         "Keep the original filename or edit SCENES at the top of this script.")
    return matches[0].resolve()


def read_cube(path, args):
    from netCDF4 import Dataset
    aliases = {
        "lines": "y", "line": "y", "frames": "y", "frame": "y", "rows": "y", "y": "y",
        "samples": "x", "sample": "x", "columns": "x", "cols": "x", "x": "x",
        "bands": "z", "band": "z", "wavelength": "z", "wavelengths": "z", "z": "z",
    }
    with Dataset(path) as ds:
        try:
            var = ds[args.variable]
        except (KeyError, IndexError) as exc:
            raise ValueError(f"{path.name}: missing raw variable {args.variable!r}.") from exc
        if var.ndim != 3 or np.dtype(var.dtype).kind not in "iu":
            raise ValueError(f"{path.name}: expected a 3-D integer DN array, got {var.shape}, {var.dtype}.")
        order = args.axis_order or "".join(aliases.get(d.lower(), "?") for d in var.dimensions)
        if len(order) != 3 or set(order) != set("xyz"):
            raise ValueError(f"Unrecognised dimensions {var.dimensions}; specify --axis-order explicitly.")
        if float(getattr(var, "scale_factor", 1)) != 1 or float(getattr(var, "add_offset", 0)) != 0:
            raise ValueError("Packed/scaled data found; choose the original integer DN variable.")
        shape = tuple(int(var.shape[order.index(axis)]) for axis in "yxz")
        height, width = shape[:2] if args.full_image else tuple(args.crop)
        row = 0 if args.full_image else (args.row if args.row is not None else (shape[0] - height) // 2)
        col = 0 if args.full_image else (args.col if args.col is not None else (shape[1] - width) // 2)
        if row < 0 or col < 0 or row + height > shape[0] or col + width > shape[1]:
            raise ValueError(f"Crop ({row}, {col}, {height}, {width}) is outside {path.name}: {shape}.")
        if min(height, width) < 2:
            raise ValueError("Use at least two rows and two columns for this comparison.")
        # Conservative allowance for the verification model's many trace arrays.
        estimated_gib = height * width * shape[2] * (2200 + 16 * (args.p + 3)) / 2**30 + 0.15
        if estimated_gib > args.max_memory_gb:
            raise ValueError(f"{path.name}: estimated RAM {estimated_gib:.1f} GiB exceeds "
                             f"--max-memory-gb {args.max_memory_gb}. Use a smaller crop.")
        selectors = {"y": slice(row, row + height), "x": slice(col, col + width), "z": slice(None)}
        var.set_auto_maskandscale(False)
        raw = var[tuple(selectors[axis] for axis in order)]
        for attr in ("_FillValue", "missing_value"):
            if attr in var.ncattrs():
                for missing in np.atleast_1d(var.getncattr(attr)):
                    if np.any(raw == missing):
                        raise ValueError(f"{path.name}: selected crop contains {attr}={missing}; choose a valid region.")
        cube = np.ascontiguousarray(np.transpose(raw, tuple(order.index(a) for a in "yxz")), dtype=np.int64)
        info = {"file": str(path), "file_bytes": path.stat().st_size,
                "variable": args.variable, "storage_dimensions": list(var.dimensions),
                "storage_dtype": str(var.dtype), "storage_axis_order": order,
                "source_shape_yxz": list(shape), "crop_origin_yx": [row, col],
                "crop_shape_yxz": list(cube.shape), "min_dn": int(cube.min()),
                "max_dn": int(cube.max()), "estimated_ram_gib": round(estimated_gib, 3),
                "crop_sha256": hashlib.sha256(cube.astype("<i8", copy=False).tobytes()).hexdigest()}
        return cube, info


def make_base_header(cube, hd, p):
    y, x, z = cube.shape
    # Filename provides only dimensions/type. No RAW copy is written.
    h = hd.Header(f"cube-u16be-{z}x{y}x{x}.raw")
    h.prediction_bands_num = p
    h.quantizer_fidelity_control_method = hd.QuantizerFidelityControlMethod.LOSSLESS
    h.periodic_error_updating_flag = hd.PeriodicErrorUpdatingFlag.NOT_USED
    h.error_update_period_exponent = 0
    h.fixed_offset_value = 0
    h.damping_offset_table_array = np.zeros(z, dtype=np.int64)
    if h.weight_init_method != hd.WeightInitMethod.DEFAULT:
        raise ValueError("This study requires default weight initialization.")
    if h.weight_exponent_offset_flag != hd.WeightExponentOffsetFlag.ALL_ZERO:
        raise ValueError("This study requires zero weight exponent offsets.")
    h.check_legal_config()
    limit = 2**h.get_dynamic_range_bits() - 1
    if int(cube.min()) < 0 or int(cube.max()) > limit:
        raise ValueError(f"DN samples outside unsigned D={h.get_dynamic_range_bits()} range.")
    return h


def reconstruction_metrics(cube, pred, a, dynamic_range_bits):
    # First spatial sample of each band is special in upstream model: t=0.
    reconstructed = pred.clipped_quantizer_bin_center.copy()
    reconstructed[0, 0, :] = (pred.predicted_sample_value[0, 0, :]
                               + pred.quantizer_index[0, 0, :])
    errors = cube - reconstructed
    abs_errors = np.abs(errors)
    max_error = int(abs_errors.max())
    if max_error > a:
        raise RuntimeError(f"Near-lossless bound violated: max error={max_error} > A={a}")
    if np.any(pred.maximum_error != a):
        raise RuntimeError("Predictor did not apply the expected uniform absolute error limit.")
    mae = float(abs_errors.mean())
    mse = float(np.mean(np.square(errors, dtype=np.float64)))
    rmse = float(np.sqrt(mse))
    psnr = float('inf') if rmse == 0 else float(20 * np.log10((2**dynamic_range_bits - 1) / rmse))

    if max_error == 0:
        mean_sam = p95_sam = 0.0
    else:
        # SAM is calculated over RAW DN spectral vectors, not calibrated radiance.
        raw_spectra = cube.reshape(-1, cube.shape[-1]).astype(np.float64)
        rec_spectra = reconstructed.reshape(-1, cube.shape[-1]).astype(np.float64)
        n1 = np.linalg.norm(raw_spectra, axis=1)
        n2 = np.linalg.norm(rec_spectra, axis=1)
        valid = (n1 > 0) & (n2 > 0)
        angles = np.degrees(np.arccos(np.clip(
            np.sum(raw_spectra[valid] * rec_spectra[valid], axis=1) /
            (n1[valid] * n2[valid]), -1.0, 1.0)))
        mean_sam = float(np.mean(angles)) if angles.size else float('nan')
        p95_sam = float(np.percentile(angles, 95)) if angles.size else float('nan')
    return {"max_error": max_error, "mae": mae, "rmse": rmse,
            "psnr_db": psnr, "mean_sam_deg": mean_sam, "p95_sam_deg": p95_sam}


def encode_one(cube, base, a, local_sum, mode, model):
    hd, constants, predictor, encoder_classes = model
    h = copy.deepcopy(base)
    h.local_sum_type = getattr(hd.LocalSumType, local_sum)
    h.prediction_mode = getattr(hd.PredictionMode, mode)
    if a == 0:
        # Match the genuine lossless baseline of the original P sweep.
        h.quantizer_fidelity_control_method = hd.QuantizerFidelityControlMethod.LOSSLESS
        h.fixed_offset_value = 0
        h.damping_offset_table_array.fill(0)
    else:
        # MUST change both the encoded scalar and per-band operational table.
        h.quantizer_fidelity_control_method = hd.QuantizerFidelityControlMethod.ABSOLUTE_ONLY
        h.absolute_error_limit_assignment_method = hd.ErrorLimitAssignmentMethod.BAND_INDEPENDENT
        h.absolute_error_limit_value = a
        h.absolute_error_limit_table.fill(a)
        # Restore upstream near-lossless default psi=7 (Theta=3).
        h.fixed_offset_value = 7
        h.damping_offset_table_array.fill(7)
    h.check_legal_config()
    image_constants = constants.ImageConstants(h)
    started = time.perf_counter()
    pred = predictor.Predictor(h, image_constants, cube)
    pred.run_predictor()
    predictor_seconds = time.perf_counter() - started
    quality = reconstruction_metrics(cube, pred, a, h.get_dynamic_range_bits())
    mapped = pred.get_predictor_output()
    del pred
    gc.collect()

    encoder = encoder_classes[h.entropy_coder_type](h, image_constants, mapped)
    started = time.perf_counter()
    encoder.run_encoder()
    encoder_seconds = time.perf_counter() - started
    header_bits, optional_bits = h.get_header_bitstreams()
    payload_count = len(encoder.bitstream)
    word_bits = 8 * (h.output_word_size or 8)
    padding = (-(len(header_bits) + payload_count)) % word_bits
    stream_bits = len(header_bits) + payload_count + padding
    external_bits = ((len(optional_bits) + 7) // 8) * 8
    if h.entropy_coder_type == hd.EntropyCoderType.HYBRID:
        n = h.z_size * (h.get_dynamic_range_bits() + encoder.initial_count_exponent)
        external_bits += ((n + 7) // 8) * 8
    result = {"a": a, "p": h.prediction_bands_num,
              "local_sum": local_sum, "prediction_mode": mode,
              "rows": cube.shape[0], "columns": cube.shape[1], "bands": cube.shape[2],
              "sample_count": cube.size, "encoder": h.entropy_coder_type.name,
              "header_bits": len(header_bits), "payload_bits": payload_count,
              "padding_bits": padding, "stream_bits": stream_bits,
              "external_info_bits": external_bits, "bits_per_sample": stream_bits / cube.size,
              "bits_per_sample_with_external_info": (stream_bits + external_bits) / cube.size,
              "compression_ratio": h.get_dynamic_range_bits() * cube.size / stream_bits,
              **quality, "predictor_seconds": predictor_seconds, "encoder_seconds": encoder_seconds}
    del encoder, mapped
    gc.collect()
    return result


def write_json(path, content):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(content, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def write_csv(path, rows):
    tmp = path.with_suffix(".csv.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def read_csv(path):
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != FIELDS:
            raise ValueError("CSV schema differs; choose a new output folder.")
        return list(reader)


def job_key(row):
    return row["image"], int(row["a"]), row["local_sum"], row["prediction_mode"]


def plot_results(rows, metadata, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    from matplotlib.lines import Line2D
    if not rows:
        return
    colours = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")
    scenes = [s for s in metadata["study"]["images"] if any(r["image"] == s for r in rows)]
    a_vals = sorted({int(r["a"]) for r in rows})
    sum_selected = metadata["study"]["sums"]
    modes_selected = metadata["study"]["modes"]
    handles = [Line2D([], [], color=colours[list(SUM_TYPES).index(s)],
                      linestyle="-" if mode == "FULL" else "--",
                      marker="o" if mode == "FULL" else "s", markersize=4,
                      label=f"{SUM_TYPES[s]} / {mode.lower()}")
               for s in sum_selected for mode in modes_selected]

    def draw(ax, scene, field):
        for sum_name in sum_selected:
            colour = colours[list(SUM_TYPES).index(sum_name)]
            for mode in modes_selected:
                vals = {int(r["a"]): float(r[field]) for r in rows
                        if r["image"] == scene and r["local_sum"] == sum_name
                        and r["prediction_mode"] == mode}
                ax.plot(a_vals, [vals.get(a, np.nan) for a in a_vals],
                        color=colour, linestyle="-" if mode == "FULL" else "--",
                        linewidth=1.5, marker="o" if mode == "FULL" else "s",
                        markersize=4, markerfacecolor=colour if mode == "FULL" else "none")
        info = metadata["study"]["inputs"][scene]
        shape = info["crop_shape_yxz"]
        completed = sum(r["image"] == scene for r in rows)
        expected = len(metadata["requested_a_values"]) * len(sum_selected) * len(modes_selected)
        status = "" if completed == expected else f" | partial: {completed}/{expected}"
        ax.set_title(f"{scene.title()} | {shape[0]} x {shape[1]} pixels, "
                     f"{shape[2]} bands{status}", fontsize=10)
        ax.set_xlabel("A (maximum absolute reconstruction error, DN)")
        ax.set_ylabel({"bits_per_sample": "Compressed bits/sample",
                       "rmse": "RMSE (DN)", "mean_sam_deg": "Mean SAM (degrees)"}[field])
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True, alpha=0.25)
        ax.set_axisbelow(True)

    for field, suffix in [("bits_per_sample", ""), ("rmse", "_rmse"),
                          ("mean_sam_deg", "_sam")]:
        for scene in scenes:
            fig, ax = plt.subplots(figsize=(9, 6))
            draw(ax, scene, field)
            fig.suptitle(f"CCSDS-123 A sweep | P={metadata['study']['p']}", fontsize=13)
            fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=9)
            fig.tight_layout(rect=(0, 0.18, 1, 0.95))
            for ext in ("png", "pdf"):
                fig.savefig(out / f"{scene}{suffix}.{ext}", dpi=180, bbox_inches="tight")
            plt.close(fig)
        nrows = (len(scenes) + 1) // 2
        fig, axes = plt.subplots(nrows, 2, figsize=(13, 3.7 * nrows + 1.6), squeeze=False)
        for ax, scene in zip(axes.flat, scenes):
            draw(ax, scene, field)
        for ax in list(axes.flat)[len(scenes):]:
            ax.set_visible(False)
        fig.suptitle(f"CCSDS-123 absolute error sweep | P={metadata['study']['p']} | {field}", fontsize=14)
        fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=9)
        fig.tight_layout(rect=(0, 0.11, 1, 0.96))
        for ext in ("png", "pdf"):
            fig.savefig(out / f"all_images{suffix}.{ext}", dpi=180, bbox_inches="tight")
        plt.close(fig)


def _evaluate(key, cubes, model, state):
    args, bases = state
    scene, value, local_sum, mode = key
    row = encode_one(cubes[scene], bases[scene], value, local_sum, mode, model)
    row.update(image=scene, capture=SCENES[scene])
    return row


def main():
    args = parse_args()
    if args.dry_run:
        return _main(args)
    if args.plot_only and not args.output.is_dir():
        raise ValueError(f"No saved output directory: {args.output}")
    with sw.output_lock(args.output.resolve()):
        return _main(args)


def _main(args):
    out = args.output.resolve()
    csv_path, meta_path = out / "results.csv", out / "metadata.json"
    if args.plot_only:
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        plot_results(read_csv(csv_path), metadata, out)
        print(f"Plots: {out}")
        return
    if not __debug__:
        raise ValueError("Run Python without -O: the verification model relies on assertions.")
    repo = find_repo(args.repo)
    hd = sw.load_model(repo, 'nc')[0]
    cubes, inputs, bases = {}, {}, {}
    for scene in args.images:
        path = find_capture(args.input_dir.resolve(), SCENES[scene])
        cubes[scene], inputs[scene] = read_cube(path, args)
        bases[scene] = make_base_header(cubes[scene], hd, args.p)
        print(f"{scene}: crop y,x={inputs[scene]['crop_origin_yx']}, "
              f"shape y,x,z={cubes[scene].shape}, DN={inputs[scene]['min_dn']}..{inputs[scene]['max_dn']}")
    package = repo / "ccsds123_i2_hlm"
    model_files = list(package.rglob("*.py")) + list(package.rglob("*.txt"))
    # RAM estimates depend on requested P but are not part of experimental identity.
    identity_inputs = copy.deepcopy(inputs)
    for info in identity_inputs.values():
        info.pop("estimated_ram_gib")
    study = {"script_version": VERSION, "worker_helper_sha256": sw.helper_sha(), "p": args.p, "sums": args.sums, "modes": args.modes,
             "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             "model_sha256": file_digest(model_files, repo),
             "python": platform.python_version(), "numpy": np.__version__,
             "images": args.images, "inputs": identity_inputs,
             "base_headers": {s: header_snapshot(h) for s, h in bases.items()},
             "metric": "CCSDS header + entropy payload/tail + output-word padding, divided by scalar samples",
             "error_check": "encoder-side reconstruction only; no independent decoder",
             "fidelity_policy": "A=0: LOSSLESS, psi=0; A>0: ABSOLUTE_ONLY, psi=7, phi unchanged",
             "validation": "encoder-side bin center error <= A; no independent decoder"}
    metadata = {"study": study, "requested_a_values": args.a_values,
                "local_sum_types": args.sums, "prediction_modes": args.modes, "model_path": str(repo)}
    example = next(iter(bases.values()))
    print(f"Fixed encoder: {example.entropy_coder_type.name}; D={example.get_dynamic_range_bits()}, "
          f"Omega={example.get_weight_component_resolution()}, R={example.register_size or 64}, "
          f"nu_min={example.get_weight_update_initial_parameter()}, "
          f"nu_max={example.get_weight_update_final_parameter()}, "
          f"t_inc={2**(example.weight_update_change_interval+4)}, "
          f"phi={example.fixed_damping_value}; A=0: lossless/psi=0; A>0: absolute-only/psi=7.")
    print(f"Fixed P={args.p}; A={args.a_values}; {len(args.images)} images "
          f"x {len(args.sums)} sums x {len(args.modes)} modes x {len(args.a_values)} A values "
          f"= {len(args.images)*len(args.sums)*len(args.modes)*len(args.a_values)} configurations.")
    jobs = [(s, a, st, mode) for s in args.images for a in args.a_values
            for st in args.sums for mode in args.modes]
    max_job_gib = max(i["estimated_ram_gib"] for i in inputs.values())
    if args.dry_run:
        sw.plan(args, len(jobs), max_job_gib, cubes)
        print("Dry run: inputs/settings checked; estimate for a fresh run. No outputs changed.")
        return
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    if not args.overwrite and (meta_path.exists() or csv_path.exists()):
        if not meta_path.exists():
            raise ValueError("CSV exists without metadata. Choose a new --output folder.")
        old = json.loads(meta_path.read_text(encoding="utf-8"))
        if not sw.matched_study(old, metadata, SERIAL_VERSION, SERIAL_DIGESTS):
            raise ValueError("Inputs, crop, model or fixed settings differ from the saved run. "
                             "Choose a new --output folder, or use --overwrite to replace its results.")
        rows = read_csv(csv_path)
        if any(int(r["a"]) not in args.a_values for r in rows):
            raise ValueError("Saved results contain A values not in this sweep; use matching --a-values or a new output folder.")
    pending = sw.pending_keys(jobs, rows, job_key)
    workers = sw.plan(args, len(pending), max_job_gib, cubes)
    metadata["execution"] = {"requested_workers": args.workers, "active_workers": workers,
                             "completed_at_start": len(rows), "total_memory_gib": args.total_memory_gb}
    if args.overwrite:
        # Only this script's named outputs are replaced, never the entire folder.
        for stem in [*SCENES, "all_images"]:
            for suffix in ("", "_rmse", "_sam"):
                for ext in ("png", "pdf"):
                    (out / f"{stem}{suffix}.{ext}").unlink(missing_ok=True)
    write_json(meta_path, metadata)
    write_csv(csv_path, rows)
    print(f"Complete: {len(rows)}; remaining: {len(pending)}. Results: {out}")
    sw.collect(jobs, rows, pending, workers, cubes, repo, "nc", _evaluate, (args, bases),
               get_key=job_key, save=lambda rs: write_csv(csv_path, rs),
               plot=lambda rs: plot_results(rs, metadata, out), out=out)
    print(f"Saved {len(rows)}/{len(jobs)} configurations. Plots and CSV: {out}")
    print("Error checks use encoder-side bin centers, not an independent decoder.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130)
    except (ValueError, FileNotFoundError, ModuleNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
