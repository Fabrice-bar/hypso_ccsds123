#!/usr/bin/env python3
r"""HYPSO damping/adaptation sweep following CCSDS Green Book Figure 4-9.

Place this file and sweep_workers.py in the same directory inside the repository.
Optional --low-memory also requires the supplied hybrid_no_traces.py alongside it.
Run from the repository root, or pass --repo PATH.

Defaults: five HYPSO scenes; centre 128x128 crops; ALL native bands of products/dn;
raw integer DN; P=3; narrow-neighbour sums; reduced prediction; HYBRID encoder;
D=16, Omega=19, R=64, Theta=3; nu_min=-6; t_inc=128; psi=7 (forced 0 for A=0).
Sweep nu_max=0..9, phi=0..5, A=0,3,15: 900 complete configurations.
This follows the figure's analysis design, with your HYPSO data and profile.
The Green Book's third error limit is 20; this study defaults to your A=15.

Examples (from repo root; PowerShell uses .\.venv\Scripts\python.exe):
  python tools/fig49/damping_sweep.py --input-dir data --workers 2
  python tools/fig49/damping_sweep.py --input-dir data --low-memory --workers 2
  python tools/fig49/damping_sweep.py --images lisbon --crop 8 8 --phi-values 0 3 5 --vmax-min 0 --vmax-max 2 --a-values 0 3 --output results/fig49_smoke

Use --crop-from previous/metadata.json instead of --crop/--row/--col to reuse
and verify exact crop origins and hashes. No calibration, binning or band removal.
Short crops may never reach the larger requested nu_max values; these are marked
with hollow symbols and shading and recorded in the CSV. Their identical rates
must not be interpreted as measured steady-state performance at those values.

Outputs: results.csv, metadata.json, best_by_phi.csv, best_overall.csv,
all_images_damping.png/.pdf (rows=scenes, columns=A), and one corresponding
three-panel figure per scene. Curves represent phi; x is nu_max; y is bits/sample.
Repeat the same command to resume. --plot-only needs only the output folder.
Use a NEW output folder when changing ranges, settings, crops, storage mode or code.
--dry-run validates inputs and reports counts/memory without encoding or writing.
--overwrite restarts only this script's known outputs; it is never implicit.

Rate includes CCSDS header, entropy payload/tail and output-word padding, divided
by scalar sample count. External initialization/tables are excluded from the main
rate; inclusive rate is also recorded. The script stores metrics, not full trace
CSV files. Error checks use encoder-side reconstruction, NOT an independent decoder.
Python timing is not FPGA throughput. Memory budgets are estimates, not OS limits.
Two workers default; --workers 1 runs serially; larger counts require more RAM.
"""

from __future__ import annotations

import argparse
import copy
import csv
import errno
import gc
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
import traceback
from contextlib import contextmanager, redirect_stdout
from enum import Enum

import numpy as np

import sweep_workers as sw

VERSION = "1.0.0-fig49"

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
    if not folder.is_dir():
        raise ValueError(f"Input folder does not exist: {folder}. Check --input-dir.")
    direct = folder / name
    if direct.is_file():
        return direct.resolve()
    matches = sorted(path for path in folder.rglob(name) if path.is_file())
    if not matches:
        raise ValueError(f"Cannot find {name} in {folder} or its subfolders. "
                         "Check --input-dir and keep the original filename.")
    if len(matches) > 1:
        locations = "\n".join(f"  {path.resolve()}" for path in matches)
        raise ValueError(f"Found {len(matches)} copies of {name} in subfolders. "
                         "Set --input-dir to the folder containing your chosen copy:\n"
                         + locations)
    return matches[0].resolve()

@contextmanager
def open_netcdf(path):
    """Read an existing local file despite native-library path encoding bugs."""
    from netCDF4 import Dataset
    path = Path(path)
    buffer = None  # Keep memory alive until Dataset.close() has finished.
    try:
        ds = Dataset(str(path), mode="r")
    except OSError as exc:
        # Retry only filename/access errors on existing non-ASCII paths.
        # Missing files, ASCII-path errors and format errors remain errors.
        if (exc.errno not in (errno.ENOENT, errno.EACCES)
                or str(path).isascii() or not path.is_file()):
            raise
        buffer = path.read_bytes()  # Python handles Unicode Windows paths.
        try:
            # This ASCII name is only an identifier; no file is created.
            ds = Dataset("hypso-input.nc", mode="r", memory=buffer)
        except OSError as memory_exc:
            raise ValueError(
                f"Python can read {path}, but NetCDF cannot open its contents. "
                f"Path error: {exc}. In-memory error: {memory_exc}."
            ) from memory_exc
        print(f"Note: opened {path.name} through Python after a NetCDF path error "
              "(in-memory fallback).", flush=True)
    try:
        yield ds
    finally:
        ds.close()

def read_cube(path, args):
    aliases = {
        "lines": "y", "line": "y", "frames": "y", "frame": "y", "rows": "y", "y": "y",
        "samples": "x", "sample": "x", "columns": "x", "cols": "x", "x": "x",
        "bands": "z", "band": "z", "wavelength": "z", "wavelengths": "z", "z": "z",
    }
    with open_netcdf(path) as ds:
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
        if args.low_memory:
            from hybrid_no_traces import estimated_job_gib
            estimated_gib = estimated_job_gib((height, width, shape[2]), args.p)
        else:
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

LOCAL_SUM_OPTIONS = {
    "wide-neighbor": "WIDE_NEIGHBOR_ORIENTED",
    "narrow-neighbor": "NARROW_NEIGHBOR_ORIENTED",
    "wide-column": "WIDE_COLUMN_ORIENTED",
    "narrow-column": "NARROW_COLUMN_ORIENTED",
}
FIELDS = [
    "image", "capture", "a", "nu_max", "nu_min", "t_inc", "p", "local_sum",
    "prediction_mode", "phi", "psi", "rows", "columns", "bands", "sample_count",
    "encoder", "effective_final_nu", "nu_max_reached", "updates_at_nu_max",
    "fraction_updates_at_nu_max", "header_bits", "payload_bits", "padding_bits",
    "stream_bits", "external_info_bits", "bits_per_sample",
    "bits_per_sample_with_external_info", "compression_ratio", "max_error", "rmse_dn",
    "payload_sha256", "predictor_seconds", "encoder_seconds",
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", type=Path, default=Path("data"))
    p.add_argument("--images", nargs="+", choices=list(SCENES), default=list(SCENES))
    p.add_argument("--crop", type=int, nargs=2, metavar=("ROWS", "COLS"), default=None,
                   help="Spatial crop; default 128 128. All native bands are retained.")
    p.add_argument("--row", type=int, help="Zero-based crop origin; default centre.")
    p.add_argument("--col", type=int, help="Zero-based crop origin; default centre.")
    p.add_argument("--crop-from", type=Path, help="Reuse crop origins/shapes and verify hashes from previous metadata.json.")
    p.add_argument("--variable", default="products/dn")
    p.add_argument("--axis-order", choices=("yxz", "yzx", "xyz", "xzy", "zxy", "zyx"))
    p.add_argument("--vmax-min", type=int, default=0, help="First nu_max value, inclusive.")
    p.add_argument("--vmax-max", type=int, default=9, help="Last nu_max value, inclusive.")
    p.add_argument("--nu-min", type=int, default=-6, help="Fixed initial adaptation parameter.")
    p.add_argument("--t-inc", type=int, default=128, help="Fixed change interval; power of two, 16..2048.")
    p.add_argument("--a-values", type=int, nargs="+", default=[0, 3, 15], help="Absolute error limits, in DN.")
    p.add_argument("--p", type=int, default=3, help="Fixed number of prediction bands.")
    p.add_argument("--local-sum", choices=list(LOCAL_SUM_OPTIONS), default="narrow-neighbor")
    p.add_argument("--prediction-mode", choices=("full", "reduced"), default="reduced")
    p.add_argument("--phi-values", type=int, nargs="+", default=list(range(6)),
                   help="Damping values to compare; default 0 1 2 3 4 5, valid range 0..7 at Theta=3.")
    p.add_argument("--psi", type=int, default=7, help="Fixed offset for A>0, 0..7; forced to zero for lossless.")
    p.add_argument("--repo", type=Path)
    p.add_argument("--output", type=Path, default=Path("results/damping_fig49_128"))
    p.add_argument("--max-memory-gb", type=float, default=8.0, help="Estimated per-run RAM budget, GiB.")
    p.add_argument("--low-memory", action="store_true",
                   help="Discard hybrid text traces with the validated adapter; codec math is unchanged.")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--plot-only", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    sw.add_options(p)
    a = p.parse_args()
    if not -6 <= a.nu_min <= a.vmax_min <= a.vmax_max <= 9:
        p.error("Require -6 <= --nu-min <= --vmax-min <= --vmax-max <= 9.")
    if not 16 <= a.t_inc <= 2048 or a.t_inc & (a.t_inc - 1):
        p.error("--t-inc must be a power of two between 16 and 2048.")
    if not 0 <= a.p <= 15 or any(not 0 <= phi <= 7 for phi in a.phi_values) or not 0 <= a.psi <= 7:
        p.error("Require P=0..15 and phi/psi=0..7.")
    if any(not 0 <= v <= 32767 for v in a.a_values):
        p.error("--a-values must be in 0..32767 for this D=16 profile.")
    if (a.crop is not None and min(a.crop) < 2) or not math.isfinite(a.max_memory_gb) or a.max_memory_gb <= 0:
        p.error("Crop dimensions must be at least 2 and the memory budget must be positive.")
    if any(v is not None and v < 0 for v in (a.row, a.col)):
        p.error("Crop origins must be nonnegative.")
    if a.crop_from and (a.row is not None or a.col is not None or a.crop is not None):
        p.error("Use --crop-from alone, without manual crop size/origin overrides.")
    a.crop = (128, 128) if a.crop is None else tuple(a.crop)
    a.images = list(dict.fromkeys(a.images))
    a.phi_values = sorted(set(a.phi_values))
    a.a_values = sorted(set(a.a_values))
    a.vmax_values = list(range(a.vmax_min, a.vmax_max + 1))
    a.full_image = False
    return a


def adaptation_schedule(rows, columns, nu_min, t_inc, nu_max):
    # Weight at t=1 is initialized. Updates for t=2..N-1 use rho(t-1).
    update_t = np.arange(1, rows * columns - 1, dtype=np.int64)
    ramp = nu_min + (update_t - columns) // t_inc
    applied = np.clip(ramp, nu_min, nu_max)
    cap = max(nu_min, int(ramp[-1]))
    reached = cap >= nu_max
    hits = int(np.count_nonzero(applied == nu_max))
    return {"effective_final_nu": int(applied[-1]), "nu_max_reached": int(reached),
            "updates_at_nu_max": hits, "fraction_updates_at_nu_max": hits / len(applied),
            "highest_reachable_nu": cap}


def make_header(cube, args, absolute_error, nu_max, hd):
    y, x, z = cube.shape
    h = hd.Header(f"cube-u16be-{z}x{y}x{x}.raw")
    if h.weight_init_method != hd.WeightInitMethod.DEFAULT:
        raise ValueError("This study expects default weight initialization, not a custom table.")
    if h.weight_exponent_offset_flag != hd.WeightExponentOffsetFlag.ALL_ZERO:
        raise ValueError("This study expects zero weight exponent offsets.")
    h.entropy_coder_type = hd.EntropyCoderType.HYBRID
    h.large_d_flag = hd.LargeDFlag.SMALL_D
    h.dynamic_range = 0  # D=16 is encoded as 0.
    h.register_size = 0  # R=64 is encoded as 0.
    h.weight_component_resolution = 19 - 4
    h.prediction_bands_num = args.p
    h.local_sum_type = getattr(hd.LocalSumType, LOCAL_SUM_OPTIONS[args.local_sum])
    h.prediction_mode = getattr(hd.PredictionMode, args.prediction_mode.upper())
    # Header fields store encoded values, not the signed physical parameters.
    h.weight_update_initial_parameter = args.nu_min + 6
    h.weight_update_final_parameter = nu_max + 6
    h.weight_update_change_interval = args.t_inc.bit_length() - 1 - 4
    h.periodic_error_updating_flag = hd.PeriodicErrorUpdatingFlag.NOT_USED
    h.error_update_period_exponent = 0
    h.quantizer_fidelity_control_method = (
        hd.QuantizerFidelityControlMethod.LOSSLESS if absolute_error == 0
        else hd.QuantizerFidelityControlMethod.ABSOLUTE_ONLY)
    h.absolute_error_limit_assignment_method = hd.ErrorLimitAssignmentMethod.BAND_INDEPENDENT
    # Keep the header field width identical for all positive A values in this run.
    h.absolute_error_limit_bit_depth = max(5, max(args.a_values).bit_length())
    h.absolute_error_limit_value = absolute_error
    h.absolute_error_limit_table = np.full(z, absolute_error, dtype=np.int64)
    h.relative_error_limit_assignment_method = hd.ErrorLimitAssignmentMethod.BAND_INDEPENDENT
    h.relative_error_limit_value = 0
    h.relative_error_limit_table = np.zeros(z, dtype=np.int64)
    h.sample_representative_flag = hd.SampleRepresentativeFlag.INCLUDED
    h.sample_representative_resolution = 3
    h.band_varying_damping_flag = hd.BandVaryingDampingFlag.BAND_INDEPENDENT
    h.damping_table_flag = hd.DampingTableFlag.NOT_INCLUDED
    h.fixed_damping_value = args.phi_values[0]
    h.damping_table_array = np.full(z, h.fixed_damping_value, dtype=np.int64)
    h.band_varying_offset_flag = hd.BandVaryingOffsetFlag.BAND_INDEPENDENT
    h.damping_offset_table_flag = hd.OffsetTableFlag.NOT_INCLUDED
    h.fixed_offset_value = 0 if absolute_error == 0 else args.psi
    h.damping_offset_table_array = np.full(z, h.fixed_offset_value, dtype=np.int64)
    h.check_legal_config()
    if int(cube.min()) < 0 or int(cube.max()) > 65535:
        raise ValueError("Raw DN samples do not fit the fixed unsigned D=16 profile.")
    return h


def encode_one(cube, header, args, absolute_error, nu_max, phi, model):
    hd, constants, predictor, hybrid_encoder = model
    h = copy.deepcopy(header)
    h.weight_update_final_parameter = nu_max + 6
    if not getattr(args, "preserve_representative_header", False):
        h.fixed_damping_value = phi
        h.damping_table_array = np.full(h.z_size, phi, dtype=np.int64)
    h.check_legal_config()
    image_constants = constants.ImageConstants(h)
    start = time.perf_counter()
    pred = predictor.Predictor(h, image_constants, cube)
    pred.run_predictor()
    predictor_seconds = time.perf_counter() - start
    reconstructed = pred.clipped_quantizer_bin_center.copy()
    # Upstream leaves this entry unset: the first spatial sample is unquantized.
    reconstructed[0, 0, :] = pred.predicted_sample_value[0, 0, :] + pred.quantizer_index[0, 0, :]
    errors = cube - reconstructed
    max_error = int(np.max(np.abs(errors)))
    if max_error > absolute_error or np.any(pred.maximum_error != absolute_error):
        raise RuntimeError(f"Reconstruction bound failed: A={absolute_error}, max error={max_error}.")
    rmse = float(np.sqrt(np.mean(errors.astype(np.float64) ** 2)))
    schedule = adaptation_schedule(cube.shape[0], cube.shape[1], args.nu_min, args.t_inc, nu_max)
    # Check the actual model's exponent against the schedule used to annotate plots.
    actual_final_nu = int(pred.weight_update_scaling_exponent[cube.shape[0] * cube.shape[1] - 2])
    actual_final_nu += pred.weight_component_resolution - image_constants.dynamic_range_bits
    if actual_final_nu != schedule["effective_final_nu"]:
        raise RuntimeError("The local model's adaptation schedule differs from this script's annotation.")
    mapped = pred.get_predictor_output()
    # The optional parallel predictor shares trace arrays through mappings.
    # All checks above have finished; only mapped indices are needed now.
    if hasattr(pred, "release_workspace"):
        pred.release_workspace()
    del pred, reconstructed, errors
    gc.collect()
    encoder_type = hybrid_encoder.HybridEncoder
    if args.low_memory:
        from hybrid_no_traces import CompactHybridEncoder
        encoder_type = CompactHybridEncoder
    encoder = encoder_type(h, image_constants, mapped)
    start = time.perf_counter()
    encoder.run_encoder()
    encoder_seconds = time.perf_counter() - start
    header_bits, optional_bits = h.get_header_bitstreams()
    payload_count = len(encoder.bitstream)
    word_bits = 8 * (h.output_word_size or 8)
    padding = (-(len(header_bits) + payload_count)) % word_bits
    stream_bits = len(header_bits) + payload_count + padding
    external_bits = ((len(optional_bits) + 7) // 8) * 8
    n = h.z_size * (h.get_dynamic_range_bits() + encoder.initial_count_exponent)
    external_bits += ((n + 7) // 8) * 8
    schedule.pop("highest_reachable_nu")
    result = {"a": absolute_error, "nu_max": nu_max, "nu_min": args.nu_min,
              "t_inc": args.t_inc, "p": args.p, "local_sum": h.local_sum_type.name,
              "prediction_mode": h.prediction_mode.name, "phi": phi,
              "psi": args.psi if getattr(args, "preserve_representative_header", False) else h.fixed_offset_value,
              "rows": cube.shape[0], "columns": cube.shape[1],
              "bands": cube.shape[2], "sample_count": cube.size, "encoder": "HYBRID",
              **schedule, "header_bits": len(header_bits), "payload_bits": payload_count,
              "padding_bits": padding, "stream_bits": stream_bits, "external_info_bits": external_bits,
              "bits_per_sample": stream_bits / cube.size,
              "bits_per_sample_with_external_info": (stream_bits + external_bits) / cube.size,
              "compression_ratio": h.get_dynamic_range_bits() * cube.size / stream_bits,
              "max_error": max_error, "rmse_dn": rmse,
              "payload_sha256": hashlib.sha256(encoder.bitstream.tobytes()).hexdigest(),
              "predictor_seconds": predictor_seconds, "encoder_seconds": encoder_seconds}
    del encoder, mapped
    gc.collect()
    return result


def job_key(row):
    return row["image"], int(row["a"]), int(row["phi"]), int(row["nu_max"])


def write_summaries(rows, metadata, out):
    """Select minima from saved rows, explicitly identifying incomplete groups."""
    study = metadata["study"]
    per_phi, overall = [], []
    columns = ["image", "a", "phi", "nu_max", "bits_per_sample", "compression_ratio",
               "max_error", "rmse_dn", "nu_max_reached", "fraction_updates_at_nu_max",
               "completed_configurations", "expected_configurations", "complete"]

    def best_record(group, expected):
        best = min(group, key=lambda r: (float(r["bits_per_sample"]),
                                       int(r["nu_max"]), int(r["phi"])))
        result = {k: best[k] for k in columns[:-3]}
        result.update(completed_configurations=len(group), expected_configurations=expected,
                      complete=int(len(group) == expected))
        return result

    for scene in study["images"]:
        for a in study["a_values"]:
            group = [r for r in rows if r["image"] == scene and int(r["a"]) == a]
            if not group:
                continue
            overall.append(best_record(group, len(study["phi_values"]) * len(study["vmax_values"])))
            for phi in study["phi_values"]:
                selected = [r for r in group if int(r["phi"]) == phi]
                if selected:
                    per_phi.append(best_record(selected, len(study["vmax_values"])))
    for name, content in (("best_by_phi.csv", per_phi), ("best_overall.csv", overall)):
        target = out / name
        tmp = target.with_suffix(".csv.tmp")
        with tmp.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(content)
        os.replace(tmp, target)


def plot_results(rows, metadata, out):
    if not rows:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator
    study = metadata["study"]
    scenes, a_values = study["images"], study["a_values"]
    vmax_values, phi_values = study["vmax_values"], study["phi_values"]
    palette = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#5B4BCE", "#333333", "#56B4E9")
    markers = ("o", "s", "D", "^", "v", "P", "X", "*")
    handles = [Line2D([], [], color=palette[phi], marker=markers[phi], markersize=4,
                      linewidth=1.4, label=rf"$\phi={phi}$") for phi in phi_values]
    profile = study["profile"]
    subtitle = (f"P={profile['p']} | {profile['local_sum']}, {profile['prediction_mode']} | "
                rf"$\nu_{{min}}$={profile['nu_min']}, $t_{{inc}}$={profile['t_inc']}, $\Theta$=3")
    note = "Rate includes header, entropy payload/tail and word padding; external initialization is excluded."
    if any(v > study["inputs"][s]["highest_reachable_nu"] for s in scenes for v in vmax_values):
        note += "\nHollow markers/shading: requested final adaptation parameter is not reached in this crop."
    expected_panel = len(vmax_values) * len(phi_values)
    indexed = {job_key(r): r for r in rows}

    def draw(ax, scene, a):
        info = study["inputs"][scene]
        h, w, bands = info["crop_shape_yxz"]
        count = sum((scene, a, phi, v) in indexed for phi in phi_values for v in vmax_values)
        title = rf"{scene.title()} | $A={a}$ DN" + f"\n{h} x {w} pixels, {bands} bands"
        if count != expected_panel:
            title += f" | partial {count}/{expected_panel}"
        ax.set_title(title, fontsize=10)
        cap = info["highest_reachable_nu"]
        if vmax_values[-1] > cap:
            ax.axvspan(max(vmax_values[0] - 0.3, cap + 0.5), vmax_values[-1] + 0.3,
                       color="#eeeeee", zorder=0)
        for phi in phi_values:
            selected = {v: indexed[(scene, a, phi, v)] for v in vmax_values
                        if (scene, a, phi, v) in indexed}
            rates = [float(selected[v]["bits_per_sample"]) if v in selected else np.nan
                     for v in vmax_values]
            ax.plot(vmax_values, rates, color=palette[phi], linewidth=1.4,
                    marker=markers[phi], markersize=3.4)
            hollow = [v for v in vmax_values if v in selected and not int(selected[v]["nu_max_reached"])]
            if hollow:
                ax.plot(hollow, [float(selected[v]["bits_per_sample"]) for v in hollow],
                        linestyle="none", marker=markers[phi], markersize=3.8,
                        markerfacecolor="white", markeredgecolor=palette[phi])
        ax.set_xlabel(r"Final adaptation parameter, $\nu_{max}$", fontsize=10)
        ax.set_ylabel("Compressed bits/sample", fontsize=10)
        ax.set_xlim(vmax_values[0] - 0.3, vmax_values[-1] + 0.3)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=10))
        ax.tick_params(labelsize=9)
        ax.grid(True, alpha=0.25)
        ax.set_axisbelow(True)

    def save(fig, name, row_count):
        height = fig.get_figheight()
        fig.suptitle("HYPSO damping/adaptation study | hybrid encoder\n" + subtitle,
                     fontsize=13 if row_count > 1 else 11, y=1 - 0.08 / height)
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.48 / height),
                   ncol=min(6, len(handles)), frameon=False, fontsize=10)
        fig.text(0.5, 0.10 / height, note, ha="center", va="bottom", fontsize=8)
        fig.tight_layout(rect=(0, 1.10 / height, 1, 1 - 0.70 / height), h_pad=2.0, w_pad=2.0)
        for ext in ("png", "pdf"):
            fig.savefig(out / f"{name}.{ext}", dpi=180, bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(len(scenes), len(a_values),
                             figsize=(5.0 * len(a_values), 3.15 * len(scenes) + 1.9), squeeze=False)
    for i, scene in enumerate(scenes):
        for j, a in enumerate(a_values):
            draw(axes[i, j], scene, a)
    save(fig, "all_images_damping", len(scenes))
    for scene in scenes:
        if not any(r["image"] == scene for r in rows):
            continue
        fig, axes = plt.subplots(1, len(a_values), figsize=(5.0 * len(a_values), 5.2), squeeze=False)
        for ax, a in zip(axes.flat, a_values):
            draw(ax, scene, a)
        save(fig, f"{scene}_damping", 1)
    write_summaries(rows, metadata, out)


def load_inputs(args):
    previous = None
    if args.crop_from:
        saved = json.loads(args.crop_from.read_text(encoding="utf-8"))
        previous = saved.get("study", {}).get("inputs")
        if not isinstance(previous, dict):
            raise ValueError("--crop-from must point to a predictor_sweep metadata.json with study.inputs.")
    cubes, inputs = {}, {}
    for scene in args.images:
        local = copy.copy(args)
        old = None
        if previous is not None:
            if scene not in previous:
                raise ValueError(f"The previous metadata has no crop for {scene}.")
            old = previous[scene]
            local.row, local.col = old["crop_origin_yx"]
            local.crop = old["crop_shape_yxz"][:2]
            local.variable = old["variable"]
            local.axis_order = old["storage_axis_order"]
        path = find_capture(args.input_dir.resolve(), SCENES[scene])
        cube, info = read_cube(path, local)
        if old is not None:
            if list(cube.shape) != old["crop_shape_yxz"]:
                raise ValueError(f"{scene}: crop shape differs from the previous study.")
            if "crop_sha256" in old and info["crop_sha256"] != old["crop_sha256"]:
                raise ValueError(f"{scene}: crop data differs from the previous study (SHA256 mismatch).")
        schedule = adaptation_schedule(cube.shape[0], cube.shape[1], args.nu_min, args.t_inc, args.vmax_max)
        info["highest_reachable_nu"] = schedule["highest_reachable_nu"]
        cubes[scene], inputs[scene] = cube, info
        print(f"{scene}: crop y,x={info['crop_origin_yx']}, shape={cube.shape}, "
              f"DN={info['min_dn']}..{info['max_dn']}, estimated RAM={info['estimated_ram_gib']:.2f} GiB")
    return cubes, inputs


def clear_outputs(out):
    # Delete only this tool's own known products, never unrelated files.
    for stem in ["all_images_damping", *(f"{scene}_damping" for scene in SCENES)]:
        for ext in ("png", "pdf"):
            (out / f"{stem}.{ext}").unlink(missing_ok=True)
    for name in ("best_by_phi.csv", "best_overall.csv", "last_error.txt"):
        (out / name).unlink(missing_ok=True)


def _evaluate(key, cubes, model, state):
    args, bases = state
    scene, a, phi, v = key
    row = encode_one(cubes[scene], bases[scene][a], args, a, v, phi, model)
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
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        rows = read_csv(csv_path)
        if not rows:
            raise ValueError("No saved result rows to plot.")
        plot_results(rows, meta, out)
        print(f"Plots: {out}")
        return
    if not __debug__:
        raise ValueError("Run without Python -O: the model relies on assertions.")
    repo = find_repo(args.repo)
    hd = sw.load_model(repo, 'vmax')[0]
    storage = {"low_memory": args.low_memory}
    if args.low_memory:
        from hybrid_no_traces import validate_reference, adapter_sha256
        storage.update(adapter_sha256=adapter_sha256(), reference_class_sha256=validate_reference())
    cubes, inputs = load_inputs(args)
    bases = {s: {a: make_header(cube, args, a, args.vmax_min, hd) for a in args.a_values}
             for s, cube in cubes.items()}
    profile = {"p": args.p, "local_sum": args.local_sum, "prediction_mode": args.prediction_mode,
               "nu_min": args.nu_min, "t_inc": args.t_inc,
               "psi_near_lossless": args.psi, "psi_lossless": 0, "theta": 3,
               "D": 16, "Omega": 19, "R": 64, "encoder": "HYBRID"}
    identity_inputs = copy.deepcopy(inputs)
    for info in identity_inputs.values():
        info.pop("estimated_ram_gib")
    package = repo / "ccsds123_i2_hlm"
    model_files = list(package.rglob("*.py")) + list(package.rglob("*.txt"))
    study = {"script_version": VERSION, "worker_helper_sha256": sw.helper_sha(),
             "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             "model_sha256": file_digest(model_files, repo), "python": platform.python_version(),
             "numpy": np.__version__, "images": args.images, "inputs": identity_inputs,
             "profile": profile, "storage": storage, "a_values": args.a_values,
             "vmax_values": args.vmax_values, "phi_values": args.phi_values,
             "headers_at_first_vmax_and_phi": {s: {str(a): header_snapshot(h) for a, h in by_a.items()}
                                       for s, by_a in bases.items()},
             "metric": "CCSDS header + entropy payload/tail + output-word padding, divided by scalar samples",
             "external_info": "Separate optional tables and hybrid initialization excluded from primary rate; inclusive rate recorded",
             "error_check": "encoder-side reconstruction, not an independent decoder",
             "reference": "CCSDS 120.2-G-2, figure 4-9; HYPSO profile/crops and A=15 replace its corpus/settings/A=20"}
    metadata = {"study": study, "model_path": str(repo),
                "crop_from": str(args.crop_from.resolve()) if args.crop_from else None}
    jobs = [(s, a, phi, v) for s in args.images for a in args.a_values
            for phi in args.phi_values for v in args.vmax_values]
    print(f"Fixed: P={args.p}, {args.local_sum}, {args.prediction_mode}, hybrid; "
          f"nu_min={args.nu_min}, t_inc={args.t_inc}, Theta=3, psi={args.psi} (0 for lossless).")
    print(f"{len(args.images)} images x {len(args.a_values)} A values x {len(args.phi_values)} phi values "
          f"x {len(args.vmax_values)} nu_max values = {len(jobs)} configurations; "
          f"A={args.a_values}, phi={args.phi_values}, nu_max={args.vmax_min}..{args.vmax_max}.")
    for s, info in inputs.items():
        cap = info["highest_reachable_nu"]
        if args.vmax_max > cap:
            print(f"NOTE {s}: this crop only reaches nu={cap}; requested nu_max > {cap} "
                  f"is never reached. Values >= {cap} have the same applied update schedule.")
    max_job_gib = max(i["estimated_ram_gib"] for i in inputs.values())
    if args.dry_run:
        sw.plan(args, len(jobs), max_job_gib, cubes)
        print("Dry run: inputs/settings checked; estimate for a fresh run. No outputs changed.")
        return
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    if not args.overwrite and (csv_path.exists() or meta_path.exists()):
        if not meta_path.exists():
            raise ValueError("Results exist without metadata. Choose a new --output folder.")
        old = json.loads(meta_path.read_text(encoding="utf-8"))
        if not isinstance(old, dict) or old.get("study") != metadata["study"]:
            raise ValueError("Inputs, ranges, settings, script or model differ from the saved run. "
                             "Choose a new --output folder, or use --overwrite to restart it.")
        rows = read_csv(csv_path)
    done = {job_key(r) for r in rows}
    if len(done) != len(rows) or not done.issubset(set(jobs)):
        raise ValueError("Saved results contain duplicate or unexpected configurations.")
    pending = sw.pending_keys(jobs, rows, job_key)
    workers = sw.plan(args, len(pending), max_job_gib, cubes)
    metadata["execution"] = {"requested_workers": args.workers, "active_workers": workers,
                             "completed_at_start": len(rows), "total_memory_gib": args.total_memory_gb}
    if args.overwrite:
        clear_outputs(out)
    write_json(meta_path, metadata)
    write_csv(csv_path, rows)
    print(f"Complete: {len(rows)}; remaining: {len(pending)}. Results: {out}")
    sw.collect(jobs, rows, pending, workers, cubes, repo, "vmax", _evaluate, (args, bases),
               get_key=job_key, save=lambda rs: write_csv(csv_path, rs),
               plot=lambda rs: plot_results(rs, metadata, out), out=out)
    print(f"Saved {len(rows)}/{len(jobs)} configurations. CSV and plots: {out}")
    print("Error checks use encoder-side reconstruction, not an independent decoder.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130)
    except (ValueError, OSError, ModuleNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
