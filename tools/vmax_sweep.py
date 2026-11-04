#!/usr/bin/env python3
r"""HYPSO predictor adaptation study, following CCSDS Green Book figure 4-7.

Standalone: save as tools/vmax_sweep.py inside the verification repository.
It does not import predictor_sweep.py and does not change the repository files.

PowerShell, from the repo root:
  .\.venv\Scripts\python.exe tools/vmax_sweep.py --input-dir data --output results/vmax_sweep

Defaults: the five original HYPSO captures, a centre 32x32 spatial crop and ALL
native bands of products/dn, with raw integer DN values and no calibration.
nu_max = -6..9 inclusive; A = 0,3,20 DN; nu_min = -6; t_inc = 128 samples.
Fixed profile: P=3, wide-neighbour sums, reduced mode, hybrid encoder, D=16,
Omega=19, R=64, Theta=3, phi=3; psi=7 for A>0 and the required psi=0 for A=0.
Other encoder settings come from your local header.py and are recorded.
This reproduces the figure's sweep design with HYPSO crops, not its numerical
results or its instrument-specific choices of prediction mode and damping.

IMPORTANT: with 32x32 crops, nu_min=-6 and t_inc=128, the update schedule only
reaches nu=1. nu_max >= 1 gives the same applied update schedule and should
give the same compressed rate. Values 2..9 are never reached. The script
keeps the requested experiment, records this limitation, and shades these
values in the plots. This short-crop experiment is not a steady-state test
of the upper part of the nu_max range. No crop or t_inc is changed silently.

Optional: --crop-from results/predictor_sweep/metadata.json uses the exact
crop origins/shapes from the previous sweep and verifies the crop hashes.
Otherwise --crop ROWS COLS, --row and --col work as in predictor_sweep.py.

Outputs: results.csv, metadata.json, all_images_vmax.png/.pdf and one
vmax_A<number>.png/.pdf per A. Each panel has one curve per HYPSO capture.
Results are saved after every run. Repeat the same command to resume.
--plot-only redraws the plots. --dry-run validates inputs without compression.
--overwrite replaces only this script's known outputs in the chosen folder.

Rate = (CCSDS header + entropy payload/tail + output-word padding) / samples.
Separate hybrid initialization/optional files are excluded from that rate;
their sizes and an inclusive rate are also recorded. No per-run bitstream
or large trace CSV files are saved. Reconstruction checks are encoder-side,
not an independent decoder. Python runtime is not FPGA throughput.
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

VERSION = "1.0.0"

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
    p.add_argument("--crop", type=int, nargs=2, metavar=("ROWS", "COLS"), default=(32, 32))
    p.add_argument("--row", type=int, help="Zero-based crop origin; default centre.")
    p.add_argument("--col", type=int, help="Zero-based crop origin; default centre.")
    p.add_argument("--crop-from", type=Path, help="Reuse crop origins/shapes and verify hashes from previous metadata.json.")
    p.add_argument("--variable", default="products/dn")
    p.add_argument("--axis-order", choices=("yxz", "yzx", "xyz", "xzy", "zxy", "zyx"))
    p.add_argument("--vmax-min", type=int, default=-6, help="First nu_max value, inclusive.")
    p.add_argument("--vmax-max", type=int, default=9, help="Last nu_max value, inclusive.")
    p.add_argument("--nu-min", type=int, default=-6, help="Fixed initial adaptation parameter.")
    p.add_argument("--t-inc", type=int, default=128, help="Fixed change interval; power of two, 16..2048.")
    p.add_argument("--a-values", type=int, nargs="+", default=[0, 3, 20], help="Absolute error limits, in DN.")
    p.add_argument("--p", type=int, default=3, help="Fixed number of prediction bands.")
    p.add_argument("--local-sum", choices=list(LOCAL_SUM_OPTIONS), default="wide-neighbor")
    p.add_argument("--prediction-mode", choices=("full", "reduced"), default="reduced")
    p.add_argument("--phi", type=int, default=3, help="Fixed sample-representative damping, 0..7.")
    p.add_argument("--psi", type=int, default=7, help="Fixed offset for A>0, 0..7; forced to zero for lossless.")
    p.add_argument("--repo", type=Path)
    p.add_argument("--output", type=Path, default=Path("results/vmax_sweep"))
    p.add_argument("--max-memory-gb", type=float, default=4.0, help="Estimated per-run RAM budget, GiB.")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--plot-only", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()
    if not -6 <= a.nu_min <= a.vmax_min <= a.vmax_max <= 9:
        p.error("Require -6 <= --nu-min <= --vmax-min <= --vmax-max <= 9.")
    if not 16 <= a.t_inc <= 2048 or a.t_inc & (a.t_inc - 1):
        p.error("--t-inc must be a power of two between 16 and 2048.")
    if not 0 <= a.p <= 15 or not 0 <= a.phi <= 7 or not 0 <= a.psi <= 7:
        p.error("Require P=0..15 and phi/psi=0..7.")
    if any(not 0 <= v <= 32767 for v in a.a_values):
        p.error("--a-values must be in 0..32767 for this D=16 profile.")
    if min(a.crop) < 2 or a.max_memory_gb <= 0:
        p.error("Crop dimensions must be at least 2 and the memory budget must be positive.")
    if any(v is not None and v < 0 for v in (a.row, a.col)):
        p.error("Crop origins must be nonnegative.")
    if a.crop_from and (a.row is not None or a.col is not None or tuple(a.crop) != (32, 32)):
        p.error("Use --crop-from alone, without manual crop size/origin overrides.")
    a.images = list(dict.fromkeys(a.images))
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
    h.fixed_damping_value = args.phi
    h.damping_table_array = np.full(z, args.phi, dtype=np.int64)
    h.band_varying_offset_flag = hd.BandVaryingOffsetFlag.BAND_INDEPENDENT
    h.damping_offset_table_flag = hd.OffsetTableFlag.NOT_INCLUDED
    h.fixed_offset_value = 0 if absolute_error == 0 else args.psi
    h.damping_offset_table_array = np.full(z, h.fixed_offset_value, dtype=np.int64)
    h.check_legal_config()
    if int(cube.min()) < 0 or int(cube.max()) > 65535:
        raise ValueError("Raw DN samples do not fit the fixed unsigned D=16 profile.")
    return h


def encode_one(cube, header, args, absolute_error, nu_max, model):
    hd, constants, predictor, hybrid_encoder = model
    h = copy.deepcopy(header)
    h.weight_update_final_parameter = nu_max + 6
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
    del pred, reconstructed, errors
    gc.collect()
    encoder = hybrid_encoder.HybridEncoder(h, image_constants, mapped)
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
              "prediction_mode": h.prediction_mode.name, "phi": args.phi,
              "psi": h.fixed_offset_value, "rows": cube.shape[0], "columns": cube.shape[1],
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
    return row["image"], int(row["a"]), int(row["nu_max"])


def plot_results(rows, metadata, out):
    if not rows:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator
    study = metadata["study"]
    scenes, a_values, vmax_values = study["images"], study["a_values"], study["vmax_values"]
    colours = dict(zip(SCENES, ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#6A5ACD")))
    markers = dict(zip(SCENES, ("o", "s", "D", "^", "v")))
    caps = [study["inputs"][s]["highest_reachable_nu"] for s in scenes]
    common_cap = caps[0] if len(set(caps)) == 1 else None
    unreachable = any(v > cap for v in vmax_values for cap in caps)
    shapes = {tuple(study["inputs"][s]["crop_shape_yxz"]) for s in scenes}
    shape_text = (f"{next(iter(shapes))[0]} x {next(iter(shapes))[1]} pixels, "
                  f"{next(iter(shapes))[2]} bands" if len(shapes) == 1 else "per-scene crops")
    profile = study["profile"]
    subtitle = (f"P={profile['p']} | {profile['local_sum']}, {profile['prediction_mode']} | "
                rf"$\nu_{{min}}$={profile['nu_min']}, $t_{{inc}}$={profile['t_inc']} | {shape_text}")
    handles = [Line2D([], [], color=colours[s], marker=markers[s], markersize=5,
                      linewidth=1.5, label=s.title()) for s in scenes]
    note = "One curve per capture crop; bits include header, entropy payload/tail and word padding."
    if unreachable:
        note = ((rf"Shaded: requested $\nu_{{max}}$ is not reached; applied schedule is identical for $\nu_{{max}}\geq {common_cap}$ ."
                 if common_cap is not None else "Hollow markers: requested nu_max is not reached in that crop.")
                + "\nShort-crop results do not measure steady-state performance at unreached values.")

    def draw(ax, a_value):
        count = sum(int(r["a"]) == a_value for r in rows)
        expected = len(scenes) * len(vmax_values)
        title = rf"Absolute error limit $A={a_value}$ DN"
        if count != expected:
            title += f"\nPartial: {count}/{expected} runs"
        ax.set_title(title, fontsize=11)
        if unreachable and common_cap is not None and vmax_values[-1] > common_cap:
            ax.axvspan(max(vmax_values[0] - 0.35, common_cap + 0.5), vmax_values[-1] + 0.35,
                       facecolor="#eeeeee", zorder=0)
        for scene in scenes:
            selected = {int(r["nu_max"]): r for r in rows
                        if r["image"] == scene and int(r["a"]) == a_value}
            rates = [float(selected[v]["bits_per_sample"]) if v in selected else np.nan for v in vmax_values]
            ax.plot(vmax_values, rates, color=colours[scene], linewidth=1.5,
                    marker=markers[scene], markersize=4, label=scene.title())
            hollow = [v for v in vmax_values if v in selected and not int(selected[v]["nu_max_reached"])]
            if hollow:
                ax.plot(hollow, [float(selected[v]["bits_per_sample"]) for v in hollow],
                        linestyle="none", marker=markers[scene], markersize=4,
                        markerfacecolor="white", markeredgecolor=colours[scene])
        ax.set_xlabel(r"Final adaptation parameter, $\nu_{max}$")
        ax.set_ylabel("Compressed bits/sample")
        ax.set_xlim(vmax_values[0] - 0.35, vmax_values[-1] + 0.35)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=9))
        ax.grid(True, alpha=0.25)
        ax.set_axisbelow(True)

    ncols = min(3, len(a_values))
    nrows = math.ceil(len(a_values) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.7 * nrows + 1.8), squeeze=False)
    for ax, a_value in zip(axes.flat, a_values):
        draw(ax, a_value)
    for ax in list(axes.flat)[len(a_values):]:
        ax.set_visible(False)
    fig.suptitle("HYPSO predictor adaptation sweep | hybrid encoder\n" + subtitle, fontsize=13)
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.08),
               ncol=min(5, len(scenes)), frameon=False, fontsize=10)
    fig.text(0.5, 0.015, note, ha="center", va="bottom", fontsize=9)
    fig.tight_layout(rect=(0, 0.19, 1, 0.88))
    for ext in ("png", "pdf"):
        fig.savefig(out / f"all_images_vmax.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    for a_value in a_values:
        if not any(int(r["a"]) == a_value for r in rows):
            continue
        fig, ax = plt.subplots(figsize=(8.5, 6.0))
        draw(ax, a_value)
        fig.suptitle("HYPSO predictor adaptation sweep | hybrid encoder\n" + subtitle, fontsize=11)
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.09),
                   ncol=min(5, len(scenes)), frameon=False, fontsize=9)
        fig.text(0.5, 0.018, note, ha="center", va="bottom", fontsize=8)
        fig.tight_layout(rect=(0, 0.19, 1, 0.89))
        for ext in ("png", "pdf"):
            fig.savefig(out / f"vmax_A{a_value}.{ext}", dpi=180, bbox_inches="tight")
        plt.close(fig)


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
    # Only known filenames belonging to this script, not unrelated outputs.
    for ext in ("png", "pdf"):
        (out / f"all_images_vmax.{ext}").unlink(missing_ok=True)
        for p in out.glob(f"vmax_A*.{ext}"):
            if p.stem.removeprefix("vmax_A").isdigit():
                p.unlink()
    (out / "last_error.txt").unlink(missing_ok=True)


def main():
    args = parse_args()
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
    sys.path.insert(0, str(repo))
    from ccsds123_i2_hlm import header as hd, constants, predictor, hybrid_encoder
    model = hd, constants, predictor, hybrid_encoder
    cubes, inputs = load_inputs(args)
    bases = {s: {a: make_header(cube, args, a, args.vmax_min, hd) for a in args.a_values}
             for s, cube in cubes.items()}
    profile = {"p": args.p, "local_sum": args.local_sum, "prediction_mode": args.prediction_mode,
               "nu_min": args.nu_min, "t_inc": args.t_inc, "phi": args.phi,
               "psi_near_lossless": args.psi, "psi_lossless": 0, "theta": 3,
               "D": 16, "Omega": 19, "R": 64, "encoder": "HYBRID"}
    identity_inputs = copy.deepcopy(inputs)
    for info in identity_inputs.values():
        info.pop("estimated_ram_gib")
    package = repo / "ccsds123_i2_hlm"
    model_files = list(package.rglob("*.py")) + list(package.rglob("*.txt"))
    study = {"script_version": VERSION,
             "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             "model_sha256": file_digest(model_files, repo), "python": platform.python_version(),
             "numpy": np.__version__, "images": args.images, "inputs": identity_inputs,
             "profile": profile, "a_values": args.a_values, "vmax_values": args.vmax_values,
             "headers_at_first_vmax": {s: {str(a): header_snapshot(h) for a, h in by_a.items()}
                                       for s, by_a in bases.items()},
             "metric": "CCSDS header + entropy payload/tail + output-word padding, divided by scalar samples",
             "external_info": "Separate optional tables and hybrid initialization excluded from primary rate; inclusive rate recorded",
             "error_check": "encoder-side reconstruction, not an independent decoder",
             "reference": "CCSDS 120.2-G-2, figure 4-7; HYPSO profile and crops replace its corpus settings"}
    metadata = {"study": study, "model_path": str(repo),
                "crop_from": str(args.crop_from.resolve()) if args.crop_from else None}
    jobs = [(s, a, v) for s in args.images for a in args.a_values for v in args.vmax_values]
    print(f"Fixed: P={args.p}, {args.local_sum}, {args.prediction_mode}, hybrid; "
          f"nu_min={args.nu_min}, t_inc={args.t_inc}, phi={args.phi}, psi={args.psi} (0 for lossless).")
    print(f"{len(args.images)} images x {len(args.a_values)} A values x {len(args.vmax_values)} nu_max values "
          f"= {len(jobs)} configurations; A={args.a_values}, nu_max={args.vmax_min}..{args.vmax_max}.")
    for s, info in inputs.items():
        cap = info["highest_reachable_nu"]
        if args.vmax_max > cap:
            print(f"NOTE {s}: this crop only reaches nu={cap}; requested nu_max > {cap} "
                  f"is never reached. Values >= {cap} have the same applied update schedule.")
    if args.dry_run:
        print("Dry run complete. No compression performed and no outputs changed.")
        return
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    if not args.overwrite and (csv_path.exists() or meta_path.exists()):
        if not meta_path.exists():
            raise ValueError("Results exist without metadata. Choose a new --output folder.")
        old = json.loads(meta_path.read_text(encoding="utf-8"))
        if old.get("study") != study:
            raise ValueError("Inputs, ranges, settings, script or model differ from the saved run. "
                             "Choose a new --output folder, or use --overwrite to restart it.")
        rows = read_csv(csv_path)
    done = {job_key(r) for r in rows}
    if len(done) != len(rows) or not done.issubset(set(jobs)):
        raise ValueError("Saved results contain duplicate or unexpected configurations.")
    if args.overwrite:
        clear_outputs(out)
    write_json(meta_path, metadata)
    write_csv(csv_path, rows)
    print(f"Complete: {len(done)}; remaining: {len(jobs) - len(done)}. Results: {out}")
    job = None
    logs = io.StringIO()
    try:
        for index, job in enumerate(jobs, 1):
            if job in done:
                continue
            scene, absolute_error, nu_max = job
            print(f"[{index}/{len(jobs)}] {scene}: A={absolute_error}, nu_max={nu_max} ...", flush=True)
            logs = io.StringIO()
            with redirect_stdout(logs):
                result = encode_one(cubes[scene], bases[scene][absolute_error], args, absolute_error, nu_max, model)
            result.update(image=scene, capture=SCENES[scene])
            rows.append(result)
            write_csv(csv_path, rows)
            done.add(job)
            print(f"  {result['bits_per_sample']:.5f} bits/sample; max error={result['max_error']} DN; "
                  f"{result['predictor_seconds'] + result['encoder_seconds']:.2f} s", flush=True)
            if nu_max == args.vmax_max and absolute_error == args.a_values[-1]:
                plot_results(rows, metadata, out)
    except KeyboardInterrupt:
        print("\nStopped. Completed runs are saved; repeat the same command to resume.")
    except Exception:
        (out / "last_error.txt").write_text(f"Configuration: {job}\n{logs.getvalue()}\n"
                                           + traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        if rows:
            plot_results(rows, metadata, out)
    print(f"Saved {len(rows)}/{len(jobs)} configurations. CSV and plots: {out}")
    print("Error checks use encoder-side reconstruction, not an independent decoder.")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, ModuleNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
