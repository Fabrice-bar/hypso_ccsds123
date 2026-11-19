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

SCENES = {
    "lisbon": "lisbon_2022-09-22T10-55-19Z-l1a.nc",
    "bangladesh": "bangladesh_2022-10-26T04-02-56Z-l1a.nc",
    "santarem": "santarem_2022-09-24T13-52-04Z-l1a.nc",
    "doha": "doha_2022-09-12T06-38-31Z-l1a.nc",
    "finnmark": "finnmark_2022-07-06T19-40-03Z-l1a.nc",
}

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
        if height < 1 or width < 2 or height * width < 3:
            raise ValueError("Use at least one row, two columns, and three spatial samples.")
        # Conservative allowance for the verification model's many trace arrays.
        if args.low_memory:
            estimated_gib = height * width * shape[2] * (320 + 16 * (args.p + 3)) / 2**30 + 0.5
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

LOCAL_SUM_OPTIONS = {
    "wide-neighbor": "WIDE_NEIGHBOR_ORIENTED",
    "narrow-neighbor": "NARROW_NEIGHBOR_ORIENTED",
    "wide-column": "WIDE_COLUMN_ORIENTED",
    "narrow-column": "NARROW_COLUMN_ORIENTED",
}

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
    h.entropy_coder_type = hd.EntropyCoderType.SAMPLE_ADAPTIVE
    h.unary_length_limit = 18
    h.rescaling_counter_size = 2
    h.initial_count_exponent = 1
    h.accumulator_init_constant = 3
    h.set_accumulator_init_table_to_default()
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
    h.sample_representative_flag = hd.SampleRepresentativeFlag.NOT_INCLUDED
    h.sample_representative_resolution = 0
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
    from ccsds123_i2_hlm.sa_encoder import SampleAdaptiveEncoder
    encoder_type = SampleAdaptiveEncoder
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
    schedule.pop("highest_reachable_nu")
    result = {"a": absolute_error, "nu_max": nu_max, "nu_min": args.nu_min,
              "t_inc": args.t_inc, "p": args.p, "local_sum": h.local_sum_type.name,
              "prediction_mode": h.prediction_mode.name, "phi": phi,
              "psi": h.fixed_offset_value, "rows": cube.shape[0], "columns": cube.shape[1],
              "bands": cube.shape[2], "sample_count": cube.size, "encoder": "SAMPLE_ADAPTIVE",
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
