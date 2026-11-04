#!/usr/bin/env python3
r"""Lossless CCSDS-123 predictor study for the five selected HYPSO-1 captures.

Put this file in the verification repository's tools/ directory. Put the five
original -l1a.nc files in data/hypso/ (subdirectories are also searched).

PowerShell, from the repository root:
  .\.venv\Scripts\python.exe -m pip install netCDF4 matplotlib
  .\.venv\Scripts\python.exe tools/predictor_sweep.py --input-dir data/hypso --p-max 8

P runs from 0 THROUGH --p-max, inclusive. Every P is tested with all four local
sum types and both prediction modes. No encoder settings are overridden: the
encoder/defaults come from your local ccsds123_i2_hlm/header.py. Currently the
upstream default is HYBRID. Required lossless changes are documented below.

Default input: a CENTER 32x32 spatial crop, ALL native bands, unscaled raw DN.
No calibration, band removal, resampling, denoising or additional binning.
These small crops are pilot tests, not whole-scene compression benchmarks.
Use --crop 64 64 for larger crops; --row/--col specify zero-based origins.
Use --images lisbon to test one capture, or --full-image for an entire cube.
The reference model is memory hungry; full cubes may require over 100 GiB.

Outputs: results.csv, metadata.json, one PNG/PDF per image, all_images.png/pdf.
Rerunning the SAME command automatically resumes completed configurations.
Increasing --p-max extends a compatible run. --overwrite starts fresh in the
chosen output folder. --plot-only redraws existing results without compression.

Metric: (CCSDS header + encoded payload including encoder tail + word padding)
        / (rows * columns * spectral bands).
This matches the upstream z-output-bitstream.bin size convention. External
initialization/table files are excluded from the primary metric, but their
sizes and an inclusive rate are also recorded. NetCDF file size is never used.

The zero-error check uses ENCODER-SIDE reconstruction, not an independent
decoder. Runtime measures this Python model, not FPGA throughput.
Tested against upstream commit 9fabb0371ebf93d72a0f83ca053e9cdb2c63e3c6.
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
    "image", "capture", "p", "local_sum", "prediction_mode", "rows", "columns",
    "bands", "sample_count", "encoder", "header_bits", "payload_bits",
    "padding_bits", "stream_bits", "external_info_bits", "bits_per_sample",
    "bits_per_sample_with_external_info", "compression_ratio", "max_error",
    "predictor_seconds", "encoder_seconds",
]
VERSION = "1.0"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", type=Path, default=Path("data/hypso"))
    p.add_argument("--images", nargs="+", choices=list(SCENES), default=list(SCENES))
    p.add_argument("--p-max", type=int, default=8, help="Sweep 0..P_MAX inclusive; 0 to 15.")
    p.add_argument("--crop", type=int, nargs=2, metavar=("ROWS", "COLS"), default=(32, 32))
    p.add_argument("--row", type=int, help="Zero-based first row; default: centred.")
    p.add_argument("--col", type=int, help="Zero-based first column; default: centred.")
    p.add_argument("--full-image", action="store_true", help="Use every spatial sample (large RAM use).")
    p.add_argument("--variable", default="products/dn", help="NetCDF raw-DN variable path.")
    p.add_argument("--axis-order", choices=("yxz", "yzx", "xyz", "xzy", "zxy", "zyx"),
                   help="Storage-axis order, only needed for unrecognised NetCDF dimension names.")
    p.add_argument("--repo", type=Path, help="Verification repo root; normally detected automatically.")
    p.add_argument("--output", type=Path, default=Path("results/predictor_sweep"))
    p.add_argument("--max-memory-gb", type=float, default=4.0,
                   help="Conservative per-run RAM budget in GiB; default 4.0.")
    p.add_argument("--overwrite", action="store_true", help="Replace this script's previous results.")
    p.add_argument("--plot-only", action="store_true", help="Plot saved CSV without rerunning the model.")
    p.add_argument("--dry-run", action="store_true", help="Validate inputs/settings and print the plan only.")
    a = p.parse_args()
    if not 0 <= a.p_max <= 15:
        p.error("--p-max must be between 0 and 15 (inclusive).")
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
        estimated_gib = height * width * shape[2] * (2200 + 16 * (args.p_max + 3)) / 2**30 + 0.15
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


def make_base_header(cube, hd):
    y, x, z = cube.shape
    # The fake raw filename supplies dimensions/type without writing a raw copy.
    h = hd.Header(f"cube-u16be-{z}x{y}x{x}.raw")
    h.quantizer_fidelity_control_method = hd.QuantizerFidelityControlMethod.LOSSLESS
    h.periodic_error_updating_flag = hd.PeriodicErrorUpdatingFlag.NOT_USED
    h.error_update_period_exponent = 0
    # The standard requires psi=0 in lossless mode. Preserve phi and Theta.
    h.fixed_offset_value = 0
    h.damping_offset_table_array = np.zeros(z, dtype=np.int64)
    if h.weight_init_method != hd.WeightInitMethod.DEFAULT:
        raise ValueError("This sweep expects the repo's default weight initialization, not custom weight tables.")
    if h.weight_exponent_offset_flag != hd.WeightExponentOffsetFlag.ALL_ZERO:
        raise ValueError("This sweep expects the repo's default zero weight exponent offsets.")
    h.check_legal_config()
    limit = 2**h.get_dynamic_range_bits() - 1
    if int(cube.min()) < 0 or int(cube.max()) > limit:
        raise ValueError(f"DN samples do not fit the fixed unsigned D={h.get_dynamic_range_bits()} setting.")
    return h


def encode_one(cube, base, p, local_sum, mode, model):
    hd, constants, predictor, encoder_classes = model
    h = copy.deepcopy(base)
    h.prediction_bands_num = p
    h.local_sum_type = getattr(hd.LocalSumType, local_sum)
    h.prediction_mode = getattr(hd.PredictionMode, mode)
    h.check_legal_config()
    image_constants = constants.ImageConstants(h)
    started = time.perf_counter()
    pred = predictor.Predictor(h, image_constants, cube)
    pred.run_predictor()
    predictor_seconds = time.perf_counter() - started
    # The upstream model leaves the first spatial sample's bin centre unset.
    reconstructed = pred.clipped_quantizer_bin_center.copy()
    reconstructed[0, 0, :] = pred.predicted_sample_value[0, 0, :] + pred.quantizer_index[0, 0, :]
    error = int(np.max(np.abs(cube - reconstructed)))
    if error != 0 or np.any(pred.maximum_error != 0):
        raise RuntimeError(f"Lossless encoder-side reconstruction failed: max error {error}.")
    mapped = pred.get_predictor_output()
    del reconstructed, pred
    gc.collect()
    encoder = encoder_classes[h.entropy_coder_type](h, image_constants, mapped)
    started = time.perf_counter()
    encoder.run_encoder()
    encoder_seconds = time.perf_counter() - started
    header_bits, optional_bits = h.get_header_bitstreams()
    payload_count = len(encoder.bitstream)  # Includes hybrid flush/termination tail.
    word_bits = 8 * (h.output_word_size or 8)
    padding = (-(len(header_bits) + payload_count)) % word_bits
    stream_bits = len(header_bits) + payload_count + padding
    external_bits = ((len(optional_bits) + 7) // 8) * 8
    if h.entropy_coder_type == hd.EntropyCoderType.HYBRID:
        # Size of upstream's separate hybrid_initial_accumulator.bin.
        n = h.z_size * (h.get_dynamic_range_bits() + encoder.initial_count_exponent)
        external_bits += ((n + 7) // 8) * 8
    result = {"p": p, "local_sum": local_sum, "prediction_mode": mode,
              "rows": cube.shape[0], "columns": cube.shape[1], "bands": cube.shape[2],
              "sample_count": cube.size, "encoder": h.entropy_coder_type.name,
              "header_bits": len(header_bits), "payload_bits": payload_count,
              "padding_bits": padding, "stream_bits": stream_bits,
              "external_info_bits": external_bits, "bits_per_sample": stream_bits / cube.size,
              "bits_per_sample_with_external_info": (stream_bits + external_bits) / cube.size,
              "compression_ratio": h.get_dynamic_range_bits() * cube.size / stream_bits,
              "max_error": error, "predictor_seconds": predictor_seconds,
              "encoder_seconds": encoder_seconds}
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
    return row["image"], int(row["p"]), row["local_sum"], row["prediction_mode"]


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
    all_p = sorted({int(r["p"]) for r in rows})
    handles = [Line2D([], [], color=colour, linestyle="-" if mode == "FULL" else "--",
                      marker="o" if mode == "FULL" else "s", markersize=4,
                      label=f"{label} / {mode.lower()}")
               for (_, label), colour in zip(SUM_TYPES.items(), colours) for mode in MODES]

    def draw(ax, scene):
        for (sum_name, _), colour in zip(SUM_TYPES.items(), colours):
            for mode in MODES:
                by_p = {int(r["p"]): float(r["bits_per_sample"]) for r in rows
                        if r["image"] == scene and r["local_sum"] == sum_name and r["prediction_mode"] == mode}
                ax.plot(all_p, [by_p.get(p, np.nan) for p in all_p], color=colour,
                        linestyle="-" if mode == "FULL" else "--", linewidth=1.5,
                        marker="o" if mode == "FULL" else "s", markersize=4,
                        markerfacecolor=colour if mode == "FULL" else "none")
        info = metadata["study"]["inputs"][scene]
        shape = info["crop_shape_yxz"]
        date = Path(info["file"]).name.split("_")[1].split("T")[0]
        completed = sum(r["image"] == scene for r in rows)
        expected = 8 * (metadata["requested_p_max"] + 1)
        status = "" if completed == expected else f" | partial: {completed}/{expected} runs"
        ax.set_title(f"{scene.title()} | {date}\n{shape[0]} x {shape[1]} pixels, {shape[2]} bands{status}", fontsize=11)
        ax.set_xlabel("P (number of preceding prediction bands)")
        ax.set_ylabel("Compressed bits/sample")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True, alpha=0.25)
        ax.set_axisbelow(True)

    encoder = next(iter(metadata["study"]["base_headers"].values()))["entropy_coder_type"]
    for scene in scenes:
        fig, ax = plt.subplots(figsize=(9, 6))
        draw(ax, scene)
        fig.suptitle(f"Lossless predictor sweep | {encoder.lower()} encoder", fontsize=13)
        fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=9)
        fig.tight_layout(rect=(0, 0.18, 1, 0.95))
        for ext in ("png", "pdf"):
            fig.savefig(out / f"{scene}.{ext}", dpi=180, bbox_inches="tight")
        plt.close(fig)
    nrows = (len(scenes) + 1) // 2
    fig, axes = plt.subplots(nrows, 2, figsize=(13, 3.7 * nrows + 1.6), squeeze=False)
    for ax, scene in zip(axes.flat, scenes):
        draw(ax, scene)
    for ax in list(axes.flat)[len(scenes):]:
        ax.set_visible(False)
    fig.suptitle(f"Lossless CCSDS-123 predictor study | {encoder.lower()} encoder", fontsize=15)
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.11, 1, 0.96))
    for ext in ("png", "pdf"):
        fig.savefig(out / f"all_images.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
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
    sys.path.insert(0, str(repo))
    from ccsds123_i2_hlm import header as hd, constants, predictor, hybrid_encoder, sa_encoder, ba_encoder
    encoder_classes = {
        hd.EntropyCoderType.HYBRID: hybrid_encoder.HybridEncoder,
        hd.EntropyCoderType.SAMPLE_ADAPTIVE: sa_encoder.SampleAdaptiveEncoder,
        hd.EntropyCoderType.BLOCK_ADAPTIVE: ba_encoder.BlockAdaptiveEncoder,
    }
    model = hd, constants, predictor, encoder_classes
    cubes, inputs, bases = {}, {}, {}
    for scene in args.images:
        path = find_capture(args.input_dir.resolve(), SCENES[scene])
        cubes[scene], inputs[scene] = read_cube(path, args)
        bases[scene] = make_base_header(cubes[scene], hd)
        print(f"{scene}: crop y,x={inputs[scene]['crop_origin_yx']}, "
              f"shape y,x,z={cubes[scene].shape}, DN={inputs[scene]['min_dn']}..{inputs[scene]['max_dn']}")
    package = repo / "ccsds123_i2_hlm"
    model_files = list(package.rglob("*.py")) + list(package.rglob("*.txt"))
    # RAM estimates depend on requested P but are not part of experimental identity.
    identity_inputs = copy.deepcopy(inputs)
    for info in identity_inputs.values():
        info.pop("estimated_ram_gib")
    study = {"script_version": VERSION,
             "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             "model_sha256": file_digest(model_files, repo),
             "python": platform.python_version(), "numpy": np.__version__,
             "images": args.images, "inputs": identity_inputs,
             "base_headers": {s: header_snapshot(h) for s, h in bases.items()},
             "metric": "CCSDS header + entropy payload/tail + output-word padding, divided by scalar samples",
             "error_check": "encoder-side reconstruction only; no independent decoder",
             "lossless_overrides": ["quantizer_fidelity_control_method=LOSSLESS", "psi=0 for every band",
                                    "periodic_error_updating_flag=NOT_USED", "error_update_period_exponent=0"]}
    metadata = {"study": study, "requested_p_max": args.p_max, "local_sum_types": list(SUM_TYPES),
                "prediction_modes": list(MODES), "model_path": str(repo)}
    example = next(iter(bases.values()))
    print(f"Fixed encoder: {example.entropy_coder_type.name}; D={example.get_dynamic_range_bits()}, "
          f"Omega={example.get_weight_component_resolution()}, R={example.register_size or 64}, "
          f"nu_min={example.get_weight_update_initial_parameter()}, "
          f"nu_max={example.get_weight_update_final_parameter()}, "
          f"t_inc={2**(example.weight_update_change_interval+4)}, "
          f"phi={example.fixed_damping_value}, psi=0; lossless.")
    print(f"P=0..{args.p_max} inclusive; {len(args.images)} images x 4 sum types x 2 modes "
          f"x {args.p_max+1} P values = {len(args.images)*8*(args.p_max+1)} configurations.")
    if args.dry_run:
        print("Dry run complete. No compression performed.")
        return
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    if not args.overwrite and (meta_path.exists() or csv_path.exists()):
        if not meta_path.exists():
            raise ValueError("CSV exists without metadata. Choose a new --output folder.")
        old = json.loads(meta_path.read_text(encoding="utf-8"))
        if old.get("study") != study:
            raise ValueError("Inputs, crop, model or fixed settings differ from the saved run. "
                             "Choose a new --output folder, or use --overwrite to replace its results.")
        rows = read_csv(csv_path)
        if any(int(r["p"]) > args.p_max for r in rows):
            raise ValueError("Saved results include higher P values. Use that --p-max or --plot-only.")
    if args.overwrite:
        # Only this script's named outputs are replaced, never the entire folder.
        for stem in [*SCENES, "all_images"]:
            for ext in ("png", "pdf"):
                (out / f"{stem}.{ext}").unlink(missing_ok=True)
    write_json(meta_path, metadata)
    write_csv(csv_path, rows)
    done = {job_key(r) for r in rows}
    if len(done) != len(rows):
        raise ValueError("Duplicate configuration rows in results.csv.")
    jobs = [(s, p, st, mode) for s in args.images for p in range(args.p_max + 1)
            for st in SUM_TYPES for mode in MODES]
    print(f"Already complete: {len(done)}; remaining: {len(jobs)-len(done)}. Results: {out}")
    current_job = None
    logs = io.StringIO()
    try:
        for index, key in enumerate(jobs, 1):
            if key in done:
                continue
            scene, p, local_sum, mode = current_job = key
            print(f"[{index}/{len(jobs)}] {scene}: P={p}, {SUM_TYPES[local_sum]}, {mode.lower()} ...", flush=True)
            logs = io.StringIO()
            with redirect_stdout(logs):
                result = encode_one(cubes[scene], bases[scene], p, local_sum, mode, model)
            result.update(image=scene, capture=SCENES[scene])
            rows.append(result)
            write_csv(csv_path, rows)
            done.add(key)
            print(f"  {result['bits_per_sample']:.5f} bits/sample; max error={result['max_error']}; "
                  f"predictor+encoder={result['predictor_seconds']+result['encoder_seconds']:.2f} s", flush=True)
            if p == args.p_max and local_sum == list(SUM_TYPES)[-1] and mode == MODES[-1]:
                plot_results(rows, metadata, out)
    except KeyboardInterrupt:
        print("\nStopped. Completed rows are saved; rerun the same command to resume.")
    except Exception:
        (out / "last_error.txt").write_text(f"Configuration: {current_job}\n{logs.getvalue()}\n"
                                           + traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        if rows:
            plot_results(rows, metadata, out)
    print(f"Saved {len(rows)}/{len(jobs)} configurations. Plots and CSV: {out}")
    print("Error checks use encoder-side reconstruction, not an independent decoder.")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, ModuleNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
