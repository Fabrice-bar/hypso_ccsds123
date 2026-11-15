#!/usr/bin/env python3
"""HYPSO adaptation of CCSDS 120.2-G-2 Figure 4-14. See README.md.

Each worker encodes one complete configuration. Spectral grouping is part of
the experiment, not a parallelization shortcut. No added noise or resampling.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import h5py
import numpy as np

SCENES = {
    "lisbon": "lisbon_2022-09-22T10-55-19Z-l1a.nc",
    "bangladesh": "bangladesh_2022-10-26T04-02-56Z-l1a.nc",
    "santarem": "santarem_2022-09-24T13-52-04Z-l1a.nc",
    "doha": "doha_2022-09-12T06-38-31Z-l1a.nc",
    "finnmark": "finnmark_2022-07-06T19-40-03Z-l1a.nc",
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def write_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    temp = path.with_suffix(".tmp")
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with temp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in row.items()})
    os.replace(temp, path)


def find_repo(explicit):
    candidates = [Path(explicit)] if explicit else [Path.cwd(), *Path(__file__).resolve().parents]
    for p in candidates:
        if (p / "ccsds123_i2_hlm" / "predictor.py").is_file():
            return p.resolve()
    raise ValueError("Run from the verification-model repository or supply --repo PATH.")


def find_scene(folder, name):
    matches = sorted({p.resolve() for p in Path(folder).rglob(SCENES[name]) if p.is_file()})
    if len(matches) != 1:
        raise ValueError(f"Expected one {SCENES[name]} under {folder}; found {len(matches)}: {matches}")
    return matches[0]


def dimension_order(dataset, requested):
    if requested != "auto":
        return requested
    aliases = {"lines": "y", "line": "y", "rows": "y", "y": "y",
               "samples": "x", "sample": "x", "columns": "x", "x": "x",
               "bands": "z", "band": "z", "wavelength": "z", "wavelengths": "z", "z": "z"}
    axes = []
    for dim in dataset.dims:
        names = [v.name.rsplit("/", 1)[-1].lower() for v in dim.values()]
        if dim.label:
            names.append(dim.label.lower())
        found = {aliases[n] for n in names if n in aliases}
        axes.append(next(iter(found)) if len(found) == 1 else "?")
    order = "".join(axes)
    if sorted(order) != ["x", "y", "z"]:
        raise ValueError("Cannot identify DN axes. Use --axis-order only after inspecting the file dimensions.")
    return order


def select_bands(w, args):
    valid = np.flatnonzero((w >= args.wavelength_min) & (w <= args.wavelength_max))
    runs = [r for r in np.split(valid, np.flatnonzero(np.diff(valid) != 1) + 1) if len(r)]
    if not runs:
        raise ValueError("No bands within the requested wavelength limits.")
    run = max(runs, key=len)
    if args.band_start is None:
        if len(run) < args.band_count:
            raise ValueError(f"Need {args.band_count} consecutive valid bands; longest run has {len(run)}.")
        start = int(run[(len(run) - args.band_count) // 2])
    else:
        start = args.band_start
    chosen = np.arange(start, start + args.band_count)
    if start < 0 or chosen[-1] >= len(w) or not np.all(np.isin(chosen, valid)):
        raise ValueError("Selected band window lies outside the valid wavelength range.")
    return chosen


def make_groups(indices, w, group_bands, strides):
    """Cover the SAME native band set exactly once for each stride."""
    groups = []
    for stride in strides:
        if len(indices) % (group_bands * stride):
            raise ValueError("band_count must be divisible by group_bands times every stride.")
        seen = []
        for block in range(len(indices) // (group_bands * stride)):
            for phase in range(stride):
                positions = block * group_bands * stride + phase + stride * np.arange(group_bands)
                native = indices[positions]
                gaps = np.diff(w[native])
                groups.append({"stride": stride, "group": block * stride + phase,
                               "positions": positions.tolist(), "native_indices": native.tolist(),
                               "wavelengths_nm": w[native].tolist(),
                               "spacing_median_nm": float(np.median(gaps)),
                               "spacing_min_nm": float(gaps.min()), "spacing_max_nm": float(gaps.max())})
                seen.extend(native.tolist())
        if sorted(seen) != sorted(indices.tolist()):
            raise RuntimeError("Spectral grouping did not cover the common band set exactly once.")
    return groups


def inspect_scene(path, name, args):
    with h5py.File(path, "r") as f:
        d = f[args.data_variable]
        if d.ndim != 3 or d.dtype.kind not in "iu":
            raise ValueError(f"{name}: expected a three-dimensional integer DN variable.")
        if float(d.attrs.get("scale_factor", 1)) != 1 or float(d.attrs.get("add_offset", 0)) != 0:
            raise ValueError("Packed/scaled DN variables require an explicit conversion; none is applied here.")
        order = dimension_order(d, args.axis_order)
        y, x, z = [int(d.shape[order.index(a)]) for a in "yxz"]
        w = np.asarray(f[args.wavelength_variable][:], dtype=np.float64).squeeze()
        if w.ndim != 1 or len(w) != z or not np.all(np.isfinite(w)) or not np.all(np.diff(w) > 0):
            raise ValueError(f"{name}: need one strictly increasing calibrated wavelength per native band.")
        units = f[args.wavelength_variable].attrs.get("units", "nm (HYPSO convention; attribute absent)")
        if isinstance(units, bytes):
            units = units.decode()
        if str(units).lower().strip() in ("um", "µm", "micrometer", "micrometers"):
            w *= 1000
        elif not str(units).lower().startswith(("nm", "nanomet")):
            raise ValueError(f"Unsupported wavelength units: {units}")
        chosen = select_bands(w, args)
        h, width = (y, x) if args.full_image else (args.crop or [256, 256])
        row = 0 if args.full_image else ((y - h) // 2 if args.row is None else args.row)
        col = 0 if args.full_image else ((x - width) // 2 if args.col is None else args.col)
        if min(h, width) < 2 or min(row, col) < 0 or row + h > y or col + width > x:
            raise ValueError(f"{name}: crop {h}x{width} at ({row},{col}) does not fit {y}x{x}.")
        if max(h, width, args.group_bands) >= 65536:
            raise ValueError("This adapter supports dimensions 2..65535.")
        return {"image": name, "path": str(path.resolve()), "file_bytes": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns, "axis_order": order, "native_shape_yxz": [y, x, z],
                "crop_yx": [h, width], "origin_yx": [row, col], "full_image": args.full_image,
                "wavelength_units_source": str(units), "native_wavelengths_nm": w.tolist(),
                "selected_native_indices": chosen.tolist(),
                "groups": make_groups(chosen, w, args.group_bands, args.strides)}


def cache_scene(scene, args, path):
    h, width = scene["crop_yx"]
    row, col = scene["origin_yx"]
    indices = scene["selected_native_indices"]
    temp = path.with_suffix(".partial.npy")
    cache = np.lib.format.open_memmap(temp, mode="w+", dtype="uint16", shape=(h, width, len(indices)))
    minimum, maximum = 65535, 0
    with h5py.File(scene["path"], "r") as f:
        d, order = f[args.data_variable], scene["axis_order"]
        fill_values = [d.attrs[k] for k in ("_FillValue", "missing_value") if k in d.attrs]
        for yy in range(0, h, 32):
            select = {"y": slice(row + yy, row + min(yy + 32, h)),
                      "x": slice(col, col + width), "z": slice(indices[0], indices[-1] + 1)}
            block = np.transpose(d[tuple(select[a] for a in order)], [order.index(a) for a in "yxz"])
            lo, hi = int(block.min()), int(block.max())
            if lo < 0 or hi > 65535:
                raise ValueError("DN samples outside unsigned 16-bit range.")
            if any(np.any(np.isin(block, v)) for v in fill_values):
                raise ValueError(f"{scene['image']}: crop contains flagged missing values. Choose a valid crop.")
            minimum, maximum = min(minimum, lo), max(maximum, hi)
            cache[yy:yy + len(block)] = block
    cache.flush()
    del cache
    os.replace(temp, path)
    return {"cache_sha256": file_sha(path), "minimum_dn": minimum, "maximum_dn": maximum}


def contexts(manifest):
    result = []
    s = manifest["settings"]
    for scene in manifest["scenes"]:
        for a, p, g in itertools.product(s["a_values"], s["p_values"], scene["groups"]):
            result.append({"id": f"{scene['image']}_A{a}_s{g['stride']}_g{g['group']:02d}_P{p:02d}",
                           "image": scene["image"], "a": a, "p": p,
                           "stride": g["stride"], "group": g["group"], "positions": g["positions"]})
    return result


def seeds(settings):
    n = settings["group_bands"]
    result = [("baseline", [0] * n, [0] * n)]
    for phi, psi in itertools.product(settings["phi_values"], settings["psi_values"]):
        if phi or psi:
            result.append((f"phi{phi}_psi{psi}", [phi] * n, [psi] * n))
    return result


def task(c, label, phi, psi, manifest):
    return {"id": c["id"] + "__" + label, "context": c, "label": label,
            "phi": phi, "psi": psi, "study_hash": manifest["study_hash"]}


def result_path(out, task_id):
    return out / "jobs" / (task_id + ".result.json")


def get_result(out, j):
    path = result_path(out, j["id"])
    if not path.exists():
        return None
    r = read_json(path)
    if r.get("job_hash") != signature(j):
        raise ValueError(f"Checkpoint does not match this job: {path}")
    return r


def proposal(c, manifest, out):
    jobs = [task(c, label, phi, psi, manifest) for label, phi, psi in seeds(manifest["settings"])]
    results = [get_result(out, j) for j in jobs]
    if any(r is None for r in results):
        return None, jobs, results
    # Entropy only proposes a profile. Its ACTUAL compressed rate is measured later.
    choice = np.argmin(np.array([r["residual_entropy_per_band"] for r in results]), axis=0)
    phi = [jobs[int(k)]["phi"][z] for z, k in enumerate(choice)]
    psi = [jobs[int(k)]["psi"][z] for z, k in enumerate(choice)]
    if any(j["phi"] == phi and j["psi"] == psi for j in jobs):
        return None, jobs, results
    return task(c, "band_profile", phi, psi, manifest), jobs, results


def estimate_gib(scene, settings):
    count = math.prod(scene["crop_yx"]) * settings["group_bands"]
    return count * (320 + 16 * (max(settings["p_values"]) + 3)) / 2**30 + 0.5


def available_gib():
    import psutil
    available = psutil.virtual_memory().available
    try:
        limit = Path("/sys/fs/cgroup/memory.max").read_text().strip()
        if limit != "max":
            current = int(Path("/sys/fs/cgroup/memory.current").read_text().strip())
            available = min(available, max(0, int(limit) - current))
    except (OSError, ValueError):
        pass
    return available / 2**30


def run_tasks(jobs, manifest, out, args):
    pending = [j for j in jobs if get_result(out, j) is None]
    if not pending:
        print("All jobs in this phase already complete.", flush=True)
        return
    largest = max(estimate_gib(s, manifest["settings"]) for s in manifest["scenes"])
    budget = min(args.memory_budget_gb, available_gib() * 0.8)
    slots = min(args.workers, os.cpu_count() or 1, len(pending), max(0, int((budget - 0.75) // largest)))
    if slots < 1:
        raise ValueError(f"Estimated {largest:.2f} GiB/job + 0.75 GiB coordinator exceeds budget {budget:.2f} GiB.")
    print(f"{len(pending)} pending encodes; {slots} workers; estimate {largest:.2f} GiB/job; budget {budget:.1f} GiB.", flush=True)
    env = os.environ.copy()
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[name] = "1"
    active, submitted, completed = {}, 0, 0
    started = last_update = time.monotonic()
    try:
        while completed < len(pending):
            while submitted < len(pending) and len(active) < slots:
                j = pending[submitted]
                jp = out / "jobs" / (j["id"] + ".job.json")
                write_json(jp, j)
                log_path = jp.with_suffix(".log")
                log = log_path.open("w", encoding="utf-8")
                process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--_worker", str(jp)],
                                           stdout=log, stderr=subprocess.STDOUT, env=env)
                active[process] = (j, log, log_path)
                submitted += 1
            for process in list(active):
                code = process.poll()
                if code is None:
                    continue
                j, log, log_path = active.pop(process)
                log.close()
                if code or get_result(out, j) is None:
                    raise RuntimeError(f"Job failed: {j['id']}. Read {log_path}. Completed jobs remain resumable.")
                completed += 1
                print(f"[{completed}/{len(pending)}] {j['id']}: {get_result(out,j)['bits_per_sample']:.6f} bits/sample", flush=True)
                last_update = time.monotonic()
            if time.monotonic() - last_update > 30:
                print(f"Still running: {len(active)} encodes; phase elapsed {(time.monotonic()-started)/60:.1f} min.", flush=True)
                last_update = time.monotonic()
            time.sleep(0.2)
    finally:
        for process, (_, log, _) in active.items():
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            log.close()


def worker(job_path):
    jp = Path(job_path).resolve()
    out = jp.parent.parent
    manifest, j = read_json(out / "manifest.json"), read_json(jp)
    if j["study_hash"] != manifest["study_hash"]:
        raise ValueError("Worker job/manifest mismatch.")
    from codec import setup, encode
    model = setup(manifest["repo"])
    mapped = np.load(out / "cache" / (j["context"]["image"] + ".npy"), mmap_mode="r")
    cube = np.array(mapped[:, :, j["context"]["positions"]], dtype=np.int64, order="C")
    del mapped
    r = encode(cube, manifest["settings"], j["context"]["a"], j["context"]["p"], j["phi"], j["psi"], model)
    r.update({"job_hash": signature(j), "job_id": j["id"], "context": j["context"],
              "label": j["label"], "phi": j["phi"], "psi": j["psi"]})
    write_json(result_path(out, j["id"]), r)


def collect(manifest, out):
    evaluations, groups, profiles = [], [], []
    settings = manifest["settings"]
    scenes = {s["image"]: s for s in manifest["scenes"]}
    for c in contexts(manifest):
        extra, jobs, results = proposal(c, manifest, out)
        if extra:
            jobs.append(extra)
            results.append(get_result(out, extra))
        key = {k: v for k, v in c.items() if k not in ("id", "positions")}
        for r in results:
            if r:
                evaluations.append({**key, **{k: v for k, v in r.items() if k != "context"}})
        if any(r is None for r in results):
            continue
        best = min(results, key=lambda r: (r["stream_bits"], r["job_id"]))
        baseline = results[0]
        groups.append({**key, "samples": best["samples"], "baseline_bits": baseline["stream_bits"],
                       "best_bits": best["stream_bits"], "baseline_bps": baseline["bits_per_sample"],
                       "best_bps": best["bits_per_sample"], "best_label": best["label"],
                       "phi": best["phi"], "psi": best["psi"], "evaluations": len(results),
                       "max_error_dn": best["max_error_dn"], "rmse_dn": best["rmse_dn"],
                       "final_nu_reached": best["final_nu_reached"]})
        scene = scenes[c["image"]]
        for z, pos in enumerate(c["positions"]):
            native = scene["selected_native_indices"][pos]
            profiles.append({**key, "group_band": z, "native_band_index": native,
                             "wavelength_nm": scene["native_wavelengths_nm"][native],
                             "phi": best["phi"][z], "psi": best["psi"][z],
                             "phi_normalized": best["phi"][z] / 2**settings["theta"],
                             "psi_normalized": best["psi"][z] / 2**settings["theta"]})
    grouped = defaultdict(list)
    for r in groups:
        grouped[(r["image"], r["a"], r["stride"], r["p"])].append(r)
    summaries = []
    expected = settings["band_count"] // settings["group_bands"]
    for (scene, a, stride, p), rows in sorted(grouped.items()):
        if len(rows) != expected:
            continue  # Never compare averages over different incomplete subsets.
        n = sum(r["samples"] for r in rows)
        baseline = sum(r["baseline_bits"] for r in rows) / n
        best = sum(r["best_bits"] for r in rows) / n
        summaries.append({"image": scene, "a": a, "stride": stride, "p": p,
                          "groups": expected, "baseline_bps": baseline, "best_bps": best,
                          "saving_percent": 100 * (baseline - best) / baseline,
                          "all_reached_final_nu": all(r["final_nu_reached"] for r in rows)})
    for name, rows in (("evaluations", evaluations), ("group_results", groups),
                       ("best_profiles", profiles), ("summary", summaries)):
        write_csv(out / (name + ".csv"), rows)
    return summaries, len(groups)


def plot(manifest, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    rows, complete = collect(manifest, out)
    settings, scenes = manifest["settings"], manifest["scenes"]
    avals = settings["a_values"]
    indexed = {(r["image"], r["a"], r["stride"], r["p"]): r for r in rows}
    colors, expected = plt.get_cmap("tab10").colors, len(contexts(manifest))

    def draw(ax, scene, a):
        for i, stride in enumerate(settings["strides"]):
            gaps = np.concatenate([np.diff(g["wavelengths_nm"]) for g in scene["groups"] if g["stride"] == stride])
            label = f"stride {stride}: median {np.median(gaps):.2f} nm"
            rr = [indexed.get((scene["image"], a, stride, p)) for p in settings["p_values"]]
            for field, style, filled in (("best_bps", "-", True), ("baseline_bps", "--", False)):
                ax.plot(settings["p_values"], [r[field] if r else np.nan for r in rr],
                        color=colors[i % len(colors)], linestyle=style, marker="o", markersize=4,
                        markerfacecolor=colors[i % len(colors)] if filled else "white",
                        label=label if filled else None)
        h, w = scene["crop_yx"]
        ax.set_title(f"{scene['image'].title()} | A = {a} DN\n{h} × {w} spatial pixels; {settings['group_bands']} bands/group")
        ax.set_xlabel("Number of preceding prediction bands, P")
        ax.set_ylabel("Compressed bits/sample")
        ax.set_xticks(settings["p_values"])
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc="best")

    def save(selected, name):
        compact = len(avals) == 1 and len(selected) > 1
        ncols = 2 if compact else len(avals)
        nrows = math.ceil(len(selected) / 2) if compact else len(selected)
        height = 3.8 * nrows + 2.4
        fig, axes = plt.subplots(nrows, ncols, figsize=(max(7.5, 6.1 * ncols), height), squeeze=False)
        if compact:
            for ax, scene in zip(axes.flat, selected):
                draw(ax, scene, avals[0])
            for ax in list(axes.flat)[len(selected):]:
                ax.set_visible(False)
        else:
            for i, scene in enumerate(selected):
                for j, a in enumerate(avals):
                    draw(axes[i, j], scene, a)
        state = "complete" if complete == expected else f"PARTIAL: {complete}/{expected} groups searched"
        fig.suptitle("HYPSO Figure 4-14 study: spectral spacing and P\n"
                     + f"Narrow-neighbor, reduced, hybrid | νmin={settings['nu_min']}, νmax={settings['nu_max']}\n"
                     + f"tinc={settings['t_inc']}, Θ={settings['theta']} | {state}", fontsize=12, y=1 - 0.12 / height)
        fig.legend(handles=[Line2D([], [], color="black", marker="o", label="Solid: best tested profile (including baseline)"),
                            Line2D([], [], color="black", marker="o", markerfacecolor="white", linestyle="--",
                                   label="Dashed: φz = ψz = 0 (same A)")],
                   loc="lower center", bbox_to_anchor=(0.5, 0.58 / height), fontsize=9)
        fig.text(0.5, 0.10 / height,
                 "Same selected native bands at every stride; no added noise or spectral averaging.\n"
                 "Rate includes profile headers, entropy payload/tail and word padding.\n"
                 "External initialization excluded.", ha="center", va="bottom", fontsize=8)
        fig.tight_layout(rect=(0, 1.23 / height, 1, 1 - 0.95 / height))
        fig.savefig(out / (name + ".png"), dpi=180)
        fig.savefig(out / (name + ".pdf"))
        plt.close(fig)
    save(scenes, "all_images_fig414")
    for scene in scenes:
        save([scene], scene["image"] + "_fig414")
    return complete, expected


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--input-dir", type=Path, default=Path("data"))
    p.add_argument("--images", nargs="+", choices=list(SCENES), default=list(SCENES))
    spatial = p.add_mutually_exclusive_group()
    spatial.add_argument("--crop", type=int, nargs=2, metavar=("ROWS", "COLS"), help="Default centered 256 256.")
    spatial.add_argument("--full-image", action="store_true", help="Each file's full spatial extent; spectral grouping remains.")
    p.add_argument("--row", type=int, help="Zero-based crop origin; default centered.")
    p.add_argument("--col", type=int, help="Zero-based crop origin; default centered.")
    p.add_argument("--data-variable", default="products/dn")
    p.add_argument("--wavelength-variable", default="metadata/corrections/wavelengths")
    p.add_argument("--axis-order", choices=["auto", "yxz", "yzx", "xyz", "xzy", "zyx", "zxy"], default="auto")
    p.add_argument("--wavelength-min", type=float, default=400)
    p.add_argument("--wavelength-max", type=float, default=800)
    p.add_argument("--band-count", type=int, default=96)
    p.add_argument("--band-start", type=int, help="Native zero-based start index; default centers the valid spectral window.")
    p.add_argument("--group-bands", type=int, default=24)
    p.add_argument("--strides", type=int, nargs="+", default=[1, 2, 4])
    p.add_argument("--p-values", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    p.add_argument("--a-values", type=int, nargs="+", default=[3], help="Near-lossless A=1..15; add 15 if wanted.")
    p.add_argument("--phi-values", type=int, nargs="+", default=[2, 3, 4], help="Uniform seed candidates.")
    p.add_argument("--psi-values", type=int, nargs="+", default=[7], help="Uniform seed candidates; baseline 0,0 always included.")
    p.add_argument("--theta", type=int, default=3)
    p.add_argument("--nu-min", type=int, default=-1)
    p.add_argument("--nu-max", type=int, default=3, help="Fixed across all phi values.")
    p.add_argument("--t-inc", type=int, default=2048)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--memory-budget-gb", type=float, default=90, help="Aggregate estimated budget in GiB; not a hard OS limit.")
    p.add_argument("--repo", type=Path)
    p.add_argument("--output", type=Path, default=Path("results/fig414_256_A3"))
    p.add_argument("--dry-run", action="store_true", help="Report plan without encoding or caching DN.")
    p.add_argument("--resume", action="store_true", help="Continue an identical study, skipping completed encodes.")
    p.add_argument("--plot-only", action="store_true", help="Rebuild CSVs/plots from --output; no source/model needed.")
    a = p.parse_args()
    if a.plot_only:
        return a
    if a.full_image and (a.row not in (None, 0) or a.col not in (None, 0)):
        p.error("--full-image requires origins zero or omitted.")
    for name in ("p_values", "a_values", "phi_values", "psi_values", "strides"):
        setattr(a, name, sorted(set(getattr(a, name))))
    a.images = list(dict.fromkeys(a.images))
    if a.band_count < 2 or a.group_bands < 2 or any(k < 1 for k in a.strides):
        p.error("Positive strides and at least two bands per group are required.")
    if any(a.band_count % (a.group_bands * k) for k in a.strides):
        p.error("band-count must be divisible by group-bands times EACH stride.")
    if any(not 1 <= v <= 15 for v in a.a_values):
        p.error("Use A=1..15; lossless is omitted in this offset/damping study.")
    if any(not 0 <= v <= 15 or v >= a.group_bands for v in a.p_values):
        p.error("Require 0<=P<=15 and P<group-bands.")
    if not 0 <= a.theta <= 4 or any(not 0 <= v < 2**a.theta for v in a.phi_values + a.psi_values):
        p.error("Require Theta=0..4 and 0<=phi,psi<2^Theta.")
    if not -6 <= a.nu_min <= a.nu_max <= 9 or a.t_inc not in [2**i for i in range(4, 12)]:
        p.error("Require -6<=nu-min<=nu-max<=9 and t-inc in 16,32,...,2048.")
    if a.workers < 1 or not math.isfinite(a.memory_budget_gb) or a.memory_budget_gb <= 0:
        p.error("Worker count and memory budget must be positive and finite.")
    if not np.isfinite(a.wavelength_min + a.wavelength_max) or a.wavelength_min >= a.wavelength_max:
        p.error("Invalid wavelength limits.")
    return a


def main():
    args = parse_args()
    out = args.output.resolve()
    if args.plot_only:
        done, total = plot(read_json(out / "manifest.json"), out)
        print(f"Plots rebuilt: {done}/{total} complete groups. {out}")
        return
    repo = find_repo(args.repo)
    scenes = [inspect_scene(find_scene(args.input_dir, s), s, args) for s in args.images]
    keys = ["p_values", "a_values", "phi_values", "psi_values", "strides", "band_count", "group_bands",
            "theta", "nu_min", "nu_max", "t_inc", "data_variable", "wavelength_variable",
            "wavelength_min", "wavelength_max"]
    settings = {k: getattr(args, k) for k in keys}
    settings.update({"search": "uniform_seeds_then_entropy_guided_band_profile_v1",
                     "sample_type": "uint16_D16", "prediction_mode": "reduced", "local_sum": "narrow-neighbor",
                     "encoder": "hybrid", "R": 64, "Omega": 19, "noise_added_dn": 0,
                     "sample_encoding_order": "BIL", "hybrid_initial_accumulator": 8})
    spec = {"settings": settings, "scenes": scenes, "repo": str(repo)}
    cases, initial = contexts(spec), seeds(settings)
    for s in scenes:
        ix, w = s["selected_native_indices"], s["native_wavelengths_nm"]
        print(f"{s['image']}: native {s['native_shape_yxz']}; crop {s['crop_yx']} at {s['origin_yx']}; "
              f"native bands {ix[0]}..{ix[-1]}, {w[ix[0]]:.2f}..{w[ix[-1]]:.2f} nm", flush=True)
        for k in settings["strides"]:
            gaps = np.concatenate([np.diff(g["wavelengths_nm"]) for g in s["groups"] if g["stride"] == k])
            print(f"  stride {k}: {len(ix)//args.group_bands} x {args.group_bands} bands; "
                  f"spacing median {np.median(gaps):.3f} nm, range {gaps.min():.3f}..{gaps.max():.3f} nm")
    print(f"{len(cases)} group/P/A cases; {len(cases)*len(initial)} seed encodes plus up to {len(cases)} proposals. "
          f"Total <= {len(cases)*(len(initial)+1)}.", flush=True)
    print("Full image means full SPATIAL size; each encoding retains the spectral group size.")
    if args.dry_run:
        for s in scenes:
            print(f"{s['image']}: conservative allowance {estimate_gib(s,settings):.2f} GiB/worker")
        return
    model_files = sorted((repo / "ccsds123_i2_hlm").rglob("*.py")) + sorted((repo / "ccsds123_i2_hlm").rglob("*.txt"))
    code_files = sorted(Path(__file__).resolve().parent.glob("*.py"))
    spec.update({"version": "1.0", "python_version": sys.version.split()[0], "numpy_version": np.__version__,
                 "model_sha256": signature({str(p.relative_to(repo)): file_sha(p) for p in model_files}),
                 "script_sha256": signature({p.name: file_sha(p) for p in code_files})})
    study_hash, mp = signature(spec), out / "manifest.json"
    if mp.exists():
        if not args.resume:
            raise ValueError("Study already exists. Add --resume for identical settings or choose a new --output.")
        manifest = read_json(mp)
        if manifest["study_hash"] != study_hash:
            raise ValueError("Inputs, settings, model, interpreter or scripts changed. Use a new --output.")
    else:
        if out.exists() and any(out.iterdir()):
            raise ValueError("Output is not empty and has no study manifest. Choose another --output.")
        out.mkdir(parents=True, exist_ok=True)
        (out / "cache").mkdir(exist_ok=True)
        (out / "jobs").mkdir(exist_ok=True)
        manifest = {**spec, "study_hash": study_hash, "cache_info": {}}
        write_json(mp, manifest)
    for s in scenes:
        cp = out / "cache" / (s["image"] + ".npy")
        info = manifest["cache_info"].get(s["image"])
        if info and cp.exists() and file_sha(cp) == info["cache_sha256"]:
            continue
        print(f"Preparing unchanged DN crop: {s['image']}", flush=True)
        new_info = cache_scene(s, args, cp)
        if info and info != new_info:
            raise ValueError("DN content differs from the checkpoint; use a new --output.")
        manifest["cache_info"][s["image"]] = new_info
        write_json(mp, manifest)
    band_rows = []
    for s in scenes:
        for g in s["groups"]:
            for z, (native, w) in enumerate(zip(g["native_indices"], g["wavelengths_nm"])):
                band_rows.append({"image": s["image"], "stride": g["stride"], "group": g["group"],
                                  "group_band": z, "native_band_index": native, "wavelength_nm": w})
    write_csv(out / "band_groups.csv", band_rows)
    try:
        print("Phase 1: baseline and uniform candidate profiles.", flush=True)
        run_tasks([task(c, label, phi, psi, manifest) for c in cases for label, phi, psi in initial], manifest, out, args)
        extras = [proposal(c, manifest, out)[0] for c in cases]
        extras = [j for j in extras if j is not None]
        print(f"Phase 2: {len(extras)} distinct band-dependent profile proposals.", flush=True)
        run_tasks(extras, manifest, out, args)
    finally:
        done, total = plot(manifest, out)
        print(f"Saved plots and CSVs: {done}/{total} complete groups. {out}", flush=True)


if __name__ == "__main__":
    try:
        if len(sys.argv) == 3 and sys.argv[1] == "--_worker":
            worker(sys.argv[2])
        else:
            main()
    except KeyboardInterrupt:
        print("Interrupted. Repeat the same command with --resume.", file=sys.stderr)
        sys.exit(130)
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        if "--_worker" in sys.argv:
            traceback.print_exc()
        sys.exit(1)
