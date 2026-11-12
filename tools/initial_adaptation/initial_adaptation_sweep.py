#!/usr/bin/env python3
r"""HYPSO initial adaptation study, following CCSDS 120.2-G-2 Figures 4-8/10/11.

Defaults: five centre 128x128 crops, all native bands, raw products/dn;
P=3, narrow-neighbour, reduced, hybrid, D=16, Omega=19, R=64, Theta=3.
Sweep phi={0,1}, nu_max={0,2,4}, nu_min={-6,0}, t_inc={16,128,1024},
A={0,3,15}: 540 configurations. Offset psi=7 for A>0, and 0 for A=0.

Run from the verification repository root (PowerShell: .\.venv\Scripts\python.exe):
  python tools/initial_adaptation/initial_adaptation_sweep.py --input-dir data --low-memory --workers 2
  python tools/initial_adaptation/initial_adaptation_sweep.py --input-dir data --images lisbon --crop 16 16 --a-values 0 3 --vmax-values 0 2 --low-memory --output results/initial_smoke

--crop-from prior/metadata.json reuses and verifies exact prior crop coordinates,
band count and DN hash. Otherwise default is the same centre 128x128 convention.
Use --dry-run before a long run. Repeat the identical command to resume;
--workers may change on resume. A new scientific grid needs a new output folder.
--plot-only --output PATH needs no input data or codec.

Each phi gets a five-image by three-A rate plot and a relative-change plot versus
nu_min=-6,t_inc=128 at the SAME phi,nu_max,A,image. Negative change is an improvement.
The baseline must be included in your grid to produce relative-change curves.
Small-crop schedules that never reach nu_max use hollow markers and CSV flags.
best_per_scene.csv and profile_ranking.csv report observed minima and common-profile
rate penalties across images, separately for each A. Partial results are labelled.
Each configuration is a full predictor/encoder run; no tiling or warm start is used.
Checks use encoder-side reconstruction, not an independent decoder.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys

import numpy as np

import _damping_core as core
import sweep_workers as sw

VERSION = "1.0.0-initial-adaptation"
SCENES = core.SCENES
BASELINE = (-6, 128)
PROFILE_KEYS = ("phi", "nu_max", "nu_min", "t_inc")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", type=Path, default=Path("data"))
    p.add_argument("--images", nargs="+", choices=list(SCENES), default=list(SCENES))
    p.add_argument("--crop", type=int, nargs=2, metavar=("ROWS", "COLS"))
    p.add_argument("--row", type=int, help="Zero-based origin; default centre.")
    p.add_argument("--col", type=int, help="Zero-based origin; default centre.")
    p.add_argument("--crop-from", type=Path, help="Previous metadata.json: reuse and verify exact crops.")
    p.add_argument("--variable", default="products/dn")
    p.add_argument("--axis-order", choices=("yxz", "yzx", "xyz", "xzy", "zxy", "zyx"))
    p.add_argument("--nu-min-values", type=int, nargs="+", default=[-6, 0])
    p.add_argument("--t-inc-values", type=int, nargs="+", default=[16, 128, 1024])
    p.add_argument("--vmax-values", type=int, nargs="+", default=[0, 2, 4])
    p.add_argument("--phi-values", type=int, nargs="+", default=[0, 1])
    p.add_argument("--a-values", type=int, nargs="+", default=[0, 3, 15])
    p.add_argument("--p", type=int, default=3)
    p.add_argument("--local-sum", choices=list(core.LOCAL_SUM_OPTIONS), default="narrow-neighbor")
    p.add_argument("--prediction-mode", choices=("full", "reduced"), default="reduced")
    p.add_argument("--psi", type=int, default=7, help="Fixed offset at A>0; forced zero for A=0.")
    p.add_argument("--repo", type=Path, help="Verification model root; otherwise find from cwd/script ancestors.")
    p.add_argument("--output", type=Path, default=Path("results/initial_adaptation_128"))
    p.add_argument("--low-memory", action="store_true", help="Use the validated hybrid adapter without text traces.")
    p.add_argument("--max-memory-gb", type=float, default=8.0, help="Estimated per-job budget, GiB; not an OS limit.")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--plot-only", action="store_true")
    p.add_argument("--overwrite", action="store_true", help="Restart only a matching-tool output folder; never implicit.")
    sw.add_options(p)
    a = p.parse_args()
    for name in ("nu_min_values", "t_inc_values", "vmax_values", "phi_values", "a_values"):
        setattr(a, name, sorted(set(getattr(a, name))))
    if any(not -6 <= x <= 9 for x in a.nu_min_values + a.vmax_values):
        p.error("nu_min and nu_max must lie in -6..9.")
    if max(a.nu_min_values) > min(a.vmax_values):
        p.error("Every nu_min must be <= every nu_max in this rectangular grid; no invalid pairs are silently skipped.")
    if any(not 16 <= x <= 2048 or x & (x - 1) for x in a.t_inc_values):
        p.error("Every t_inc must be a power of two from 16 through 2048.")
    if not 0 <= a.p <= 15 or not 0 <= a.psi <= 7 or any(not 0 <= x <= 7 for x in a.phi_values):
        p.error("Require P=0..15 and phi/psi=0..7 at Theta=3.")
    if any(not 0 <= x <= 32767 for x in a.a_values):
        p.error("Require A=0..32767 for this fixed unsigned D=16 profile.")
    if (a.crop is not None and min(a.crop) < 2) or any(x is not None and x < 0 for x in (a.row, a.col)):
        p.error("Crop dimensions must be >=2 and origins nonnegative.")
    if not math.isfinite(a.max_memory_gb) or a.max_memory_gb <= 0:
        p.error("--max-memory-gb must be positive and finite.")
    if a.crop_from and (a.crop is not None or a.row is not None or a.col is not None):
        p.error("--crop-from cannot be combined with --crop/--row/--col.")
    if a.overwrite and (a.dry_run or a.plot_only):
        p.error("--overwrite cannot be combined with --dry-run or --plot-only.")
    a.images = list(dict.fromkeys(a.images))
    a.crop = tuple(a.crop or (128, 128))
    a.full_image = False
    # Used only to load crops/build header templates; each job replaces the schedule.
    a.nu_min, a.t_inc = BASELINE
    a.vmax_min, a.vmax_max = min(a.vmax_values), max(a.vmax_values)
    return a


def job_key(row):
    return (row["image"], int(row["a"]), *(int(row[k]) for k in PROFILE_KEYS))


def profile_key(row):
    return tuple(int(row[k]) for k in PROFILE_KEYS)


def make_jobs(args):
    return [(s, a, phi, v, n, t) for s in args.images for a in args.a_values
            for phi in args.phi_values for v in args.vmax_values
            for n in args.nu_min_values for t in args.t_inc_values]


def _evaluate(key, cubes, model, state):
    args, bases = state
    scene, absolute_error, phi, nu_max, nu_min, t_inc = key
    local = copy.copy(args)
    local.nu_min, local.t_inc = nu_min, t_inc
    header = copy.deepcopy(bases[scene][absolute_error])
    header.weight_update_initial_parameter = nu_min + 6
    header.weight_update_change_interval = t_inc.bit_length() - 1 - 4
    # The shared, unchanged runner sets nu_max/phi, validates the header and bounds,
    # and checks the model's actual final update exponent against its schedule.
    row = core.encode_one(cubes[scene], header, local, absolute_error, nu_max, phi, model)
    row.update(image=scene, capture=SCENES[scene])
    return row


def save_table(path, fields, rows):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def write_summaries(rows, metadata, out):
    st = metadata["study"]
    grid = st["grid"]
    expected = math.prod(len(grid[k]) for k in ("phi_values", "vmax_values", "nu_min_values", "t_inc_values"))
    groups = {(s, a): [r for r in rows if r["image"] == s and int(r["a"]) == a]
              for s in st["images"] for a in st["a_values"]}
    best, minima = [], {}
    for (scene, a), group in groups.items():
        if not group:
            continue
        row = min(group, key=lambda r: (float(r["bits_per_sample"]), profile_key(r)))
        minima[(scene, a)] = float(row["bits_per_sample"])
        best.append({k: row[k] for k in ["image", "a", *PROFILE_KEYS, "bits_per_sample", "max_error", "rmse_dn", "nu_max_reached"]}
                    | {"completed_configurations": len(group), "expected_configurations": expected,
                       "complete": int(len(group) == expected)})
    best_fields = ["image", "a", *PROFILE_KEYS, "bits_per_sample", "max_error", "rmse_dn", "nu_max_reached",
                   "completed_configurations", "expected_configurations", "complete"]
    save_table(out / "best_per_scene.csv", best_fields, best)
    ranking = []
    for a in st["a_values"]:
        a_rows = [r for r in rows if int(r["a"]) == a]
        for key in sorted({profile_key(r) for r in a_rows}):
            selected = [r for r in a_rows if profile_key(r) == key]
            penalties = [100 * (float(r["bits_per_sample"]) / minima[(r["image"], a)] - 1) for r in selected]
            ranking.append(dict(zip(PROFILE_KEYS, key)) | {
                "a": a, "images_completed": len(selected), "images_expected": len(st["images"]),
                "all_images_present": int(len(selected) == len(st["images"])),
                "full_grid_complete": int(all(len(groups[(s, a)]) == expected for s in st["images"])),
                "mean_bits_per_sample": float(np.mean([float(r["bits_per_sample"]) for r in selected])),
                "pooled_bits_per_sample": sum(int(r["stream_bits"]) for r in selected) / sum(int(r["sample_count"]) for r in selected),
                "mean_rate_penalty_pct": float(np.mean(penalties)), "worst_rate_penalty_pct": max(penalties),
                "all_schedules_reach_vmax": int(all(int(r["nu_max_reached"]) for r in selected))})
    ranking.sort(key=lambda r: (r["a"], -r["all_images_present"], r["worst_rate_penalty_pct"], r["mean_rate_penalty_pct"],
                                *(r[k] for k in PROFILE_KEYS)))
    fields = ["a", *PROFILE_KEYS, "images_completed", "images_expected", "all_images_present", "full_grid_complete",
              "mean_bits_per_sample", "pooled_bits_per_sample", "mean_rate_penalty_pct", "worst_rate_penalty_pct",
              "all_schedules_reach_vmax"]
    save_table(out / "profile_ranking.csv", fields, ranking)


def plot_results(rows, metadata, out):
    if not rows:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    st, indexed = metadata["study"], {job_key(r): r for r in rows}
    grid, profile = st["grid"], st["profile"]
    scenes, a_values, vmax_values = st["images"], st["a_values"], grid["vmax_values"]
    schedules = [(n, t) for n in grid["nu_min_values"] for t in grid["t_inc_values"]]
    palette = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#5B4BCE", "#333333", "#56B4E9"]
    markers = ["o", "s", "D", "^", "v", "P", "X", "*"]
    handles = [Line2D([], [], color=palette[i % 8], marker=markers[i % 8], linewidth=1.3, markersize=4,
                      linestyle="-" if i < 8 else "--", label=rf"$\nu_{{min}}={n},\ t_{{inc}}={t}$")
               for i, (n, t) in enumerate(schedules)]
    expected_panel = len(schedules) * len(vmax_values)
    relative_available = BASELINE in schedules

    def draw(ax, scene, a, phi, relative):
        shape = st["inputs"][scene]["crop_shape_yxz"]
        count = sum((scene, a, phi, v, n, t) in indexed for v in vmax_values for n, t in schedules)
        title = f"{scene.title()} | A={a} DN\n{shape[0]} x {shape[1]} pixels, {shape[2]} bands"
        if count != expected_panel:
            title += f" | partial {count}/{expected_panel}"
        ax.set_title(title, fontsize=10)
        for i, (n, t) in enumerate(schedules):
            curve = [indexed.get((scene, a, phi, v, n, t)) for v in vmax_values]
            values = []
            for v, row in zip(vmax_values, curve):
                baseline = indexed.get((scene, a, phi, v, *BASELINE))
                value = float(row["bits_per_sample"]) if row else np.nan
                if relative:
                    value = 100 * (value / float(baseline["bits_per_sample"]) - 1) if baseline else np.nan
                values.append(value)
            color, marker = palette[i % 8], markers[i % 8]
            ax.plot(vmax_values, values, color=color, marker=marker, markersize=3.6,
                    linewidth=1.3, linestyle="-" if i < 8 else "--")
            hollow = [(v, y) for v, y, r in zip(vmax_values, values, curve)
                      if r is not None and not int(r["nu_max_reached"])]
            if hollow:
                ax.plot(*zip(*hollow), linestyle="none", marker=marker, markersize=4.4,
                        markerfacecolor="white", markeredgecolor=color)
        if relative:
            ax.axhline(0, color="#555555", linewidth=0.7)
        ax.set_xlabel(r"Final adaptation parameter, $\nu_{max}$", fontsize=10)
        ax.set_ylabel("Rate change from baseline (%)" if relative else "Compressed bits/sample", fontsize=10)
        ax.set_xticks(vmax_values)
        ax.set_xlim(min(vmax_values) - 0.25, max(vmax_values) + 0.25)
        ax.tick_params(labelsize=9)
        ax.grid(True, alpha=0.25)
        ax.set_axisbelow(True)

    def render(selected_scenes, phi, relative, stem):
        legend_rows = math.ceil(len(handles) / 3)
        bottom = 0.78 + 0.24 * legend_rows
        height = 3.10 * len(selected_scenes) + 0.75 + bottom
        fig, axes = plt.subplots(len(selected_scenes), len(a_values),
                                 figsize=(4.9 * len(a_values), height), squeeze=False)
        for i, scene in enumerate(selected_scenes):
            for j, a in enumerate(a_values):
                draw(axes[i, j], scene, a, phi, relative)
        title = "HYPSO initial adaptation study | hybrid encoder"
        subtitle = (f"P={profile['p']} | {profile['local_sum']}, {profile['prediction_mode']} | "
                    rf"$\phi={phi}$, $\Theta=3$ | six schedules" if len(schedules) == 6 else
                    f"P={profile['p']} | {profile['local_sum']}, {profile['prediction_mode']} | "
                    rf"$\phi={phi}$, $\Theta=3$ | {len(schedules)} schedules")
        fig.suptitle(title + "\n" + subtitle, fontsize=13, y=1 - 0.08 / height)
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.46 / height),
                   ncol=min(3, len(handles)), frameon=False, fontsize=9)
        note = ("Baseline: nu_min=-6, t_inc=128 at the SAME image, A, phi and nu_max; negative is better."
                if relative else "Rate includes header, entropy payload/tail and padding; external initialization is excluded.")
        note += "\nHollow markers: the requested nu_max was not reached by this crop."
        fig.text(0.5, 0.10 / height, note, ha="center", va="bottom", fontsize=8)
        fig.tight_layout(rect=(0, bottom / height, 1, 1 - 0.72 / height), h_pad=2.1, w_pad=1.8)
        for ext in ("png", "pdf"):
            fig.savefig(out / f"{stem}.{ext}", dpi=180, bbox_inches="tight")
        plt.close(fig)

    for phi in grid["phi_values"]:
        render(scenes, phi, False, f"all_images_initial_phi{phi}")
        if relative_available:
            render(scenes, phi, True, f"all_images_change_phi{phi}")
        for scene in scenes:
            if any(r["image"] == scene and int(r["phi"]) == phi for r in rows):
                render([scene], phi, False, f"{scene}_initial_phi{phi}")
    write_summaries(rows, metadata, out)


def find_repo(explicit):
    candidates = [explicit] if explicit else [Path.cwd(), *Path(__file__).resolve().parents]
    for root in candidates:
        if root and (root / "ccsds123_i2_hlm" / "header.py").is_file():
            return root.resolve()
    raise ValueError("Cannot find ccsds123_i2_hlm/header.py; run from the repository root or pass --repo PATH.")


def validate_prior_profile(args, previous, bases, model_sha):
    """--crop-from also guards fixed profile fields when source metadata has them."""
    if not previous:
        return
    st = previous.get("study", {})
    old_profile = st.get("profile", {})
    wanted = {"p": args.p, "local_sum": args.local_sum, "prediction_mode": args.prediction_mode,
              "psi_near_lossless": args.psi, "psi_lossless": 0, "theta": 3,
              "D": 16, "Omega": 19, "R": 64, "encoder": "HYBRID"}
    for key, value in wanted.items():
        if key in old_profile and old_profile[key] != value:
            raise ValueError(f"Previous study fixed setting {key}={old_profile[key]!r} differs from {value!r}. "
                             "Match the old setting, or omit --crop-from for a deliberately separate study.")
    if st.get("model_sha256") and st["model_sha256"] != model_sha:
        raise ValueError("The previous study used different model code/tables. Use the same model for a direct comparison.")
    previous_headers = st.get("headers_at_first_vmax_and_phi", st.get("header_templates", {}))
    varies = {"weight_update_initial_parameter", "weight_update_final_parameter", "weight_update_change_interval",
              "fixed_damping_value", "damping_table_array", "absolute_error_limit_bit_depth"}
    for scene, headers in bases.items():
        for a, header in headers.items():
            old = previous_headers.get(scene, {}).get(str(a))
            if not old:
                continue
            current = core.header_snapshot(header)
            for key in set(old) & set(current) - varies:
                if old[key] != current[key]:
                    raise ValueError(f"{scene}, A={a}: previous fixed header field {key} differs. "
                                     "Review the model/settings before comparing studies.")
            if old.get("absolute_error_limit_bit_depth") != current["absolute_error_limit_bit_depth"]:
                print("NOTE: absolute-error header field width differs from the previous run; rate includes this overhead.")


def clear_outputs(out):
    for pattern in ("all_images_initial_phi*.*", "all_images_change_phi*.*", "*_initial_phi*.png", "*_initial_phi*.pdf"):
        for path in out.glob(pattern):
            if path.suffix in (".png", ".pdf"):
                path.unlink()
    for name in ("best_per_scene.csv", "profile_ranking.csv", "last_error.txt"):
        (out / name).unlink(missing_ok=True)


def _main(args):
    out = args.output.resolve()
    csv_path, meta_path = out / "results.csv", out / "metadata.json"
    if args.plot_only:
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        if metadata.get("study", {}).get("tool") != "initial_adaptation":
            raise ValueError("This output is not an initial-adaptation study.")
        rows = core.read_csv(csv_path)
        if not rows:
            raise ValueError("No completed results to plot.")
        plot_results(rows, metadata, out)
        print(f"Plots: {out}")
        return
    if not __debug__:
        raise ValueError("Do not use python -O; the verification model relies on assertions.")
    if args.overwrite and (csv_path.exists() or meta_path.exists()):
        old = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        if old.get("study", {}).get("tool") != "initial_adaptation":
            raise ValueError("Refusing to overwrite a different study; choose a new output folder.")
    repo = find_repo(args.repo)
    hd = sw.load_model(repo, "vmax")[0]
    storage = {"low_memory": args.low_memory}
    if args.low_memory:
        from hybrid_no_traces import validate_reference, adapter_sha256
        storage.update(adapter_sha256=adapter_sha256(), reference_class_sha256=validate_reference())
    previous = json.loads(args.crop_from.read_text(encoding="utf-8")) if args.crop_from else None
    if previous:
        previous_inputs = previous.get("study", {}).get("inputs", {})
        if any(not previous_inputs.get(s, {}).get("crop_sha256") for s in args.images):
            raise ValueError("--crop-from requires a stored crop SHA256 for every selected scene.")
    cubes, inputs = core.load_inputs(args)
    for info in inputs.values():
        info.pop("highest_reachable_nu", None)  # Reachability varies with the swept schedule.
    bases = {s: {a: core.make_header(cube, args, a, min(args.vmax_values), hd) for a in args.a_values}
             for s, cube in cubes.items()}
    model_files = list((repo / "ccsds123_i2_hlm").rglob("*.py")) + list((repo / "ccsds123_i2_hlm").rglob("*.txt"))
    model_sha = core.file_digest(model_files, repo)
    validate_prior_profile(args, previous, bases, model_sha)
    identity_inputs = copy.deepcopy(inputs)
    for info in identity_inputs.values():
        info.pop("estimated_ram_gib")
    own_dir = Path(__file__).resolve().parent
    own_files = [own_dir / name for name in (Path(__file__).name, "_damping_core.py", "sweep_workers.py", "hybrid_no_traces.py")]
    profile = {"p": args.p, "local_sum": args.local_sum, "prediction_mode": args.prediction_mode,
               "psi_near_lossless": args.psi, "psi_lossless": 0, "theta": 3,
               "D": 16, "Omega": 19, "R": 64, "encoder": "HYBRID"}
    study = {"tool": "initial_adaptation", "script_version": VERSION,
             "script_bundle_sha256": core.file_digest(own_files, own_dir), "model_sha256": model_sha,
             "python": platform.python_version(), "numpy": np.__version__,
             "images": args.images, "inputs": identity_inputs, "profile": profile, "storage": storage,
             "a_values": args.a_values,
             "grid": {name: getattr(args, name) for name in ("phi_values", "vmax_values", "nu_min_values", "t_inc_values")},
             "header_templates": {s: {str(a): core.header_snapshot(h) for a, h in headers.items()} for s, headers in bases.items()},
             "baseline_schedule": {"nu_min": BASELINE[0], "t_inc": BASELINE[1]},
             "metric": "header + entropy payload/tail + output-word padding per scalar sample",
             "external_info": "External initialization excluded from primary rate; inclusive rate recorded.",
             "error_check": "encoder-side reconstruction, not an independent decoder",
             "reference": "CCSDS 120.2-G-2 section 4.2.2, Figures 4-8, 4-10 and 4-11; targeted HYPSO follow-up"}
    metadata = {"study": study, "model_path": str(repo), "crop_from": str(args.crop_from.resolve()) if args.crop_from else None}
    jobs = make_jobs(args)
    print(f"{len(jobs)} configurations: {len(args.images)} images x {len(args.a_values)} A x {len(args.phi_values)} phi "
          f"x {len(args.vmax_values)} nu_max x {len(args.nu_min_values)} nu_min x {len(args.t_inc_values)} t_inc.")
    print(f"P={args.p}, {args.local_sum}, {args.prediction_mode}, HYBRID; A={args.a_values}; "
          f"phi={args.phi_values}; nu_max={args.vmax_values}; nu_min={args.nu_min_values}; t_inc={args.t_inc_values}.")
    unreached = 0
    for s, a, phi, v, n, t in jobs:
        shape = cubes[s].shape
        unreached += not core.adaptation_schedule(shape[0], shape[1], n, t, v)["nu_max_reached"]
    if unreached:
        print(f"NOTE: {unreached}/{len(jobs)} configurations never reach their requested nu_max in these crops. "
              "They will have hollow markers and CSV flags; do not interpret these as steady-state measurements.")
    if BASELINE[0] not in args.nu_min_values or BASELINE[1] not in args.t_inc_values:
        print("NOTE: baseline (-6,128) is absent; only absolute-rate plots will be produced.")
    max_job = max(info["estimated_ram_gib"] for info in inputs.values())
    if args.dry_run:
        sw.plan(args, len(jobs), max_job, cubes)
        print("Dry run complete: no encoding and no output files changed.")
        return
    rows = []
    if not args.overwrite and (csv_path.exists() or meta_path.exists()):
        if not meta_path.exists():
            raise ValueError("Results exist without metadata; choose a new output folder.")
        old = json.loads(meta_path.read_text(encoding="utf-8"))
        if old.get("study") != study:
            raise ValueError("Saved inputs, grid, fixed settings, code or model differ. Use a new output folder.")
        rows = core.read_csv(csv_path)
    pending = sw.pending_keys(jobs, rows, job_key)
    workers = sw.plan(args, len(pending), max_job, cubes)
    metadata["execution"] = {"requested_workers": args.workers, "active_workers": workers,
                             "completed_at_start": len(rows), "total_memory_gib": args.total_memory_gb}
    out.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        clear_outputs(out)
    core.write_json(meta_path, metadata)
    core.write_csv(csv_path, rows)
    print(f"Complete: {len(rows)}; remaining: {len(pending)}. Results: {out}")
    sw.collect(jobs, rows, pending, workers, cubes, repo, "vmax", _evaluate, (args, bases),
               get_key=job_key, save=lambda rs: core.write_csv(csv_path, rs),
               plot=lambda rs: plot_results(rs, metadata, out), out=out, grouped=False)
    print(f"Saved {len(rows)}/{len(jobs)} configurations. CSV and plots: {out}")
    print("Checks use encoder-side reconstruction, not an independent decoder.")


def main():
    args = parse_args()
    if args.dry_run:
        return _main(args)
    if args.plot_only and not args.output.is_dir():
        raise ValueError(f"No saved output directory: {args.output}")
    with sw.output_lock(args.output.resolve()):
        return _main(args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130)
    except (ValueError, OSError, ModuleNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
