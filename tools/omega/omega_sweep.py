#!/usr/bin/env python3
"""HYPSO weight-resolution study, following Green Book Figures 4-17 and 4-18.

Examples, from the verification-model repository root:
  python tools/omega/omega_sweep.py --figure 417 --crop 256 256 --workers 2
  python tools/omega/omega_sweep.py --figure 418 --a-values 0 1 3 7 15 --crop 256 256 --workers 2
  python tools/omega/omega_sweep.py --figure both --full-image --a-values 0 3 15 --workers 2

One complete configuration per process. All native bands, integer DN, no added
noise. Omega=19 is always encoded as a separate matched reference for each
image and A. These are HYPSO adaptations, not the Green Book's original corpus.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import sys

import numpy as np
import _damping_core as core
import _omega_engine as engine
import sweep_workers as sw

VERSION = "1.0.0-omega"
REFERENCE_OMEGA = 19


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--figure", choices=("417", "418", "both"), default="417")
    p.add_argument("--input-dir", type=Path, default=Path("data"))
    select = p.add_mutually_exclusive_group()
    select.add_argument("--images", nargs="+", choices=list(core.SCENES), help="Default: all five study scenes.")
    select.add_argument("--files", type=Path, nargs="+", help="Explicit NetCDF files, e.g. Adelaide or test files.")
    size = p.add_mutually_exclusive_group()
    size.add_argument("--crop", type=int, nargs=2, metavar=("HEIGHT", "WIDTH"), default=[256, 256])
    size.add_argument("--full-image", action="store_true", help="Use each input's complete spatial dimensions.")
    p.add_argument("--anchor", choices=("center", "top-left"), default="center")
    p.add_argument("--row", type=int, help="Explicit zero-based crop row; requires --col.")
    p.add_argument("--col", type=int, help="Explicit zero-based crop column; requires --row.")
    p.add_argument("--omega-min", type=int, help="Inclusive range start, default 4.")
    p.add_argument("--omega-max", type=int, help="Inclusive range end, default 19.")
    p.add_argument("--omega-values", type=int, nargs="+", help="Selected resolutions instead of a range, e.g. 8 11 14 19.")
    p.add_argument("--a-values", type=int, nargs="+", help="Default: 0 for 417; 0 1 3 7 15 for 418/both. Add 31 for the book's full list.")
    p.add_argument("--nu-min", "--vmin", dest="nu_min", type=int, default=-1)
    p.add_argument("--nu-max", "--vmax", dest="nu_max", type=int, default=3)
    p.add_argument("--t-inc", type=int, default=2048, help="Literal power-of-two interval 16..2048; HYPSO default 2048.")
    p.add_argument("--phi", type=int, default=3, help="Fixed damping for every Omega and A; default 3.")
    p.add_argument("--psi", type=int, default=7, help="Fixed offset for A>0; zero for lossless.")
    p.add_argument("--p", type=int, default=3)
    p.add_argument("--prediction-mode", choices=("reduced", "full"), default="reduced")
    p.add_argument("--local-sum", choices=list(core.LOCAL_SUM_OPTIONS), default="narrow-neighbor")
    p.add_argument("--variable", default="products/dn")
    p.add_argument("--axis-order", choices=("yxz", "yzx", "xyz", "xzy", "zxy", "zyx"))
    storage = p.add_mutually_exclusive_group()
    storage.add_argument("--low-memory", dest="low_memory", action="store_true")
    storage.add_argument("--standard-memory", dest="low_memory", action="store_false")
    p.set_defaults(low_memory=True)
    p.add_argument("--workers", type=int, default=2, help="Maximum concurrent complete configurations, subject to CPU/RAM.")
    p.add_argument("--max-memory-gb", type=float, default=90, help="Estimated per-job RAM ceiling in GiB.")
    p.add_argument("--total-memory-gb", type=float, default=90, help="Aggregate planning budget in GiB; not an OS hard limit.")
    p.add_argument("--memory-poll-ms", type=int, default=50)
    p.add_argument("--repo", type=Path)
    p.add_argument("--output", type=Path, default=Path("results/omega_study"))
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--plot-only", action="store_true", help="Recreate selected figures from saved jobs without encoding.")
    args = p.parse_args(argv)
    if args.omega_values is not None:
        if args.omega_min is not None or args.omega_max is not None:
            p.error("Use --omega-values OR --omega-min/--omega-max.")
        args.omega_requested = sorted(set(args.omega_values))
    else:
        first = 4 if args.omega_min is None else args.omega_min
        last = 19 if args.omega_max is None else args.omega_max
        if first > last:
            p.error("--omega-min must be <= --omega-max.")
        args.omega_requested = list(range(first, last + 1))
    if any(not 4 <= w <= 19 for w in args.omega_requested):
        p.error("Weight resolution Omega must be 4..19 inclusive.")
    args.omega_values = sorted(set(args.omega_requested + [REFERENCE_OMEGA]))
    defaults = [0] if args.figure == "417" else [0, 1, 3, 7, 15]
    args.a_values = sorted(set(defaults if args.a_values is None else args.a_values))
    if any(not 0 <= a <= 32767 for a in args.a_values):
        p.error("A must be 0..32767 DN.")
    if not args.plot_only and args.figure in ("417", "both") and 0 not in args.a_values:
        p.error("Figure 4-17 requires A=0; include 0 in --a-values or select --figure 418.")
    if not -6 <= args.nu_min <= args.nu_max <= 9:
        p.error("Require -6 <= nu_min <= nu_max <= 9.")
    if not 0 <= args.phi <= 7 or not 0 <= args.psi <= 7 or not 0 <= args.p <= 15:
        p.error("Require phi/psi in 0..7 and P in 0..15.")
    if args.t_inc < 16 or args.t_inc > 2048 or args.t_inc & (args.t_inc - 1):
        p.error("t_inc must be a power of two from 16 to 2048.")
    if any(not 2 <= n <= 65535 for n in args.crop):
        p.error("Crop dimensions must be 2..65535.")
    if (args.row is None) != (args.col is None):
        p.error("Specify --row and --col together.")
    if args.row is not None and (args.full_image or min(args.row, args.col) < 0):
        p.error("Explicit origins must be nonnegative and cannot be used with --full-image.")
    if args.workers < 1 or not 10 <= args.memory_poll_ms <= 1000:
        p.error("Workers must be positive and memory-poll-ms must be 10..1000.")
    if any(not math.isfinite(v) or v <= 0 for v in (args.max_memory_gb, args.total_memory_gb)):
        p.error("Memory budgets must be positive and finite.")
    args.images = list(dict.fromkeys(args.images or list(core.SCENES)))
    return args


def build_jobs(args, inputs):
    jobs = []
    # References first, so partial relative plots become useful promptly.
    order = [REFERENCE_OMEGA] + [w for w in args.omega_values if w != REFERENCE_OMEGA]
    for scene, info in inputs.items():
        sy, sx, z = info["source_shape_yxz"]
        y, x = (sy, sx) if args.full_image else args.crop
        if args.full_image:
            row, col = 0, 0
        elif args.row is not None:
            row, col = args.row, args.col
        elif args.anchor == "top-left":
            row, col = 0, 0
        else:
            row, col = (sy-y)//2, (sx-x)//2
        issue = "" if min(row, col) >= 0 and row+y <= sy and col+x <= sx else f"Crop ({row},{col},{y},{x}) does not fit source {sy}x{sx}."
        for A in args.a_values:
            for omega in order:
                identity = [scene, y, x, row, col, A, omega, args.phi, args.nu_min, args.nu_max, args.t_inc]
                job_id = hashlib.sha256(json.dumps(identity).encode()).hexdigest()[:16]
                jobs.append(dict(job_id=job_id, image=scene, size=f"{y}x{x}", repeat=1,
                                 is_full_image=int(row == col == 0 and (y, x) == (sy, sx)),
                                 rows=y, columns=x, bands=z, crop_row=row, crop_col=col,
                                 omega=omega, t_inc=args.t_inc, nu_min=args.nu_min, phi=args.phi, a=A,
                                 estimated_ram_gib=engine.estimate_gib((y, x, z), args.p, args.low_memory),
                                 dimension_issue=issue))
    return jobs


def comparison_rows(records):
    """Use the matching image/A reference, never a global minimum or another A."""
    refs = {(r["row"]["image"], int(r["row"]["a"])): r["row"] for r in records
            if int(r["row"]["omega"]) == REFERENCE_OMEGA}
    comparisons = []
    for record in records:
        r = record["row"]
        ref = refs.get((r["image"], int(r["a"])))
        if ref and any(r[k] != ref[k] for k in ("crop_sha256", "sample_count", "p", "local_sum", "prediction_mode",
                                                "nu_min", "nu_max", "t_inc", "phi", "psi", "encoder")):
            raise ValueError("Reference and test configuration differ in more than Omega.")
        rate = float(r["bits_per_sample"])
        baseline = float(ref["bits_per_sample"]) if ref else None
        if baseline is not None and baseline <= 0:
            raise ValueError("Reference rate must be positive.")
        comparisons.append(dict(image=r["image"], a=int(r["a"]), omega=int(r["omega"]),
                                rows=r["rows"], columns=r["columns"], bands=r["bands"],
                                bits_per_sample=rate, reference_omega=REFERENCE_OMEGA,
                                reference_bits_per_sample=baseline if baseline is not None else "",
                                rate_increase_percent=100*(rate/baseline-1) if baseline is not None else "",
                                reference_available=int(ref is not None), job_id=r["job_id"]))
    return comparisons


def plot_results(records, metadata, out, figure="417"):
    study = metadata["study"]
    if figure in ("417", "both") and 0 not in study["a_values"]:
        raise ValueError("Saved study has no lossless A=0 grid for Figure 4-17.")
    rows = comparison_rows(records)
    fields = ["image", "a", "omega", "rows", "columns", "bands", "bits_per_sample", "reference_omega",
              "reference_bits_per_sample", "rate_increase_percent", "reference_available", "job_id"]
    engine.table(out/"rate_comparison.csv", fields, rows)
    if not rows:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    values = {(r["image"], r["a"], r["omega"]): r for r in rows}
    images = list(study["inputs"])
    omegas = study["omega_values"]
    avals = study["a_values"]
    profile = study["profile"]
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#5B4BCE", "#333333"]
    markers = ["o", "s", "D", "^", "v", "P", "X"]
    jobs = {s: next(j for j in study["jobs"] if j["image"] == s) for s in images}
    sizes = {(j["rows"], j["columns"], j["bands"]) for j in jobs.values()}
    size_label = ("full images (native dimensions)" if study["full_image"] else
                  f"{next(iter(sizes))[0]} x {next(iter(sizes))[1]} crops")
    config = (f"P={profile['p']} | {profile['local_sum']}, {profile['prediction_mode']} | hybrid | "
              rf"$\nu_{{min}}={profile['nu_min']},\ \nu_{{max}}={profile['nu_max']},\ t_{{inc}}={profile['t_inc']},\ \phi={profile['phi']}$")
    note = "Rate includes header, entropy payload/tail and word padding; external initialization excluded. Reference: same image/A at Omega=19."

    def curve(scene, A, field):
        return [float(values[(scene, A, w)][field]) if (scene, A, w) in values and values[(scene, A, w)][field] != ""
                else np.nan for w in omegas]

    def decorate(ax, ylabel):
        ax.set_xlabel(r"Weight resolution, $\Omega$")
        ax.set_ylabel(ylabel)
        ticks = [w for w in omegas if w % 2 == 0 or w == 19] if len(omegas) > 10 else omegas
        ax.set_xticks(ticks)
        ax.set_xlim(min(omegas)-.3, max(omegas)+.3)
        ax.ticklabel_format(axis="y", useOffset=False)
        ax.grid(alpha=.25)
        ax.set_axisbelow(True)

    def save(fig, basename, handles):
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .045),
                   ncol=min(6, len(handles)), frameon=False)
        fig.text(.5, .012, note, ha="center", fontsize=7.5)
        fig.tight_layout(rect=(0, .13, 1, .97), h_pad=2.4, w_pad=2.2)
        for ext in ("png", "pdf"):
            fig.savefig(out/f"{basename}.{ext}", dpi=170, bbox_inches="tight")
        plt.close(fig)

    if figure in ("417", "both"):
        fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8))
        handles = []
        for i, scene in enumerate(images):
            color, marker = colors[i % len(colors)], markers[i % len(markers)]
            label = study["inputs"][scene]["label"]
            for ax, field in zip(axes, ("bits_per_sample", "rate_increase_percent")):
                ax.plot(omegas, curve(scene, 0, field), color=color, marker=marker, markersize=4, linewidth=1.5)
            handles.append(Line2D([], [], color=color, marker=marker, label=label))
        decorate(axes[0], "Compressed bits/sample")
        decorate(axes[1], "Rate increase (%)")
        axes[1].axhline(0, color="grey", linewidth=.6)
        done = sum(r["a"] == 0 for r in rows)
        fig.suptitle(f"HYPSO weight-resolution study | Figure 4-17 style | A=0 | {done}/{len(images)*len(omegas)} points\n"
                     +config+f"\n{size_label}; all native bands", fontsize=12)
        save(fig, "fig417_lossless", handles)

    if figure in ("418", "both"):
        ncols = min(3, len(images))
        nrows = math.ceil(len(images)/ncols)
        fig, axes = plt.subplots(nrows, ncols, squeeze=False, figsize=(5*ncols, 3.5*nrows+1.5))
        for ax, scene in zip(axes.flat, images):
            job = jobs[scene]
            for i, A in enumerate(avals):
                ax.plot(omegas, curve(scene, A, "rate_increase_percent"), color=colors[i % len(colors)],
                        marker=markers[i % len(markers)], markersize=4, linewidth=1.4)
            count = sum(r["image"] == scene and r["reference_available"] for r in rows)
            expected = len(avals)*len(omegas)
            partial = f" | {count}/{expected} points" if count != expected else ""
            ax.set_title(f"{study['inputs'][scene]['label']}{partial}\n{job['rows']} x {job['columns']} pixels, {job['bands']} bands", fontsize=10.5)
            decorate(ax, "Rate increase (%)")
            ax.axhline(0, color="grey", linewidth=.6)
            if count == 0:
                ax.text(.5, .5, "Waiting for Omega=19 references", ha="center", transform=ax.transAxes)
        for ax in list(axes.flat)[len(images):]:
            ax.set_visible(False)
        handles = [Line2D([], [], color=colors[i % len(colors)], marker=markers[i % len(markers)], label=f"A={A} DN")
                   for i, A in enumerate(avals)]
        fig.suptitle(f"HYPSO weight-resolution study | Figure 4-18 style | {len(rows)}/{len(study['jobs'])} points\n"+config,
                     fontsize=12)
        save(fig, "fig418_error_limits", handles)


def make_metadata(args, repo, inputs, jobs):
    own = Path(__file__).resolve().parent
    profile = {k: getattr(args, k) for k in ("p", "nu_min", "nu_max", "t_inc", "phi", "psi", "local_sum", "prediction_mode")}
    profile.update(D=16, R=64, Theta=3, encoder="HYBRID", psi_lossless=0,
                   weight_initialization="DEFAULT", weight_exponent_offsets="ALL_ZERO")
    model_files = list((repo/"ccsds123_i2_hlm").rglob("*.py")) + list((repo/"ccsds123_i2_hlm").rglob("*.txt"))
    study = dict(tool="omega_sweep", version=VERSION, profile=profile, inputs=inputs, jobs=jobs,
                 omega_values=args.omega_values, omega_requested=args.omega_requested, reference_omega=REFERENCE_OMEGA,
                 a_values=args.a_values, full_image=args.full_image, low_memory=args.low_memory,
                 model_sha256=core.file_digest(model_files, repo),
                 bundle_sha256=core.file_digest(list(own.glob("*.py")), own),
                 python=platform.python_version(), numpy=np.__version__,
                 reference="CCSDS 120.2-G-2 section 4.2.4, Figures 4-17/4-18; HYPSO inputs and explicitly selected fixed settings.",
                 metric="100*(rate(Omega)/rate(19)-1) per image/A; rate includes header/payload/tail/word padding, excludes external initialization.",
                 execution="One complete configuration per worker; original serial predictor within each worker.",
                 checks="Encoder-side reconstruction and applied adaptation schedule; no independent decoder.")
    return dict(study_id=hashlib.sha256(json.dumps(study, sort_keys=True).encode()).hexdigest(),
                study=study, model_path=str(repo))


def main():
    args = parse_args()
    out = args.output.resolve()
    if args.plot_only:
        if not out.is_dir():
            raise ValueError("Output directory does not exist.")
        with sw.output_lock(out):
            meta = json.loads((out/"metadata.json").read_text())
            if meta.get("study", {}).get("tool") != "omega_sweep":
                raise ValueError("Not an omega study output directory.")
            plot_results(engine.regenerate(out, meta), meta, out, args.figure)
        print(f"Plots and rate comparison: {out}")
        return 0
    if not __debug__:
        raise ValueError("Do not use python -O; model assertions must remain enabled.")
    repo = engine.find_repo(args.repo)
    sw.load_model(repo, "vmax")
    if args.low_memory:
        from hybrid_no_traces import validate_reference
        validate_reference()
    paths = ([(p.stem, p.resolve()) for p in args.files] if args.files else
             [(s, core.find_capture(args.input_dir.resolve(), core.SCENES[s])) for s in args.images])
    inputs = {}
    for scene, path in paths:
        if scene in inputs:
            raise ValueError("Input names collide; use distinct filenames.")
        info = engine.inspect_input(path, args)
        info["label"] = scene.title() if scene in core.SCENES else scene.replace("_", " ")
        inputs[scene] = info
    jobs = build_jobs(args, inputs)
    metadata = make_metadata(args, repo, inputs, jobs)
    limit = min(engine.allowed_configuration_workers(args.workers), len(jobs))
    print(f"{len(inputs)} images x {len(args.a_values)} A values x {len(args.omega_values)} Omega values = {len(jobs)} configurations.")
    print(f"Omega={args.omega_values}; A={args.a_values}; reference Omega=19 per image/A.")
    print(f"Fixed P={args.p}, phi={args.phi}, nu_min={args.nu_min}, nu_max={args.nu_max}, t_inc={args.t_inc}; HYBRID.")
    print(f"One complete configuration per worker; CPU allowance {limit}; RAM can reduce concurrency.")
    if REFERENCE_OMEGA not in args.omega_requested:
        print("Omega=19 was added automatically to supply the matching reference rates.")
    for scene, info in inputs.items():
        j = next(j for j in jobs if j["image"] == scene)
        need = j["estimated_ram_gib"]
        budget = engine.execution_budget(args)
        slots = max(0, min(limit, math.floor((budget-.35)/need))) if need <= args.max_memory_gb else 0
        print(f"{info['label']}: source {info['source_shape_yxz']}; selected {j['rows']}x{j['columns']}x{j['bands']} "
              f"at row={j['crop_row']}, col={j['crop_col']}; estimate {need:.2f} GiB/config; up to {slots} fit now."+
              (f" SKIP: {j['dimension_issue']}" if j["dimension_issue"] else ""))
    if args.dry_run:
        print("Dry run: no compression or output writes.")
        return 0
    with sw.output_lock(out):
        meta_path = out/"metadata.json"
        if meta_path.exists():
            old = json.loads(meta_path.read_text())
            if old.get("study_id") != metadata["study_id"] or old.get("study") != metadata["study"]:
                raise ValueError("Inputs, grid, profile or source code changed. Use a new output directory.")
        elif (out/"jobs").exists() or (out/"results.csv").exists():
            raise ValueError("Results exist without matching metadata; use a new output directory.")
        (out/"jobs").mkdir(exist_ok=True)
        core.write_json(meta_path, metadata)
        engine.table(out/"plan.csv", list(jobs[0]), jobs)
        return engine.run_jobs(args, metadata, repo, out,
                               lambda rows, meta, dest: plot_results(rows, meta, dest, args.figure))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted. Completed configurations are saved; repeat the command to resume.")
        raise SystemExit(130)
    except (ValueError, OSError, ModuleNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
