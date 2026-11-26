#!/usr/bin/env python3
"""
nu_max sweep (weight-update adaptation rate) for HYPSO-1 images with the
CCSDS 123.0-B-2 model, at one or more absolute error limits A*.

Background (CCSDS 123.0-B-2 sec. 4.10, Green Book 120.2-G-2): the weights are
updated with a step size 2^-(rho(t)), where
    rho(t) = clip(nu_min + floor((t - NX) / t_inc), nu_min, nu_max) + D - Omega.
The exponent starts at nu_min (large steps: fast learning) and grows by one
every t_inc pixels until it reaches nu_max (small steps: fine, stable weights).
nu_max therefore sets how finely the weights settle in the steady state, which
covers almost the whole image.

Swept:      nu_max over --vmax (default -1..9; values below nu_min are skipped)
            x A* over --abs-error (default 0 2 4 8; 0 = lossless)
Fixed:      nu_min = -1, t_inc = 2^11 (change with --vmin / --tinc), the
            predictor configuration in --P / --local-sum / --mode (default:
            P=3, wide neighbor, full, the best configuration from the P sweep),
            and everything else at Vorhaug 2024, Table 3.1.
Relative error limit: off by default (absolute limit only). --rel-error R adds it.

Needs p_sweep.py and fast_engine.py in the same folder (it reuses their image
readers, header builder, engines and --verify check).

Examples (from the repository root)
    python tools/p_sweep/vmax_sweep.py --model-dir . --images data --verify --dry-run
    python tools/p_sweep/vmax_sweep.py --model-dir . --images data --verify --out results/vmax_sweep
    python tools/p_sweep/vmax_sweep.py ... --center-crop 256x256 --vmax 0 2 4 6 8 --abs-error 0 4
    python tools/p_sweep/vmax_sweep.py ... --abs-error 4 8 --rel-error 16 --tinc 256 --vmin -2
    python tools/p_sweep/vmax_sweep.py --plot-only --out results/vmax_sweep

Outputs in --out: results.csv (one row per run, appended as runs finish; rerun
the same command to resume), and for each crop / predictor / schedule setting:
    vmax_delta_*.png/.pdf  rate change (%) vs nu_max relative to --ref-vmax (default 4, Vorhaug)
    vmax_rate_*.png/.pdf   bits/sample vs nu_max
    vmax_psnr_*.png/.pdf   PSNR of the decoded image vs nu_max (lossy A* only)
    vmax_summary_*.md      best nu_max per image and A*
"""

import argparse
import csv
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p_sweep as ps  # noqa: E402

LS_ALIASES = {
    "wide-neighbor": "WIDE_NEIGHBOR_ORIENTED", "narrow-neighbor": "NARROW_NEIGHBOR_ORIENTED",
    "wide-column": "WIDE_COLUMN_ORIENTED", "narrow-column": "NARROW_COLUMN_ORIENTED",
}
FIELDS = [
    "image", "crop", "D", "P", "local_sum", "mode", "omega", "vmin", "tinc", "vmax", "A", "R",
    "ny", "nx", "nz", "header_bits", "body_bits", "total_bits", "bps", "cr_vs_16bit",
    "max_abs_err", "mse", "psnr_db", "snr_db", "seconds",
]
KEY = ["image", "crop", "D", "P", "local_sum", "mode", "omega", "vmin", "tinc", "vmax", "A", "R"]
# categorical colours (reference palette slots 1-4, then extra distinct hues) + markers
COLORS = ["#257de0", "#f47230", "#00b894", "#f2a600", "#8e44ad", "#d63031", "#6c5ce7", "#636e72"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
INK, INK2, GRID = "#191919", "#666666", "#e5e5e5"


def key_of(d):
    return tuple(str(d[k]) for k in KEY)


# ----------------------------------------------------------------------------
# Worker
# ----------------------------------------------------------------------------


def run_job(job):
    t0 = time.perf_counter()
    engine = job["engine"]
    cube, signed = ps.load_cube(job["path"], job["crop"], native=engine == "fast")
    D = job["D"]
    lo, hi = (-(2 ** (D - 1)), 2 ** (D - 1) - 1) if signed else (0, 2**D - 1)
    if cube.min() < lo or cube.max() > hi:
        raise ValueError(f"{job['image']}: sample values {cube.min()}..{cube.max()} do not fit D={D} bits")
    ov = {"nu_min": job["vmin"], "nu_max": job["vmax"], "t_inc": job["tinc"], "omega": job["omega"]}
    fid = (job["A"], job["R"])
    args = (signed, job["P"], job["local_sum"], job["mode"], fid, D)

    if job.get("verify"):
        hb, b_eng, s_eng, _ = ps.compress_bits(cube, *args, engine=engine, overrides=ov)
        _, b_ref, s_ref, _ = ps.compress_bits(cube.astype(np.int64), *args, lean=False, engine="model", overrides=ov)
        if engine == "fast":
            import fast_engine
            s_eng = fast_engine.bits_to01(s_eng, b_eng)
        else:
            s_eng = s_eng.to01()
        return {**job, "identical": b_eng == b_ref and s_eng == s_ref.to01()}

    hb, body, _, st = ps.compress_bits(cube, *args, engine=engine, overrides=ov)
    ny, nx, nz = cube.shape
    n = ny * nx * nz
    word = 8 * ps.VORHAUG["word_bytes"]
    total = hb + body
    total += (word - total % word) % word
    return {
        **{k: job[k] for k in KEY if k != "crop"}, "crop": job["crop_label"],
        "ny": ny, "nx": nx, "nz": nz, "header_bits": hb, "body_bits": body, "total_bits": total,
        "bps": total / n, "cr_vs_16bit": 16 * n / total,
        **ps.distortion_fields(st, job["psnr_bits"]), "seconds": round(time.perf_counter() - t0, 1),
    }


# ----------------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------------


def make_plots(rows, out_dir, ref_vmax):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    groups = {}
    for r in rows:
        g = (r["crop"], r["D"], r["P"], r["local_sum"], r["mode"], r["omega"], r["vmin"], r["tinc"], r["R"])
        groups.setdefault(g, []).append(r)

    for (crop, D, P, ls, mode, om, vmin, tinc, R), rs in groups.items():
        images = sorted({r["image"] for r in rs})
        As = sorted({int(r["A"]) for r in rs})
        vm = sorted({int(r["vmax"]) for r in rs})
        bps = {(r["image"], int(r["A"]), int(r["vmax"])): float(r["bps"]) for r in rs}

        def num(v):
            try:
                x = float(v)
                return x if np.isfinite(x) else None
            except (TypeError, ValueError):
                return None
        psnr = {(r["image"], int(r["A"]), int(r["vmax"])): num(r["psnr_db"]) for r in rs}
        ref = ref_vmax if ref_vmax in vm else vm[len(vm) // 2]
        rel_txt = "no relative limit" if int(R) == 0 else f"R*={R}"
        setting = (f"P={P}, {ps.SHORT[ls]}, {ps.SHORT[mode]}, Ω={om}, ν_min={vmin}, t_inc={tinc}, "
                   f"D={D}, {rel_txt}, crop: {crop}")
        tag = f"{crop}_P{P}_{ls.split('_')[0].lower()}{ls.split('_')[1][0].lower()}_{mode.lower()}_om{om}_vmin{vmin}_tinc{tinc}_D{D}_R{R}"
        a_label = {a: ("lossless (A*=0)" if a == 0 and int(R) == 0 else f"A*={a}") for a in As}
        legend = [Line2D([], [], color=COLORS[i % len(COLORS)], marker=MARKERS[i % len(MARKERS)], markersize=5,
                         linewidth=1.8, label=a_label[a]) for i, a in enumerate(As)]
        legend.append(Line2D([], [], color=INK, linestyle=":", linewidth=1, label=f"ν_max = {ref} (reference)"))

        def series(table, panel, a, fn=None):
            ys = []
            for v in vm:
                if panel == "Mean over images":
                    vals = [fn(im, a, v) if fn else table.get((im, a, v)) for im in images]
                    ys.append(np.mean(vals) if all(x is not None for x in vals) else np.nan)
                else:
                    x = fn(panel, a, v) if fn else table.get((panel, a, v))
                    ys.append(np.nan if x is None else x)
            return ys

        def delta(im, a, v):
            b, r0 = bps.get((im, a, v)), bps.get((im, a, ref))
            return None if b is None or r0 is None else 100 * (b / r0 - 1)

        def figure(title, ylabel, fname, table=None, fn=None, a_list=As, zero_line=False):
            panels = images + (["Mean over images"] if len(images) > 1 else [])
            ncol = min(3, len(panels))
            nrow = math.ceil(len(panels) / ncol)
            top_in, leg_in = 0.5, 0.55
            fig_h = 3.2 * nrow + top_in + leg_in
            fig, axes = plt.subplots(nrow, ncol, figsize=(4.3 * ncol, fig_h), squeeze=False)
            for ax, panel in zip(axes.flat, panels):
                for i, a in enumerate(As):
                    if a not in a_list:
                        continue
                    ax.plot(vm, series(table, panel, a, fn), color=COLORS[i % len(COLORS)],
                            marker=MARKERS[i % len(MARKERS)], markersize=4, linewidth=1.6)
                ax.axvline(ref, color=INK, linestyle=":", linewidth=1)
                if zero_line:
                    ax.axhline(0, color=INK2, linewidth=0.8)
                ax.set_title(panel, fontsize=10, color=INK, loc="left")
                ax.set_xlabel("ν_max", fontsize=9, color=INK2)
                ax.set_ylabel(ylabel, fontsize=9, color=INK2)
                ax.set_xticks(vm)
                ax.grid(True, color=GRID, linewidth=0.8)
                ax.set_axisbelow(True)
                for sp in ("top", "right"):
                    ax.spines[sp].set_visible(False)
                ax.tick_params(colors=INK2, labelsize=8)
            for ax in list(axes.flat)[len(panels):]:
                ax.axis("off")
            fig.suptitle(f"{title}\n{setting}", x=0.01, ha="left", y=1 - 0.08 / fig_h, va="top", fontsize=10, color=INK)
            items = [it for it, a in zip(legend, As + [None]) if a is None or a in a_list]
            fig.legend(handles=items, loc="lower center", ncol=min(len(items), 5), fontsize=8, frameon=False)
            fig.tight_layout(rect=(0, leg_in / fig_h, 1, 1 - top_in / fig_h))
            for ext in ("png", "pdf"):
                fig.savefig(out_dir / f"{fname}_{tag}.{ext}", dpi=200)
            plt.close(fig)

        figure(f"Rate change vs ν_max, relative to ν_max = {ref} for each A* (lower is better)",
               f"Rate change vs ν_max={ref} (%)", "vmax_delta", fn=delta, zero_line=True)
        figure("Compressed rate vs ν_max", "Rate (bits/sample)", "vmax_rate", table=bps)
        lossy = [a for a in As if not (a == 0 and int(R) == 0) and any(psnr.get((im, a, v)) for im in images for v in vm)]
        if lossy:
            figure("Decoded image PSNR vs ν_max (higher is better)", "PSNR (dB)", "vmax_psnr", table=psnr, a_list=lossy)

        # summary
        lines = [f"# ν_max sweep summary", "", setting, "",
                 f"| Image | A* | Best ν_max | Best bps | bps at ν_max={ref} | Change | PSNR at best (dB) |",
                 "|---|---|---|---|---|---|---|"]
        for a in As:
            for im in images + ["Mean over images"]:
                if im == "Mean over images":
                    cand = {v: np.mean([bps[(i, a, v)] for i in images]) for v in vm if all((i, a, v) in bps for i in images)}
                else:
                    cand = {v: bps[(im, a, v)] for v in vm if (im, a, v) in bps}
                if not cand or ref not in cand:
                    continue
                vb = min(cand, key=cand.get)
                p = psnr.get((im, a, vb)) if im != "Mean over images" else None
                lines.append(f"| {im} | {a} | {vb} | {cand[vb]:.4f} | {cand[ref]:.4f} | {100 * (cand[vb] / cand[ref] - 1):+.2f}% | "
                             f"{'-' if p is None else f'{p:.2f}'} |")
        (out_dir / f"vmax_summary_{tag}.md").write_text("\n".join(lines) + "\n")
        print("\n".join(lines))


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------


def read_rows(path):
    if not path.exists() or path.stat().st_size == 0:
        return []
    with open(path, newline="") as f:
        rd = csv.DictReader(f)
        if any(k not in (rd.fieldnames or []) for k in KEY + ["bps"]):
            raise SystemExit(f"{path} was not written by this script. Use another --out folder. Nothing was changed.")
        return list(rd)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", help="folder that contains ccsds123_i2_hlm (your repository root: .)")
    ap.add_argument("--images", nargs="+", help=".nc files, raw files, or folders")
    ap.add_argument("--out", default="vmax_sweep_results")
    ap.add_argument("--crop", help="Y0:Y1,X0:X1 (lines, samples)")
    ap.add_argument("--center-crop", help="HxW, e.g. 256x256")
    ap.add_argument("--bands", help="Z0:Z1")
    ap.add_argument("--vmax", type=int, nargs="+", default=list(range(-1, 10)), help="nu_max sweep points (-6..9). Default -1..9")
    ap.add_argument("--abs-error", type=int, nargs="+", default=[0, 2, 4, 8], help="A* values. 0 = lossless (unless --rel-error). Default 0 2 4 8")
    ap.add_argument("--rel-error", type=int, default=0, help="relative error limit R* (default 0 = not used)")
    ap.add_argument("--vmin", type=int, default=-1, help="nu_min (-6..9). Default -1")
    ap.add_argument("--tinc", type=int, default=2**11, help="t_inc: power of two 16..2048. Default 2048 (2^11)")
    ap.add_argument("--omega", type=int, default=ps.VORHAUG["omega"], help="weight resolution Omega (4..19). Default 19")
    ap.add_argument("--P", type=int, default=3, help="prediction bands (default 3)")
    ap.add_argument("--local-sum", default="wide-neighbor", choices=list(LS_ALIASES), help="default wide-neighbor")
    ap.add_argument("--mode", default="full", choices=["full", "reduced"], help="default full")
    ap.add_argument("--dynamic-range", type=int, default=ps.VORHAUG["D"], help="D (default 16, as Vorhaug)")
    ap.add_argument("--psnr-bits", type=int, default=12, help="sensor bit depth for PSNR peak (HYPSO-1: 12)")
    ap.add_argument("--ref-vmax", type=int, default=ps.VORHAUG["nu_max"], help="reference nu_max for the % plot (default 4, Vorhaug)")
    ap.add_argument("--nc-var", help="NetCDF variable (default products/dn)")
    ap.add_argument("--nc-axes", help="stored axis order of the NetCDF variable, e.g. yxz")
    ap.add_argument("--engine", choices=["auto", "fast", "lean"], default="auto")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--verify", action="store_true", help="check the engine against the unmodified model before sweeping")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "results.csv"
    if args.plot_only:
        make_plots(read_rows(csv_path), out, args.ref_vmax)
        return
    if not args.model_dir or not args.images:
        ap.error("--model-dir and --images are required (unless --plot-only)")

    # parameter checks (ranges from CCSDS 123.0-B-2)
    if not -6 <= args.vmin <= 9:
        ap.error("--vmin must be in -6..9")
    if args.tinc not in [2**k for k in range(4, 12)]:
        ap.error("--tinc must be a power of two from 16 to 2048")
    if not 4 <= args.omega <= 19:
        ap.error("--omega must be in 4..19")
    bad = [v for v in args.vmax if not -6 <= v <= 9]
    if bad:
        ap.error(f"--vmax values must be in -6..9: {bad}")
    vmaxes = sorted({v for v in args.vmax if v >= args.vmin})
    skipped = sorted({v for v in args.vmax if v < args.vmin})
    if any(a < 0 for a in args.abs_error) or args.rel_error < 0:
        ap.error("error limits must be >= 0")
    if args.engine == "auto":
        try:
            import numba  # noqa: F401
            args.engine = "fast"
        except ImportError:
            args.engine = "lean"
            print("numba not installed: using the model's Python code (slow). pip install numba")
    ps.NC_OPTS.update(var=args.nc_var, axes=args.nc_axes)
    ls = LS_ALIASES[args.local_sum]
    mode = args.mode.upper()

    images = ps.collect_images(args.images)
    jobs = []
    for path in images:
        name, _, nz, ny, nx = ps.parse_image_name(path)
        c = ps.resolve_crop(args, ny, nx, nz)
        for a, v in product(sorted(set(args.abs_error)), vmaxes):
            jobs.append({"path": str(path), "image": name, "crop": c, "crop_label": ps.crop_label(c, ny, nx, nz),
                         "samples": (c[1] - c[0]) * (c[3] - c[2]) * (c[5] - c[4]), "D": args.dynamic_range,
                         "P": args.P, "local_sum": ls, "mode": mode, "omega": args.omega, "vmin": args.vmin,
                         "tinc": args.tinc, "vmax": v, "A": a, "R": args.rel_error, "engine": args.engine,
                         "psnr_bits": args.psnr_bits})
    done = {key_of(r) for r in read_rows(csv_path)}
    todo = [j for j in jobs if key_of({**j, "crop": j["crop_label"]}) not in done]
    todo.sort(key=lambda j: j["samples"], reverse=True)

    workers = args.workers or os.cpu_count() or 1
    try:
        import psutil
        per = (todo[0]["samples"] * (4 if args.engine == "fast" else 18) / 2**20 + 400) if todo else 1
        workers = min(workers, max(1, int(0.8 * psutil.virtual_memory().available / 2**20 // per)))
    except ImportError:
        pass
    if os.name == "nt":
        workers = min(workers, 61)
    workers = max(1, min(workers, len(todo) or 1))

    rel = "not used" if args.rel_error == 0 else f"R*={args.rel_error}"
    print(f"nu_max: {vmaxes}" + (f"  (skipped below nu_min: {skipped})" if skipped else ""))
    print(f"A*: {sorted(set(args.abs_error))}   relative limit: {rel}   nu_min={args.vmin}  t_inc={args.tinc}  Omega={args.omega}  D={args.dynamic_range}")
    print(f"Predictor: P={args.P}, {ps.SHORT[ls]}, {args.mode}   engine: {args.engine}   workers: {workers}")
    print(f"Images: {len(images)}   runs per image: {len(vmaxes) * len(set(args.abs_error))}   total: {len(jobs)}   "
          f"already done: {len(jobs) - len(todo)}")
    for path in images:
        j = next(j for j in jobs if j["path"] == str(path))
        print(f"  {j['image']}: crop {j['crop_label']}, {j['samples'] / 1e6:.2f} M samples")
    if args.dry_run:
        return

    with ProcessPoolExecutor(max_workers=workers, initializer=ps._init_worker,
                             initargs=(args.model_dir, dict(ps.NC_OPTS))) as pool:
        if args.verify:
            # 16x16 crops; t_inc=16 so that the nu ramp (nu_min -> nu_max) happens inside the crop
            vj = []
            for path in images:
                name, _, nz, ny, nx = ps.parse_image_name(path)
                _, _, _, _, z0, z1 = ps.resolve_crop(args, ny, nx, nz)
                h, w = min(16, ny), min(16, nx)
                y0, x0 = (ny - h) // 2, (nx - w) // 2
                for a, v in product(sorted(set(args.abs_error)), vmaxes):
                    vj.append({"path": str(path), "image": name, "crop": (y0, y0 + h, x0, x0 + w, z0, z1), "D": args.dynamic_range,
                               "P": args.P, "local_sum": ls, "mode": mode, "omega": args.omega, "vmin": args.vmin,
                               "tinc": 16, "vmax": v, "A": a, "R": args.rel_error, "engine": args.engine, "verify": True})
            print(f"\nVerifying the {args.engine} engine against the unmodified model: {len(vj)} runs on 16x16 crops (t_inc=16) ...")
            badv = [r for r in pool.map(run_job, vj) if not r["identical"]]
            if badv:
                for r in badv:
                    print(f"  MISMATCH: {r['image']} A*={r['A']} nu_max={r['vmax']}")
                sys.exit("Verification failed - not running the sweep.")
            print("  all bitstreams identical to the reference model.")

        if todo:
            print(f"\nRunning {len(todo)} jobs ...")
            new = not csv_path.exists() or csv_path.stat().st_size == 0
            with open(csv_path, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=FIELDS)
                if new:
                    w.writeheader()
                t0 = time.time()
                futs = {pool.submit(run_job, j): j for j in todo}
                for i, fut in enumerate(as_completed(futs), 1):
                    j = futs[fut]
                    try:
                        r = fut.result()
                    except Exception as exc:
                        print(f"  [{i}/{len(todo)}] FAILED {j['image']} A*={j['A']} nu_max={j['vmax']}: {exc}")
                        continue
                    w.writerow(r)
                    f.flush()
                    el = time.time() - t0
                    print(f"  [{i}/{len(todo)}] {r['image']:<30} A*={r['A']:<3} nu_max={r['vmax']:<3} {r['bps']:.4f} bps  "
                          f"PSNR {r['psnr_db'] or '-'}  {r['seconds']:>6.1f} s   elapsed {el / 60:.1f} min, "
                          f"ETA {el / i * (len(todo) - i) / 60:.1f} min")

    make_plots(read_rows(csv_path), out, args.ref_vmax)
    print(f"\nResults: {csv_path}\nPlots and summary: {out}")


if __name__ == "__main__":
    main()
