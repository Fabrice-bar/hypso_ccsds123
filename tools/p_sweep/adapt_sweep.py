#!/usr/bin/env python3
"""
Initial adaptation studies (nu_min and t_inc) on HYPSO-1 images, following the
CCSDS Green Book 120.2-G-2 (Dec. 2022), sec. 4.2.2.

Background: the weights are updated with step 2^-(rho(t)),
    rho(t) = clip(nu_min + floor((t - NX) / t_inc), nu_min, nu_max) + D - Omega.
nu_min sets the step size at the start (low = large steps = fast learning),
t_inc sets how many pixels pass before the exponent rises by one, and nu_max
sets the final step size. nu_min and t_inc therefore only act during the
start-up ramp, which lasts (nu_max - nu_min) * t_inc pixels. The Green Book
notes that their effect shrinks as the number of samples per band grows.

Three studies (--study), each reproducing one Green Book figure on your images:

  vmin   Figure 4-10: rate against nu_min (default -6..2), one curve per damping
         value phi (default 0..5), one panel per image and A* (default 0, 3, 20).
  tinc   Figure 4-11: rate against log2(t_inc) (default 4..11), one curve per phi,
         one panel per image and A*.
  fig48  Figure 4-8: rate against nu_max, one curve per extreme (nu_min, t_inc)
         combination: nu_min in {-6, 0} x t_inc in {2^4, 2^7, 2^10}; lossless.

Book settings used as defaults:
  * nu_max = 7 - 2^(3-Theta) * phi   (the book: "a good choice in experimental
    results"; at Theta = 3 this is 7 - phi, so phi = 3 gives Vorhaug's nu_max = 4)
    Use --vmax N to hold nu_max fixed instead.
  * psi = 2^Theta - 1 (the book's suggested default; 0 when lossless), Theta = 3.
  * Figure 4-10 (vmin): t_inc = 2^6 (Annex C default; the figure gives no value).
  * Figure 4-11 (tinc): nu_min = -1 (Annex C default; "set as described above").
  * Figure 4-8: nu_max on the x axis, phi = 0, lossless.
  * A* = 0, 3, 20 with no relative limit, as in the figures.
HYPSO settings used as defaults (the book's predictor settings are per instrument):
  P = 3, wide neighbor, full prediction, Omega = 19, D = 16, hybrid coder, BIL.
Everything is configurable.

Image size matters here more than for any other parameter: on a full HYPSO
image (about 650,000 pixels per band) a ramp of e.g. 13 * 2^6 = 832 pixels is
0.1% of the image. Run full images for the real HYPSO answer, and optionally
a small crop (e.g. --center-crop 64x64) to see the "few samples per band"
regime the book describes.

Needs p_sweep.py and fast_engine.py in the same folder.

Examples (from the repository root)
    python tools/p_sweep/adapt_sweep.py --study vmin --model-dir . --images data --verify --dry-run
    python tools/p_sweep/adapt_sweep.py --study vmin --model-dir . --images data --verify --out results/adapt_vmin
    python tools/p_sweep/adapt_sweep.py --study tinc --model-dir . --images data --verify --out results/adapt_tinc
    python tools/p_sweep/adapt_sweep.py --study fig48 --model-dir . --images data --verify --out results/adapt_fig48
    python tools/p_sweep/adapt_sweep.py --study vmin ... --abs-error 0 4 16 --phi 0 3 5 --tinc 2048
    python tools/p_sweep/adapt_sweep.py --plot-only --out results/adapt_vmin

Outputs in --out:
    results.csv                      one row per run (rerun the same command to resume)
    <study>_rate_*.png/.pdf          bits/sample, rows = images, columns = A*
    <study>_delta_*.png/.pdf         % above the best point of each panel, with a mean row
    <study>_summary_*.md             best setting per image and A*, and the cost of the reference
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
KEY = ["study", "image", "crop", "A", "R", "theta", "phi", "psi", "vmin", "vmax", "tinc",
       "P", "local_sum", "mode", "omega", "D"]
FIELDS = KEY + ["ny", "nx", "nz", "header_bits", "body_bits", "total_bits", "bps",
                "max_abs_err", "mse", "psnr_db", "snr_db", "ramp_pixels", "ramp_pct", "seconds"]
COLORS = ["#257de0", "#f47230", "#00b894", "#f2a600", "#8e44ad", "#d63031", "#6c5ce7", "#636e72"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
INK, INK2, GRID = "#191919", "#666666", "#e5e5e5"


def key_of(d):
    return tuple(str(d[k]) for k in KEY)


def vmax_rule(phi, theta):
    """Green Book: nu_max = 7 - 2^(3 - Theta) * phi."""
    return int(round(7 - 2 ** (3 - theta) * phi))


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
    ov = {"nu_min": job["vmin"], "nu_max": job["vmax"], "t_inc": job["tinc"], "omega": job["omega"],
          "theta": job["theta"], "phi": job["phi"], "psi": job["psi"]}
    args = (signed, job["P"], job["local_sum"], job["mode"], (job["A"], job["R"]), D)

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
    ramp = min((job["vmax"] - job["vmin"]) * job["tinc"], max(ny * nx - nx, 0))  # pixels until nu_max
    return {
        **{k: job[k] for k in KEY if k != "crop"}, "crop": job["crop_label"],
        "ny": ny, "nx": nx, "nz": nz, "header_bits": hb, "body_bits": body, "total_bits": total,
        "bps": total / n, **ps.distortion_fields(st, job["psnr_bits"]),
        "ramp_pixels": ramp, "ramp_pct": round(100 * ramp / (ny * nx), 4),
        "seconds": round(time.perf_counter() - t0, 1),
    }


# ----------------------------------------------------------------------------
# Plots and summary
# ----------------------------------------------------------------------------

STUDY = {
    "vmin": {"x": "vmin", "curve": "phi", "xlabel": "ν_min", "fig": "Figure 4-10",
             "title": "Rate against ν_min for different damping ϕ (Green Book Fig. 4-10 style)"},
    "tinc": {"x": "tinc", "curve": "phi", "xlabel": "log₂(t_inc)", "fig": "Figure 4-11",
             "title": "Rate against t_inc for different damping ϕ (Green Book Fig. 4-11 style)"},
    "fig48": {"x": "vmax", "curve": "vmin_tinc", "xlabel": "ν_max", "fig": "Figure 4-8",
              "title": "Rate against ν_max for extreme (ν_min, t_inc) choices (Green Book Fig. 4-8 style)"},
}


def make_plots(rows, out_dir, refs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    for study in sorted({r["study"] for r in rows}):
        spec = STUDY[study]
        rs = [r for r in rows if r["study"] == study]
        # one figure set per fixed setting (everything except image, A, x and curve variables)
        varying = {"image", "A", "phi", "psi", "vmin", "vmax", "tinc"}
        groups = {}
        for r in rs:
            g = tuple((k, r[k]) for k in KEY if k not in varying and k != "study")
            groups.setdefault(g, []).append(r)
        for g, grs in groups.items():
            gd = dict(g)
            images = sorted({r["image"] for r in grs})
            As = sorted({int(r["A"]) for r in grs})

            def xval(r):
                return int(round(math.log2(int(r["tinc"])))) if spec["x"] == "tinc" else int(r[spec["x"]])

            def cval(r):
                return (int(r["vmin"]), int(r["tinc"])) if spec["curve"] == "vmin_tinc" else int(r["phi"])
            xs = sorted({xval(r) for r in grs})
            curves = sorted({cval(r) for r in grs})
            bps = {(r["image"], int(r["A"]), cval(r), xval(r)): float(r["bps"]) for r in grs}
            vmx = {(r["image"], int(r["A"]), cval(r)): int(r["vmax"]) for r in grs}

            def clabel(c):
                if spec["curve"] == "vmin_tinc":
                    return f"ν_min={c[0]}, t_inc=2^{int(math.log2(c[1]))}"
                vm = sorted({v for (im, a, cc), v in vmx.items() if cc == c})
                vm_txt = f" (ν_max={vm[0]})" if len(vm) == 1 and spec["x"] != "vmax" else ""
                return f"ϕ={c}{vm_txt}"

            def cstyle(i, c):
                if spec["curve"] == "vmin_tinc":
                    ts = sorted({cc[1] for cc in curves})
                    return dict(color=COLORS[ts.index(c[1]) % len(COLORS)], linestyle="-" if c[0] == min(cc[0] for cc in curves) else "--",
                                marker=MARKERS[ts.index(c[1]) % len(MARKERS)])
                return dict(color=COLORS[i % len(COLORS)], linestyle="-", marker=MARKERS[i % len(MARKERS)])

            fixed = []
            if study == "vmin":
                fixed.append("t_inc=" + ",".join(str(t) for t in sorted({int(r["tinc"]) for r in grs})))
            if study == "tinc":
                fixed.append(f"ν_min={','.join(sorted({r['vmin'] for r in grs}))}")
            if study != "fig48":
                vm_set = sorted({int(r["vmax"]) for r in grs})
                rule = all(int(r["vmax"]) == vmax_rule(int(r["phi"]), int(r["theta"])) for r in grs)
                fixed.append("ν_max = 7 - 2^(3-Θ)ϕ (book rule)" if rule and len(vm_set) > 1 else f"ν_max={','.join(map(str, vm_set))}")
            else:
                fixed.append(f"ϕ={','.join(sorted({r['phi'] for r in grs}))}")
            setting = (f"P={gd['P']}, {ps.SHORT[gd['local_sum']]}, {ps.SHORT[gd['mode']]}, Θ={gd['theta']}, Ω={gd['omega']}, "
                       f"D={gd['D']}, {'R*=' + gd['R'] if int(gd['R']) else 'no relative limit'}, " + ", ".join(fixed) + f", crop: {gd['crop']}")
            tag = (f"{gd['crop']}_P{gd['P']}_{gd['local_sum'].split('_')[0].lower()}{gd['local_sum'].split('_')[1][0].lower()}_"
                   f"{gd['mode'].lower()}_th{gd['theta']}_om{gd['omega']}_D{gd['D']}_R{gd['R']}")
            legend = [Line2D([], [], linewidth=1.8, markersize=5, label=clabel(c), **cstyle(i, c)) for i, c in enumerate(curves)]
            ref = refs.get(study)
            if ref is not None:
                legend.append(Line2D([], [], color=INK, linestyle=":", linewidth=1, label=f"reference {spec['xlabel']} = {ref[1]}"))

            def a_txt(a):
                return "lossless (A*=0)" if a == 0 and int(gd["R"]) == 0 else f"A*={a}"

            def figure(kind):
                with_mean = kind == "delta" and len(images) > 1
                rows_lab = images + (["Mean over images"] if with_mean else [])
                nr, nc = len(rows_lab), len(As)
                fig_w, fig_h = 3.6 * nc + 0.4, 2.6 * nr + 1.6
                fig, axes = plt.subplots(nr, nc, figsize=(fig_w, fig_h), squeeze=False)
                for i, rl in enumerate(rows_lab):
                    for j, a in enumerate(As):
                        ax = axes[i, j]
                        best_pt = None
                        if rl != "Mean over images":
                            panel = {k: v for k, v in bps.items() if k[0] == rl and k[1] == a}
                            pmin = min(panel.values()) if panel else None
                        for ci, c in enumerate(curves):
                            ys = []
                            for x in xs:
                                if rl == "Mean over images":
                                    vals = []
                                    for im in images:
                                        p = [v for k, v in bps.items() if k[0] == im and k[1] == a]
                                        v = bps.get((im, a, c, x))
                                        if v is not None and p:
                                            vals.append(100 * (v / min(p) - 1))
                                    ys.append(np.mean(vals) if len(vals) == len(images) else np.nan)
                                else:
                                    v = bps.get((rl, a, c, x))
                                    if v is None:
                                        ys.append(np.nan)
                                    else:
                                        ys.append(100 * (v / pmin - 1) if kind == "delta" else v)
                            ax.plot(xs, ys, linewidth=1.5, markersize=3.5, **cstyle(ci, c))
                            yarr = np.array(ys, dtype=float)
                            if np.isfinite(yarr).any():
                                k = int(np.nanargmin(yarr))
                                if best_pt is None or yarr[k] < best_pt[1]:
                                    best_pt = (xs[k], yarr[k])
                        if best_pt is not None:
                            ax.plot([best_pt[0]], [best_pt[1]], marker="*", color=INK, markersize=10, linestyle="none", zorder=5)
                        if ref is not None:
                            ax.axvline(ref[0], color=INK, linestyle=":", linewidth=1)
                        ax.set_title(f"{rl.split('_')[0]}  {a_txt(a)}", fontsize=9, color=INK, loc="left")
                        ax.set_xticks(xs)
                        ax.tick_params(labelsize=7, colors=INK2)
                        ax.grid(True, color=GRID, linewidth=0.8)
                        ax.set_axisbelow(True)
                        for sp in ("top", "right"):
                            ax.spines[sp].set_visible(False)
                        if i == nr - 1:
                            ax.set_xlabel(spec["xlabel"], fontsize=8, color=INK2)
                        if j == 0:
                            ax.set_ylabel("% above panel best" if kind == "delta" else "Rate (bits/sample)", fontsize=8, color=INK2)
                head = spec["title"] + (" - % above each panel's best point" if kind == "delta" else "")
                fig.suptitle(f"{head}\n{setting}", x=0.01, ha="left", y=1 - 0.1 / fig_h, va="top", fontsize=9, color=INK)
                fig.legend(handles=legend, loc="lower center", ncol=min(len(legend), 4), fontsize=7.5, frameon=False)
                leg_rows = math.ceil(len(legend) / 4)
                fig.tight_layout(rect=(0, (0.25 + 0.2 * leg_rows) / fig_h, 1, 1 - 0.75 / fig_h))
                for ext in ("png", "pdf"):
                    fig.savefig(out_dir / f"{study}_{kind}_{tag}.{ext}", dpi=170)
                plt.close(fig)

            figure("rate")
            figure("delta")

            # summary
            xname = spec["xlabel"]
            lines = [f"# {spec['fig']} study ({study}) summary", "", setting, ""]
            hdr = f"| Image | A* | best {xname} | best curve | best bps |"
            sep = "|---|---|---|---|---|"
            if ref is not None:
                hdr += f" bps at {xname}={ref[1]} (same curve) | cost of {xname}={ref[1]} |"
                sep += "---|---|"
            lines += [hdr, sep]
            for a in As:
                for im in images:
                    panel = {k: v for k, v in bps.items() if k[0] == im and k[1] == a}
                    if not panel:
                        continue
                    (_, _, cb, xb), vb = min(panel.items(), key=lambda kv: kv[1])
                    xb_txt = f"2^{int(xb)}" if spec["x"] == "tinc" else f"{xb}"
                    line = f"| {im.split('_')[0]} | {a} | {xb_txt} | {clabel(cb)} | {vb:.4f} |"
                    if ref is not None:
                        on = {k: v for k, v in panel.items() if k[2] == cb}
                        vr = on.get((im, a, cb, ref[0]))
                        line += f" {'-' if vr is None else f'{vr:.4f}'} | {'-' if vr is None else f'{100 * (vr / min(on.values()) - 1):+.3f}%'} |"
                    lines.append(line)
            # sensitivity: spread along x for each curve, mean over images
            lines += ["", f"Spread along {xname} (max - min, % of min), mean over images:", "",
                      "| A* | " + " | ".join(clabel(c) for c in curves) + " |", "|---|" + "---|" * len(curves)]
            for a in As:
                cells = []
                for c in curves:
                    sp = []
                    for im in images:
                        v = [bps[(im, a, c, x)] for x in xs if (im, a, c, x) in bps]
                        if len(v) > 1:
                            sp.append(100 * (max(v) / min(v) - 1))
                    cells.append(f"{np.mean(sp):.3f}%" if sp else "-")
                lines.append(f"| {a} | " + " | ".join(cells) + " |")
            (out_dir / f"{study}_summary_{tag}.md").write_text("\n".join(lines) + "\n")
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
    ap.add_argument("--study", choices=list(STUDY), help="vmin (Fig. 4-10), tinc (Fig. 4-11) or fig48 (Fig. 4-8)")
    ap.add_argument("--model-dir", help="folder that contains ccsds123_i2_hlm (your repository root: .)")
    ap.add_argument("--images", nargs="+")
    ap.add_argument("--out", help="output folder (default: adapt_<study>_results)")
    ap.add_argument("--crop", help="Y0:Y1,X0:X1")
    ap.add_argument("--center-crop", help="HxW, e.g. 64x64 for the book's 'few samples per band' regime")
    ap.add_argument("--bands", help="Z0:Z1")
    ap.add_argument("--abs-error", type=int, nargs="+", help="A* values (default 0 3 20 as the book; fig48: 0)")
    ap.add_argument("--rel-error", type=int, default=0, help="R* (default 0 = not used, as the book)")
    ap.add_argument("--theta", type=int, default=3, help="Theta (default 3, as the book)")
    ap.add_argument("--phi", type=int, nargs="+", help="damping curves (default 0..5 as the book; fig48: 0)")
    ap.add_argument("--psi", type=int, help="offset (default 2^Theta - 1 as the book; 0 when lossless)")
    ap.add_argument("--vmin-values", type=int, nargs="+", help="vmin study: nu_min sweep (default -6..2)")
    ap.add_argument("--tinc-exp", type=int, nargs="+", help="tinc study: log2(t_inc) sweep (default 4..11)")
    ap.add_argument("--vmax-values", type=int, nargs="+", help="fig48: nu_max sweep (default -2..8)")
    ap.add_argument("--vmax", type=int, help="hold nu_max fixed (default: book rule 7 - 2^(3-Theta)*phi)")
    ap.add_argument("--vmin", type=int, default=-1, help="tinc study: nu_min (default -1, Annex C)")
    ap.add_argument("--tinc", type=int, default=2**6, help="vmin study: t_inc (default 2^6, Annex C)")
    ap.add_argument("--P", type=int, default=3)
    ap.add_argument("--local-sum", default="wide-neighbor", choices=list(LS_ALIASES))
    ap.add_argument("--mode", default="full", choices=["full", "reduced"])
    ap.add_argument("--omega", type=int, default=19)
    ap.add_argument("--dynamic-range", type=int, default=16)
    ap.add_argument("--psnr-bits", type=int, default=16)
    ap.add_argument("--nc-var")
    ap.add_argument("--nc-axes")
    ap.add_argument("--engine", choices=["auto", "fast", "lean"], default="auto")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()

    out = Path(args.out or f"adapt_{args.study or 'study'}_results")
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "results.csv"
    refs = {"vmin": (-1, -1), "tinc": (6, "2^6"), "fig48": None}  # Annex C / Vorhaug values
    if args.plot_only:
        make_plots(read_rows(csv_path), out, refs)
        return
    if not args.study or not args.model_dir or not args.images:
        ap.error("--study, --model-dir and --images are required (unless --plot-only)")

    q = 2**args.theta
    As = sorted(set(args.abs_error if args.abs_error else ([0] if args.study == "fig48" else [0, 3, 20])))
    phis = sorted(set(args.phi if args.phi is not None else ([0] if args.study == "fig48" else range(6))))
    if any(not 0 <= p < q for p in phis):
        ap.error(f"phi must be in 0..{q - 1} for Theta={args.theta}")
    psi_lossy = (q - 1) if args.psi is None else args.psi
    if not 0 <= psi_lossy < q:
        ap.error(f"psi must be in 0..{q - 1}")
    if args.tinc not in [2**k for k in range(4, 12)]:
        ap.error("--tinc must be a power of two from 16 to 2048")
    if not -6 <= args.vmin <= 9:
        ap.error("--vmin must be in -6..9")
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

    # (vmin, vmax, tinc, phi) grid for the chosen study
    combos, skipped = [], []
    if args.study == "vmin":
        xs = sorted(set(args.vmin_values if args.vmin_values else range(-6, 3)))
        for phi, v in product(phis, xs):
            vmax = args.vmax if args.vmax is not None else vmax_rule(phi, args.theta)
            (combos if v <= vmax else skipped).append((v, vmax, args.tinc, phi))
    elif args.study == "tinc":
        es = sorted(set(args.tinc_exp if args.tinc_exp else range(4, 12)))
        if any(not 4 <= e <= 11 for e in es):
            ap.error("--tinc-exp values must be 4..11")
        for phi, e in product(phis, es):
            vmax = args.vmax if args.vmax is not None else vmax_rule(phi, args.theta)
            (combos if args.vmin <= vmax else skipped).append((args.vmin, vmax, 2**e, phi))
    else:
        vms = sorted(set(args.vmax_values if args.vmax_values else range(-2, 9)))
        for phi, vmin, e, vmax in product(phis, (-6, 0), (4, 7, 10), vms):
            (combos if vmin <= vmax else skipped).append((vmin, vmax, 2**e, phi))
    bad = [c for c in combos if not (-6 <= c[0] <= c[1] <= 9)]
    if bad:
        ap.error(f"nu values must satisfy -6 <= nu_min <= nu_max <= 9 (e.g. {bad[0]})")

    images = ps.collect_images(args.images)
    jobs = []
    for path in images:
        name, _, nz, ny, nx = ps.parse_image_name(path)
        c = ps.resolve_crop(args, ny, nx, nz)
        for a, (vmin, vmax, tinc, phi) in product(As, combos):
            lossless = (a, args.rel_error) == (0, 0)
            jobs.append({"path": str(path), "image": name, "crop": c, "crop_label": ps.crop_label(c, ny, nx, nz),
                         "samples": (c[1] - c[0]) * (c[3] - c[2]) * (c[5] - c[4]), "study": args.study,
                         "A": a, "R": args.rel_error, "theta": args.theta, "phi": phi, "psi": 0 if lossless else psi_lossy,
                         "vmin": vmin, "vmax": vmax, "tinc": tinc, "P": args.P, "local_sum": ls, "mode": mode,
                         "omega": args.omega, "D": args.dynamic_range, "engine": args.engine, "psnr_bits": args.psnr_bits})
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

    print(f"Study: {args.study} ({STUDY[args.study]['fig']})   A*: {As}   R*: {args.rel_error or 'not used'}   "
          f"Θ={args.theta}   ϕ: {phis}   ψ (lossy): {psi_lossy}")
    if args.study == "vmin":
        print(f"ν_min: {sorted({c[0] for c in combos})}   t_inc={args.tinc}   "
              f"ν_max: {'fixed ' + str(args.vmax) if args.vmax is not None else 'book rule 7 - 2^(3-Θ)ϕ'}")
    elif args.study == "tinc":
        print(f"t_inc: {sorted({c[2] for c in combos})}   ν_min={args.vmin}   "
              f"ν_max: {'fixed ' + str(args.vmax) if args.vmax is not None else 'book rule 7 - 2^(3-Θ)ϕ'}")
    else:
        print(f"ν_max: {sorted({c[1] for c in combos})}   (ν_min, t_inc): {sorted({(c[0], c[2]) for c in combos})}")
    if skipped:
        print(f"Skipped {len(skipped)} combinations with ν_min > ν_max (not allowed by the standard)")
    print(f"Predictor: P={args.P}, {ps.SHORT[ls]}, {args.mode}, Ω={args.omega}, D={args.dynamic_range}   "
          f"engine: {args.engine}   workers: {workers}")
    print(f"Images: {len(images)}   runs: {len(jobs)}   already done: {len(jobs) - len(todo)}   to run: {len(todo)}")
    for path in images:
        j = next(j for j in jobs if j["path"] == str(path))
        npx = (j["crop"][1] - j["crop"][0]) * (j["crop"][3] - j["crop"][2])
        longest = max((c[1] - c[0]) * c[2] for c in combos)
        print(f"  {j['image']}: crop {j['crop_label']}, {j['samples'] / 1e6:.2f} M samples, {npx:,} pixels per band; "
              f"longest ramp {longest:,} pixels = {100 * min(longest, npx) / npx:.1f}% of the image")
    if args.dry_run:
        return

    with ProcessPoolExecutor(max_workers=workers, initializer=ps._init_worker,
                             initargs=(args.model_dir, dict(ps.NC_OPTS))) as pool:
        if args.verify:
            # 16x16 crops with t_inc = 16, so the nu_min -> nu_max ramp happens inside the crop
            vj = []
            pick = sorted({combos[0], combos[-1], combos[len(combos) // 2], combos[len(combos) // 3]})
            for path in images:
                name, _, nz, ny, nx = ps.parse_image_name(path)
                _, _, _, _, z0, z1 = ps.resolve_crop(args, ny, nx, nz)
                h, w = min(16, ny), min(16, nx)
                y0, x0 = (ny - h) // 2, (nx - w) // 2
                for a, (vmin, vmax, tinc, phi) in product(As, pick):
                    lossless = (a, args.rel_error) == (0, 0)
                    vj.append({"path": str(path), "image": name, "crop": (y0, y0 + h, x0, x0 + w, z0, z1), "study": args.study,
                               "A": a, "R": args.rel_error, "theta": args.theta, "phi": phi,
                               "psi": 0 if lossless else psi_lossy, "vmin": vmin, "vmax": vmax, "tinc": 16,
                               "P": args.P, "local_sum": ls, "mode": mode, "omega": args.omega, "D": args.dynamic_range,
                               "engine": args.engine, "verify": True})
            print(f"\nVerifying the {args.engine} engine against the unmodified model: {len(vj)} runs on 16x16 crops (t_inc=16) ...")
            badv = [r for r in pool.map(run_job, vj) if not r["identical"]]
            if badv:
                for r in badv:
                    print(f"  MISMATCH: {r['image']} A*={r['A']} ϕ={r['phi']} ν_min={r['vmin']} ν_max={r['vmax']}")
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
                        print(f"  [{i}/{len(todo)}] FAILED {j['image']} A*={j['A']} ϕ={j['phi']} "
                              f"ν={j['vmin']}..{j['vmax']} t_inc={j['tinc']}: {exc}")
                        continue
                    w.writerow(r)
                    f.flush()
                    el = time.time() - t0
                    if i % 5 == 0 or i == len(todo):
                        print(f"  [{i}/{len(todo)}] {r['image'].split('_')[0]:<10} A*={r['A']:<3} ϕ={r['phi']} "
                              f"ν={r['vmin']}..{r['vmax']} t_inc={r['tinc']:<5} {r['bps']:.4f} bps   "
                              f"elapsed {el / 60:.1f} min, ETA {el / i * (len(todo) - i) / 60:.1f} min")

    make_plots(read_rows(csv_path), out, refs)
    print(f"\nResults: {csv_path}\nPlots and summary: {out}")


if __name__ == "__main__":
    main()
