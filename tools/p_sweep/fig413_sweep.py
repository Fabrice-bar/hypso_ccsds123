#!/usr/bin/env python3
"""
Green Book Figure 4-13 study on HYPSO-1 images: compressed-size increase over
the best sample-representative pair (phi', psi') at different noise levels.

What the Green Book did (CCSDS 120.2-G-2, Dec. 2022, sec. 4.2.2, Figure 4-13):
  * a processed AVIRIS-NG image (300 x 512 x 64, high SNR) with Gaussian noise
    of standard deviation sigma = 0, 25, 50, 100, 200, 400 added;
  * A* = 40, P = 2, Theta = 4, Omega = 19;
  * one fixed pair (phi, psi) in all bands except the first, where phi = psi = 0;
  * normalized damping phi' = phi / 2^Theta and offset psi' = psi / 2^Theta;
  * plotted: expansion (%) = 100 * (compressed size / size at the best pair - 1),
    one heatmap per noise level. Finding: more noise -> larger phi', psi' best.

What this script does: the same grid search on your HYPSO images.
  For every image, noise level sigma and A*, it compresses the image with every
  (phi, psi) pair (default: all of 0 .. 2^Theta - 1), finds the smallest
  compressed size and plots every pair's expansion relative to it.

Defaults follow the book where the book states a value:
  sigma 0 25 50 100 200 400, A* = 40, P = 2, Theta = 4, Omega = 19, first band 0.
Where the Figure 4-13 text gives no value, the defaults are:
  nu_min = -1, nu_max = 3   (the values the book states for the companion
                             Figure 4-14 experiment on the same image)
  t_inc = 2^6               (Green Book Annex C default)
  local sum / mode = wide neighbor / full   (best for HYPSO in the P sweep; the
                             book's Annex C uses wide column / reduced for
                             AVIRIS-NG because of that sensor's streaking)
  D = 16, hybrid coder and the rest as Vorhaug 2024.
Everything is configurable.

Noise: Gaussian noise (in DN) is added to the raw counts, rounded, and a
constant offset (default 4 x the largest sigma) is added to every sample at
every noise level so that noise never gets clipped at 0. A constant offset
does not change prediction. The noise is the same for every (phi, psi) pair of
an image and sigma (fixed seed), so the heatmap compares like with like.
Distortion (MSE, PSNR) is measured against the noisy input that is compressed.
HYPSO raw data already contain sensor noise: sigma = 0 means "no added noise".

Size of the study: images x sigmas x A* x phi values x psi values runs.
The default grid is 16 x 16 = 256 runs per image, noise level and A*: with
6 images and 6 noise levels that is 9216 runs. On full images that takes days;
use a crop (the book's image was 300 x 512 x 64) and/or fewer grid points, e.g.
    --center-crop 512x300 --bands 0:64        (book-sized crop)
    --phi 0 2 4 6 8 10 12 14 --psi 0 2 4 6 8 10 12 14   (8 x 8 grid)
--dry-run prints the number of runs.

Examples (from the repository root)
    python tools/p_sweep/fig413_sweep.py --model-dir . --images data --center-crop 512x300 --dry-run
    python tools/p_sweep/fig413_sweep.py --model-dir . --images data --center-crop 512x300 --verify --out results/fig413
    python tools/p_sweep/fig413_sweep.py --plot-only --out results/fig413

Needs p_sweep.py and fast_engine.py in the same folder.

Outputs in --out:
    results.csv                one row per run (rerun the same command to resume)
    fig413_expansion_*.png/pdf heatmaps: rows = images (+ mean), columns = sigma;
                               the white star marks the best pair of each panel
    fig413_optimum_*.png/pdf   best phi' and psi' against sigma, per image
    fig413_mse_*.png/pdf       MSE relative to the best pair's MSE (A* > 0)
    fig413_summary_*.md        best pair, its rate and PSNR, and the expansion at
                               (0, 0) and at Vorhaug's normalized setting
"""

import argparse
import csv
import math
import os
import sys
import time
import zlib
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
KEY = ["image", "crop", "sigma", "noise_offset", "seed", "A", "R", "theta", "phi", "psi", "first_band",
       "P", "local_sum", "mode", "omega", "vmin", "vmax", "tinc", "D"]
FIELDS = KEY + ["ny", "nx", "nz", "header_bits", "body_bits", "total_bits", "bps",
                "max_abs_err", "mse", "psnr_db", "snr_db", "clipped_pct", "seconds"]
INK, INK2 = "#191919", "#666666"


def key_of(d):
    return tuple(str(d[k]) for k in KEY)


# ----------------------------------------------------------------------------
# Noisy image (identical for every (phi, psi) of one image / sigma)
# ----------------------------------------------------------------------------


def noisy_cube(cube, image, sigma, offset, seed, D):
    """Add rounded Gaussian noise (std sigma DN) band by band, plus a constant offset."""
    Y, X, Z = cube.shape
    hi = 2**D - 1
    out = np.empty((Y, X, Z), dtype=np.uint16 if D <= 16 else np.int64)
    clipped = 0
    tag = zlib.crc32(image.encode())
    for z in range(Z):
        b = cube[:, :, z].astype(np.float64) + offset
        if sigma > 0:
            rng = np.random.default_rng([seed, tag, int(round(sigma * 1000)), z])
            b += rng.normal(0.0, sigma, size=(Y, X))
        b = np.rint(b)
        clipped += int(np.count_nonzero((b < 0) | (b > hi)))
        out[:, :, z] = np.clip(b, 0, hi)
    return out, 100.0 * clipped / cube.size


# ----------------------------------------------------------------------------
# Worker
# ----------------------------------------------------------------------------


def tables(job, Z):
    phi = np.full(Z, job["phi"], dtype=np.int64)
    psi = np.full(Z, job["psi"], dtype=np.int64)
    if job["first_band"] == "zero":
        phi[0] = psi[0] = 0
    return phi, psi


def run_job(job):
    t0 = time.perf_counter()
    engine = job["engine"]
    cube, signed = ps.load_cube(job["path"], job["crop"], native=True)
    if signed:
        raise ValueError(f"{job['image']}: signed samples are not supported by this study")
    cube, clipped = noisy_cube(cube, job["image"], job["sigma"], job["noise_offset"], job["seed"], job["D"])
    if engine != "fast":
        cube = cube.astype(np.int64)
    Z = cube.shape[2]
    phi_t, psi_t = tables(job, Z)
    ov = {"nu_min": job["vmin"], "nu_max": job["vmax"], "t_inc": job["tinc"], "omega": job["omega"],
          "theta": job["theta"], "phi_table": phi_t, "psi_table": psi_t}
    fid = (job["A"], job["R"])
    args = (False, job["P"], job["local_sum"], job["mode"], fid, job["D"])

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
        "bps": total / n, **ps.distortion_fields(st, job["psnr_bits"]),
        "clipped_pct": round(clipped, 6), "seconds": round(time.perf_counter() - t0, 1),
    }


# ----------------------------------------------------------------------------
# Plots and summary
# ----------------------------------------------------------------------------


def make_plots(rows, out_dir, cmax_arg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    groups = {}
    for r in rows:
        g = tuple(r[k] for k in KEY if k not in ("image", "sigma", "phi", "psi"))
        groups.setdefault(g, []).append(r)

    for g, rs in groups.items():
        gd = dict(zip([k for k in KEY if k not in ("image", "sigma", "phi", "psi")], g))
        theta = int(gd["theta"])
        q = 2**theta
        images = sorted({r["image"] for r in rs})
        sigmas = sorted({float(r["sigma"]) for r in rs})
        phis = sorted({int(r["phi"]) for r in rs})
        psis = sorted({int(r["psi"]) for r in rs})
        bits = {(r["image"], float(r["sigma"]), int(r["phi"]), int(r["psi"])): float(r["total_bits"]) for r in rs}

        def fnum(v):
            try:
                x = float(v)
                return x if np.isfinite(x) else None
            except (TypeError, ValueError):
                return None
        mse = {(r["image"], float(r["sigma"]), int(r["phi"]), int(r["psi"])): fnum(r["mse"]) for r in rs}
        bps = {(r["image"], float(r["sigma"]), int(r["phi"]), int(r["psi"])): float(r["bps"]) for r in rs}
        psnr = {(r["image"], float(r["sigma"]), int(r["phi"]), int(r["psi"])): fnum(r["psnr_db"]) for r in rs}

        def grid(table, im, s):
            m = np.full((len(phis), len(psis)), np.nan)
            for i, f in enumerate(phis):
                for j, p in enumerate(psis):
                    v = table.get((im, s, f, p))
                    if v is not None:
                        m[i, j] = v
            return m

        def expansion(table, im, s):
            m = grid(table, im, s)
            return 100 * (m / np.nanmin(m) - 1) if np.isfinite(m).any() else m

        panels = {}  # (row label, sigma) -> expansion grid
        for s in sigmas:
            per = [expansion(bits, im, s) for im in images]
            for im, e in zip(images, per):
                panels[(im, s)] = e
            full = [e for e in per if np.isfinite(e).all()]
            if len(images) > 1 and len(full) == len(images):
                panels[("Mean over images", s)] = np.mean(full, axis=0)
        rows_lab = images + (["Mean over images"] if len(images) > 1 else [])
        a_txt = "lossless" if (int(gd["A"]), int(gd["R"])) == (0, 0) else f"A*={gd['A']}" + (f", R*={gd['R']}" if int(gd["R"]) else "")
        setting = (f"{a_txt}, P={gd['P']}, {ps.SHORT[gd['local_sum']]}, {ps.SHORT[gd['mode']]}, Θ={theta}, Ω={gd['omega']}, "
                   f"ν={gd['vmin']}..{gd['vmax']}, t_inc={gd['tinc']}, D={gd['D']}, first band {'0' if gd['first_band'] == 'zero' else 'same'}, "
                   f"noise offset {gd['noise_offset']} DN, crop: {gd['crop']}")
        tag = (f"{gd['crop']}_A{gd['A']}_R{gd['R']}_P{gd['P']}_{gd['local_sum'].split('_')[0].lower()}"
               f"{gd['local_sum'].split('_')[1][0].lower()}_{gd['mode'].lower()}_th{theta}_v{gd['vmin']}_{gd['vmax']}_t{gd['tinc']}_D{gd['D']}")
        def edges(vals):
            """Cell edges halfway between grid values (works for uneven grids such as 0 2 4 ...)."""
            c = np.asarray(vals, dtype=float)
            if c.size == 1:
                return np.array([c[0] - 0.5, c[0] + 0.5]) / q
            mid = (c[1:] + c[:-1]) / 2
            return np.concatenate([[c[0] - (c[1] - c[0]) / 2], mid, [c[-1] + (c[-1] - c[-2]) / 2]]) / q
        x_edges, y_edges = edges(psis), edges(phis)

        def heatmaps(pan, title, cbar_label, fname, mark_best=True, cmax=None):
            vals = np.concatenate([v[np.isfinite(v)] for v in pan.values()]) if pan else np.array([0.0])
            top = cmax if cmax is not None else min(200.0, float(np.nanmax(vals)) if vals.size else 1.0)
            top = max(top, 1e-6)
            nr, nc = len(rows_lab), len(sigmas)
            fig_w, fig_h = 2.25 * nc + 1.3, 2.05 * nr + 0.9
            fig, axes = plt.subplots(nr, nc, figsize=(fig_w, fig_h), squeeze=False)
            norm = Normalize(0, top)
            im_obj = None
            for i, rl in enumerate(rows_lab):
                for j, s in enumerate(sigmas):
                    ax = axes[i, j]
                    m = pan.get((rl, s))
                    if m is None:
                        ax.axis("off")
                        continue
                    im_obj = ax.pcolormesh(x_edges, y_edges, np.clip(m, 0, top), cmap="Blues", norm=norm, shading="flat")
                    if mark_best and np.isfinite(m).any():
                        bi, bj = np.unravel_index(np.nanargmin(m), m.shape)
                        ax.plot(psis[bj] / q, phis[bi] / q, marker="*", color="white", markeredgecolor=INK,
                                markersize=10, linestyle="none")
                    ax.set_aspect("equal")
                    ax.tick_params(labelsize=6.5, colors=INK2)
                    if i == 0:
                        ax.set_title(f"σ = {s:g}", fontsize=9, color=INK)
                    if j == 0:
                        ax.set_ylabel(f"{rl.split('_')[0]}\nϕ'", fontsize=8, color=INK)
                    if i == nr - 1:
                        ax.set_xlabel("ψ'", fontsize=8, color=INK)
            fig.suptitle(f"{title}\n{setting}", x=0.01, ha="left", fontsize=9, color=INK)
            fig.tight_layout(rect=(0, 0, 1 - 0.9 / fig_w, 1 - 0.5 / fig_h))
            if im_obj is not None:
                cax = fig.add_axes([1 - 0.75 / fig_w, 0.15, 0.18 / fig_w, 0.7])
                cb = fig.colorbar(im_obj, cax=cax)
                cb.set_label(cbar_label, fontsize=8)
                cb.ax.tick_params(labelsize=7)
            for ext in ("png", "pdf"):
                fig.savefig(out_dir / f"{fname}_{tag}.{ext}", dpi=200)
            plt.close(fig)

        heatmaps(panels, "Increase in compressed size vs the best (ϕ', ψ') of each panel (Green Book Fig. 4-13 style)",
                 "expansion (%)", "fig413_expansion", cmax=cmax_arg)

        if any(v is not None and v > 0 for v in mse.values()):
            mpan = {}
            for s in sigmas:
                for im in images:
                    m = grid(mse, im, s)
                    b = expansion(bits, im, s)
                    if np.isfinite(b).any() and np.isfinite(m).any():
                        bi, bj = np.unravel_index(np.nanargmin(b), b.shape)
                        mpan[(im, s)] = 100 * (m / m[bi, bj] - 1)
            if mpan:
                vals = np.concatenate([np.abs(v[np.isfinite(v)]) for v in mpan.values()])
                lim = float(np.nanmax(vals)) if vals.size else 1.0
                try:
                    # MSE change can be negative: use a diverging map centred on 0
                    nr, nc = len(images), len(sigmas)
                    fig_w, fig_h = 2.25 * nc + 1.3, 2.05 * nr + 0.9
                    fig, axes = plt.subplots(nr, nc, figsize=(fig_w, fig_h), squeeze=False)
                    norm = Normalize(-max(lim, 1e-6), max(lim, 1e-6))
                    im_obj = None
                    for i, rl in enumerate(images):
                        for j, s in enumerate(sigmas):
                            ax = axes[i, j]
                            m = mpan.get((rl, s))
                            if m is None:
                                ax.axis("off")
                                continue
                            im_obj = ax.pcolormesh(x_edges, y_edges, m, cmap="RdBu_r", norm=norm, shading="flat")
                            ax.set_aspect("equal")
                            ax.tick_params(labelsize=6.5, colors=INK2)
                            if i == 0:
                                ax.set_title(f"σ = {s:g}", fontsize=9, color=INK)
                            if j == 0:
                                ax.set_ylabel(f"{rl.split('_')[0]}\nϕ'", fontsize=8, color=INK)
                            if i == nr - 1:
                                ax.set_xlabel("ψ'", fontsize=8, color=INK)
                    fig.suptitle(f"MSE of the decoded image relative to the rate-optimal pair (red = more distortion)\n{setting}",
                                 x=0.01, ha="left", fontsize=9, color=INK)
                    fig.tight_layout(rect=(0, 0, 1 - 0.9 / fig_w, 1 - 0.5 / fig_h))
                    cax = fig.add_axes([1 - 0.75 / fig_w, 0.15, 0.18 / fig_w, 0.7])
                    cb = fig.colorbar(im_obj, cax=cax)
                    cb.set_label("MSE change (%)", fontsize=8)
                    cb.ax.tick_params(labelsize=7)
                    for ext in ("png", "pdf"):
                        fig.savefig(out_dir / f"fig413_mse_{tag}.{ext}", dpi=200)
                    plt.close(fig)
                except Exception as exc:  # plotting must never lose results
                    print(f"  (MSE figure skipped: {exc})")

        # optimum (phi', psi') against sigma
        fig, axs = plt.subplots(1, 2, figsize=(9, 3.6))
        colors = ["#257de0", "#f47230", "#00b894", "#f2a600", "#8e44ad", "#d63031", "#6c5ce7", "#636e72"]
        for k, im in enumerate(images):
            best = []
            for s in sigmas:
                e = panels.get((im, s))
                if e is None or not np.isfinite(e).any():
                    best.append((np.nan, np.nan))
                    continue
                bi, bj = np.unravel_index(np.nanargmin(e), e.shape)
                best.append((phis[bi] / q, psis[bj] / q))
            best = np.array(best)
            for a_i, lab in enumerate(("best ϕ'", "best ψ'")):
                axs[a_i].plot(sigmas, best[:, a_i], marker="os^Dv*PX"[k % 8], color=colors[k % len(colors)],
                              linewidth=1.6, markersize=5, label=im.split("_")[0])
        for a_i, lab in enumerate(("best ϕ' (damping)", "best ψ' (offset)")):
            axs[a_i].set_xlabel("added noise σ (DN)", fontsize=9)
            axs[a_i].set_ylabel(lab, fontsize=9)
            axs[a_i].set_ylim(-0.05, 1.0)
            axs[a_i].grid(True, color="#e5e5e5")
            for sp in ("top", "right"):
                axs[a_i].spines[sp].set_visible(False)
        axs[1].legend(fontsize=7.5, frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))
        fig.suptitle(f"Best sample-representative pair against added noise\n{setting}", x=0.01, ha="left", fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.86))
        for ext in ("png", "pdf"):
            fig.savefig(out_dir / f"fig413_optimum_{tag}.{ext}", dpi=200)
        plt.close(fig)

        # summary table
        vor_phi, vor_psi = round(3 / 8 * q), round(7 / 8 * q)  # Vorhaug: phi=3, psi=7 at Theta=3
        lines = ["# Figure 4-13 study summary", "", setting, "",
                 f"Vorhaug's (phi=3, psi=7, Theta=3) is (phi', psi') = (0.375, 0.875), i.e. (phi, psi) = ({vor_phi}, {vor_psi}) at Theta={theta}.", "",
                 "| Image | σ | best ϕ (ϕ') | best ψ (ψ') | bps at best | PSNR at best (dB) | expansion at (0,0) | expansion at Vorhaug | 2nd best |",
                 "|---|---|---|---|---|---|---|---|---|"]
        for im in images:
            for s in sigmas:
                e = panels.get((im, s))
                if e is None or not np.isfinite(e).any():
                    continue
                bi, bj = np.unravel_index(np.nanargmin(e), e.shape)
                f, p = phis[bi], psis[bj]
                flat = np.sort(e[np.isfinite(e)])
                second = flat[1] if flat.size > 1 else np.nan

                def at(fv, pv):
                    if fv in phis and pv in psis:
                        v = e[phis.index(fv), psis.index(pv)]
                        return f"{v:.2f}%" if np.isfinite(v) else "-"
                    return "not on grid"
                pv = psnr.get((im, s, f, p))
                lines.append(f"| {im.split('_')[0]} | {s:g} | {f} ({f / q:.3f}) | {p} ({p / q:.3f}) | {bps[(im, s, f, p)]:.4f} | "
                             f"{'-' if pv is None else f'{pv:.2f}'} | {at(0, 0)} | {at(vor_phi, vor_psi)} | +{second:.2f}% |")
        (out_dir / f"fig413_summary_{tag}.md").write_text("\n".join(lines) + "\n")
        print("\n".join(lines))


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------


def read_rows(path):
    if not path.exists() or path.stat().st_size == 0:
        return []
    with open(path, newline="") as f:
        rd = csv.DictReader(f)
        if any(k not in (rd.fieldnames or []) for k in KEY + ["total_bits"]):
            raise SystemExit(f"{path} was not written by this script. Use another --out folder. Nothing was changed.")
        return list(rd)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", help="folder that contains ccsds123_i2_hlm (your repository root: .)")
    ap.add_argument("--images", nargs="+", help=".nc files, raw files, or folders")
    ap.add_argument("--out", default="fig413_results")
    ap.add_argument("--crop", help="Y0:Y1,X0:X1 (lines, samples)")
    ap.add_argument("--center-crop", help="HxW (lines x samples), e.g. 512x300 like the book's image")
    ap.add_argument("--bands", help="Z0:Z1, e.g. 0:64 for 64 bands like the book")
    ap.add_argument("--sigma", type=float, nargs="+", default=[0, 25, 50, 100, 200, 400],
                    help="added Gaussian noise std in DN (default: the book's 0 25 50 100 200 400)")
    ap.add_argument("--noise-offset", default="auto",
                    help="constant DN added to every sample so noise is not clipped at 0 (default: 4 x max sigma)")
    ap.add_argument("--seed", type=int, default=12345, help="noise seed")
    ap.add_argument("--abs-error", type=int, nargs="+", default=[40], help="A* values (default 40, as the book)")
    ap.add_argument("--rel-error", type=int, default=0, help="relative limit R* (default 0 = not used, as the book)")
    ap.add_argument("--theta", type=int, default=4, help="sample representative resolution Theta (default 4, as the book)")
    ap.add_argument("--phi", type=int, nargs="+", help="damping values phi (default: all 0 .. 2^Theta-1)")
    ap.add_argument("--psi", type=int, nargs="+", help="offset values psi (default: all 0 .. 2^Theta-1)")
    ap.add_argument("--first-band", choices=["zero", "same"], default="zero",
                    help="zero: phi = psi = 0 in the first band, as the book (default); same: one pair in all bands")
    ap.add_argument("--P", type=int, default=2, help="prediction bands (default 2, as the book)")
    ap.add_argument("--local-sum", default="wide-neighbor", choices=list(LS_ALIASES))
    ap.add_argument("--mode", default="full", choices=["full", "reduced"])
    ap.add_argument("--omega", type=int, default=19)
    ap.add_argument("--vmin", type=int, default=-1)
    ap.add_argument("--vmax", type=int, default=3)
    ap.add_argument("--tinc", type=int, default=2**6)
    ap.add_argument("--dynamic-range", type=int, default=16)
    ap.add_argument("--psnr-bits", type=int, default=12, help="sensor bit depth for the PSNR peak (HYPSO-1: 12)")
    ap.add_argument("--cmax", type=float, help="top of the expansion colour scale in %% (default: max, capped at 200 like the book)")
    ap.add_argument("--nc-var")
    ap.add_argument("--nc-axes")
    ap.add_argument("--engine", choices=["auto", "fast", "lean"], default="auto")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--verify", action="store_true", help="check the engine against the unmodified model first")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "results.csv"
    if args.plot_only:
        make_plots(read_rows(csv_path), out, args.cmax)
        return
    if not args.model_dir or not args.images:
        ap.error("--model-dir and --images are required (unless --plot-only)")

    q = 2**args.theta
    if not 0 <= args.theta <= 4:
        ap.error("--theta must be 0..4")
    phis = sorted(set(args.phi if args.phi else range(q)))
    psis = sorted(set(args.psi if args.psi else range(q)))
    if any(not 0 <= v < q for v in phis + psis):
        ap.error(f"phi and psi must be in 0..{q - 1} for Theta={args.theta}")
    if not -6 <= args.vmin <= args.vmax <= 9:
        ap.error("need -6 <= vmin <= vmax <= 9")
    if args.tinc not in [2**k for k in range(4, 12)]:
        ap.error("--tinc must be a power of two from 16 to 2048")
    if any(s < 0 for s in args.sigma):
        ap.error("--sigma must be >= 0")
    offset = int(math.ceil(4 * max(args.sigma))) if args.noise_offset == "auto" else int(args.noise_offset)
    if args.engine == "auto":
        try:
            import numba  # noqa: F401
            args.engine = "fast"
        except ImportError:
            args.engine = "lean"
            print("numba not installed: using the model's Python code (very slow for this study). pip install numba")
    ps.NC_OPTS.update(var=args.nc_var, axes=args.nc_axes)
    ls = LS_ALIASES[args.local_sum]
    mode = args.mode.upper()

    images = ps.collect_images(args.images)
    jobs = []
    for path in images:
        name, _, nz, ny, nx = ps.parse_image_name(path)
        c = ps.resolve_crop(args, ny, nx, nz)
        for s, a in product(sorted(set(args.sigma)), sorted(set(args.abs_error))):
            lossless = (a, args.rel_error) == (0, 0)
            for f, p in product(phis, [0] if lossless else psis):  # psi has no effect (must be 0) when lossless
                jobs.append({"path": str(path), "image": name, "crop": c, "crop_label": ps.crop_label(c, ny, nx, nz),
                             "samples": (c[1] - c[0]) * (c[3] - c[2]) * (c[5] - c[4]),
                             "sigma": float(s), "noise_offset": offset, "seed": args.seed, "A": a, "R": args.rel_error,
                             "theta": args.theta, "phi": f, "psi": p, "first_band": args.first_band, "P": args.P,
                             "local_sum": ls, "mode": mode, "omega": args.omega, "vmin": args.vmin, "vmax": args.vmax,
                             "tinc": args.tinc, "D": args.dynamic_range, "engine": args.engine, "psnr_bits": args.psnr_bits})
    done = {key_of(r) for r in read_rows(csv_path)}
    todo = [j for j in jobs if key_of({**j, "crop": j["crop_label"]}) not in done]
    todo.sort(key=lambda j: j["samples"], reverse=True)

    workers = args.workers or os.cpu_count() or 1
    try:
        import psutil
        per = (todo[0]["samples"] * (6 if args.engine == "fast" else 20) / 2**20 + 400) if todo else 1
        workers = min(workers, max(1, int(0.8 * psutil.virtual_memory().available / 2**20 // per)))
    except ImportError:
        pass
    if os.name == "nt":
        workers = min(workers, 61)
    workers = max(1, min(workers, len(todo) or 1))

    print(f"Noise σ (DN): {sorted(set(args.sigma))}   constant offset: {offset} DN   seed: {args.seed}")
    print(f"A*: {sorted(set(args.abs_error))}   R*: {args.rel_error or 'not used'}   Θ={args.theta}   "
          f"ϕ: {phis}   ψ: {psis}   first band: {args.first_band}")
    print(f"Predictor: P={args.P}, {ps.SHORT[ls]}, {args.mode}, Ω={args.omega}, ν {args.vmin}..{args.vmax}, "
          f"t_inc={args.tinc}, D={args.dynamic_range}   engine: {args.engine}   workers: {workers}")
    print(f"Images: {len(images)}   runs: {len(jobs)}   already done: {len(jobs) - len(todo)}   to run: {len(todo)}")
    for path in images:
        j = next(j for j in jobs if j["path"] == str(path))
        print(f"  {j['image']}: crop {j['crop_label']}, {j['samples'] / 1e6:.2f} M samples")
    if todo:
        ms = sum(j["samples"] for j in todo) / 1e6
        print(f"Work: {ms:,.0f} M samples in total. At ~0.3-1 us/sample/core (fast engine, estimate) "
              f"that is roughly {ms * 0.3 / 3600 / workers:.1f}-{ms * 1.0 / 3600 / workers:.1f} h with {workers} workers.")
    if args.dry_run:
        return

    with ProcessPoolExecutor(max_workers=workers, initializer=ps._init_worker,
                             initargs=(args.model_dir, dict(ps.NC_OPTS))) as pool:
        if args.verify:
            vj = []
            rng = np.random.default_rng(0)
            corners = {(phis[0], psis[0]), (phis[0], psis[-1]), (phis[-1], psis[0]), (phis[-1], psis[-1]),
                       (phis[len(phis) // 2], psis[len(psis) // 2])}
            extra = {(int(rng.choice(phis)), int(rng.choice(psis))) for _ in range(4)}
            for path in images:
                name, _, nz, ny, nx = ps.parse_image_name(path)
                _, _, _, _, z0, z1 = ps.resolve_crop(args, ny, nx, nz)
                h, w = min(16, ny), min(16, nx)
                y0, x0 = (ny - h) // 2, (nx - w) // 2
                for s, a in product(sorted({min(args.sigma), max(args.sigma)}), sorted(set(args.abs_error))):
                    lossless = (a, args.rel_error) == (0, 0)
                    for f, p in sorted(corners | extra):
                        vj.append({"path": str(path), "image": name, "crop": (y0, y0 + h, x0, x0 + w, z0, z1),
                                   "sigma": float(s), "noise_offset": offset, "seed": args.seed, "A": a, "R": args.rel_error,
                                   "theta": args.theta, "phi": f, "psi": 0 if lossless else p, "first_band": args.first_band,
                                   "P": args.P, "local_sum": ls, "mode": mode, "omega": args.omega, "vmin": args.vmin,
                                   "vmax": args.vmax, "tinc": 16, "D": args.dynamic_range, "engine": args.engine,
                                   "verify": True})
            print(f"\nVerifying the {args.engine} engine against the unmodified model: {len(vj)} runs on 16x16 crops ...")
            badv = [r for r in pool.map(run_job, vj) if not r["identical"]]
            if badv:
                for r in badv:
                    print(f"  MISMATCH: {r['image']} σ={r['sigma']} A*={r['A']} ϕ={r['phi']} ψ={r['psi']}")
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
                        print(f"  [{i}/{len(todo)}] FAILED {j['image']} σ={j['sigma']} ϕ={j['phi']} ψ={j['psi']}: {exc}")
                        continue
                    w.writerow(r)
                    f.flush()
                    el = time.time() - t0
                    if i % 10 == 0 or i == len(todo):
                        print(f"  [{i}/{len(todo)}] {r['image'].split('_')[0]:<10} σ={r['sigma']:<6g} A*={r['A']:<3} "
                              f"ϕ={r['phi']:<2} ψ={r['psi']:<2} {r['bps']:.4f} bps   elapsed {el / 60:.1f} min, "
                              f"ETA {el / i * (len(todo) - i) / 60:.1f} min")

    make_plots(read_rows(csv_path), out, args.cmax)
    print(f"\nResults: {csv_path}\nPlots and summary: {out}")


if __name__ == "__main__":
    main()
