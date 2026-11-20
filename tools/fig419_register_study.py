#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HYPSO register-size experiment — CCSDS 120.2-G-2 §4.2.5, Figure 4-19.

Save this one file as tools/fig419_register_study.py in the NTNU verification
model. Use its Python environment. From the repository root:

  python tools/fig419_register_study.py --input-dir data --crop 256 256 --dry-run
  python tools/fig419_register_study.py --input-dir data --crop 256 256 --output results/fig419_256
  python tools/fig419_register_study.py --input-dir data --full-image --output results/fig419_full
  python tools/fig419_register_study.py --output results/fig419_full --plot-only

Default: centered Lisbon and Finnmark; all native bands; A = 0 and 15 DN.
Coded unsigned D=16; P=3; narrow-neighbor/reduced; hybrid; vmin=-1, vmax=3,
t_inc=2048; theta=3, phi=3, psi=7 (zero for lossless); default weights/zero offsets.
Test physical R (bits), not the 6-bit header representation (0 means 64):
  Omega 12: R=32,34,35,64   (R*=35)
  Omega 14: R=32,34,37,64   (R*=37)
  Omega 19: R=37,39,42,64   (R*=42)
Two scenes x two A values x 12 (Omega,R) pairs = 48 points at ONE size.
--dense fills in every legal width between Rmin and R*, plus R64: 76 points.
For the same study at two sizes, run twice in different output directories.
--row Y --col X selects a common zero-based origin; --origin IMAGE Y X
sets it for a particular scene. --full-image reads each file's own dimensions.
--images lisbon bangladesh santarem doha finnmark selects all five: 120 points.
--workers 2 (default): one complete configuration per process; neither crops
nor individual predictions are split. A 90 GiB aggregate planning allowance and
available machine memory limit concurrency. This is an estimate, not an OS cap.

The runner uses the public NTNU ccsds123_issue_2_verification_model at commit
9fabb0371ebf93d72a0f83ca053e9cdb2c63e3c6 (with its lookup tables). Tested
with Python 3.11, NumPy 1.26, bitarray 2.8.2, netCDF4, matplotlib >= 3.7,
and psutil. Install the repo's requirements plus any missing plotting/NetCDF
dependencies into the same environment. The trace-saving hybrid encoder adapter
verifies that the upstream encoder source matches the supported implementation.

Outputs:
  fig419_register_rate.png/.pdf: signed change vs that scene/A/Omega's R=64,
    100*(bits_R-bits_64)/bits_64. Green shading = guaranteed no overflow (R>=R*).
  fig419_absolute_rate.png/.pdf: actual compressed bits per original sample.
  register_comparison.csv: every planned point, matching reference, change in
    actual stream bits, wrap count and whether entropy payload equals R64.
  results.csv, plan.csv, metadata.json and jobs/*.json are auditable checkpoints.
The stream includes header, compressed payload/tail, and output word padding;
it excludes externally supplied initialization. Raw unsigned DN only: all
bands are kept, with no masking, radiometric conversion or synthetic noise.
Per working region, two adjacent widths can give equal size with different
payloads. The R=64 check therefore compares entropy bit lengths and hashes, not
only compressed size. Different R is signaled in the header, so full-stream
hashes would be inappropriate. Encoder-side checks enforce max error <= A;
these checks are not an independent decoder/bitstream conformance test.

Existing reference data: 90 R=64 observations for Omega=12,14,19 taken from
the user's completed omega_fig418_256 (centered 256 x 256) and omega_fig417_256
(native sizes despite its folder name). Reuse requires identical NetCDF bytes,
raw variable/axes, spatial region, model digest and predictor/encoder settings.
If all match, 12 crop or 6 full-image controls are reused in the default study.
Archive-only controls have no observed wrap count; the blank in the CSV is
intentional. --fresh measures all the reference points as well. Rerunning the
same command resumes completed measurements. A different grid needs its own
output directory; to regenerate figures alone use --plot-only --output DIR.

The main application and all study helpers are included below in readable
Python. The launcher materializes private worker files in a temporary folder
so Windows and Unix can start clean configuration processes. It never edits
the verification model or the inputs.
"""
from pathlib import Path
import runpy
import sys
import tempfile


def _run():
    for stream in (sys.stdout,sys.stderr):
        if hasattr(stream,"reconfigure"):
            stream.reconfigure(errors="backslashreplace")
    with tempfile.TemporaryDirectory(prefix="hypso_fig419_") as name:
        folder=Path(name)
        for filename,contents in _SOURCES.items():
            (folder/filename).write_bytes(contents.encode("utf-8"))
        sys.path.insert(0,name)
        try:
            runpy.run_path(str(folder/"register_sweep.py"),run_name="__main__")
        finally:
            sys.path.remove(name)


# The entry point comes first; the archived R=64 measurements come last.

_SOURCES={
'register_sweep.py': r'''#!/usr/bin/env python3
"""HYPSO / CCSDS 120.2-G-2, Figure 4-19: register width at fixed weight resolution.

From the verification-model root: python tools/fig419/register_sweep.py --help
One configuration = one scene/region, one A, one Omega and one physical R.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import _damping_core as core
import _register_engine as engine
from _register_report import make_report
from sweep_workers import output_lock

HERE = Path(__file__).resolve().parent
D = 16


def register_grid(omega, p, mode, explicit=None, dense=False):
    lower = max(32, D + omega + 2)  # CCSDS 123.0-B-2, 4.7.2
    kappa = 1 if mode == "reduced" else 19
    # ceil(log2(n+1)) == n.bit_length() for a positive integer n (Green Book eq. 23).
    star = omega + 2 + (((1 << D) - 1) * (8*p + kappa)).bit_length()
    if explicit is not None:
        illegal = [r for r in explicit if not lower <= r <= 64]
        if illegal:
            raise ValueError(f"Illegal R={illegal} for Ω={omega}, D={D}; legal interval is [{lower}, 64]. "
                             "Omit --r-values for an automatically legal grid.")
        widths = set(explicit)
    elif dense:
        widths = set(range(lower, max(lower,star)+1))
    else:
        widths = {lower, min(lower+2,64), max(lower,star)}
    return lower, star, sorted(widths | {64})


def arguments():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir",type=Path,default=Path("data"))
    p.add_argument("--repo",type=Path,help="Verification-model root (normally the current directory).")
    p.add_argument("--images",nargs="+",choices=sorted(core.SCENES),default=["lisbon","finnmark"])
    s=p.add_mutually_exclusive_group()
    s.add_argument("--crop",type=int,nargs=2,metavar=("HEIGHT","WIDTH"),help="Centered spatial crop; default 256 256.")
    s.add_argument("--full-image",action="store_true",help="Use every spatial sample at the native size of each file.")
    p.add_argument("--row",type=int,help="Common crop origin: zero-based line. Supply with --col.")
    p.add_argument("--col",type=int,help="Common crop origin: zero-based sample.")
    p.add_argument("--origin",nargs=3,action="append",default=[],metavar=("IMAGE","ROW","COL"),
                   help="Per-image crop origin, repeatable. Overrides common row/col for that image.")
    p.add_argument("--variable",default="products/dn",help="Raw integer DN variable in the NetCDF file.")
    p.add_argument("--axis-order",choices=("yxz","yzx","xyz","xzy","zyx","zxy"),help="Override stored dimension order.")
    p.add_argument("--omega-values",type=int,nargs="+",default=[12,14,19])
    p.add_argument("--a-values",type=int,nargs="+",default=[0,15],help="Error bounds in DN (0..15); zero is lossless.")
    w=p.add_mutually_exclusive_group()
    w.add_argument("--r-values",type=int,nargs="+",help="Exact widths, each legal for every selected Ω; 64 is always added.")
    w.add_argument("--dense",action="store_true",help="Test every legal width from Rmin through R*, plus 64.")
    p.add_argument("--p",type=int,default=3,help="Number of preceding prediction bands.")
    p.add_argument("--local-sum",choices=list(core.LOCAL_SUM_OPTIONS),default="narrow-neighbor")
    p.add_argument("--prediction-mode",choices=("reduced","full"),default="reduced")
    p.add_argument("--workers",type=int,default=2,help="Maximum concurrent complete configurations. Default: 2.")
    p.add_argument("--total-memory-gb",type=float,default=90,help="Aggregate planning budget in GiB; allow OS headroom.")
    p.add_argument("--max-memory-gb",type=float,default=90,help="Per-configuration planning budget in GiB.")
    p.add_argument("--fresh",action="store_true",help="Remeasure archive references; keep this output's own measured jobs.")
    p.add_argument("--low-memory",action="store_true",default=True,help="Use trace-free hybrid encoding (always enabled).")
    p.add_argument("--output",type=Path,help="Output directory (default results/fig419_{full|HxW}).")
    run=p.add_mutually_exclusive_group()
    run.add_argument("--dry-run",action="store_true",help="Check inputs, print legal grid and exact planned/reused/new counts; do not encode.")
    run.add_argument("--plot-only",action="store_true",help="Regenerate figures from the specified --output; needs no input files or model.")
    args=p.parse_args()
    args.crop=args.crop or [256,256]
    args.output=(args.output or Path("results")/("fig419_full" if args.full_image else "fig419_%sx%s"%tuple(args.crop))).resolve()
    if args.plot_only:
        return args
    if len(args.images)!=len(set(args.images)):p.error("Each scene should be selected once.")
    if min(args.crop)<2:p.error("Use a crop at least 2 x 2.")
    if (args.row is None)!=(args.col is None):p.error("Supply both --row and --col.")
    if args.full_image and (args.row is not None or args.origin):p.error("A full image cannot have a crop origin.")
    if args.row is not None and min(args.row,args.col)<0:p.error("Origins are zero-based and non-negative.")
    args.origins={}
    for name,row,col in args.origin:
        if name not in args.images or name in args.origins:p.error("--origin must name a selected image only once.")
        try: r,c=int(row),int(col)
        except ValueError:p.error("--origin ROW COL must be integers.")
        if min(r,c)<0:p.error("Negative --origin.")
        args.origins[name]=(r,c)
    args.omega_values=sorted(set(args.omega_values));args.a_values=sorted(set(args.a_values))
    if not all(4<=k<=19 for k in args.omega_values):p.error("Ω must be in [4, 19].")
    if not all(0<=k<=15 for k in args.a_values):p.error("A must be in [0, 15] DN.")
    if not 0<=args.p<=15:p.error("P must be in [0, 15].")
    if args.p==0 and args.prediction_mode=="reduced":p.error("Use P>=1 with the reduced-mode verification runner.")
    if args.workers<1 or any(not math.isfinite(n) or n<=0 for n in (args.max_memory_gb,args.total_memory_gb)):
        p.error("Worker count and memory budgets must be positive and finite.")
    args.nu_min,args.nu_max,args.t_inc,args.phi,args.psi=-1,3,2048,3,7
    args.memory_poll_ms=150
    return args


def build_study(args,repo):
    grids={o:register_grid(o,args.p,args.prediction_mode,args.r_values,args.dense) for o in args.omega_values}
    profile={k:getattr(args,k) for k in ("p","nu_min","nu_max","t_inc","phi","psi","local_sum","prediction_mode")}
    profile.update(D=D,Theta=3,encoder="HYBRID",psi_lossless=0,weight_initialization="DEFAULT",weight_exponent_offsets="ALL_ZERO")
    inputs={};jobs=[]
    for name in args.images:
        path=core.find_capture(args.input_dir,core.SCENES[name])
        s=inputs[name]=engine.inspect_input(path,args);sy,sx,nz=s["source_shape_yxz"]
        h,w=(sy,sx) if args.full_image else args.crop
        if args.full_image: r,c=0,0
        elif name in args.origins:r,c=args.origins[name]
        elif args.row is not None:r,c=args.row,args.col
        else:r,c=(sy-h)//2,(sx-w)//2
        if h<2 or w<2 or min(r,c)<0 or r+h>sy or c+w>sx:
            raise ValueError(f"{name}: crop origin ({r},{c}), size {h}×{w}, outside native shape {sy}×{sx} or too small.")
        print(f"  {name}: native {sy}×{sx}×{nz}; using rows [{r}:{r+h}), columns [{c}:{c+w}), all {nz} bands.")
        for a in args.a_values:
            for o,(lo,star,widths) in grids.items():
                for width in [64]+[v for v in widths if v!=64]:
                    jobs.append(dict(job_id=f"{name}_y{r}x{c}_{h}x{w}_a{a}_om{o}_r{width}",
                        image=name,size=f"{h}x{w}",repeat=1,is_full_image=int(args.full_image),rows=h,columns=w,bands=nz,
                        crop_row=r,crop_col=c,omega=o,a=a,register_bits=width,r_min=lo,r_star=star,
                        t_inc=args.t_inc,nu_min=args.nu_min,phi=args.phi,dimension_issue="",
                        estimated_ram_gib=engine.estimate_gib((h,w,nz),args.p,args.low_memory)))
    files=list((repo/"ccsds123_i2_hlm").rglob("*.py"))+list((repo/"ccsds123_i2_hlm").rglob("*.txt"))
    model_digest=core.file_digest(files,repo)
    guard=core.file_digest(list(HERE.glob("*.py")),HERE)
    stable=dict(tool="register_sweep",version=1,model_sha256=model_digest,runner_sha256=guard,
                profile=profile,images=args.images,omega_values=args.omega_values,a_values=args.a_values,
                inputs={k:{f:v[f] for f in ("file_sha256","source_shape_yxz","variable","axis_order")} for k,v in inputs.items()},
                jobs=jobs)
    ident=hashlib.sha256(json.dumps(stable,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    return {"study_id":ident,"study":dict(stable,inputs=inputs)}


def imported_references(metadata):
    """Only a matching file, variable, axis order, region, model and complete codec profile can be reused."""
    archive=json.loads((HERE/"archived_r64.json").read_text(encoding="utf-8"))
    s=metadata["study"]
    if archive["model_sha256"]!=s["model_sha256"] or archive["profile"]!=dict(s["profile"],R=64):return {}
    fields=("image","crop_row","crop_col","rows","columns","bands","a","omega")
    def key(row):return tuple(row[k] for k in fields)
    bykey={key(r):r for r in archive["rows"]}
    output={}
    for j in s["jobs"]:
        if j["register_bits"]!=64:continue
        source=archive["inputs"].get(j["image"]);actual=s["inputs"][j["image"]]
        if not source or any(actual[f]!=x for f,x in source.items()):continue
        old=bykey.get(key(j))
        if not old:continue
        row={k:v for k,v in old.items() if k in engine.FIELDS}
        n=old["sample_count"];bits=old["stream_bits"]
        if (bits!=sum(old[k] for k in ("header_bits","payload_bits","padding_bits"))
            or n!=j["rows"]*j["columns"]*j["bands"] or old["max_error"]>j["a"]):
            raise ValueError("Invalid built-in historical reference "+old["source_job"])
        row.update(job_id=j["job_id"],size=j["size"],repeat=1,is_full_image=j["is_full_image"],spatial_samples=j["rows"]*j["columns"],
                   bits_per_sample=bits/n,bits_per_sample_with_external_info=(bits+old["external_info_bits"])/n,
                   compression_ratio=D*n/bits,register_bits=64,r_min=j["r_min"],r_star=j["r_star"],
                   result_source="archived:"+old["source_study"]+"/"+old["source_job"],wrap_count="",modulo_operations="")
        output[j["job_id"]]=dict(study_id=metadata["study_id"],row=row,progression=[],
                            note="Previously measured R=64; wrap instrumentation was not present in the archived study.")
    return output


def preflight(out,metadata,fresh):
    mpath=out/"metadata.json"
    if mpath.exists():
        if json.loads(mpath.read_text(encoding="utf-8")).get("study_id")!=metadata["study_id"]:
            raise ValueError(f"{out} belongs to a different study (grid, files, or code changed). Choose a different --output.")
    elif out.is_dir() and any(x.name!=".sweep.lock" for x in out.iterdir()):
        raise ValueError(f"Output directory {out} is not empty and has no matching study metadata.")
    archive={} if fresh else imported_references(metadata)
    saved=set(); stale=[]
    for j in metadata["study"]["jobs"]:
        key=j["job_id"];path=out/"jobs"/(key+".json")
        if path.exists():
            data=json.loads(path.read_text(encoding="utf-8"))
            if data.get("study_id")!=metadata["study_id"] or data["row"]["job_id"]!=key:
                raise ValueError("Unmatched checkpoint: "+str(path))
            if fresh and data["row"]["result_source"].startswith("archived:"):stale.append(path)
            else:saved.add(key)
    return saved,{k:v for k,v in archive.items() if k not in saved},stale


def main():
    args=arguments();out=args.output
    if args.plot_only:
        if not (out/"metadata.json").is_file():raise ValueError("No completed/planned run under --output.")
        with output_lock(out):
            m=json.loads((out/"metadata.json").read_text(encoding="utf-8"))
            make_report(engine.regenerate(out,m),m,out)
        print("Updated register_comparison.csv and plots in",out)
        return
    repo=engine.find_repo(args.repo)
    metadata=build_study(args,repo);s=metadata["study"];jobs=s["jobs"]
    for o in s["omega_values"]:
        same=[j for j in jobs if j["omega"]==o]
        print(f"  Ω={o}: legal R≥{same[0]['r_min']}; no-overflow R*={same[0]['r_star']}; testing {sorted({j['register_bits'] for j in same})}")
    def status(saved,archive):
        print(f"{len(jobs)} configuration points = {len(s['images'])} scene(s) × {len(s['a_values'])} A level(s) × "
              f"{len(jobs)//(len(s['images'])*len(s['a_values']))} legal (Ω,R) pairs. "
              f"{len(saved)} already saved; {len(archive)} reusable archived R=64; {len(jobs)-len(saved)-len(archive)} new compressions.",
              flush=True)
    if args.dry_run:
        saved,archive,_=preflight(out,metadata,args.fresh);status(saved,archive)
        return
    with output_lock(out):
        saved,archive,stale=preflight(out,metadata,args.fresh);status(saved,archive)
        (out/"jobs").mkdir(exist_ok=True)
        core.write_json(out/"metadata.json",metadata)
        engine.table(out/"plan.csv",list(jobs[0]),jobs)
        for path in stale:path.unlink()
        for k,data in archive.items():core.write_json(out/"jobs"/(k+".json"),data)
        engine.run_jobs(args,metadata,repo,out,make_report)
        final=json.loads((out/"last_execution.json").read_text(encoding="utf-8"))
        if final["completed_total"]!=len(jobs):
            raise SystemExit(f"Incomplete: {final['completed_total']}/{len(jobs)} points. See failures.csv, skipped.csv, and jobs/*.log. Rerun to resume.")
        print("Finished. Figure: "+str(out/"fig419_register_rate.png"))


if __name__=="__main__":
    try:main()
    except (ValueError, OSError, ImportError) as exc:raise SystemExit(f"ERROR: {exc}")
''',
'_register_report.py': r'''"""Reduce completed points using a matched R=64 reference; render the two plots."""
from __future__ import annotations

import math
from pathlib import Path

import _register_engine as engine


def make_report(records, metadata, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    study = metadata["study"]
    jobs = study["jobs"]
    rows = {r["row"]["job_id"]: r["row"] for r in records}
    def key(j):
        return (j["image"], int(j["crop_row"]), int(j["crop_col"]),
                int(j["rows"]), int(j["columns"]), int(j["bands"]), int(j["a"]), int(j["omega"]))
    ref = {key(r): r for r in rows.values() if int(r["register_bits"]) == 64}
    points = []
    for j in jobs:
        entry = {k: j[k] for k in ("job_id", "image", "crop_row", "crop_col", "rows", "columns",
                 "bands", "a", "omega", "register_bits", "r_min", "r_star")}
        r = rows.get(j["job_id"])
        reference = ref.get(key(j))
        entry.update(status="pending", result_source="", bits_per_sample="", ref64_bits_per_sample="",
                     delta_bits="", rate_increase_percent="", wrap_count="", modulo_operations="",
                     payload_matches_r64="", max_error_dn="", ref64_source="")
        if r:
            entry.update(status="no R64 yet", result_source=r["result_source"], bits_per_sample=r["bits_per_sample"],
                         wrap_count=r.get("wrap_count",""), modulo_operations=r.get("modulo_operations",""),
                         max_error_dn=r["max_error"])
            if int(r["max_error"]) > int(j["a"]):
                raise ValueError("Reconstruction error exceeds A for "+j["job_id"])
            if reference:
                if r["crop_sha256"] != reference["crop_sha256"]:
                    raise ValueError("Reference used different samples for "+j["job_id"])
                bits, b64 = int(r["stream_bits"]), int(reference["stream_bits"])
                same = (r["payload_sha256"] == reference["payload_sha256"]
                        and int(r["payload_bits"]) == int(reference["payload_bits"]))
                if ((j["register_bits"] >= j["r_star"] or r.get("wrap_count") == 0)
                        and (not same or bits != b64)):
                    raise ValueError("Expected identical payload and length for "+j["job_id"]+
                                     "; check the model and R64 reference. To remeasure, use a new output with --fresh.")
                entry.update(status="complete", ref64_bits_per_sample=reference["bits_per_sample"],
                             delta_bits=bits-b64, rate_increase_percent=100*(bits-b64)/b64,
                             payload_matches_r64=int(same), ref64_source=reference["result_source"])
        points.append(entry)
    engine.table(out/"register_comparison.csv", list(points[0]), points)
    if not rows:
        return
    images, omegas, avals = study["images"], study["omega_values"], study["a_values"]
    colors = list(plt.get_cmap("tab10").colors); markers = ["o","s","D","^","v","P","X","*"]
    for metric, ylabel, filename in (
            ("rate_increase_percent","Rate change vs. R=64 (%)","fig419_register_rate"),
            ("bits_per_sample","Compressed bits/sample","fig419_absolute_rate")):
        fig, axes = plt.subplots(len(images),len(omegas),squeeze=False,
                                 figsize=(4.25*len(omegas)+0.8,2.6*len(images)+1.35), layout="constrained")
        for i,image in enumerate(images):
            for col,omega in enumerate(omegas):
                ax=axes[i,col]
                pp=[p for p in points if p["image"]==image and p["omega"]==omega]
                widths=sorted({p["register_bits"] for p in pp}); star=pp[0]["r_star"]
                ax.axvspan(max(min(widths)-.6,star),64.5,color="#e2ebdf",zorder=0)
                if min(widths)-1 <= star <= 64:
                    ax.axvline(star,color="#51724c",lw=.85,ls=":",zorder=1)
                if metric=="rate_increase_percent": ax.axhline(0,color=".45",lw=.7)
                for k,a in enumerate(avals):
                    line=sorted((p for p in pp if p["a"]==a),key=lambda p:p["register_bits"])
                    ax.plot([p["register_bits"] for p in line],
                            [math.nan if p[metric]=="" else float(p[metric]) for p in line],
                            ls="-" if k%2==0 else "--",marker=markers[k%len(markers)],ms=6 if k%2==0 else 4,
                            mfc="none" if k%2==0 else colors[k%len(colors)],color=colors[k%len(colors)],lw=1.25)
                ax.set_title(f"{image.title()} · Ω={omega}  |  R*={star}\n"
                             f"{pp[0]['rows']} × {pp[0]['columns']} pixels",fontsize=10)
                ax.set_xlim(min(widths)-.9,64.9)
                if len(widths)<=9: ax.set_xticks(widths)
                ax.grid(axis="y",alpha=.24)
                ax.ticklabel_format(axis="y",style="plain",useOffset=False)
                if i==len(images)-1: ax.set_xlabel("Register size R (bits)")
                if col==0: ax.set_ylabel(ylabel)
        complete=sum(p[metric]!="" for p in points)
        fig.suptitle(f"HYPSO register size · {complete}/{len(points)} points\n"
                     f"R=64 reference per scene, Ω and A · D=16 · P={study['profile']['p']} · "
                     f"{study['profile']['prediction_mode']} prediction · hybrid encoder",fontsize=13)
        handles=[Line2D([],[],color=colors[k%len(colors)],marker=markers[k%len(markers)],lw=1.25,
                        ls="-" if k%2==0 else "--",label=f"A={a} DN") for k,a in enumerate(avals)]
        handles.append(Patch(facecolor="#e2ebdf",label="R ≥ R*: overflow impossible"))
        fig.legend(handles=handles,loc="outside lower center",ncol=min(6,len(handles)),frameon=False,fontsize=9)
        for ext in ("png","pdf"):
            fig.savefig(out/(filename+"."+ext),dpi=165,bbox_inches="tight")
        plt.close(fig)
''',
'_register_engine.py': r'''#!/usr/bin/env python3
"""Execution engine for the Figure 4-19 study. Use register_sweep.py.

Each worker runs one complete configuration. Codec maths are unchanged.
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
import signal
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

import numpy as np
import psutil
import _damping_core as core
import sweep_workers as sw

VERSION = "1.0.0-register-config-workers"
GIB = 1024**3
ALIASES = {"lines":"y", "line":"y", "frames":"y", "frame":"y", "rows":"y", "y":"y",
           "samples":"x", "sample":"x", "columns":"x", "cols":"x", "x":"x",
           "bands":"z", "band":"z", "wavelength":"z", "wavelengths":"z", "z":"z"}
EXTRA_FIELDS = ["omega", "job_id", "size", "repeat", "is_full_image", "spatial_samples", "crop_row", "crop_col",
                "crop_sha256", "data_load_seconds", "model_seconds", "codec_and_checks_seconds", "wall_seconds",
                "peak_rss_bytes", "peak_rss_gib", "peak_rss_method", "sampled_peak_rss_bytes",
                "workers_requested", "peak_concurrent_workers", "native_threads", "estimated_ram_gib",
                "schedule_verified", "first_update_at_vmax", "adaptation_verification_seconds",
                "predictor_backend", "predictor_pool_size", "predictor_processes_used",
                "predictor_peak_tasks_in_flight", "predictor_tasks", "rows_per_task", "shared_workspace_bytes",
                "sampled_tree_peak_rss_gib", "sampled_tree_peak_pss_gib", "tree_memory_method",
                "concurrent_crops"]
FIELDS = core.FIELDS + EXTRA_FIELDS + ["register_bits", "r_min", "r_star",
    "wrap_count", "modulo_operations", "result_source"]


def count_wraps(pred, width, omega):
    """Observe the *pre*-modulo quantity (Green Book eq. 20), not clipped output.

    For D=16, P<=15, Omega<=19 this sum fits signed int64 even for R=64.
    At t=0 the model bypasses this calculation for every band.
    Limit scratch arrays to about one million samples at a time.
    """
    if pred.register_size != width or pred.weight_component_resolution != omega:
        raise ValueError("The actual predictor used a different R or Omega.")
    delta = pred.predicted_central_local_difference.reshape(-1)
    sigma = pred.local_sum.reshape(-1)
    nz = pred.header.z_size
    lower, upper = -(1 << (width - 1)), (1 << (width - 1)) - 1
    total = 0
    for i in range(nz, sigma.size, 1 << 20):
        sl = slice(i, i + (1 << 20))
        value = (sigma[sl] - 4 * int(pred.image_constants.middle_sample_value)) * (1 << omega)
        value += delta[sl]
        total += int(np.count_nonzero(value < lower)) + int(np.count_nonzero(value > upper))
    return total, int(sigma.size - nz)




def find_repo(explicit):
    for root in ([explicit] if explicit else [Path.cwd(),*Path(__file__).resolve().parents]):
        if root and (root/"ccsds123_i2_hlm/header.py").is_file():return root.resolve()
    raise ValueError("Cannot find verification model; run from its root or pass --repo PATH.")


def sha_file(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for data in iter(lambda:f.read(8*1024**2),b""):h.update(data)
    return h.hexdigest()


def inspect_input(path,args):
    with core.open_netcdf(path) as ds:
        try:var=ds[args.variable]
        except (KeyError,IndexError) as exc:raise ValueError(f"Missing {args.variable} in {path}.") from exc
        order=args.axis_order or "".join(ALIASES.get(d.lower(),"?") for d in var.dimensions)
        if var.ndim!=3 or np.dtype(var.dtype).kind not in "iu" or len(order)!=3 or set(order)!=set("xyz"):
            raise ValueError(f"Expected 3-D integer DN; dimensions {var.dimensions}. Use --axis-order if needed.")
        if float(getattr(var,"scale_factor",1))!=1 or float(getattr(var,"add_offset",0))!=0:
            raise ValueError("Use unscaled raw integer DN, not packed/calibrated floating-point data.")
        shape=[int(var.shape[order.index(a)]) for a in "yxz"]
        if min(shape)<1 or max(shape)>65535:raise ValueError("This runner supports dimensions 1..65535.")
        info={"file":str(path.resolve()),"source_shape_yxz":shape,"variable":args.variable,
              "axis_order":order,"storage_dimensions":list(var.dimensions),"storage_dtype":str(var.dtype)}
    stat=path.stat()
    info.update(file_bytes=stat.st_size,file_mtime_ns=stat.st_mtime_ns,file_sha256=sha_file(path))
    return info


def estimate_gib(shape,p,low):
    count=math.prod(shape)
    return count*(320+16*(p+3))/GIB+0.5 if low else count*(2200+16*(p+3))/GIB+0.15


def schedule_segments(rows,columns,nmin,tinc,nmax,D=16,Omega=19):
    """Actual updates at spatial t=2..N-1 use rho[t-1]. No factor of band count."""
    N=rows*columns
    segments=[];j=1
    while j<N-1:
        nu=max(nmin,min(nmax,nmin+(j-columns)//tinc))
        end=min(N-1,columns+(nu-nmin+1)*tinc) if nu<nmax else N-1
        assert end>j
        segments.append({"spatial_index_start":j+1,"spatial_index_end_inclusive":end,
                         "updates_per_band":end-j,"nu":nu,"rho":nu+D-Omega})
        j=end
    return segments




def os_peak_rss():
    if sys.platform.startswith("linux") or sys.platform=="darwin":
        import resource
        peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(peak*(1 if sys.platform=="darwin" else 1024)),"OS ru_maxrss (fresh process)"
    mem=psutil.Process().memory_info()
    for attr in ("peak_rss","peak_wset"):
        if hasattr(mem,attr):return int(getattr(mem,attr)),f"OS {attr} (fresh process)"
    return int(mem.rss),"sampled RSS lower bound (no OS high-water API)"


def worker(spec_path):
    """Internal child entry point: input arrays never live in the coordinator."""
    spec=json.loads(Path(spec_path).read_text(encoding="utf-8"))
    args=SimpleNamespace(**spec["args"]);job=spec["job"];source=spec["input"]
    signal.signal(signal.SIGINT,signal.SIG_IGN)
    model=sw.load_model(Path(spec["repo"]),"vmax")
    if args.low_memory:
        from hybrid_no_traces import validate_reference
        validate_reference()
    path=Path(source["file"])
    stat=path.stat()
    if (stat.st_size,stat.st_mtime_ns)!=(source["file_bytes"],source["file_mtime_ns"]):
        raise ValueError("Input file changed after planning; restart with a consistent source.")
    args.full_image=bool(job["is_full_image"]);args.crop=(job["rows"],job["columns"])
    args.row,args.col=job["crop_row"],job["crop_col"]
    args.axis_order=source["axis_order"];args.t_inc=job["t_inc"]
    args.phi_values=[args.phi]  # args.a_values retains the complete study grid.
    start=time.perf_counter()
    cube,info=core.read_cube(path,args)
    load_seconds=time.perf_counter()-start
    if list(cube.shape)!=[job["rows"],job["columns"],job["bands"]]:raise ValueError("Input dimensions changed.")
    h=core.make_header(cube,args,args.a,args.nu_max,model[0])
    h.weight_component_resolution = args.omega - 4
    h.register_size = job["register_bits"] % 64  # the 6-bit header encodes 64 as zero
    h.check_legal_config()
    segments=schedule_segments(h.y_size,h.x_size,args.nu_min,args.t_inc,args.nu_max,
                               D=h.get_dynamic_range_bits(),Omega=args.omega)
    verification_seconds=0.0
    statistics={"predictor_backend":"original-serial", "predictor_pool_size":1,
                "predictor_processes_used":1, "predictor_worker_pids":[os.getpid()],
                "predictor_peak_tasks_in_flight":1, "predictor_tasks":1,
                "rows_per_task":0, "shared_workspace_bytes":0}
    # No nested process pool: this child owns prediction AND entropy encoding.
    wraps = {}
    class ObservedPredictor(model[2].Predictor):
        def run_predictor(self):
            nonlocal verification_seconds
            super().run_predictor()
            begin=time.perf_counter()
            wraps["wrap_count"], wraps["modulo_operations"] = count_wraps(self, job["register_bits"], args.omega)
            if job["register_bits"] >= job["r_star"] and wraps["wrap_count"]:
                raise RuntimeError("Observed wrap at or above the theoretical no-overflow bound.")
            for seg in segments:
                # Applied update at t uses exponent stored at t-1.
                j0=seg["spatial_index_start"]-1;j1=seg["spatial_index_end_inclusive"]
                if not np.all(self.weight_update_scaling_exponent[j0:j1]==seg["rho"]):
                    raise RuntimeError("Actual model adaptation schedule differs from the reported progression.")
            verification_seconds=time.perf_counter()-begin
    observed=(model[0],model[1],SimpleNamespace(Predictor=ObservedPredictor),model[3])
    begin=time.perf_counter()
    row=core.encode_one(cube,h,args,args.a,args.nu_max,args.phi,observed)
    row.update(wraps,register_bits=job["register_bits"],r_min=job["r_min"],
               r_star=job["r_star"],result_source="measured")
    total_codec=time.perf_counter()-begin
    row.update({k:v for k,v in statistics.items() if k!="predictor_worker_pids"})
    # Remove the observer scan from predictor time; total wall time retains all work.
    row["predictor_seconds"]=max(0.0,row["predictor_seconds"]-verification_seconds)
    peak,method=os_peak_rss()
    cap=[s for s in segments if s["nu"]==args.nu_max]
    row.update(omega=args.omega,image=job["image"],capture=path.name,job_id=job["job_id"],size=job["size"],repeat=job["repeat"],
               is_full_image=job["is_full_image"],spatial_samples=h.y_size*h.x_size,
               crop_row=job["crop_row"],crop_col=job["crop_col"],crop_sha256=info["crop_sha256"],
               data_load_seconds=load_seconds,model_seconds=row["predictor_seconds"]+row["encoder_seconds"],
               codec_and_checks_seconds=total_codec,peak_rss_bytes=peak,peak_rss_gib=peak/GIB,
               peak_rss_method=method+"; complete configuration worker",native_threads=1,estimated_ram_gib=spec["estimated_ram_gib"],
               schedule_verified=1,first_update_at_vmax=cap[0]["spatial_index_start"] if cap else "",
               adaptation_verification_seconds=verification_seconds)
    core.write_json(Path(spec["worker_result"]),{"row":row,"crop":info,"progression":segments,
                    "header":core.header_snapshot(h),"parallel_execution":statistics})


def table(path,fields,rows):
    tmp=path.with_suffix(path.suffix+".tmp")
    with tmp.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    os.replace(tmp,path)


def regenerate(out,metadata):
    records=[]
    for job in metadata["study"]["jobs"]:
        path=out/"jobs"/(job["job_id"]+".json")
        if path.exists():
            record=json.loads(path.read_text(encoding="utf-8"))
            if record.get("study_id")!=metadata["study_id"]:raise ValueError("Saved result identity differs.")
            if record["row"]["job_id"]!=job["job_id"]:raise ValueError("Saved job identity differs.")
            records.append(record)
    table(out/"results.csv",FIELDS,[r["row"] for r in records])
    progression=[]
    for r in records:
        row=r["row"]
        for seg in r["progression"]:
            progression.append({k:row[k] for k in ("job_id","image","size","omega","t_inc","nu_min","phi","a","repeat")} | seg)
    table(out/"adaptation_progression.csv",["job_id","image","size","omega","t_inc","nu_min","phi","a","repeat","spatial_index_start",
          "spatial_index_end_inclusive","updates_per_band","nu","rho"],progression)
    return records




def execution_budget(args):
    return min(args.total_memory_gb,sw._available_memory())


def allowed_configuration_workers(requested):
    limit=requested
    try:limit=min(limit,len(os.sched_getaffinity(0)))
    except AttributeError:limit=min(limit,os.cpu_count() or 1)
    allocation=os.environ.get("SLURM_CPUS_PER_TASK","")
    if allocation.isdecimal():limit=min(limit,max(1,int(allocation)))
    return max(1,limit)


def sample_process_tree(root,include_pss=False):
    """RSS sum double-counts shared mappings; Linux PSS apportions them."""
    try:processes=[root,*root.children(recursive=True)]
    except (psutil.NoSuchProcess,psutil.AccessDenied):return 0,None,[]
    rss=0;pss=0;pss_available=include_pss and sys.platform.startswith("linux")
    for process in processes:
        try:
            rss+=process.memory_info().rss
            if pss_available:
                value=getattr(process.memory_full_info(),"pss",None)
                if value is None:pss_available=False
                else:pss+=value
        except (psutil.NoSuchProcess,psutil.AccessDenied):
            pss_available=False
    return rss,pss if pss_available else None,processes


def terminate_job(item):
    """Stop this configuration process and any unexpected descendants."""
    proc=item["proc"]
    if os.name=="posix":
        try:os.killpg(proc.pid,signal.SIGTERM)
        except ProcessLookupError:pass
        try:proc.wait(timeout=5)
        except subprocess.TimeoutExpired:pass
        # Parent exit alone does not prove its children have exited.
        try:os.killpg(proc.pid,signal.SIGKILL)
        except ProcessLookupError:pass
        proc.wait()
    else:
        known=list(item.get("known_processes",{}).values())
        try:known.extend(item["ps"].children(recursive=True));known.append(item["ps"])
        except psutil.NoSuchProcess:pass
        for p in known:
            try:p.terminate()
            except psutil.NoSuchProcess:pass
        _,alive=psutil.wait_procs(known,timeout=5)
        for p in alive:
            try:p.kill()
            except psutil.NoSuchProcess:pass
        psutil.wait_procs(alive,timeout=5)
        proc.wait()


def run_jobs(args,metadata,repo,out,plot_results):
    jobs=metadata["study"]["jobs"];inputs=metadata["study"]["inputs"]
    records=regenerate(out,metadata);done={r["row"]["job_id"] for r in records}
    pending=[j for j in jobs if j["job_id"] not in done]
    skipped=[];failed=[];active={};aggregate_peak=0;run_start=time.perf_counter()
    diagnostic_fields=["job_id","image","size","omega","register_bits","t_inc","nu_min","phi","a","repeat","reason"]
    cpu_limit=allowed_configuration_workers(args.workers)
    limit=min(cpu_limit,len(pending))
    peak_concurrent=0
    waiting_for_memory=False
    print(f"Completed {len(done)}/{len(jobs)}; ONE COMPLETE configuration per worker; "
          f"requested {args.workers}, CPU allowance {cpu_limit}, pending-work limit {limit}. "
          "RAM budget may reduce concurrency.",flush=True)
    child_args={k:getattr(args,k) for k in ("p","nu_max","psi","a_values","local_sum","prediction_mode",
                                         "low_memory","max_memory_gb","variable")}
    try:
        while pending or active:
            while pending and len(active)<limit:
                job=pending[0]
                need=job["estimated_ram_gib"]
                reserved=sum(max(item["job"]["estimated_ram_gib"],item["last_rss"]/GIB) for item in active.values())
                reason=job["dimension_issue"]
                if not reason and need>args.max_memory_gb:reason=f"Estimated {need:.2f} GiB exceeds per-job limit {args.max_memory_gb:.2f}."
                # Active workers already consume some available RAM. Reserve their remaining allowance.
                remaining_reserve=sum(max(0,item["job"]["estimated_ram_gib"]-item["last_rss"]/GIB) for item in active.values())
                enough=need+.35<=execution_budget(args)-remaining_reserve and need+reserved+.35<=args.total_memory_gb
                if not reason and not enough:
                    if active:
                        if not waiting_for_memory:
                            print(f"RAM budget: keeping {len(active)} configuration worker(s) active; "
                                  "remaining jobs wait for memory.",flush=True)
                        waiting_for_memory=True
                        break
                    reason=f"Estimated {need:.2f} GiB + coordinator allowance exceeds current budget {execution_budget(args):.2f} GiB."
                pending.pop(0)
                if reason:
                    skipped.append({k:job[k] for k in diagnostic_fields[:-1]}|{"reason":reason})
                    table(out/"skipped.csv",diagnostic_fields,skipped)
                    print(f"SKIP {job['image']} {job['size']} Omega={job['omega']}, R={job['register_bits']}, A={job['a']}: {reason}",flush=True)
                    continue
                stem=out/"jobs"/job["job_id"]
                result=stem.with_suffix(".worker.json");result.unlink(missing_ok=True)
                spec={"args":child_args | {k:job[k] for k in ("nu_min","phi","a","omega")},"job":job,"input":inputs[job["image"]],"repo":str(repo),
                      "worker_result":str(result),"estimated_ram_gib":need}
                spec_path=stem.with_suffix(".spec.json");log=None
                try:
                    core.write_json(spec_path,spec)
                    log=stem.with_suffix(".log").open("w",encoding="utf-8")
                    env=dict(os.environ)
                    for name in sw._NATIVE_THREADS:env[name]="1"
                    begin=time.perf_counter()
                    proc=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),"--_worker",str(spec_path)],
                                          stdout=log,stderr=subprocess.STDOUT,env=env,cwd=repo,start_new_session=os.name=="posix")
                except BaseException:
                    if log is not None:log.close()
                    raise
                active[job["job_id"]]={"job":job,"proc":proc,"ps":psutil.Process(proc.pid),"log":log,
                                      "start":begin,"peak":0,"last_rss":0,"peak_concurrent":1,"result":result,
                                      "tree_rss_peak":0,"tree_pss_peak":None,
                                      "pss_last_time":0,"known_processes":{}}
                waiting_for_memory=False
                peak_concurrent=max(peak_concurrent,len(active))
                for item in active.values():item["peak_concurrent"]=max(item["peak_concurrent"],len(active))
                print(f"START {job['image']} {job['rows']}x{job['columns']}x{job['bands']} "
                      f"({'full' if job['is_full_image'] else 'crop'}), Omega={job['omega']}, R={job['register_bits']}, A={job['a']}, "
                      f"configuration worker PID={proc.pid}, active={len(active)}/{limit}, estimate={need:.2f} GiB",flush=True)
            rss_sum=psutil.Process().memory_info().rss
            for key,item in list(active.items()):
                try:rss=item["ps"].memory_info().rss
                except (psutil.NoSuchProcess,psutil.AccessDenied):rss=0
                item["last_rss"]=rss;item["peak"]=max(item["peak"],rss)
                now=time.monotonic();sample_pss=now-item["pss_last_time"]>=0.5
                tree_rss,tree_pss,processes=sample_process_tree(item["ps"],sample_pss)
                for process in processes:item["known_processes"][process.pid]=process
                item["tree_rss_peak"]=max(item["tree_rss_peak"],tree_rss);rss_sum+=tree_rss
                if sample_pss:item["pss_last_time"]=now
                if tree_pss is not None:item["tree_pss_peak"]=max(item["tree_pss_peak"] or 0,tree_pss)
                code=item["proc"].poll()
                if code is None:continue
                elapsed=time.perf_counter()-item["start"];item["log"].close();job=item["job"]
                terminate_job(item)
                del active[key]
                waiting_for_memory=False
                if code!=0 or not item["result"].exists():
                    reason=f"Worker exited with code {code}; inspect jobs/{key}.log (possible memory/process failure)."
                    failed.append({k:job[k] for k in diagnostic_fields[:-1]}|{"reason":reason})
                    table(out/"failures.csv",diagnostic_fields,failed)
                    print(f"FAILED {job['image']} {job['size']} Omega={job['omega']}, R={job['register_bits']}, A={job['a']}: {reason}",flush=True)
                    continue
                record=json.loads(item["result"].read_text(encoding="utf-8"));row=record["row"]
                if row["job_id"]!=key:raise ValueError("Worker returned the wrong job.")
                if not row["peak_rss_method"].startswith("OS"):
                    row["peak_rss_bytes"]=max(row["peak_rss_bytes"],item["peak"])
                    row["peak_rss_gib"]=row["peak_rss_bytes"]/GIB
                row.update(wall_seconds=elapsed,sampled_peak_rss_bytes=item["peak"],workers_requested=args.workers,
                           peak_concurrent_workers=item["peak_concurrent"],concurrent_crops=item["peak_concurrent"],
                           sampled_tree_peak_rss_gib=item["tree_rss_peak"]/GIB,
                           sampled_tree_peak_pss_gib=item["tree_pss_peak"]/GIB if item["tree_pss_peak"] is not None else "",
                           tree_memory_method="Sampled complete configuration process: RSS includes shared library pages; Linux PSS apportions sharing. Sampling can miss peaks.")
                record["study_id"]=metadata["study_id"]
                core.write_json(out/"jobs"/(key+".json"),record)
                item["result"].unlink(missing_ok=True)
                done.add(key);records=regenerate(out,metadata)
                print(f"[{len(done)}/{len(jobs)}] {job['image']} {job['size']} Omega={job['omega']}, R={job['register_bits']}, A={job['a']}, wraps={row['wrap_count']}: "
                      f"{row['bits_per_sample']:.6f} bits/sample; model={row['model_seconds']:.2f}s; "
                      f"predictor processes used={row['predictor_processes_used']}; nu_max reached={bool(row['nu_max_reached'])}",flush=True)
            aggregate_peak=max(aggregate_peak,rss_sum)
            if active:time.sleep(args.memory_poll_ms/1000)
    finally:
        for item in active.values():
            terminate_job(item)
            item["log"].close()
        records=regenerate(out,metadata)
        table(out/"skipped.csv",diagnostic_fields,skipped)
        table(out/"failures.csv",diagnostic_fields,failed)
        core.write_json(out/"last_execution.json",{"completed_total":len(records),"requested_total":len(jobs),
                        "skipped_this_invocation":len(skipped),"failed_this_invocation":len(failed),
                        "wall_seconds":time.perf_counter()-run_start,"sampled_aggregate_peak_rss_gib":aggregate_peak/GIB,
                        "memory_note":"Sampled sum of coordinator and configuration worker RSS; shared library pages can be counted repeatedly. See per-job PSS on Linux.",
                        "execution_mode":"one-complete-configuration-per-worker",
                        "workers_requested":args.workers,"allowed_configuration_workers":cpu_limit,
                        "peak_concurrent_workers":peak_concurrent,"predictor_workers_per_configuration":1,
                        "concurrent_crops":peak_concurrent,"memory_poll_ms":args.memory_poll_ms})
        plot_results(records,metadata,out)
    print(f"Complete {len(records)}/{len(jobs)}; skipped {len(skipped)}; failed {len(failed)}. Results: {out}")
    return 1 if failed or len(records)!=len(jobs) else 0





if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--_worker":
        try:
            worker(sys.argv[2])
        except BaseException:
            traceback.print_exc()
            raise SystemExit(1)
    else:
        raise SystemExit("Run register_sweep.py, not this internal worker module.")
''',
'_damping_core.py': r'''#!/usr/bin/env python3
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
''',
'sweep_workers.py': r'''"""Process scheduling for the HYPSO sweep tools. No codec math lives here."""
from __future__ import annotations

import argparse
from contextlib import closing, contextmanager, redirect_stdout
import hashlib
import io
import math
import multiprocessing as mp
from multiprocessing.connection import wait
import os
from pathlib import Path
import signal
import sys
import tempfile
import time
import traceback

_GIB = 1024 ** 3
_NATIVE_THREADS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                   "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")


def add_options(parser):
    def positive_int(text):
        n = int(text)
        if n < 1:
            raise argparse.ArgumentTypeError("must be at least one")
        return n

    def positive_float(text):
        n = float(text)
        if not math.isfinite(n) or n <= 0:
            raise argparse.ArgumentTypeError("must be positive and finite")
        return n

    parser.add_argument("--workers", type=positive_int, default=2, metavar="N",
                        help="Concurrent complete configurations (default: 2; 1 runs serially).")
    parser.add_argument("--total-memory-gb", type=positive_float, metavar="GIB",
                        help="Optional total sweep RAM budget in GiB, including all workers. "
                             "This is an estimate, not an OS limit.")


def helper_sha():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def matched_study(old_meta, metadata, legacy_version, legacy_digests):
    """Accept the current identity or a known, otherwise identical serial release."""
    if not isinstance(old_meta, dict):
        return False
    old, new = old_meta.get("study"), metadata["study"]
    if not isinstance(old, dict):
        return False
    if old == new:
        if "serial_origin" in old_meta:
            metadata["serial_origin"] = old_meta["serial_origin"]
        return True
    if old.get("script_version") != legacy_version or old.get("script_sha256") not in legacy_digests:
        return False
    omit = {"script_version", "script_sha256", "worker_helper_sha256"}
    scientific = lambda data: {k: v for k, v in data.items() if k not in omit}
    if scientific(old) != scientific(new):
        return False
    metadata["serial_origin"] = {"script_version": old["script_version"],
                                 "script_sha256": old["script_sha256"]}
    print("Matched the previous serial release, crop, model and configuration; preserving completed rows.")
    return True


@contextmanager
def output_lock(path):
    """One coordinator per output directory; the OS releases this lock on exit."""
    path.mkdir(parents=True, exist_ok=True)
    with (path / ".sweep.lock").open("a+b") as f:
        if os.name == "nt":
            import msvcrt
            if f.seek(0, 2) == 0:
                f.write(b"0")
                f.flush()
            f.seek(0)
            try:
                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as e:
                raise ValueError(f"Another sweep is using {path}, or the filesystem cannot lock it.") from e
        else:
            import fcntl
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as e:
                raise ValueError(f"Another sweep is using {path}, or the filesystem cannot lock it.") from e
        yield  # Closing the descriptor releases either kind of lock.


def _available_memory():
    import psutil
    available = psutil.virtual_memory().available
    # The host may have much more RAM than the Linux job/container allocation.
    root = Path("/sys/fs/cgroup")
    if (root / "cgroup.controllers").is_file():
        roots = {root}
        try:
            rel = next(x[3:] for x in Path("/proc/self/cgroup").read_text().splitlines()
                       if x.startswith("0::"))
            here = (root / rel.lstrip("/")).resolve()
            if here.is_relative_to(root):
                roots.update(p for p in (here, *here.parents) if p.is_relative_to(root))
        except (OSError, StopIteration):
            pass
        for d in roots:
            try:
                limit = int((d / "memory.max").read_text().strip())
                used = int((d / "memory.current").read_text().strip())
                available = min(available, max(0, limit - used))
            except (OSError, ValueError):
                pass
    return available / _GIB


def plan(args, pending_count, max_job_gib, arrays):
    try:
        cpus = len(os.sched_getaffinity(0))
    except AttributeError:
        cpus = os.cpu_count() or 1
    allocation = os.environ.get("SLURM_CPUS_PER_TASK", "")
    if allocation.isdecimal() and int(allocation) > 0:
        cpus = min(cpus, int(allocation))
    n = min(args.workers, cpus, pending_count)
    if not n:
        print("No outstanding configurations.")
        return 0
    # Account for the parent's crop arrays and a modest coordinator allowance.
    shared = sum(a.nbytes for a in arrays.values()) / _GIB
    estimate = max_job_gib * n + shared + 0.25
    available = _available_memory()
    budget = min(available, args.total_memory_gb) if args.total_memory_gb is not None else available
    print(f"{pending_count} pending; {n} active worker(s) requested/available for this run "
          f"(--workers {args.workers}); estimated aggregate RAM {estimate:.2f} GiB "
          f"(~{max_job_gib:.2f} GiB per configuration); budget {budget:.2f} GiB.", flush=True)
    if estimate > budget:
        raise ValueError(f"Estimated total {estimate:.2f} GiB exceeds {budget:.2f} GiB. "
                         "Choose fewer workers or a smaller crop, or request a larger RAM allocation.")
    return n


def load_model(repo, kind):
    loc = str(repo)
    if loc not in sys.path:
        sys.path.insert(0, loc)
    from ccsds123_i2_hlm import header as hd
    if kind == "raw":
        from ccsds123_i2_hlm import ccsds123
        return hd, ccsds123
    from ccsds123_i2_hlm import constants, predictor, hybrid_encoder
    if kind == "vmax":
        return hd, constants, predictor, hybrid_encoder
    from ccsds123_i2_hlm import sa_encoder, ba_encoder
    return hd, constants, predictor, {
        hd.EntropyCoderType.HYBRID: hybrid_encoder.HybridEncoder,
        hd.EntropyCoderType.SAMPLE_ADAPTIVE: sa_encoder.SampleAdaptiveEncoder,
        hd.EntropyCoderType.BLOCK_ADAPTIVE: ba_encoder.BlockAdaptiveEncoder}


def _calculate(evaluate, key, arrays, model, state):
    logs = io.StringIO()
    try:
        with redirect_stdout(logs):
            return evaluate(key, arrays, model, state)
    except Exception as exc:
        raise RuntimeError(f"Configuration {key!r} failed. Model output (tail):\n"
                           + logs.getvalue()[-4000:]) from exc


def _worker(conn, paths, repo, kind, evaluate, state):
    # Parent owns Ctrl+C, so it can checkpoint completed rows and stop both children.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        import numpy as np
        arrays = {s: np.load(p, mmap_mode="r", allow_pickle=False) for s, p in paths.items()}
        model = load_model(repo, kind)
        while True:
            key = conn.recv()
            if key is None:
                return
            row = _calculate(evaluate, key, arrays, model, state)
            conn.send((True, row))
    except (EOFError, BrokenPipeError):
        return
    except Exception:
        try:
            conn.send((False, traceback.format_exc()))
        except (EOFError, BrokenPipeError, OSError):
            pass
    finally:
        conn.close()


@contextmanager
def _single_native_thread():
    # Set before spawn imports NumPy; restore the calling program's environment.
    old = {k: os.environ.get(k) for k in _NATIVE_THREADS}
    try:
        os.environ.update({k: "1" for k in _NATIVE_THREADS})
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def results(keys, n, arrays, repo, kind, evaluate, state):
    """Yield keyed results as they finish. Keep one task in flight per process."""
    if not keys:
        return
    if n == 1:
        model = load_model(repo, kind)
        for key in keys:
            print(f"  [PID {os.getpid()}] start {key}", flush=True)
            yield key, _calculate(evaluate, key, arrays, model, state)
        return
    import numpy as np
    ctx = mp.get_context("spawn")  # same process contract on Linux and Windows
    with tempfile.TemporaryDirectory(prefix="hypso_sweep_") as tmp:
        paths = {}
        for i, (s, arr) in enumerate(arrays.items()):
            file = Path(tmp) / f"crop_{i}.npy"
            np.save(file, arr, allow_pickle=False)
            paths[s] = str(file)
        workers, active = {}, {}
        rest = iter(keys)
        clean = False

        def dispatch(conn):
            key = next(rest, None)
            conn.send(key)
            if key is not None:
                active[conn] = key
                print(f"  [PID {workers[conn].pid}] start {key}", flush=True)

        try:
            with _single_native_thread():
                for _ in range(n):
                    parent, child = ctx.Pipe()
                    proc = ctx.Process(target=_worker, args=(child, paths, repo, kind, evaluate, state))
                    try:
                        proc.start()
                    except BaseException:
                        parent.close()
                        raise
                    finally:
                        child.close()
                    workers[parent] = proc
                    dispatch(parent)
            # A pipe also becomes ready at EOF if a worker is killed (e.g. OOM).
            while active:
                for conn in wait(list(active)):
                    key = active.pop(conn)
                    try:
                        ok, msg = conn.recv()
                    except (EOFError, OSError) as exc:
                        raise RuntimeError(f"Worker PID {workers[conn].pid} exited during {key}; "
                                           "check job memory and system logs.") from exc
                    if not ok:
                        raise RuntimeError(f"Worker PID {workers[conn].pid} failed on {key}:\n{msg}")
                    # Persist the result before giving this worker another job.
                    yield key, msg
                    dispatch(conn)
            clean = True
        finally:
            for conn, proc in workers.items():
                if not clean and proc.is_alive():
                    proc.terminate()
                conn.close()
            for proc in workers.values():
                proc.join(timeout=5)
                if proc.is_alive():
                    proc.kill()
                    proc.join()
                proc.close()
            # Both children (and thus all file mappings) are closed before cleanup.


def pending_keys(jobs, rows, get_key):
    done = {get_key(r) for r in rows}
    if len(set(jobs)) != len(jobs) or len(done) != len(rows) or not done.issubset(set(jobs)):
        raise ValueError("Duplicate or unexpected saved/requested configuration; no results changed.")
    return [k for k in jobs if k not in done]


def collect(jobs, rows, pending, n, arrays, repo, kind, evaluate, state, *,
            get_key, save, plot=None, grouped=True, out=None):
    """Single writer; CSV stays in experiment order even if workers finish out of order."""
    start = time.monotonic()
    ranks = {key: i for i, key in enumerate(jobs)}
    remaining = {}
    if grouped:
        for k in pending:
            remaining[k[0]] = remaining.get(k[0], 0) + 1
    try:
        with closing(results(pending, n, arrays, repo, kind, evaluate, state)) as completed:
            for key, row in completed:
                if get_key(row) != key:
                    raise RuntimeError(f"Worker returned the wrong result for {key}.")
                rows.append(row)
                rows.sort(key=lambda r: ranks[get_key(r)])
                save(rows)
                rate = row["bits_per_sample"]
                err = row.get("max_error", row.get("max_abs_error_model"))
                seconds = (row["predictor_seconds"] + row["encoder_seconds"] if "predictor_seconds" in row
                           else row["model_seconds_including_debug_write"])
                print(f"[{len(rows)}/{len(jobs)}] {key}: {rate:.5f} bits/sample; "
                      f"max error={err}; model time={seconds:.2f} s", flush=True)
                if grouped:
                    remaining[key[0]] -= 1
                    if not remaining[key[0]] and len(rows) < len(jobs) and plot:
                        plot(rows)
    except KeyboardInterrupt:
        print("\nInterrupted; completed CSV rows have been checkpointed.", flush=True)
        if plot and rows:
            plot(rows)
        raise
    except Exception:
        if out is not None:
            (out / "last_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    if plot and rows:
        plot(rows)
    if pending:
        print(f"{len(pending)} new configurations in {time.monotonic() - start:.2f} s wall time "
              f"(workers, checkpointing and plots included).", flush=True)
''',
'hybrid_no_traces.py': r'''"""Optional memory-saving storage for the NTNU CCSDS-123 hybrid encoder.

All coding methods are inherited unchanged. Only eight full-cube Unicode
debugging arrays are replaced by discard-only objects. Numeric codec state,
the binary bitstream, initialization and termination are retained.

The upstream class is checked before use because this adapter relies on those
eight attributes being diagnostic only in the validated reference version.
"""
from __future__ import annotations

import ast
from functools import lru_cache
import hashlib
import inspect
from pathlib import Path
import textwrap

import numpy as np
from bitarray import bitarray
from ccsds123_i2_hlm.hybrid_encoder import HybridEncoder

REFERENCE_CLASS_SHA256 = "eccc91082666bc1ffc9a229bd24751d85deff319b327dd87402452398aeabed0"
DISCARDED_TEXT_BYTES_PER_SAMPLE = 1200


@lru_cache(maxsize=1)
def validate_reference():
    """Allow whitespace/comments to differ, but refuse changed reference code."""
    source = textwrap.dedent(inspect.getsource(HybridEncoder))
    node = ast.parse(source).body[0]
    # Python 3.12 adds empty type_params fields to classes/functions. Omit
    # those empty fields so the same Python source also validates on 3.11.
    for item in ast.walk(node):
        if "type_params" in item._fields and not getattr(item, "type_params", None):
            item._fields = tuple(name for name in item._fields if name != "type_params")
    actual = hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
    if actual != REFERENCE_CLASS_SHA256:
        raise ValueError(
            "--low-memory requires the validated NTNU HybridEncoder implementation "
            "(upstream commit 9fabb0371ebf93d72a0f83ca053e9cdb2c63e3c6). "
            "Your encoder code differs; validate this adapter against that version "
            "before using it. The source check must not simply be disabled."
        )
    return actual


def adapter_sha256():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def estimated_job_gib(shape, p_max):
    """Conservative predictor/check/encoder allowance; not an enforced limit.

    Predictor retains 14 int64 cubes and two int64 vectors of up to P+3
    components, plus reconstruction/error temporaries. Predictor is released
    before encoding. A base 320 bytes/sample includes extra working allowance;
    the surviving hybrid arrays use about 33 bytes/sample, excluding input.
    The coordinator separately budgets its input arrays and fixed overhead.
    """
    count = int(np.prod(shape, dtype=np.int64))
    return count * (320 + 16 * (p_max + 3)) / 2**30 + 0.5


class _DiscardedTextTrace:
    """Accept scalar diagnostic writes without retaining a sample history.

    The reference reads these fields only for append-to-trace operations and
    one assertion on an initially empty input-symbol trace. They never feed
    coding decisions. Access to actual stored traces (e.g. reshape) is absent.
    """
    __slots__ = ("initial",)

    def __init__(self, initial=""):
        self.initial = initial

    def __getitem__(self, key):
        return self.initial

    def __setitem__(self, key, value):
        pass


class CompactHybridEncoder(HybridEncoder):
    def __init__(self, header, image_constants, mapped_quantizer_index):
        validate_reference()
        super().__init__(header, image_constants, mapped_quantizer_index)

    # Explicit parent-mangled name: run_encoder() calls this initializer.
    def _HybridEncoder__init_encoder_arrays(self):
        image_shape = self.mapped_quantizer_index.shape
        self.accumulator = np.zeros(image_shape, dtype=np.int64)
        self.counter = np.zeros(image_shape[:2], dtype=np.int64)
        self.variable_length_code = np.full(image_shape, fill_value=-1, dtype=np.int64)
        self.active_prefix = [""] * 16
        self.code_index = np.full(image_shape, fill_value=-1, dtype=np.int64)
        self.input_symbol = _DiscardedTextTrace()
        self.low_entropy_codes = _DiscardedTextTrace()
        self.high_entropy_codes = _DiscardedTextTrace()
        self.rescale_bits = _DiscardedTextTrace()
        self.flush_codes = np.full((16), fill_value="", dtype="U10")
        self.accumulator_final = np.full((self.header.z_size), fill_value="", dtype="U46")
        self.codewords = _DiscardedTextTrace()
        self.codewords_binary = _DiscardedTextTrace()
        self.entropy_type = np.full(image_shape, fill_value=2, dtype=np.uint8)
        self.current_active_prefix = _DiscardedTextTrace("-")
        self.prefix_match_index = np.full(image_shape, fill_value=-2, dtype=np.int64)

        self.counter[0, 0] = 2**self.initial_count_exponent
        if self.use_accu_init_file:
            with open(self.accu_init_file, "rb") as file:
                data = file.read()
                data = "".join([bin(byte)[2:].zfill(8) for byte in data])
                num_bits = self.image_constants.dynamic_range_bits + self.initial_count_exponent
                data = data[: num_bits * self.header.z_size]
                data = [int(data[i : i + num_bits], 2) for i in range(0, len(data), num_bits)]
                assert len(data) == self.header.z_size
                self.accumulator[0, 0] = np.array(data)
        else:
            self.accumulator[0, 0] = 4 * self.counter[0, 0]

        self.bitstream = bitarray()
        self.bitstream_readable = _DiscardedTextTrace()

    def save_data(self, output_folder, header_bitstream):
        raise NotImplementedError(
            "CompactHybridEncoder discards text traces and cannot use the reference "
            "save_data() trace exporter. Use the sweep's CSV/plot metrics, or write "
            "header_bitstream + bitstream with the required output-word padding."
        )
''',
'archived_r64.json': r'''{"profile":{"p":3,"nu_min":-1,"nu_max":3,"t_inc":2048,"phi":3,"psi":7,"local_sum":"narrow-neighbor","prediction_mode":"reduced","D":16,"R":64,"Theta":3,"encoder":"HYBRID","psi_lossless":0,"weight_initialization":"DEFAULT","weight_exponent_offsets":"ALL_ZERO"},"model_sha256":"d21132b89c873965518d6ed7d11619d7792d174f3a146931fb6395c86799c3b9","inputs":{"lisbon":{"file_sha256":"c1a618541c7c6a2a8fcd33e1650a007a50cc846de30e54b21e25ad6b358178bc","source_shape_yxz":[956,684,120],"axis_order":"yxz","variable":"products/dn"},"bangladesh":{"file_sha256":"81e4d20872bdb2c87c950db3280ba35769b5ecdac3875df457e286d1922889a8","source_shape_yxz":[956,684,120],"axis_order":"yxz","variable":"products/dn"},"santarem":{"file_sha256":"67e7d15625164a65a26aaa3798cde8591184754281d5245f76f8290a212e38d1","source_shape_yxz":[956,684,120],"axis_order":"yxz","variable":"products/dn"},"doha":{"file_sha256":"463d58fa18e7fe8dbe6e2f2fa5d927a08e2dd9c0ce864f1502ec3cab19146e2e","source_shape_yxz":[956,684,120],"axis_order":"yxz","variable":"products/dn"},"finnmark":{"file_sha256":"af6f04a8847f41b89d047e6e6c5fe36251c56a1b9bad9d50f193c1baceb74323","source_shape_yxz":[956,684,120],"axis_order":"yxz","variable":"products/dn"}},"rows":[{"image":"lisbon","capture":"lisbon_2022-09-22T10-55-19Z-l1a.nc","crop_sha256":"ce3eea19d9a04730c38d170f5887104de08a256b5e397912b155167bdc5e1969","payload_sha256":"4532e8ffa6678f38104fde1864d52904dfd5a48b4fa41992263c25c563410f60","local_sum":"NARROW_NEIGHBOR_ORIENTED","prediction_mode":"REDUCED","encoder":"HYBRID","a":0,"omega":19,"nu_min":-1,"nu_max":3,"phi":3,"psi":0,"p":3,"t_inc":2048,"rows":256,"columns":256,"bands":120,"crop_row":350,"crop_col":214,"sample_count":7864320,"header_bits":176,"payload_bits":54036872,"padding_bits":8,"stream_bits":54037056,"external_info_bits":2040,"max_error":0,"rmse_dn":0.0,"source_study":"omega_fig418_256","source_job":"4097da209567b6c8"},{"image":"lisbon","capture":"lisbon_2022-09-22T10-55-19Z-l1a.nc","crop_sha256":"ce3eea19d9a04730c38d170f5887104de08a256b5e397912b155167bdc5e1969","payload_sha256":"b26ab5c089bc4eff2e6d13bf725f51336a693ed881391c4b232237348695e42e","local_sum":"NARROW_NEIGHBOR_ORIENTED","prediction_mode":"REDUCED","encoder":"HYBRID","a":0,"omega":12,"nu_min":-1,"nu_max":3,"phi":3,"psi":0,"p":3,"t_inc":2048,"rows":256,"columns":256,"bands":120,"crop_row":350,"crop_col":214,"sample_count":7864320,"header_bits":176,"payload_bits":54159746,"padding_bits":14,"stream_bits":54159936,"external_info_bits":2040,"max_error":0,"rmse_dn":0.0,"source_study":"omega_fig418_256","source_job":"43d530381eaf5c04"},{"image":"lisbon","capture":"lisbon_2022-09-22T10-55-19Z-l1a.nc","crop_sha256":"ce3eea19d9a04730c38d170f5887104de08a256b5e397912b155167bdc5e1969","payload_sha256":"c9e66b56f860780d76b1dfcf0ae5ac0c33bb79965137191639360c4be3347a36","local_sum":"NARROW_NEIGHBOR_ORIENTED","prediction_mode":"REDUCED","encoder":"HYBRID","a":0,"omega":14,"nu_min":-1,"nu_max":3,"phi":3,"psi":0,"p":3,"t_inc":2048,"rows":256,"columns":256,"bands":120,"crop_row":350,"crop_col":214,"sample_count":7864320,"header_bits":176,"payload_bits":54038734,"padding_bits":2,"stream_bits":54038912,"external_info_bits":2040,"max_error":0,"rmse_dn":0.0,"source_study":"omega_fig418_256","source_job":"9d7b3e4177d59b37"},{"image":"lisbon","capture":"lisbon_2022-09-22T10-55-19Z-l1a.nc","crop_sha256":"ce3eea19d9a04730c38d170f5887104de08a256b5e397912b155167bdc5e1969","payload_sha256":"60f4bd9fd3676fa7e32b7dd97bb42da49f56c62e8a52618811c4d0beef0495af","local_sum":"NARROW_NEIGHBOR_ORIENTED","prediction_mode":"REDUCED","encoder":"HYBRID","a":1,"omega":19,"nu_min":-1,"nu_max":3,"phi":3,"psi":7,"p":3,"t_inc":2048,"rows":256,"columns":256,"bands":120,"crop_row":350,"crop_col":214,"sample_count":7864320,"header_bits":200,"payload_bits":41492179,"padding_bits":37,"stream_bits":41492416,"external_info_bits":2040,"max_error":1,"rmse_dn":0.8164776589832787,"source_study":"omega_fig418_256","source_job":"85dbb22c040ce546"},{"image":"lisbon","capture":"lisbon_2022-09-22T10-55-19Z-l1a.nc","crop_sha256":"ce3eea19d9a04730c38d170f5887104de08a256b5e397912b155167bdc5e1969","payload_sha256":"21679c8ab33fe473abfc353e67d735044a42f0a55f58a34d6a188058f4a59b90","local_sum":"NARROW_NEIGHBOR_ORIENTED","prediction_mode":"REDUCED","encoder":"HYBRID","a":1,"omega":12,"nu_min":-1,"nu_max":3,"phi":3,"psi":7,"p":3,"t_inc":2048,"rows":256,"columns":256,"bands":120,"crop_row":350,"crop_col":214,"sample_count":7864320,"header_bits":200,"payload_bits":41613379,"padding_bits":53,"stream_bits":41613632,"external_info_bits":2040,"max_error":1,"rmse_dn":0.8165274936137474,"source_study":"omega_fig418_256","source_job":"173a24fb85ce611b"},{"image":"lisbon","capture":"lisbon_2022-09-22T10-55-19Z-l1a.nc","crop_sha256":"ce3eea19d9a04730c38d170f5887104de08a256b5e397912b155167bdc5e1969","payload_sha256":"c509ebd98914108450378be3c0f463859563018a37302f64d046cd6c7e1ba4bc","local_sum":"NARROW_NEIGHBOR_ORIENTED","prediction_mode":"REDUCED","encoder":"HYBRID","a":1,"omega":14,"nu_min":-1,"nu_max":3,"phi":3,"psi":7,"p":3,"t_inc":2048,"rows":256,"columns":256,"bands":120,"crop_row":350,"crop_col":214,"sample_count":7864320,"header_bits":200,"payload_bits":41493026,"padding_bits":22,"stream_bits":41493248,"external_info_bits":2040,"max_error":1,"rmse_dn":0.8165327105007542,"source_study":"omega_fig418_256","source_job":"8fff8f1400bec3d3"},{"image":"lisbon","capture":"lisbon_2022-09-22T10-55-19Z-l1a.nc","crop_sha256":"ce3eea19d9a04730c38d170f5887104de08a256b5e397912b155167bdc5e1969","payload_sha256":"8ea085b5bcd2338ca8c6ed99d382526e0cea44da192365974b4c69a3073be465","local_sum":"NARROW_NEIGHBOR_ORIENTED","prediction_mode":"REDUCED","encoder":"HYBRID","a":3,"omega":19,"nu_min":-1,"nu_max":3,"phi":3,"psi":7,"p":3,"t_inc":2048,"rows":256,"columns":256,"bands":120,"crop_row":350,"crop_col":214,"sample_count":7864320,"header_bits":200,"payload_bits":31680262,"padding_bits":50,"stream_bits":31680512,"external_info_bits":2040,"max_error":3,"rmse_dn":2.0000029245990985,"source_study":"omega_fig418_256","source_job":"a6c0bd73d9718402"},{"image":"lisbon","capture":"lisbon_2022-09-22T10-55-19Z-l1a.nc","crop_sha256":"ce3eea19d9a04730c38d170f5887104de08a256b5e397912b155167bdc5e1969","payload_sha256":"f7ce35dfb739d33da9d5a50d6d32a15130d5e377f7f6267c412c7482a5749513","local_sum":"NARROW_NEIGHBOR_ORIENTED","prediction_mode":"REDUCED","encoder":"HYBRID","a":3,"omega":12,"nu_min":-1,"nu_max":3,"phi":3,"psi":7,"p":3,"t_inc":2048,"rows":256,"columns":256,"bands":120,"crop_row":350,"crop_col":214,"sample_count":7864320,"header_bits":200,"payload_bits":31788080,"padding_bits":8,"stream_bits":31788288,"external_info_bits":2040,"max_error":3,"rmse_dn":2.000533128253869,"source_study":"omega_fig418_256","source_job":"95fc0641317abf6b"},{"image":"lisbon","capture":"lisbon_2022-09-22T10-55-19Z-l1a.nc","crop_sha256":"ce3eea19d9a04730c38d170f5887104de08a256b5e397912b155167bdc5e1969","payload_sha256":"8ab864c5e72f8e7fd4ddf536b1c5e3e1862503258d28ace820ac6c73c4f8e0d9","local_sum":"NARROW_NEIGHBOR_ORIENTED","prediction_mode":"REDUCED","encoder":"HYBRID","a":3,"omega":14,"nu_min":-1,"nu_max":3,"phi":3,"psi":7,"p":3,"t_inc":2048,"rows":256,"columns":256,"bands":120,"crop_row":350,"crop_col":214,"sample_count":7864320,"header_bits":200,"payload_bits":31682049,"padding_bits":55,"stream_bits":31682304,"external_info_bits":2040,"max_error":3,"rmse_dn":1.9997785127704384,"source_study":"omega_fig418_256","source_job":"04579a71ff6f3cb2"},{"image":"lisbon","capture":"lisbon_2022-09-22T10-55-19Z-l1a.nc","crop_sha256":"ce3eea19d9a04730c38d170f5887104de08a256b5e397912b155167bdc5e1969","payload_sha256":"9b25f1e399eb30bf5e00ff50fdb02a1391b9e898f5cb567b61a1d70167c2d948","local_sum":"NARROW_NEIGHBOR_ORIENTED","prediction_mode":"REDUCED","encoder":"HYBRID","a":7,"omega":19,"nu_min":-1,"nu_max":3,"phi":3,"psi":7,"p":3,"t_inc":2048,"rows":256,"columns":256,"bands":120,"crop_row":350,"crop_col":214,"sample_count":7864320,"header_bits":200,"payload_bits":22890429,"padding_bits":59,"stream_bits":22890688,"external_info_bits":2040,"max_error":7,"rmse_dn":4.320315207767385,"source_study":"omega_fig418_256","source_job":"6a23611547a73319"},{"image":"lisbon","capture":"lisbon_2022-09-22T10-55-19Z-l1a.nc","crop_sha256":"ce3eea19d9a04730c38d170f5887104de08a256b5e397912b155167bdc5e1969","payload_sha256":"74381c7d92c4f5b42e651e91c2d5b363bc7f5844df1318846465ef246157810f","local_sum":"NARROW_NEIGHBOR_ORIENTED","prediction_mode":"REDUCED","encoder":"HYBRID","a":7,"omega":12,"nu_min":-1,"nu_max":3,"phi":3,"psi":7,"p":3,"t_inc":2048,"rows":256,"columns":256,"bands":120,"crop_row":350,"crop_col":214,"sample_count":7864320,"header_bits":200,"payload_bits":22976776,"padding_bits":48,"stream_bits":22977024,"external_info_bits":2040,"max_error":7,"rmse_dn":4.320306701840217,"source_study":"omega_fig418_256","source_job":"7cbfd5c7c1609d64"},{"image":"lisbon","capture":"lisbon_2022-09-22T10-55-19Z-l1a.nc","crop_sha256":"ce3eea19d9a04730c38d170f5887104de08a256b5e397912b155167bdc5e1969","payload_sha256":"f81302c4238068b539f1255e0fcc8f30f697f041d4b98724a9bbc1e3f6c5ef01","local_sum":"NARROW_NEIGHBOR_ORIENTED","prediction_mode":"REDUCED","encoder":"HYBRID","a":7,"omega":14,"nu_min":-1,"nu_max":3,"phi":3,"psi":7,"p":3,"t_inc":2048,"rows":256,"columns":256,"bands":120,"crop_row":350,"crop_col":214,"sample_count":7864320,"header_bits":200,"payload_bits":22896786,"padding_bits":38,"stream_bits":22897024,"external_info_bits":2040,"max_error":7,"rmse_dn":4.3194275725192925,"source_study":"omega_fig418_256","source_job":"8df61a34b237ca60"},{"image":"lisbon","capture":"lisbon_2022-09-22T10-55-19Z-l1a.nc","crop_sha256":"ce3eea19d9a04730c38d170f5887104de08a256b5e397912b155167bdc5e1969","payload_sha256":"571834e74194ab016ce6699fbee424f75c147e426f18758dde3a3a8b961cd3c0","local_sum":"NARROW_NEIGHBOR_ORIENTED","prediction_mode":"REDUCED","encoder":"HYBRID","a":15,"omega":19,"nu_min":-1,"nu_max":3,"phi":3,"psi":7,"p":3,"t_inc":2048,"rows":256,"columns":256,"bands":120,"crop_row":350,"crop_col":214,"sample_count":7864320,"header_bits":200,"payload_bits":15196562,"padding_bits":38,"stream_bits":15196800,"external_info_bits":2040,"max_error":15,"rmse_dn":8.809475975544995,"source_study":"omega_fig418_256","source_job":"54226eb2399649bc"},{"image":"lisbon","capture":"lisbon_2022-09-22T10-55-19Z-l1a.nc","crop_sha256":"ce3eea19d9a04730c38d170f5887104de08a256b5e397912b155167bdc5e1969","payload_sha256":"e2b7f819cbff52264533c1ecfea3b4b7929e7c064eaf7e800ae629314945942f","local_sum":"NARROW_NEIGHBOR_ORIENTED","prediction_mode":"REDUCED","encoder":"HYBRID","a":15,"omega":12,"nu_min":-1,"nu_max":3,"phi":3,"psi":7,"p":3,"t_inc":2048,"rows":256,"columns":256,"bands":120,"crop_row":350,"crop_col":214,"sample_count":7864320,"header_bits":200,"payload_bits":15288727,"padding_bits":33,"stream_bits":15288960,"external_info_bits":2040,"max_error":15,"rmse_dn":8.818929973291544,"source_study":"omega_fig418_256","source_job":"2bda4365fedcfaf1"},{"image":"lisbon","capture":"lisbon_2022-09-22T10-55-19Z-l1a.nc","crop_sha256":"ce3eea19d9a04730c38d170f5887104de08a256b5e397912b155167bdc5e1969","payload_sha256":"f1a69845b2d03651cfe71176b3b9f52030f771d0cf44c6af161eceff09739004","local_sum":"NARROW_NEIGHBOR_ORIENTED","prediction_mode":"REDUCED","encoder":"HYBRID","a":15,"omega":14,"nu_min":-1,"nu_max":3,"phi":3,"psi":7,"p":3,"t_inc":2048,"rows":256,"columns":256,"bands":120,"crop_row":350,"crop_col":214,"sample_count":7864320,"header_bits":200,"payload_bits":15201032,"padding_bits":48,"stream_bits":15201280,"external_info_bits":2040,"max_error":15,"rmse_dn":8.810478963486608,"source_study":"omega_fig418_256","source_job":"dd1defa77e9bf373"},{"image":"bangladesh","capture":"bangladesh_2022-10-26T04-02-56Z-l1a.nc","crop_sha256":"456a44f7f7d0645290a165e8eac772cb31041eca7aa7ca137e5b80d5aaad4186","payload_sha256":"7140ac60fd6851c36be5fb6e1136020f836b7f9b853894465b72ff302a3b21bb","local_sum":"NARROW_NEIGHBOR_ORIENTED","prediction_mode":"REDUCED","encoder":"HYBRID","a":0,"omega":19,"nu_min":-1,"nu_max":3,"phi":3,"psi":0,"p":3,"t_inc":2048,"rows":256,"columns":256,"bands":120,"crop_row":350,"crop_col":214,"sample_count":7864320,"header_bits":176,"payload_bits":60899185,"padding_bits":31,"stream_bits":60899392,"external_info_bits":2040,"max_error":0,"rmse_dn":0.0,"source_study":"omega_fig418_256","source_job":"d84df1a51d6153fb"},{"image":"bangladesh","capture":"bangladesh_2022-10-26T04-02-56Z-l1a.nc","crop_sha256":"456a44f7f7d0645290a165e8eac772cb31041eca7aa7ca137e5b80d5aaad4186","payload_sha256":"f2fc3b41cc4709fec29463b5de61f38cd465b868bef866a8e86b4f7543cd2fbe","local_sum":"NARROW_NEIGHBOR_ORIENTED","prediction_mode":"REDUCED","encoder":"HYBRID","a":0,"omega":12,"nu_min":-1,"nu_max":3,"phi":3,"psi":0,"p":3,"t_inc":2048,"rows":256,"columns":256,"bands":120,"crop_row":350,"crop_col":214,"sample_count":7864320,"header_bits":176,"payload_bits":60898565,"padding_bits":11,"stream_bits":60898752,"external_info_bits":2040,"max_error":0,"rmse_dn":0.0,"source_study":"omega_fig418_256","source_job":"410379a1dc5efa40"},{"image":"bangladesh","capture":"bangladesh_2022-10-26T04-02-56Z-l1a.nc","crop_sha256":"456a44f7f7d0645290a165e8eac772cb31041eca7aa7ca137e5b80d5aaad4186","payload_sha256":"8ae81d1e03d4049d86b15239e6d373ae9c05fdee68f0edc12b5618266fc0796a","local_sum":"NARROW_NEIGHBOR_ORIENTED","prediction_mode":"REDUCED","encoder":"HYBRID","a":0,"omega":14,"nu_min":-1,"nu_max":3,"phi":3,"psi":0,"p":3,"t_inc":2048,"rows":256,"columns":256,"bands":120,"crop_row":350,"crop_col":214,"sample_count":7864320,"header_bits":176,"payload_bits":60899272,"padding_bits":8,"stream_bits":60899456,"external_info_bits":2040,"max_error":0,"rmse_dn":0.0,"source_study":"omega_fig418_256","source_job":"0e3f0ac7b67a642d"},{"image":"bangladesh","capture":"bangladesh_2022-10-26T04-02-56Z-l1a.nc","crop_sha256":"456a44f7f7d0645290a165e8eac772cb31041eca7aa7ca137e5b80d5aaad4186","payload_sha256":"3295bb20daec1a30cfcec70c49e4e731c826ac0920a61ecd6359b41bc11f5169","local_sum":"NARROW_NEIGHBOR_ORIENTED","prediction_mode":"REDUCED","encoder":"HYBRID","a":1,"omega":19,"nu_min":-1,"nu_max":3,"phi":3,"psi":7,"p":3,"t_inc":2048,"rows":256,"columns":256,"bands":120,"crop_row":350,"crop_col":214,"sample_count":7864320,"header_bits":200,"payload_bits":48446708,"padding_bits":4,"stream_bits":48446912,"external_info_bits":2040,"max_error":1,"rmse_dn":0.8164898064547997,"source_study":"omega_fig418_256","source_job":"c5d3d70cb9682ebc"},{"image":"bangladesh","capture":"bangladesh_2022-10-26T04-02-56Z-l1a.nc","crop_sha256":"456a44f7f7d0645290a165e8eac772cb31041eca7aa7ca137e5b80d5aaad4186","payload_sha256":"bd54784248fe1fab65e62469ebc66fac88aa391bff84c703bce22e53829513e9","local_sum":"NARROW_NEIGHBOR_ORIENTED","prediction_mode":"REDUCED","encoder":"HYBRID","a":1,"omega":12,"nu_min":-1,"nu_max":3,"phi":3,"psi":7,"p":3,"t_inc":2048,"rows":256,"columns":256,"bands":120,"crop_row":350,"crop_col":214,"sample_count":7864320,"header_bits":200,"payload_bits":48448959,"padding_bits":57,"stream_bits":48449216,"external_info_bits":2040,"max_error":1,"rmse_dn":0.8165555242568235,"source_study":"omega_fig418_256","source_job":"2e3121e38f32b50c"},{"image":"bangladesh","capture":"bangladesh_2022-10-26T04-02-56Z-l1a.nc","crop_sha256":"456a44f7f7d0645290a165e8eac772cb31041eca7aa7ca137e5b80d5aaad4186","payload_sha256":"6c5b3a12e8e231c8f5d70c332f226e00b6a7ce67c5904b30cb3669639c8cf7ff","local_sum":"NARROW_NEIGHBOR_ORIENTED","prediction_mode":"REDUCED","encoder":"HYBRID","a":1,"omega":14,"nu_min":-1,"nu_max":3,"phi":3,"psi":7,"p":3,"t_inc":2048,"rows":256,"columns":256,"bands":120,"crop_row":350,"crop_col":214,"sample_count":7864320,"header_bits":200,"payload_bits":48448448,"padding_bits":56,"stream_bits":48448704,"external_info_bits":2040,"max_error":1,"rmse_dn":0.8165918069381597,"source_study":"omega_fig418_256","source_job":"e0039157fc082ddd"},{"image":"bangladesh","capture":"bangladesh_2022-10-26T04-02-56Z-l1a.nc","crop_sha256":"456a44f7f7d0645290a165e8eac772cb31041eca7aa7ca137e5b80d5aaad4186","payload_sha256":"9ea3969f5cf366fb487f6e2fceb942ef8470522fec91c1fab33e6dc2ad5f8dc0","local_sum":"NARROW_NEIGHBOR_ORIENTED","prediction_mode":"REDUCED","encoder":"HYBRID","a":3,"omega":19,"nu_min":-1,"nu_max":3,"phi":3,"psi":7,"p":3,"t_inc":2048,"rows":256,"columns":256,"bands":120,"crop_row":350,"crop_col":214,"sample_count":7864320,"header_bits":200,"payload_bits":38797721,"padding_bits":31,"stream_bits":38797952,"external_info_bits":2040,"max_error":3,"rmse_dn":2.0003046757372656,"source_study":"omega_fig418_256","source_job":"2fd644222fe906cc"},{"image":"bangladesh","capture":"bangladesh_2022-10-26T04-02-56Z-l1a.nc","crop_sha256":"456a44f7f7d0645290a165e8eac772cb31041eca7aa7ca137e5b80d5aaad4186","payload_sha256":"a69a3deecd9b2b1e81c133b5caefcca4655377ca9ccc3429d0e752fb74deeed7","local_sum":"NARROW_NEIGHBOR_ORIENTED","prediction_mode":"REDUCED","encoder":"HYBRID","a":3,"omega":12,"nu_min":-1,"nu_max":3,"phi":3,"psi":7,"p":3,"t_inc":2048,"rows":256,"columns":256,"bands":120,"crop_row":350,"crop_col":214,"sample_count":7864320,"header_bits":200,"payload_bits":38799685,"padding_bits":51,"stream_bits":38799936,"external_info_bits":2040,"max_error":3,"rmse_dn":2.0002748300550985,"source_study":"omega_fig418_256","source_job":"f767b6522c81fb7f"},{"image":"bangladesh","capture":"bangladesh_2022-10-26T04-02-56Z-l1a.nc","crop_sha256":"456a44f7f7d0645290a165e8eac772cb31041eca7aa7ca137e5b80d5aaad4186","payload_sha256":"0f1fed0f97552aa94fe81cc12f8185b38fe0ad165dcd6b769736721b73f95577","local_sum":"NARROW_NEIGHBOR_ORIENTED","prediction_mode":"REDUCED","encoder":"HYBRID","a":3,"omega":14,"nu_min":-1,"nu_max":3,"phi":3,"psi":7,"p":3,"t_inc":2048,"rows":256,"columns":256,"bands":120,"crop_row":350,"crop_col":214,"sample_count":7864320,"header_bits":200,"payload_bits":38797737,"padding_bits":15,"stream_bits":38797952,"external_info_bits":2040,"max_error":3,"rmse_dn":2.000089261924093,"source_study":"omega_fig418_256","source_job":"204f6ce26dff793b"},{"image":"bangladesh","capture":"bangladesh_2022-10-26T04-02-56Z-l1a.nc","crop_sha256":"456a44f7f7d0645290a165e8eac772cb31041eca7aa7ca137e5b80d5aaad4186","payload_sha256":"0e13341b8f6844536f9e66102941e8b35abe2713558d574cc5f3bbfb559e3726","local_sum":"NARROW_NEIGHBOR_ORIENTED","prediction_mode":"REDUCED","encoder":"HYBRID","a":7,"omega":19,"nu_min":-1,"nu_max":3,"phi":3,"psi":7,"p":3,"t_inc":2048,"rows":256,"columns":256,"bands":120,"crop_row":350,"crop_col":214,"sample_count":7864320,"header_bits":200,"payload_bits":29906799,"padding_bits":9,"stream_bits":29907008,"external_info_bits":2040,"max_error":7,"rmse_dn":4.321438416969347,"source_study":"omega_fig418_256","source_job":"deef80daccb9f3a5"},{"image":"bangladesh","capture":"bangladesh_2022-10-26T04-02-56Z-l1a.nc","crop_sha256":"456a44f7f7d0645290a165e8eac772cb31041eca7aa7ca137e5b80d5aaad4186","payload_sha256":"e306aaca9855f4ecc0314453325a1abde36c02ee1246a715321c7febf5a6960d","local_sum":"NARROW_NEIGHBOR_ORIENTED","prediction_mode":"REDUCED","encoder":"HYBRID","a":7,"omega":12,"nu_min":-1,"nu_max":3,"phi":3,"psi":7,"p":3,"t_inc":2048,"rows":256,"columns":256,"bands":120,"crop_row":350,"crop_col":214,"sample_count":7864320,"header_bits":200,"payload_bits":29907176,"padding_bits":16,"stream_bits":29907392,"external_info_bits":2040,"max_error":7,"rmse_dn":4.320795471162853,"source_study":"omega_fig418_256","source_job":"4189134efe7644ab"},{"image":"bangladesh","capture":"bangladesh_2022-10-26T04-02-56Z-l1a.nc","crop_sha256":"456a44f7f7d0645290a165e8eac772cb31041eca7aa7ca137e5b80d5aaad4186","payload_sha256":"c75fb03216ef041726cef57ddacd9ec1b350c04c20ec85fd7fb75238d2b6c475","local_sum":"NARROW_NEIGHBOR_ORIENTED","prediction_mode":"REDUCED","encoder":"HYBRID","a":7,"omega":14,"nu_min":-1,"nu_max":3,"phi":3,"psi":7,"p":3,"t_inc":2048,"rows":256,"columns":256,"bands":120,"crop_row":350,"crop_col":214,"sample_count":7864320,"header_bits":200,"payload_bits":29907791,"padding_bits":41,"stream_bits":29908032,"external_info_bits":2040,"max_error":7,"rmse_dn":4.31963584342286,"source_study":"omega_fig418_256","source_job":"70e7bd9fa5cf54dc"},{"image":"bangladesh","capture":"bangladesh_2022-10-26T04-02-56Z-l1a.nc","crop_sha256":"456a44f7f7d0645290a165e8eac772cb31041eca7aa7ca137e5b80d5aaad4186","payload_sha256":"75ff7ca9fecc05e85827f7c0e821f6a6dc997623116586f845ea986a2fa6f8f0","local_sum":"NARROW_NEIGHBOR_ORIENTED","prediction_mode":"REDUCED","encoder":"HYBRID","a":15,"omega":19,"nu_min":-1,"nu_max":3,"phi":3,"psi":7,"p":3,"t_inc":2048,"rows":256,"columns":256,"bands":120,"crop_row":350,"crop_col":214,"sample_count":7864320,"header_bits":200,"payload_bits":21547577,"padding_bits":63,"stream_bits":21547840,"external_info_bits":2040,"max_error":15,"rmse_dn":8.932453379302524,"source_study":"omega_fig418_256","source_job":"a00d24c4693659dd"},{"image":"bangladesh","capture":"bangladesh_2022-10-26T04-02-56Z-l1a.nc","crop_sha256":"456a44f7f7d0645290a165e8eac772cb31041eca7aa7ca137e5b80d5aaad4186","payload_sha256":"e1b17b6eeaa216ea2e482aba3c1f924fb88d4d961462c1365c8216b8d64f88b1","local_sum":"NARROW_NEIGHBOR_ORIENTED","prediction_mode":"REDUCED","encoder":"HYBRID","a":15,"omega":12,"nu_min":-1,"nu_max":3,"phi":3,"psi":7,"p":3,"t_inc":2048,"rows":256,"columns":256,"bands":120,"crop_row":350,"crop_col":214,"sample_count":7864320,"header_bits":200,"payload_bits":21548096,"padding_bits":56,"stream_bits":21548352,"external_info_bits":2040,"max_error":15,"rmse_dn":8.938264365066777,"source_study":"omega_fig418_256","source_job":"fc77635d9adff7c7"},{"image":"bangladesh","capture":"bangladesh_2022-10-26T04-02-56Z-l1a.nc","crop_sha256":"456a44f7f7d0645290a165e8eac772cb31041eca7aa7ca137e5b80d5aaad4186","payload_sha256":"156f9451016029220b4165986f76c39993cbe64ca9f868646c448f89074ab810","local_sum":"NARROW_NEIGHBOR_ORIENTED","prediction_mode":"REDUCED","encoder":"HYBRID","a":15,"omega":14,"nu_min":-1,"nu_max":3,"phi":3,"psi":7,"p":3,"t_inc":2048,"rows":256,"columns":256,"bands":120,"crop_row":350,"crop_col":214,"sample_count":7864320,"header_bits":200,"payload_bits":21551778,"padding_bits":22,"stream_bits":21552000,"external_info_bits":2040,"max_error":15,"rmse_dn":8.936728206291257,"source_study":"omega_fig418_256","source_job":"969dd570f4398fdd"},{"image":"santarem","capture":"santarem_2022-09-24T13-52-04Z-l1a.nc","crop_sha256":"b58b314a249ed8a9a95bcc01108cb3daf8cef364166d1910ace1eb371ae7c1df","payload_sha256":"61aca7d301af2ef531e79f0b4af89b74d4caebc31c6001fea47ac4b0542ba184","local_sum":"NARROW_NEIGHBOR_ORIENTED","prediction_mode":"REDUCED","encoder":"HYBRID","a":0,"omega":19,"nu_min":-1,"nu_max":3,"phi":3,"psi":0,"p":3,"t_inc":2048,"rows":256,"columns":256,"bands":120,"crop_row":350,"crop_col":214,"sample_count":7864320,"header_bits":176,"payload_bits":57915062,"padding_bits":26,"stream_bits":57915264,"external_info_bits":2040,"max_error":0,"rmse_dn":0.0,"source_study":"omega_fig418_256","source_job":"4e65f9a412f97a0a"},{"image":"santarem","capture":"santarem_2022-09-24T13-52-04Z-l1a.nc","crop_sha256":"b58b314a249ed8a9a95bcc01108cb3daf8cef364166d1910ace1eb371ae7c1df","payload_sha256":"7685d00e767fda2278bf45d3695255f8e192e5c4da2f28be76d83a27a6fa1049","local_sum":"NARROW_NEIGHBOR_ORIENTED","prediction_mode":"REDUCED","encoder":"HYBRID","a":0,"omega":12,"nu_min":-1,"nu_max":3,"phi":3,"psi":0,"p":3,"t_inc":2048,"rows":256,"columns":256,"bands":120,"crop_row":350,"crop_col":214,"sample_count":7864320,"header_bits":176,"payload_bits":57920285,"padding_bits":51,"stream_bits":57920512,"external_info_bits":2040,"max_error":0,"rmse_dn":0.0,"source_study":"omega_fig418_256","source_job":"6b09cb81eb5df8a5"},{"image":"santarem","capture":"santarem_2022-09-24T13-52-04Z-l1a.nc","crop_sha256":"b58b314a249ed8a9a95bcc01108cb3daf8cef364166d1910ace1eb371ae7c1df","payload_sha256":"ee3b67de9a5dac9fc6e53357a0247cf24a0c676bd7811217c826aef74fb9b987","local_sum":"NARROW_NEIGHBOR_ORIENTED","prediction_mode":"REDUCED","encoder":"HYBRID","a":0,"omega":14,"nu_min":-1,"nu_max":3,"phi":3,"psi":0,"p":3,"t_inc":2048,"rows":256,"columns":256,"bands":120,"crop_row":350,"crop_col":214,"sample_count":7864320,"header_bits":176,"payload_bits":57914353,"padding_bits":31,"stream_bits":57914560,"external_info_bits":2040,"max_error":0,"rmse_dn":0.0,"source_study":"omega_fig418_256","source_job":"9d9a9ba83a903231"},{"image":"santarem","capture":"santarem_2022-09-24T13-52-04Z-l1a.nc","crop_sha256":"b58b314a249ed8a9a95bcc01108cb3daf8cef364166d1910ace1eb371ae7c1df","payload_sha256":"cc2c8668eeb2015f012ed31acc21d600432a4f2d82c7e73c33464e94ed844cd9","local_sum":"NARROW_NEIGHBOR_ORIENTED","prediction_mode":"REDUCED","encoder":"HYBRID","a":1,"omega":19,"nu_min":-1,"nu_max":3,"phi":3,"psi":7,"p":3,"t_inc":2048,"rows":256,"columns":256,"bands":120,"crop_row":350,"crop_col":214,"sample_count":7864320,"header_bits":200,"payload_bits":45413648,"padding_bits":40,"stream_bits":45413888,"external_info_bits":2040,"max_error":1,"rmse_dn":0.8162535991657505,"source_study":"omega_fig418_256","source_job":"13d692291e20e4f6"},{"image":"santarem","capture":"santarem_2022-09-24T13-52-04Z-l1a.nc","crop_sha256":"b58b314a249ed8a9a95bcc01108cb3daf8cef364166d1910ace1eb371ae7c1df","payload_sha256":"49e0fb3dd403d515ed46833aaddcbe74c4a0cdd030b852773a0e7e626b92dd56","local_sum":"NARROW_NEIGHBOR_ORIENTED","prediction_mode":"REDUCED","encoder":"HYBRID","a":1,"omega":12,"nu_min":-1,"nu_max":3,"phi":3,"psi":7,"p":3,"t_inc":2048,"rows":256,"columns":256,"bands":120,"crop_row":350,"crop_col":214,"sample_count":7864320,"header_bits":200,"payload_bits":45419926,"padding_bits":34,"stream_bits":45420160,"external_info_bits":2040,"max_error":1,"rmse_dn":0.8164022004484893,"source_study":"omega_fig418_256","source_job":"a39539e65e95b51a"},{"image":"santarem","capture":"santarem_2022-09-24T13-52-04Z-l1a.nc","crop_sha256":"b58b314a249ed8a9a95bcc01108cb3daf8cef364166d1910ace1eb371ae7c1df","payload_sha256":"80aae58cec95f26a717916e0427fa33eff49222db84531c473df724d2850aabc","local_sum":"NARROW_NEIGHBOR_ORIENTED","prediction_mode":"REDUCED","encoder":"HYBRID","a":1,"omega":14,"nu_min":-1,"nu_max":3,"phi":3,"psi":7,"p":3,"t_inc":2048,"rows":256,"columns":256,"bands":120,"crop_row":350,"crop_col":214,"sample_count":7864320,"header_bits":200,"payload_bits":45412945,"padding_bits":39,"stream_bits":45413184,"external_info_bits":2040,"max_error":1,"rmse_dn":0.8165703956795521,"source_study":"omega_fig418_256","source_job":"24dcb7cd50779346"},{"image":"santarem","capture":"santarem_2022-09-24T13-52-04Z-l1a.nc","crop_sha256":"b58b314a249ed8a9a95bcc01108cb3daf8cef364166d1910ace1eb371ae7c1df","payload_sha256":"53430695268326ffee7b4bd569958db08dea9911d92c5d2439457b67342da381","local_sum":"NARROW_NEIGHBOR_ORIENTED","prediction_mode":"REDUCED","encoder":"HYBRID","a":3,"omega":19,"nu_min":-1,"nu_max":3,"phi":3,"psi":7,"p":3,"t_inc":2048,"rows":256,"columns":256,"bands":120,"crop_row":350,"crop_col":214,"sample_count":7864320,"header_bits":200,"payload_bits":35755187,"padding_bits":5,"stream_bits":35755392,"external_info_bits":2040,"max_error":3,"rmse_dn":2.000037606203654,"source_study":"omega_fig418_256","source_job":"b0158158900fcbed"},{"image":"santarem","capture":"santarem_2022-09-24T13-52-04Z-l1a.nc","crop_sha256":"b58b314a249ed8a9a95bcc01108cb3daf8cef364166d1910ace1eb371ae7c1df","payload_sha256":"e7e99b67d91ff8508da8eabc2870cca7796149572eb1d48576179154d365d464","local_sum":"NARROW_NEIGHBOR_ORIENTED","prediction_mode":"REDUCED","encoder":"HYBRID","a":3,"omega":12,"nu_min":-1,"nu_max":3,"phi":3,"psi":7,"p":3,"t_inc":2048,"rows":256,"columns":256,"bands":120,"crop_row":350,"crop_col":214,"sample_count":7864320,"header_bits":200,"payload_bits":35764028,"padding_bits":60,"stream_bits":35764288,"external_info_bits":2040,"max_error":3,"rmse_dn":2.0000521653048025,"source_study":"omega_fig418_256","source_job":"4469e9f723fcce8b"},{"image":"santarem","capture":"santarem_2022-09-24T13-52-04Z-l1a.nc","crop_sha256":"b58b314a249ed8a9a95bcc01108cb3daf8cef364166d1910ace1eb371ae7c1df","payload_sha256":"089e6fb0d92c6f4fd0785d09a8c1f5082f1e35e10f0b74a2d3a37eb18392ca43","local_sum":"NARROW_NEIGHBOR_ORIENTED","prediction_mode":"REDUCED","encoder":"HYBRID","a":3,"omega":14,"nu_min":-1,"nu_max":3,"phi":3,"psi":7,"p":3,"t_inc":2048,"rows":256,"columns":256,"bands":120,"crop_row":350,"crop_col":214,"sample_count":7864320,"header_bits":200,"payload_bits":35756434,"padding_bits":38,"stream_bits":35756672,"external_info_bits":2040,"max_error":3,"rmse_dn":2.000248194235322,"source_study":"omega_fig418_256","source_job":"c2f20cbeb59969ba"},{"image":"santarem","capture":"santarem_2022-09-24T13-52-04Z-l1a.nc","crop_sha256":"b58b314a249ed8a9a95bcc01108cb3daf8cef364166d1910ace1eb371ae7c1df","payload_sha256":"ec5fbcc7270715be34bfbf4c4a771dffcdece555d16f95a4d8e32057ad47222d","local_sum":"NARROW_NEIGHBOR_ORIENTED","prediction_mode":"REDUCED","encoder":"HYBRID","a":7,"omega":19,"nu_min":-1,"nu_max":3,"phi":3,"psi":7,"p":3,"t_inc":2048,"rows":256,"columns":256,"bands":120,"crop_row":350,"crop_col":214,"sample_count":7864320,"header_bits":200,"payload_bits":26649866,"padding_bits":46,"stream_bits":26650112,"external_info_bits":2040,"max_error":7,"rmse_dn":4.319905918297797,"source_study":"omega_fig418_256","source_job":"ccf422bb7adc86d9"},{"image":"santarem","capture":"santarem_2022-09-24T13-52-04Z-l1a.nc","crop_sha256":"b58b314a249ed8a9a95bcc01108cb3daf8cef364166d1910ace1eb371ae7c1df","payload_sha256":"6989eb94c33f0fa7fb42caee1ed6c6c18c5a8ca3976b3b88342357e719bac838","local_sum":"NARROW_NEIGHBOR_ORIENTED","prediction_mode":"REDUCED","encoder":"HYBRID","a":7,"omega":12,"nu_min":-1,"nu_max":3,"phi":3,"psi":7,"p":3,"t_inc":2048,"rows":256,"columns":256,"bands":120,"crop_row":350,"crop_col":214,"sample_count":7864320,"header_bits":200,"payload_bits":26653949,"padding_bits":59,"stream_bits":26654208,"external_info_bits":2040,"max_error":7,"rmse_dn":4.321695639124178,"source_study":"omega_fig418_256","source_job":"b67fbf57b469fbd5"},{"image":"santarem","capture":"santarem_2022-09-24T13-52-04Z-l1a.nc","crop_sha256":"b58b314a249ed8a9a95bcc01108cb3daf8cef364166d1910ace1eb371ae7c1df","payload_sha256":"ba5c413789744479f28ec87dfa5b72f8e853e5181d4564a1960f42c64b9d74bf","local_sum":"NARROW_NEIGHBOR_ORIENTED","prediction_mode":"REDUCED","encoder":"HYBRID","a":7,"omega":14,"nu_min":-1,"nu_max":3,"phi":3,"psi":7,"p":3,"t_inc":2048,"rows":256,"columns":256,"bands":120,"crop_row":350,"crop_col":214,"sample_count":7864320,"header_bits":200,"payload_bits":26648289,"padding_bits":23,"stream_bits":26648512,"external_info_bits":2040,"max_error":7,"rmse_dn":4.320636419348036,"source_study":"omega_fig418_256","source_job":"0f4774a42e1c4f8d"},{"image":"santarem","capture":"santarem_2022-09-24T13-52-04Z-l1a.nc","crop_sha256":"b58b314a249ed8a9a95bcc01108cb3daf8cef364166d1910ace1eb371ae7c1df","payload_sha256":"6f728dde85708b18d31ea5cfbf45f05198df6d1141d6647919cb04c3c5192a43","local_sum":"NARROW_NEIGHBOR_ORIENTED","prediction_mode":"REDUCED","encoder":"HYBRID","a":15,"omega":19,"nu_min":-1,"nu_max":3,"phi":3,"psi":7,"p":3,"t_inc":2048,"rows":256,"columns":256,"bands":120,"crop_row":350,"crop_col":214,"sample_count":7864320,"header_bits":200,"payload_bits":18598357,"padding_bits":35,"stream_bits":18598592,"external_info_bits":2040,"max_error":15,"rmse_dn":8.929379631157516,"source_study":"omega_fig418_256","source_job":"c744769d794a5cd0"},{"image":"santarem","capture":"santarem_2022-09-24T13-52-04Z-l1a.nc","crop_sha256":"b58b314a249ed8a9a95bcc01108cb3daf8cef364166d1910ace1eb371ae7c1df","payload_sha256":"e4e7821baee468ad518eefcc399f811dc2bce5b6d4e4dc5b8baeaf72d5a8de6a","local_sum":"NARROW_NEIGHBOR_ORIENTED","prediction_mode":"REDUCED","encoder":"HYBRID","a":15,"omega":12,"nu_min":-1,"nu_max":3,"phi":3,"psi":7,"p":3,"t_inc":2048,"rows":256,"columns":256,"bands":120,"crop_row":350,"crop_col":214,"sample_count":7864320,"header_bits":200,"payload_bits":18605200,"padding_bits":40,"stream_bits":18605440,"external_info_bits":2040,"max_error":15,"rmse_dn":8.930926301365563,"source_study":"omega_fig418_256","source_job":"4edce670bbf48ec1"},{"image":"santarem","capture":"santarem_2022-09-24T13-52-04Z-l1a.nc","crop_sha256":"b58b314a249ed8a9a95bcc01108cb3daf8cef364166d1910ace1eb371ae7c1df","payload_sha256":"e842e7c5b509e6e7b008db2180e43d75cb84e5332dd710b2cba7cfabbdf63607","local_sum":"NARROW_NEIGHBOR_ORIENTED","prediction_mode":"REDUCED","encoder":"HYBRID","a":15,"omega":14,"nu_min":-1,"nu_max":3,"phi":3,"psi":7,"p":3,"t_inc":2048,"rows":256,"columns":256,"bands":120,"crop_row":350,"crop_col":214,"sample_count":7864320,"header_bits":200,"payload_bits":18598734,"padding_bits":42,"stream_bits":18598976,"external_info_bits":2040,"max_error":15,"rmse_dn":8.930759853032155,"source_study":"omega_fig418_256","source_job":"1e46e160040bde8d"},{"image":"doha","capture":"doha_2022-09-12T06-38-31Z-l1a.nc","crop_sha256":"6905c8b2f23714aeb52966a39d23e3503eaff8f4003fd9c28fa5f52f48261b90","payload_sha256":"1abe1817d28e19f1ebd7d4cb8aec33138486b4a8e45b27482ca1852c24f36034","local_sum":"NARROW_NEIGHBOR_ORIENTED","prediction_mode":"REDUCED","encoder":"HYBRID","a":0,"omega":19,"nu_min":-1,"nu_max":3,"phi":3,"psi":0,"p":3,"t_inc":2048,"rows":256,"columns":256,"bands":120,"crop_row":350,"crop_col":214,"sample_count":7864320,"header_bits":176,"payload_bits":62656981,"padding_bits":59,"stream_bits":62657216,"external_info_bits":2040,"max_error":0,"rmse_dn":0.0,"source_study":"omega_fig418_256","source_job":"e67fd76ce589f3a2"},{"image":"doha","capture":"doha_2022-09-12T06-38-31Z-l1a.nc","crop_sha256":"6905c8b2f23714aeb52966a39d23e3503eaff8f4003fd9c28fa5f52f48261b90","payload_sha256":"1451fc8673520cffa41db1cd5455fea87893439a26b38a4c57e96bd13e9e6779","local_sum":"NARROW_NEIGHBOR_ORIENTED","prediction_mode":"REDUCED","encoder":"HYBRID","a":0,"omega":12,"nu_min":-1,"nu_max":3,"phi":3,"psi":0,"p":3,"t_inc":2048,"rows":256,"columns":256,"bands":120,"crop_row":350,"crop_col":214,"sample_count":7864320,"header_bits":176,"payload_bits":62660340,"padding_bits":28,"stream_bits":62660544,"external_info_bits":2040,"max_error":0,"rmse_dn":0.0,"source_study":"omega_fig418_256","source_job":"14f7a96e714fc4da"},{"image":"doha","capture":"doha_2022-09-12T06-38-31Z-l1a.nc","crop_sha256":"6905c8b2f23714aeb52966a39d23e3503eaff8f4003fd9c28fa5f52f48261b90","payload_sha256":"9aff6892a043dba961576e2124d1e3fe02e0b5f9c22b16857c3820e7432edcf2","local_sum":"NARROW_NEIGHBOR_ORIENTED","prediction_mode":"REDUCED","encoder":"HYBRID","a":0,"omega":14,"nu_min":-1,"nu_max":3,"phi":3,"psi":0,"p":3,"t_inc":2048,"rows":256,"columns":256,"bands":120,"crop_row":350,"crop_col":214,"sample_count":7864320,"header_bits":176,"payload_bits":62656289,"padding_bits":47,"stream_bits":62656512,"external_info_bits":2040,"max_error":0,"rmse_dn":0.0,"source_study":"omega_fig418_256","source_job":"672fbb81c07841e5"},{"image":"doha","capture":"doha_2022-09-12T06-38-31Z-l1a.nc","crop_sha256":"6905c8b2f23714aeb52966a39d23e3503eaff8f4003fd9c28fa5f52f48261b90","payload_sha256":"be56e5ef1c79e0fe8e7c7136319a465f6c5fb5678d05b82a3e52fe871d60a595","local_sum":"NARROW_NEIGHBOR_ORIENTED","prediction_mode":"REDUCED","encoder":"HYBRID","a":1,"omega":19,"nu_min":-1,"nu_max":3,"phi":3,"psi":7,"p":3,"t_inc":2048,"rows":256,"columns":256,"bands":120,"crop_row":350,"crop_col":214,"sample_count":7864320,"header_bits":200,"payload_bits":50199369,"padding_bits":47,"stream_bits":50199616,"external_info_bits":2040,"max_error":1,"rmse_dn":0.8167088192811048,"source_study":"omega_fig418_256","source_job":"059e1166e8c07a5c"},{"image":"doha","capture":"doha_2022-09-12T06-38-31Z-l1a.nc","crop_sha256":"6905c8b2f23714aeb52966a39d23e3503eaff8f4003fd9c28fa5f52f48261b90","payload_sha256":"90ee63a9e1afecb25cfa06d1bdf1844661b4957d3727259033c5bb9c03021bf8","local_sum":"NARROW_NEIGHBOR_ORIENTED","prediction_mode":"REDUCED","encoder":"HYBRID","a":1,"omega":12,"nu_min":-1,"nu_max":3,"phi":3,"psi":7,"p":3,"t_inc":2048,"rows":256,"columns":256,"bands":120,"crop_row":350,"crop_col":214,"sample_count":7864320,"header_bits":200,"payload_bits":50204733,"padding_bits":59,"stream_bits":50204992,"external_info_bits":2040,"max_error":1,"rmse_dn":0.8165890819000615,"source_study":"omega_fig418_256","source_job":"8c0164633dc1b944"},{"image":"doha","capture":"doha_2022-09-12T06-38-31Z-l1a.nc","crop_sha256":"6905c8b2f23714aeb52966a39d23e3503eaff8f4003fd9c28fa5f52f48261b90","payload_sha256":"77f30e485b995ab6489b751309aef3dd0ea91cbf982550267c040442ee189615","local_sum":"NARROW_NEIGHBOR_ORIENTED","prediction_mode":"REDUCED","encoder":"HYBRID","a":1,"omega":14,"nu_min":-1,"nu_max":3,"phi":3,"psi":7,"p":3,"t_inc":2048,"rows":256,"columns":256,"bands":120,"crop_row":350,"crop_col":214,"sample_count":7864320,"header_bits":200,"payload_bits":50199487,"padding_bits":57,"stream_bits":50199744,"external_info_bits":2040,"max_error":1,"rmse_dn":0.8164816302919292,"source_study":"omega_fig418_256","source_job":"4aed1e5634eb72ad"},{"image":"doha","capture":"doha_2022-09-12T06-38-31Z-l1a.nc","crop_sha256":"6905c8b2f23714aeb52966a39d23e3503eaff8f4003fd9c28fa5f52f48261b90","payload_sha256":"60ed217fc7e08ed013225beba988c5d904c2fd214f1074cdc2c29d6655744ef0","local_sum":"NARROW_NEIGHBOR_ORIENTED","prediction_mode":"REDUCED","encoder":"HYBRID","a":3,"omega":19,"nu_min":-1,"nu_max":3,"phi":3,"psi":7,"p":3,"t_inc":2048,"rows":256,"columns":256,"bands":120,"crop_row":350,"crop_col":214,"sample_count":7864320,"header_bits":200,"payload_bits":40580201,"padding_bits":15,"stream_bits":40580416,"external_info_bits":2040,"max_error":3,"rmse_dn":1.9998907377279227,"source_study":"omega_fig418_256","source_job":"2859d19372327d52"},{"image":"doha","capture":"doha_2022-09-12T06-38-31Z-l1a.nc","crop_sha256":"6905c8b2f23714aeb52966a39d23e3503eaff8f4003fd9c28fa5f52f48261b90","payload_sha256":"5c0e22e66caa445ea8c15ead479de874fe063682932fb746b6b5d8d63f4ce546","local_sum":"NARROW_NEIGHBOR_ORIENTED","prediction_mode":"REDUCED","encoder":"HYBRID","a":3,"omega":12,"nu_min":-1,"nu_max":3,"phi":3,"psi":7,"p":3,"t_inc":2048,"rows":256,"columns":256,"bands":120,"crop_row":350,"crop_col":214,"sample_count":7864320,"header_bits":200,"payload_bits":40587607,"padding_bits":33,"stream_bits":40587840,"external_info_bits":2040,"max_error":3,"rmse_dn":1.9994556321944061,"source_study":"omega_fig418_256","source_job":"de2456306bd07474"},{"image":"doha","capture":"doha_2022-09-12T06-38-31Z-l1a.nc","crop_sha256":"6905c8b2f23714aeb52966a39d23e3503eaff8f4003fd9c28fa5f52f48261b90","payload_sha256":"b398051a81a9d83cb742ee7b7d5d94ba85d53c5f08e4bc174d65600ad4a62680","local_sum":"NARROW_NEIGHBOR_ORIENTED","prediction_mode":"REDUCED","encoder":"HYBRID","a":3,"omega":14,"nu_min":-1,"nu_max":3,"phi":3,"psi":7,"p":3,"t_inc":2048,"rows":256,"columns":256,"bands":120,"crop_row":350,"crop_col":214,"sample_count":7864320,"header_bits":200,"payload_bits":40577473,"padding_bits":55,"stream_bits":40577728,"external_info_bits":2040,"max_error":3,"rmse_dn":2.0002135798536425,"source_study":"omega_fig418_256","source_job":"e8a610cffef88851"},{"image":"doha","capture":"doha_2022-09-12T06-38-31Z-l1a.nc","crop_sha256":"6905c8b2f23714aeb52966a39d23e3503eaff8f4003fd9c28fa5f52f48261b90","payload_sha256":"a47f896e82e137102c6946eeb9611e73fed61ccd69c52cbe0b8249e5008fb2bc","local_sum":"NARROW_NEIGHBOR_ORIENTED","prediction_mode":"REDUCED","encoder":"HYBRID","a":7,"omega":19,"nu_min":-1,"nu_max":3,"phi":3,"psi":7,"p":3,"t_inc":2048,"rows":256,"columns":256,"bands":120,"crop_row":350,"crop_col":214,"sample_count":7864320,"header_bits":200,"payload_bits":31775078,"padding_bits":18,"stream_bits":31775296,"external_info_bits":2040,"max_error":7,"rmse_dn":4.32091830595232,"source_study":"omega_fig418_256","source_job":"e719fb87762e5583"},{"image":"doha","capture":"doha_2022-09-12T06-38-31Z-l1a.nc","crop_sha256":"6905c8b2f23714aeb52966a39d23e3503eaff8f4003fd9c28fa5f52f48261b90","payload_sha256":"8beea2b1fedd1844c8377a4beccf8e233523908014d2f9107c1d4c4c16910ce0","local_sum":"NARROW_NEIGHBOR_ORIENTED","prediction_mode":"REDUCED","encoder":"HYBRID","a":7,"omega":12,"nu_min":-1,"nu_max":3,"phi":3,"psi":7,"p":3,"t_inc":2048,"rows":256,"columns":256,"bands":120,"crop_row":350,"crop_col":214,"sample_count":7864320,"header_bits":200,"payload_bits":31781877,"padding_bits":3,"stream_bits":31782080,"external_info_bits":2040,"max_error":7,"rmse_dn":4.320054238885484,"source_study":"omega_fig418_256","source_job":"67a8e771c6e99263"},{"image":"doha","capture":"doha_2022-09-12T06-38-31Z-l1a.nc","crop_sha256":"6905c8b2f23714aeb52966a39d23e3503eaff8f4003fd9c28fa5f52f48261b90","payload_sha256":"e0d4c5e6ef92c8530ec26797079718d2233e394d6234194baba2514067e34ca4","local_sum":"NARROW_NEIGHBOR_ORIENTED","prediction_mode":"REDUCED","encoder":"HYBRID","a":7,"omega":14,"nu_min":-1,"nu_max":3,"phi":3,"psi":7,"p":3,"t_inc":2048,"rows":256,"columns":256,"bands":120,"crop_row":350,"crop_col":214,"sample_count":7864320,"header_bits":200,"payload_bits":31779161,"padding_bits":31,"stream_bits":31779392,"external_info_bits":2040,"max_error":7,"rmse_dn":4.3211560493299865,"source_study":"omega_fig418_256","source_job":"350bd3e5abfd95a5"},{"image":"doha","capture":"doha_2022-09-12T06-38-31Z-l1a.nc","crop_sha256":"6905c8b2f23714aeb52966a39d23e3503eaff8f4003fd9c28fa5f52f48261b90","payload_sha256":"17ee30e2aa72761889621c679bb9d940725944ed805e18ca430c468047bc0dca","local_sum":"NARROW_NEIGHBOR_ORIENTED","prediction_mode":"REDUCED","encoder":"HYBRID","a":15,"omega":19,"nu_min":-1,"nu_max":3,"phi":3,"psi":7,"p":3,"t_inc":2048,"rows":256,"columns":256,"bands":120,"crop_row":350,"crop_col":214,"sample_count":7864320,"header_bits":200,"payload_bits":23245940,"padding_bits":4,"stream_bits":23246144,"external_info_bits":2040,"max_error":15,"rmse_dn":8.936341430691508,"source_study":"omega_fig418_256","source_job":"d9cdefb18e3e23aa"},{"image":"doha","capture":"doha_2022-09-12T06-38-31Z-l1a.nc","crop_sha256":"6905c8b2f23714aeb52966a39d23e3503eaff8f4003fd9c28fa5f52f48261b90","payload_sha256":"321bc63143f50b44c6bbf4f761aa58e14dc72dc1646876e492e1804773b1e4f8","local_sum":"NARROW_NEIGHBOR_ORIENTED","prediction_mode":"REDUCED","encoder":"HYBRID","a":15,"omega":12,"nu_min":-1,"nu_max":3,"phi":3,"psi":7,"p":3,"t_inc":2048,"rows":256,"columns":256,"bands":120,"crop_row":350,"crop_col":214,"sample_count":7864320,"header_bits":200,"payload_bits":23253659,"padding_bits":29,"stream_bits":23253888,"external_info_bits":2040,"max_error":15,"rmse_dn":8.934657016825032,"source_study":"omega_fig418_256","source_job":"0d2ed6e60df6d257"},{"image":"doha","capture":"doha_2022-09-12T06-38-31Z-l1a.nc","crop_sha256":"6905c8b2f23714aeb52966a39d23e3503eaff8f4003fd9c28fa5f52f48261b90","payload_sha256":"c5de2f208a3de27c79a6e88d8ef692ee317a43ec3911e632659a87e1b70f32ce","local_sum":"NARROW_NEIGHBOR_ORIENTED","prediction_mode":"REDUCED","encoder":"HYBRID","a":15,"omega":14,"nu_min":-1,"nu_max":3,"phi":3,"psi":7,"p":3,"t_inc":2048,"rows":256,"columns":256,"bands":120,"crop_row":350,"crop_col":214,"sample_count":7864320,"header_bits":200,"payload_bits":23242990,"padding_bits":10,"stream_bits":23243200,"external_info_bits":2040,"max_error":15,"rmse_dn":8.937499708340157,"source_study":"omega_fig418_256","source_job":"da7fd5b9a1d97e72"},{"image":"finnmark","capture":"finnmark_2022-07-06T19-40-03Z-l1a.nc","crop_sha256":"66d85dcba6cf40a76cb5e9544d2ef78f5fd06e9a7b99629924f8dc0514f954cb","payload_sha256":"81b4613d4e663cb7ca305694e711eec6e497db272d83fc89e431b6fab79ad671","local_sum":"NARROW_NEIGHBOR_ORIENTED","prediction_mode":"REDUCED","encoder":"HYBRID","a":0,"omega":19,"nu_min":-1,"nu_max":3,"phi":3,"psi":0,"p":3,"t_inc":2048,"rows":256,"columns":256,"bands":120,"crop_row":350,"crop_col":214,"sample_count":7864320,"header_bits":176,"payload_bits":56661428,"padding_bits":28,"stream_bits":56661632,"external_info_bits":2040,"max_error":0,"rmse_dn":0.0,"source_study":"omega_fig418_256","source_job":"269994a4dfa53b54"},{"image":"finnmark","capture":"finnmark_2022-07-06T19-40-03Z-l1a.nc","crop_sha256":"66d85dcba6cf40a76cb5e9544d2ef78f5fd06e9a7b99629924f8dc0514f954cb","payload_sha256":"532373acb3b6644c317fc9c73c2d849f2efe97698bd85afce1df94d6ff8c6cd3","local_sum":"NARROW_NEIGHBOR_ORIENTED","prediction_mode":"REDUCED","encoder":"HYBRID","a":0,"omega":12,"nu_min":-1,"nu_max":3,"phi":3,"psi":0,"p":3,"t_inc":2048,"rows":256,"columns":256,"bands":120,"crop_row":350,"crop_col":214,"sample_count":7864320,"header_bits":176,"payload_bits":56662293,"padding_bits":59,"stream_bits":56662528,"external_info_bits":2040,"max_error":0,"rmse_dn":0.0,"source_study":"omega_fig418_256","source_job":"b367be9665688f6d"},{"image":"finnmark","capture":"finnmark_2022-07-06T19-40-03Z-l1a.nc","crop_sha256":"66d85dcba6cf40a76cb5e9544d2ef78f5fd06e9a7b99629924f8dc0514f954cb","payload_sha256":"00cf8fff4a23fe79fab5cb7a1fdabf5060304fa465ead9471ae75c3330b531fc","local_sum":"NARROW_NEIGHBOR_ORIENTED","prediction_mode":"REDUCED","encoder":"HYBRID","a":0,"omega":14,"nu_min":-1,"nu_max":3,"phi":3,"psi":0,"p":3,"t_inc":2048,"rows":256,"columns":256,"bands":120,"crop_row":350,"crop_col":214,"sample_count":7864320,"header_bits":176,"payload_bits":56657964,"padding_bits":36,"stream_bits":56658176,"external_info_bits":2040,"max_error":0,"rmse_dn":0.0,"source_study":"omega_fig418_256","source_job":"1517fdd32f45b13d"},{"image":"finnmark","capture":"finnmark_2022-07-06T19-40-03Z-l1a.nc","crop_sha256":"66d85dcba6cf40a76cb5e9544d2ef78f5fd06e9a7b99629924f8dc0514f954cb","payload_sha256":"e73c17b7a43bb15d9787ef1f7cd2996c52eed83b1ac5f61d51ae780a7559a077","local_sum":"NARROW_NEIGHBOR_ORIENTED","prediction_mode":"REDUCED","encoder":"HYBRID","a":1,"omega":19,"nu_min":-1,"nu_max":3,"phi":3,"psi":7,"p":3,"t_inc":2048,"rows":256,"columns":256,"bands":120,"crop_row":350,"crop_col":214,"sample_count":7864320,"header_bits":200,"payload_bits":44175186,"padding_bits":38,"stream_bits":44175424,"external_info_bits":2040,"max_error":1,"rmse_dn":0.8164772696382536,"source_study":"omega_fig418_256","source_job":"0576d3d15405d2a6"},{"image":"finnmark","capture":"finnmark_2022-07-06T19-40-03Z-l1a.nc","crop_sha256":"66d85dcba6cf40a76cb5e9544d2ef78f5fd06e9a7b99629924f8dc0514f954cb","payload_sha256":"2912f8cf292ba02338a2062908b1c719e617fad2ed198c0160239d68d6ec0731","local_sum":"NARROW_NEIGHBOR_ORIENTED","prediction_mode":"REDUCED","encoder":"HYBRID","a":1,"omega":12,"nu_min":-1,"nu_max":3,"phi":3,"psi":7,"p":3,"t_inc":2048,"rows":256,"columns":256,"bands":120,"crop_row":350,"crop_col":214,"sample_count":7864320,"header_bits":200,"payload_bits":44181950,"padding_bits":58,"stream_bits":44182208,"external_info_bits":2040,"max_error":1,"rmse_dn":0.8164930768970232,"source_study":"omega_fig418_256","source_job":"fc3417d354fe026a"},{"image":"finnmark","capture":"finnmark_2022-07-06T19-40-03Z-l1a.nc","crop_sha256":"66d85dcba6cf40a76cb5e9544d2ef78f5fd06e9a7b99629924f8dc0514f954cb","payload_sha256":"70090e623bef06dc8f7fb6c60d018f20704df5af297c3fb371c4d21e83eda0c4","local_sum":"NARROW_NEIGHBOR_ORIENTED","prediction_mode":"REDUCED","encoder":"HYBRID","a":1,"omega":14,"nu_min":-1,"nu_max":3,"phi":3,"psi":7,"p":3,"t_inc":2048,"rows":256,"columns":256,"bands":120,"crop_row":350,"crop_col":214,"sample_count":7864320,"header_bits":200,"payload_bits":44174986,"padding_bits":46,"stream_bits":44175232,"external_info_bits":2040,"max_error":1,"rmse_dn":0.816545402191157,"source_study":"omega_fig418_256","source_job":"d17da925580a0739"},{"image":"finnmark","capture":"finnmark_2022-07-06T19-40-03Z-l1a.nc","crop_sha256":"66d85dcba6cf40a76cb5e9544d2ef78f5fd06e9a7b99629924f8dc0514f954cb","payload_sha256":"273f1ddb77e5ded7d24dcc171f90e8921deed3c7eae5f6c177c2e32029f39a7c","local_sum":"NARROW_NEIGHBOR_ORIENTED","prediction_mode":"REDUCED","encoder":"HYBRID","a":3,"omega":19,"nu_min":-1,"nu_max":3,"phi":3,"psi":7,"p":3,"t_inc":2048,"rows":256,"columns":256,"bands":120,"crop_row":350,"crop_col":214,"sample_count":7864320,"header_bits":200,"payload_bits":34496564,"padding_bits":4,"stream_bits":34496768,"external_info_bits":2040,"max_error":3,"rmse_dn":2.000198990141421,"source_study":"omega_fig418_256","source_job":"a75358bdf9ad69ee"},{"image":"finnmark","capture":"finnmark_2022-07-06T19-40-03Z-l1a.nc","crop_sha256":"66d85dcba6cf40a76cb5e9544d2ef78f5fd06e9a7b99629924f8dc0514f954cb","payload_sha256":"1ea06379002bbb7329a0eaac6972eddbad2ff8c015dc16cb90c6ba9774d59f1e","local_sum":"NARROW_NEIGHBOR_ORIENTED","prediction_mode":"REDUCED","encoder":"HYBRID","a":3,"omega":12,"nu_min":-1,"nu_max":3,"phi":3,"psi":7,"p":3,"t_inc":2048,"rows":256,"columns":256,"bands":120,"crop_row":350,"crop_col":214,"sample_count":7864320,"header_bits":200,"payload_bits":34506885,"padding_bits":51,"stream_bits":34507136,"external_info_bits":2040,"max_error":3,"rmse_dn":2.000170891136555,"source_study":"omega_fig418_256","source_job":"abc18e86599c53e1"},{"image":"finnmark","capture":"finnmark_2022-07-06T19-40-03Z-l1a.nc","crop_sha256":"66d85dcba6cf40a76cb5e9544d2ef78f5fd06e9a7b99629924f8dc0514f954cb","payload_sha256":"99e3168ef2a1f0e56887cb9e518e12347568a92ca50132b9fe2d1348cdf4ab58","local_sum":"NARROW_NEIGHBOR_ORIENTED","prediction_mode":"REDUCED","encoder":"HYBRID","a":3,"omega":14,"nu_min":-1,"nu_max":3,"phi":3,"psi":7,"p":3,"t_inc":2048,"rows":256,"columns":256,"bands":120,"crop_row":350,"crop_col":214,"sample_count":7864320,"header_bits":200,"payload_bits":34496439,"padding_bits":1,"stream_bits":34496640,"external_info_bits":2040,"max_error":3,"rmse_dn":1.9996355996359387,"source_study":"omega_fig418_256","source_job":"db21f13c77b8b809"},{"image":"finnmark","capture":"finnmark_2022-07-06T19-40-03Z-l1a.nc","crop_sha256":"66d85dcba6cf40a76cb5e9544d2ef78f5fd06e9a7b99629924f8dc0514f954cb","payload_sha256":"92dd4578b392d6cdda6c55331dcc25fb340b36366dea5dfd81fc338b14baec93","local_sum":"NARROW_NEIGHBOR_ORIENTED","prediction_mode":"REDUCED","encoder":"HYBRID","a":7,"omega":19,"nu_min":-1,"nu_max":3,"phi":3,"psi":7,"p":3,"t_inc":2048,"rows":256,"columns":256,"bands":120,"crop_row":350,"crop_col":214,"sample_count":7864320,"header_bits":200,"payload_bits":25670297,"padding_bits":31,"stream_bits":25670528,"external_info_bits":2040,"max_error":7,"rmse_dn":4.321015138151841,"source_study":"omega_fig418_256","source_job":"e30879c28dc368b8"},{"image":"finnmark","capture":"finnmark_2022-07-06T19-40-03Z-l1a.nc","crop_sha256":"66d85dcba6cf40a76cb5e9544d2ef78f5fd06e9a7b99629924f8dc0514f954cb","payload_sha256":"3a885cca4ee70cf7b9c6cd22ed9a0e34513ffbc8476635b90c3504aca1c7efc7","local_sum":"NARROW_NEIGHBOR_ORIENTED","prediction_mode":"REDUCED","encoder":"HYBRID","a":7,"omega":12,"nu_min":-1,"nu_max":3,"phi":3,"psi":7,"p":3,"t_inc":2048,"rows":256,"columns":256,"bands":120,"crop_row":350,"crop_col":214,"sample_count":7864320,"header_bits":200,"payload_bits":25683987,"padding_bits":37,"stream_bits":25684224,"external_info_bits":2040,"max_error":7,"rmse_dn":4.320232355203359,"source_study":"omega_fig418_256","source_job":"3ecb591802050d30"},{"image":"finnmark","capture":"finnmark_2022-07-06T19-40-03Z-l1a.nc","crop_sha256":"66d85dcba6cf40a76cb5e9544d2ef78f5fd06e9a7b99629924f8dc0514f954cb","payload_sha256":"8380e6efa6c4a42f1bb70e8bbb10249d39e34df0b2c776376f2a343bc425e009","local_sum":"NARROW_NEIGHBOR_ORIENTED","prediction_mode":"REDUCED","encoder":"HYBRID","a":7,"omega":14,"nu_min":-1,"nu_max":3,"phi":3,"psi":7,"p":3,"t_inc":2048,"rows":256,"columns":256,"bands":120,"crop_row":350,"crop_col":214,"sample_count":7864320,"header_bits":200,"payload_bits":25676942,"padding_bits":42,"stream_bits":25677184,"external_info_bits":2040,"max_error":7,"rmse_dn":4.3193841655378025,"source_study":"omega_fig418_256","source_job":"4dbff306907c8482"},{"image":"finnmark","capture":"finnmark_2022-07-06T19-40-03Z-l1a.nc","crop_sha256":"66d85dcba6cf40a76cb5e9544d2ef78f5fd06e9a7b99629924f8dc0514f954cb","payload_sha256":"30106deff9fe0faa86ba059f07156871c052349970be08c8a522d98d14a1e0b3","local_sum":"NARROW_NEIGHBOR_ORIENTED","prediction_mode":"REDUCED","encoder":"HYBRID","a":15,"omega":19,"nu_min":-1,"nu_max":3,"phi":3,"psi":7,"p":3,"t_inc":2048,"rows":256,"columns":256,"bands":120,"crop_row":350,"crop_col":214,"sample_count":7864320,"header_bits":200,"payload_bits":17955566,"padding_bits":10,"stream_bits":17955776,"external_info_bits":2040,"max_error":15,"rmse_dn":8.927610040985472,"source_study":"omega_fig418_256","source_job":"fdecd6cb20f05124"},{"image":"finnmark","capture":"finnmark_2022-07-06T19-40-03Z-l1a.nc","crop_sha256":"66d85dcba6cf40a76cb5e9544d2ef78f5fd06e9a7b99629924f8dc0514f954cb","payload_sha256":"f31d2fc0b1754d651835bf225ad5c14a84d17c567c53c650ac76b651583f2567","local_sum":"NARROW_NEIGHBOR_ORIENTED","prediction_mode":"REDUCED","encoder":"HYBRID","a":15,"omega":12,"nu_min":-1,"nu_max":3,"phi":3,"psi":7,"p":3,"t_inc":2048,"rows":256,"columns":256,"bands":120,"crop_row":350,"crop_col":214,"sample_count":7864320,"header_bits":200,"payload_bits":17963279,"padding_bits":41,"stream_bits":17963520,"external_info_bits":2040,"max_error":15,"rmse_dn":8.92628238130964,"source_study":"omega_fig418_256","source_job":"d26a086b17b3a0aa"},{"image":"finnmark","capture":"finnmark_2022-07-06T19-40-03Z-l1a.nc","crop_sha256":"66d85dcba6cf40a76cb5e9544d2ef78f5fd06e9a7b99629924f8dc0514f954cb","payload_sha256":"04189468badd0385b592a5b3dc9cd22371adf0114ddf658a32ae90fcb00141c2","local_sum":"NARROW_NEIGHBOR_ORIENTED","prediction_mode":"REDUCED","encoder":"HYBRID","a":15,"omega":14,"nu_min":-1,"nu_max":3,"phi":3,"psi":7,"p":3,"t_inc":2048,"rows":256,"columns":256,"bands":120,"crop_row":350,"crop_col":214,"sample_count":7864320,"header_bits":200,"payload_bits":17952101,"padding_bits":19,"stream_bits":17952320,"external_info_bits":2040,"max_error":15,"rmse_dn":8.925960491225467,"source_study":"omega_fig418_256","source_job":"92d9d91931d9eb8b"},{"image":"lisbon","capture":"lisbon_2022-09-22T10-55-19Z-l1a.nc","crop_sha256":"4205b566d512d7710fb9ac46854722da1f0be1857cf7b724840d900c943b8dd4","payload_sha256":"dff6d6cb06425b69ff17ec927524769abd41ca43f315b0f629e40acdcbd99e73","local_sum":"NARROW_NEIGHBOR_ORIENTED","prediction_mode":"REDUCED","encoder":"HYBRID","a":0,"omega":19,"nu_min":-1,"nu_max":3,"phi":3,"psi":0,"p":3,"t_inc":2048,"rows":956,"columns":684,"bands":120,"crop_row":0,"crop_col":0,"sample_count":78468480,"header_bits":176,"payload_bits":553476104,"padding_bits":8,"stream_bits":553476288,"external_info_bits":2040,"max_error":0,"rmse_dn":0.0,"source_study":"omega_fig417_256","source_job":"6045c64d68d3270f"},{"image":"lisbon","capture":"lisbon_2022-09-22T10-55-19Z-l1a.nc","crop_sha256":"4205b566d512d7710fb9ac46854722da1f0be1857cf7b724840d900c943b8dd4","payload_sha256":"db392926d0700c30a4dd891a46a2a4b39df67d4c26022cae79245aae04d28811","local_sum":"NARROW_NEIGHBOR_ORIENTED","prediction_mode":"REDUCED","encoder":"HYBRID","a":0,"omega":12,"nu_min":-1,"nu_max":3,"phi":3,"psi":0,"p":3,"t_inc":2048,"rows":956,"columns":684,"bands":120,"crop_row":0,"crop_col":0,"sample_count":78468480,"header_bits":176,"payload_bits":554093121,"padding_bits":15,"stream_bits":554093312,"external_info_bits":2040,"max_error":0,"rmse_dn":0.0,"source_study":"omega_fig417_256","source_job":"151ccdb44aad0dd3"},{"image":"lisbon","capture":"lisbon_2022-09-22T10-55-19Z-l1a.nc","crop_sha256":"4205b566d512d7710fb9ac46854722da1f0be1857cf7b724840d900c943b8dd4","payload_sha256":"f2b54862812e29247bd9a2fd11150a5789fc1a0dad9d8a71f8d367c6652c6c1c","local_sum":"NARROW_NEIGHBOR_ORIENTED","prediction_mode":"REDUCED","encoder":"HYBRID","a":0,"omega":14,"nu_min":-1,"nu_max":3,"phi":3,"psi":0,"p":3,"t_inc":2048,"rows":956,"columns":684,"bands":120,"crop_row":0,"crop_col":0,"sample_count":78468480,"header_bits":176,"payload_bits":553481391,"padding_bits":33,"stream_bits":553481600,"external_info_bits":2040,"max_error":0,"rmse_dn":0.0,"source_study":"omega_fig417_256","source_job":"75c9e2da80ff4e73"},{"image":"bangladesh","capture":"bangladesh_2022-10-26T04-02-56Z-l1a.nc","crop_sha256":"75a01f5cd7ebb73a680d53c8d96e4b2d1c5ee6e454b42c6fe89de379618537c6","payload_sha256":"3010c720024e9df6e9bdc3a9a4d48637c004dac7d2ac2fe2c900e34fbe2046ac","local_sum":"NARROW_NEIGHBOR_ORIENTED","prediction_mode":"REDUCED","encoder":"HYBRID","a":0,"omega":19,"nu_min":-1,"nu_max":3,"phi":3,"psi":0,"p":3,"t_inc":2048,"rows":956,"columns":684,"bands":120,"crop_row":0,"crop_col":0,"sample_count":78468480,"header_bits":176,"payload_bits":584374728,"padding_bits":8,"stream_bits":584374912,"external_info_bits":2040,"max_error":0,"rmse_dn":0.0,"source_study":"omega_fig417_256","source_job":"9683006ef1c9d17b"},{"image":"bangladesh","capture":"bangladesh_2022-10-26T04-02-56Z-l1a.nc","crop_sha256":"75a01f5cd7ebb73a680d53c8d96e4b2d1c5ee6e454b42c6fe89de379618537c6","payload_sha256":"469119bd6e1dd7084c8c778e97e3e7bb730c4e1ba017e09279d4c382a60999fb","local_sum":"NARROW_NEIGHBOR_ORIENTED","prediction_mode":"REDUCED","encoder":"HYBRID","a":0,"omega":12,"nu_min":-1,"nu_max":3,"phi":3,"psi":0,"p":3,"t_inc":2048,"rows":956,"columns":684,"bands":120,"crop_row":0,"crop_col":0,"sample_count":78468480,"header_bits":176,"payload_bits":584736017,"padding_bits":63,"stream_bits":584736256,"external_info_bits":2040,"max_error":0,"rmse_dn":0.0,"source_study":"omega_fig417_256","source_job":"ee50870b64b04284"},{"image":"bangladesh","capture":"bangladesh_2022-10-26T04-02-56Z-l1a.nc","crop_sha256":"75a01f5cd7ebb73a680d53c8d96e4b2d1c5ee6e454b42c6fe89de379618537c6","payload_sha256":"8591f3111faab2c32df804a0a6460091c349ec6b528f5251fe86af7cc7c61599","local_sum":"NARROW_NEIGHBOR_ORIENTED","prediction_mode":"REDUCED","encoder":"HYBRID","a":0,"omega":14,"nu_min":-1,"nu_max":3,"phi":3,"psi":0,"p":3,"t_inc":2048,"rows":956,"columns":684,"bands":120,"crop_row":0,"crop_col":0,"sample_count":78468480,"header_bits":176,"payload_bits":584378616,"padding_bits":24,"stream_bits":584378816,"external_info_bits":2040,"max_error":0,"rmse_dn":0.0,"source_study":"omega_fig417_256","source_job":"c9b33c9b2055e18e"},{"image":"santarem","capture":"santarem_2022-09-24T13-52-04Z-l1a.nc","crop_sha256":"22c5977df1287ea6f8085d8466ed656028f1b8c4fc23b1a591637ab4897e777d","payload_sha256":"eed1d21f1047ead63dca38955ef9c4d52c4f61c2df405c2fa240e662c6d1cdc4","local_sum":"NARROW_NEIGHBOR_ORIENTED","prediction_mode":"REDUCED","encoder":"HYBRID","a":0,"omega":19,"nu_min":-1,"nu_max":3,"phi":3,"psi":0,"p":3,"t_inc":2048,"rows":956,"columns":684,"bands":120,"crop_row":0,"crop_col":0,"sample_count":78468480,"header_bits":176,"payload_bits":570473671,"padding_bits":9,"stream_bits":570473856,"external_info_bits":2040,"max_error":0,"rmse_dn":0.0,"source_study":"omega_fig417_256","source_job":"610fb85b52a3a1d3"},{"image":"santarem","capture":"santarem_2022-09-24T13-52-04Z-l1a.nc","crop_sha256":"22c5977df1287ea6f8085d8466ed656028f1b8c4fc23b1a591637ab4897e777d","payload_sha256":"b19b803cd859fb46c0d4727038ddfb5866bf592c9ab497496b570621ffc6f334","local_sum":"NARROW_NEIGHBOR_ORIENTED","prediction_mode":"REDUCED","encoder":"HYBRID","a":0,"omega":12,"nu_min":-1,"nu_max":3,"phi":3,"psi":0,"p":3,"t_inc":2048,"rows":956,"columns":684,"bands":120,"crop_row":0,"crop_col":0,"sample_count":78468480,"header_bits":176,"payload_bits":570549744,"padding_bits":32,"stream_bits":570549952,"external_info_bits":2040,"max_error":0,"rmse_dn":0.0,"source_study":"omega_fig417_256","source_job":"b296a4d872a46456"},{"image":"santarem","capture":"santarem_2022-09-24T13-52-04Z-l1a.nc","crop_sha256":"22c5977df1287ea6f8085d8466ed656028f1b8c4fc23b1a591637ab4897e777d","payload_sha256":"61bca894e1edda42e35e91006cda8d3cabff4a4d3b77f1c9fe8265425e3d71a2","local_sum":"NARROW_NEIGHBOR_ORIENTED","prediction_mode":"REDUCED","encoder":"HYBRID","a":0,"omega":14,"nu_min":-1,"nu_max":3,"phi":3,"psi":0,"p":3,"t_inc":2048,"rows":956,"columns":684,"bands":120,"crop_row":0,"crop_col":0,"sample_count":78468480,"header_bits":176,"payload_bits":570471444,"padding_bits":60,"stream_bits":570471680,"external_info_bits":2040,"max_error":0,"rmse_dn":0.0,"source_study":"omega_fig417_256","source_job":"4b00aa5848d7135c"},{"image":"doha","capture":"doha_2022-09-12T06-38-31Z-l1a.nc","crop_sha256":"7f19a39fb549e5fe9af3bf21d581cf9e9e942949bdf2c104684eaa4ff3a64c8a","payload_sha256":"ba9741e64b8bdbd8447053cdbe7eefd80ea6f2eb08e991dc5fb8679cb58bb772","local_sum":"NARROW_NEIGHBOR_ORIENTED","prediction_mode":"REDUCED","encoder":"HYBRID","a":0,"omega":19,"nu_min":-1,"nu_max":3,"phi":3,"psi":0,"p":3,"t_inc":2048,"rows":956,"columns":684,"bands":120,"crop_row":0,"crop_col":0,"sample_count":78468480,"header_bits":176,"payload_bits":592922516,"padding_bits":60,"stream_bits":592922752,"external_info_bits":2040,"max_error":0,"rmse_dn":0.0,"source_study":"omega_fig417_256","source_job":"93c2fa5b781b1b21"},{"image":"doha","capture":"doha_2022-09-12T06-38-31Z-l1a.nc","crop_sha256":"7f19a39fb549e5fe9af3bf21d581cf9e9e942949bdf2c104684eaa4ff3a64c8a","payload_sha256":"d15ebf768c9ec51271722755d2111cec7cbea5772dd6cb4b29657997b08df157","local_sum":"NARROW_NEIGHBOR_ORIENTED","prediction_mode":"REDUCED","encoder":"HYBRID","a":0,"omega":12,"nu_min":-1,"nu_max":3,"phi":3,"psi":0,"p":3,"t_inc":2048,"rows":956,"columns":684,"bands":120,"crop_row":0,"crop_col":0,"sample_count":78468480,"header_bits":176,"payload_bits":593146872,"padding_bits":24,"stream_bits":593147072,"external_info_bits":2040,"max_error":0,"rmse_dn":0.0,"source_study":"omega_fig417_256","source_job":"1f4e853833364ab7"},{"image":"doha","capture":"doha_2022-09-12T06-38-31Z-l1a.nc","crop_sha256":"7f19a39fb549e5fe9af3bf21d581cf9e9e942949bdf2c104684eaa4ff3a64c8a","payload_sha256":"25a8c4dc7ea734877bbc1051ac92291ca7c8eba52a3ecf3197e3ce583645243b","local_sum":"NARROW_NEIGHBOR_ORIENTED","prediction_mode":"REDUCED","encoder":"HYBRID","a":0,"omega":14,"nu_min":-1,"nu_max":3,"phi":3,"psi":0,"p":3,"t_inc":2048,"rows":956,"columns":684,"bands":120,"crop_row":0,"crop_col":0,"sample_count":78468480,"header_bits":176,"payload_bits":592924512,"padding_bits":48,"stream_bits":592924736,"external_info_bits":2040,"max_error":0,"rmse_dn":0.0,"source_study":"omega_fig417_256","source_job":"ffbbc0d902c9d430"},{"image":"finnmark","capture":"finnmark_2022-07-06T19-40-03Z-l1a.nc","crop_sha256":"dbc7667a9fabe362e97f77a64af28bb73f75c11ac7b77ade4f2adf7ef00188d3","payload_sha256":"250dc211f59c0ecdd3ef833130df427f73eca23be5df3030e791908e1f47c6a8","local_sum":"NARROW_NEIGHBOR_ORIENTED","prediction_mode":"REDUCED","encoder":"HYBRID","a":0,"omega":19,"nu_min":-1,"nu_max":3,"phi":3,"psi":0,"p":3,"t_inc":2048,"rows":956,"columns":684,"bands":120,"crop_row":0,"crop_col":0,"sample_count":78468480,"header_bits":176,"payload_bits":546887300,"padding_bits":12,"stream_bits":546887488,"external_info_bits":2040,"max_error":0,"rmse_dn":0.0,"source_study":"omega_fig417_256","source_job":"86b721aad87f670e"},{"image":"finnmark","capture":"finnmark_2022-07-06T19-40-03Z-l1a.nc","crop_sha256":"dbc7667a9fabe362e97f77a64af28bb73f75c11ac7b77ade4f2adf7ef00188d3","payload_sha256":"91a57cf7c61c23fb696e0301ab2e87e7a56d54cb1850caa77b8513f559e075a7","local_sum":"NARROW_NEIGHBOR_ORIENTED","prediction_mode":"REDUCED","encoder":"HYBRID","a":0,"omega":12,"nu_min":-1,"nu_max":3,"phi":3,"psi":0,"p":3,"t_inc":2048,"rows":956,"columns":684,"bands":120,"crop_row":0,"crop_col":0,"sample_count":78468480,"header_bits":176,"payload_bits":547203170,"padding_bits":46,"stream_bits":547203392,"external_info_bits":2040,"max_error":0,"rmse_dn":0.0,"source_study":"omega_fig417_256","source_job":"21856c9b5ff30a21"},{"image":"finnmark","capture":"finnmark_2022-07-06T19-40-03Z-l1a.nc","crop_sha256":"dbc7667a9fabe362e97f77a64af28bb73f75c11ac7b77ade4f2adf7ef00188d3","payload_sha256":"034fddddbda046e131842f6f7a1cc0d7d0782bfdad026747cfb43526b39b2f03","local_sum":"NARROW_NEIGHBOR_ORIENTED","prediction_mode":"REDUCED","encoder":"HYBRID","a":0,"omega":14,"nu_min":-1,"nu_max":3,"phi":3,"psi":0,"p":3,"t_inc":2048,"rows":956,"columns":684,"bands":120,"crop_row":0,"crop_col":0,"sample_count":78468480,"header_bits":176,"payload_bits":546883053,"padding_bits":35,"stream_bits":546883264,"external_info_bits":2040,"max_error":0,"rmse_dn":0.0,"source_study":"omega_fig417_256","source_job":"ae01a215d388fb70"}]}
''',
}

if __name__=="__main__":
    _run()
