#!/usr/bin/env python3
r"""HYPSO sensitivity to spatial image size, with genuine full-image runs.

Defaults: five HYPSO scenes; nested centre crops 128,256,512 and full image;
P=3, reduced, narrow-neighbour, nu_min=-6, nu_max=4, phi=0, A=0, hybrid;
D=16, Omega=19, R=64, Theta=3. All native bands and unscaled products/dn.
Choose --t-inc 128 or --t-inc-values 128 512 2048. One fresh process per run.
Low-memory hybrid storage is ON by default; --standard-memory disables it.

Run using your Python interpreter, from the verification repository root:
  python tools/spatial_size/spatial_size_sweep.py --input-dir data --t-inc 128 --dry-run
  python tools/spatial_size/spatial_size_sweep.py --input-dir data --t-inc 128
  python tools/spatial_size/spatial_size_sweep.py --input-dir data --images lisbon --sizes full --t-inc 2048
  python tools/spatial_size/spatial_size_sweep.py --input-dir data --t-inc-values 128 512 2048

Use --files PATH.nc for explicit inputs instead of the five-scene list.
--sizes full encodes every spatial sample as one image; it does not tile.
Oversized crops and jobs that do not fit the estimated RAM budget are recorded
as skipped. They are reconsidered on resume. Nothing is silently downsized.
Repeat the identical scientific command to resume, or use --plot-only --output DIR.
Changing worker count or RAM limits is allowed; changing the grid needs a new DIR.

Measured outputs: bits/sample, predictor/encoder time, total job wall time,
per-process OS peak resident RAM, plus sampled parent/worker RAM diagnostics.
The adaptation plot is the actual verified update-exponent schedule, NOT a
measurement of weight convergence. Different crops also contain different data.
Correctness checks use encoder-side reconstruction, not an independent decoder.
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
from contextlib import redirect_stdout, redirect_stderr
from types import SimpleNamespace

import numpy as np
import psutil
import _damping_core as core
import sweep_workers as sw

VERSION = "1.0.0-spatial-size"
GIB = 1024**3
ALIASES = {"lines":"y", "line":"y", "frames":"y", "frame":"y", "rows":"y", "y":"y",
           "samples":"x", "sample":"x", "columns":"x", "cols":"x", "x":"x",
           "bands":"z", "band":"z", "wavelength":"z", "wavelengths":"z", "z":"z"}
EXTRA_FIELDS = ["job_id", "size", "repeat", "is_full_image", "spatial_samples", "crop_row", "crop_col",
                "crop_sha256", "data_load_seconds", "model_seconds", "codec_and_checks_seconds", "wall_seconds",
                "peak_rss_bytes", "peak_rss_gib", "peak_rss_method", "sampled_peak_rss_bytes",
                "workers_requested", "peak_concurrent_workers", "native_threads", "estimated_ram_gib",
                "schedule_verified", "first_update_at_vmax", "adaptation_verification_seconds"]
FIELDS = core.FIELDS + EXTRA_FIELDS


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", type=Path, default=Path("data"))
    select=p.add_mutually_exclusive_group()
    select.add_argument("--images", nargs="+", choices=list(core.SCENES))
    select.add_argument("--files", type=Path, nargs="+", help="Explicit NetCDF files; bypass scene filenames.")
    p.add_argument("--sizes", nargs="+", default=["128","256","512","full"], help="Square side lengths and/or 'full'.")
    timing=p.add_mutually_exclusive_group()
    timing.add_argument("--t-inc", type=int, help="One adaptation interval, e.g. 128.")
    timing.add_argument("--t-inc-values", type=int, nargs="+", help="Several intervals, e.g. 128 512 2048.")
    p.add_argument("--nu-min", type=int, default=-6)
    p.add_argument("--nu-max", type=int, default=4)
    p.add_argument("--phi", type=int, default=0, help="Fixed damping; default zero based on your recent shortlist.")
    p.add_argument("--a", type=int, default=0, help="Absolute error limit; default lossless.")
    p.add_argument("--p", type=int, default=3)
    p.add_argument("--psi", type=int, default=7, help="Offset only for A>0; zero for lossless.")
    p.add_argument("--local-sum", choices=list(core.LOCAL_SUM_OPTIONS), default="narrow-neighbor")
    p.add_argument("--prediction-mode", choices=("reduced","full"), default="reduced")
    p.add_argument("--variable", default="products/dn")
    p.add_argument("--axis-order", choices=("yxz","yzx","xyz","xzy","zxy","zyx"))
    p.add_argument("--anchor", choices=("center","top-left"), default="center")
    storage=p.add_mutually_exclusive_group()
    storage.add_argument("--low-memory", dest="low_memory", action="store_true", help="Default: validated hybrid storage without text traces.")
    storage.add_argument("--standard-memory", dest="low_memory", action="store_false", help="Original diagnostic arrays (much larger RAM use).")
    p.set_defaults(low_memory=True)
    p.add_argument("--workers", type=int, default=1, help="Fresh concurrent processes; default 1 for timing/memory comparisons.")
    p.add_argument("--repeats", type=int, default=1, help="Independent runs for timing variability; never treated as new image samples.")
    p.add_argument("--max-memory-gb", type=float, default=90, help="Estimated per-job RAM ceiling, GiB.")
    p.add_argument("--total-memory-gb", type=float, default=90, help="Estimated aggregate RAM ceiling, GiB (not an OS limit).")
    p.add_argument("--memory-poll-ms", type=int, default=50, help="Fallback RSS sampling interval, 10..1000 ms.")
    p.add_argument("--repo", type=Path)
    p.add_argument("--output", type=Path, default=Path("results/spatial_size"))
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--plot-only", action="store_true")
    a=p.parse_args()
    a.t_inc_values=sorted(set(a.t_inc_values if a.t_inc_values is not None else [a.t_inc if a.t_inc is not None else 128]))
    if any(t <16 or t>2048 or t&(t-1) for t in a.t_inc_values):p.error("t_inc must be a power of two from 16 to 2048.")
    if not -6<=a.nu_min<=a.nu_max<=9:p.error("Require -6 <= nu_min <= nu_max <= 9.")
    if not 0<=a.p<=15 or not 0<=a.phi<=7 or not 0<=a.psi<=7 or not 0<=a.a<=32767:p.error("Require P=0..15, phi/psi=0..7, A=0..32767.")
    if a.workers<1 or a.repeats<1:p.error("Workers and repeats must be positive.")
    if not 10<=a.memory_poll_ms<=1000:p.error("--memory-poll-ms must be 10..1000.")
    if any(not math.isfinite(v) or v<=0 for v in (a.max_memory_gb,a.total_memory_gb)):p.error("Memory budgets must be positive and finite.")
    converted=[]
    for s in a.sizes:
        if s.lower()=="full":converted.append("full")
        elif s.isdecimal() and 2<=int(s)<=65535:converted.append(str(int(s)))
        else:p.error("--sizes accepts integer sides 2..65535 and 'full'.")
    a.sizes=sorted(set(converted),key=lambda s:math.inf if s=="full" else int(s))
    a.images=list(dict.fromkeys(a.images or list(core.SCENES)))
    return a


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


def make_plan(args,inputs):
    jobs=[]
    for size in args.sizes:
        for scene,info in inputs.items():
            sy,sx,z=info["source_shape_yxz"]
            y,x=(sy,sx) if size=="full" else (int(size),int(size))
            row,col=(0,0) if size=="full" or args.anchor=="top-left" else ((sy-y)//2,(sx-x)//2)
            issue="" if 2<=y<=sy and 2<=x<=sx else f"Requested {y}x{x} does not fit source {sy}x{sx}."
            for tinc in args.t_inc_values:
                for repeat in range(1,args.repeats+1):
                    identity=[scene,size,tinc,repeat]
                    job_id=hashlib.sha256(json.dumps(identity).encode()).hexdigest()[:16]
                    jobs.append({"job_id":job_id,"image":scene,"size":size,"t_inc":tinc,"repeat":repeat,
                                 "rows":y,"columns":x,"bands":z,"crop_row":row,"crop_col":col,
                                 "is_full_image":int(size=="full"),"estimated_ram_gib":estimate_gib((y,x,z),args.p,args.low_memory),
                                 "dimension_issue":issue})
    return jobs


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
    args.a_values=[args.a];args.phi_values=[args.phi]
    start=time.perf_counter()
    cube,info=core.read_cube(path,args)
    load_seconds=time.perf_counter()-start
    if list(cube.shape)!=[job["rows"],job["columns"],job["bands"]]:raise ValueError("Input dimensions changed.")
    h=core.make_header(cube,args,args.a,args.nu_max,model[0])
    segments=schedule_segments(h.y_size,h.x_size,args.nu_min,args.t_inc,args.nu_max)
    verification_seconds=0.0
    class ObservedPredictor(model[2].Predictor):
        def run_predictor(self):
            nonlocal verification_seconds
            super().run_predictor()
            begin=time.perf_counter()
            for seg in segments:
                # Applied update at t uses exponent stored at t-1.
                j0=seg["spatial_index_start"]-1;j1=seg["spatial_index_end_inclusive"]
                if not np.all(self.weight_update_scaling_exponent[j0:j1]==seg["rho"]):
                    raise RuntimeError("Actual model adaptation schedule differs from the reported progression.")
            verification_seconds=time.perf_counter()-begin
    observed=(model[0],model[1],SimpleNamespace(Predictor=ObservedPredictor),model[3])
    begin=time.perf_counter()
    row=core.encode_one(cube,h,args,args.a,args.nu_max,args.phi,observed)
    total_codec=time.perf_counter()-begin
    # Remove the observer scan from predictor time; total wall time retains all work.
    row["predictor_seconds"]=max(0.0,row["predictor_seconds"]-verification_seconds)
    peak,method=os_peak_rss()
    cap=[s for s in segments if s["nu"]==args.nu_max]
    row.update(image=job["image"],capture=path.name,job_id=job["job_id"],size=job["size"],repeat=job["repeat"],
               is_full_image=job["is_full_image"],spatial_samples=h.y_size*h.x_size,
               crop_row=job["crop_row"],crop_col=job["crop_col"],crop_sha256=info["crop_sha256"],
               data_load_seconds=load_seconds,model_seconds=row["predictor_seconds"]+row["encoder_seconds"],
               codec_and_checks_seconds=total_codec,peak_rss_bytes=peak,peak_rss_gib=peak/GIB,
               peak_rss_method=method,native_threads=1,estimated_ram_gib=job["estimated_ram_gib"],
               schedule_verified=1,first_update_at_vmax=cap[0]["spatial_index_start"] if cap else "",
               adaptation_verification_seconds=verification_seconds)
    core.write_json(Path(spec["worker_result"]),{"row":row,"crop":info,"progression":segments,"header":core.header_snapshot(h)})


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
            progression.append({k:row[k] for k in ("job_id","image","size","t_inc","repeat")} | seg)
    table(out/"adaptation_progression.csv",["job_id","image","size","t_inc","repeat","spatial_index_start",
          "spatial_index_end_inclusive","updates_per_band","nu","rho"],progression)
    return records


def plot_results(records,metadata,out):
    if not records:return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    st=metadata["study"];rows=[r["row"] for r in records]
    colors=["#0072B2","#D55E00","#009E73","#CC79A7","#E69F00","#5B4BCE"]
    tvalues=st["t_inc_values"]
    metrics=[("bits_per_sample","Compressed bits/sample"),("model_seconds","Predictor + encoder (s)"),
             ("peak_rss_gib","Peak worker resident RAM (GiB)"),("fraction_updates_at_nu_max","Updates at final parameter (%)")]
    title=(f"P={st['profile']['p']} | narrow-neighbor, reduced | "
           f"A={st['profile']['a']}, phi={st['profile']['phi']}, nu_min={st['profile']['nu_min']}, nu_max={st['profile']['nu_max']}")
    # Profile text also supports deliberate CLI overrides.
    title=title.replace("narrow-neighbor, reduced",f"{st['profile']['local_sum']}, {st['profile']['prediction_mode']}")
    for scene,info in st["inputs"].items():
        selected=[r for r in rows if r["image"]==scene]
        if not selected:continue
        groups={}
        for r in selected:groups.setdefault((r["size"],int(r["t_inc"])),[]).append(r)
        sizes=sorted({r["size"] for r in selected},key=lambda s:next(r["spatial_samples"] for r in selected if r["size"]==s))
        ticks={}
        for size in sizes:
            r=next(r for r in selected if r["size"]==size)
            label=f"{r['rows']} x {r['columns']}"+("\nfull" if size=="full" else "")
            ticks[r["spatial_samples"]]=label
        fig,axes=plt.subplots(2,2,figsize=(12,8.4))
        for ax,(metric,label) in zip(axes.flat,metrics):
            for i,tinc in enumerate(tvalues):
                pts=[]
                for size in sizes:
                    g=groups.get((size,tinc),[])
                    if g:
                        values=[float(r[metric])*(100 if metric=="fraction_updates_at_nu_max" else 1) for r in g]
                        pts.append((g[0]["spatial_samples"],float(np.median(values)),min(values),max(values),size))
                if not pts:continue
                xs,ys,lo,hi,_=zip(*pts)
                ax.errorbar(xs,ys,yerr=[np.array(ys)-lo,np.array(hi)-ys],color=colors[i%6],marker="o",
                            capsize=3,linewidth=1.5,label=f"t_inc={tinc}")
                full=[pt for pt in pts if pt[4]=="full"]
                if full:ax.plot([p[0] for p in full],[p[1] for p in full],linestyle="none",marker="*",markersize=12,color=colors[i%6])
            ax.set_xscale("log",base=2);ax.set_xticks(list(ticks),list(ticks.values()),fontsize=9)
            ax.set_xlabel("Spatial dimensions (positioned by pixel count)");ax.set_ylabel(label)
            ax.grid(True,alpha=.25);ax.set_axisbelow(True)
            if metric=="fraction_updates_at_nu_max":ax.set_ylim(-2,102)
        expected=sum(j["image"]==scene and not j["dimension_issue"] for j in st["jobs"])
        fig.suptitle(f"{info['label']} | spatial-size sensitivity | completed {len(selected)}/{expected}\n"+title,fontsize=12)
        handles=[Line2D([],[],color=colors[i%6],marker="o",label=f"t_inc={t}") for i,t in enumerate(tvalues)]
        fig.legend(handles=handles,loc="lower center",bbox_to_anchor=(.5,.055),ncol=min(4,len(handles)),frameon=False)
        fig.text(.5,.012,"Stars: full capture. Medians across repeats; whiskers: min-max. Crop content changes with size.\n"
                 "Peak RAM is per fresh worker, including input/model/checks; resource values are measured, not estimates.",ha="center",fontsize=8)
        fig.tight_layout(rect=(0,.12,1,.92))
        for ext in ("png","pdf"):fig.savefig(out/f"{scene}_size_sensitivity.{ext}",dpi=170,bbox_inches="tight")
        plt.close(fig)
        for tinc in tvalues:
            available=[r for r in records if r["row"]["image"]==scene and int(r["row"]["t_inc"])==tinc]
            if not available:continue
            fig,axes=plt.subplots(1,2,figsize=(12,5.5));handles=[]
            for i,size in enumerate(sizes):
                rec=next((r for r in available if r["row"]["size"]==size),None)
                if rec is None:continue
                rr=rec["row"];seg=rec["progression"];N=rr["spatial_samples"]
                x=[s["spatial_index_start"] for s in seg]+[N-1]
                y=[s["nu"] for s in seg]+[seg[-1]["nu"]]
                label=f"{rr['rows']} x {rr['columns']}"+(" (full)" if size=="full" else "")
                for ax,xx in zip(axes,(x,np.array(x)/(N-1)*100)):
                    ax.step(xx,y,where="post",color=colors[i%6],label=label)
                    ax.plot(xx[-1],y[-1],marker="o",markersize=5,color=colors[i%6],
                            markerfacecolor=colors[i%6] if int(rr["nu_max_reached"]) else "white")
                handles.append(Line2D([],[],color=colors[i%6],label=label))
            axes[0].set_xscale("log");axes[0].set_xlabel("Spatial raster index t per band (log scale)")
            axes[1].set_xlabel("Position through spatial raster (%)");axes[1].set_xlim(0,100)
            for ax in axes:
                ax.set_ylabel(r"Applied adaptation parameter $\nu(t-1)$")
                ax.axhline(st["profile"]["nu_max"],color="#777777",linestyle=":",linewidth=.8)
                ax.grid(True,alpha=.25);ax.set_axisbelow(True)
            fig.suptitle(f"{info['label']} | adaptation progression | t_inc={tinc}\n"+title,fontsize=11)
            fig.legend(handles=handles,loc="lower center",bbox_to_anchor=(.5,.07),ncol=min(4,len(handles)),frameon=False,fontsize=9)
            fig.text(.5,.015,"Verified update schedule, not measured predictor convergence. Hollow endpoint: final parameter not reached.\n"
                     "Updates start at spatial t=2; t=0,1 are initialization. Schedule is per band, not multiplied by band count.",ha="center",fontsize=8)
            fig.tight_layout(rect=(0,.18,1,.90))
            for ext in ("png","pdf"):fig.savefig(out/f"{scene}_adaptation_tinc{tinc}.{ext}",dpi=170,bbox_inches="tight")
            plt.close(fig)


def execution_budget(args):
    return min(args.total_memory_gb,sw._available_memory())


def run_jobs(args,metadata,repo,out):
    jobs=metadata["study"]["jobs"];inputs=metadata["study"]["inputs"]
    records=regenerate(out,metadata);done={r["row"]["job_id"] for r in records}
    pending=[j for j in jobs if j["job_id"] not in done]
    skipped=[];failed=[];active={};aggregate_peak=0;run_start=time.perf_counter()
    diagnostic_fields=["job_id","image","size","t_inc","repeat","reason"]
    limit=args.workers
    try:limit=min(limit,len(os.sched_getaffinity(0)))
    except AttributeError:limit=min(limit,os.cpu_count() or 1)
    allocation=os.environ.get("SLURM_CPUS_PER_TASK","")
    if allocation.isdecimal():limit=min(limit,max(1,int(allocation)))
    print(f"Completed {len(done)}/{len(jobs)}; requested workers {args.workers}, allowed {limit}.",flush=True)
    child_args={k:getattr(args,k) for k in ("p","nu_min","nu_max","phi","psi","a","local_sum","prediction_mode",
                                         "low_memory","max_memory_gb","variable")}
    try:
        while pending or active:
            while pending and len(active)<limit:
                job=pending[0];need=job["estimated_ram_gib"]
                reserved=sum(item["job"]["estimated_ram_gib"] for item in active.values())
                reason=job["dimension_issue"]
                if not reason and need>args.max_memory_gb:reason=f"Estimated {need:.2f} GiB exceeds per-job limit {args.max_memory_gb:.2f}."
                # Active workers already consume some available RAM. Reserve their remaining allowance.
                remaining_reserve=sum(max(0,item["job"]["estimated_ram_gib"]-item["last_rss"]/GIB) for item in active.values())
                enough=need+.35<=execution_budget(args)-remaining_reserve and need+reserved+.35<=args.total_memory_gb
                if not reason and not enough:
                    if active:break
                    reason=f"Estimated {need:.2f} GiB + coordinator allowance exceeds current budget {execution_budget(args):.2f} GiB."
                pending.pop(0)
                if reason:
                    skipped.append({k:job[k] for k in diagnostic_fields[:-1]}|{"reason":reason})
                    table(out/"skipped.csv",diagnostic_fields,skipped)
                    print(f"SKIP {job['image']} {job['size']} t_inc={job['t_inc']}: {reason}",flush=True)
                    continue
                stem=out/"jobs"/job["job_id"]
                result=stem.with_suffix(".worker.json");result.unlink(missing_ok=True)
                spec={"args":child_args,"job":job,"input":inputs[job["image"]],"repo":str(repo),"worker_result":str(result)}
                spec_path=stem.with_suffix(".spec.json");core.write_json(spec_path,spec)
                log=stem.with_suffix(".log").open("w",encoding="utf-8")
                env=dict(os.environ)
                for name in sw._NATIVE_THREADS:env[name]="1"
                begin=time.perf_counter()
                proc=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),"--_worker",str(spec_path)],
                                      stdout=log,stderr=subprocess.STDOUT,env=env,cwd=repo)
                active[job["job_id"]]={"job":job,"proc":proc,"ps":psutil.Process(proc.pid),"log":log,
                                      "start":begin,"peak":0,"last_rss":0,"peak_concurrent":1,"result":result}
                for item in active.values():item["peak_concurrent"]=max(item["peak_concurrent"],len(active))
                print(f"START {job['image']} {job['rows']}x{job['columns']}x{job['bands']} "
                      f"({'full' if job['is_full_image'] else 'crop'}), t_inc={job['t_inc']}, repeat={job['repeat']}, "
                      f"PID={proc.pid}, estimate={need:.2f} GiB",flush=True)
            rss_sum=psutil.Process().memory_info().rss
            for key,item in list(active.items()):
                try:rss=item["ps"].memory_info().rss
                except (psutil.NoSuchProcess,psutil.AccessDenied):rss=0
                item["last_rss"]=rss;item["peak"]=max(item["peak"],rss);rss_sum+=rss
                code=item["proc"].poll()
                if code is None:continue
                elapsed=time.perf_counter()-item["start"];item["log"].close();job=item["job"]
                del active[key]
                if code!=0 or not item["result"].exists():
                    reason=f"Worker exited with code {code}; inspect jobs/{key}.log (possible memory/process failure)."
                    failed.append({k:job[k] for k in diagnostic_fields[:-1]}|{"reason":reason})
                    table(out/"failures.csv",diagnostic_fields,failed)
                    print(f"FAILED {job['image']} {job['size']} t_inc={job['t_inc']}: {reason}",flush=True)
                    continue
                record=json.loads(item["result"].read_text(encoding="utf-8"));row=record["row"]
                if row["job_id"]!=key:raise ValueError("Worker returned the wrong job.")
                if not row["peak_rss_method"].startswith("OS"):
                    row["peak_rss_bytes"]=max(row["peak_rss_bytes"],item["peak"])
                    row["peak_rss_gib"]=row["peak_rss_bytes"]/GIB
                row.update(wall_seconds=elapsed,sampled_peak_rss_bytes=item["peak"],workers_requested=args.workers,
                           peak_concurrent_workers=item["peak_concurrent"])
                record["study_id"]=metadata["study_id"]
                core.write_json(out/"jobs"/(key+".json"),record)
                item["result"].unlink(missing_ok=True)
                done.add(key);records=regenerate(out,metadata)
                print(f"[{len(done)}/{len(jobs)}] {job['image']} {job['size']} t_inc={job['t_inc']}: "
                      f"{row['bits_per_sample']:.6f} bits/sample; model={row['model_seconds']:.2f}s; "
                      f"peak={row['peak_rss_gib']:.3f} GiB; nu_max reached={bool(row['nu_max_reached'])}",flush=True)
            aggregate_peak=max(aggregate_peak,rss_sum)
            if active:time.sleep(args.memory_poll_ms/1000)
    finally:
        for item in active.values():item["proc"].terminate()
        for item in active.values():
            try:item["proc"].wait(timeout=5)
            except subprocess.TimeoutExpired:item["proc"].kill();item["proc"].wait()
            item["log"].close()
        records=regenerate(out,metadata)
        table(out/"skipped.csv",diagnostic_fields,skipped)
        table(out/"failures.csv",diagnostic_fields,failed)
        core.write_json(out/"last_execution.json",{"completed_total":len(records),"requested_total":len(jobs),
                        "skipped_this_invocation":len(skipped),"failed_this_invocation":len(failed),
                        "wall_seconds":time.perf_counter()-run_start,"sampled_aggregate_peak_rss_gib":aggregate_peak/GIB,
                        "memory_note":"Sampled sum of coordinator and active worker RSS; shared pages may be counted more than once.",
                        "workers_requested":args.workers,"memory_poll_ms":args.memory_poll_ms})
        plot_results(records,metadata,out)
    print(f"Complete {len(records)}/{len(jobs)}; skipped {len(skipped)}; failed {len(failed)}. Results: {out}")
    return 1 if failed or not records else 0


def main():
    args=parse_args();out=args.output.resolve()
    if args.plot_only:
        if not out.is_dir():raise ValueError("Output folder does not exist.")
        with sw.output_lock(out):
            metadata=json.loads((out/"metadata.json").read_text(encoding="utf-8"))
            if metadata.get("study",{}).get("tool")!="spatial_size":raise ValueError("Not a spatial-size study.")
            records=regenerate(out,metadata);plot_results(records,metadata,out)
        print(f"Plots: {out}");return 0
    if not __debug__:raise ValueError("Do not run python -O; the model uses assertions.")
    repo=find_repo(args.repo);sw.load_model(repo,"vmax")
    adapter={}
    if args.low_memory:
        from hybrid_no_traces import validate_reference,adapter_sha256
        adapter={"reference_class_sha256":validate_reference(),"adapter_sha256":adapter_sha256()}
    inputs={}
    paths=([(p.stem,p.resolve()) for p in args.files] if args.files else
           [(s,core.find_capture(args.input_dir.resolve(),core.SCENES[s])) for s in args.images])
    for scene,path in paths:
        if scene in inputs:raise ValueError("Input identifiers collide; use distinct filenames.")
        # IDs become output filenames; reject problematic filesystem names.
        if not scene or any(ch in scene for ch in '/\\:*?"<>|'):raise ValueError("Use a filename without special path characters.")
        info=inspect_input(path,args);info["label"]=scene.split('_')[0].title();inputs[scene]=info
        print(f"{info['label']}: source shape {info['source_shape_yxz']}; {path.name}")
    jobs=make_plan(args,inputs)
    own=Path(__file__).resolve().parent
    model_files=list((repo/"ccsds123_i2_hlm").rglob("*.py"))+list((repo/"ccsds123_i2_hlm").rglob("*.txt"))
    profile={k:getattr(args,k) for k in ("p","nu_min","nu_max","phi","psi","a","local_sum","prediction_mode")}
    profile.update(D=16,Omega=19,R=64,Theta=3,encoder="HYBRID",psi_lossless=0)
    study={"tool":"spatial_size","version":VERSION,"profile":profile,"inputs":inputs,"sizes":args.sizes,
           "anchor":args.anchor,"t_inc_values":args.t_inc_values,"repeats":args.repeats,"jobs":jobs,
           "low_memory":args.low_memory,"adapter":adapter,"python":platform.python_version(),"numpy":np.__version__,
           "psutil":psutil.__version__,"platform":platform.platform(),"processor":platform.processor(),
           "model_sha256":core.file_digest(model_files,repo),
           "bundle_sha256":core.file_digest([own/n for n in (Path(__file__).name,"_damping_core.py","sweep_workers.py","hybrid_no_traces.py")],own),
           "metric":"Header + entropy payload/tail + output-word padding per scalar sample; external initialization excluded, inclusive column also saved.",
           "timing":"Fresh process per configuration; one native thread. Predictor time excludes the added schedule-verification scan.",
           "memory":"Per-worker OS peak RSS where supported, including input, imports, codec and correctness checks; sampled fallback is a lower bound.",
           "adaptation":"Actual rho[t-1] verified for every used update, t=2..Nx*Ny-1. This is the programmed schedule, not convergence.",
           "crop_caveat":"Changing size changes image content, boundaries, headers and initial/steady-state proportions. Full images are not tiled."}
    identity=hashlib.sha256(json.dumps(study,sort_keys=True).encode()).hexdigest()
    metadata={"study_id":identity,"study":study,"model_path":str(repo)}
    print(f"{len(jobs)} requested jobs; low-memory={args.low_memory}; current available/budget={execution_budget(args):.2f} GiB.")
    for job in jobs:
        reason=job["dimension_issue"]
        if not reason and job["estimated_ram_gib"]>args.max_memory_gb:reason="exceeds per-job memory ceiling"
        if not reason and job["estimated_ram_gib"]+.35>execution_budget(args):reason="does not fit current available/total memory"
        print(f"  {job['image']}: {job['size']}, t_inc={job['t_inc']}, repeat={job['repeat']}, "
              f"{job['estimated_ram_gib']:.2f} GiB estimate"+(f" — SKIP NOW: {reason}" if reason else ""))
    if args.dry_run:
        print("Dry run: input headers/hashes and plan checked; no image encoding or output writes.");return 0
    if args.workers>1:print("NOTE: parallel runs share CPU/memory bandwidth; use workers=1 for resource comparisons.")
    with sw.output_lock(out):
        meta_path=out/"metadata.json"
        if meta_path.exists():
            old=json.loads(meta_path.read_text(encoding="utf-8"))
            if old.get("study_id")!=identity or old.get("study")!=study:
                raise ValueError("Inputs, scientific settings, grid, storage mode or code changed. Choose a new output folder.")
        elif (out/"results.csv").exists() or (out/"jobs").exists():
            raise ValueError("Results exist without matching metadata. Choose a new output folder.")
        (out/"jobs").mkdir(exist_ok=True)
        core.write_json(meta_path,metadata)
        table(out/"plan.csv",list(jobs[0]),jobs)
        return run_jobs(args,metadata,repo,out)


if __name__=="__main__":
    if len(sys.argv)==3 and sys.argv[1]=="--_worker":
        try:worker(sys.argv[2])
        except BaseException:traceback.print_exc();raise SystemExit(1)
    else:
        try:raise SystemExit(main())
        except KeyboardInterrupt:print("Interrupted; completed configurations are saved. Repeat the same command to resume.");raise SystemExit(130)
        except (ValueError,OSError,ModuleNotFoundError) as exc:print(f"ERROR: {exc}",file=sys.stderr);raise SystemExit(1)
