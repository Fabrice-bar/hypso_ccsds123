#!/usr/bin/env python3
"""One complete codec configuration per process; use fig413_sweep.py."""
from __future__ import annotations
import csv
import hashlib
import json
import math
import os
from pathlib import Path
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
import _noise_study as ns

GIB=1024**3
ALIASES={"lines":"y","line":"y","frames":"y","frame":"y","rows":"y","y":"y",
         "samples":"x","sample":"x","columns":"x","cols":"x","x":"x",
         "bands":"z","band":"z","wavelength":"z","wavelengths":"z","z":"z"}
EXTRA_FIELDS=["job_id","size","crop_row","crop_col","crop_sha256","log2_t_inc",
              "spatial_samples","data_load_seconds","model_seconds","codec_and_checks_seconds",
              "wall_seconds","peak_rss_bytes","peak_rss_gib","peak_rss_method",
              "sampled_peak_rss_bytes","workers_requested","peak_concurrent_workers",
              "native_threads","estimated_ram_gib","schedule_verified","first_update_at_vmax",
              "adaptation_verification_seconds","worker_pid","predictor_processes_per_config"]
FIELDS=core.FIELDS+EXTRA_FIELDS+ns.FIELDS


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

def table(path,fields,rows):
    tmp=path.with_suffix(path.suffix+".tmp")
    with tmp.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    os.replace(tmp,path)

def execution_budget(args):
    return min(args.total_memory_gb,sw._available_memory())

def allowed_workers(requested):
    limit=requested
    try:limit=min(limit,len(os.sched_getaffinity(0)))
    except AttributeError:limit=min(limit,os.cpu_count() or 1)
    allocation=os.environ.get("SLURM_CPUS_PER_TASK","")
    if allocation.isdecimal():limit=min(limit,max(1,int(allocation)))
    return max(1,limit)


def worker(spec_path):
    """Load, predict, encode and check one configuration in this process."""
    spec=json.loads(Path(spec_path).read_text(encoding="utf-8"))
    args=SimpleNamespace(**spec["args"]);job=spec["job"];source=spec["input"]
    signal.signal(signal.SIGINT,signal.SIG_IGN)
    if not (1<=args.a<=15 and 0<=args.psi<2**args.theta and 0<=args.phi<2**args.theta
            and -6<=args.nu_min<=args.nu_max<=9):
        raise ValueError("Illegal error limit, representative value or adaptation bounds.")
    model=sw.load_model(Path(spec["repo"]),"vmax")
    if args.low_memory:
        from hybrid_no_traces import validate_reference
        validate_reference()
    path=Path(source["file"]);stat=path.stat()
    if (stat.st_size,stat.st_mtime_ns)!=(source["file_bytes"],source["file_mtime_ns"]):
        raise ValueError("Input changed after planning; restart with a consistent source.")
    args.full_image=False;args.crop=(job["rows"],job["columns"])
    args.row,args.col=job["crop_row"],job["crop_col"]
    args.axis_order=source["axis_order"];args.t_inc=job["t_inc"]
    args.phi_values=[args.phi]  # Keep a_values as the full grid for common header widths.
    start=time.perf_counter();cube,info=core.read_cube(path,args)
    load_seconds=time.perf_counter()-start
    if list(cube.shape)!=[job["rows"],job["columns"],job["bands"]]:raise ValueError("Input dimensions changed.")
    cube,noise_info=ns.add_noise(cube,job["sigma_dn"],job["noise_seed"])
    h=ns.make_header(cube,args,model[0])
    args.preserve_representative_header=True
    segments=schedule_segments(h.y_size,h.x_size,args.nu_min,args.t_inc,args.nu_max)
    verification_seconds=0.0

    class ObservedPredictor(model[2].Predictor):
        def run_predictor(self):
            nonlocal verification_seconds
            super().run_predictor()  # Original sequential sample/band order.
            begin=time.perf_counter()
            for seg in segments:
                j0=seg["spatial_index_start"]-1;j1=seg["spatial_index_end_inclusive"]
                if not np.all(self.weight_update_scaling_exponent[j0:j1]==seg["rho"]):
                    raise RuntimeError("Applied adaptation schedule differs from the reported schedule.")
            verification_seconds=time.perf_counter()-begin

    observed=(model[0],model[1],SimpleNamespace(Predictor=ObservedPredictor),model[3])
    begin=time.perf_counter()
    row=core.encode_one(cube,h,args,args.a,args.nu_max,args.phi,observed)
    total_codec=time.perf_counter()-begin
    row["predictor_seconds"]=max(0.0,row["predictor_seconds"]-verification_seconds)
    peak,method=os_peak_rss()
    cap=[s for s in segments if s["nu"]==args.nu_max]
    row.update(image=job["image"],capture=path.name,job_id=job["job_id"],size=job["size"],
               crop_row=job["crop_row"],crop_col=job["crop_col"],crop_sha256=info["crop_sha256"],
               spatial_samples=h.y_size*h.x_size,log2_t_inc=args.t_inc.bit_length()-1,
               data_load_seconds=load_seconds,model_seconds=row["predictor_seconds"]+row["encoder_seconds"],
               codec_and_checks_seconds=total_codec,peak_rss_bytes=peak,peak_rss_gib=peak/GIB,
               peak_rss_method=method+"; whole configuration process",native_threads=1,
               estimated_ram_gib=job["estimated_ram_gib"],schedule_verified=1,
               first_update_at_vmax=cap[0]["spatial_index_start"] if cap else "",
               adaptation_verification_seconds=verification_seconds,worker_pid=os.getpid(),
               predictor_processes_per_config=1)
    row.update({k:job[k] for k in ("noise_level","sigma_dn","sigma_over_a","noise_rep","noise_seed")})
    row.update(theta=args.theta,phi_normalized=args.phi/2**args.theta,
               psi_normalized=args.psi/2**args.theta,first_band=args.first_band,**noise_info)
    core.write_json(Path(spec["worker_result"]),{"row":row,"crop":info,"progression":segments,
                                                "header":core.header_snapshot(h)})


def regenerate(out,metadata):
    records=[]
    for job in metadata["study"]["jobs"]:
        path=out/"jobs"/(job["job_id"]+".json")
        if not path.exists():continue
        record=json.loads(path.read_text(encoding="utf-8"))
        if record.get("study_id")!=metadata["study_id"]:raise ValueError("Saved result identity differs.")
        row=record["row"]
        for key in ("job_id","image","t_inc","phi","psi","a","nu_min","nu_max","rows","columns","bands",
                    "noise_level","sigma_dn","noise_rep","noise_seed"):
            if row[key]!=job[key]:raise ValueError(f"Saved result has a different {key}.")
        records.append(record)
    write_tables(out,records)
    return records


def write_tables(out,records):
    table(out/"results.csv",FIELDS,[r["row"] for r in records])
    progression=[]
    params=("job_id","image","size","t_inc","nu_min","nu_max","phi","psi","a","noise_level","sigma_dn","noise_rep")
    for record in records:
        for seg in record["progression"]:
            progression.append({k:record["row"][k] for k in params}|seg)
    table(out/"adaptation_progression.csv",list(params)+["spatial_index_start","spatial_index_end_inclusive",
          "updates_per_band","nu","rho"],progression)


def run_jobs(args,metadata,repo,out,plot_results):
    jobs=metadata["study"]["jobs"];inputs=metadata["study"]["inputs"]
    records=regenerate(out,metadata);done={r["row"]["job_id"] for r in records}
    by_id={r["row"]["job_id"]:r for r in records}
    pending=[j for j in jobs if j["job_id"] not in done]
    skipped=[];failed=[];active={};aggregate_peak=0;peak_active=0
    run_start=time.perf_counter()
    diag=["job_id","image","size","t_inc","nu_min","nu_max","phi","psi","a","sigma_dn","noise_rep","reason"]
    limit=allowed_workers(args.workers)
    print(f"Completed {len(done)}/{len(jobs)}. One COMPLETE configuration per worker; "
          f"requested {args.workers}, CPU allowance {limit}. RAM can lower concurrency.",flush=True)
    base_args={k:getattr(args,k) for k in ("p","nu_min","a_values","local_sum","prediction_mode",
                                         "low_memory","max_memory_gb","variable","theta","first_band")}
    try:
        while pending or active:
            while pending and len(active)<limit:
                job=pending[0];need=job["estimated_ram_gib"]
                reserved=sum(item["job"]["estimated_ram_gib"] for item in active.values())
                remaining_reserve=sum(max(0,item["job"]["estimated_ram_gib"]-item["last_rss"]/GIB)
                                      for item in active.values())
                enough=(need+.35<=execution_budget(args)-remaining_reserve and
                        need+reserved+.35<=args.total_memory_gb)
                reason=job["dimension_issue"]
                if not reason and need>args.max_memory_gb:
                    reason=f"Estimated {need:.2f} GiB exceeds per-job ceiling {args.max_memory_gb:.2f}."
                if not reason and not enough:
                    if active:break  # Wait for memory to be released, do not drop the job.
                    reason=f"Estimated {need:.2f} GiB + coordinator allowance exceeds available/budget {execution_budget(args):.2f} GiB."
                pending.pop(0)
                if reason:
                    skipped.append({k:job[k] for k in diag[:-1]}|{"reason":reason})
                    table(out/"skipped.csv",diag,skipped)
                    print(f"SKIP {job['image']} phi={job['phi']} psi={job['psi']} A={job['a']} sigma={job['sigma_dn']:g} rep={job['noise_rep']} t_inc={job['t_inc']}: {reason}",flush=True)
                    continue
                stem=out/"jobs"/job["job_id"]
                result=stem.with_suffix(".worker.json");result.unlink(missing_ok=True)
                spec={"args":base_args|{k:job[k] for k in ("nu_max","phi","psi","a")},"job":job,
                      "input":inputs[job["image"]],"repo":str(repo),"worker_result":str(result)}
                spec_path=stem.with_suffix(".spec.json");log=None
                try:
                    core.write_json(spec_path,spec)
                    log=stem.with_suffix(".log").open("w",encoding="utf-8")
                    env=dict(os.environ)
                    for name in sw._NATIVE_THREADS:env[name]="1"
                    begin=time.perf_counter()
                    proc=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),"--_worker",str(spec_path)],
                                          stdout=log,stderr=subprocess.STDOUT,env=env,cwd=repo,
                                          start_new_session=os.name=="posix")
                except BaseException:
                    if log is not None:log.close()
                    raise
                active[job["job_id"]]={"job":job,"proc":proc,"ps":psutil.Process(proc.pid),"log":log,
                                      "start":begin,"peak":0,"last_rss":0,"peak_concurrent":1,"result":result}
                peak_active=max(peak_active,len(active))
                for item in active.values():item["peak_concurrent"]=max(item["peak_concurrent"],len(active))
                print(f"START {job['image']} {job['size']}, phi={job['phi']}, psi={job['psi']}, nu_max={job['nu_max']}, "
                      f"A={job['a']}, sigma={job['sigma_dn']:g}, rep={job['noise_rep']}, t_inc={job['t_inc']}; PID={proc.pid}; active={len(active)}/{limit}",flush=True)
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
                    reason=f"Process exited with code {code}; inspect jobs/{key}.log."
                    failed.append({k:job[k] for k in diag[:-1]}|{"reason":reason})
                    table(out/"failures.csv",diag,failed)
                    print(f"FAILED {job['image']} phi={job['phi']} psi={job['psi']} A={job['a']} sigma={job['sigma_dn']:g} rep={job['noise_rep']} t_inc={job['t_inc']}: {reason}",flush=True)
                    continue
                record=json.loads(item["result"].read_text(encoding="utf-8"));row=record["row"]
                if row["job_id"]!=key:raise ValueError("Worker returned the wrong job.")
                if not row["peak_rss_method"].startswith("OS"):
                    row["peak_rss_bytes"]=max(row["peak_rss_bytes"],item["peak"])
                    row["peak_rss_gib"]=row["peak_rss_bytes"]/GIB
                row.update(wall_seconds=elapsed,sampled_peak_rss_bytes=item["peak"],
                           workers_requested=args.workers,peak_concurrent_workers=item["peak_concurrent"])
                record["study_id"]=metadata["study_id"]
                core.write_json(out/"jobs"/(key+".json"),record)
                item["result"].unlink(missing_ok=True)
                done.add(key);by_id[key]=record
                records=[by_id[j["job_id"]] for j in jobs if j["job_id"] in by_id]
                write_tables(out,records)
                print(f"[{len(done)}/{len(jobs)}] {job['image']} phi={job['phi']} psi={job['psi']} A={job['a']} sigma={job['sigma_dn']:g} rep={job['noise_rep']} "
                      f"t_inc={job['t_inc']} (2^{row['log2_t_inc']}), nu_max={job['nu_max']}: "
                      f"{row['bits_per_sample']:.6f} bits/sample; max error={row['max_error']}; "
                      f"nu_max reached={bool(row['nu_max_reached'])}",flush=True)
            aggregate_peak=max(aggregate_peak,rss_sum)
            if active:time.sleep(args.memory_poll_ms/1000)
    finally:
        # No nested pools: every child is one complete sequential codec job.
        for item in active.values():
            if item["proc"].poll() is None:item["proc"].terminate()
        for item in active.values():
            try:item["proc"].wait(timeout=5)
            except subprocess.TimeoutExpired:item["proc"].kill();item["proc"].wait()
            item["log"].close()
        records=regenerate(out,metadata)
        table(out/"skipped.csv",diag,skipped);table(out/"failures.csv",diag,failed)
        core.write_json(out/"last_execution.json",{"completed_total":len(records),"requested_total":len(jobs),
                        "skipped_this_invocation":len(skipped),"failed_this_invocation":len(failed),
                        "wall_seconds":time.perf_counter()-run_start,
                        "sampled_aggregate_peak_rss_gib":aggregate_peak/GIB,
                        "memory_note":"Sampled coordinator + configuration-process RSS; shared libraries can be counted more than once.",
                        "workers_requested":args.workers,"allowed_workers":limit,
                        "peak_concurrent_configurations":peak_active,"predictor_processes_per_config":1,
                        "native_threads_per_process":1,"memory_poll_ms":args.memory_poll_ms})
        plot_results(records,metadata,out)
    print(f"Complete {len(records)}/{len(jobs)}; skipped {len(skipped)}; failed {len(failed)}. Results: {out}")
    return 1 if failed or len(records)!=len(jobs) else 0


if __name__=="__main__":
    if len(sys.argv)==3 and sys.argv[1]=="--_worker":
        try:worker(sys.argv[2])
        except BaseException:traceback.print_exc();raise SystemExit(1)
    else:
        raise SystemExit("Run fig413_sweep.py, not this internal worker module.")
