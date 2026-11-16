#!/usr/bin/env python3
"""Execution engine for the weight-resolution study. Use omega_sweep.py.

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

VERSION = "1.0.0-omega-config-workers"
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
FIELDS = core.FIELDS + EXTRA_FIELDS




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
    h.check_legal_config()
    segments=schedule_segments(h.y_size,h.x_size,args.nu_min,args.t_inc,args.nu_max,
                               D=h.get_dynamic_range_bits(),Omega=args.omega)
    verification_seconds=0.0
    statistics={"predictor_backend":"original-serial", "predictor_pool_size":1,
                "predictor_processes_used":1, "predictor_worker_pids":[os.getpid()],
                "predictor_peak_tasks_in_flight":1, "predictor_tasks":1,
                "rows_per_task":0, "shared_workspace_bytes":0}
    # No nested process pool: this child owns prediction AND entropy encoding.
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
    diagnostic_fields=["job_id","image","size","omega","t_inc","nu_min","phi","a","repeat","reason"]
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
                    print(f"SKIP {job['image']} {job['size']} Omega={job['omega']}, nu_min={job['nu_min']}, phi={job['phi']}, A={job['a']}, t_inc={job['t_inc']}: {reason}",flush=True)
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
                      f"({'full' if job['is_full_image'] else 'crop'}), Omega={job['omega']}, nu_min={job['nu_min']}, phi={job['phi']}, A={job['a']}, t_inc={job['t_inc']}, repeat={job['repeat']}, "
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
                    print(f"FAILED {job['image']} {job['size']} Omega={job['omega']}, nu_min={job['nu_min']}, phi={job['phi']}, A={job['a']}, t_inc={job['t_inc']}: {reason}",flush=True)
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
                print(f"[{len(done)}/{len(jobs)}] {job['image']} {job['size']} Omega={job['omega']}, nu_min={job['nu_min']}, phi={job['phi']}, A={job['a']}, t_inc={job['t_inc']}: "
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
        raise SystemExit("Run omega_sweep.py, not this internal worker module.")
