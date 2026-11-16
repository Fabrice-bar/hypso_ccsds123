"""Process scheduling for the HYPSO sweep tools. No codec math lives here."""
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
