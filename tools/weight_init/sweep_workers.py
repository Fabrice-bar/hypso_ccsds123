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
