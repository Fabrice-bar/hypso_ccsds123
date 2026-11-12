"""Bit-exact parallel scheduling of the validated NTNU predictor.

One task computes consecutive complete rows in ONE band. It waits for the
preceding row block in that band and the same block in the preceding band.
These dependencies cover every read by the original predictor methods. No
prediction, learning, quantization or boundary rule is changed, and no state
is reset at task boundaries. The hybrid entropy encoder remains sequential.

Arrays are shared through temporary memory-mapped files, not copied into each
process. Use a local disk with enough free space; the runner cleans up after
each crop. Spawn is used on every OS to exercise Windows-compatible startup.
"""
from __future__ import annotations

import ast
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from functools import lru_cache
import hashlib
import heapq
import inspect
import math
import multiprocessing as mp
import os
from pathlib import Path
import time
import textwrap

import numpy as np

REFERENCE_CLASS_SHA256 = "469eb56ff821834d6d86a92799111ca05b9cec200e8fe43b8078fee8eebacc8c"
SCALAR_FIELDS = (
    "local_sum", "predicted_central_local_difference",
    "high_resolution_predicted_sample_value", "double_resolution_predicted_sample_value",
    "predicted_sample_value", "prediction_residual", "maximum_error", "quantizer_index",
    "clipped_quantizer_bin_center", "double_resolution_sample_representative",
    "sample_representative", "double_resolution_prediction_error",
    "scaled_prediction_endpoint_difference", "mapped_quantizer_index",
)
VECTOR_FIELDS = ("local_difference_vector", "weight_vector")
METHODS = (
    "calculate_local_sum", "calculate_local_difference_vector", "calculate_weight_vector",
    "calculate_predicted_central_local_difference", "calculate_prediction",
    "calculate_maximum_error", "calculate_quantization", "calculate_sample_representative",
    "calculate_prediction_error", "calculate_mapped_quantizer_index",
)
_STATE = None
_METHODS = ()


@lru_cache(maxsize=1)
def validate_reference():
    from ccsds123_i2_hlm.predictor import Predictor
    node = ast.parse(textwrap.dedent(inspect.getsource(Predictor))).body[0]
    for item in ast.walk(node):
        if "type_params" in item._fields and not getattr(item, "type_params", None):
            item._fields = tuple(n for n in item._fields if n != "type_params")
    actual = hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
    if actual != REFERENCE_CLASS_SHA256:
        raise ValueError(
            "Parallel prediction requires the validated NTNU Predictor at commit "
            "9fabb0371ebf93d72a0f83ca053e9cdb2c63e3c6. Its methods have changed; "
            "revalidate the dependency scheduler against your version. "
            "Do not bypass this source check. --workers 1 uses the original predictor."
        )
    return actual


def workspace_bytes(shape, p, full_mode):
    components = p + (3 if full_mode else 0)
    return math.prod(shape) * 8 * (1 + len(SCALAR_FIELDS) + 2 * components)


def task_geometry(rows, bands, workers, rows_per_task=0):
    # Enough row blocks to fill the pipeline, without one IPC message per pixel.
    block = rows_per_task or max(1, min(8, rows // (4 * workers)))
    blocks = math.ceil(rows / block)
    return block, blocks, min(workers, bands, blocks)


class SharedWorkspace:
    """Own mappings until encoding and reconstruction checks are finished."""
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.arrays = {}
        self.descriptors = {}
        self.bytes = 0

    def allocate(self, name, shape, fill=None):
        count = math.prod(shape)
        if count == 0:
            # P=0/reduced has zero-length vectors, which cannot be mmap'ed.
            arr = np.zeros(shape, dtype=np.int64)
            self.descriptors[name] = {"path": None, "shape": list(shape)}
        else:
            path = self.directory / (name + ".i64")
            # Exclusive creation protects unrelated existing files.
            with path.open("xb") as f:
                f.truncate(count * 8)
            arr = np.memmap(path, dtype=np.int64, mode="r+", shape=shape)
            if fill is not None and fill != 0:
                arr.fill(fill)
            self.descriptors[name] = {"path": str(path), "shape": list(shape)}
            self.bytes += count * 8
        self.arrays[name] = arr
        # Scalar indexing of a memmap subclass adds Python overhead. An ndarray
        # view uses the SAME shared buffer with the reference array semantics.
        return np.asarray(arr)

    def release_except(self, keep=()):
        for name in list(self.arrays):
            if name in keep:
                continue
            arr = self.arrays.pop(name)
            if isinstance(arr, np.memmap):
                arr._mmap.close()
            path = self.descriptors[name]["path"]
            if path:
                Path(path).unlink(missing_ok=True)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.release_except()


def _initialize_worker(header, constants, descriptors):
    global _STATE, _METHODS
    from ccsds123_i2_hlm.predictor import Predictor
    validate_reference()
    arrays = {}
    for name, desc in descriptors.items():
        shape = tuple(desc["shape"])
        arrays[name] = (np.asarray(np.memmap(desc["path"], dtype=np.int64,
                                 mode="r" if name == "image_sample" else "r+", shape=shape)
                        ) if desc["path"] else np.zeros(shape, dtype=np.int64))
    _STATE = Predictor(header, constants, arrays.pop("image_sample"))
    _STATE._Predictor__init_predictor_constants()
    for name, arr in arrays.items():
        setattr(_STATE, name, arr)
    _METHODS = tuple(getattr(_STATE, "_Predictor__" + name) for name in METHODS)


def _process_strip(task):
    block_index, band, first_row, stop_row = task
    width = _STATE.header.x_size
    for y in range(first_row, stop_row):
        for x in range(width):
            t = x + y * width
            for method in _METHODS:
                method(x, y, band, t)
    # The result message is sent only after all shared-array writes complete.
    return block_index, band, os.getpid()


def predictor_class(workspace, workers, rows_per_task, statistics):
    """Build a drop-in class; only scheduling and array allocation change."""
    from ccsds123_i2_hlm.predictor import Predictor
    validate_reference()

    class WavefrontPredictor(Predictor):
        def run_predictor(self):
            self._Predictor__init_predictor_constants()
            shape = self.image_sample.shape
            shared_input = workspace.allocate("image_sample", shape)
            shared_input[:] = self.image_sample
            for name in SCALAR_FIELDS:
                setattr(self, name, workspace.allocate(name, shape, -1))
            for name in VECTOR_FIELDS:
                setattr(self, name, workspace.allocate(
                    name, shape + (self.local_difference_values_num,), 0))
            block, blocks, count = task_geometry(shape[0], shape[2], workers, rows_per_task)
            predecessors = np.full((blocks, shape[2]), 2, dtype=np.int8)
            predecessors[0, :] -= 1
            predecessors[:, 0] -= 1
            ready = [(0, 0, 0)]  # (diagonal, row block, band)
            active = {}
            completed = 0
            total = blocks * shape[2]
            pids = set()
            peak_active = 0
            last_report = time.monotonic()
            print(f"Parallel predictor: {count} processes, {block} rows/task, "
                  f"{total} dependent tasks; shared files {workspace.bytes / 2**30:.3f} GiB.", flush=True)
            with ProcessPoolExecutor(
                max_workers=count, mp_context=mp.get_context("spawn"),
                initializer=_initialize_worker,
                initargs=(self.header, self.image_constants, workspace.descriptors),
            ) as pool:
                while ready or active:
                    while ready and len(active) < count:
                        _, b, z = heapq.heappop(ready)
                        task = (b, z, b * block, min(shape[0], (b + 1) * block))
                        active[pool.submit(_process_strip, task)] = (b, z)
                    peak_active = max(peak_active, len(active))
                    done, _ = wait(active, return_when=FIRST_COMPLETED)
                    for future in done:
                        expected = active.pop(future)
                        b, z, pid = future.result()
                        if (b, z) != expected:
                            raise RuntimeError("Predictor worker returned the wrong task.")
                        pids.add(pid)
                        completed += 1
                        for nb, nz in ((b + 1, z), (b, z + 1)):
                            if nb < blocks and nz < shape[2]:
                                predecessors[nb, nz] -= 1
                                if predecessors[nb, nz] == 0:
                                    heapq.heappush(ready, (nb + nz, nb, nz))
                    if time.monotonic() - last_report >= 10:
                        print(f"Predictor tasks: {completed}/{total}", flush=True)
                        last_report = time.monotonic()
            if completed != total:
                raise RuntimeError(f"Incomplete predictor: {completed}/{total} tasks.")
            statistics.update(
                predictor_backend="wavefront-shared-mmap",
                predictor_pool_size=count, predictor_processes_used=len(pids),
                predictor_worker_pids=sorted(pids), predictor_peak_tasks_in_flight=peak_active,
                predictor_tasks=total, rows_per_task=block, shared_workspace_bytes=workspace.bytes,
            )
            print(f"Predictor complete: {len(pids)} processes performed work. "
                  "Hybrid encoding now runs sequentially.", flush=True)

        def release_workspace(self):
            # Called by encode_one only after reconstruction/error validation.
            # Keep mapped indices for the entropy encoder, release all traces.
            for name in SCALAR_FIELDS + VECTOR_FIELDS:
                if name != "mapped_quantizer_index":
                    setattr(self, name, None)
            workspace.release_except(("mapped_quantizer_index",))

    return WavefrontPredictor
