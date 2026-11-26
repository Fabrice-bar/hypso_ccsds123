#!/usr/bin/env python3
"""
P / local-sum / prediction-mode sweep for HYPSO-1 images with the NTNU
CCSDS 123.0-B-2 verification model, compared against Vorhaug's flight baseline.

Model: https://github.com/NTNU-SmallSat-Lab/ccsds123_issue_2_verification_model
(Vorhaug, Boyle, Orlandic, WHISPERS 2024). Clone it and point --model-dir at it:
    git clone https://github.com/NTNU-SmallSat-Lab/ccsds123_issue_2_verification_model
    pip install -r ccsds123_issue_2_verification_model/requirements.txt matplotlib netCDF4 numba
Keep fast_engine.py in the same folder as this script.

What is swept (everything else is held at Vorhaug 2024, Table 3.1):
    P              1..10                        (10 values)
    local sum      wide/narrow x neighbor/column (4 values)
    pred. mode     full / reduced               (2 values)
    -> 80 configurations per image. The baseline (P=3, narrow column, reduced)
       is one of the 80, so no extra runs are needed for it.

Memory (lean engine): the model keeps every intermediate value of every sample (about
1.3 kB/sample, ~100 GB for a full 120x956x684 HYPSO cube). This script runs the
model's own Predictor and HybridEncoder code unchanged and only swaps their
storage arrays:
  * values read back only at the current sample     -> one stored value
  * values read back at earlier bands / previous t  -> one stored row
  * sample representatives (also read at row y-1)   -> two stored rows
  * debug-only arrays (strings, readable bitstream) -> discarded
That leaves the input cube and the predictor output (2 x 8 B/sample) plus the
bitstream, about 1.4 GB per full-cube job. The compressed bitstream is bit-identical to the
unmodified model; --verify checks this on your own data before the sweep.

Image files follow the model's naming convention, which gives type and size:
    <name>-<type>-<Z>x<Y>x<X>.raw     e.g. lisbon-u16be-120x956x684.raw  (BSQ, as the model)
The extension selects the layout on disk, so HYPSO .bip cubes need no conversion:
    .raw / .bsq = band sequential, .bil = band interleaved by line, .bip = by pixel
    e.g. lisbon-u16le-120x956x684.bip  (Z bands, Y frames/lines, X pixels per line)
NetCDF (.nc) files, e.g. HYPSO L1a products, are read directly (pip install
netCDF4). The variable products/dn (raw counts) is used when present; the plan
printout shows the variable, the dimension -> Y/X/Z mapping and the size, so
check those once (override with --nc-var / --nc-axes).
The plan printout shows each image's min..max value; values far above 4095 for
12-bit HYPSO data usually mean the endianness (le/be) in the name is wrong.

Examples
    # full images, all cores
    python p_sweep.py --model-dir ../ccsds123_issue_2_verification_model \
        --images data/*.raw --out results_full

    # 256x256 centre crops, 8 workers, check against the reference model first
    python p_sweep.py --model-dir ../ccsds123_issue_2_verification_model \
        --images data/*.raw --center-crop 256x256 --workers 8 --verify --out results_crop

    # lossless, or lossy with other error limits (the baseline uses the same setting)
    python p_sweep.py ... --fidelity lossless
    python p_sweep.py ... --fidelity lossy --abs-error 8 --rel-error 16

    # re-plot existing results without running anything
    python p_sweep.py --plot-only --out results_full

Engines (--engine):
  fast  fast_engine.py: the model's predictor and hybrid encoder re-written as
        one compiled (Numba) loop, with every constant and code table taken
        from the model. Bit-identical to the model in all tests; --verify checks
        it against the unmodified model on your own images before the sweep.
  lean  the model's own Python code with lean storage (described below).
        About 18-30 us per sample per core: 25-40 min per full cube and
        configuration, about 200 CPU-hours for 5 images x 80 configurations.
  auto (default) uses fast when numba is installed, otherwise lean.

Results are appended to <out>/results.csv after every finished job, so an
interrupted run continues where it stopped when you rerun the same command.
"""

import argparse
import contextlib
import csv
import glob
import math
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import as_strided

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

LOCAL_SUMS = ["WIDE_NEIGHBOR_ORIENTED", "NARROW_NEIGHBOR_ORIENTED", "WIDE_COLUMN_ORIENTED", "NARROW_COLUMN_ORIENTED"]
MODES = ["FULL", "REDUCED"]
SHORT = {
    "WIDE_NEIGHBOR_ORIENTED": "wide neighbor",
    "NARROW_NEIGHBOR_ORIENTED": "narrow neighbor",
    "WIDE_COLUMN_ORIENTED": "wide column",
    "NARROW_COLUMN_ORIENTED": "narrow column",
    "FULL": "full",
    "REDUCED": "reduced",
}

# Vorhaug 2024, MSc thesis Table 3.1 / ESA OBPDC 2024 Table 3 (flight configuration)
BASELINE_POINT = {"P": 3, "local_sum": "NARROW_COLUMN_ORIENTED", "mode": "REDUCED"}
VORHAUG = {
    "D": 16,
    "omega": 19,
    "R": 64,
    "nu_min": -1,
    "nu_max": 4,
    "t_inc": 2**6,
    "theta": 3,
    "phi": 3,
    "psi": 7,
    "A": 4,
    "R_rel": 16,
    "D_A": 5,  # bit depth of the absolute error limit field (model default)
    "D_R": 7,  # bit depth of the relative error limit field (model default)
    "u_max": 18,
    "gamma_star": 6,
    "gamma_0": 1,
    "word_bytes": 8,  # output word size B (model default, header value 0)
}

CSV_FIELDS = [
    "image", "crop", "fidelity", "D", "P", "local_sum", "mode", "is_baseline",
    "ny", "nx", "nz", "header_bits", "body_bits", "total_bits", "bps", "cr_vs_16bit",
    "seconds", "rss_mb", "max_abs_err", "mse", "psnr_db", "snr_db",
]
CSV_KEY_FIELDS = ["image", "crop", "fidelity", "D", "P", "local_sum", "mode", "bps"]


def fidelity_tag(fid):
    """fid = (A*, R*); (0, 0) is lossless."""
    return "lossless" if fid == (0, 0) else f"A{fid[0]}_R{fid[1]}"


def fidelity_text(tag):
    if tag == "lossless":
        return "lossless"
    a, r = tag[1:].split("_R")
    return f"A*={a}, R*={r}"

FNAME_RE = re.compile(r"^(?P<name>.+)-(?P<fmt>[us](?:8|16|32|64)(?:be|le))-(?P<z>\d+)x(?P<y>\d+)x(?P<x>\d+)\.(?P<ext>raw|bsq|bil|bip)$")
# File layout on disk by extension: .raw/.bsq = band sequential (the model's convention)
LAYOUT = {"raw": "bsq", "bsq": "bsq", "bil": "bil", "bip": "bip"}
DTYPES = {f"{s}{b}{e}": np.dtype(("<" if e == "le" else ">") + ("u" if s == "u" else "i") + str(b // 8))
          for s in "us" for b in (8, 16, 32, 64) for e in ("be", "le")}

# ----------------------------------------------------------------------------
# Lean storage used in place of the model's full-image arrays
# ----------------------------------------------------------------------------
#
# The model visits samples in order t = 0, 1, 2, ... (BIP inside the predictor,
# BIL inside the encoder) and, apart from sample_representative, only ever looks
# back at (a) the current sample, (b) earlier bands of the current sample or
# (c) the previous sample t-1 in the same band. For those arrays one row of
# storage is enough: a numpy view whose y-stride is 0 makes every y alias the
# same row, so slot (x, z) holds the most recent value written at (x, z).
# When the model reads t-1 = (y-1, X-1) at x = 0, that slot has not been
# overwritten yet in row y, so the value is still the right one. These views
# are plain numpy arrays, so there is no Python overhead per access.


def one_row(shape, fill):
    """Array that looks like `shape` = (Y, ...) but stores a single y-row."""
    buf = np.full(shape[1:], fill, dtype=np.int64)
    return as_strided(buf, shape=shape, strides=(0,) + buf.strides)


def one_value(shape, fill):
    """Array that looks like `shape` but stores one value (read back only at the index just written)."""
    buf = np.full(1, fill, dtype=np.int64)
    return as_strided(buf, shape=shape, strides=(0,) * len(shape))


class TwoRows:
    """sample_representative is read at rows y and y-1 (local sums, directional
    differences), so it needs a 2-row ring buffer: row y is stored at y % 2."""

    __slots__ = ("shape", "buf")

    def __init__(self, shape, fill):
        self.shape = tuple(shape)
        self.buf = np.full((2,) + self.shape[1:], fill, dtype=np.int64)

    def __getitem__(self, k):
        return self.buf[k[0] & 1, k[1], k[2]]

    def __setitem__(self, k, value):
        self.buf[k[0] & 1, k[1], k[2]] = value


class Sink:
    """Stands in for a debug-only array: writes are dropped, reads return ''."""

    __slots__ = ()

    def __getitem__(self, key):
        return ""

    def __setitem__(self, key, value):
        pass


# Predictor arrays only read back at the sample just written
PRED_SAME_SAMPLE = [
    "predicted_central_local_difference", "high_resolution_predicted_sample_value",
    "double_resolution_predicted_sample_value", "predicted_sample_value", "prediction_residual",
    "maximum_error", "quantizer_index", "clipped_quantizer_bin_center",
    "double_resolution_sample_representative", "scaled_prediction_endpoint_difference",
]
# Hybrid-encoder arrays that are only written for debugging/CSV output
ENC_DEBUG_ONLY = [
    "code_index", "input_symbol", "low_entropy_codes", "high_entropy_codes", "rescale_bits",
    "codewords", "codewords_binary", "entropy_type", "current_active_prefix",
    "prefix_match_index", "bitstream_readable",
]

# ----------------------------------------------------------------------------
# Model import (done once per process)
# ----------------------------------------------------------------------------

_M = {}


def _init_model(model_dir):
    if _M:
        return _M
    model_dir = str(Path(model_dir).resolve())
    if model_dir not in sys.path:
        sys.path.insert(0, model_dir)
    from bitarray import bitarray
    from ccsds123_i2_hlm import constants as const
    from ccsds123_i2_hlm import header as hd
    from ccsds123_i2_hlm import hybrid_encoder as henc
    from ccsds123_i2_hlm import hybrid_encoder_tables as henc_tables
    from ccsds123_i2_hlm import predictor as pred

    class LeanPredictor(pred.Predictor):
        # Overrides Predictor.__init_predictor_arrays (name-mangled)
        def _Predictor__init_predictor_arrays(self):
            Y, X, Z = self.image_sample.shape
            C = self.local_difference_values_num
            self.local_sum = one_row((Y, X, Z), -1)
            self.local_difference_vector = one_row((Y, X, Z, C), 0)
            self.weight_vector = one_row((Y, X, Z, C), 0)
            self.double_resolution_prediction_error = one_row((Y, X, Z), -1)
            self.sample_representative = TwoRows((Y, X, Z), -1)
            self.mapped_quantizer_index = np.full((Y, X, Z), -1, dtype=np.int64)  # the output
            for name in PRED_SAME_SAMPLE:
                setattr(self, name, one_value((Y, X, Z), -1))

    class LeanHybridEncoder(henc.HybridEncoder):
        # Overrides HybridEncoder.__init_encoder_arrays (name-mangled)
        def _HybridEncoder__init_encoder_arrays(self):
            Y, X, Z = self.mapped_quantizer_index.shape
            self.accumulator = one_row((Y, X, Z), 0)
            # The counter has no z index but is updated once per band in BIL order,
            # so one row is not enough; the full (Y, X) array is only 8 B/pixel.
            self.counter = np.zeros((Y, X), dtype=np.int64)
            self.variable_length_code = one_value((Y, X, Z), -1)
            self.active_prefix = [""] * 16
            for name in ENC_DEBUG_ONLY:
                setattr(self, name, Sink())
            self.flush_codes = np.full((16), fill_value="", dtype="U10")
            self.accumulator_final = np.full((Z), fill_value="", dtype="U46")
            self.counter[0, 0] = 2**self.initial_count_exponent
            # Same arbitrary default as the model (no accumulator init file): 4 * 2^gamma_0
            self.accumulator[0, 0] = 4 * self.counter[0, 0]
            self.bitstream = bitarray()

    _M.update(hd=hd, const=const, pred=pred, henc=henc, henc_tables=henc_tables,
              LeanPredictor=LeanPredictor, LeanHybridEncoder=LeanHybridEncoder)
    return _M


# ----------------------------------------------------------------------------
# Image handling
# ----------------------------------------------------------------------------


# --- NetCDF (.nc) input ------------------------------------------------------
# HYPSO L1a products are NetCDF-4 files with the raw uint16 sensor counts in
# products/dn (lines x samples x bands). The script reads that variable (or
# products/Lt) when it exists; otherwise
# it takes the largest 3-D integer variable. --nc-var picks one explicitly.
# The band axis is found from the dimension names (band, wavelength, ...) or,
# failing that, as the smallest axis; the other two keep their stored order as
# Y (lines/frames) then X (pixels). --nc-axes overrides this, e.g. "yxz" or "zyx".
# Values are read raw: no scale_factor/offset, no fill-value masking.

NC_OPTS = {"var": None, "axes": None}  # set from the command line, copied to the workers
NC_DEFAULT_VARS = ["products/dn", "products/Lt", "dn", "Lt"]
BAND_WORDS = ("band", "wavelength", "wl", "lambda", "spectral", "channel")


def _nc_open(path):
    """Open a NetCDF file with whichever reader is installed. Returns (kind, handle)."""
    errors = []
    try:
        import netCDF4
        return "netcdf4", netCDF4.Dataset(path, "r")
    except ImportError:
        errors.append("netCDF4 not installed")
    except OSError as e:
        errors.append(f"netCDF4: {e}")
    try:
        import h5py  # NetCDF-4 files are HDF5 files
        return "h5py", h5py.File(path, "r")
    except ImportError:
        errors.append("h5py not installed")
    except OSError as e:
        errors.append(f"h5py: {e}")
    try:
        from scipy.io import netcdf_file  # classic NetCDF-3 only
        return "scipy", netcdf_file(path, "r", mmap=False, maskandscale=False)
    except ImportError:
        errors.append("scipy not installed")
    except Exception as e:
        errors.append(f"scipy: {e}")
    raise RuntimeError(f"Cannot open {path} ({'; '.join(errors)}). Install a reader: pip install netCDF4")


def _nc_variables(kind, f):
    """Yield (path, variable, dimension names) for every variable in the file."""
    if kind == "netcdf4":
        def walk(g, prefix):
            for name, v in g.variables.items():
                yield prefix + name, v, tuple(v.dimensions)
            for gname, sub in g.groups.items():
                yield from walk(sub, prefix + gname + "/")
        yield from walk(f, "")
    elif kind == "h5py":
        import h5py
        found = []

        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                dims = []
                for d in obj.dims:
                    try:
                        dims.append(d[0].name.split("/")[-1] if len(d) else (d.label or ""))
                    except Exception:
                        dims.append("")
                found.append((name, obj, tuple(dims)))
        f.visititems(visit)
        yield from found
    else:
        for name, v in f.variables.items():
            yield name, v, tuple(v.dimensions)


def _vdtype(v):
    dt = getattr(v, "dtype", None)
    return np.dtype(dt if dt is not None else v.data.dtype)  # scipy variables have no .dtype


def _nc_attr(kind, v, key):
    try:
        if kind == "netcdf4":
            return v.getncattr(key) if key in v.ncattrs() else None
        if kind == "h5py":
            val = v.attrs.get(key)
            return val.decode() if isinstance(val, bytes) else val
        val = v._attributes.get(key)
        return val.decode() if isinstance(val, bytes) else val
    except Exception:
        return None


def _nc_pick(kind, f, path):
    """Choose the image variable and work out its axes. Returns an info dict."""
    allvars = {p.lstrip("/"): (v, d) for p, v, d in _nc_variables(kind, f)}
    if NC_OPTS["var"]:
        key = NC_OPTS["var"].lstrip("/")
        if key not in allvars:
            raise ValueError(f"{path}: no variable {key}. Variables: {', '.join(sorted(allvars))}")
    else:
        key = next((k for k in NC_DEFAULT_VARS if k in allvars and len(allvars[k][0].shape) == 3), None)
        if key is None:
            cands = [(int(np.prod(v.shape)), k) for k, (v, _) in allvars.items()
                     if len(v.shape) == 3 and _vdtype(v).kind in "iu"]
            if not cands:
                three_d = [f"{k} {tuple(v.shape)} {_vdtype(v)}" for k, (v, _) in allvars.items() if len(v.shape) == 3]
                raise ValueError(f"{path}: no 3-D integer variable found (3-D variables: {three_d or 'none'}). "
                                 "CCSDS 123 needs integer samples, e.g. raw L1a counts; choose one with --nc-var")
            key = max(cands)[1]
    v, dims = allvars[key]
    shape = tuple(int(s) for s in v.shape)
    if len(shape) != 3:
        raise ValueError(f"{path}: variable {key} has shape {shape}, need 3-D")
    dt = _vdtype(v)
    if dt.kind not in "iu":
        raise ValueError(f"{path}: variable {key} is {dt}; CCSDS 123 needs integer samples (use raw L1a counts)")
    unsigned_attr = str(_nc_attr(kind, v, "_Unsigned")).lower() == "true"
    signed = dt.kind == "i" and not unsigned_attr

    if NC_OPTS["axes"]:
        order = NC_OPTS["axes"].lower().replace(",", "")
        if sorted(order) != ["x", "y", "z"]:
            raise ValueError("--nc-axes must name the stored axes with y, x, z, e.g. yxz")
        pos = {a: order.index(a) for a in "yxz"}
        guessed = False
    else:
        named = [i for i, d in enumerate(dims) if any(w in (d or "").lower() for w in BAND_WORDS)]
        guessed = len(named) != 1
        zpos = named[0] if not guessed else int(np.argmin(shape))
        rest = [i for i in range(3) if i != zpos]
        pos = {"y": rest[0], "x": rest[1], "z": zpos}
    nz, ny, nx = shape[pos["z"]], shape[pos["y"]], shape[pos["x"]]
    name = re.sub(r"[-_.]?l1a$", "", Path(path).stem, flags=re.I) or Path(path).stem
    return {"name": name, "var": key, "dims": dims, "pos": pos, "nz": nz, "ny": ny, "nx": nx,
            "signed": signed, "bits": dt.itemsize * 8, "unsigned_attr": unsigned_attr, "guessed": guessed}


_NC_INFO = {}


def nc_info(path):
    key = (str(Path(path).resolve()), NC_OPTS["var"], NC_OPTS["axes"])
    if key not in _NC_INFO:
        kind, f = _nc_open(path)
        try:
            _NC_INFO[key] = _nc_pick(kind, f, path)
        finally:
            f.close()
    return _NC_INFO[key]


class NcCube:
    """Reads a (Y, X, Z) block of a NetCDF variable on demand, like a memmap."""

    def __init__(self, path):
        self.path = path
        self.info = nc_info(path)

    def __getitem__(self, key):
        ys, xs, zs = key
        pos = self.info["pos"]
        stored = [None] * 3
        stored[pos["y"]], stored[pos["x"]], stored[pos["z"]] = ys, xs, zs
        kind, f = _nc_open(self.path)
        try:
            v = dict((p.lstrip("/"), var) for p, var, _ in _nc_variables(kind, f))[self.info["var"]]
            if kind == "netcdf4":
                v.set_auto_maskandscale(False)
            arr = np.asarray(v[tuple(stored)])
        finally:
            f.close()
        if self.info["unsigned_attr"] and arr.dtype.kind == "i":
            arr = arr.view(arr.dtype.str.replace("i", "u"))
        return arr.transpose(pos["y"], pos["x"], pos["z"])


def is_nc(path):
    return Path(path).suffix.lower() == ".nc"


def parse_image_name(path):
    """Return (name, type, Z, Y, X). Type is e.g. 'u16be'; its first letter gives the sign."""
    if is_nc(path):
        i = nc_info(path)
        return i["name"], f"{'s' if i['signed'] else 'u'}{i['bits']}", i["nz"], i["ny"], i["nx"]
    m = FNAME_RE.match(Path(path).name)
    if not m:
        raise ValueError(f"{path}: use a .nc file or name raw files <name>-<type>-<Z>x<Y>x<X>.<raw|bsq|bil|bip> "
                         "(e.g. lisbon-u16be-120x956x684.raw or lisbon-u16le-120x956x684.bip)")
    return m["name"], m["fmt"], int(m["z"]), int(m["y"]), int(m["x"])


def open_memmap(path):
    """Open an image as a (Y, X, Z) array-like, whatever the file format or layout."""
    if is_nc(path):
        return NcCube(path)
    _, fmt, nz, ny, nx = parse_image_name(path)
    layout = LAYOUT[FNAME_RE.match(Path(path).name)["ext"]]
    expected = nz * ny * nx * DTYPES[fmt].itemsize
    if os.path.getsize(path) != expected:
        raise ValueError(f"{path}: file is {os.path.getsize(path)} bytes, name says {expected} bytes")
    if layout == "bsq":
        return np.memmap(path, dtype=DTYPES[fmt], mode="r", shape=(nz, ny, nx)).transpose(1, 2, 0)
    if layout == "bil":
        return np.memmap(path, dtype=DTYPES[fmt], mode="r", shape=(ny, nz, nx)).transpose(0, 2, 1)
    return np.memmap(path, dtype=DTYPES[fmt], mode="r", shape=(ny, nx, nz))


def parse_slice(text, size, what):
    a, b = text.split(":")
    a = int(a) if a else 0
    b = int(b) if b else size
    if not (0 <= a < b <= size):
        raise ValueError(f"{what} range {text} outside 0..{size}")
    return a, b


def resolve_crop(args, ny, nx, nz):
    """Return (y0, y1, x0, x1, z0, z1) for one image."""
    y0, y1, x0, x1 = 0, ny, 0, nx
    if args.center_crop:
        h, w = (int(v) for v in args.center_crop.lower().split("x"))
        if h > ny or w > nx:
            raise ValueError(f"--center-crop {args.center_crop} larger than image {ny}x{nx}")
        y0, x0 = (ny - h) // 2, (nx - w) // 2
        y1, x1 = y0 + h, x0 + w
    elif args.crop:
        ys, xs = args.crop.split(",")
        y0, y1 = parse_slice(ys, ny, "y")
        x0, x1 = parse_slice(xs, nx, "x")
    z0, z1 = parse_slice(args.bands, nz, "band") if args.bands else (0, nz)
    return y0, y1, x0, x1, z0, z1


def crop_label(c, ny, nx, nz):
    y0, y1, x0, x1, z0, z1 = c
    if (y0, y1, x0, x1, z0, z1) == (0, ny, 0, nx, 0, nz):
        return "full"
    return f"y{y0}:{y1}_x{x0}:{x1}_z{z0}:{z1}"


def load_cube(path, c, native=False):
    """Load a (cropped) cube as (Y, X, Z), the array layout the model works on.
    int64 for the model; native=True keeps a small integer type (fast engine)."""
    _, fmt, _, _, _ = parse_image_name(path)
    mm = open_memmap(path)
    y0, y1, x0, x1, z0, z1 = c
    block = mm[y0:y1, x0:x1, z0:z1]
    keep = native and block.dtype.kind in "iu" and block.dtype.itemsize <= 4
    cube = np.ascontiguousarray(block, dtype=block.dtype.newbyteorder("=") if keep else np.int64)
    del mm
    return cube, fmt[0] == "s"


# ----------------------------------------------------------------------------
# Header (all values set explicitly, in the model's encoded form)
# ----------------------------------------------------------------------------


def make_header(ny, nx, nz, signed, P, local_sum, mode, fid, D, overrides=None):
    """Build the header from scratch. The model's set_* methods take the real
    parameter value and do the standard's encoding themselves (e.g. Omega -> Omega-4).
    `overrides` replaces any VORHAUG entry, e.g. {"nu_max": 6, "t_inc": 2**11}."""
    hd = _M["hd"]
    v = {**VORHAUG, **(overrides or {})}
    h = hd.Header()  # no image name: nothing is derived from a file
    h.set_x_size(nx)
    h.set_y_size(ny)
    h.set_z_size(nz)
    h.set_sample_type(hd.SampleType.SIGNED_INTEGER if signed else hd.SampleType.UNSIGNED_INTEGER)
    h.set_dynamic_range(D)
    h.set_encoding_order_bil()
    h.set_output_word_size(v["word_bytes"])
    h.set_entropy_coder_type(hd.EntropyCoderType.HYBRID)

    h.set_sample_representative_flag(hd.SampleRepresentativeFlag.INCLUDED)
    h.set_prediction_bands_num(P)
    h.set_prediction_mode(hd.PredictionMode[mode])
    h.set_local_sum_type(hd.LocalSumType[local_sum])
    h.set_weight_exponent_offset_flag(hd.WeightExponentOffsetFlag.ALL_ZERO)
    h.set_weight_exponent_offset_table_flag(hd.WeightExponentOffsetTableFlag.NOT_INCLUDED)
    h.set_register_size(v["R"])
    h.set_weight_component_resolution(v["omega"])
    h.set_weight_update_change_interval(v["t_inc"])
    h.set_weight_update_initial_parameter(v["nu_min"])
    h.set_weight_update_final_parameter(v["nu_max"])
    h.set_weight_init_method(hd.WeightInitMethod.DEFAULT)
    h.set_weight_init_table_flag(hd.WeightInitTableFlag.NOT_INCLUDED)
    h.set_weight_init_resolution(0)

    h.set_periodic_error_updating_flag(hd.PeriodicErrorUpdatingFlag.NOT_USED)
    h.set_error_update_period_exponent(0)
    h.set_absolute_error_limit_assignment_method(hd.ErrorLimitAssignmentMethod.BAND_INDEPENDENT)
    h.set_relative_error_limit_assignment_method(hd.ErrorLimitAssignmentMethod.BAND_INDEPENDENT)
    A, Rr = fid
    lossless = (A, Rr) == (0, 0)
    if lossless:  # error-limit fields are not written; keep Vorhaug's values as before
        A, Rr = v["A"], v["R_rel"]
    # widen the error-limit fields only if a value does not fit Vorhaug's bit depths
    h.set_absolute_error_limit_bit_depth(max(v["D_A"], int(A).bit_length()))
    h.set_relative_error_limit_bit_depth(max(v["D_R"], int(Rr).bit_length()))
    method = hd.QuantizerFidelityControlMethod
    if lossless:
        h.set_quantizer_fidelity_control_method(method.LOSSLESS)
    elif fid[1] == 0:
        h.set_quantizer_fidelity_control_method(method.ABSOLUTE_ONLY)
    elif fid[0] == 0:
        h.set_quantizer_fidelity_control_method(method.RELATIVE_ONLY)
    else:
        h.set_quantizer_fidelity_control_method(method.ABSOLUTE_AND_RELATIVE)
    h.set_absolute_error_limit_value(A)  # also fills the per-band table
    h.set_relative_error_limit_value(Rr)

    h.set_sample_representative_resolution(v["theta"])
    h.set_band_varying_damping_flag(hd.BandVaryingDampingFlag.BAND_INDEPENDENT)
    h.set_damping_table_flag(hd.DampingTableFlag.NOT_INCLUDED)
    h.set_fixed_damping_value(v["phi"])
    h.set_band_varying_offset_flag(hd.BandVaryingOffsetFlag.BAND_INDEPENDENT)
    h.set_damping_offset_table_flag(hd.OffsetTableFlag.NOT_INCLUDED)
    h.set_fixed_offset_value(0 if lossless else v["psi"])  # psi must be 0 when lossless

    h.set_unary_length_limit(v["u_max"])
    h.set_rescaling_counter_size(v["gamma_star"])
    h.set_initial_count_exponent(v["gamma_0"])
    h.set_accumulator_init_constant(0)
    h.set_accumulator_init_table_flag(hd.AccumulatorInitTableFlag.NOT_INCLUDED)

    h.set_damping_table_array_to_default()
    h.set_damping_offset_table_array_to_default()
    h.set_accumulator_init_table_to_default()
    h.check_legal_config()
    return h


# ----------------------------------------------------------------------------
# One compression run (executed in a worker process)
# ----------------------------------------------------------------------------


def _silenced():
    return contextlib.redirect_stdout(open(os.devnull, "w"))


def compress_bits(cube, signed, P, local_sum, mode, fid, D, lean=True, engine="lean", overrides=None):
    """Compress a (Y,X,Z) cube with error limits fid = (A*, R*), (0, 0) = lossless.

    Returns (header_bits, body_bits, bitstream, distortion stats or None).

    engine="fast": compiled re-implementation (fast_engine.py)
    engine="lean": the model's own code with lean storage (lean=False: unmodified model)
    """
    M = _M
    ny, nx, nz = cube.shape
    h = make_header(ny, nx, nz, signed, P, local_sum, mode, tuple(fid), D, overrides)
    k = M["const"].ImageConstants(h)
    header_bits = len(h.get_header_bitstreams()[0])
    if engine == "fast":
        import fast_engine
        nbits, packed, stats = fast_engine.compress(cube, h, k, M)
        return header_bits, nbits, packed, stats
    P_cls = M["LeanPredictor"] if lean else M["pred"].Predictor
    E_cls = M["LeanHybridEncoder"] if lean else M["henc"].HybridEncoder
    with _silenced():
        p = P_cls(h, k, cube)
        p.run_predictor()
        mqi = p.get_predictor_output()
        del p
        e = E_cls(h, k, mqi)
        e.run_encoder()
    return header_bits, len(e.bitstream), e.bitstream, None


def _init_worker(model_dir, nc_opts=None):
    if nc_opts:
        NC_OPTS.update(nc_opts)
    sys.path.insert(0, str(Path(__file__).resolve().parent))  # fast_engine.py lives next to this script
    _init_model(model_dir)


def run_job(job):
    t0 = time.perf_counter()
    engine = job.get("engine", "lean")
    cube, signed = load_cube(job["path"], job["crop"], native=engine == "fast")
    lo, hi = (-(2 ** (job["D"] - 1)), 2 ** (job["D"] - 1) - 1) if signed else (0, 2 ** job["D"] - 1)
    if cube.min() < lo or cube.max() > hi:
        raise ValueError(f"{job['image']}: sample values {cube.min()}..{cube.max()} do not fit D={job['D']} bits")
    ny, nx, nz = cube.shape

    args = (signed, job["P"], job["local_sum"], job["mode"], tuple(job["fid"]), job["D"])
    if job.get("verify"):  # sweep engine vs the unmodified model, bit by bit
        hb, b_eng, s_eng, _ = compress_bits(cube, *args, engine=engine)
        _, b_ref, s_ref, _ = compress_bits(cube.astype(np.int64), *args, lean=False, engine="model")
        if engine == "fast":
            import fast_engine
            s_eng = fast_engine.bits_to01(s_eng, b_eng)
        else:
            s_eng = s_eng.to01()
        return {**job, "identical": b_eng == b_ref and s_eng == s_ref.to01(), "bits": hb + b_eng}

    header_bits, body_bits, _, st = compress_bits(cube, *args, engine=engine)
    word_bits = 8 * VORHAUG["word_bytes"]
    total = header_bits + body_bits
    total += (word_bits - total % word_bits) % word_bits  # pad to output word, as the model's file
    n = ny * nx * nz
    try:
        import psutil
        rss = psutil.Process(os.getpid()).memory_info().rss / 2**20
    except Exception:
        rss = float("nan")
    return {
        "image": job["image"], "crop": job["crop_label"], "fidelity": job["fidelity"], "D": job["D"],
        "P": job["P"], "local_sum": job["local_sum"], "mode": job["mode"], "is_baseline": int(job["is_baseline"]),
        "ny": ny, "nx": nx, "nz": nz, "header_bits": header_bits, "body_bits": body_bits, "total_bits": total,
        "bps": total / n, "cr_vs_16bit": 16 * n / total, "seconds": round(time.perf_counter() - t0, 1), "rss_mb": round(rss),
        **distortion_fields(st, job.get("psnr_bits", 12)),
    }


def distortion_fields(st, bits):
    """Reconstruction error of the decoded image (fast engine only; blank otherwise).
    PSNR uses the sensor's peak 2^bits - 1 (HYPSO-1: 12-bit, so 4095)."""
    if st is None:
        return {"max_abs_err": "", "mse": "", "psnr_db": "", "snr_db": ""}
    mse = st["sse"] / st["n"]
    if mse == 0:
        return {"max_abs_err": 0, "mse": 0.0, "psnr_db": "inf", "snr_db": "inf"}
    return {"max_abs_err": st["max_abs_err"], "mse": round(mse, 6),
            "psnr_db": round(10 * math.log10((2**bits - 1) ** 2 / mse), 4),
            "snr_db": round(10 * math.log10(st["sum_sq"] / st["sse"]), 4)}


# ----------------------------------------------------------------------------
# Job planning
# ----------------------------------------------------------------------------


def job_key(d):
    return (d["image"], d["crop"] if "crop_label" not in d else d["crop_label"], d["fidelity"], int(d["D"]), int(d["P"]), d["local_sum"], d["mode"])


def plan_jobs(args, images):
    fidelity = fidelity_tag(args.fid)
    grid = list(product(range(args.p_min, args.p_max + 1), LOCAL_SUMS, MODES))
    if (BASELINE_POINT["P"], BASELINE_POINT["local_sum"], BASELINE_POINT["mode"]) not in grid:
        grid.append((BASELINE_POINT["P"], BASELINE_POINT["local_sum"], BASELINE_POINT["mode"]))
    jobs = []
    for path in images:
        name, _, nz, ny, nx = parse_image_name(path)
        c = resolve_crop(args, ny, nx, nz)
        for P, ls, mode in grid:
            is_base = (P, ls, mode) == (BASELINE_POINT["P"], BASELINE_POINT["local_sum"], BASELINE_POINT["mode"])
            jobs.append({
                "path": str(path), "image": name, "crop": c, "crop_label": crop_label(c, ny, nx, nz),
                "samples": (c[1] - c[0]) * (c[3] - c[2]) * (c[5] - c[4]),
                "P": P, "local_sum": ls, "mode": mode, "is_baseline": is_base,
                "fid": args.fid, "fidelity": fidelity, "D": args.dynamic_range, "engine": args.engine,
                "psnr_bits": args.psnr_bits,
            })
    return jobs, len(grid)


def est_cost(job):
    """Relative runtime: ~ samples x (weights + local differences per sample)."""
    C = job["P"] + 3 * (job["mode"] == "FULL")
    return job["samples"] * (6 + C)


def est_job_mem_mb(job):
    if job.get("engine") == "fast":  # 16-bit cube + packed bitstream buffer + Numba runtime
        return job["samples"] * (2 + 2) / 2**20 + 400
    # model engine: int64 cube + mapped quantizer index + bitstream + interpreter/model overhead
    return job["samples"] * (8 + 8 + 2) / 2**20 + 250


def read_results(csv_path):
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        missing = [c for c in CSV_KEY_FIELDS if c not in (reader.fieldnames or [])]
        if missing:
            raise SystemExit(f"{csv_path} exists but was not written by this script (missing columns: "
                             f"{', '.join(missing[:5])}{'...' if len(missing) > 5 else ''}).\n"
                             "Use another --out folder, or move/rename that file. Nothing was changed.")
        return [r for r in reader if r.get("image")]


def upgrade_csv(csv_path):
    """Rewrite a results.csv from an older version of this script with the current
    columns (new columns left blank), so new rows can be appended to it."""
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames == CSV_FIELDS:
            return
        rows = list(reader)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in CSV_FIELDS})


# ----------------------------------------------------------------------------
# Plots and summary
# ----------------------------------------------------------------------------

# Categorical slots 1-4 of the reference palette (validated for adjacent lines)
COLORS = {
    "WIDE_NEIGHBOR_ORIENTED": "#257de0",
    "NARROW_NEIGHBOR_ORIENTED": "#f47230",
    "WIDE_COLUMN_ORIENTED": "#00b894",
    "NARROW_COLUMN_ORIENTED": "#f2a600",
}
LINESTYLE = {"FULL": "-", "REDUCED": "--"}
MARKER = {"FULL": "o", "REDUCED": "s"}
INK, INK2, GRID = "#191919", "#666666", "#e5e5e5"


def make_plots(rows, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    groups = {}
    for r in rows:
        groups.setdefault((r["crop"], r["fidelity"], r["D"]), []).append(r)

    for (crop, fid, D), rs in groups.items():
        images = sorted({r["image"] for r in rs})
        tab = {(r["image"], int(r["P"]), r["local_sum"], r["mode"]): float(r["bps"]) for r in rs}
        base = {im: tab.get((im, BASELINE_POINT["P"], BASELINE_POINT["local_sum"], BASELINE_POINT["mode"])) for im in images}
        Ps = sorted({int(r["P"]) for r in rs})
        tag = f"{crop}_{fid}_D{D}"
        fid_txt = fidelity_text(fid)

        def style(ax):
            ax.grid(True, color=GRID, linewidth=0.8)
            ax.set_axisbelow(True)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            for s in ("left", "bottom"):
                ax.spines[s].set_color(INK2)
            ax.tick_params(colors=INK2, labelsize=8)
            ax.set_xticks(Ps)

        legend_items = [Line2D([], [], color=COLORS[ls], linestyle=LINESTYLE[m], marker=MARKER[m], markersize=5, linewidth=2,
                               label=f"{SHORT[ls]}, {SHORT[m]}") for ls in LOCAL_SUMS for m in MODES]
        legend_items.append(Line2D([], [], color=INK, marker="*", linestyle=":", markersize=11, linewidth=1,
                                   label="Vorhaug 2024 baseline (P=3, narrow column, reduced)"))

        # --- Figure 1: metric vs P, one panel per image + mean --------------
        def per_image_figure(t, b_of, ylabel, title, fname):
            panels = images + (["Mean over images"] if len(images) > 1 else [])
            ncol = min(3, len(panels))
            nrow = math.ceil(len(panels) / ncol)
            top_in, legend_in = 0.45, 0.95  # space for title and legend (inches)
            fig_h = 3.3 * nrow + top_in + legend_in
            fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, fig_h), squeeze=False)
            for ax, panel in zip(axes.flat, panels):
                for ls, m in product(LOCAL_SUMS, MODES):
                    if panel == "Mean over images":
                        ys = [np.mean([t[(im, P, ls, m)] for im in images]) if all((im, P, ls, m) in t for im in images) else np.nan for P in Ps]
                    else:
                        ys = [t.get((panel, P, ls, m), np.nan) for P in Ps]
                    ax.plot(Ps, ys, color=COLORS[ls], linestyle=LINESTYLE[m], marker=MARKER[m], markersize=4, linewidth=1.6)
                if panel == "Mean over images":
                    vals = [b_of[im] for im in images]
                    b = np.mean(vals) if all(v is not None for v in vals) else None
                else:
                    b = b_of[panel]
                if b is not None and not np.isnan(b):
                    ax.axhline(b, color=INK, linestyle=":", linewidth=1)
                    ax.plot([BASELINE_POINT["P"]], [b], marker="*", color=INK, markersize=11, zorder=5, linestyle="none")
                ax.set_title(panel, fontsize=10, color=INK, loc="left")
                ax.set_xlabel("Prediction bands P", fontsize=9, color=INK2)
                ax.set_ylabel(ylabel, fontsize=9, color=INK2)
                style(ax)
            for ax in list(axes.flat)[len(panels):]:
                ax.axis("off")
            fig.suptitle(f"{title}  ({fid_txt}, D={D}, crop: {crop})", x=0.01, ha="left",
                         y=1 - 0.12 / fig_h, va="top", fontsize=11, color=INK)
            fig.legend(handles=legend_items, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=3, fontsize=8, frameon=False)
            fig.tight_layout(rect=(0, legend_in / fig_h, 1, 1 - top_in / fig_h))
            for ext in ("png", "pdf"):
                fig.savefig(out_dir / f"{fname}_{tag}.{ext}", dpi=200)
            plt.close(fig)

        per_image_figure(tab, base, "Rate (bits/sample)", "Compressed rate vs P", "rate_vs_P")

        # PSNR of the decoded image (lossy runs with the fast engine)
        def num(v):
            try:
                return float(v)
            except (TypeError, ValueError):
                return None
        ptab = {(r["image"], int(r["P"]), r["local_sum"], r["mode"]): num(r.get("psnr_db")) for r in rs}
        ptab = {k: v for k, v in ptab.items() if v is not None and np.isfinite(v)}
        if fid != "lossless" and ptab:
            pbase = {im: ptab.get((im, BASELINE_POINT["P"], BASELINE_POINT["local_sum"], BASELINE_POINT["mode"])) for im in images}
            per_image_figure(ptab, pbase, "PSNR (dB)  - higher is better", "Decoded image PSNR vs P", "psnr_vs_P")

        # --- Figure 2: % change vs baseline, mean over images --------------
        fig, ax = plt.subplots(figsize=(7.5, 4.6))
        ax.axhline(0, color=INK, linestyle=":", linewidth=1)
        for ls, m in product(LOCAL_SUMS, MODES):
            ys = []
            for P in Ps:
                vals = [100 * (tab[(im, P, ls, m)] / base[im] - 1) for im in images if (im, P, ls, m) in tab and base[im]]
                ys.append(np.mean(vals) if len(vals) == len(images) else np.nan)
            ax.plot(Ps, ys, color=COLORS[ls], linestyle=LINESTYLE[m], marker=MARKER[m], markersize=4, linewidth=1.6)
        ax.plot([BASELINE_POINT["P"]], [0], marker="*", color=INK, markersize=11, zorder=5, linestyle="none")
        ax.set_xlabel("Prediction bands P", fontsize=9, color=INK2)
        ax.set_ylabel("Rate change vs baseline (%)  - lower is better", fontsize=9, color=INK2)
        ax.set_title(f"Mean rate change vs Vorhaug baseline, {len(images)} images ({fid_txt}, crop: {crop})", fontsize=10, color=INK, loc="left")
        style(ax)
        ax.legend(handles=legend_items, fontsize=7.5, frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(out_dir / f"delta_vs_baseline_{tag}.{ext}", dpi=200)
        plt.close(fig)

        # --- Summary table ---------------------------------------------------
        lines = [f"# Summary ({fid_txt}, D={D}, crop: {crop})", "",
                 "| Image | Baseline bps | Best config | Best bps | Change vs baseline | PSNR baseline / best (dB) |",
                 "|---|---|---|---|---|---|"]
        pt = {(r["image"], int(r["P"]), r["local_sum"], r["mode"]): r.get("psnr_db", "") for r in rs}
        for im in images:
            cand = [(v, k) for k, v in tab.items() if k[0] == im]
            v, k = min(cand)
            bk = (im, BASELINE_POINT["P"], BASELINE_POINT["local_sum"], BASELINE_POINT["mode"])
            ps_txt = f"{pt.get(bk) or '-'} / {pt.get(k) or '-'}"
            lines.append(f"| {im} | {base[im]:.4f} | P={k[1]}, {SHORT[k[2]]}, {SHORT[k[3]]} | {v:.4f} | "
                         f"{100 * (v / base[im] - 1):+.2f}% | {ps_txt} |")
        mean_delta = {}
        for P, ls, m in product(Ps, LOCAL_SUMS, MODES):
            vals = [100 * (tab[(im, P, ls, m)] / base[im] - 1) for im in images if (im, P, ls, m) in tab and base[im]]
            if len(vals) == len(images):
                mean_delta[(P, ls, m)] = np.mean(vals)
        if mean_delta:
            (P, ls, m), d = min(mean_delta.items(), key=lambda kv: kv[1])
            wins = sum(tab[(im, P, ls, m)] < base[im] for im in images)
            lines += ["", f"Best single configuration for all images: P={P}, {SHORT[ls]}, {SHORT[m]}: "
                          f"mean change {d:+.2f}% vs baseline, better than baseline on {wins}/{len(images)} images."]
        (out_dir / f"summary_{tag}.md").write_text("\n".join(lines) + "\n")
        print("\n".join(lines))


# ----------------------------------------------------------------------------
# Scene report: quicklooks and simple statistics that relate to compressibility
# ----------------------------------------------------------------------------


def nc_wavelengths(path):
    """Band centre wavelengths (nm) from a HYPSO L1a file, or None."""
    if not is_nc(path):
        return None
    try:
        kind, f = _nc_open(path)
    except Exception:
        return None
    try:
        for p, v, _ in _nc_variables(kind, f):
            if p.lstrip("/").lower() in ("metadata/corrections/wavelengths", "wavelengths", "wavelength"):
                w = np.asarray(v[:] if kind != "scipy" else v.data, dtype=float).ravel()
                return w
    except Exception:
        return None
    finally:
        f.close()
    return None


def _robust_noise(d, k):
    """Noise sigma from k-th order differences d (robust to edges): MAD scaled."""
    mad = np.median(np.abs(d - np.median(d)))
    return 1.4826 * mad / math.sqrt({1: 2, 2: 6}[k])


def scene_report(images, out_dir, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sdir = out_dir / "scenes"
    sdir.mkdir(parents=True, exist_ok=True)
    rows, spectra = [], {}
    for path in images:
        name, _, nz, ny, nx = parse_image_name(path)
        c = resolve_crop(args, ny, nx, nz)
        cube, _ = load_cube(path, c, native=True)  # (lines, samples, bands)
        Y, X, Z = cube.shape
        wl = nc_wavelengths(path)
        wl = wl[c[4]:c[5]] if wl is not None and len(wl) >= c[5] else None

        def band_near(nm, frac):
            return int(np.argmin(np.abs(wl - nm))) if wl is not None else int(frac * (Z - 1))
        rgb_idx = [band_near(630, 0.62), band_near(550, 0.38), band_near(470, 0.17)]
        rgb = np.stack([cube[:, :, i].astype(np.float32) for i in rgb_idx], axis=-1)
        lo, hi = np.percentile(rgb, 2, axis=(0, 1)), np.percentile(rgb, 98, axis=(0, 1))
        rgb = np.clip((rgb - lo) / np.maximum(hi - lo, 1), 0, 1) ** (1 / 1.8)
        plt.imsave(sdir / f"{name}_rgb.png", rgb)

        corr_band, corr_line, corr_pix, noise, mean_spec = [], [], [], [], []
        for z in range(Z):
            b = cube[:, :, z].astype(np.float32)
            mean_spec.append(float(b.mean()))
            if z + 1 < Z:
                corr_band.append(np.corrcoef(b.ravel(), cube[:, :, z + 1].astype(np.float32).ravel())[0, 1])
            corr_line.append(np.corrcoef(b[1:].ravel(), b[:-1].ravel())[0, 1])
            corr_pix.append(np.corrcoef(b[:, 1:].ravel(), b[:, :-1].ravel())[0, 1])
            noise.append(_robust_noise((b[2:] - 2 * b[1:-1] + b[:-2]).ravel(), 2))
        spectra[name] = (wl if wl is not None else np.arange(Z), np.array(mean_spec))
        rows.append({
            "image": name, "lines": Y, "samples": X, "bands": Z,
            "min_dn": int(cube.min()), "max_dn": int(cube.max()), "mean_dn": round(float(cube.mean()), 1),
            "saturated_pct": round(100 * float(np.mean(cube >= 4095)), 4),
            "noise_sigma_dn": round(float(np.median(noise)), 2),
            "corr_adjacent_band": round(float(np.nanmean(corr_band)), 5),
            "corr_along_track": round(float(np.nanmean(corr_line)), 5),
            "corr_cross_track": round(float(np.nanmean(corr_pix)), 5),
            "rgb_bands": "/".join(str(i) for i in rgb_idx),
        })
        print(f"  {name}: {Y}x{X}x{Z}, mean {rows[-1]['mean_dn']} DN, noise ~{rows[-1]['noise_sigma_dn']} DN, "
              f"corr band/along/cross {rows[-1]['corr_adjacent_band']}/{rows[-1]['corr_along_track']}/{rows[-1]['corr_cross_track']}")

    with open(sdir / "scene_stats.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    for (name, (x, y)), col in zip(sorted(spectra.items()), ["#257de0", "#f47230", "#00b894", "#f2a600", "#8e44ad", "#666666"] * 3):
        ax.plot(x, y, color=col, linewidth=1.6, label=name.split("_")[0])
    ax.set_xlabel("Wavelength (nm)" if any(len(v[0]) and v[0][0] > 100 for v in spectra.values()) else "Band", fontsize=9)
    ax.set_ylabel("Mean raw DN", fontsize=9)
    ax.set_title("Mean spectrum per scene (raw counts)", fontsize=10, loc="left")
    ax.grid(True, color=GRID)
    ax.legend(fontsize=8, frameon=False)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(sdir / "scene_spectra.png", dpi=200)
    plt.close(fig)
    print(f"\nQuicklooks, scene_stats.csv and scene_spectra.png written to {sdir}")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------


def collect_images(items):
    out = []
    for it in items:
        if any(ch in it for ch in "*?["):  # PowerShell/cmd do not expand wildcards
            out += sorted(Path(q) for q in glob.glob(it))
            continue
        p = Path(it)
        out += sorted(q for q in p.iterdir() if FNAME_RE.match(q.name) or is_nc(q)) if p.is_dir() else [p]
    if not out:
        raise SystemExit(f"No image files found in: {' '.join(items)}")
    names = [parse_image_name(p)[0] for p in out]  # also fails early on bad names
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup:
        raise SystemExit(f"Image names must be unique (results are keyed by name): {', '.join(dup)}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", help="path to the cloned ccsds123_issue_2_verification_model repository")
    ap.add_argument("--images", nargs="+", help="image files or folders: .nc files, or raw files named <name>-<type>-<Z>x<Y>x<X>.<raw|bsq|bil|bip>")
    ap.add_argument("--nc-var", help="NetCDF variable to read (default: products/dn or products/Lt, else the largest 3-D integer variable)")
    ap.add_argument("--nc-axes", help="stored axis order of the NetCDF variable as y/x/z letters, e.g. yxz (default: from dimension names)")
    ap.add_argument("--out", default="p_sweep_results", help="output folder (default: p_sweep_results)")
    ap.add_argument("--crop", help="crop as Y0:Y1,X0:X1 (frames, cross-track pixels), e.g. 300:556,200:456")
    ap.add_argument("--center-crop", help="centred crop HxW, e.g. 256x256 (overrides --crop)")
    ap.add_argument("--bands", help="band range Z0:Z1 (default: all bands)")
    ap.add_argument("--p-min", type=int, default=1)
    ap.add_argument("--p-max", type=int, default=10)
    ap.add_argument("--fidelity", choices=["lossy", "lossless"], default="lossy",
                    help="lossy (near-lossless, error limits below; default) or lossless. The baseline uses the same setting")
    ap.add_argument("--abs-error", type=int, default=VORHAUG["A"],
                    help="lossy: absolute error limit A* (default 4, Vorhaug). 0 = relative limit only")
    ap.add_argument("--rel-error", type=int, default=VORHAUG["R_rel"],
                    help="lossy: relative error limit R* (default 16, Vorhaug). 0 = absolute limit only")
    ap.add_argument("--lossless", action="store_true", help="same as --fidelity lossless")
    ap.add_argument("--psnr-bits", type=int, default=12, help="sensor bit depth for the PSNR peak 2^bits-1 (HYPSO-1: 12)")
    ap.add_argument("--dynamic-range", type=int, default=VORHAUG["D"], help="D in bits (default 16, as Vorhaug)")
    ap.add_argument("--engine", choices=["auto", "fast", "lean"], default="auto",
                    help="fast = compiled engine (needs numba, ~100x faster); lean = the model's own Python code. "
                         "auto: fast if numba is installed")
    ap.add_argument("--workers", type=int, default=0, help="parallel processes (default: all cores, capped by free memory)")
    ap.add_argument("--verify", action="store_true", help="first check bit-exactness against the unmodified model on a 12x12 crop of each image")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    ap.add_argument("--plot-only", action="store_true", help="only (re)make plots from <out>/results.csv")
    ap.add_argument("--scene-report", action="store_true",
                    help="write RGB quicklooks and scene statistics for the images to <out>/scenes, then exit")
    args = ap.parse_args()

    NC_OPTS.update(var=args.nc_var, axes=args.nc_axes)
    if args.lossless:
        args.fidelity = "lossless"
    if args.fidelity == "lossless":
        args.fid = (0, 0)
    else:
        if args.abs_error < 0 or args.rel_error < 0 or (args.abs_error, args.rel_error) == (0, 0):
            ap.error("lossy needs --abs-error and/or --rel-error > 0 (both 0 is lossless)")
        if args.abs_error >= 2**16 or args.rel_error >= 2**16:
            ap.error("error limits must be below 65536")
        args.fid = (args.abs_error, args.rel_error)
    if args.engine == "auto":
        try:
            import numba  # noqa: F401
            args.engine = "fast"
        except ImportError:
            args.engine = "lean"
            print("numba not installed: using the model's Python code (slow). pip install numba for the fast engine.")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "results.csv"

    if args.plot_only:
        make_plots(read_results(csv_path), out_dir)
        return
    if not args.images or not (args.model_dir or args.scene_report):
        ap.error("--model-dir and --images are required (--scene-report needs only --images)")

    images = collect_images(args.images)
    if args.scene_report:
        scene_report(images, out_dir, args)
        return
    jobs, n_cfg = plan_jobs(args, images)
    done = {job_key(r) for r in read_results(csv_path)}
    todo = [j for j in jobs if job_key(j) not in done]
    todo.sort(key=est_cost, reverse=True)  # longest first for better load balance

    n_cpu = os.cpu_count() or 1
    workers = args.workers or n_cpu
    mem_note = ""
    try:
        import psutil
        avail = psutil.virtual_memory().available / 2**20
        need = max((est_job_mem_mb(j) for j in todo), default=1)
        cap = max(1, int(0.8 * avail // need))
        if cap < workers:
            mem_note = f" (capped from {workers} by free memory: {avail / 1024:.1f} GB available, ~{need / 1024:.1f} GB per job)"
            workers = cap
    except ImportError:
        pass
    if os.name == "nt":
        workers = min(workers, 61)  # Windows limit for ProcessPoolExecutor
    workers = max(1, min(workers, len(todo) or 1))

    print(f"Images: {len(images)}   configurations per image: {n_cfg} "
          f"(P {args.p_min}-{args.p_max} x {len(LOCAL_SUMS)} local sums x {len(MODES)} modes)")
    print(f"Total runs: {len(jobs)}   already done: {len(jobs) - len(todo)}   to run: {len(todo)}")
    print(f"Fidelity: {fidelity_text(fidelity_tag(args.fid))}   D={args.dynamic_range}   "
          f"engine: {args.engine}   workers: {workers}{mem_note}")
    for path in images:
        j = next(j for j in jobs if j["path"] == str(path))
        y0, y1, x0, x1, z0, z1 = j["crop"]
        mm = open_memmap(path)[y0:y1, x0:x1, z0:z1]
        lo, hi = int(mm.min()), int(mm.max())
        del mm
        flag = "   <-- outside D bits: wrong type/endianness or D?" if hi >= 2 ** args.dynamic_range or lo < -(2 ** (args.dynamic_range - 1)) else ""
        print(f"  {j['image']}: crop {j['crop_label']}, {j['samples'] / 1e6:.2f} M samples, values {lo}..{hi}{flag}")
        if is_nc(path):
            i = nc_info(path)
            axes = ",".join(f"{d or '?'}->{a.upper()}" for d, a in zip(i["dims"] or ("",) * 3, sorted("yxz", key=i["pos"].get)))
            print(f"      {Path(path).name}: variable {i['var']}, {'signed' if i['signed'] else 'unsigned'} {i['bits']}-bit, "
                  f"Z x Y x X = {i['nz']} x {i['ny']} x {i['nx']}, axes {axes}")
            if i["guessed"]:
                print("      WARNING: no band dimension name found; the smallest axis was taken as the band axis. "
                      "Check the sizes above, or set --nc-axes")
    if args.dry_run:
        return

    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(args.model_dir, dict(NC_OPTS))) as pool:
        if args.verify:
            vjobs = []
            for path in images:
                name, _, nz, ny, nx = parse_image_name(path)
                _, _, _, _, z0, z1 = resolve_crop(args, ny, nx, nz)
                h, w = min(12, ny), min(12, nx)  # 144 pixels: enough for counter rescaling (gamma*=6)
                y0, x0 = (ny - h) // 2, (nx - w) // 2
                c = (y0, y0 + h, x0, x0 + w, z0, z1)
                for P, ls, m in product(range(args.p_min, args.p_max + 1), LOCAL_SUMS, MODES):
                    vjobs.append({"path": str(path), "image": name, "crop": c, "P": P, "local_sum": ls, "mode": m,
                                  "fid": args.fid, "D": args.dynamic_range, "verify": True, "engine": args.engine})
            print(f"\nVerifying the {args.engine} engine against the unmodified model: {len(vjobs)} runs on 12x12 crops ...")
            bad = [r for r in pool.map(run_job, vjobs) if not r["identical"]]
            if bad:
                for r in bad:
                    print(f"  MISMATCH: {r['image']} P={r['P']} {r['local_sum']} {r['mode']}")
                sys.exit("Verification failed - not running the sweep.")
            print("  all bitstreams identical to the reference model.")

        if todo:
            print(f"\nRunning {len(todo)} jobs ...")
            upgrade_csv(csv_path)
            new_file = not csv_path.exists() or csv_path.stat().st_size == 0
            with open(csv_path, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
                if new_file:
                    w.writeheader()
                t0 = time.time()
                total_cost = sum(est_cost(j) for j in todo)
                cost_done = 0
                futs = {pool.submit(run_job, j): j for j in todo}
                for i, fut in enumerate(as_completed(futs), 1):
                    j = futs[fut]
                    try:
                        r = fut.result()
                    except Exception as exc:
                        print(f"  [{i}/{len(todo)}] FAILED {j['image']} P={j['P']} {j['local_sum']} {j['mode']}: {exc}")
                        continue
                    w.writerow(r)
                    f.flush()
                    cost_done += est_cost(j)
                    el = time.time() - t0
                    eta = el * (total_cost / cost_done - 1)
                    print(f"  [{i}/{len(todo)}] {r['image']:<12} P={r['P']:<2} {SHORT[r['local_sum']]:<15} {SHORT[r['mode']]:<7} "
                          f"{r['bps']:.4f} bps  {r['seconds']:>7.1f} s  {r['rss_mb']} MB   elapsed {el / 60:.1f} min, ETA {eta / 60:.1f} min")

    rows = [r for r in read_results(csv_path) if r["fidelity"] == fidelity_tag(args.fid)]
    make_plots(rows, out_dir)
    print(f"\nResults: {csv_path}\nPlots and summary: {out_dir}")


if __name__ == "__main__":
    main()
