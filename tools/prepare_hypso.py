"""Extract an unchanged uint16 HYPSO DN crop as the model's BSQ input.

Example:
    python prepare_hypso.py data/adelaide_2025-09-09T01-12-29Z-l1a.nc
The default keeps all bands and selects a central 16-line x 32-column crop.
"""
import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("--output", type=Path, default=Path("raw_images"))
    p.add_argument("--height", type=int, default=16)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--row", type=int, help="First along-track line, zero-based")
    p.add_argument("--column", type=int, help="First across-track sample, zero-based")
    args = p.parse_args()
    with h5py.File(args.source, "r") as f:
        d = f["products/dn"]
        if d.ndim != 3 or d.dtype.kind != "u" or d.dtype.itemsize != 2:
            raise ValueError("Expected products/dn to contain a 3-D uint16 cube.")
        scales = [[scale.name for scale in dim.values()] for dim in d.dims]
        if scales != [["/lines"], ["/samples"], ["/bands"]]:
            raise ValueError(f"Unexpected dimension order: {scales}")
        ny, nx, nz = d.shape
        h, w = args.height, args.width
        y = args.row if args.row is not None else (ny - h) // 2
        x = args.column if args.column is not None else (nx - w) // 2
        if min(h, w) < 1 or min(y, x) < 0 or y + h > ny or x + w > nx:
            raise ValueError("Requested crop lies outside the source cube.")
        # h5py returns the stored integers without applying NetCDF scale/offset.
        cube = d[y:y + h, x:x + w, :]
        wavelength_values = f["metadata/corrections/wavelengths"][:].tolist()
        source_shape = list(d.shape)
        attrs = {k: str(v) for k, v in d.attrs.items()
                 if k not in ("DIMENSION_LIST", "REFERENCE_LIST")}
    args.output.mkdir(parents=True, exist_ok=True)
    # Avoid an 'x' in the prefix: the upstream filename parser splits on it.
    raw = args.output / f"adelaide_r{y}_c{x}-u16be-{nz}x{h}x{w}.raw"
    meta = raw.with_suffix(".json")
    if raw.exists() or meta.exists():
        raise FileExistsError(f"Choose another output directory; {raw} already exists.")
    # Input disk ordering is z,y,x (BSQ), independent of encoder sample order.
    bsq = np.ascontiguousarray(cube.transpose(2, 0, 1), dtype=">u2")
    bsq.tofile(raw)
    loaded = np.fromfile(raw, dtype=">u2").reshape(nz, h, w).transpose(1, 2, 0)
    if raw.stat().st_size != cube.size * 2 or not np.array_equal(loaded, cube):
        raise RuntimeError("Conversion verification failed.")
    metadata = {
        "source_filename": args.source.name,
        "source_variable": "products/dn",
        "source_shape_y_x_z": source_shape,
        "source_dataset_attributes": attrs,
        "crop_row_start": y, "crop_column_start": x,
        "crop_shape_y_x_z": list(cube.shape),
        "crop_shape_z_y_x_on_disk": list(bsq.shape),
        "storage": "BSQ, unsigned 16-bit, big-endian",
        "sample_min": int(cube.min()), "sample_max": int(cube.max()),
        "raw_bytes": raw.stat().st_size,
        "raw_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
        "sample_values_preserved_exactly": True,
        "wavelength_values_as_stored": wavelength_values,
        "band_order": "original order, all spectral bands retained",
        "note": "Small crop for setup and exploratory plots, not representative validation."
    }
    meta.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Source: {source_shape} (lines, samples, bands)")
    print(f"Crop: {cube.shape}; rows [{y}, {y+h}), columns [{x}, {x+w})")
    print(f"Output: {raw}")
    print(f"Bytes: {raw.stat().st_size}; values: {cube.min()}..{cube.max()}")
    print("Verified that all stored sample values survived conversion exactly.")


if __name__ == "__main__":
    main()
