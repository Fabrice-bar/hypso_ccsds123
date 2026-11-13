"""Noise generation and representative tables for the Figure 4-13-style study."""
from __future__ import annotations
import hashlib
from types import SimpleNamespace
import numpy as np
import _damping_core as core

FIELDS=["noise_level","sigma_dn","sigma_over_a","noise_rep","noise_seed","theta",
        "phi_normalized","psi_normalized","first_band","input_sha256",
        "added_noise_mean_dn","added_noise_std_dn","added_noise_rmse_dn",
        "clipped_samples","clipped_fraction"]


def array_hash(cube):
    return hashlib.sha256(cube.astype("<i8",copy=False).tobytes()).hexdigest()


def noise_seed(source_sha,rows,cols,row,col,seed,rep):
    """Exclude A, sigma, phi and psi: use paired common random draws."""
    key=f"{source_sha}|{rows}|{cols}|{row}|{col}|{seed}|{rep}"
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],"little")


def add_noise(cube,sigma,seed):
    """Add N(0,sigma^2), round to nearest integer, clip to unsigned D=16."""
    if sigma==0:
        result=cube
        mean=std=rmse=0.0;clipped=0
    else:
        generator=np.random.Generator(np.random.PCG64(seed))
        values=generator.standard_normal(cube.shape)
        values*=sigma
        values+=cube
        np.rint(values,out=values)
        clipped=int(np.count_nonzero((values<0)|(values>65535)))
        np.clip(values,0,65535,out=values)
        result=values.astype(np.int64)
        delta=result-cube
        mean=float(np.mean(delta));std=float(np.std(delta,dtype=np.float64))
        rmse=float(np.sqrt(np.mean(delta.astype(np.float64)**2)))
    return result,dict(input_sha256=array_hash(result),added_noise_mean_dn=mean,
                       added_noise_std_dn=std,added_noise_rmse_dn=rmse,
                       clipped_samples=clipped,clipped_fraction=clipped/cube.size)


def make_header(cube,args,hd):
    # Construct the previous fixed predictor/encoder profile with legal zeros,
    # then install the requested resolution and tables before encoding.
    base=SimpleNamespace(**vars(args));base.phi_values=[0];base.psi=0
    h=core.make_header(cube,base,args.a,args.nu_max,hd)
    h.sample_representative_resolution=args.theta
    h.damping_table_array=np.full(h.z_size,args.phi,dtype=np.int64)
    h.damping_offset_table_array=np.full(h.z_size,args.psi,dtype=np.int64)
    if args.first_band=="zero":
        h.damping_table_array[0]=0;h.damping_offset_table_array[0]=0
        # Embed both tables for every grid point, giving uniform header overhead.
        h.band_varying_damping_flag=hd.BandVaryingDampingFlag.BAND_DEPENDENT
        h.damping_table_flag=hd.DampingTableFlag.INCLUDED
        h.fixed_damping_value=0
        h.band_varying_offset_flag=hd.BandVaryingOffsetFlag.BAND_DEPENDENT
        h.damping_offset_table_flag=hd.OffsetTableFlag.INCLUDED
        h.fixed_offset_value=0
    else:
        h.fixed_damping_value=args.phi;h.fixed_offset_value=args.psi
    h.check_legal_config()
    return h
