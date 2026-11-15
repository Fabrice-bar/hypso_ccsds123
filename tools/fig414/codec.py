"""Verified-model adapter; predictor and entropy coding equations are unchanged."""
from __future__ import annotations
import gc
import hashlib
import math
import os
from pathlib import Path
import sys
import time
import numpy as np


def setup(repo):
    repo = Path(repo).resolve()
    sys.path.insert(0, str(repo))
    os.chdir(repo)  # Upstream locates hybrid code tables relative to the CWD.
    from ccsds123_i2_hlm import header, constants, predictor, hybrid_encoder
    return header, constants, predictor, hybrid_encoder


def make_header(shape, settings, a, p, phi, psi, hd):
    y, x, z = map(int, shape)
    h = hd.Header(f"cube-u16be-{z}x{y}x{x}.raw")
    h.sample_type = hd.SampleType.UNSIGNED_INTEGER
    h.large_d_flag = hd.LargeDFlag.SMALL_D
    h.dynamic_range = 0  # D=16.
    h.register_size = 0  # R=64.
    h.weight_component_resolution = 15  # Omega=19.
    h.entropy_coder_type = hd.EntropyCoderType.HYBRID
    h.set_encoding_order_bil()
    h.output_word_size = 0  # Eight-byte words.
    h.prediction_bands_num = p
    h.prediction_mode = hd.PredictionMode.REDUCED
    h.local_sum_type = hd.LocalSumType.NARROW_NEIGHBOR_ORIENTED
    h.weight_init_method = hd.WeightInitMethod.DEFAULT
    h.weight_init_table_flag = hd.WeightInitTableFlag.NOT_INCLUDED
    h.weight_init_resolution = 0
    h.weight_exponent_offset_flag = hd.WeightExponentOffsetFlag.ALL_ZERO
    h.weight_exponent_offset_table_flag = hd.WeightExponentOffsetTableFlag.NOT_INCLUDED
    h.weight_update_initial_parameter = settings["nu_min"] + 6
    h.weight_update_final_parameter = settings["nu_max"] + 6
    h.weight_update_change_interval = settings["t_inc"].bit_length() - 5
    h.periodic_error_updating_flag = hd.PeriodicErrorUpdatingFlag.NOT_USED
    h.error_update_period_exponent = 0
    h.quantizer_fidelity_control_method = hd.QuantizerFidelityControlMethod.ABSOLUTE_ONLY
    h.absolute_error_limit_assignment_method = hd.ErrorLimitAssignmentMethod.BAND_INDEPENDENT
    h.absolute_error_limit_bit_depth = 5
    h.absolute_error_limit_value = a
    h.absolute_error_limit_table = np.full(z, a, dtype=np.int64)
    h.relative_error_limit_assignment_method = hd.ErrorLimitAssignmentMethod.BAND_INDEPENDENT
    h.relative_error_limit_value = 0
    h.relative_error_limit_table = np.zeros(z, dtype=np.int64)
    h.sample_representative_flag = hd.SampleRepresentativeFlag.INCLUDED
    h.sample_representative_resolution = settings["theta"]
    for name, values in (("phi", phi), ("psi", psi)):
        if len(values) != z or any(not 0 <= v < 2**settings["theta"] for v in values):
            raise ValueError(f"Invalid {name} profile.")
    varying_phi, varying_psi = len(set(phi)) > 1, len(set(psi)) > 1
    h.band_varying_damping_flag = (hd.BandVaryingDampingFlag.BAND_DEPENDENT if varying_phi
                                  else hd.BandVaryingDampingFlag.BAND_INDEPENDENT)
    h.damping_table_flag = hd.DampingTableFlag.INCLUDED if varying_phi else hd.DampingTableFlag.NOT_INCLUDED
    h.fixed_damping_value = 0 if varying_phi else phi[0]
    h.damping_table_array = np.array(phi, dtype=np.int64)
    h.band_varying_offset_flag = (hd.BandVaryingOffsetFlag.BAND_DEPENDENT if varying_psi
                                 else hd.BandVaryingOffsetFlag.BAND_INDEPENDENT)
    h.damping_offset_table_flag = hd.OffsetTableFlag.INCLUDED if varying_psi else hd.OffsetTableFlag.NOT_INCLUDED
    h.fixed_offset_value = 0 if varying_psi else psi[0]
    h.damping_offset_table_array = np.array(psi, dtype=np.int64)
    h.unary_length_limit = 18
    h.rescaling_counter_size = 2  # gamma*=6.
    h.initial_count_exponent = 1
    h.check_legal_config()
    return h


def encode(cube, settings, a, p, phi, psi, model, compact=True, return_stream=False):
    hd, constants, predictor, hybrid_encoder = model
    h = make_header(cube.shape, settings, a, p, phi, psi, hd)
    if cube.dtype != np.int64 or cube.min() < 0 or cube.max() > 65535:
        raise ValueError("Expected unsigned 16-bit DN in an int64 working array.")
    start = time.perf_counter()
    image_constants = constants.ImageConstants(h)
    pred = predictor.Predictor(h, image_constants, cube)
    pred.run_predictor()
    squared_sum, max_error, entropies = 0.0, 0, []
    mapped = pred.get_predictor_output()
    for z in range(cube.shape[2]):
        recon = pred.clipped_quantizer_bin_center[:, :, z].copy()
        # Upstream leaves this entry unset; the first sample is unquantized.
        recon[0, 0] = pred.predicted_sample_value[0, 0, z] + pred.quantizer_index[0, 0, z]
        err = cube[:, :, z] - recon
        max_error = max(max_error, int(np.max(np.abs(err))))
        squared_sum += float(np.sum(err.astype(np.float64) ** 2))
        counts = np.bincount(mapped[:, :, z].ravel())
        probabilities = counts[counts > 0].astype(np.float64) / counts.sum()
        entropies.append(float(-np.sum(probabilities * np.log2(probabilities))))
    if max_error > a or np.any(pred.maximum_error != a):
        raise RuntimeError(f"Encoder-side error check failed: A={a}, max error={max_error}.")
    nxy = cube.shape[0] * cube.shape[1]
    effective_nu = int(pred.weight_update_scaling_exponent[nxy - 2]) + 19 - 16
    final_update = (1 if settings["nu_min"] == settings["nu_max"] else
                    cube.shape[1] + (settings["nu_max"] - settings["nu_min"]) * settings["t_inc"] + 1)
    del pred, recon, err, counts, probabilities
    gc.collect()
    if compact:
        from hybrid_no_traces import CompactHybridEncoder
        cls = CompactHybridEncoder
    else:
        cls = hybrid_encoder.HybridEncoder
    enc = cls(h, image_constants, mapped)
    enc.use_accu_init_file = False  # Same external initial value: 4*2^gamma0=8.
    enc.run_encoder()
    header_bits, external_tables = h.get_header_bitstreams()
    if len(external_tables):
        raise RuntimeError("All profile tables must be included in the header.")
    padding = (-(len(header_bits) + len(enc.bitstream))) % 64
    stream_bits = len(header_bits) + len(enc.bitstream) + padding
    digest = hashlib.sha256(header_bits.tobytes() + enc.bitstream.tobytes()
                            + str(len(enc.bitstream)).encode()).hexdigest()
    result = {
        "header_bits": len(header_bits), "payload_bits": len(enc.bitstream),
        "padding_bits": padding, "stream_bits": stream_bits,
        "samples": int(cube.size), "bits_per_sample": stream_bits / cube.size,
        "max_error_dn": max_error, "rmse_dn": math.sqrt(squared_sum / cube.size),
        "residual_entropy_per_band": entropies,
        "effective_final_nu": effective_nu,
        "final_nu_reached": effective_nu == settings["nu_max"],
        "first_final_nu_update_index": final_update,
        "seconds": time.perf_counter() - start, "encoding_sha256": digest,
        "external_initialization_bits_if_sent_as_array": math.ceil(cube.shape[2] * 17 / 8) * 8,
    }
    if return_stream:
        result["_header"], result["_payload"] = header_bits, enc.bitstream
    return result
