"""Optional memory-saving storage for the NTNU CCSDS-123 hybrid encoder.

All coding methods are inherited unchanged. Only eight full-cube Unicode
debugging arrays are replaced by discard-only objects. Numeric codec state,
the binary bitstream, initialization and termination are retained.

The upstream class is checked before use because this adapter relies on those
eight attributes being diagnostic only in the validated reference version.
"""
from __future__ import annotations

import ast
from functools import lru_cache
import hashlib
import inspect
from pathlib import Path
import textwrap

import numpy as np
from bitarray import bitarray
from ccsds123_i2_hlm.hybrid_encoder import HybridEncoder

REFERENCE_CLASS_SHA256 = "eccc91082666bc1ffc9a229bd24751d85deff319b327dd87402452398aeabed0"
DISCARDED_TEXT_BYTES_PER_SAMPLE = 1200


@lru_cache(maxsize=1)
def validate_reference():
    """Allow whitespace/comments to differ, but refuse changed reference code."""
    source = textwrap.dedent(inspect.getsource(HybridEncoder))
    node = ast.parse(source).body[0]
    # Python 3.12 adds empty type_params fields to classes/functions. Omit
    # those empty fields so the same Python source also validates on 3.11.
    for item in ast.walk(node):
        if "type_params" in item._fields and not getattr(item, "type_params", None):
            item._fields = tuple(name for name in item._fields if name != "type_params")
    actual = hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
    if actual != REFERENCE_CLASS_SHA256:
        raise ValueError(
            "--low-memory requires the validated NTNU HybridEncoder implementation "
            "(upstream commit 9fabb0371ebf93d72a0f83ca053e9cdb2c63e3c6). "
            "Your encoder code differs; validate this adapter against that version "
            "before using it. The source check must not simply be disabled."
        )
    return actual


def adapter_sha256():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def estimated_job_gib(shape, p_max):
    """Conservative predictor/check/encoder allowance; not an enforced limit.

    Predictor retains 14 int64 cubes and two int64 vectors of up to P+3
    components, plus reconstruction/error temporaries. Predictor is released
    before encoding. A base 320 bytes/sample includes extra working allowance;
    the surviving hybrid arrays use about 33 bytes/sample, excluding input.
    The coordinator separately budgets its input arrays and fixed overhead.
    """
    count = int(np.prod(shape, dtype=np.int64))
    return count * (320 + 16 * (p_max + 3)) / 2**30 + 0.5


class _DiscardedTextTrace:
    """Accept scalar diagnostic writes without retaining a sample history.

    The reference reads these fields only for append-to-trace operations and
    one assertion on an initially empty input-symbol trace. They never feed
    coding decisions. Access to actual stored traces (e.g. reshape) is absent.
    """
    __slots__ = ("initial",)

    def __init__(self, initial=""):
        self.initial = initial

    def __getitem__(self, key):
        return self.initial

    def __setitem__(self, key, value):
        pass


class CompactHybridEncoder(HybridEncoder):
    def __init__(self, header, image_constants, mapped_quantizer_index):
        validate_reference()
        super().__init__(header, image_constants, mapped_quantizer_index)

    # Explicit parent-mangled name: run_encoder() calls this initializer.
    def _HybridEncoder__init_encoder_arrays(self):
        image_shape = self.mapped_quantizer_index.shape
        self.accumulator = np.zeros(image_shape, dtype=np.int64)
        self.counter = np.zeros(image_shape[:2], dtype=np.int64)
        self.variable_length_code = np.full(image_shape, fill_value=-1, dtype=np.int64)
        self.active_prefix = [""] * 16
        self.code_index = np.full(image_shape, fill_value=-1, dtype=np.int64)
        self.input_symbol = _DiscardedTextTrace()
        self.low_entropy_codes = _DiscardedTextTrace()
        self.high_entropy_codes = _DiscardedTextTrace()
        self.rescale_bits = _DiscardedTextTrace()
        self.flush_codes = np.full((16), fill_value="", dtype="U10")
        self.accumulator_final = np.full((self.header.z_size), fill_value="", dtype="U46")
        self.codewords = _DiscardedTextTrace()
        self.codewords_binary = _DiscardedTextTrace()
        self.entropy_type = np.full(image_shape, fill_value=2, dtype=np.uint8)
        self.current_active_prefix = _DiscardedTextTrace("-")
        self.prefix_match_index = np.full(image_shape, fill_value=-2, dtype=np.int64)

        self.counter[0, 0] = 2**self.initial_count_exponent
        if self.use_accu_init_file:
            with open(self.accu_init_file, "rb") as file:
                data = file.read()
                data = "".join([bin(byte)[2:].zfill(8) for byte in data])
                num_bits = self.image_constants.dynamic_range_bits + self.initial_count_exponent
                data = data[: num_bits * self.header.z_size]
                data = [int(data[i : i + num_bits], 2) for i in range(0, len(data), num_bits)]
                assert len(data) == self.header.z_size
                self.accumulator[0, 0] = np.array(data)
        else:
            self.accumulator[0, 0] = 4 * self.counter[0, 0]

        self.bitstream = bitarray()
        self.bitstream_readable = _DiscardedTextTrace()

    def save_data(self, output_folder, header_bitstream):
        raise NotImplementedError(
            "CompactHybridEncoder discards text traces and cannot use the reference "
            "save_data() trace exporter. Use the sweep's CSV/plot metrics, or write "
            "header_bitstream + bitstream with the required output-word padding."
        )
