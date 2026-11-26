"""
Compiled (Numba) re-implementation of the NTNU CCSDS 123.0-B-2 verification
model's predictor and hybrid encoder, for fast parameter sweeps.

It follows the model's code step by step (predictor.py, hybrid_encoder.py) and
takes every constant and table from the model itself, so the output bitstream
is meant to be bit-identical to the model's. p_sweep.py --verify checks that
against the unmodified model before a sweep starts.

Supported (everything p_sweep.py uses): hybrid entropy coder, BI encoding order
(any sub-frame interleaving depth; BIL is depth 1), full/reduced prediction,
all four local sums, default weight initialisation, lossless / absolute /
relative / absolute+relative error limits without periodic updating,
band-independent or band-dependent damping and offset tables. Anything else
raises NotImplementedError so it can never be silently wrong.

Without Numba installed the same functions run as plain Python (very slow,
only useful for testing).
"""

import numpy as np

VERSION = 2  # 2: compress() also returns distortion statistics

try:
    from numba import njit

    HAVE_NUMBA = True
except ImportError:  # plain-Python fallback, same code
    HAVE_NUMBA = False

    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        return lambda f: f


# Indices into the int64 parameter vector passed to the kernel
(I_Y, I_X, I_Z, I_D, I_SMIN, I_SMAX, I_SMID, I_P, I_C, I_FULL, I_LS, I_FID, I_OMEGA, I_R, I_WMIN, I_WMAX,
 I_THETA, I_UMAX, I_GSTAR, I_G0, I_M, N_PRM) = range(22)
LS_CODE = {"WIDE_NEIGHBOR_ORIENTED": 0, "NARROW_NEIGHBOR_ORIENTED": 1, "WIDE_COLUMN_ORIENTED": 2, "NARROW_COLUMN_ORIENTED": 3}
FID_CODE = {"LOSSLESS": 0, "ABSOLUTE_ONLY": 1, "RELATIVE_ONLY": 2, "ABSOLUTE_AND_RELATIVE": 3}
SYMBOLS = "0123456789ABC"  # low-entropy input symbols; 'X' (escape) is index 13
N_SYM = 14


# ----------------------------------------------------------------------------
# Low-entropy code tables -> tries (built once per process from the model's tables)
# ----------------------------------------------------------------------------

def _parse_word(w):
    n, v = w.strip().split("'h")
    return int(v, 16), int(n)


def build_tables(tables_module):
    """Turn the model's 16 code and flush tables into trie arrays.

    The model appends one input symbol to the active prefix of the chosen table
    and emits a codeword when the prefix equals a table entry exactly. Because
    each table is prefix-free (checked below), walking a trie one symbol at a
    time and emitting at a leaf is the same thing.
    """
    tm = tables_module
    tm.tables_init()
    tries = []
    for i in range(16):
        nodes = [{}]  # node -> {symbol: child}
        leaf = {}
        for inp, out in zip(tm.code_table_input[i], tm.code_table_output[i]):
            if inp == "Z":
                continue
            n = 0
            for ch in inp:
                s = 13 if ch == "X" else SYMBOLS.index(ch)
                if s not in nodes[n]:
                    nodes.append({})
                    nodes[n][s] = len(nodes) - 1
                n = nodes[n][s]
            if n in leaf:
                raise NotImplementedError(f"code table {i}: duplicate entry {inp}")
            leaf[n] = _parse_word(out)
        for n in leaf:
            if nodes[n]:
                raise NotImplementedError(f"code table {i} is not prefix-free")
        flush = {}
        for k, (pre, word) in enumerate(zip(tm.flush_table_prefix[i], tm.flush_table_word[i])):
            if pre == "Z":
                continue
            if k == 0:
                flush[0] = _parse_word(word)  # model: empty active prefix -> entry 0
                continue
            n = 0
            for ch in pre:
                s = 13 if ch == "X" else (SYMBOLS.index(ch) if ch in SYMBOLS else -1)
                n = nodes[n].get(s, -1) if n >= 0 and s >= 0 else -1
            if n > 0 and n not in flush:
                flush[n] = _parse_word(word)  # model takes the first matching entry
        tries.append((nodes, leaf, flush))

    nn = max(len(t[0]) for t in tries)
    child = np.full((16, nn, N_SYM), -1, dtype=np.int64)
    wval = np.zeros((16, nn), dtype=np.int64)
    wlen = np.full((16, nn), -1, dtype=np.int64)
    fval = np.zeros((16, nn), dtype=np.int64)
    flen = np.full((16, nn), -1, dtype=np.int64)
    for i, (nodes, leaf, flush) in enumerate(tries):
        for n, kids in enumerate(nodes):
            for s, c in kids.items():
                child[i, n, s] = c
        for n, (v, ln) in leaf.items():
            if ln > 62 or v >= 2**62:
                raise NotImplementedError("codeword longer than 62 bits")
            wval[i, n], wlen[i, n] = v, ln
        for n, (v, ln) in flush.items():
            fval[i, n], flen[i, n] = v, ln
    limit = np.asarray(tm.input_symbol_limit, dtype=np.int64)
    thr = np.asarray(tm.threshold, dtype=np.int64)
    return child, wval, wlen, fval, flen, limit, thr


# ----------------------------------------------------------------------------
# Kernel
# ----------------------------------------------------------------------------

@njit(cache=True)
def _bitlen(v):
    n = 0
    while v > 0:
        v >>= 1
        n += 1
    return n


@njit(cache=True)
def _put(buf, pos, val, n):
    """Write the n low bits of val, most significant first (buffer starts zeroed)."""
    for b in range(n - 1, -1, -1):
        if (val >> b) & 1:
            buf[pos >> 3] |= np.uint8(128 >> (pos & 7))
        pos += 1
    return pos


@njit(cache=True)
def _put_binary_zfill(buf, pos, val, width):
    """bin(val)[2:].zfill(width): at least `width` bits, more if val needs them."""
    n = _bitlen(val)
    if n < width:
        n = width
    return _put(buf, pos, val, n)


@njit(cache=True)
def _reverse_gpo2(buf, pos, j, k, D, umax):
    """HybridEncoder.__reverse_gpo2 followed by __add_to_bitstream."""
    zeros = j >> k
    if zeros < umax:
        if k > 0:
            pos = _put(buf, pos, j & ((1 << k) - 1), k)
        pos = _put(buf, pos, 1, 1)
        return pos + zeros  # zeros are already in the buffer
    pos = _put_binary_zfill(buf, pos, j, D)
    return pos + umax


@njit(cache=True)
def _clip(v, lo, hi):
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


@njit(cache=True)
def _sign(v):
    if v > 0:
        return 1
    if v < 0:
        return -1
    return 0


@njit(cache=True)
def _grow(buf, pos, need_bits):
    if pos + need_bits <= buf.shape[0] * 8:
        return buf
    nb = np.zeros(max(buf.shape[0] * 2, (pos + need_bits) // 8 + 1), dtype=np.uint8)
    nb[: buf.shape[0]] = buf
    return nb


@njit(cache=True)
def compress_kernel(img, prm, pstar, rho, wexp, alim, rlim, phi, psi,
                    child, wval, wlen, fval, flen, limit, thr):
    """Predictor (BIP order) and hybrid encoder (BI order) fused line by line.

    Returns (packed bitstream bytes, number of bits, error code, sum of squared
    reconstruction errors, max absolute error, sum of squared samples). Error code 0 = ok.
    """
    Y, X, Z, D = prm[I_Y], prm[I_X], prm[I_Z], prm[I_D]
    smin, smax, smid = prm[I_SMIN], prm[I_SMAX], prm[I_SMID]
    P, C, full, lstype, fid = prm[I_P], prm[I_C], prm[I_FULL], prm[I_LS], prm[I_FID]
    om, R, wmin, wmax, theta = prm[I_OMEGA], prm[I_R], prm[I_WMIN], prm[I_WMAX], prm[I_THETA]
    umax, gstar, g0, M = prm[I_UMAX], prm[I_GSTAR], prm[I_G0], prm[I_M]
    off = 3 if full == 1 else 0
    dyn = np.int64(1) << D
    two_om = np.int64(1) << om
    half_r = np.int64(1) << (R - 1) if R < 64 else np.int64(0)
    hr_lo = (np.int64(1) << (om + 2)) * smin
    hr_hi = (np.int64(1) << (om + 2)) * smax + (np.int64(1) << (om + 1))
    hr_add = (np.int64(1) << (om + 2)) * smid + (np.int64(1) << (om + 1))

    # predictor state: only what later samples read back
    sr = np.zeros((2, X, Z), dtype=np.int64)  # sample representatives, rows y % 2
    W = np.zeros((Z, C), dtype=np.int64)  # weights of the previous t, per band
    U = np.zeros((Z, C), dtype=np.int64)  # local differences, per band
    err = np.zeros(Z, dtype=np.int64)  # double-resolution prediction error of the previous t
    ls_cur = np.zeros(Z, dtype=np.int64)  # local sums of the current t (read by band z+1)
    mqi = np.zeros((X, Z), dtype=np.int64)  # predictor output for the current line

    # encoder state
    acc = np.empty(Z, dtype=np.int64)
    acc[:] = 4 * (np.int64(1) << g0)  # model default: 4 * 2^gamma_0
    resc = (np.int64(1) << gstar) - 1
    cnt = np.empty(Y * X, dtype=np.int64)  # counter Gamma(t): the same for every band
    cnt[0] = np.int64(1) << g0
    for t in range(1, Y * X):
        cnt[t] = (cnt[t - 1] + 1) // 2 if cnt[t - 1] == resc else cnt[t - 1] + 1
    kmax = max(D - 2, 2)
    state = np.zeros(16, dtype=np.int64)  # active prefix of each low-entropy code (trie node)
    buf = np.zeros(max(Y * X * Z // 2, 1024), dtype=np.uint8)
    pos = np.int64(0)
    n_i = (Z + M - 1) // M
    sse = 0.0  # distortion of the decoder's reconstruction (clipped quantizer bin centre)
    ss2 = 0.0
    maxe = np.int64(0)

    for y in range(Y):
        r = y & 1
        rp = r ^ 1
        # ------------------------------ predictor, line y (BIP) ------------------------------
        for x in range(X):
            t = x + y * X
            for z in range(Z):
                s = np.int64(img[y, x, z])

                # local sum (Predictor.__calculate_local_sum)
                lsum = np.int64(-1)
                if t > 0:
                    if lstype == 0:  # wide neighbor
                        if y > 0 and 0 < x and x < X - 1:
                            lsum = sr[r, x - 1, z] + sr[rp, x - 1, z] + sr[rp, x, z] + sr[rp, x + 1, z]
                        elif y == 0 and x > 0:
                            lsum = sr[r, x - 1, z] * 4
                        elif y > 0 and x == 0:
                            lsum = (sr[rp, x, z] + sr[rp, x + 1, z]) * 2
                        elif y > 0 and x == X - 1:
                            lsum = sr[r, x - 1, z] + sr[rp, x - 1, z] + sr[rp, x, z] * 2
                    elif lstype == 1:  # narrow neighbor
                        if y > 0 and 0 < x and x < X - 1:
                            lsum = sr[rp, x - 1, z] + sr[rp, x, z] * 2 + sr[rp, x + 1, z]
                        elif y == 0 and x > 0 and z > 0:
                            lsum = sr[r, x - 1, z - 1] * 4
                        elif y > 0 and x == 0:
                            lsum = (sr[rp, x, z] + sr[rp, x + 1, z]) * 2
                        elif y > 0 and x == X - 1:
                            lsum = (sr[rp, x - 1, z] + sr[rp, x, z]) * 2
                        elif y == 0 and x > 0 and z == 0:
                            lsum = smid * 4
                    elif lstype == 2:  # wide column
                        if y > 0:
                            lsum = sr[rp, x, z] * 4
                        elif y == 0 and x > 0:
                            lsum = sr[r, x - 1, z] * 4
                    else:  # narrow column
                        if y > 0:
                            lsum = sr[rp, x, z] * 4
                        elif y == 0 and x > 0 and z > 0:
                            lsum = sr[r, x - 1, z - 1] * 4
                        elif y == 0 and x > 0 and z == 0:
                            lsum = smid * 4
                    ls_cur[z] = lsum

                # weights (Predictor.__calculate_weight_vector); uses U and err of t-1,
                # so it runs before U is overwritten with this t's local differences
                if t == 1:
                    for i in range(C):
                        W[z, i] = 0
                    if z > 0 and pstar[z] > 0:
                        W[z, off] = two_om * 7 // 8
                        for i in range(1, pstar[z]):
                            W[z, off + i] = W[z, off + i - 1] // 8
                elif t >= 2:
                    sp = 1 if err[z] >= 0 else -1
                    for i in range(C):
                        # floor(sp * u * 2^-(rho + offset)) done exactly with shifts
                        e = rho[t - 1] + wexp[z, i]
                        v = sp * U[z, i]
                        if e >= 0:
                            f = v >> e
                        else:
                            f = v << (-e)
                        W[z, i] = _clip(W[z, i] + (f + 1) // 2, wmin, wmax)

                # local differences (Predictor.__calculate_local_difference_vector)
                if t > 0:
                    if full == 1:
                        if x > 0 and y > 0:
                            U[z, 0] = 4 * sr[rp, x, z] - lsum
                            U[z, 1] = 4 * sr[r, x - 1, z] - lsum
                            U[z, 2] = 4 * sr[rp, x - 1, z] - lsum
                        elif x == 0 and y > 0:
                            U[z, 0] = 4 * sr[rp, x, z] - lsum
                            U[z, 1] = 4 * sr[rp, x, z] - lsum
                            U[z, 2] = 4 * sr[rp, x, z] - lsum
                        else:
                            U[z, 0] = 0
                            U[z, 1] = 0
                            U[z, 2] = 0
                    if z > 0 and pstar[z] > 0:
                        U[z, off] = 4 * sr[r, x, z - 1] - ls_cur[z - 1]
                        for i in range(1, pstar[z]):
                            U[z, off + i] = U[z - 1, off + i - 1]

                # prediction
                hr = np.int64(-1)
                if t > 0:
                    if full == 0 and z == 0:
                        d = np.int64(0)
                    else:
                        d = np.int64(0)
                        for i in range(C):
                            d += W[z, i] * U[z, i]
                    v = d + two_om * (lsum - 4 * smid)
                    if R < 64 and (v < -half_r or v >= half_r):
                        v = ((v + half_r) % (2 * half_r)) - half_r
                    hr = _clip(v + hr_add, hr_lo, hr_hi)
                    drp = hr >> (om + 1)
                elif t == 0 and P > 0 and z > 0:
                    drp = 2 * np.int64(img[y, x, z - 1])
                else:
                    drp = 2 * smid
                pred = drp >> 1

                # maximum error
                if fid == 0:
                    m = np.int64(0)
                elif fid == 1:
                    m = alim[y, z]
                elif fid == 2:
                    m = rlim[y, z] * pred // dyn
                else:
                    m = min(alim[y, z], rlim[y, z] * pred // dyn)

                # quantization
                res = s - pred
                if t == 0:
                    q = res
                else:
                    q = _sign(res) * ((abs(res) + m) // (2 * m + 1))

                # sample representative
                if t == 0:
                    cq = np.int64(-1)  # never written at t = 0 in the model
                    srv = s
                else:
                    if m == 0:
                        cq = s
                    else:
                        cq = _clip(pred + q * (2 * m + 1), smin, smax)
                    if phi[z] == 0 and psi[z] == 0:
                        srv = cq
                    else:
                        dsr = (4 * ((np.int64(1) << theta) - phi[z]) * (cq * two_om - _sign(q) * m * psi[z] * (np.int64(1) << (om - theta)))
                               + phi[z] * hr - phi[z] * (np.int64(1) << (om + 1))) >> (om + theta + 1)
                        srv = (dsr + 1) // 2
                sr[r, x, z] = srv

                # reconstruction error: the decoder outputs the clipped bin centre (t > 0)
                # and the exact sample at t = 0 (q is the unquantized residual there)
                if t > 0:
                    de = s - cq
                    if de < 0:
                        de = -de
                    if de > maxe:
                        maxe = de
                    sse += float(de) * float(de)
                ss2 += float(s) * float(s)

                # prediction error and mapped quantizer index
                err[z] = 2 * cq - drp
                if t == 0:
                    th = min(pred - smin, smax - pred)
                else:
                    th = min((pred - smin + m) // (2 * m + 1), (smax - pred + m) // (2 * m + 1))
                term = q if drp % 2 == 0 else -q
                aq = abs(q)
                if aq > th:
                    mqi[x, z] = aq + th
                elif 0 <= term and term <= th:
                    mqi[x, z] = 2 * aq
                else:
                    mqi[x, z] = 2 * aq - 1

        # ------------------------------ encoder, line y (BI order) ------------------------------
        buf = _grow(buf, pos, X * Z * (2 * (D + umax) + 70))
        for ii in range(n_i):
            z0 = ii * M
            z1 = min((ii + 1) * M, Z)
            for x in range(X):
                t = x + y * X
                for z in range(z0, z1):
                    j = mqi[x, z]
                    if t == 0:  # first pixel: raw D-bit value
                        pos = _put_binary_zfill(buf, pos, j, D)
                        continue
                    c_prev = cnt[t - 1]
                    c = cnt[t]
                    if c_prev == resc:  # rescaling: emit LSB of the old accumulator
                        pos = _put(buf, pos, acc[z] & 1, 1)
                        acc[z] = (acc[z] + 4 * j + 1) // 2
                    else:
                        acc[z] = acc[z] + 4 * j
                    a = acc[z]
                    if a * 16384 >= thr[0] * c:  # high-entropy: reversed GPO2 code
                        v = (a + c * 49 // 32) // c
                        k = _bitlen(v) - 1 - 2  # floor(log2(v)) - 2
                        if k > kmax:
                            k = kmax
                        if k < 2:
                            return buf, pos, 1, sse, maxe, ss2  # model asserts k >= 2
                        pos = _reverse_gpo2(buf, pos, j, k, D, umax)
                    else:  # low-entropy: variable-to-variable code of the selected table
                        ci = -2
                        for i in range(15, -1, -1):
                            if a * 16384 < c * thr[i]:
                                ci = i
                                break
                        if j <= limit[ci]:
                            sym = j
                        else:
                            sym = 13  # 'X' escape symbol, followed by the residual
                            pos = _reverse_gpo2(buf, pos, j - limit[ci] - 1, 0, D, umax)
                        nxt = child[ci, state[ci], sym]
                        if nxt < 0:
                            return buf, pos, 2, sse, maxe, ss2  # prefix not in the code table
                        if wlen[ci, nxt] >= 0:
                            pos = _put_binary_zfill(buf, pos, wval[ci, nxt], wlen[ci, nxt])
                            state[ci] = 0
                        else:
                            state[ci] = nxt

    # ------------------------------ image tail ------------------------------
    buf = _grow(buf, pos, 16 * 64 + Z * 70 + 8)
    for i in range(16):
        if flen[i, state[i]] < 0:
            return buf, pos, 3, sse, maxe, ss2  # no flush word for this prefix
        pos = _put_binary_zfill(buf, pos, fval[i, state[i]], flen[i, state[i]])
    for z in range(Z):
        pos = _put_binary_zfill(buf, pos, acc[z], 2 + D + gstar)
    pos = _put(buf, pos, 1, 1)
    return buf, pos, 0, sse, maxe, ss2


# ----------------------------------------------------------------------------
# Python driver
# ----------------------------------------------------------------------------

ERRORS = {1: "high-entropy code parameter k < 2", 2: "low-entropy prefix not in code table",
          3: "no flush codeword for active prefix"}
_TABLES = {}


def compress(cube, header, consts, M):
    """Compress a (Y, X, Z) integer cube with the settings in a model Header.

    M is the dict of imported model modules from p_sweep (hd, pred, henc_tables).
    Returns (body_bits, packed_bytes, distortion stats).
    """
    hd = M["hd"]
    h = header
    if h.entropy_coder_type != hd.EntropyCoderType.HYBRID:
        raise NotImplementedError("only the hybrid entropy coder")
    if h.sample_encoding_order != hd.SampleEncodingOrder.BI:
        raise NotImplementedError("only BI (e.g. BIL) sample encoding order")
    if h.periodic_error_updating_flag != hd.PeriodicErrorUpdatingFlag.NOT_USED:
        raise NotImplementedError("periodic error limit updating")
    if h.weight_init_method != hd.WeightInitMethod.DEFAULT:
        raise NotImplementedError("custom weight initialisation")
    Y, X, Z = cube.shape
    if consts.lower_sample_limit < 0 and FID_CODE[hd.QuantizerFidelityControlMethod(h.quantizer_fidelity_control_method).name] >= 2:
        # The model computes m = r * s_hat // 2^D without |s_hat|, which goes negative for
        # signed data; the standard uses |s_hat|. Refuse rather than copy that.
        raise NotImplementedError("relative error limits with signed samples")
    if X < 2:
        raise NotImplementedError("images must be at least 2 pixels wide")

    # Predictor constants computed by the model's own code
    p = M["pred"].Predictor(h, consts, None)
    p._Predictor__init_predictor_constants()
    C = int(p.local_difference_values_num)
    wexp_f = np.asarray(p.weight_exponent_offset, dtype=np.float64).reshape(Z, C) if C > 0 else np.zeros((Z, 1))
    if not np.all(wexp_f == np.round(wexp_f)):
        raise NotImplementedError("non-integer weight exponent offsets")
    wexp = np.round(wexp_f).astype(np.int64)

    if "tables" not in _TABLES:
        _TABLES["tables"] = build_tables(M["henc_tables"])
    child, wval, wlen, fval, flen, limit, thr = _TABLES["tables"]

    prm = np.zeros(N_PRM, dtype=np.int64)
    prm[I_Y], prm[I_X], prm[I_Z], prm[I_D] = Y, X, Z, consts.dynamic_range_bits
    prm[I_SMIN], prm[I_SMAX], prm[I_SMID] = consts.lower_sample_limit, consts.upper_sample_limit, consts.middle_sample_value
    prm[I_P], prm[I_C] = h.prediction_bands_num, max(C, 1)
    prm[I_FULL] = int(h.prediction_mode == hd.PredictionMode.FULL)
    prm[I_LS] = LS_CODE[hd.LocalSumType(h.local_sum_type).name]
    prm[I_FID] = FID_CODE[hd.QuantizerFidelityControlMethod(h.quantizer_fidelity_control_method).name]
    prm[I_OMEGA], prm[I_R] = p.weight_component_resolution, p.register_size
    prm[I_WMIN], prm[I_WMAX] = p.weight_min, p.weight_max
    prm[I_THETA] = h.sample_representative_resolution
    prm[I_UMAX] = h.unary_length_limit + 32 * (h.unary_length_limit == 0)
    prm[I_GSTAR] = h.rescaling_counter_size + 4
    prm[I_G0] = h.initial_count_exponent + 8 * (h.initial_count_exponent == 0)
    prm[I_M] = h.sub_frame_interleaving_depth
    if prm[I_M] <= 0:
        raise NotImplementedError("sub-frame interleaving depth must be > 0")

    phi = np.asarray(h.damping_table_array, dtype=np.int64).reshape(-1)
    psi = np.asarray(h.damping_offset_table_array, dtype=np.int64).reshape(-1)
    if phi.size == 1:
        phi = np.full(Z, phi[0], dtype=np.int64)
    if psi.size == 1:
        psi = np.full(Z, psi[0], dtype=np.int64)
    img = cube if cube.dtype.kind in "iu" and cube.dtype.itemsize <= 4 or cube.dtype == np.int64 else cube.astype(np.int64)
    buf, nbits, code, sse, maxe, ss2 = compress_kernel(
        np.ascontiguousarray(img), prm, np.asarray(p.spectral_bands_used, dtype=np.int64),
        np.asarray(p.weight_update_scaling_exponent, dtype=np.int64), wexp,
        np.asarray(p.absolute_error_limits, dtype=np.int64), np.asarray(p.relative_error_limits, dtype=np.int64),
        phi, psi, child, wval, wlen, fval, flen, limit, thr)
    if code:
        raise RuntimeError(f"fast engine stopped: {ERRORS.get(code, code)}")
    stats = {"sse": float(sse), "max_abs_err": int(maxe), "sum_sq": float(ss2), "n": Y * X * Z}
    return int(nbits), buf[: (int(nbits) + 7) // 8], stats


def bits_to01(packed, nbits):
    return "".join(np.unpackbits(np.frombuffer(bytes(packed), dtype=np.uint8))[:nbits].astype(str))
