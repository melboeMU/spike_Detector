"""
spikedet.py -- Interictal epileptiform spike detection for iEEG.

Design goal: detect single-trial epileptiform transients without firing on
beta/gamma oscillatory bursts, and without a long analysis window (two spikes
200 ms apart are both reported).

Pipeline
--------
Stage 1 (touches every sample, vectorised over channels):
    multiscale Mexican-hat (Ricker) matched filter -> detection function D
    adaptive robust threshold (MAD over time blocks)
    local-maximum picking with a short refractory period

Stage 2 (runs only on windows around candidates, batched):
    Gate A  cycle count   -- reject rhythmic bursts (the important one)
    Gate B  broadband     -- require simultaneous power in >= N log-spaced bands
    Gate C  low-frequency -- require a coincident 2-8 Hz deflection
    Gate D  half-width    -- reject too-narrow / too-broad deflections

Stage 3: cross-channel coincidence rejection (artifact removal).

Author's note on efficiency: Stage 1 uses overlap-add FFT convolution in
float32 with a running maximum across scales, processed in channel blocks and
time blocks, so peak memory is bounded by
``channel_block * time_block_s * fs * 4 bytes * ~6``.
Stage 2 cost scales with the number of candidates, not with recording length.

Usage
-----
    import numpy as np
    from spikedet import SpikeParams, detect_spikes

    p = SpikeParams(fs=1000.0, notch_freqs=(60., 120., 180.))
    p.thresh_mad = 6.0
    ev = detect_spikes(data, p, ch_names=names)   # data: (n_channels, n_samples)

    # ev is a dict of equal-length numpy arrays; wrap in a DataFrame if you like:
    # import pandas as pd; df = pd.DataFrame(ev)

Dependencies: numpy, scipy.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional, Sequence, Tuple, Dict

import numpy as np
from scipy import signal

__all__ = ["SpikeParams", "detect_spikes", "detection_function"]


# ----------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------

@dataclass
class SpikeParams:
    """All tunable parameters. Only ``fs`` is required.

    Tuning order that usually works: thresh_mad -> max_cycles -> min_bands.
    Score on false positives per minute, not accuracy.
    """

    # --- acquisition -------------------------------------------------------
    fs: float                                   # Hz. >= 500 Hz strongly preferred.
    notch_freqs: Tuple[float, ...] = ()         # e.g. (60., 120., 180.) or (50., 100., 150.)
    notch_q: float = 30.0
    highpass: float = 1.0                       # Hz, applied before everything else

    # --- filter bands ------------------------------------------------------
    wide_band: Tuple[float, float] = (1.0, 80.0)    # morphology / detection function
    hf_band:   Tuple[float, float] = (20.0, 70.0)   # cycle-counting gate
    lf_band:   Tuple[float, float] = (2.0, 8.0)     # low-frequency involvement gate
    filter_order: int = 4                           # butterworth order (per sosfiltfilt)

    # --- stage 1: candidate detection --------------------------------------
    # Ricker sigma values in ms. A spike whose central lobe is ~2*sigma wide
    # responds maximally at that scale. 10-66 ms covers spikes -> sharp waves.
    scales_ms: Tuple[float, ...] = (10.0, 15.0, 22.0, 32.0, 46.0, 66.0)
    thresh_mad: float = 6.0                     # threshold = median + k * 1.4826 * MAD of D
    baseline_block_s: float = 60.0              # adaptive baseline block length
    max_amp_mad: float = 60.0                   # above this -> artifact, not spike (0 disables)
    refractory_ms: float = 40.0                 # dead time; caps rate at 1/refractory
    peak_refine_ms: float = 12.0                # search radius for the morphological peak

    # --- Gate A: cycle count (rejects oscillatory bursts) -------------------
    use_cycle_gate: bool = True
    cycle_win_ms: float = 150.0                 # half-window searched around the event
    cycle_rel_thresh: float = 0.5               # envelope width measured at this fraction of max
    max_cycles: float = 3.0                     # >= this many sustained cycles -> reject

    # --- Gate B: broadband coincidence -------------------------------------
    use_band_gate: bool = True
    band_edges: Tuple[Tuple[float, float], ...] = ((8., 16.), (16., 32.),
                                                   (32., 64.), (64., 100.))
    band_win_ms: float = 200.0                  # half-window for the spectrum
    band_z: float = 2.0                         # per-band z threshold vs. background
    min_bands: int = 3                          # how many bands must exceed band_z
    n_baseline_windows: int = 500               # random windows used for background spectra

    # --- Gate C: low-frequency involvement ---------------------------------
    use_lf_gate: bool = True
    lf_z: float = 2.0                           # |x_lf| at event, in robust z units
    lf_win_ms: float = 100.0                    # half-window searched for the LF peak

    # --- Gate D: half-width ------------------------------------------------
    use_width_gate: bool = True
    min_halfwidth_ms: float = 10.0
    max_halfwidth_ms: float = 80.0

    # --- Stage 3: cross-channel artifact rejection -------------------------
    use_coincidence_gate: bool = True
    coincidence_ms: float = 20.0
    max_coincident_frac: float = 0.5            # fraction of channels; 1.0 disables

    # --- performance -------------------------------------------------------
    channel_block: int = 16                     # channels held in memory at once
    time_block_s: float = 300.0                 # seconds held in memory at once (0 = all)
    overlap_s: float = 5.0                      # overlap between time blocks
    dtype: str = "float32"
    random_seed: int = 0

    def as_dict(self) -> dict:
        return asdict(self)


# ----------------------------------------------------------------------------
# Filter helpers
# ----------------------------------------------------------------------------

def _sos_band(band, fs, order):
    nyq = fs / 2.0
    lo, hi = band
    hi = min(hi, nyq * 0.98)
    if lo <= 0:
        return signal.butter(order, hi / nyq, btype="low", output="sos")
    if hi >= nyq * 0.98:
        return signal.butter(order, lo / nyq, btype="high", output="sos")
    return signal.butter(order, [lo / nyq, hi / nyq], btype="band", output="sos")


def _notch_sos(freqs, fs, q):
    nyq = fs / 2.0
    sections = []
    for f0 in freqs:
        if f0 <= 0 or f0 >= nyq * 0.98:
            continue
        b, a = signal.iirnotch(f0 / nyq, q)
        sections.append(signal.tf2sos(b, a))
    return np.vstack(sections) if sections else None


def _filt(sos, x):
    """Zero-phase filtering along the last axis, preserving dtype."""
    if sos is None:
        return x
    return signal.sosfiltfilt(sos, x, axis=-1).astype(x.dtype, copy=False)


def _ricker_kernels(scales_ms, fs, dtype=np.float32):
    """Unit-L2-norm Mexican-hat kernels, so responses are comparable across scales."""
    kernels = []
    for s_ms in scales_ms:
        sigma = max(s_ms * 1e-3 * fs, 1.0)
        half = int(np.ceil(4.0 * sigma))
        t = np.arange(-half, half + 1, dtype=np.float64)
        tt = (t / sigma) ** 2
        k = (1.0 - tt) * np.exp(-tt / 2.0)
        k -= k.mean()                       # exactly zero-mean -> no DC response
        k /= np.linalg.norm(k)
        kernels.append(k.astype(dtype))
    return kernels


# ----------------------------------------------------------------------------
# Stage 1
# ----------------------------------------------------------------------------

def detection_function(x_wide: np.ndarray, kernels) -> Tuple[np.ndarray, np.ndarray]:
    """Running max over scales of |x * ricker_s|.

    Returns (D, best_scale_index), both shape (n_ch, n_t).
    Memory: two arrays the size of x_wide, regardless of the number of scales.
    """
    D = None
    best = None
    for si, k in enumerate(kernels):
        c = np.abs(signal.oaconvolve(x_wide, k[None, :], mode="same", axes=-1))
        if D is None:
            D = c
            best = np.zeros(c.shape, dtype=np.uint8)
        else:
            newer = c > D
            D = np.where(newer, c, D)
            best[newer] = si
            del newer
        del c
    return D, best


def _robust_stats_blocks(D, block_len):
    """Per-channel, per-block median and MAD-based sigma, expanded back to full length."""
    n_ch, n_t = D.shape
    n_blk = max(1, int(np.ceil(n_t / block_len)))
    med = np.empty((n_ch, n_blk), dtype=np.float64)
    sig = np.empty((n_ch, n_blk), dtype=np.float64)
    for b in range(n_blk):
        seg = D[:, b * block_len:(b + 1) * block_len]
        m = np.median(seg, axis=1)
        med[:, b] = m
        sig[:, b] = 1.4826 * np.median(np.abs(seg - m[:, None]), axis=1)
    sig[sig <= 0] = np.finfo(np.float64).eps
    return med, sig, n_blk


def _pick_peaks(d, thr, refractory):
    """Local maxima above threshold, greedily thinned by amplitude with a refractory.

    d: 1-D detection function. Returns sorted sample indices.
    """
    above = d > thr
    if not above.any():
        return np.empty(0, dtype=np.int64)
    # strict local maxima
    loc = np.zeros_like(above)
    loc[1:-1] = (d[1:-1] >= d[:-2]) & (d[1:-1] > d[2:])
    cand = np.flatnonzero(above & loc)
    if cand.size == 0:
        return cand
    order = cand[np.argsort(d[cand])[::-1]]      # strongest first
    keep = []
    taken = np.zeros(0, dtype=np.int64)
    for i in order:
        if taken.size and np.min(np.abs(taken - i)) < refractory:
            continue
        keep.append(i)
        taken = np.append(taken, i)
    return np.sort(np.asarray(keep, dtype=np.int64))


# ----------------------------------------------------------------------------
# Window extraction (batched)
# ----------------------------------------------------------------------------

def _windows(x_row, idx, half):
    """(n_events, 2*half+1) window matrix, zero-padded at the edges."""
    n = x_row.size
    offs = np.arange(-half, half + 1)
    ii = idx[:, None] + offs[None, :]
    valid = (ii >= 0) & (ii < n)
    ii = np.clip(ii, 0, n - 1)
    w = x_row[ii]
    w[~valid] = 0.0
    return w


# ----------------------------------------------------------------------------
# Gates
# ----------------------------------------------------------------------------

def _gate_cycles(w_hf, fs, rel_thresh):
    """Effective number of oscillation cycles in the HF band, per event.

    Do NOT count peaks of the filtered trace: bandpass-filtering a sharp
    transient produces ringing that mimics an oscillation. Instead measure how
    *long* the HF envelope is sustained, in units of its own dominant period:

        n_cycles = (envelope width at `rel_thresh` of its max) * f_dominant

    Filter ringing from a spike gives a single narrow envelope blip
    (n_cycles ~ 1). A genuine beta/gamma burst sustains its envelope over
    several periods (n_cycles >= 3).
    """
    n = w_hf.shape[1]
    env = np.abs(signal.hilbert(w_hf.astype(np.float64), axis=1))

    # dominant HF frequency per window
    win = np.hanning(n)
    P = np.abs(np.fft.rfft(w_hf * win, axis=1)) ** 2
    f = np.fft.rfftfreq(n, d=1.0 / fs)
    f0 = f[P.argmax(axis=1)]
    f0[f0 <= 0] = np.finfo(np.float64).eps

    # envelope width at rel_thresh * max, around the envelope peak
    pk = env.argmax(axis=1)
    emax = env.max(axis=1)
    emax[emax <= 0] = np.finfo(np.float64).eps
    below = env < rel_thresh * emax[:, None]
    ar = np.arange(n)[None, :]

    rm = below & (ar > pk[:, None])
    first_r = np.where(rm.any(axis=1), rm.argmax(axis=1), n - 1)
    lm = (below & (ar < pk[:, None]))[:, ::-1]
    last_l = np.where(lm.any(axis=1), n - 1 - lm.argmax(axis=1), 0)

    width_s = np.maximum(first_r - last_l, 1) / fs
    return width_s * f0


def _band_logpower(w, fs, band_edges):
    """(n_windows, n_bands) log band power via one rfft per window."""
    n = w.shape[1]
    win = np.hanning(n).astype(w.dtype)
    W = np.fft.rfft(w * win, axis=1)
    P = (W.real ** 2 + W.imag ** 2)
    f = np.fft.rfftfreq(n, d=1.0 / fs)
    out = np.empty((w.shape[0], len(band_edges)), dtype=np.float64)
    for bi, (lo, hi) in enumerate(band_edges):
        m = (f >= lo) & (f < hi)
        out[:, bi] = P[:, m].sum(axis=1) if m.any() else 0.0
    return np.log10(out + 1e-20)


def _gate_bands(w_ev, w_bg, fs, band_edges, band_z):
    """Number of bands whose power at the event exceeds background by band_z."""
    lp_bg = _band_logpower(w_bg, fs, band_edges)
    med = np.median(lp_bg, axis=0)
    mad = 1.4826 * np.median(np.abs(lp_bg - med), axis=0)
    mad[mad <= 0] = np.finfo(np.float64).eps
    lp = _band_logpower(w_ev, fs, band_edges)
    z = (lp - med) / mad
    return (z > band_z).sum(axis=1), z


def _gate_halfwidth(w_wide, fs):
    """Full width at half maximum of the central deflection, in ms."""
    a = np.abs(w_wide)
    c = a.shape[1] // 2
    peak = a[:, c]
    peak[peak <= 0] = np.finfo(np.float32).eps
    below = a < 0.5 * peak[:, None]
    right = below[:, c:]
    left = below[:, :c + 1][:, ::-1]
    # argmax on a boolean row gives the first True; if none, use the full arm
    r = np.where(right.any(axis=1), right.argmax(axis=1), right.shape[1])
    l = np.where(left.any(axis=1), left.argmax(axis=1), left.shape[1])
    return (r + l) * 1000.0 / fs


# ----------------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------------

def detect_spikes(data: np.ndarray,
                  params: SpikeParams,
                  ch_names: Optional[Sequence[str]] = None,
                  return_rejected: bool = False,
                  verbose: bool = False) -> Dict[str, np.ndarray]:
    """Detect interictal epileptiform spikes.

    Parameters
    ----------
    data : (n_channels, n_samples) array. Bipolar or Laplacian referencing is
        strongly recommended for iEEG before calling this.
    params : SpikeParams
    ch_names : optional channel labels
    return_rejected : if True, also return candidates that failed a gate, with a
        ``reject_reason`` field. Very useful while tuning.

    Returns
    -------
    dict of equal-length arrays:
        channel, channel_name, sample, time_s, amplitude_uv, det_z, scale_ms,
        n_cycles, n_bands, lf_z, halfwidth_ms, accepted, reject_reason
    """
    p = params
    data = np.atleast_2d(np.asarray(data))
    n_ch, n_t = data.shape
    dt = np.dtype(p.dtype)
    fs = float(p.fs)
    rng = np.random.default_rng(p.random_seed)

    kernels = _ricker_kernels(p.scales_ms, fs, dt)
    sos_hp = (signal.butter(p.filter_order, p.highpass / (fs / 2.0),
                            btype="high", output="sos")
              if p.highpass > 0 else None)
    sos_notch = _notch_sos(p.notch_freqs, fs, p.notch_q)
    sos_wide = _sos_band(p.wide_band, fs, p.filter_order)
    sos_hf = _sos_band(p.hf_band, fs, p.filter_order)
    sos_lf = _sos_band(p.lf_band, fs, p.filter_order)

    band_edges = tuple((lo, hi) for lo, hi in p.band_edges if lo < fs / 2.0)

    refr = max(1, int(round(p.refractory_ms * 1e-3 * fs)))
    refine = max(1, int(round(p.peak_refine_ms * 1e-3 * fs)))
    h_cyc = max(2, int(round(p.cycle_win_ms * 1e-3 * fs)))
    h_bnd = max(8, int(round(p.band_win_ms * 1e-3 * fs)))
    h_lf = max(2, int(round(p.lf_win_ms * 1e-3 * fs)))
    h_wid = max(2, int(round(p.max_halfwidth_ms * 1e-3 * fs)) + 2)
    blk_len = max(int(round(p.baseline_block_s * fs)), 4 * refr)

    t_block = int(round(p.time_block_s * fs)) if p.time_block_s > 0 else n_t
    t_block = max(t_block, 4 * blk_len)
    ov = int(round(p.overlap_s * fs))
    pad = int(4 * fs)                     # filter settling pad, discarded

    cols = {k: [] for k in ("channel", "sample", "amplitude_uv", "det_z", "scale_ms",
                            "n_cycles", "n_bands", "lf_z", "halfwidth_ms",
                            "accepted", "reject_reason")}

    for c0 in range(0, n_ch, p.channel_block):
        c1 = min(c0 + p.channel_block, n_ch)
        for t0 in range(0, n_t, t_block):
            t1 = min(t0 + t_block, n_t)
            if t1 - t0 < int(2 * fs):
                continue
            s0 = max(0, t0 - ov - pad)
            s1 = min(n_t, t1 + ov + pad)
            raw = np.ascontiguousarray(data[c0:c1, s0:s1], dtype=dt)

            x_clean = _filt(sos_notch, _filt(sos_hp, raw))
            del raw
            x_wide = _filt(sos_wide, x_clean)
            x_hf = _filt(sos_hf, x_clean)
            x_lf = _filt(sos_lf, x_clean)

            D, best = detection_function(x_wide, kernels)
            med, sig, n_blk = _robust_stats_blocks(D, blk_len)

            # robust scale of the LF signal, once per channel-block
            lf_med = np.median(x_lf, axis=1)
            lf_sig = 1.4826 * np.median(np.abs(x_lf - lf_med[:, None]), axis=1)
            lf_sig[lf_sig <= 0] = np.finfo(np.float32).eps

            # keep only events whose *global* sample lies in the core region
            core_lo = t0 - s0
            core_hi = t1 - s0

            for ci in range(c1 - c0):
                d = D[ci]
                blk_id = np.minimum(np.arange(d.size) // blk_len, n_blk - 1)
                thr = med[ci][blk_id] + p.thresh_mad * sig[ci][blk_id]
                idx = _pick_peaks(d, thr.astype(d.dtype), refr)
                idx = idx[(idx >= core_lo) & (idx < core_hi)]
                if idx.size == 0:
                    continue

                # refine to the morphological peak in the wideband trace
                wref = _windows(x_wide[ci], idx, refine)
                idx = idx + (np.abs(wref).argmax(axis=1) - refine)
                idx = np.clip(idx, 0, d.size - 1)
                del wref

                det_z = (d[idx] - med[ci][np.minimum(idx // blk_len, n_blk - 1)]) / \
                        sig[ci][np.minimum(idx // blk_len, n_blk - 1)]
                amp = x_wide[ci][idx]
                scale = np.asarray(p.scales_ms, dtype=np.float64)[best[ci][idx]]

                reason = np.zeros(idx.size, dtype=object)
                reason[:] = ""
                ok = np.ones(idx.size, dtype=bool)

                if p.max_amp_mad > 0:
                    bad = det_z > p.max_amp_mad
                    reason[bad & ok] = "artifact_amplitude"
                    ok &= ~bad

                # --- Gate A: cycle count
                if p.use_cycle_gate:
                    ncyc = _gate_cycles(_windows(x_hf[ci], idx, h_cyc), fs,
                                        p.cycle_rel_thresh)
                    bad = ncyc >= p.max_cycles
                    reason[bad & ok] = "oscillatory_burst"
                    ok &= ~bad
                else:
                    ncyc = np.full(idx.size, np.nan)

                # --- Gate B: broadband coincidence
                if p.use_band_gate and len(band_edges) >= 2:
                    nbg = min(p.n_baseline_windows, max(50, idx.size * 10))
                    bg_idx = rng.integers(h_bnd, max(h_bnd + 1, d.size - h_bnd), size=nbg)
                    w_bg = _windows(x_clean[ci], bg_idx, h_bnd)
                    w_ev = _windows(x_clean[ci], idx, h_bnd)
                    nband, _ = _gate_bands(w_ev, w_bg, fs, band_edges, p.band_z)
                    del w_bg, w_ev
                    bad = nband < p.min_bands
                    reason[bad & ok] = "not_broadband"
                    ok &= ~bad
                else:
                    nband = np.full(idx.size, -1)

                # --- Gate C: low-frequency involvement
                if p.use_lf_gate:
                    w_lf = np.abs(_windows(x_lf[ci], idx, h_lf))
                    lfz = (w_lf.max(axis=1) - lf_med[ci]) / lf_sig[ci]
                    bad = lfz < p.lf_z
                    reason[bad & ok] = "no_lf_component"
                    ok &= ~bad
                else:
                    lfz = np.full(idx.size, np.nan)

                # --- Gate D: half-width
                if p.use_width_gate:
                    hw = _gate_halfwidth(_windows(x_wide[ci], idx, h_wid), fs)
                    bad = (hw < p.min_halfwidth_ms) | (hw > p.max_halfwidth_ms)
                    reason[bad & ok] = "bad_halfwidth"
                    ok &= ~bad
                else:
                    hw = np.full(idx.size, np.nan)

                sel = np.ones(idx.size, dtype=bool) if return_rejected else ok
                if not sel.any():
                    continue
                cols["channel"].append(np.full(sel.sum(), c0 + ci, dtype=np.int32))
                cols["sample"].append(idx[sel] + s0)
                cols["amplitude_uv"].append(amp[sel])
                cols["det_z"].append(det_z[sel])
                cols["scale_ms"].append(scale[sel])
                cols["n_cycles"].append(np.asarray(ncyc)[sel])
                cols["n_bands"].append(np.asarray(nband)[sel])
                cols["lf_z"].append(lfz[sel])
                cols["halfwidth_ms"].append(hw[sel])
                cols["accepted"].append(ok[sel])
                cols["reject_reason"].append(reason[sel])

            del D, best, x_clean, x_wide, x_hf, x_lf
        if verbose:
            print(f"  channels {c0}-{c1 - 1} done", flush=True)

    out = {}
    for k, v in cols.items():
        out[k] = np.concatenate(v) if v else np.empty(0, dtype=np.float64)

    # sort by time then channel
    if out["sample"].size:
        o = np.lexsort((out["channel"], out["sample"]))
        for k in out:
            out[k] = out[k][o]

    # --- Stage 3: cross-channel coincidence rejection
    if p.use_coincidence_gate and out["sample"].size and n_ch > 2 \
            and p.max_coincident_frac < 1.0:
        win = int(round(p.coincidence_ms * 1e-3 * fs))
        s = out["sample"].astype(np.int64)
        acc = out["accepted"].astype(bool)
        lo = np.searchsorted(s, s - win, side="left")
        hi = np.searchsorted(s, s + win, side="right")
        n_near = np.array([len(np.unique(out["channel"][a:b][acc[a:b]]))
                           for a, b in zip(lo, hi)])
        bad = (n_near > p.max_coincident_frac * n_ch) & acc
        out["accepted"] = acc & ~bad
        out["reject_reason"] = np.where(bad, "cross_channel_artifact",
                                        out["reject_reason"])

    out["time_s"] = out["sample"] / fs if out["sample"].size else np.empty(0)
    if ch_names is not None and out["sample"].size:
        out["channel_name"] = np.asarray(ch_names, dtype=object)[out["channel"]]

    if not return_rejected and out["sample"].size:
        keep = out["accepted"].astype(bool)
        for k in out:
            out[k] = out[k][keep]

    return out
