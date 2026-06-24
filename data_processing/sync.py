"""
sync.py  –  Causal frequency alignment (zero-order hold / last-known-value).

Design constraint:
    At every target timestep t, ALL data values must originate from a source
    sample at time ≤ t.  Future samples are never used.  This is implemented
    as a "previous or coincident" nearest-neighbour search (ZOH).

Key functions
─────────────
causal_latest_indices(master_ts, source_ts)
    For each timestamp in master_ts, return the index of the most recent
    sample in source_ts that is ≤ master_ts[i], plus a validity mask.

temporal_downsample_causal(ts_ns, target_fps)
    Given the raw timestamps of the master stream, select a subset whose
    spacing approximates 1/target_fps seconds using causal selection.

build_valid_mask(master_ts, stream_ts_dict, max_gap_ns)
    Combine per-stream validity masks into one boolean array.
"""

from __future__ import annotations

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────


def causal_latest_indices(
    master_ts: np.ndarray,
    source_ts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """For each master timestamp, find the most recent source sample at or before it.

    Parameters
    ----------
    master_ts : (M,) int64 – target timestamps in nanoseconds (must be sorted).
    source_ts : (S,) int64 – source timestamps in nanoseconds (need not be sorted).

    Returns
    -------
    indices : (M,) int64
        Index into *source_ts* of the most recent causal sample.
        For steps where no causal sample exists, the index is 0 (sentinel –
        check the *valid* mask to identify these positions).
    valid : (M,) bool
        True  → a causal sample exists (indices[i] is meaningful).
        False → no source sample exists at or before master_ts[i].
    """
    if len(source_ts) == 0:
        return np.zeros(len(master_ts), dtype=np.int64), np.zeros(len(master_ts), dtype=bool)

    # Sort source timestamps; we'll un-sort the final indices.
    sort_order = np.argsort(source_ts, kind="stable")
    sorted_ts = source_ts[sort_order]

    # searchsorted(..., side='right') → insertion point after all equal values.
    # Subtract 1 to get the last index with value ≤ master_ts[i].
    pos = np.searchsorted(sorted_ts, master_ts, side="right") - 1  # (M,) may be -1

    valid = pos >= 0  # -1 means master_ts[i] is before all source samples

    pos_safe = pos.clip(0, len(sorted_ts) - 1)
    indices = sort_order[pos_safe]

    return indices, valid


def temporal_downsample_causal(
    ts_ns: np.ndarray,
    target_fps: float,
) -> np.ndarray:
    """Select frame indices that best approximate a regular causal grid.

    For each ideal target time t_k = t_0 + k*(1/target_fps) the last actual
    frame at or before t_k is chosen.  This guarantees:
      • No future data is used.
      • Each source frame is selected at most once.
      • Gaps in the source (e.g. pause artifacts) produce no output frames.

    Parameters
    ----------
    ts_ns      : (N,) int64 – sorted source timestamps in nanoseconds.
    target_fps : float      – desired output frame rate.

    Returns
    -------
    keep : (K,) int64 – indices into ts_ns to keep (K ≤ N).
    """
    if len(ts_ns) == 0:
        return np.empty(0, dtype=np.int64)

    interval_ns = int(1e9 / target_fps)
    ideal_ts = np.arange(ts_ns[0], ts_ns[-1] + interval_ns, interval_ns, dtype=np.int64)

    keep: list[int] = []
    last_kept_idx = -1

    for t in ideal_ts:
        # Rightmost position with value ≤ t.
        pos = int(np.searchsorted(ts_ns, t, side="right")) - 1
        if pos < 0:
            continue  # no frame at or before this ideal tick

        # Enforce strictly increasing frame selection (no duplicates).
        if pos > last_kept_idx:
            keep.append(pos)
            last_kept_idx = pos

    return np.array(keep, dtype=np.int64)


def make_target_grid(
    start_ns: int,
    end_ns: int,
    target_fps: float,
) -> np.ndarray:
    """Return a regular causal grid of nanosecond timestamps.

    The grid starts at *start_ns* and steps by ``1e9 / target_fps`` until
    *end_ns* (inclusive if it lands exactly on the grid).  This is the
    master timeline used for ZOH alignment.

    Using a regular grid (rather than actual camera timestamps) allows
    target_fps > source_fps – frames will simply repeat via ZOH.

    Parameters
    ----------
    start_ns / end_ns : episode window in nanoseconds.
    target_fps        : desired output frame rate (Hz).

    Returns
    -------
    (K,) int64 timestamps, K ≈ (end_ns - start_ns) * target_fps / 1e9.
    """
    interval_ns = int(round(1e9 / target_fps))
    return np.arange(start_ns, end_ns + 1, interval_ns, dtype=np.int64)


def build_valid_mask(
    master_ts: np.ndarray,
    stream_indices: dict[str, np.ndarray],
    stream_ts:      dict[str, np.ndarray],
    stream_valid:   dict[str, np.ndarray],
    max_gap_ns: int,
) -> np.ndarray:
    """Build a per-step validity mask across all non-master streams.

    A step is valid if every stream has a causal sample within max_gap_ns.

    Parameters
    ----------
    master_ts      : (N,) sorted target timestamps.
    stream_indices : stream_name → (N,) index array (from causal_latest_indices).
    stream_ts      : stream_name → raw source timestamp array.
    stream_valid   : stream_name → (N,) bool from causal_latest_indices.
    max_gap_ns     : maximum allowed age of a causal sample.

    Returns
    -------
    valid : (N,) bool
    """
    valid = np.ones(len(master_ts), dtype=bool)

    for name, idx in stream_indices.items():
        # Steps where causal_latest_indices found no sample.
        sv = stream_valid[name]
        valid &= sv

        # Steps where the most recent sample is older than the threshold.
        src_ts = stream_ts[name]
        age_ns = master_ts - src_ts[idx]   # positive = sample is in the past
        valid &= age_ns <= max_gap_ns

    return valid
