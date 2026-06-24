"""
extract.py  –  Extract raw data streams from a Rerun dataset view.

Functions are stateless – each accepts a ``ds_view`` (a Rerun DatasetEntry)
and returns NumPy arrays plus timestamps.

Camera frames are never accumulated in RAM; timestamps are read separately
from the bulk pixel pass so we can build sync indices before decoding pixels.

Stereo head camera
──────────────────
The /cameras/head stream encodes a side-by-side stereo frame:
    [LEFT HALF | RIGHT HALF]
``get_camera_timestamps`` and ``stream_camera_frames`` both accept a
``split_stereo`` flag.  When True, each decoded frame is yielded as a
``(left, right)`` pair instead of the full-width image.

H.264 / video time-windowing note
──────────────────────────────────
Video is delta-compressed: P-frames depend on earlier I-frames.  Feeding
PyAV packets starting mid-stream raises InvalidDataError.  Therefore ALL
video functions decode from the beginning of the stream (no packet-level
time filter).  Time-window filtering is applied to the *decoded* frame
timestamps instead, and full-stream frame indices are returned so callers
can build "needed" dicts that work with a full-stream decode pass in
``stream_camera_frames``.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Generator, Iterator

import av
import numpy as np
import pandas as pd

# ──────────────────── codec lookup ────────────────────────────────────────────
_CODEC_MAP: dict[int, str] = {
    1635148593: "h264",
    1751480163: "hevc",
}


# ──────────────────── helpers ─────────────────────────────────────────────────

def _to_ns(ts_series: pd.Series) -> np.ndarray:
    """Convert datetime64[ns] Series to int64 nanoseconds since epoch."""
    return ts_series.values.astype(np.int64)


def _scale_av_frame(frame: av.VideoFrame, scale: float | None) -> av.VideoFrame:
    """Resize an av.VideoFrame using PyAV reformat (no numpy round-trip)."""
    if scale is None or scale == 1.0:
        return frame
    new_w = max(1, int(frame.width  * scale))
    new_h = max(1, int(frame.height * scale))
    return frame.reformat(width=new_w, height=new_h)


def _packet_df(ds_view, stream: str) -> tuple[pd.DataFrame, str, str]:
    """Read the full compressed-packet dataframe for a video stream.

    Never filtered by time – callers must always start from packet 0 so
    H.264 has the I-frame it needs.
    """
    sample_col = f"{stream}:VideoStream:sample"
    codec_col  = f"{stream}:VideoStream:codec"
    df = (
        ds_view.reader(index="video_pts")
        .select(sample_col, codec_col, "ts")
        .to_pandas()
        .dropna(subset=[sample_col])
    )
    return df, sample_col, codec_col


def _make_codec_ctx(df: pd.DataFrame, codec_col: str) -> av.CodecContext:
    codec_val = int(df[codec_col].dropna().iloc[0][0])
    return av.CodecContext.create(_CODEC_MAP.get(codec_val, "h264"), "r")


# ──────────────────── camera – timestamps only ────────────────────────────────

def collect_frame_timestamps(
    ds_view,
    stream: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode the full video stream collecting (frame_index, ts_ns) pairs.

    No pixel data is produced – only codec tokens are decoded, so this is
    ~10-20× faster than a full pixel decode pass.  Call this once per camera
    per .rrd file; the returned arrays cover the entire recording.

    Returns
    -------
    frame_indices : (N,) int64  sequential frame numbers 0, 1, 2, …
    ts_ns         : (N,) int64  nanosecond timestamp for each frame
    """
    df, sample_col, codec_col = _packet_df(ds_view, stream)
    if df.empty:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    codec_ctx = _make_codec_ctx(df, codec_col)
    ts_list: list[int] = []

    for ts, sample in zip(df["ts"], df[sample_col]):
        t = int(pd.Timestamp(ts).value)
        for _ in codec_ctx.decode(av.Packet(bytes(sample[0]))):
            ts_list.append(t)
    for _ in codec_ctx.decode(None):
        ts_list.append(ts_list[-1] if ts_list else 0)

    ts_ns = np.array(ts_list, dtype=np.int64)
    frame_indices = np.arange(len(ts_ns), dtype=np.int64)
    return frame_indices, ts_ns


def get_camera_timestamps(
    ds_view,
    stream: str,
    start_ns: int | None = None,
    end_ns: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode the full video stream, return timestamps filtered to a window.

    The full stream is always decoded (H.264 I-frame requirement).
    After collecting all decoded-frame timestamps, the array is filtered to
    ``[start_ns, end_ns]``.

    Parameters
    ----------
    start_ns / end_ns : optional nanosecond bounds for the episode window.

    Returns
    -------
    ts_window    : (K,) int64  timestamps within [start_ns, end_ns].
    full_indices : (K,) int64  indices into the *full* decoded-frame sequence.
                   Use these in the ``needed`` dict for stream_camera_frames.
    """
    df, sample_col, codec_col = _packet_df(ds_view, stream)
    if df.empty:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    codec_ctx = _make_codec_ctx(df, codec_col)
    ts_ns: list[int] = []

    for ts, sample in zip(df["ts"], df[sample_col]):
        t = int(pd.Timestamp(ts).value)
        for _ in codec_ctx.decode(av.Packet(bytes(sample[0]))):
            ts_ns.append(t)
    for _ in codec_ctx.decode(None):
        ts_ns.append(ts_ns[-1] if ts_ns else 0)

    ts_all = np.array(ts_ns, dtype=np.int64)
    full_indices = np.arange(len(ts_all), dtype=np.int64)

    mask = np.ones(len(ts_all), dtype=bool)
    if start_ns is not None:
        mask &= ts_all >= start_ns
    if end_ns is not None:
        mask &= ts_all <= end_ns

    return ts_all[mask], full_indices[mask]


def get_camera_frame_shape(
    ds_view,
    stream: str,
    split_stereo: bool = False,
    start_ns: int | None = None,
    end_ns: int | None = None,
) -> tuple[int, int] | None:
    """Return (H, W) of the first decoded frame (post-split if applicable).

    start_ns / end_ns are accepted for API compatibility but ignored –
    frame dimensions are constant across the stream.
    """
    df, sample_col, codec_col = _packet_df(ds_view, stream)
    if df.empty:
        return None
    codec_ctx = _make_codec_ctx(df, codec_col)
    for sample in df[sample_col]:
        for frame in codec_ctx.decode(av.Packet(bytes(sample[0]))):
            h, w = frame.height, frame.width
            if split_stereo:
                w = w // 2
            return h, w
    return None


# ──────────────────── camera – streaming pixel decode ─────────────────────────

@dataclass
class CameraFrame:
    frame_idx: int
    ts_ns: int
    rgb: np.ndarray           # (H, W, 3) uint8 – full or left half
    rgb_right: np.ndarray | None = None  # right half when split_stereo=True


def stream_camera_frames(
    ds_view,
    stream: str,
    needed: dict[int, list[int]],
    split_stereo: bool = False,
    scale: float | None = None,
) -> Generator[CameraFrame, None, None]:
    """Decode the full video stream, yield only frames listed in *needed*.

    Parameters
    ----------
    needed      : {full_stream_frame_idx: [output_row, ...]}
                  Keys are indices into the *full* decoded frame sequence
                  (as returned by get_camera_timestamps full_indices).
                  Frames absent from this dict are decoded and discarded.
    split_stereo: If True, yield rgb (left half) and rgb_right (right half).
    scale       : Resize factor applied after optional stereo split.

    Note: always decodes from frame 0 for H.264 I-frame correctness.
    """
    df, sample_col, codec_col = _packet_df(ds_view, stream)
    if df.empty:
        return

    codec_ctx = _make_codec_ctx(df, codec_col)
    frame_idx = 0

    def _process_frame(frame: av.VideoFrame) -> tuple[np.ndarray, np.ndarray | None]:
        scaled = _scale_av_frame(frame, scale)
        if split_stereo:
            half_w = scaled.width // 2
            full   = scaled.to_ndarray(format="rgb24")
            return full[:, :half_w, :], full[:, half_w:, :]
        return scaled.to_ndarray(format="rgb24"), None

    def _inner() -> Iterator[CameraFrame]:
        nonlocal frame_idx
        for _ts, sample in zip(df["ts"], df[sample_col]):
            for frame in codec_ctx.decode(av.Packet(bytes(sample[0]))):
                rows = needed.get(frame_idx)
                if rows:
                    left, right = _process_frame(frame)
                    for row in rows:
                        yield CameraFrame(frame_idx=row, ts_ns=0, rgb=left, rgb_right=right)
                frame_idx += 1
        for frame in codec_ctx.decode(None):
            rows = needed.get(frame_idx)
            if rows:
                left, right = _process_frame(frame)
                for row in rows:
                    yield CameraFrame(frame_idx=row, ts_ns=0, rgb=left, rgb_right=right)
            frame_idx += 1

    yield from _inner()


# ──────────────────── non-video extractors ────────────────────────────────────

def extract_pose(
    ds_view,
    stream: str,
    start_ns: int | None = None,
    end_ns: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return translation (N,3), quaternion (N,4), timestamps (N,) ns."""
    base      = f"{stream}:Transform3D"
    trans_col = f"{base}:translation"
    quat_col  = f"{base}:quaternion"

    df = (
        ds_view.reader(index="robot_ts")
        .select(trans_col, quat_col, "ts")
        .to_pandas()
        .dropna(subset=[trans_col, quat_col])
    )
    if start_ns is not None:
        df = df[df["ts"].values.astype(np.int64) >= start_ns]
    if end_ns is not None:
        df = df[df["ts"].values.astype(np.int64) <= end_ns]
    if df.empty:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 4), dtype=np.float32),
            np.empty(0, dtype=np.int64),
        )

    translations = np.stack([r[0] for r in df[trans_col]], axis=0).astype(np.float32)
    quaternions  = np.stack([r[0] for r in df[quat_col]],  axis=0).astype(np.float32)
    return translations, quaternions, _to_ns(df["ts"])


def extract_scalars(
    ds_view,
    entity: str,
    start_ns: int | None = None,
    end_ns: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return scalars (N,D) float32 and timestamps (N,) ns."""
    col = f"{entity}:Scalars:scalars"
    df = (
        ds_view.reader(index="robot_ts")
        .select(col, "ts")
        .to_pandas()
        .dropna(subset=[col])
    )
    if start_ns is not None:
        df = df[df["ts"].values.astype(np.int64) >= start_ns]
    if end_ns is not None:
        df = df[df["ts"].values.astype(np.int64) <= end_ns]
    if df.empty:
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.int64)

    data = np.stack(df[col].values, axis=0).astype(np.float32)
    return data, _to_ns(df["ts"])
