"""
rrd_to_hdf5.py  –  Phase 1: convert paired .rrd files to a single HDF5 dataset.

Episode definition
──────────────────
Each episode corresponds to one ``generic_subtask`` entry in the paired
workflow .rrd file.  The workflow file contains alternating
``generic_subtask`` / ``idle`` rows; only the former define robot episodes.

The episode's time window comes directly from the workflow event:
    start_time  ->  episode start (nanoseconds)
    end_time    ->  episode end   (nanoseconds)

All data streams are aligned to a regular ``target_fps`` grid using causal
ZOH (last-known-value – no future data).  The master timeline is a fixed
frequency grid, so ``target_fps`` > source camera fps produces repeated
frames (e.g. 50 Hz output from a 30 Hz head camera).

Decode-once architecture
────────────────────────
Video streams are decoded exactly TWICE per camera per .rrd file regardless
of how many episodes the file contains:
  1. ``collect_frame_timestamps``  – fast token decode, no pixel conversion.
  2. ``stream_camera_frames``      – full pixel decode with a combined
     ``needed`` dict covering ALL episodes at once.

Non-video data (scalars, poses) is loaded once per file via a single Parquet
read, then episode slices are ZOH-aligned from that preloaded DataFrame.

HDF5 writing
────────────
Camera datasets are created as empty resizable datasets
    shape=(0, H, W, 3),  maxshape=(None, H, W, 3)
and grown one frame at a time via resize+write.  This avoids having to know
n_valid upfront and eliminates all row-index mapping complexity.

Usage
─────
    python -m data_processing.rrd_to_hdf5 \\
        --data-dir     data/cube-to-box/data \\
        --workflow-dir data/cube-to-box/workflow \\
        --output-dir   output/hdf5/cube-to-box \\
        --fps 50 --stereo-side both

        python -m data_processing.rrd_to_hdf5 \\
        --data-dir     data/04-06-26/data \\
        --workflow-dir data/04-06-26/workflow \\
        --output-dir   output/hdf5 \\
        --fps 50 --stereo-side both

        python -m data_processing.rrd_to_hdf5 --data-dir     data/validation/data --workflow-dir data/validation/workflow --output-dir   output/hdf5/validation --fps 50 --stereo-side both
"""

from __future__ import annotations

import sys
import traceback
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

from .config import Config
from .extract import (
    collect_frame_timestamps,
    get_camera_frame_shape,
    stream_camera_frames,
)
from .rrd_loader import pair_rrd_files, RRDDatasetPair
from .sync import causal_latest_indices, make_target_grid
from .workflow import WorkflowEvent, load_generic_subtask_events

_VERSION = "0.4.0"


# ──────────────────────────────────────────────────────────────────────────────
# Non-video ZOH alignment from a preloaded DataFrame
# ──────────────────────────────────────────────────────────────────────────────

def _zoh_align_df(
    ep_df: pd.DataFrame,
    target_ts_ns: np.ndarray,
    pose_streams: list[str],
    scalar_entities: list[str],
) -> dict[str, Any]:
    """ZOH-align a pre-loaded episode DataFrame to target_ts_ns.

    ``ep_df`` must have a ``robot_ts_ns`` int64 column covering at least the
    episode window.  Returns a flat dict of column → aligned array, plus a
    ``valid`` boolean mask indicating which target ticks had causal data.
    """
    src_ts = ep_df["robot_ts_ns"].values.astype(np.int64)
    idx, valid = causal_latest_indices(target_ts_ns, src_ts)

    result: dict[str, Any] = {"valid": valid}

    for s in pose_streams:
        t_col = f"{s}:Transform3D:translation"
        q_col = f"{s}:Transform3D:quaternion"
        if t_col in ep_df.columns:
            trans = np.stack([r[0] for r in ep_df[t_col].values], 0)[idx]
            quat  = np.stack([r[0] for r in ep_df[q_col].values], 0)[idx]
            result[t_col] = trans.astype(np.float32)
            result[q_col] = quat.astype(np.float32)

    for e in scalar_entities:
        col = f"{e}:Scalars:scalars"
        if col in ep_df.columns:
            data = np.stack(ep_df[col].values, 0)[idx]
            result[col] = data.astype(np.float32)

    return result


# ──────────────────────────────────────────────────────────────────────────────
# Per-episode record
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class _EpRecord:
    """Transient state for one episode during the file-level processing pass."""
    event:       WorkflowEvent
    ep_key:      str
    source_file: str
    target_ts:   np.ndarray        # (N,) int64 ns – the regular grid
    valid_mask:  np.ndarray        # (N,) bool
    zoh:         dict[str, Any]    # aligned non-video arrays + valid mask

    # stream → [full_stream_frame_idx, ...] with one entry per valid output row,
    # in output row order.  Frame repeats (ZOH holds) appear as duplicate indices.
    cam_frame_map: dict[str, list[int]] = field(default_factory=dict)

    @property
    def n_valid(self) -> int:
        return int(self.valid_mask.sum())


def _build_ep_record(
    event: WorkflowEvent,
    ep_key: str,
    source_file: str,
    cam_all: dict[str, tuple[np.ndarray, np.ndarray]],
    all_df: pd.DataFrame,
    cfg: Config,
) -> _EpRecord | None:
    """Build an _EpRecord for one episode, or return None to skip."""
    start_ns = event.start_ns
    end_ns   = event.end_ns

    # ── 1. Regular target grid ─────────────────────────────────────────────
    target_ts = make_target_grid(start_ns, end_ns, cfg.target_fps)
    if len(target_ts) < 2:
        print(f"    {ep_key}: SKIP – target grid has < 2 ticks")
        return None

    N = len(target_ts)
    window_s = (end_ns - start_ns) / 1e9
    print(f"    {ep_key}: window {window_s:.1f}s → {N} ticks at {cfg.target_fps:.0f} Hz")

    # ── 2. Non-video ZOH from preloaded DataFrame ──────────────────────────
    ep_df = all_df[
        (all_df["robot_ts_ns"] >= start_ns) &
        (all_df["robot_ts_ns"] <= end_ns)
    ]
    if ep_df.empty:
        print(f"    {ep_key}: SKIP – no robot_ts rows in episode window")
        return None

    zoh = _zoh_align_df(ep_df, target_ts, cfg.streams.poses, cfg.streams.scalars)
    valid_mask = zoh["valid"].copy()

    # ── 3. Camera ZOH: build per-stream validity and age ──────────────────
    cam_local_idx:  dict[str, np.ndarray] = {}
    stream_valid:   dict[str, np.ndarray] = {}
    stream_age_ns:  dict[str, np.ndarray] = {}

    for stream in cfg.streams.cameras:
        fi_all, ts_all = cam_all[stream]
        if len(ts_all) == 0:
            print(f"    {ep_key}: SKIP – {stream} has no frames")
            return None
        local_idx, valid_s = causal_latest_indices(target_ts, ts_all)
        cam_local_idx[stream] = local_idx
        stream_valid[stream]  = valid_s
        stream_age_ns[stream] = target_ts - ts_all[local_idx]

    # ── 4. Combined validity mask ──────────────────────────────────────────
    for stream in cfg.streams.cameras:
        valid_mask &= stream_valid[stream]
        valid_mask &= stream_age_ns[stream] <= cfg.sync.max_sync_gap_ns

    n_valid = int(valid_mask.sum())
    print(
        f"    {ep_key}: {n_valid}/{N} valid frames "
        f"(dropped {N - n_valid} outside {cfg.sync.max_sync_gap_ns / 1e6:.0f} ms gap)"
    )
    if n_valid == 0:
        print(f"    {ep_key}: SKIP – no valid frames")
        return None

    # ── 5. Build cam_frame_map (output-row order, valid ticks only) ───────
    # For each valid output row, record which full-stream frame index to use.
    # Duplicate full_frame_idx values appear when ZOH repeats a frame.
    cam_frame_map: dict[str, list[int]] = {}
    for stream in cfg.streams.cameras:
        fi_all     = cam_all[stream][0]
        local_idx  = cam_local_idx[stream]
        valid_locs = np.where(valid_mask)[0]
        cam_frame_map[stream] = [int(fi_all[local_idx[g]]) for g in valid_locs]

    return _EpRecord(
        event        = event,
        ep_key       = ep_key,
        source_file  = source_file,
        target_ts    = target_ts,
        valid_mask   = valid_mask,
        zoh          = zoh,
        cam_frame_map= cam_frame_map,
    )


# ──────────────────────────────────────────────────────────────────────────────
# HDF5 helpers
# ──────────────────────────────────────────────────────────────────────────────

def _write_ep_attrs(ep_grp: h5py.Group, rec: _EpRecord, cfg: Config) -> None:
    for k, v in rec.event.as_hdf5_attrs().items():
        ep_grp.attrs[k] = v

    valid_ts = rec.target_ts[rec.valid_mask]
    ep_grp.attrs.update({
        "source_file":        rec.source_file,
        "n_frames":           rec.n_valid,
        "target_fps":         cfg.target_fps,
        "master_stream":      cfg.master_stream,
        "max_sync_gap_ns":    cfg.sync.max_sync_gap_ns,
        "image_scale_factor": cfg.camera.image_scale_factor or 1.0,
        "start_ns":           int(valid_ts[0]),
        "end_ns":             int(valid_ts[-1]),
    })
    ep_grp.create_dataset("timestamps", data=valid_ts, dtype=np.int64)


def _write_nonvideo(ep_grp: h5py.Group, rec: _EpRecord, cfg: Config) -> None:
    """Write poses and scalars into an episode group."""
    valid = rec.valid_mask
    zoh   = rec.zoh

    if cfg.streams.poses:
        pose_grp = ep_grp.create_group("commanded_pose")
        for s in cfg.streams.poses:
            t_col = f"{s}:Transform3D:translation"
            q_col = f"{s}:Transform3D:quaternion"
            if t_col not in zoh:
                continue
            name = s.rstrip("/").split("/")[-1]
            g = pose_grp.create_group(name)
            g.create_dataset("translation", data=zoh[t_col][valid])
            g.create_dataset("quaternion",  data=zoh[q_col][valid])

    for e in cfg.streams.scalars:
        col = f"{e}:Scalars:scalars"
        if col not in zoh:
            continue
        parts = e.lstrip("/").split("/")
        grp = ep_grp
        for part in parts[:-1]:
            grp = grp.require_group(part)
        grp.create_dataset(parts[-1], data=zoh[col][valid])


def _create_camera_datasets(
    ep_grp: h5py.Group,
    rec: _EpRecord,
    data_ds,
    cfg: Config,
) -> dict[str, tuple[bool, h5py.Dataset | None, h5py.Dataset | None]]:
    """Create empty resizable camera datasets for one episode.

    Returns stream → (is_stereo, dset_left_or_mono, dset_right_or_None).
    """
    cam_grp  = ep_grp.create_group("cameras")
    comp     = {"compression": cfg.camera.compression,
                "compression_opts": cfg.camera.compression_opts}
    is_stereo = set(cfg.camera.stereo_streams)
    info: dict[str, tuple] = {}

    for stream in cfg.streams.cameras:
        raw_name = stream.rstrip("/").split("/")[-1]
        stereo   = stream in is_stereo
        shape    = get_camera_frame_shape(data_ds, stream, split_stereo=stereo)
        if shape is None:
            print(f"    WARNING: cannot determine shape for {stream}, skipping")
            continue

        H, W = shape
        sc = cfg.camera.image_scale_factor
        if sc and sc != 1.0:
            H = max(1, int(H * sc))
            W = max(1, int(W * sc))

        # Start empty; grows via resize() during video streaming.
        def _make(name: str) -> h5py.Dataset:
            return cam_grp.create_dataset(
                name,
                shape=(0, H, W, 3),
                maxshape=(None, H, W, 3),
                dtype=np.uint8,
                chunks=(1, H, W, 3),
                **comp,
            )

        if stereo:
            sides = cfg.camera.stereo_side
            dl = _make(f"{raw_name}_left")  if sides in ("left",  "both") else None
            dr = _make(f"{raw_name}_right") if sides in ("right", "both") else None
            info[stream] = (True, dl, dr)
        else:
            info[stream] = (False, _make(raw_name), None)

    return info


# ──────────────────────────────────────────────────────────────────────────────
# Stream auto-discovery
# ──────────────────────────────────────────────────────────────────────────────

def _auto_discover_streams(ds, cfg: Config) -> None:
    if None not in (cfg.streams.cameras, cfg.streams.poses, cfg.streams.scalars):
        return
    all_cols = [c.name for c in ds.schema().component_columns()]
    if cfg.streams.cameras is None:
        cfg.streams.cameras = sorted({
            c.rsplit(":", 2)[0] for c in all_cols if ":VideoStream:sample" in c
        })
    if cfg.streams.poses is None:
        cfg.streams.poses = sorted({
            c.rsplit(":", 2)[0]
            for c in all_cols
            if ":Transform3D:translation" in c and not c.startswith("property:")
        })
    if cfg.streams.scalars is None:
        cfg.streams.scalars = sorted({
            c.rsplit(":", 2)[0]
            for c in all_cols
            if ":Scalars:scalars" in c
            and not c.startswith("property:")
            and not c.startswith("/system/")
        })


# ──────────────────────────────────────────────────────────────────────────────
# File-level conversion
# ──────────────────────────────────────────────────────────────────────────────

def convert_file(
    data_rrd: Path,
    wf_rrd: Path,
    f: h5py.File,
    global_ep_idx: int,
    cfg: Config,
    streams_found: bool,
) -> tuple[int, bool, list[tuple]]:
    """Convert all episodes from one (data_rrd, wf_rrd) pair."""
    errors: list[tuple] = []
    print(f"[{data_rrd.name}]")

    try:
        with RRDDatasetPair(data_rrd, wf_rrd) as pair:
            if not streams_found:
                _auto_discover_streams(pair.data_ds, cfg)
                streams_found = True

            subtask_events = load_generic_subtask_events(
                pair.workflow_ds, cfg.streams.workflow_fields
            )
            print(f"  generic_subtask events: {len(subtask_events)}")
            if not subtask_events:
                print("  WARNING: no generic_subtask events – skipping pair")
                return global_ep_idx, streams_found, errors

            # ── Step 1: collect video frame timestamps (one pass per camera) ──
            print("  Collecting camera frame timestamps …")
            cam_all: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            for stream in cfg.streams.cameras:
                fi, ts = collect_frame_timestamps(pair.data_ds, stream)
                cam_all[stream] = (fi, ts)
                duration_s = (ts[-1] - ts[0]) / 1e9 if len(ts) > 1 else 0.0
                print(f"    {stream}: {len(ts)} frames  {duration_s:.1f}s")

            # ── Step 2: load all non-video data once ──────────────────────────
            # Load all robot_ts rows, drop compressed video columns, then
            # forward-fill (ZOH) so every row has a value for every stream
            # (streams that log less frequently would otherwise leave NaN gaps).
            raw = (
                pair.data_ds
                .reader(index="robot_ts")
                .to_arrow_table()
                .to_pandas()
            )
            drop = [c for c in raw.columns if "VideoStream:sample" in c]
            raw = raw.drop(columns=drop)
            raw["robot_ts_ns"] = raw["robot_ts"].astype(np.int64)
            all_df = raw.sort_values("robot_ts_ns").ffill()
            print(f"  Non-video preload: {len(all_df)} rows")

            # ── Step 3: build _EpRecord for every episode ─────────────────────
            records: list[_EpRecord] = []
            for ev_idx, event in enumerate(subtask_events):
                ep_key = f"episode_{global_ep_idx + len(records):06d}"
                try:
                    rec = _build_ep_record(
                        event, ep_key, data_rrd.name, cam_all, all_df, cfg,
                    )
                except Exception:
                    tb = traceback.format_exc()
                    errors.append((data_rrd.name, ev_idx, tb))
                    print(f"    ERROR building record for event {ev_idx}:\n{tb}")
                    continue
                if rec is not None:
                    records.append(rec)

            if not records:
                print("  No valid episodes in this file.")
                return global_ep_idx, streams_found, errors

            # ── Step 4: build combined camera needed dict (all episodes) ──────
            # combined_needed[stream][full_frame_idx] = [ep_local, ep_local, ...]
            # One ep_local entry per output row that maps to this frame.
            # Duplicate entries handle ZOH frame repeats within an episode.
            combined_needed: dict[str, dict[int, list[int]]] = {
                s: defaultdict(list) for s in cfg.streams.cameras
            }
            for ep_local, rec in enumerate(records):
                for stream, frame_list in rec.cam_frame_map.items():
                    for full_i in frame_list:
                        combined_needed[stream][full_i].append(ep_local)

            # ── Step 5: write HDF5 attrs, non-video, create camera datasets ───
            ep_groups:       list[h5py.Group] = []
            cam_dsets_per_ep: list[dict]      = []

            for ep_local, rec in enumerate(records):
                ep_grp = f.create_group(rec.ep_key)
                ep_groups.append(ep_grp)
                try:
                    _write_ep_attrs(ep_grp, rec, cfg)
                    _write_nonvideo(ep_grp, rec, cfg)
                    cam_info = _create_camera_datasets(ep_grp, rec, pair.data_ds, cfg)
                    cam_dsets_per_ep.append(cam_info)
                except Exception:
                    tb = traceback.format_exc()
                    errors.append((data_rrd.name, ep_local, tb))
                    print(f"    ERROR writing attrs/non-video for {rec.ep_key}:\n{tb}")
                    del f[rec.ep_key]
                    cam_dsets_per_ep.append({})

            # ── Step 6: single streaming video decode pass per camera ─────────
            # Frames arrive in full_frame_idx order.  For each frame we append
            # to every episode that needs it (ep_local entry in combined_needed).
            is_stereo = set(cfg.camera.stereo_streams)
            try:
                for stream in cfg.streams.cameras:
                    if not combined_needed[stream]:
                        continue
                    stereo = stream in is_stereo
                    # Total output-row writes expected across all episodes.
                    total_writes = sum(
                        len(v) for v in combined_needed[stream].values()
                    )
                    n_src_frames = len(cam_all[stream][0])
                    print(f"  Streaming video → HDF5: {stream} "
                          f"({'stereo' if stereo else 'mono'})  "
                          f"[{n_src_frames} src frames → {total_writes} writes]")

                    writes_done = 0
                    _REPORT_EVERY = 250
                    for cf in stream_camera_frames(
                        pair.data_ds, stream,
                        dict(combined_needed[stream]),   # {full_i: [ep_local, ...]}
                        split_stereo=stereo,
                        scale=cfg.camera.image_scale_factor,
                    ):
                        ep_local = cf.frame_idx          # single int
                        ci = cam_dsets_per_ep[ep_local]
                        if stream not in ci:
                            continue
                        _stereo_flag, dl, dr = ci[stream]
                        if _stereo_flag:
                            if dl is not None and cf.rgb is not None:
                                dl.resize(dl.shape[0] + 1, axis=0)
                                dl[-1] = cf.rgb
                            if dr is not None and cf.rgb_right is not None:
                                dr.resize(dr.shape[0] + 1, axis=0)
                                dr[-1] = cf.rgb_right
                        else:
                            if dl is not None and cf.rgb is not None:
                                dl.resize(dl.shape[0] + 1, axis=0)
                                dl[-1] = cf.rgb
                        writes_done += 1
                        if writes_done % _REPORT_EVERY == 0:
                            pct = 100 * writes_done / max(1, total_writes)
                            print(f"    … {writes_done}/{total_writes} "
                                  f"({pct:.0f}%)", flush=True)
                    print(f"    done  {writes_done}/{total_writes} writes")
            except Exception:
                tb = traceback.format_exc()
                errors.append((data_rrd.name, -1, tb))
                print(f"  ERROR during video streaming:\n{tb}")

            # ── Step 7: always runs – count episodes still present in file ────
            for ep_local, rec in enumerate(records):
                if rec.ep_key in f:
                    global_ep_idx += 1
                    f.attrs["n_episodes"] = global_ep_idx

    except Exception:
        tb = traceback.format_exc()
        errors.append((data_rrd.name, -1, tb))
        print(f"  ERROR opening pair:\n{tb}")

    return global_ep_idx, streams_found, errors


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def convert_all(cfg: Config) -> None:
    output_path = cfg.output_dir / "dataset.h5"

    print(f"Data dir     : {cfg.data_dir.resolve()}")
    print(f"Workflow dir : {cfg.workflow_dir.resolve()}")
    print(f"Output file  : {output_path.resolve()}")
    print(f"Target FPS   : {cfg.target_fps}")
    print(f"Max sync gap : {cfg.sync.max_sync_gap_ns / 1e6:.0f} ms")
    print(f"Stereo side  : {cfg.camera.stereo_side}")
    print(f"Image scale  : {cfg.camera.image_scale_factor}")
    print()

    if cfg.skip_existing and output_path.exists():
        print(f"SKIP – output already exists: {output_path}")
        print("Run with --no-skip to overwrite.")
        return

    pairs = pair_rrd_files(cfg.data_dir, cfg.workflow_dir)
    print(f"Found {len(pairs)} .rrd pair(s)\n")

    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    global_ep_idx = 0
    streams_found = False
    all_errors: list[tuple] = []

    with h5py.File(output_path, "w") as f:
        f.attrs["pipeline_version"] = _VERSION
        f.attrs["n_episodes"]       = 0

        for data_rrd, wf_rrd in pairs:
            global_ep_idx, streams_found, errors = convert_file(
                data_rrd, wf_rrd, f, global_ep_idx, cfg, streams_found
            )
            all_errors.extend(errors)
            print()

    if global_ep_idx == 0:
        output_path.unlink(missing_ok=True)
        print("No episodes written – output file removed.")
    else:
        size_mb = output_path.stat().st_size / 1024 ** 2
        print(
            f"Wrote {output_path}  "
            f"({global_ep_idx} episode(s), {size_mb:.1f} MB)"
        )

    if all_errors:
        print(f"\n{len(all_errors)} error(s) occurred.")
        sys.exit(1)
    else:
        print("Done.")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _build_parser():
    import argparse
    p = argparse.ArgumentParser(
        description=(
            "Convert paired .rrd files to a single HDF5 dataset.\n"
            "Episodes are defined by 'generic_subtask' entries in the "
            "workflow .rrd file."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-dir",     type=Path, required=True)
    p.add_argument("--workflow-dir", type=Path, required=True)
    p.add_argument("--output-dir",   type=Path, default=Path("output/hdf5"))
    p.add_argument("--fps",          type=float, default=50.0,
                   help="Target output frame rate (Hz).")
    p.add_argument("--max-gap-ms",   type=float, default=50.0,
                   help="Max allowed age of a causal sample (ms).")
    p.add_argument("--stereo-side",  choices=["left", "right", "both"], default="both")
    p.add_argument("--image-scale",  type=float, default=0.5,
                   help="Resize factor for all cameras (1.0 = full res).")
    p.add_argument("--no-skip",      action="store_true")
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    cfg = Config(
        data_dir     = args.data_dir,
        workflow_dir = args.workflow_dir,
        output_dir   = args.output_dir,
        skip_existing= not args.no_skip,
    )
    cfg.sync.max_sync_gap_ns      = int(args.max_gap_ms * 1_000_000)
    cfg.sync.target_fps           = args.fps
    cfg.camera.stereo_side        = args.stereo_side
    cfg.camera.image_scale_factor = args.image_scale if args.image_scale != 1.0 else None
    convert_all(cfg)
