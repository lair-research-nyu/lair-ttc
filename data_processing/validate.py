"""
validate.py  –  Inspect, verify, and visualise HDF5 episode files.

CLI usage:
    python -m data_processing.validate output/hdf5/data1.h5
    python -m data_processing.validate output/hdf5/data1.h5 --episode 0 --rerun

Programmatic usage:
    from data_processing.validate import inspect_file, verify_episode
    inspect_file(Path("output/hdf5/data1.h5"))
    verify_episode(Path("output/hdf5/data1.h5"), episode_idx=0)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────

def _indent(n: int) -> str:
    return "  " * n


def _print_group(grp: h5py.Group, depth: int = 0) -> None:
    """Recursively print an HDF5 group tree."""
    for key in grp.keys():
        item = grp[key]
        if isinstance(item, h5py.Dataset):
            extra = f"  dtype={item.dtype}  chunks={item.chunks}"
            print(f"{_indent(depth)}[dataset] {key}  shape={item.shape}{extra}")
        elif isinstance(item, h5py.Group):
            print(f"{_indent(depth)}[group]   {key}/")
            _print_group(item, depth + 1)


def inspect_file(path: Path) -> None:
    """Print a full tree + summary statistics for an HDF5 file."""
    path = Path(path)
    if not path.exists():
        print(f"ERROR: file not found: {path}")
        return

    with h5py.File(path, "r") as f:
        print(f"\n{'='*60}")
        print(f"  {path.name}")
        print(f"{'='*60}")

        # File-level attrs.
        print("\nFile attributes:")
        for k, v in f.attrs.items():
            print(f"  {k}: {v}")

        n_episodes = f.attrs.get("n_episodes", "?")
        print(f"\nEpisodes: {n_episodes}")

        for ep_key in sorted(f.keys()):
            ep = f[ep_key]
            if not isinstance(ep, h5py.Group):
                continue
            n = ep.attrs.get("n_frames", "?")
            fps = ep.attrs.get("target_fps", "?")
            dur = ""
            if "timestamps" in ep:
                ts = ep["timestamps"][:]
                dur = f"  duration={( ts[-1]-ts[0])/1e9:.2f}s"

            print(f"\n  {ep_key}  (n_frames={n}, fps={fps}{dur})")

            # Print episode attrs.
            for k, v in ep.attrs.items():
                if k not in ("n_frames", "target_fps", "segment_id"):
                    print(f"    {k}: {v}")

            # Print dataset tree.
            _print_group(ep, depth=2)

            # Camera summary.
            if "cameras" in ep:
                for cam_name in ep["cameras"].keys():
                    ds = ep["cameras"][cam_name]
                    print(
                        f"    camera/{cam_name}: "
                        f"{ds.shape}  "
                        f"min={ds[0].min()}  max={ds[0].max()}"
                    )


def verify_episode(path: Path, episode_idx: int = 0) -> bool:
    """Run basic integrity checks on one episode.  Returns True if OK."""
    path = Path(path)
    ok = True

    with h5py.File(path, "r") as f:
        ep_keys = sorted(k for k in f.keys() if k.startswith("episode_"))
        if episode_idx >= len(ep_keys):
            print(f"ERROR: episode index {episode_idx} out of range (have {len(ep_keys)})")
            return False

        ep = f[ep_keys[episode_idx]]
        n  = ep.attrs.get("n_frames", -1)
        print(f"\nVerifying {path.name}  ->  {ep_keys[episode_idx]}  (n={n})")

        # ── Timestamps monotonically increasing? ──────────────────────────
        if "timestamps" in ep:
            ts = ep["timestamps"][:]
            if len(ts) != n:
                print(f"  FAIL: timestamps length {len(ts)} ≠ n_frames {n}")
                ok = False
            if not np.all(np.diff(ts) > 0):
                n_bad = int((np.diff(ts) <= 0).sum())
                print(f"  FAIL: timestamps not strictly increasing ({n_bad} violations)")
                ok = False
            else:
                print(f"  OK  : timestamps monotonically increasing")
        else:
            print("  FAIL: 'timestamps' dataset missing")
            ok = False

        # ── Camera frames non-zero? ────────────────────────────────────────
        if "cameras" in ep:
            for cam_name, cam_ds in ep["cameras"].items():
                if cam_ds.shape[0] != n:
                    print(f"  FAIL: cameras/{cam_name} has {cam_ds.shape[0]} frames ≠ {n}")
                    ok = False
                    continue
                # Spot-check first and last frame for all-black.
                for idx in (0, -1):
                    frame = cam_ds[idx]
                    if frame.max() == 0:
                        print(f"  WARN: cameras/{cam_name} frame {idx} is all-black")
                print(f"  OK  : cameras/{cam_name}  shape={cam_ds.shape}")

        # ── Scalar datasets shape ─────────────────────────────────────────
        def _check_scalars(grp: h5py.Group, prefix: str) -> None:
            nonlocal ok
            for key in grp.keys():
                item = grp[key]
                if isinstance(item, h5py.Dataset):
                    if item.shape[0] != n:
                        print(
                            f"  FAIL: {prefix}/{key} has {item.shape[0]} rows ≠ {n}"
                        )
                        ok = False
                    else:
                        print(f"  OK  : {prefix}/{key}  shape={item.shape}")
                elif isinstance(item, h5py.Group):
                    _check_scalars(item, f"{prefix}/{key}")

        for section in ("commanded_pose", "commanded_qpos", "state"):
            if section in ep:
                _check_scalars(ep[section], section)

        if ok:
            print("  -> All checks passed.")
        else:
            print("  -> Some checks FAILED.")

    return ok


def visualise_episode_rerun(path: Path, episode_idx: int = 0) -> None:
    """Write an episode to a .rrd file and open it in the Rerun viewer.

    Uses rr.save() rather than spawn=True to avoid the gRPC race condition
    where data is logged before the viewer process is ready to receive it.

    The .rrd file is written next to the HDF5 file, then opened with
    ``rerun <file>``.  The viewer scrub bar shows both a frame-sequence
    timeline and a real-seconds timeline.
    """
    try:
        import rerun as rr
    except ImportError:
        print("ERROR: rerun-sdk not installed.  Run: pip install rerun-sdk")
        return

    import subprocess

    path = Path(path)
    rrd_path = path.with_suffix("") / f"episode_{episode_idx:06d}.rrd"
    rrd_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, "r") as f:
        ep_keys = sorted(k for k in f.keys() if k.startswith("episode_"))
        if episode_idx >= len(ep_keys):
            print(f"ERROR: episode index {episode_idx} out of range "
                  f"(have {len(ep_keys)})")
            return

        ep_key = ep_keys[episode_idx]
        ep     = f[ep_key]
        ts     = ep["timestamps"][:]
        n      = len(ts)

        print(f"Writing {ep_key} ({n} frames) → {rrd_path} …")

        rr.init("hdf5_viewer", recording_id=ep_key)
        rr.save(str(rrd_path))

        t0 = int(ts[0])

        for i in range(n):
            t_ns = int(ts[i])
            rr.set_time_sequence("frame", i)
            rr.set_time_seconds("timestamp", (t_ns - t0) / 1e9)

            # Cameras.
            if "cameras" in ep:
                for cam_name in ep["cameras"].keys():
                    frame = ep["cameras"][cam_name][i]
                    rr.log(f"cameras/{cam_name}", rr.Image(frame))

            # Scalars.
            def _log_scalars(grp: h5py.Group, prefix: str, idx: int) -> None:
                for key in grp.keys():
                    item = grp[key]
                    if isinstance(item, h5py.Dataset):
                        vals = item[idx]
                        for j, v in enumerate(np.asarray(vals).flat):
                            rr.log(f"{prefix}/{key}/{j}",
                                   rr.Scalar(float(v)))
                    elif isinstance(item, h5py.Group):
                        _log_scalars(item, f"{prefix}/{key}", idx)

            for section in ("commanded_pose", "commanded_qpos", "state"):
                if section in ep:
                    _log_scalars(ep[section], section, i)

    print(f"Saved.  Opening viewer …")
    subprocess.Popen(["rerun", str(rrd_path)])


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect and verify HDF5 episode files."
    )
    parser.add_argument("path", type=Path, help="Path to .h5 file or directory")
    parser.add_argument(
        "--episode", type=int, default=None,
        help="Episode index to verify (default: all)"
    )
    parser.add_argument(
        "--rerun", action="store_true",
        help="Open episode in Rerun viewer after verification"
    )
    args = parser.parse_args()

    target = Path(args.path)

    if target.is_dir():
        h5_files = sorted(target.glob("*.h5"))
        if not h5_files:
            print(f"No .h5 files found in {target}")
            sys.exit(1)
        for p in h5_files:
            inspect_file(p)
    else:
        inspect_file(target)
        ep_idx = args.episode if args.episode is not None else 0
        ok = verify_episode(target, ep_idx)
        if args.rerun:
            visualise_episode_rerun(target, ep_idx)
        if not ok:
            sys.exit(1)


if __name__ == "__main__":
    main()
