"""
annotate.py  –  Interactive annotation of HDF5 episodes using the Rerun viewer.

Workflow
────────
1. Open an HDF5 episode file in the Rerun viewer (spawned as a subprocess).
2. The operator watches the episode play back and uses keyboard shortcuts to
   mark the episode as SUCCESS / FAILURE / SKIP.
3. The annotation is written back to the HDF5 file as:
       episode_<N>/attrs["annotation"] = "success" | "failure" | "skip"
       episode_<N>/attrs["annotated_by"] = os.getlogin()
       episode_<N>/attrs["annotated_at_ns"] = current time (ns)
4. A summary CSV is written alongside the HDF5 directory for easy review.

─────────────────────────────────────────────────────────────────────────────
STATUS: STUB – annotation callback API not yet implemented.
        The Rerun "callbacks" / Blueprint event API landed in rerun-sdk 0.22+.
        See: https://rerun.io/docs/howto/visualization/callbacks

        Until the callback API is wired up, the stub below:
          • streams the episode to the viewer (same as validate.visualise_episode_rerun)
          • prompts the operator in the terminal for annotation
          • writes the result back to HDF5
─────────────────────────────────────────────────────────────────────────────

Usage:
    python -m data_processing.annotate output/hdf5/data1.h5
    python -m data_processing.annotate output/hdf5/  --all
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import h5py


# ──────────────────────────────────────────────────────────────────────────────

_VALID_LABELS = ("success", "failure", "skip")


def annotate_episode(h5_path: Path, episode_idx: int = 0) -> str:
    """Stream episode to Rerun viewer, prompt operator, write annotation.

    Returns the label chosen by the operator.
    """
    from data_processing.validate import visualise_episode_rerun

    visualise_episode_rerun(h5_path, episode_idx)

    print(f"\nAnnotate episode {episode_idx} in {h5_path.name}")
    print("  Labels: success (s) | failure (f) | skip (k)")

    label = ""
    while label not in _VALID_LABELS:
        raw = input("  Enter label [s/f/k]: ").strip().lower()
        label = {"s": "success", "f": "failure", "k": "skip"}.get(raw, raw)
        if label not in _VALID_LABELS:
            print(f"  Invalid label '{raw}'.  Try s, f, or k.")

    _write_annotation(h5_path, episode_idx, label)
    print(f"  Saved annotation: {label}")
    return label


def _write_annotation(h5_path: Path, episode_idx: int, label: str) -> None:
    ep_keys = _get_episode_keys(h5_path)
    if episode_idx >= len(ep_keys):
        raise IndexError(f"Episode {episode_idx} not in {h5_path}")

    with h5py.File(h5_path, "a") as f:
        ep = f[ep_keys[episode_idx]]
        ep.attrs["annotation"]       = label
        ep.attrs["annotated_by"]     = os.getlogin()
        ep.attrs["annotated_at_ns"]  = time.time_ns()


def _get_episode_keys(h5_path: Path) -> list[str]:
    with h5py.File(h5_path, "r") as f:
        return sorted(k for k in f.keys() if k.startswith("episode_"))


def annotate_all(h5_dir: Path) -> None:
    """Iterate over all HDF5 files and episodes, prompting for each."""
    h5_files = sorted(Path(h5_dir).glob("*.h5"))
    if not h5_files:
        print(f"No .h5 files found in {h5_dir}")
        return

    for h5_path in h5_files:
        keys = _get_episode_keys(h5_path)
        for ep_idx in range(len(keys)):
            # Skip already-annotated episodes.
            with h5py.File(h5_path, "r") as f:
                ep = f[keys[ep_idx]]
                if "annotation" in ep.attrs:
                    existing = ep.attrs["annotation"]
                    print(
                        f"  SKIP {h5_path.name} / {keys[ep_idx]}  "
                        f"(already annotated: {existing})"
                    )
                    continue

            annotate_episode(h5_path, ep_idx)


def export_annotation_csv(h5_dir: Path, csv_path: Path | None = None) -> Path:
    """Write a CSV summary of all annotations in *h5_dir*."""
    import csv

    csv_path = csv_path or Path(h5_dir) / "annotations.csv"
    rows = []

    for h5_path in sorted(Path(h5_dir).glob("*.h5")):
        with h5py.File(h5_path, "r") as f:
            for ep_key in sorted(k for k in f.keys() if k.startswith("episode_")):
                ep = f[ep_key]
                rows.append({
                    "file":         h5_path.name,
                    "episode":      ep_key,
                    "annotation":   ep.attrs.get("annotation",       ""),
                    "annotated_by": ep.attrs.get("annotated_by",     ""),
                    "n_frames":     ep.attrs.get("n_frames",          ""),
                    "duration_ms":  ep.attrs.get("duration_ms",       ""),
                    "task_type":    ep.attrs.get("workflow_type",      ""),
                    "exit_type":    ep.attrs.get("exit_type",          ""),
                })

    with open(csv_path, "w", newline="") as csvf:
        writer = csv.DictWriter(csvf, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} row(s) to {csv_path}")
    return csv_path


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Annotate HDF5 episodes as success / failure / skip."
    )
    parser.add_argument("path", type=Path,
                        help="Path to a single .h5 file or a directory")
    parser.add_argument("--episode", type=int, default=0,
                        help="Episode index (only for single-file mode)")
    parser.add_argument("--all", action="store_true",
                        help="Annotate all un-annotated episodes in directory")
    parser.add_argument("--export-csv", action="store_true",
                        help="Export annotation summary CSV and exit")
    args = parser.parse_args()

    target = Path(args.path)

    if args.export_csv:
        d = target if target.is_dir() else target.parent
        export_annotation_csv(d)
        return

    if target.is_dir():
        annotate_all(target)
    else:
        if args.all:
            keys = _get_episode_keys(target)
            for i in range(len(keys)):
                annotate_episode(target, i)
        else:
            annotate_episode(target, args.episode)


if __name__ == "__main__":
    main()
