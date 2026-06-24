"""
rrd_loader.py  –  Discover and pair .rrd files; open Rerun catalog datasets.

Pairing strategy (in priority order):
  1. UUID match – filenames share the pattern  <prefix>_<uuid>.rrd, so data and
     workflow files with the same UUID are paired automatically.
  2. Index match – files are sorted alphanumerically and paired by position.
     Use this for the "renamedN.rrd" convention.

Usage:
    pairs = pair_rrd_files(data_dir, workflow_dir)
    for data_rrd, workflow_rrd in pairs:
        with RRDDatasetPair(data_rrd, workflow_rrd) as pair:
            data_ds, wf_ds = pair.data_ds, pair.workflow_ds
            ...
"""

from __future__ import annotations

import re
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import rerun as rr

# ──────────────────────────────────────────────────────────────────────────────

_UUID_RE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
    re.IGNORECASE,
)


def _extract_uuid(path: Path) -> str | None:
    """Return the first UUID found in the filename, or None."""
    m = _UUID_RE.search(path.stem)
    return m.group(0).lower() if m else None


def pair_rrd_files(
    data_dir: Path,
    workflow_dir: Path,
) -> list[tuple[Path, Path]]:
    """Return a list of (data_rrd, workflow_rrd) pairs.

    Raises ValueError if the two directories cannot be paired.
    """
    data_files = sorted(Path(data_dir).glob("*.rrd"))
    wf_files   = sorted(Path(workflow_dir).glob("*.rrd"))

    if not data_files:
        raise ValueError(f"No .rrd files found in data_dir: {data_dir}")
    if not wf_files:
        raise ValueError(f"No .rrd files found in workflow_dir: {workflow_dir}")

    # ── Try UUID-based pairing ─────────────────────────────────────────────
    data_uuid_map = {_extract_uuid(f): f for f in data_files if _extract_uuid(f)}
    wf_uuid_map   = {_extract_uuid(f): f for f in wf_files   if _extract_uuid(f)}

    shared = data_uuid_map.keys() & wf_uuid_map.keys()
    if shared:
        # Preserve sorted(data_files) order so episodes are processed
        # chronologically (timestamp is the prefix, not the UUID suffix).
        pairs = [
            (data_uuid_map[u], wf_uuid_map[u])
            for f in data_files
            if (u := _extract_uuid(f)) in shared
        ]
        unpaired_data = [f for f in data_files if _extract_uuid(f) not in shared]
        unpaired_wf   = [f for f in wf_files   if _extract_uuid(f) not in shared]
        if unpaired_data or unpaired_wf:
            print(
                f"  WARNING: {len(unpaired_data)} data file(s) and "
                f"{len(unpaired_wf)} workflow file(s) have no UUID match; "
                "they will be skipped."
            )
        return pairs

    # ── Fall back to index-based pairing ──────────────────────────────────
    if len(data_files) != len(wf_files):
        raise ValueError(
            f"Cannot pair by index: {len(data_files)} data file(s) vs "
            f"{len(wf_files)} workflow file(s).  "
            "Ensure filenames share a UUID or that both directories have the "
            "same number of files."
        )
    print(
        "  INFO: No UUID found in filenames - pairing data/workflow files "
        "by sorted index."
    )
    return list(zip(data_files, wf_files))


# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class RRDDatasetPair:
    """Context manager that opens a Rerun catalog server for one episode pair.

    Usage::

        with RRDDatasetPair(data_rrd, workflow_rrd) as p:
            segment_ids = p.data_ds.segment_ids()
            wf_events   = p.workflow_ds.reader(...).to_pandas()
    """

    data_rrd: Path
    workflow_rrd: Path | None = None  # None = no workflow file for this episode

    _server: rr.server.Server | None = None

    def __enter__(self) -> "RRDDatasetPair":
        datasets: dict[str, list[Path]] = {"data": [self.data_rrd]}
        if self.workflow_rrd is not None:
            datasets["workflow"] = [self.workflow_rrd]

        self._server = rr.server.Server(datasets=datasets)
        client = self._server.client()

        self.data_ds = client.get_dataset("data")
        self.workflow_ds = (
            client.get_dataset("workflow") if self.workflow_rrd is not None else None
        )
        return self

    def __exit__(self, *_) -> None:
        # rr.server.Server has no explicit close(); the Python GC handles it.
        self._server = None

    # Convenience: name of the source file (stem without extension).
    @property
    def episode_source(self) -> str:
        return self.data_rrd.stem


def open_dataset_dir(data_dir: Path) -> tuple[rr.server.Server, object]:
    """Open all .rrd files in *data_dir* as a single multi-segment dataset.

    Returns (server, dataset).  The caller is responsible for keeping a
    reference to *server* for the lifetime of *dataset*.
    """
    server = rr.server.Server(datasets={"data": str(data_dir)})
    client = server.client()
    return server, client.get_dataset("data")
