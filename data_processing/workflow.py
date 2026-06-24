"""
workflow.py  –  Extract episode metadata from paired workflow .rrd files.

Each workflow .rrd contains one or more /workflow_duration_logs/events rows.
A single row typically describes one full episode attempt with fields such as:

    start_time, end_time, duration_ms, event_kind, workflow_type,
    workflow_status, exit_type, teleoperator_email, station_slug, …

Usage::

    with RRDDatasetPair(data_rrd, workflow_rrd) as pair:
        events = load_workflow_events(pair.workflow_ds, fields=WF_FIELDS)
        meta   = best_matching_event(events, data_start_ns, data_end_ns)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Entity path used in every workflow .rrd observed so far.
_WF_ENTITY = "/workflow_duration_logs/events"

# The full list of fields present in the workflow schema.
ALL_WORKFLOW_FIELDS: list[str] = [
    "start_time",
    "end_time",
    "duration_ms",
    "duration_log_type",
    "event_kind",
    "workflow_type",
    "workflow_status",
    "exit_type",
    "teleoperator_email",
    "teleoperator_name",
    "teleoperator_id",
    "station_slug",
    "station_display_name",
    "organization_id",
    "organization_display_name",
    "workflow_id",
    "workflow_duration_log_id",
    "workflow_created_by_email",
    "workflow_created_by_name",
    "workflow_created_by_id",
    "start_metadata_json",
    "end_metadata_json",
]


# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class WorkflowEvent:
    """Metadata for one episode attempt."""
    start_ns: int   = 0
    end_ns: int     = 0
    duration_ms: float = 0.0
    event_kind: str = ""
    workflow_type: str = ""
    workflow_status: str = ""
    exit_type: str = ""
    log_type: str = ""   # "generic_subtask" or "idle"
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_s(self) -> float:
        return self.duration_ms / 1000.0

    def as_hdf5_attrs(self) -> dict[str, Any]:
        """Return a flat dict suitable for writing to an HDF5 group's attrs."""
        attrs: dict[str, Any] = {
            "start_ns":       self.start_ns,
            "end_ns":         self.end_ns,
            "duration_ms":    self.duration_ms,
            "event_kind":     self.event_kind,
            "workflow_type":  self.workflow_type,
            "workflow_status": self.workflow_status,
            "exit_type":      self.exit_type,
            "log_type":       self.log_type,
        }
        for k, v in self.extra.items():
            if v is None:
                continue
            if isinstance(v, (str, int, float, bool)):
                attrs[k] = v
            else:
                attrs[k] = str(v)
        return attrs


_EMPTY_EVENT = WorkflowEvent()


# ──────────────────────────────────────────────────────────────────────────────

def _parse_timestamp_ns(value: Any) -> int:
    """Best-effort conversion of a workflow timestamp to int nanoseconds."""
    if value is None:
        return 0
    if isinstance(value, (int, float)):
        return int(value)
    try:
        return int(pd.Timestamp(value).value)
    except Exception:
        return 0


def load_workflow_events(
    wf_ds,
    fields: list[str] | None = None,
) -> list[WorkflowEvent]:
    """Read workflow events from a Rerun dataset.

    Parameters
    ----------
    wf_ds  : Rerun DatasetEntry for the workflow dataset.
    fields : subset of ALL_WORKFLOW_FIELDS to fetch (None = all).

    Returns
    -------
    List of WorkflowEvent objects, one per row in the workflow log.
    Empty list if the dataset has no workflow data.
    """
    if wf_ds is None:
        return []

    fetch = fields or ALL_WORKFLOW_FIELDS
    col_map = {f: f"{_WF_ENTITY}:{f}" for f in fetch}

    # Try each timeline in order; different workflow files use different ones.
    _TIMELINE_CANDIDATES = ["log_time", "ts", "log_tick", "wall_time", "time"]
    df = None
    for timeline in _TIMELINE_CANDIDATES:
        try:
            df = (
                wf_ds.reader(index=timeline)
                .select(*col_map.values(), "ts")
                .to_pandas()
            )
            break
        except Exception:
            continue

    # Last resort: discover available timelines from the dataset schema.
    if df is None:
        try:
            schema = wf_ds.schema()
            idx_cols = schema.index_columns()
            for col in idx_cols:
                # IndexColumnDescriptor may expose name via .name or str()
                timeline = getattr(col, "name", None) or str(col)
                if timeline in _TIMELINE_CANDIDATES:
                    continue
                try:
                    df = (
                        wf_ds.reader(index=timeline)
                        .select(*col_map.values(), "ts")
                        .to_pandas()
                    )
                    print(f"  INFO: workflow read via discovered timeline '{timeline}'")
                    break
                except Exception:
                    continue
        except Exception:
            pass

    if df is None:
        print("  WARNING: could not read workflow dataset with any known timeline index")
        return []

    if df.empty:
        return []

    events: list[WorkflowEvent] = []
    for _, row in df.iterrows():
        def _get(field_name: str) -> Any:
            col = col_map.get(field_name)
            if col is None or col not in row or row[col] is None:
                return None
            val = row[col]
            # Rerun scalar fields come back as 1-element arrays.
            if hasattr(val, "__len__") and not isinstance(val, str):
                try:
                    val = val[0]
                except (IndexError, TypeError):
                    pass
            return val

        start_ns = _parse_timestamp_ns(_get("start_time"))
        end_ns   = _parse_timestamp_ns(_get("end_time"))

        extra_keys = [
            f for f in fetch
            if f not in ("start_time", "end_time", "duration_ms",
                         "duration_log_type",
                         "event_kind", "workflow_type", "workflow_status",
                         "exit_type")
        ]

        events.append(WorkflowEvent(
            start_ns     = start_ns,
            end_ns       = end_ns,
            duration_ms  = float(_get("duration_ms") or 0.0),
            event_kind   = str(_get("event_kind")   or ""),
            workflow_type= str(_get("workflow_type") or ""),
            workflow_status= str(_get("workflow_status") or ""),
            exit_type    = str(_get("exit_type")    or ""),
            log_type     = str(_get("duration_log_type") or ""),
            extra        = {k: _get(k) for k in extra_keys},
        ))

    # Rerun logs each workflow event across multiple ticks (e.g. one row when
    # start_time is written, another when end_time is written).  The reader
    # forward-fills all columns, so both rows appear identical.  Deduplicate
    # by (start_ns, end_ns), keeping only complete events (both non-zero).
    seen: set[tuple[int, int]] = set()
    unique: list[WorkflowEvent] = []
    for ev in events:
        if ev.start_ns == 0 or ev.end_ns == 0:
            continue   # incomplete row – end_time not yet logged at this tick
        key = (ev.start_ns, ev.end_ns)
        if key not in seen:
            seen.add(key)
            unique.append(ev)

    return unique


def best_matching_event(
    events: list[WorkflowEvent],
    data_start_ns: int,
    data_end_ns:   int,
) -> WorkflowEvent:
    """Return the workflow event with the greatest temporal overlap with the
    data segment [data_start_ns, data_end_ns].

    Falls back to the first event if no overlap exists (time-zone or clock
    offset between data and workflow recorders).
    """
    if not events:
        return _EMPTY_EVENT

    best, best_overlap = events[0], -1

    for ev in events:
        overlap_start = max(ev.start_ns, data_start_ns)
        overlap_end   = min(ev.end_ns,   data_end_ns)
        overlap       = max(0, overlap_end - overlap_start)
        if overlap > best_overlap:
            best, best_overlap = ev, overlap

    return best


def load_generic_subtask_events(
    wf_ds,
    fields: list[str] | None = None,
) -> list[WorkflowEvent]:
    """Return only the ``generic_subtask`` events from a workflow dataset.

    These define individual robot episodes: the episode's time window is
    ``[event.start_ns, event.end_ns]``.

    Parameters
    ----------
    wf_ds  : Rerun DatasetEntry for the workflow dataset (may be None).
    fields : subset of ALL_WORKFLOW_FIELDS to fetch (None = all).

    Returns
    -------
    List of WorkflowEvent objects with log_type == "generic_subtask".
    """
    fetch = list(fields) if fields else list(ALL_WORKFLOW_FIELDS)
    # duration_log_type is required for filtering; ensure it is always fetched.
    if "duration_log_type" not in fetch:
        fetch = ["duration_log_type"] + fetch

    all_events = load_workflow_events(wf_ds, fetch)
    return [e for e in all_events if e.log_type.lower() == "generic_subtask"]
