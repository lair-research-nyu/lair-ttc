"""
config.py  –  Pipeline configuration.

Edit the CONFIG block at the bottom, or import Config and construct
one programmatically.  Every downstream module imports Config from here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


# ─────────────────────────────── data types ───────────────────────────────────

@dataclass
class SyncConfig:
    """Controls how streams are aligned to a common timeline."""

    # Output frame rate for the HDF5 dataset.
    # All streams are resampled to this rate using causal ZOH
    # (last-known-value – no future data is ever used).
    target_fps: float = 30.0

    # Maximum allowed age (ns) of a non-master sample at any target step.
    # Steps where any enabled stream has no sample within this window are dropped.
    # Default: 50 ms = 50_000_000 ns
    max_sync_gap_ns: int = 50_000_000


@dataclass
class CameraConfig:
    """Camera stream settings."""

    # The camera whose (downsampled) timestamps define the master timeline.
    # Must appear in streams.cameras.
    master_stream: str = "/cameras/head"

    # Streams that are side-by-side stereo images.
    # Each stereo stream is split into <name>_left / <name>_right in HDF5.
    stereo_streams: list[str] = field(
        default_factory=lambda: ["/cameras/head"]
    )

    # Which stereo halves to keep.  One of "left", "right", "both".
    stereo_side: str = "both"

    # Resize factor applied uniformly to all camera images.
    # 0.5 = half width & height.  None / 1.0 = keep original.
    image_scale_factor: float | None = 0.5

    # HDF5 compression for camera datasets.
    compression: str = "gzip"
    compression_opts: int = 4


@dataclass
class StreamConfig:
    """Which Rerun entity paths to extract."""

    cameras: list[str] = field(
        default_factory=lambda: [
            "/cameras/head",
            "/cameras/left_wrist",
            "/cameras/right_wrist",
        ]
    )

    # Each pose entity → two datasets: translation (N,3) and quaternion (N,4).
    poses: list[str] = field(
        default_factory=lambda: [
            "/commanded_pose/left_arm",
            "/commanded_pose/neck",
            "/commanded_pose/right_arm",
        ]
    )

    # Each scalar entity → one dataset of shape (N, D).
    scalars: list[str] = field(
        default_factory=lambda: [
            "/commanded_qpos/chest",
            "/commanded_qpos/left_arm",
            "/commanded_qpos/left_arm_ee",
            "/commanded_qpos/neck",
            "/commanded_qpos/right_arm",
            "/commanded_qpos/right_arm_ee",
            "/state/chest",
            "/state/left_arm",
            "/state/neck",
            "/state/right_arm",
            "/teleop_data_sequence",
        ]
    )

    # Workflow fields to read from the paired workflow .rrd.
    # These become per-episode HDF5 attributes.
    workflow_fields: list[str] = field(
        default_factory=lambda: [
            "start_time",
            "end_time",
            "duration_ms",
            "duration_log_type",
            "event_kind",
            "workflow_type",
            "workflow_status",
            "exit_type",
            "teleoperator_email",
            "station_slug",
        ]
    )


@dataclass
class Config:
    """Top-level pipeline configuration."""

    # ── Paths ─────────────────────────────────────────────────────────────────
    # Directory containing data .rrd files.  Required – no default.
    data_dir: Path = field(default=None)  # type: ignore[assignment]

    # Directory containing workflow .rrd files.  Required – no default.
    workflow_dir: Path = field(default=None)  # type: ignore[assignment]

    # Where HDF5 episodes are written.  Created if absent.
    output_dir: Path = field(default_factory=lambda: Path("output/hdf5"))

    # ── Behaviour ─────────────────────────────────────────────────────────────
    # Skip files whose HDF5 output already exists.
    skip_existing: bool = True

    # Maximum parallel worker processes for conversion.
    # None = one worker per episode (auto).  1 = serial.
    max_workers: int | None = 1

    # ── Sub-configs ───────────────────────────────────────────────────────────
    sync: SyncConfig = field(default_factory=SyncConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    streams: StreamConfig = field(default_factory=StreamConfig)

    def __post_init__(self) -> None:
        if self.data_dir is None:
            raise ValueError("Config.data_dir is required.")
        if self.workflow_dir is None:
            raise ValueError("Config.workflow_dir is required.")
        self.data_dir     = Path(self.data_dir)
        self.workflow_dir = Path(self.workflow_dir)
        self.output_dir   = Path(self.output_dir)

    # ── Helpers ───────────────────────────────────────────────────────────────
    @property
    def master_stream(self) -> str:
        return self.camera.master_stream

    @property
    def target_fps(self) -> float:
        return self.sync.target_fps
