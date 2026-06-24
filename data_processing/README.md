# data_processing – RRD → HDF5 Pipeline

Converts paired Rerun `.rrd` recording files into a single `dataset.h5` ready
for training.  All 7 (or however many) recording pairs together form **one
dataset**; episodes are numbered sequentially across all pairs.

---

## Table of contents

1. [Environment setup](#1-environment-setup)
2. [Data layout](#2-data-layout)
3. [Episode definition](#3-episode-definition)
4. [Running the conversion](#4-running-the-conversion)
5. [HDF5 output structure](#5-hdf5-output-structure)
6. [Validation](#6-validation)
7. [Pipeline modules](#7-pipeline-modules)
8. [Phase 2 – training formats (planned)](#8-phase-2--training-formats-planned)

---

## 1. Environment setup

Use the `hf-env` conda environment (Python 3.12, rerun-sdk 0.31.1).

```bash
# Create from the spec file
conda env create -f data_processing/environment.yaml
conda activate hf-env

# Or pip-only
pip install -r data_processing/requirements.txt
```

---

## 2. Data layout

Each recording session produces a pair of `.rrd` files:

| File | Contents |
|------|----------|
| `data/<session>/GEN2-015-<ts>_<uuid>.rrd` | Camera frames (H.264), robot poses, joint angles, teleop data |
| `workflow/<session>/GEN2-015-<ts>_<uuid>.rrd` | Workflow event log – episode start/end times, operator info, success/failure |

Files are matched by shared UUID in the filename.  Seven pairs = one dataset.

### Data streams inside a data `.rrd`

| Stream | Type | Rate |
|--------|------|------|
| `/cameras/head` | stereo video (H.264, side-by-side) | 60 Hz |
| `/cameras/left_wrist` | video (H.264) | 30 Hz |
| `/cameras/right_wrist` | video (H.264) | 30 Hz |
| `/commanded_pose/{left_arm,neck,right_arm}` | Transform3D (xyz + quaternion) | 50 Hz |
| `/commanded_qpos/{chest,left_arm,left_arm_ee,neck,right_arm,right_arm_ee}` | Scalars | 500 Hz |
| `/state/{chest,left_arm,neck,right_arm}` | Scalars | 500 Hz |
| `/teleop_data_sequence` | Scalars | 500 Hz |

### Head camera – stereo split

`/cameras/head` encodes a **side-by-side** stereo frame `[LEFT | RIGHT]`.
The pipeline splits this into `head_left` and `head_right` datasets in HDF5.
Use `--stereo-side left|right|both` to control which halves are stored.

---

## 3. Episode definition

Episodes come from the **workflow file**, not from Rerun segments.

The workflow log (`/workflow_duration_logs/events`) contains alternating rows:

```
generic_subtask   <- robot is executing a task  [EPISODE]
idle              <- robot is idle between tasks [DISCARDED]
generic_subtask
idle
...
```

Each `generic_subtask` row defines **one episode**.  The fields used are:

| Field | Meaning |
|-------|---------|
| `start_time` | Episode start (nanoseconds since epoch) |
| `end_time` | Episode end (nanoseconds since epoch) |
| `duration_ms` | Duration in milliseconds |
| `workflow_status` | `"success"` / `"failure"` / etc. |
| `exit_type` | How the episode ended |
| `teleoperator_email` | Operator who collected the episode |

All data streams are **windowed to `[start_time, end_time]`** before
processing.

---

## 4. Running the conversion

### Delete previous (incorrect) output first

A previous run produced 7 per-file HDF5 files using the wrong episode
definition (Rerun segments, not workflow events).  Delete them before
re-running:

```bash
rm output/hdf5/*.h5
```

### Run the conversion

```bash
conda activate hf-env

python -m data_processing.rrd_to_hdf5 \
    --data-dir     data/04-06-26/data \
    --workflow-dir data/04-06-26/workflow \
    --output-dir   output/hdf5 \
    --fps          30 \
    --stereo-side  both \
    --image-scale  0.5
```

**Output:** `output/hdf5/dataset.h5`

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--data-dir` | required | Directory containing data `.rrd` files |
| `--workflow-dir` | required | Directory containing workflow `.rrd` files |
| `--output-dir` | `output/hdf5` | Where `dataset.h5` is written |
| `--fps` | `30` | Target output frame rate (Hz) |
| `--max-gap-ms` | `50` | Max age of a causal sample before step is dropped (ms) |
| `--stereo-side` | `both` | Which stereo halves to store: `left`, `right`, `both` |
| `--image-scale` | `0.5` | Resize factor for all cameras (1.0 = full resolution) |
| `--no-skip` | off | Re-convert even if `dataset.h5` already exists |

### Temporal alignment (causal ZOH)

All streams are aligned to the master camera (`/cameras/head`) using
**causal zero-order hold** (last-known-value at or before each target tick).
Future data is never used.  Steps where any stream has no sample within
`--max-gap-ms` are dropped.

---

## 5. HDF5 output structure

```
output/hdf5/dataset.h5
├── attrs
│   ├── pipeline_version   "0.2.0"
│   └── n_episodes         <int>
│
├── episode_000000/
│   ├── attrs
│   │   ├── source_file        "GEN2-015-..._<uuid>.rrd"
│   │   ├── n_frames           <int>
│   │   ├── target_fps         30.0
│   │   ├── start_ns           <int>   first master timestamp
│   │   ├── end_ns             <int>   last  master timestamp
│   │   ├── duration_ms        <float> from workflow
│   │   ├── workflow_status    "success" / "failure" / ...
│   │   ├── exit_type          <str>
│   │   ├── log_type           "generic_subtask"
│   │   └── teleoperator_email <str>
│   │
│   ├── timestamps            (N,)       int64   nanoseconds
│   ├── cameras/
│   │   ├── head_left         (N, H, W, 3) uint8
│   │   ├── head_right        (N, H, W, 3) uint8
│   │   ├── left_wrist        (N, H, W, 3) uint8
│   │   └── right_wrist       (N, H, W, 3) uint8
│   ├── commanded_pose/
│   │   ├── left_arm/
│   │   │   ├── translation   (N, 3)    float32
│   │   │   └── quaternion    (N, 4)    float32
│   │   ├── neck/  ...
│   │   └── right_arm/  ...
│   ├── commanded_qpos/
│   │   ├── chest             (N, D)    float32
│   │   ├── left_arm          ...
│   │   └── ...
│   ├── state/
│   │   ├── chest  ...
│   │   └── ...
│   └── teleop_data_sequence  (N, D)    float32
│
├── episode_000001/  ...
└── ...
```

All camera datasets use gzip compression (level 4), chunked at 1 frame.

---

## 6. Validation

```bash
# Inspect file structure and episode summary
python -m data_processing.validate output/hdf5/dataset.h5

# Verify a specific episode
python -m data_processing.validate output/hdf5/dataset.h5 --episode 3

# Stream episode 0 to the Rerun viewer
python -m data_processing.validate output/hdf5/dataset.h5 --episode 0 --rerun
```

---

## 7. Pipeline modules

| Module | Purpose |
|--------|---------|
| `config.py` | `Config` dataclass – all tunable parameters |
| `rrd_loader.py` | File pairing (UUID match → index fallback); `RRDDatasetPair` context manager |
| `workflow.py` | Load workflow events; `load_generic_subtask_events()` filters to episodes |
| `extract.py` | Time-windowed extraction of cameras, poses, scalars from a Rerun dataset |
| `sync.py` | Causal ZOH alignment; temporal downsampling |
| `rrd_to_hdf5.py` | Main conversion pipeline (entry point) |
| `validate.py` | Inspection, integrity checks, Rerun visualisation |
| `hdf5_to_lerobot.py` | **(Phase 2, stub)** HDF5 → LeRobot v2 format |
| `hdf5_to_webdataset.py` | **(Phase 2, stub)** HDF5 → WebDataset tar shards |
| `annotate.py` | **(Phase 2, stub)** Terminal tool to annotate episodes success/failure |

---

## 8. Phase 2 – training formats (planned)

### LeRobot

```bash
python -m data_processing.hdf5_to_lerobot \
    --input  output/hdf5/dataset.h5 \
    --output output/lerobot/
```

### WebDataset

```bash
python -m data_processing.hdf5_to_webdataset \
    --input     output/hdf5/dataset.h5 \
    --output    output/webdataset/ \
    --shard-size 1000
```

These modules are stubs – see the source files for design notes.
