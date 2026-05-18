# Electron Integration Contract

This document is the proposed boundary between an Electron shell and the DLB
Python backend. The current app is still PyQt-first; this contract is meant to
keep future integration from depending on GUI internals.

## Recommended Architecture

Electron should launch DLB as a local Python worker and communicate through a
stable JSON interface.

Recommended phases:

1. **CLI JSON commands** for simple synchronous jobs.
2. **Local HTTP or WebSocket service** for progress events and cancellation.
3. **Electron UI replaces PyQt** only after the backend contract is stable.

## Core Concepts

- `video_path`: source video file.
- `session_dir`: cache directory for frames and DINO features.
- `anchor_frame`: frame index for the user-provided annotation.
- `annotation_path`: LabelMe / X-AnyLabeling compatible JSON.
- `inference_range`: inclusive `[start_frame, end_frame]`.
- `output_dir`: destination folder for exported `jpg + json`.

## Proposed Commands

### prepare_video

```json
{
  "cmd": "prepare_video",
  "video_path": "C:/data/video.mp4",
  "session_dir": "tmp/video",
  "start_frame": 100,
  "end_frame": 160
}
```

Expected response:

```json
{
  "ok": true,
  "video": {
    "fps": 29.97,
    "width": 1920,
    "height": 1080,
    "total_frames": 1200
  },
  "cache": {
    "frames_ready": 61,
    "features_ready": 61
  }
}
```

### propagate

```json
{
  "cmd": "propagate",
  "video_path": "C:/data/video.mp4",
  "session_dir": "tmp/video",
  "annotation_path": "C:/data/frame_000120.json",
  "anchor_frame": 120,
  "inference_range": [90, 150]
}
```

Expected response:

```json
{
  "ok": true,
  "inference_range": [90, 150],
  "summary": {
    "annotated_frames": 61,
    "flagged_frames": 4,
    "boxes": 122
  },
  "audit": [
    {
      "frame_idx": 132,
      "label": "eye",
      "confidence": 0.42,
      "reason": "low_confidence"
    }
  ]
}
```

### export_annotations

Export always uses the full inference interval. Electron should not ask users
to pick export start/end frames separately. Frames without boxes should still
be exported with an empty `shapes: []` JSON so frame sequences stay continuous.

```json
{
  "cmd": "export_annotations",
  "session_dir": "tmp/video",
  "output_dir": "C:/data/export",
  "inference_range": [90, 150],
  "format": "labelme_compatible"
}
```

Expected response:

```json
{
  "ok": true,
  "output_dir": "C:/data/export",
  "exported_frames": 61,
  "skipped_frames": 0,
  "format": "labelme_compatible"
}
```

## Annotation Format

The interchange format should remain LabelMe-compatible JSON so both LabelMe
and X-AnyLabeling can read the files:

- `shape_type` is `rectangle`.
- `points` uses two corners: `[[x1, y1], [x2, y2]]`.
- AI confidence is stored in `description`, for example `conf:0.923`.
- `flags` remains `{}` to avoid tool-specific rendering issues.

## Progress Events

When moving to HTTP/WebSocket, emit progress events with this shape:

```json
{
  "event": "progress",
  "stage": "extract_features",
  "done": 20,
  "total": 61,
  "message": "Computing DINOv2 features"
}
```

Suggested stages:

- `load_video`
- `extract_frames`
- `load_model`
- `extract_features`
- `propagate`
- `write_annotations`
- `export`

## Error Shape

```json
{
  "ok": false,
  "error": {
    "code": "annotation_empty",
    "message": "The selected JSON has no rectangle annotations.",
    "details": "Optional traceback or debug text"
  }
}
```

The Electron UI should show `message` by default and keep `details` behind a
technical details disclosure.
