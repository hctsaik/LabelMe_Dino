"""Background video jobs. Reuses src.video_core / dino_engine / propagator, never PyQt."""

from __future__ import annotations

import json
import os
import sys
import threading
import traceback
import uuid
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

JOBS: dict[str, dict[str, Any]] = {}
LOCK = threading.Lock()
_DINO = None
_MODEL_READY = False
_MODEL_ERROR = ""


def model_status() -> dict[str, Any]:
    return {"model_ready": _MODEL_READY, "model_error": _MODEL_ERROR or None}


def get_job(job_id: str) -> dict[str, Any] | None:
    with LOCK:
        rec = JOBS.get(job_id)
        return dict(rec) if rec else None


def _update(job_id: str, **fields: Any) -> None:
    with LOCK:
        if job_id in JOBS:
            JOBS[job_id].update(fields)


def _start(kind: str, fn: Callable[[str], None]) -> str:
    job_id = uuid.uuid4().hex[:12]
    with LOCK:
        JOBS[job_id] = {
            "id": job_id,
            "kind": kind,
            "status": "queued",
            "progress": {"done": 0, "total": 0},
            "error": None,
            "result": None,
        }

    def run() -> None:
        _update(job_id, status="running")
        try:
            fn(job_id)
            with LOCK:
                if JOBS[job_id]["status"] != "failed":
                    JOBS[job_id]["status"] = "succeeded"
        except Exception as exc:  # noqa: BLE001
            _update(job_id, status="failed", error=str(exc)[:800])
            traceback.print_exc()

    threading.Thread(target=run, daemon=True).start()
    return job_id


def _dino():
    global _DINO, _MODEL_READY, _MODEL_ERROR
    if _DINO is not None:
        return _DINO
    from src.dino_engine import DinoEngine

    _DINO = DinoEngine()
    _MODEL_READY = True
    _MODEL_ERROR = ""
    return _DINO


def start_prepare(video_path: str, start_sec: float, end_sec: float, session_dir: Path) -> str:
    video_path = str(Path(video_path))
    if not Path(video_path).is_file():
        raise FileNotFoundError(video_path)
    if end_sec <= start_sec:
        raise ValueError("end_sec must be > start_sec")

    def work(job_id: str) -> None:
        from src.video_core import VideoCore

        vc = VideoCore(session_dir)
        meta = vc.load(video_path)
        end = min(float(end_sec), meta.duration)
        start = max(0.0, float(start_sec))

        def on_frames(done: int, total: int) -> None:
            _update(job_id, progress={"done": done, "total": total, "stage": "extract_frames"})

        paths = vc.extract_frames(start, end, progress_callback=on_frames)
        engine = None
        if os.environ.get("LABEL_WEB_SKIP_DINO", "").strip() in {"1", "true"}:
            pass
        else:
            try:
                engine = _dino()
            except Exception as exc:  # noqa: BLE001
                global _MODEL_ERROR
                _MODEL_ERROR = str(exc)
        if engine is not None:
            for i, path in enumerate(paths):
                idx = int(path.stem.split("_")[1])
                feat = vc.feature_path(idx)
                if not feat.exists():
                    frame = vc.read_frame(idx)
                    engine.extract_and_save(frame, feat)
                _update(
                    job_id,
                    progress={"done": i + 1, "total": len(paths), "stage": "dino_features"},
                )
        _update(
            job_id,
            result={
                "session_dir": str(session_dir),
                "frames": len(paths),
                "fps": meta.fps,
                "duration": meta.duration,
                "features": engine is not None,
            },
        )

    return _start("prepare_video", work)


def start_propagate(
    *,
    session_dir: Path,
    anchor_frame: int,
    boxes: list[dict[str, Any]],
    radius_seconds: float,
    classes: list[str],
) -> str:
    def work(job_id: str) -> None:
        from src.class_manager import ClassManager
        from src.propagator import LabeledBox, Propagator
        from src.video_core import VideoCore

        vc = VideoCore(session_dir)
        vc.load_image_sequence(session_dir / "frames")
        engine = _dino()
        cm = ClassManager()
        for i, name in enumerate(classes or ["object"]):
            try:
                cm.add(name, i)
            except ValueError:
                pass
        prop = Propagator(vc, engine, cm, radius_seconds=radius_seconds)
        labeled = []
        for box in boxes:
            pts = box.get("points") or []
            if len(pts) < 2:
                continue
            xs = [float(p[0]) for p in pts]
            ys = [float(p[1]) for p in pts]
            labeled.append(
                LabeledBox(
                    label=str(box.get("label") or "object"),
                    bbox=(min(xs), min(ys), max(xs), max(ys)),
                    confidence=1.0,
                    ai_generated=False,
                )
            )
        if not labeled:
            raise ValueError("anchor has no rectangles")
        prop.set_anchor(int(anchor_frame), labeled)

        def on_prog(done: int, total: int) -> None:
            _update(job_id, progress={"done": done, "total": total, "stage": "propagate"})

        audits = prop.propagate_from_anchor(
            int(anchor_frame),
            forward=True,
            backward=True,
            radius_seconds=radius_seconds,
            progress_callback=on_prog,
        )
        written = []
        for idx, ann in prop.annotations.items():
            dest = vc.label_path(idx)
            shapes = []
            for box in ann.boxes:
                x1, y1, x2, y2 = box.bbox
                item = {
                    "label": box.label,
                    "points": [[float(x1), float(y1)], [float(x2), float(y2)]],
                    "shape_type": "rectangle",
                    "flags": {},
                }
                if box.ai_generated:
                    item["score"] = box.confidence
                shapes.append(item)
            dest.write_text(
                json.dumps(
                    {
                        "version": "5.0.1",
                        "flags": {},
                        "shapes": shapes,
                        "imagePath": dest.with_suffix(".jpg").name,
                        "imageData": None,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            written.append(idx)
        _update(
            job_id,
            result={
                "written": written,
                "flagged": [
                    {"frame": a.frame_idx, "label": a.label, "reason": a.reason, "confidence": a.confidence}
                    for a in audits
                ],
            },
        )

    return _start("propagate", work)
