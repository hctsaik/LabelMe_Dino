from __future__ import annotations

import json
import re
import subprocess
import threading
from pathlib import Path
from typing import Callable, Optional

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from PyQt5.QtCore import QObject, pyqtSignal

from src.propagator import LabeledBox


_CONF_RE = re.compile(r"conf(?:idence)?\s*[=:]\s*([0-9]*\.?[0-9]+)", re.IGNORECASE)


class _DebounceWatcher(FileSystemEventHandler):
    """Fires callback after debounce_ms of silence following any create/modify/move."""

    def __init__(self, watch_path: Path, callback: Callable, debounce_ms: int):
        self._path = watch_path.resolve()
        self._callback = callback
        self._debounce = debounce_ms / 1000.0
        self._timer: Optional[threading.Timer] = None
        self._lock = threading.Lock()

    def _matches(self, path: str) -> bool:
        return Path(path).resolve() == self._path

    def _schedule(self):
        with self._lock:
            if self._timer:
                self._timer.cancel()
            self._timer = threading.Timer(self._debounce, self._fire)
            self._timer.start()

    def on_modified(self, event):
        if not event.is_directory and self._matches(event.src_path):
            self._schedule()

    def on_created(self, event):
        if not event.is_directory and self._matches(event.src_path):
            self._schedule()

    def on_moved(self, event):
        # Atomic write: editor writes to temp file then renames to target
        if not event.is_directory and self._matches(event.dest_path):
            self._schedule()

    def _fire(self):
        with self._lock:
            self._timer = None
        self._callback()

    def cancel(self):
        with self._lock:
            if self._timer:
                self._timer.cancel()
                self._timer = None


class LabelBridge(QObject):
    """
    Manages LabelMe subprocess interaction and LabelMe JSON I/O.

    Workflow:
      open_frame() → launches LabelMe exe, starts File Watcher on label JSON.
      Each save in LabelMe triggers on_save(boxes) via the debounced watcher.
      write_boxes() writes AI-generated annotations back to JSON for LabelMe to display.

    Thread safety: watchdog fires on a background thread. _file_updated signal
    marshals the callback to the Qt main thread via AutoConnection.
    """

    _file_updated = pyqtSignal()

    def __init__(self, exe_path: str, debounce_ms: int = 500):
        super().__init__()
        self.exe_path = exe_path
        self.debounce_ms = debounce_ms

        self._process: Optional[subprocess.Popen] = None
        self._observer: Optional[Observer] = None
        self._watcher: Optional[_DebounceWatcher] = None
        self._pending_callback: Optional[Callable] = None
        self._file_updated.connect(self._dispatch_to_main_thread)

    def _dispatch_to_main_thread(self):
        cb = self._pending_callback
        self._pending_callback = None
        if cb:
            cb()

    # ------------------------------------------------------------------
    # LabelMe subprocess
    # ------------------------------------------------------------------

    def open_frame(
        self,
        frame_path: Path,
        label_path: Path,
        on_save: Callable[[list[LabeledBox]], None],
        img_w: int,
        img_h: int,
    ):
        """
        Open LabelMe for frame_path. JSON is written to label_path (in labels_dir).
        on_save(boxes) is called on every save event (debounced 500ms).
        Completes in < 2s by launching process asynchronously.
        """
        frame_path = frame_path.resolve()
        label_path = label_path.resolve()
        if not self.exe_path:
            raise RuntimeError(
                "LabelMe executable path not configured. "
                "Set it via Toolbar → LabelMe Path."
            )

        self._stop_watcher()

        # Ensure JSON exists so LabelMe can open it
        if not label_path.exists():
            self.write_boxes(label_path, frame_path, [], img_w, img_h)

        # Start watcher before launching so no save is missed
        self._start_watcher(label_path, on_save, img_w, img_h)

        # LabelMe 6.x expects --output to be a directory. It writes
        # <image_stem>.json inside that directory, which matches label_path.
        cmd = [self.exe_path, str(frame_path), "--output", str(label_path.parent)]
        self._process = subprocess.Popen(cmd)

    def close(self):
        self._stop_watcher()
        if self._process and self._process.poll() is None:
            self._process.terminate()
            self._process = None

    def watch_label_file(
        self,
        label_path: Path,
        on_save: Callable[[list[LabeledBox]], None],
        img_w: int,
        img_h: int,
    ):
        """Watch an existing LabelMe/X-AnyLabeling JSON and call on_save after saves."""
        label_path = label_path.resolve()
        self._stop_watcher()
        label_path.parent.mkdir(parents=True, exist_ok=True)
        self._start_watcher(label_path, on_save, img_w, img_h)

    # ------------------------------------------------------------------
    # JSON I/O
    # ------------------------------------------------------------------

    def read_boxes(
        self,
        label_path: Path,
        img_w: int,
        img_h: int,
    ) -> list[LabeledBox]:
        if not label_path.exists():
            return []
        try:
            with open(label_path, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            return []

        boxes: list[LabeledBox] = []
        for shape in data.get("shapes", []):
            if shape.get("shape_type") != "rectangle":
                continue
            pts = shape["points"]
            if len(pts) < 2:
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            flags = shape.get("flags") or {}
            confidence = flags.get("confidence")
            if confidence is None:
                match = _CONF_RE.search(shape.get("description") or "")
                confidence = float(match.group(1)) if match else 1.0
            boxes.append(LabeledBox(
                label=shape["label"],
                bbox=(float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))),
                confidence=float(confidence),
                ai_generated=bool(flags.get("ai_generated", False)),
            ))
        return boxes

    def write_boxes(
        self,
        label_path: Path,
        frame_path: Path,
        boxes: list[LabeledBox],
        img_w: int,
        img_h: int,
    ):
        label_path = label_path.resolve()
        frame_path = frame_path.resolve()
        shapes = []
        for box in boxes:
            x1, y1, x2, y2 = box.bbox
            # Store confidence in description so LabelMe and X-AnyLabeling keep
            # rendering a normal rectangle without custom flag UI interference.
            desc = f"confidence={box.confidence:.3f}" if box.ai_generated else ""
            shapes.append({
                "label": box.label,
                "points": [[x1, y1], [x2, y2]],  # LabelMe 6.x uses 2-point rectangle
                "group_id": None,
                "description": desc,
                "shape_type": "rectangle",
                "flags": {},
                "mask": None,
            })

        payload = {
            "version": "6.0.0",
            "flags": {},
            "shapes": shapes,
            "imagePath": frame_path.name,
            "imageData": None,
            "imageHeight": img_h,
            "imageWidth": img_w,
        }
        label_path.parent.mkdir(parents=True, exist_ok=True)
        with open(label_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _start_watcher(
        self,
        label_path: Path,
        on_save: Callable[[list[LabeledBox]], None],
        img_w: int,
        img_h: int,
    ):
        def handle():
            # Called from watchdog background thread — marshal to Qt main thread
            boxes = self.read_boxes(label_path, img_w, img_h)
            self._pending_callback = lambda: on_save(boxes)
            self._file_updated.emit()

        self._watcher = _DebounceWatcher(label_path, handle, self.debounce_ms)
        self._observer = Observer()
        self._observer.schedule(
            self._watcher, str(label_path.parent), recursive=False
        )
        self._observer.start()

    def _stop_watcher(self):
        if self._watcher:
            self._watcher.cancel()
            self._watcher = None
        if self._observer:
            self._observer.stop()
            self._observer.join(timeout=2)
            self._observer = None
