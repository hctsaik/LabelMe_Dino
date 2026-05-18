from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class VideoMeta:
    fps: float
    width: int
    height: int
    total_frames: int
    duration: float  # seconds


class VideoCore:
    def __init__(self, session_dir: str | Path):
        self.session_dir = Path(session_dir)
        self.frames_dir = self.session_dir / "frames"
        self.features_dir = self.session_dir / "features"

        for d in (self.frames_dir, self.features_dir):
            d.mkdir(parents=True, exist_ok=True)

        self._cap: Optional[cv2.VideoCapture] = None
        self.meta: Optional[VideoMeta] = None
        self.video_path: Optional[Path] = None

    # ------------------------------------------------------------------
    # Video loading
    # ------------------------------------------------------------------

    def load(self, path: str | Path) -> VideoMeta:
        if self._cap:
            self._cap.release()

        self.video_path = Path(path)
        self._cap = cv2.VideoCapture(str(path))

        if not self._cap.isOpened():
            raise ValueError(f"Cannot open video: {path}")

        fps = self._cap.get(cv2.CAP_PROP_FPS)
        width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0.0

        self.meta = VideoMeta(fps, width, height, total_frames, duration)
        return self.meta

    # ------------------------------------------------------------------
    # Time / frame index helpers
    # ------------------------------------------------------------------

    def time_to_frame(self, t: float) -> int:
        assert self.meta, "load() must be called first"
        return max(0, min(int(t * self.meta.fps), self.meta.total_frames - 1))

    def frame_to_time(self, idx: int) -> float:
        assert self.meta
        return idx / self.meta.fps

    # ------------------------------------------------------------------
    # Frame extraction & caching
    # ------------------------------------------------------------------

    def extract_frames(
        self,
        start_time: float,
        end_time: float,
        progress_callback=None,
    ) -> list[Path]:
        """
        Extract frames from [start_time, end_time] to session frames_dir.
        Skips frames already on disk. Returns sorted list of .jpg paths.
        progress_callback(done: int, total: int) called if provided.
        """
        assert self._cap and self.meta, "load() must be called first"

        start_idx = self.time_to_frame(start_time)
        end_idx = self.time_to_frame(end_time)
        indices = range(start_idx, end_idx + 1)
        total = len(indices)

        self._cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)

        paths: list[Path] = []
        for done, idx in enumerate(indices):
            path = self.frames_dir / f"frame_{idx:06d}.jpg"
            if not path.exists():
                ret, frame = self._cap.read()
                if not ret:
                    break
                cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            else:
                # advance cap even on cache hit to keep position in sync
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, idx + 1)

            paths.append(path)
            if progress_callback:
                progress_callback(done + 1, total)

        return sorted(paths)

    def read_frame(self, frame_idx: int) -> np.ndarray:
        """Read a frame — from disk cache if available, otherwise from video."""
        cached = self.frames_dir / f"frame_{frame_idx:06d}.jpg"
        if cached.exists():
            return cv2.imread(str(cached))

        assert self._cap, "load() must be called first"
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self._cap.read()
        if not ret:
            raise ValueError(f"Cannot read frame {frame_idx}")
        return frame

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def frame_path(self, frame_idx: int) -> Path:
        return self.frames_dir / f"frame_{frame_idx:06d}.jpg"

    def feature_path(self, frame_idx: int) -> Path:
        return self.features_dir / f"frame_{frame_idx:06d}.npy"

    def label_path(self, frame_idx: int) -> Path:
        # JSON lives next to its frame jpg so LabelMe finds it when opening the folder
        return self.frames_dir / f"frame_{frame_idx:06d}.json"

    def cached_frame_indices(self) -> list[int]:
        return sorted(
            int(p.stem.split("_")[1])
            for p in self.frames_dir.glob("frame_*.jpg")
        )

    def cached_feature_indices(self) -> list[int]:
        return sorted(
            int(p.stem.split("_")[1])
            for p in self.features_dir.glob("frame_*.npy")
        )

    # ------------------------------------------------------------------
    # Rescaling guard
    # ------------------------------------------------------------------

    def scale_bbox(
        self,
        bbox: tuple[float, float, float, float],
        src_w: int,
        src_h: int,
    ) -> tuple[float, float, float, float]:
        """Rescale bbox if video resolution differs from src_w/src_h."""
        assert self.meta
        if self.meta.width == src_w and self.meta.height == src_h:
            return bbox
        sx = self.meta.width / src_w
        sy = self.meta.height / src_h
        x1, y1, x2, y2 = bbox
        return x1 * sx, y1 * sy, x2 * sx, y2 * sy

    def release(self):
        if self._cap:
            self._cap.release()
            self._cap = None
