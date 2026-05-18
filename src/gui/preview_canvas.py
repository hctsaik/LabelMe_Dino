from __future__ import annotations

import numpy as np
import cv2
from PyQt5.QtWidgets import QLabel
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap

from src.propagator import FrameAnnotation

# BGR colors
_COLOR_ANCHOR = (30, 160, 220)   # amber/gold
_COLOR_OK = (50, 200, 50)        # green
_COLOR_WARN = (40, 40, 220)      # red
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.52
_THICKNESS = 1


class PreviewCanvas(QLabel):
    """
    Displays the current video frame with DINO/anchor bbox overlays.

    Color coding:
      - Anchor frame boxes → amber
      - AI boxes with conf ≥ threshold → green
      - AI boxes with conf < threshold → red
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(640, 480)
        self.setStyleSheet("background-color: #111111;")

        self._frame: np.ndarray | None = None
        self._annotation: FrameAnnotation | None = None
        self._thresholds: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def show_frame(
        self,
        frame: np.ndarray,
        annotation: FrameAnnotation | None = None,
        conf_thresholds: dict[str, float] | None = None,
    ):
        self._frame = frame.copy()
        self._annotation = annotation
        self._thresholds = conf_thresholds or {}
        self._render()

    def clear(self):
        self._frame = None
        self._annotation = None
        self.clear()

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render(self):
        if self._frame is None:
            return

        canvas = self._frame.copy()

        if self._annotation:
            for box in self._annotation.boxes:
                threshold = self._thresholds.get(box.label, 0.6)

                if self._annotation.is_anchor:
                    color = _COLOR_ANCHOR
                elif box.confidence >= threshold:
                    color = _COLOR_OK
                else:
                    color = _COLOR_WARN

                x1, y1, x2, y2 = (int(v) for v in box.bbox)
                cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

                text = f"{box.label} {box.confidence:.2f}"
                (tw, th), baseline = cv2.getTextSize(
                    text, _FONT, _FONT_SCALE, _THICKNESS
                )
                bg_y1 = max(0, y1 - th - baseline - 4)
                cv2.rectangle(
                    canvas, (x1, bg_y1), (x1 + tw + 4, y1), color, cv2.FILLED
                )
                cv2.putText(
                    canvas, text, (x1 + 2, y1 - baseline - 2),
                    _FONT, _FONT_SCALE, (255, 255, 255), _THICKNESS, cv2.LINE_AA,
                )

        self.setPixmap(
            QPixmap.fromImage(self._to_qimage(canvas)).scaled(
                self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._render()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_qimage(bgr: np.ndarray) -> QImage:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        return QImage(rgb.data.tobytes(), w, h, ch * w, QImage.Format_RGB888)
