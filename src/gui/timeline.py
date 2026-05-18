from __future__ import annotations

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSlider, QLabel, QPushButton,
    QDoubleSpinBox, QSpinBox, QSizePolicy,
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QPainter, QColor, QPen


_STATE_COLORS: dict[str, QColor] = {
    "unannotated": QColor("#444444"),
    "anchor": QColor("#E6A800"),
    "ai": QColor("#35B56A"),
    "anomaly": QColor("#CC3333"),
}


class TimelineBar(QWidget):
    """Custom bar widget painted with per-frame annotation states."""

    frameClicked = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(20)
        self._total_frames = 1
        self._start_frame = 0
        self._end_frame = 0
        self._current = 0
        self._states: dict[int, str] = {}

    def configure(self, total_frames: int, start_frame: int, end_frame: int):
        self._total_frames = max(1, total_frames)
        self._start_frame = start_frame
        self._end_frame = max(start_frame, end_frame)
        self._states.clear()
        self.update()

    def set_frame_state(self, frame_idx: int, state: str):
        self._states[frame_idx] = state
        self.update()

    def set_current(self, frame_idx: int):
        self._current = frame_idx
        self.update()

    def paintEvent(self, event):
        if self._end_frame <= self._start_frame:
            return
        painter = QPainter(self)
        w, h = self.width(), self.height()
        span = self._end_frame - self._start_frame

        for idx in range(self._start_frame, self._end_frame + 1):
            state = self._states.get(idx, "unannotated")
            x = int((idx - self._start_frame) / span * w)
            x2 = int((idx + 1 - self._start_frame) / span * w)
            painter.fillRect(x, 0, max(1, x2 - x), h, _STATE_COLORS[state])

        cx = int((self._current - self._start_frame) / span * w)
        painter.setPen(QPen(QColor("white"), 2))
        painter.drawLine(cx, 0, cx, h)

    def _frame_at(self, x: int) -> int:
        if self._end_frame <= self._start_frame:
            return self._start_frame
        span = self._end_frame - self._start_frame
        ratio = x / max(1, self.width())
        idx = int(ratio * span) + self._start_frame
        return max(self._start_frame, min(idx, self._end_frame))

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.frameClicked.emit(self._frame_at(event.x()))


class TimelinePanel(QWidget):
    """
    Bottom control area.

    Start/End are now read-only inference interval indicators. Export always
    uses the inference interval, not a user-selected frame range.
    """

    frameChanged = pyqtSignal(int)
    rangeChanged = pyqtSignal(float, float)  # kept for compatibility

    def __init__(self, parent=None):
        super().__init__(parent)
        self._fps = 30.0
        self._total_frames = 0
        self._current = 0
        self._play_dir = 0
        self._play_timer = QTimer(self)
        self._play_timer.setInterval(100)
        self._play_timer.timeout.connect(self._on_play_tick)
        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 2, 4, 4)
        root.setSpacing(3)

        self.bar = TimelineBar()
        self.bar.frameClicked.connect(self._on_bar_click)
        root.addWidget(self.bar)

        ctrl = QHBoxLayout()

        self.btn_prev = QPushButton("<")
        self.btn_prev.setFixedWidth(30)
        self.btn_play_prev = QPushButton("自動<")
        self.btn_play_prev.setFixedWidth(56)
        self.btn_next = QPushButton(">")
        self.btn_next.setFixedWidth(30)
        self.btn_play_next = QPushButton("自動>")
        self.btn_play_next.setFixedWidth(56)
        self.lbl_time = QLabel("0.00s / 0.00s")

        self.spin_goto = QSpinBox()
        self.spin_goto.setMinimum(0)
        self.spin_goto.setMaximum(0)
        self.spin_goto.setFixedWidth(80)
        self.spin_goto.setToolTip("輸入 frame 編號後按「跳到」")
        self.btn_goto = QPushButton("跳到")
        self.btn_goto.setFixedWidth(48)

        ctrl.addWidget(self.btn_prev)
        ctrl.addWidget(self.btn_play_prev)
        ctrl.addWidget(self.btn_next)
        ctrl.addWidget(self.btn_play_next)
        ctrl.addWidget(self.lbl_time)
        ctrl.addSpacing(12)
        ctrl.addWidget(QLabel("Frame:"))
        ctrl.addWidget(self.spin_goto)
        ctrl.addWidget(self.btn_goto)
        ctrl.addStretch()

        self.lbl_range_summary = QLabel("Inference 區間：尚未設定")
        self.lbl_range_summary.setStyleSheet("color: #888;")
        self.lbl_range_summary.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        ctrl.addWidget(self.lbl_range_summary)

        ctrl.addWidget(QLabel("Start:"))
        self.spin_start = QDoubleSpinBox()
        self.spin_start.setDecimals(2)
        self.spin_start.setSuffix("s")
        self.spin_start.setFixedWidth(76)
        self.spin_start.setEnabled(False)
        ctrl.addWidget(self.spin_start)

        ctrl.addWidget(QLabel("End:"))
        self.spin_end = QDoubleSpinBox()
        self.spin_end.setDecimals(2)
        self.spin_end.setSuffix("s")
        self.spin_end.setFixedWidth(76)
        self.spin_end.setEnabled(False)
        ctrl.addWidget(self.spin_end)

        root.addLayout(ctrl)

        legend = QLabel("圖例：灰=未處理  黃=Anchor  綠=已傳播  紅=待檢查")
        legend.setStyleSheet("color:#888; font-size:10px;")
        root.addWidget(legend)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        root.addWidget(self.slider)

        self.btn_prev.clicked.connect(lambda: self._seek_delta(-1))
        self.btn_play_prev.clicked.connect(lambda: self._toggle_play(-1))
        self.btn_next.clicked.connect(lambda: self._seek_delta(1))
        self.btn_play_next.clicked.connect(lambda: self._toggle_play(1))
        self.btn_goto.clicked.connect(self._on_goto)
        self.slider.valueChanged.connect(self._on_slider)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_video(self, fps: float, total_frames: int, duration: float):
        self._stop_play()
        self._fps = fps
        self._total_frames = total_frames
        self.slider.setMaximum(max(0, total_frames - 1))
        self.spin_start.setMaximum(duration)
        self.spin_end.setMaximum(duration)
        self.spin_start.setValue(0.0)
        self.spin_end.setValue(duration)
        self.spin_goto.setMaximum(max(0, total_frames - 1))
        self.bar.configure(total_frames, 0, total_frames - 1)
        self.lbl_time.setText(f"0.00s / {duration:.2f}s")
        self.lbl_range_summary.setText("Inference 區間：尚未設定")
        self._update_play_buttons()

    def set_current_frame(self, frame_idx: int):
        self._current = frame_idx
        self.slider.blockSignals(True)
        self.slider.setValue(frame_idx)
        self.slider.blockSignals(False)
        self.bar.set_current(frame_idx)
        t = frame_idx / max(1.0, self._fps)
        total = self._total_frames / max(1.0, self._fps)
        self.lbl_time.setText(f"{t:.2f}s / {total:.2f}s")
        self._update_play_buttons()

    def set_frame_state(self, frame_idx: int, state: str):
        self.bar.set_frame_state(frame_idx, state)

    def set_inference_range(self, start_frame: int, end_frame: int):
        start_t = start_frame / max(1.0, self._fps)
        end_t = end_frame / max(1.0, self._fps)
        total = max(0, end_frame - start_frame + 1)
        self.spin_start.setValue(start_t)
        self.spin_end.setValue(end_t)
        self.lbl_range_summary.setText(
            f"Inference 區間：frame {start_frame} - {end_frame}，共 {total} frames"
        )

    def get_range(self) -> tuple[float, float]:
        return self.spin_start.value(), self.spin_end.value()

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_bar_click(self, frame_idx: int):
        self._stop_play()
        self.frameChanged.emit(frame_idx)

    def _on_goto(self):
        self._stop_play()
        target = max(0, min(self.spin_goto.value(), self._total_frames - 1))
        self.frameChanged.emit(target)

    def _on_slider(self, value: int):
        self._stop_play()
        self.frameChanged.emit(value)

    def _seek_delta(self, delta: int, stop_play: bool = True):
        if stop_play:
            self._stop_play()
        target = max(0, min(self._current + delta, self._total_frames - 1))
        self.frameChanged.emit(target)

    def _toggle_play(self, direction: int):
        if self._play_timer.isActive() and self._play_dir == direction:
            self._stop_play()
            return
        if self._total_frames <= 0:
            return
        self._play_dir = direction
        self._play_timer.start()
        self._update_play_buttons()

    def _stop_play(self):
        if self._play_timer.isActive():
            self._play_timer.stop()
        self._play_dir = 0
        if hasattr(self, "btn_play_prev") and hasattr(self, "btn_play_next"):
            self._update_play_buttons()

    def _update_play_buttons(self):
        self.btn_play_prev.setText("停止" if self._play_dir == -1 else "自動<")
        self.btn_play_next.setText("停止" if self._play_dir == 1 else "自動>")
        self.btn_play_prev.setEnabled(self._current > 0 or self._play_dir == -1)
        self.btn_play_next.setEnabled(
            self._current < self._total_frames - 1 or self._play_dir == 1
        )

    def _on_play_tick(self):
        if self._total_frames <= 0 or self._play_dir == 0:
            self._stop_play()
            return

        step = max(1, int(round(self._fps * 0.1)))
        target = max(
            0,
            min(self._current + self._play_dir * step, self._total_frames - 1),
        )
        if target == self._current:
            self._stop_play()
            return

        self.frameChanged.emit(target)
        if target in (0, self._total_frames - 1):
            self._stop_play()
