from __future__ import annotations

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QSpinBox, QDoubleSpinBox, QLabel,
    QPushButton, QGroupBox,
)


class LoadAnnotationDialog(QDialog):
    def __init__(
        self,
        parent=None,
        total_duration: float = 0.0,
        fps: float = 30.0,
        initial_frame: int | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("載入標記")
        self.setMinimumWidth(400)
        self._fps = max(fps, 1.0)
        self._total_frames = max(1, int(total_duration * self._fps))
        self._syncing = False
        self._build_ui(total_duration, fps)
        if initial_frame is not None:
            self._set_frame(initial_frame)

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self, total_duration: float, fps: float):
        layout = QVBoxLayout(self)
        layout.setSpacing(14)

        group = QGroupBox("這個 LabelMe 標記對應影片的哪一個 frame？")
        f1 = QVBoxLayout(group)

        hint = QLabel(
            "在主畫面拖動時間軸找到與標記圖片相符的 frame，\n"
            "再輸入下方的 Frame # 或時間。"
        )
        hint.setStyleSheet("color:#888; font-size: 10px;")
        f1.addWidget(hint)

        form = QFormLayout()
        form.setRowWrapPolicy(QFormLayout.DontWrapRows)

        self.spin_frame = QSpinBox()
        self.spin_frame.setRange(0, self._total_frames - 1)
        self.spin_frame.setMinimumWidth(110)
        self.spin_frame.setToolTip("從 0 開始的 frame 編號")
        frame_row = QHBoxLayout()
        frame_row.addWidget(self.spin_frame)
        frame_row.addWidget(QLabel(f"   （共 {self._total_frames} frames  @  {fps:.2f} fps）"))
        frame_row.addStretch()
        form.addRow("Frame #:", frame_row)

        self.spin_min = QSpinBox()
        self.spin_min.setRange(0, int(total_duration // 60))
        self.spin_min.setSuffix(" 分")
        self.spin_min.setFixedWidth(80)

        self.spin_sec = QDoubleSpinBox()
        self.spin_sec.setRange(0.0, 59.99)
        self.spin_sec.setDecimals(2)
        self.spin_sec.setSuffix(" 秒")
        self.spin_sec.setFixedWidth(95)

        time_row = QHBoxLayout()
        time_row.addWidget(self.spin_min)
        time_row.addWidget(QLabel(":"))
        time_row.addWidget(self.spin_sec)
        time_row.addWidget(QLabel("  ← 自動換算"))
        time_row.addStretch()
        form.addRow("= 時間:", time_row)
        f1.addLayout(form)
        layout.addWidget(group)

        # ── Buttons ──────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self.btn_ok = QPushButton("載入並預覽  →")
        self.btn_ok.setDefault(True)
        self.btn_ok.setToolTip(
            "擷取前後 0.5 秒的 features，顯示 DINO 追蹤結果預覽。\n"
            "確認無誤後再執行完整傳播。"
        )
        self.btn_cancel = QPushButton("取消")
        btn_row.addWidget(self.btn_ok)
        btn_row.addWidget(self.btn_cancel)
        layout.addLayout(btn_row)

        self.btn_ok.clicked.connect(self.accept)
        self.btn_cancel.clicked.connect(self.reject)

        self.spin_frame.valueChanged.connect(self._on_frame_changed)
        self.spin_min.valueChanged.connect(self._on_time_changed)
        self.spin_sec.valueChanged.connect(self._on_time_changed)

    # ------------------------------------------------------------------
    # Sync helpers
    # ------------------------------------------------------------------

    def _set_frame(self, frame: int):
        self._syncing = True
        frame = max(0, min(frame, self._total_frames - 1))
        self.spin_frame.setValue(frame)
        total_sec = frame / self._fps
        mins = int(total_sec // 60)
        self.spin_min.setValue(mins)
        self.spin_sec.setValue(round(total_sec - mins * 60, 2))
        self._syncing = False

    def _on_frame_changed(self, frame: int):
        if self._syncing:
            return
        self._syncing = True
        total_sec = frame / self._fps
        mins = int(total_sec // 60)
        self.spin_min.setValue(mins)
        self.spin_sec.setValue(round(total_sec - mins * 60, 2))
        self._syncing = False

    def _on_time_changed(self):
        if self._syncing:
            return
        self._syncing = True
        total_sec = self.spin_min.value() * 60.0 + self.spin_sec.value()
        frame = max(0, min(int(total_sec * self._fps), self._total_frames - 1))
        self.spin_frame.setValue(frame)
        self._syncing = False

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    def frame_index(self) -> int:
        return self.spin_frame.value()

    def timestamp_seconds(self) -> float:
        return self.spin_min.value() * 60.0 + self.spin_sec.value()
