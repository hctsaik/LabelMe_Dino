from __future__ import annotations

from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QFrame,
    QSizePolicy,
    QInputDialog,
    QHBoxLayout,
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor

from src.class_manager import ClassManager
from src.propagator import AuditEntry


class RightPanel(QWidget):
    """Right sidebar with class settings first, then review/export details."""

    editClassesRequested = pyqtSignal()
    frameSelected = pyqtSignal(int)
    thresholdChanged = pyqtSignal(str, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(280)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self._labels: list[str] = []
        self._audit_frames: list[int] = []
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 6)
        layout.setSpacing(8)

        lbl_classes = QLabel("標籤類別")
        lbl_classes.setStyleSheet("font-weight: bold; font-size: 13px;")
        layout.addWidget(lbl_classes)

        lbl_classes_hint = QLabel(
            "每個 class 可設定 YOLO ID 與最低信心門檻。\n"
            "點兩下 class 可快速調整門檻。"
        )
        lbl_classes_hint.setStyleSheet("color: #888; font-size: 10px;")
        lbl_classes_hint.setWordWrap(True)
        layout.addWidget(lbl_classes_hint)

        self.class_list = QListWidget()
        self.class_list.setMinimumHeight(280)
        self.class_list.setAlternatingRowColors(True)
        self.class_list.setFocusPolicy(Qt.NoFocus)
        self.class_list.setToolTip("點兩下可修改該 class 的 confidence threshold")
        layout.addWidget(self.class_list, stretch=2)

        class_btns = QHBoxLayout()
        self.btn_edit = QPushButton("編輯標籤類別")
        class_btns.addWidget(self.btn_edit)
        layout.addLayout(class_btns)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)
        layout.addWidget(sep)

        self.lbl_audit = QLabel("異常待檢查")
        self.lbl_audit.setStyleSheet("font-weight: bold;")
        layout.addWidget(self.lbl_audit)

        lbl_audit_hint = QLabel(
            "列出追蹤傳播後需要人工檢查的 frame。\n"
            "點兩下可跳到指定 frame。"
        )
        lbl_audit_hint.setStyleSheet("color: #888; font-size: 10px;")
        lbl_audit_hint.setWordWrap(True)
        layout.addWidget(lbl_audit_hint)

        self.audit_list = QListWidget()
        self.audit_list.setMaximumHeight(180)
        self.audit_list.setAlternatingRowColors(True)
        self.audit_list.setToolTip("點兩下可跳到指定 frame")
        layout.addWidget(self.audit_list, stretch=1)

        self.btn_next_issue = QPushButton("下一個待檢查")
        self.btn_next_issue.setEnabled(False)
        layout.addWidget(self.btn_next_issue)

        sep2 = QFrame()
        sep2.setFrameShape(QFrame.HLine)
        sep2.setFrameShadow(QFrame.Sunken)
        layout.addWidget(sep2)

        self.lbl_export_summary = QLabel("匯出狀態\n尚未設定 inference 範圍")
        self.lbl_export_summary.setStyleSheet("color: #888; font-size: 10px;")
        self.lbl_export_summary.setWordWrap(True)
        layout.addWidget(self.lbl_export_summary)

        self.btn_edit.clicked.connect(self.editClassesRequested)
        self.btn_next_issue.clicked.connect(self._on_next_issue)
        self.audit_list.itemDoubleClicked.connect(self._on_audit_click)
        self.class_list.itemDoubleClicked.connect(self._on_class_double_click)

    def update_classes(self, cm: ClassManager):
        self._labels = list(cm.labels)
        self.class_list.clear()
        for label in self._labels:
            yid = cm.yolo_id(label)
            conf = cm.conf_threshold(label)
            self.class_list.addItem(f"{label} | YOLO {yid} | min {conf:.2f}")

    def update_audit(self, entries: list[AuditEntry]):
        self.audit_list.clear()
        self._audit_frames = []
        for entry in entries:
            reason = "目標遺失" if entry.reason == "object_lost" else "低信心"
            text = f"#{entry.frame_idx:05d}  {entry.label}  {entry.confidence:.2f}  {reason}"
            item = QListWidgetItem(text)
            item.setData(Qt.UserRole, entry.frame_idx)
            color = "#FF5555" if entry.reason == "object_lost" else "#D8A500"
            item.setForeground(QColor(color))
            self.audit_list.addItem(item)
            if entry.frame_idx not in self._audit_frames:
                self._audit_frames.append(entry.frame_idx)

        n = len(entries)
        self.lbl_audit.setText(f"異常待檢查 ({n})" if n else "異常待檢查")
        self.btn_next_issue.setEnabled(bool(self._audit_frames))
        if not n:
            self.audit_list.addItem("目前沒有異常待檢查")

    def update_export_summary(
        self,
        inference_range: tuple[int, int] | None,
        annotated_frames: int = 0,
        flagged_frames: int = 0,
        output_dir: str = "",
    ):
        if inference_range is None:
            self.lbl_export_summary.setText("匯出狀態\n尚未設定 inference 範圍")
            return
        start, end = inference_range
        total = max(0, end - start + 1)
        out = output_dir or "尚未選擇"
        self.lbl_export_summary.setText(
            "匯出狀態\n"
            f"Inference 範圍：{start} - {end}，共 {total} frames\n"
            f"已有標註：{annotated_frames} frames\n"
            f"異常標記：{flagged_frames} frames\n"
            "格式：LabelMe / X-AnyLabeling 相容 JSON + JPG\n"
            f"輸出資料夾：{out}"
        )

    def _on_next_issue(self):
        if self._audit_frames:
            self.frameSelected.emit(self._audit_frames[0])

    def _on_audit_click(self, item: QListWidgetItem):
        idx = item.data(Qt.UserRole)
        if idx is not None:
            self.frameSelected.emit(int(idx))

    def _on_class_double_click(self, item: QListWidgetItem):
        row = self.class_list.row(item)
        if row < 0 or row >= len(self._labels):
            return
        label = self._labels[row]
        text = item.text()
        try:
            current = float(text.split("min")[-1].strip())
        except (ValueError, IndexError):
            current = 0.6

        val, ok = QInputDialog.getDouble(
            self,
            f"調整門檻：{label}",
            f"'{label}' 的最低 confidence\n(0.0 = 全部保留，1.0 = 只保留最高信心)：",
            value=current,
            min=0.0,
            max=1.0,
            decimals=2,
        )
        if ok:
            self.thresholdChanged.emit(label, val)
