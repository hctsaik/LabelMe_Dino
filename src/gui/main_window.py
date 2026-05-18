from __future__ import annotations

import logging
import json
import os
import re
import shutil
import subprocess
import sys
import traceback
import zipfile
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

import cv2
import yaml
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QToolBar, QFileDialog, QMessageBox,
    QProgressBar, QLabel, QStatusBar, QPushButton,
    QComboBox, QDoubleSpinBox, QStackedLayout, QFrame,
)
from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal

from src.video_core import VideoCore
from src.dino_engine import DinoEngine
from src.class_manager import ClassManager
from src.propagator import Propagator, LabeledBox, FrameAnnotation
from src.label_bridge import LabelBridge
from src.gui.preview_canvas import PreviewCanvas
from src.gui.timeline import TimelinePanel
from src.gui.right_panel import RightPanel
from src.gui.class_editor import ClassEditorDialog
from src.gui.load_annotation_dialog import LoadAnnotationDialog

APP_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = APP_ROOT / "config.yaml"

# Default preview radius in seconds (used before full propagation)
PREVIEW_RADIUS_SEC = 0.5


# ------------------------------------------------------------------
# Background workers
# ------------------------------------------------------------------

class PrepareWorker(QThread):
    """
    Runs entirely off the main thread:
      1. Extract video frames for the given range
      2. Load DinoEngine if not yet loaded  (can take several seconds)
      3. Extract DINOv2 features for each frame (skip if cached)
    """
    stage    = pyqtSignal(str)       # human-readable current step
    progress = pyqtSignal(int, int)  # done, total
    finished = pyqtSignal(list)      # list[Path] of frame paths
    error    = pyqtSignal(str)

    def __init__(
        self,
        start_sec: float,
        end_sec: float,
        vc: VideoCore,
        engine_holder: list,         # [None] or [DinoEngine] — mutated in-thread
        parent=None,
    ):
        super().__init__(parent)
        self.start_sec = start_sec
        self.end_sec = end_sec
        self.vc = vc
        self.engine_holder = engine_holder

    def run(self):
        try:
            n_steps = 3 if self.engine_holder[0] is None else 2

            self.stage.emit(f"[1/{n_steps}] Extracting video frames…")
            frame_paths = self.vc.extract_frames(self.start_sec, self.end_sec)

            step = 2
            if self.engine_holder[0] is None:
                self.stage.emit(f"[{step}/{n_steps}] Loading DINOv2 model — first run, please wait (~10 s)…")
                self.engine_holder[0] = DinoEngine()
                step += 1

            engine: DinoEngine = self.engine_holder[0]
            total = len(frame_paths)
            self.stage.emit(f"[{step}/{n_steps}] Computing DINOv2 features — 0 / {total} frames…")

            for done, fp in enumerate(frame_paths):
                idx = int(Path(fp).stem.split("_")[1])
                feat_path = self.vc.feature_path(idx)
                if not feat_path.exists():
                    img = cv2.imread(str(fp))
                    engine.extract_and_save(img, feat_path)
                self.progress.emit(done + 1, total)
                self.stage.emit(
                    f"[{step}/{n_steps}] Computing DINOv2 features — {done + 1} / {total} frames…"
                )

            self.finished.emit(frame_paths)
        except Exception as exc:
            tb = traceback.format_exc()
            logging.getLogger(__name__).error("PrepareWorker failed:\n%s", tb)
            self.error.emit(tb)


class PropagationWorker(QThread):
    progress = pyqtSignal(int, int)
    finished = pyqtSignal()
    error    = pyqtSignal(str)

    def __init__(
        self,
        propagator: Propagator,
        anchor_idx: int,
        forward: bool = True,
        backward: bool = True,
        radius_seconds: float | None = None,
    ):
        super().__init__()
        self.propagator     = propagator
        self.anchor_idx     = anchor_idx
        self.forward        = forward
        self.backward       = backward
        self.radius_seconds = radius_seconds

    def run(self):
        try:
            self.propagator.propagate_from_anchor(
                self.anchor_idx,
                forward=self.forward,
                backward=self.backward,
                radius_seconds=self.radius_seconds,
                progress_callback=lambda d, t: self.progress.emit(d, t),
            )
            self.finished.emit()
        except Exception as exc:
            tb = traceback.format_exc()
            logging.getLogger(__name__).error("PropagationWorker failed:\n%s", tb)
            self.error.emit(tb)


# ------------------------------------------------------------------
# Main window
# ------------------------------------------------------------------

class MainWindow(QMainWindow):
    """
    Workflow
    ────────
    1. Import Video
    2. Load Annotation  → JSON + dialog (frame #)
                        → PrepareWorker extracts ±0.5 s preview range (off main thread)
                        → mini-propagation → Preview mode
    3. Preview mode     → ← / → navigate freely, DINO result visible
                        → 延伸 ±1s to expand range, Confirm to run full propagation
    4. 匯出標記         → export annotated frames as jpg + LabelMe JSON
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("video_annotator")

        self._cfg = self._load_config()

        _tmp = self._cfg.get("session", {}).get("tmp_dir", "tmp")
        self.video_core    = VideoCore(_tmp)   # replaced per-video in _on_import_video
        self._engine_holder: list = [None]   # shared mutable ref for PrepareWorker
        self.dino_engine: DinoEngine | None = None
        self.class_manager = ClassManager()
        self.class_manager.load_from_config(CONFIG_PATH)
        self.propagator:    Propagator   | None = None
        self.label_bridge:  LabelBridge  | None = None

        self._current_frame_idx: int  = 0
        self._worker: QThread | None  = None

        # Pending data kept across the async Load Annotation pipeline
        self._pending_anchor_frame: int  | None = None
        self._pending_anchor_boxes: list | None = None
        self._pending_prop_forward:  bool  = True
        self._pending_prop_backward: bool  = True
        self._preview_radius_sec: float = PREVIEW_RADIUS_SEC

        # Preview mode
        self._preview_mode:        bool      = False
        self._preview_frames:      list[int] = []
        self._preview_pos:         int       = 0
        self._preview_start_idx:   int       = 0   # current preview boundary (backward)
        self._preview_end_idx:     int       = 0   # current preview boundary (forward)
        self._inference_start_idx: int | None = None
        self._inference_end_idx:   int | None = None
        self._last_export_dir:     str        = ""
        self._preview_play_dir: int = 0
        self._preview_play_timer = QTimer(self)
        self._preview_play_timer.setInterval(100)
        self._preview_play_timer.timeout.connect(self._on_preview_play_tick)

        self._build_ui()
        self._rebuild_bridges()
        self.right_panel.update_classes(self.class_manager)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        self._build_toolbar()

        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(4, 4, 4, 4)
        ll.setSpacing(4)
        self.canvas = PreviewCanvas()
        self.canvas_area = QWidget()
        self.canvas_stack = QStackedLayout(self.canvas_area)
        self.canvas_stack.setContentsMargins(0, 0, 0, 0)
        self.canvas_stack.setStackingMode(QStackedLayout.StackAll)
        self.canvas_stack.addWidget(self.canvas)
        self.progress_overlay = self._build_progress_overlay()
        self.progress_overlay.setVisible(False)
        self.canvas_stack.addWidget(self.progress_overlay)
        ll.addWidget(self.canvas_area, stretch=1)
        ll.addWidget(self._build_action_bar())
        ll.addWidget(self._build_timeline())
        root.addWidget(left, stretch=1)

        self.right_panel = RightPanel()
        self.right_panel.editClassesRequested.connect(self._on_edit_classes)
        self.right_panel.frameSelected.connect(self._on_frame_changed)
        self.right_panel.thresholdChanged.connect(self._on_threshold_changed)
        root.addWidget(self.right_panel)

        self.lbl_status = QLabel("步驟 1：請先載入影片")
        sb = QStatusBar()
        sb.addWidget(self.lbl_status, stretch=1)
        self.setStatusBar(sb)

    def _build_progress_overlay(self) -> QWidget:
        overlay = QWidget()
        overlay.setAttribute(Qt.WA_StyledBackground, True)
        overlay.setStyleSheet("background-color: rgba(0, 0, 0, 150);")

        root = QVBoxLayout(overlay)
        root.setContentsMargins(24, 24, 24, 24)
        root.addStretch(1)

        panel = QFrame()
        panel.setObjectName("progressOverlayPanel")
        panel.setFixedWidth(460)
        panel.setStyleSheet(
            "#progressOverlayPanel {"
            "background-color: rgba(24, 28, 34, 235);"
            "border: 1px solid #4b5563;"
            "border-radius: 8px;"
            "}"
            "QLabel { color: #f3f4f6; }"
        )

        pl = QVBoxLayout(panel)
        pl.setContentsMargins(22, 18, 22, 18)
        pl.setSpacing(10)

        self.progress_overlay_title = QLabel("Inference 執行中")
        self.progress_overlay_title.setAlignment(Qt.AlignCenter)
        self.progress_overlay_title.setStyleSheet("font-size: 18px; font-weight: bold;")
        pl.addWidget(self.progress_overlay_title)

        self.progress_overlay_detail = QLabel("準備中...")
        self.progress_overlay_detail.setAlignment(Qt.AlignCenter)
        self.progress_overlay_detail.setWordWrap(True)
        self.progress_overlay_detail.setStyleSheet("font-size: 12px; color: #d1d5db;")
        pl.addWidget(self.progress_overlay_detail)

        self.progress_overlay_bar = QProgressBar()
        self.progress_overlay_bar.setRange(0, 100)
        self.progress_overlay_bar.setTextVisible(False)
        self.progress_overlay_bar.setFixedHeight(18)
        pl.addWidget(self.progress_overlay_bar)

        self.progress_overlay_percent = QLabel("0%")
        self.progress_overlay_percent.setAlignment(Qt.AlignCenter)
        self.progress_overlay_percent.setStyleSheet("font-size: 13px; color: #d1d5db;")
        pl.addWidget(self.progress_overlay_percent)

        root.addWidget(panel, alignment=Qt.AlignCenter)
        root.addStretch(1)
        return overlay

    def _build_toolbar(self):
        tb = QToolBar()
        tb.setMovable(False)
        tb.setFloatable(False)
        self.addToolBar(tb)
        self.lbl_video_info = QLabel("  尚未載入影片")
        self.lbl_video_info.setStyleSheet("padding: 0 8px; color: #aaaaaa;")
        tb.addWidget(self.lbl_video_info)

    def _build_action_bar(self) -> QWidget:
        container = QWidget()
        cl = QVBoxLayout(container)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.setSpacing(0)

        # ── Normal bar ────────────────────────────────────────────────
        self.normal_bar = QWidget()
        nl = QHBoxLayout(self.normal_bar)
        nl.setContentsMargins(0, 0, 0, 0)
        nl.setSpacing(6)

        self.btn_import_video = QPushButton("載入影片")
        self.btn_load_ann     = QPushButton("載入 LabelMe 標註")
        self.btn_load_xany_ann = QPushButton("載入 X-AnyLabeling 標註")
        self.btn_labelme_annotate = QPushButton("LabelMe 標註")
        self.btn_anylabeling_annotate = QPushButton("X-AnyLabeling 標註")
        self.cmb_preview_direction = QComboBox()
        self.cmb_preview_direction.addItems(["雙向", "往後", "往前"])
        self.cmb_preview_direction.setToolTip("選擇從 anchor frame 往哪個方向做 tracking + DINO 預覽")
        self.spin_preview_sec = QDoubleSpinBox()
        self.spin_preview_sec.setRange(0.5, 30.0)
        self.spin_preview_sec.setSingleStep(0.5)
        self.spin_preview_sec.setDecimals(1)
        self.spin_preview_sec.setValue(PREVIEW_RADIUS_SEC)
        self.spin_preview_sec.setSuffix(" s")
        self.spin_preview_sec.setToolTip("預覽 propagation 的秒數，可用 0.5 秒為單位增加")
        self.btn_preview_current = QPushButton("預覽 tracking + DINO")
        self.btn_preview_current.setToolTip("使用目前 frame 的 JSON 標註作為 anchor，依方向與秒數建立預覽區間")

        self.btn_import_video.setToolTip("開啟 MP4 / AVI / MOV 影片")
        self.btn_load_ann.setToolTip(
            "載入 LabelMe 相容 JSON，指定 anchor frame 後預覽"
        )
        self.btn_load_xany_ann.setToolTip(
            "載入 X-AnyLabeling 標註 JSON，指定 anchor frame 後預覽"
        )
        self.btn_labelme_annotate.setToolTip(
            "用 LabelMe 開啟目前 frame，JSON 會存成同名 frame_XXXXXX.json"
        )
        self.btn_anylabeling_annotate.setToolTip(
            "用 X-AnyLabeling 開啟目前 frame，沿用同資料夾同名 JSON"
        )

        nl.addWidget(self.btn_import_video)
        nl.addWidget(self.btn_labelme_annotate)
        nl.addWidget(self.btn_anylabeling_annotate)
        nl.addWidget(self.btn_load_ann)
        nl.addWidget(self.btn_load_xany_ann)
        nl.addSpacing(10)
        nl.addWidget(QLabel("方向"))
        nl.addWidget(self.cmb_preview_direction)
        nl.addWidget(QLabel("秒數"))
        nl.addWidget(self.spin_preview_sec)
        nl.addWidget(self.btn_preview_current)
        nl.addStretch()

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setFixedWidth(200)
        nl.addWidget(self.progress_bar)
        nl.addStretch()

        self.btn_export = QPushButton("匯出 inference 區間")
        self.btn_export.setToolTip(
            "匯出整個 inference 區間內所有已產生標註的 frame，不再手動選 Export 起點/終點"
        )
        nl.addWidget(self.btn_export)

        self.btn_import_video.clicked.connect(self._on_import_video)
        self.btn_load_ann.clicked.connect(self._on_load_annotation)
        self.btn_load_xany_ann.clicked.connect(self._on_load_annotation)
        self.btn_labelme_annotate.clicked.connect(self._on_open_labelme)
        self.btn_anylabeling_annotate.clicked.connect(self._on_open_anylabeling)
        self.btn_preview_current.clicked.connect(self._on_preview_current_annotation)
        self.btn_export.clicked.connect(self._on_export_yolo)

        cl.addWidget(self.normal_bar)

        # ── Preview bar (hidden until preview mode) ───────────────────
        self.preview_bar = QWidget()
        self.preview_bar.setStyleSheet(
            "QWidget{background:#1e2e1e;border-radius:4px;}"
            "QPushButton{background:#2e4a2e;color:#bbffbb;"
            "border:1px solid #3a6a3a;padding:4px 10px;border-radius:3px;}"
            "QPushButton:hover{background:#3a5a3a;}"
            "QPushButton:disabled{color:#557755;border-color:#2a4a2a;}"
        )
        pl = QHBoxLayout(self.preview_bar)
        pl.setContentsMargins(8, 4, 8, 4)
        pl.setSpacing(8)

        lbl = QLabel("預覽模式")
        lbl.setStyleSheet("color:#88ff88;font-weight:bold;")
        pl.addWidget(lbl)

        self.btn_prev10   = QPushButton("-10")
        self.btn_prev1    = QPushButton("-1")
        self.btn_play_backward = QPushButton("往前播放")
        self.btn_play_forward = QPushButton("往後播放")
        self.btn_next1    = QPushButton("+1")
        self.btn_next10   = QPushButton("+10")
        self.lbl_preview_pos = QLabel("")
        self.lbl_preview_pos.setStyleSheet("color:#88ff88;min-width:240px;")

        for b in (self.btn_prev10, self.btn_prev1, self.btn_next1, self.btn_next10):
            b.setFixedWidth(40)
        for b in (self.btn_play_backward, self.btn_play_forward):
            b.setFixedWidth(82)

        pl.addWidget(self.btn_prev10)
        pl.addWidget(self.btn_prev1)
        pl.addWidget(self.btn_play_backward)
        pl.addWidget(self.lbl_preview_pos)
        pl.addWidget(self.btn_play_forward)
        pl.addWidget(self.btn_next1)
        pl.addWidget(self.btn_next10)
        pl.addSpacing(16)

        # Extend preview range button — extends both directions, auto-updates export range
        self.btn_extend = QPushButton("擴大 ±0.5s")
        self.btn_extend.setToolTip("前後各延伸 0.5 秒；確認後這段會成為 inference 區間")
        self.lbl_preview_range = QLabel("")
        self.lbl_preview_range.setStyleSheet("color:#aaffaa;font-size:10px;min-width:160px;")
        pl.addWidget(self.btn_extend)
        pl.addWidget(self.lbl_preview_range)
        pl.addStretch()

        self.btn_confirm = QPushButton("使用此範圍開始傳播")
        self.btn_confirm.setStyleSheet(
            "QPushButton{background:#1a4a1a;color:#aaffaa;font-weight:bold;"
            "border:1px solid #3a9a3a;padding:4px 14px;border-radius:3px;}"
            "QPushButton:hover{background:#2a5a2a;}"
        )
        self.btn_cancel_preview = QPushButton("取消預覽")
        pl.addWidget(self.btn_confirm)
        pl.addWidget(self.btn_cancel_preview)

        self.btn_prev10.clicked.connect(lambda: self._seek_preview(-10))
        self.btn_prev1.clicked.connect(lambda: self._seek_preview(-1))
        self.btn_play_backward.clicked.connect(lambda: self._toggle_preview_play(-1))
        self.btn_play_forward.clicked.connect(lambda: self._toggle_preview_play(1))
        self.btn_next1.clicked.connect(lambda: self._seek_preview(1))
        self.btn_next10.clicked.connect(lambda: self._seek_preview(10))
        self.btn_extend.clicked.connect(self._on_extend_preview)
        self.btn_confirm.clicked.connect(self._on_confirm_propagation)
        self.btn_cancel_preview.clicked.connect(self._on_cancel_preview)

        self.preview_bar.setVisible(False)
        cl.addWidget(self.preview_bar)

        return container

    def _build_timeline(self) -> QWidget:
        self.timeline = TimelinePanel()
        self.timeline.frameChanged.connect(self._on_frame_changed)
        self.timeline.rangeChanged.connect(self._on_range_changed)
        return self.timeline

    # ------------------------------------------------------------------
    # Import video
    # ------------------------------------------------------------------

    def _on_import_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Video", "",
            "Video Files (*.mp4 *.avi *.mov);;All Files (*)"
        )
        if not path:
            return

        # Each video gets its own session directory to avoid cross-video cache collisions
        tmp_base = Path(self._cfg.get("session", {}).get("tmp_dir", "tmp"))
        video_stem = Path(path).stem
        self.video_core.release()
        self.video_core = VideoCore(tmp_base / video_stem)
        self.propagator = None
        self._pending_anchor_frame = None
        self._pending_anchor_boxes = None
        self._preview_frames = []
        self._preview_start_idx = 0
        self._preview_end_idx = 0
        self._inference_start_idx = None
        self._inference_end_idx = None
        self._last_export_dir = ""
        self.right_panel.update_export_summary(None)

        try:
            meta = self.video_core.load(path)
        except ValueError as exc:
            QMessageBox.critical(self, "Cannot Open Video", str(exc))
            return

        self.timeline.set_video(meta.fps, meta.total_frames, meta.duration)
        self._show_frame(0)
        self.lbl_video_info.setText(
            f"  {Path(path).name}   {meta.width}×{meta.height}   "
            f"{meta.fps:.2f} fps   {meta.duration:.1f}s   {meta.total_frames} frames"
        )
        self.lbl_status.setText(
            "步驟 2：請在時間軸找到已標註的 frame，然後載入 LabelMe / X-AnyLabeling JSON"
        )

    # ------------------------------------------------------------------
    # Load Annotation  →  Preview  →  Confirm & Propagate
    # ------------------------------------------------------------------

    def _on_load_annotation(self):
        if not self.video_core.meta:
            QMessageBox.warning(self, "Not Ready", "Import a video first.")
            return

        json_path, _ = QFileDialog.getOpenFileName(
            self, "Select LabelMe JSON", "", "LabelMe JSON (*.json);;All Files (*)"
        )
        if not json_path:
            return

        import json as _json
        try:
            with open(json_path, encoding="utf-8") as f:
                data = _json.load(f)
        except Exception as exc:
            QMessageBox.critical(self, "Cannot Read JSON", str(exc))
            return

        # Auto-fill only if our own frame_XXXXXX filename
        stem = Path(data.get("imagePath", "")).stem
        initial_frame: int | None = None
        m = re.match(r"frame_(\d+)$", stem)
        if m:
            initial_frame = int(m.group(1))

        meta = self.video_core.meta

        dlg = LoadAnnotationDialog(
            parent=self,
            total_duration=meta.duration,
            fps=meta.fps,
            initial_frame=initial_frame,
        )
        if dlg.exec_() != LoadAnnotationDialog.Accepted:
            return

        frame_idx   = dlg.frame_index()
        do_forward, do_backward = self._selected_preview_direction()
        self._preview_radius_sec = float(self.spin_preview_sec.value())

        # Copy JSON and read boxes
        import shutil as _shutil
        label_path = self.video_core.label_path(frame_idx)
        _shutil.copy2(json_path, label_path)

        boxes = self.label_bridge.read_boxes(label_path, meta.width, meta.height)
        if not boxes:
            QMessageBox.warning(self, "No Annotations",
                                "The JSON has no rectangle shapes.")
            return

        for box in boxes:
            if not self.class_manager.has_label(box.label):
                next_id = max(
                    (self.class_manager.yolo_id(l) for l in self.class_manager.labels),
                    default=-1,
                ) + 1
                self.class_manager.add(box.label, yolo_id=next_id)
        self.class_manager.save_to_config(CONFIG_PATH)
        self.right_panel.update_classes(self.class_manager)

        if self.propagator is None:
            self._rebuild_bridges()

        self._pending_anchor_frame   = frame_idx
        self._pending_anchor_boxes   = boxes
        self._pending_prop_forward   = do_forward
        self._pending_prop_backward  = do_backward

        # ── Immediate visual feedback ─────────────────────────────────
        # Show the anchor frame with its boxes right now, before any feature
        # extraction, so the user can see something happened instantly.
        try:
            frame_img = self.video_core.read_frame(frame_idx)
            from src.propagator import FrameAnnotation as _FA
            temp_ann = _FA(frame_idx=frame_idx, boxes=boxes, is_anchor=True)
            thresholds = {
                lbl: self.class_manager.conf_threshold(lbl)
                for lbl in self.class_manager.labels
            }
            self.canvas.show_frame(frame_img, temp_ann, thresholds)
            self.timeline.set_current_frame(frame_idx)
            self._current_frame_idx = frame_idx
        except Exception:
            pass

        labels_str = ", ".join({b.label for b in boxes})
        t = self.video_core.frame_to_time(frame_idx)
        mins, secs = divmod(t, 60)
        self.lbl_status.setText(
            f"Annotation loaded — frame {frame_idx}  ({int(mins)}:{secs:05.2f})  "
            f"labels: [{labels_str}]  |  Preparing preview features…"
        )

        # Disable button to prevent double-click while working
        self.btn_load_ann.setEnabled(False)

        # Preview uses the selected short radius; the full radius runs on Confirm
        start_sec = self.video_core.frame_to_time(
            max(0, frame_idx - int(self._preview_radius_sec * meta.fps))
            if do_backward else frame_idx
        )
        end_sec = self.video_core.frame_to_time(
            min(meta.total_frames - 1, frame_idx + int(self._preview_radius_sec * meta.fps))
            if do_forward else frame_idx
        )

        self._start_prepare_worker(
            start_sec, end_sec,
            on_finished=self._on_preview_prepare_done,
        )

    def _on_preview_prepare_done(self, frame_paths: list):
        """Preview features ready — set anchor, mini-propagate, enter preview mode."""
        frame_idx = self._pending_anchor_frame
        boxes     = self._pending_anchor_boxes
        if frame_idx is None or not boxes:
            self._show_progress(False)
            return

        # Build the ordered list of preview frames from cached features
        meta = self.video_core.meta
        feat_dir = self.video_core.features_dir
        preview_frame_list = sorted(
            int(p.stem.split("_")[1])
            for p in feat_dir.glob("frame_*.npy")
            if abs(int(p.stem.split("_")[1]) - frame_idx) <=
               int(self._preview_radius_sec * meta.fps) + 2
        )
        if not preview_frame_list:
            preview_frame_list = [frame_idx]

        self.propagator.set_anchor(frame_idx, boxes)

        n_preview = len(preview_frame_list) - 1  # frames to propagate
        preview_radius = self._preview_radius_sec + 1.0 / meta.fps
        self._show_progress(
            True,
            f"Running preview propagation ({n_preview} frames) — "
            "please wait, this uses optical flow + DINO…"
        )

        worker = PropagationWorker(
            self.propagator, frame_idx,
            forward=self._pending_prop_forward,
            backward=self._pending_prop_backward,
            radius_seconds=preview_radius,
        )
        worker.progress.connect(
            lambda d, t: self._set_progress_value(d, t)
        )

        # Capture preview_frame_list in closure
        def on_mini_done():
            self._show_progress(False)
            self.btn_load_ann.setEnabled(True)
            self.btn_preview_current.setEnabled(True)
            self._preview_frames = preview_frame_list
            try:
                self._preview_pos = self._preview_frames.index(frame_idx)
            except ValueError:
                self._preview_pos = 0
            self._enter_preview_mode()

        worker.finished.connect(on_mini_done)
        worker.error.connect(
            lambda e: (
                logger.error("DINO Error: %s", e),
                self._show_progress(False),
                self.btn_load_ann.setEnabled(True),
                self.btn_preview_current.setEnabled(True),
                QMessageBox.critical(self, "DINO Error", e),
            )
        )
        self._worker = worker
        worker.start()

    def _enter_preview_mode(self):
        self._preview_mode = True
        self.normal_bar.setVisible(False)
        self.preview_bar.setVisible(True)
        self.timeline.set_frame_state(self._pending_anchor_frame, "anchor")
        self._refresh_timeline_states()

        if self._preview_frames:
            self._preview_start_idx = self._preview_frames[0]
            self._preview_end_idx   = self._preview_frames[-1]
            self._sync_export_range()

        self._update_preview_nav()
        self._on_frame_changed(self._preview_frames[self._preview_pos])

    def _exit_preview_mode(self):
        self._stop_preview_play()
        self._preview_mode = False
        self.preview_bar.setVisible(False)
        self.normal_bar.setVisible(True)

    def _update_preview_nav(self):
        pos    = self._preview_pos
        frames = self._preview_frames
        anchor = self._pending_anchor_frame

        self.btn_prev10.setEnabled(pos > 0)
        self.btn_prev1.setEnabled(pos > 0)
        self.btn_play_backward.setEnabled(pos > 0 or self._preview_play_dir == -1)
        self.btn_play_forward.setEnabled(pos < len(frames) - 1 or self._preview_play_dir == 1)
        self.btn_play_backward.setText("停止" if self._preview_play_dir == -1 else "往前播放")
        self.btn_play_forward.setText("停止" if self._preview_play_dir == 1 else "往後播放")
        self.btn_next1.setEnabled(pos < len(frames) - 1)
        self.btn_next10.setEnabled(pos < len(frames) - 1)

        # Update extend-range label
        if self.video_core.meta and frames:
            bwd_sec = self.video_core.frame_to_time(
                anchor - self._preview_start_idx) if anchor > self._preview_start_idx else 0
            fwd_sec = self.video_core.frame_to_time(
                self._preview_end_idx - anchor) if self._preview_end_idx > anchor else 0
            self.lbl_preview_range.setText(
                f"範圍：-{bwd_sec:.1f}s ~ +{fwd_sec:.1f}s"
            )

        f = frames[pos]
        if f == anchor:
            role = "Anchor（人工標註）"
        elif f < anchor:
            offset = anchor - f
            role = f"Anchor 前 {offset} frames（AI 預測）"
        else:
            offset = f - anchor
            role = f"Anchor 後 {offset} frames（AI 預測）"

        self.lbl_preview_pos.setText(f"Frame {f}  —  {role}")

        t = self.video_core.frame_to_time(f)
        mins, secs = divmod(t, 60)
        self.lbl_status.setText(
            f"預覽中：Frame {f}  ({int(mins)}:{secs:05.2f})  —  {role}。"
            "確認穩定後可使用此範圍開始傳播。"
        )

    def _seek_preview(self, delta: int):
        self._stop_preview_play()
        new_pos = max(0, min(self._preview_pos + delta, len(self._preview_frames) - 1))
        if new_pos != self._preview_pos:
            self._preview_pos = new_pos
            self._update_preview_nav()
            self._on_frame_changed(self._preview_frames[self._preview_pos])

    def _toggle_preview_play(self, direction: int):
        if self._preview_play_timer.isActive() and self._preview_play_dir == direction:
            self._stop_preview_play()
            return
        if not self._preview_frames:
            return
        self._preview_play_dir = direction
        self._preview_play_timer.start()
        self._update_preview_nav()

    def _stop_preview_play(self):
        if hasattr(self, "_preview_play_timer") and self._preview_play_timer.isActive():
            self._preview_play_timer.stop()
        self._preview_play_dir = 0
        if hasattr(self, "btn_play_backward") and hasattr(self, "btn_play_forward"):
            self.btn_play_backward.setText("往前播放")
            self.btn_play_forward.setText("往後播放")

    def _on_preview_play_tick(self):
        if not self._preview_frames or not self.video_core.meta:
            self._stop_preview_play()
            return

        pos = self._preview_pos
        direction = self._preview_play_dir
        if direction == 0:
            self._stop_preview_play()
            return

        frame_step = max(1, int(round(self.video_core.meta.fps * 0.1)))
        current_frame = self._preview_frames[pos]
        target_frame = current_frame + direction * frame_step

        if direction > 0:
            candidates = [
                i for i, frame_idx in enumerate(self._preview_frames)
                if i > pos and frame_idx >= target_frame
            ]
            new_pos = candidates[0] if candidates else len(self._preview_frames) - 1
        else:
            candidates = [
                i for i, frame_idx in enumerate(self._preview_frames)
                if i < pos and frame_idx <= target_frame
            ]
            new_pos = candidates[-1] if candidates else 0

        if new_pos == pos:
            self._stop_preview_play()
            self._update_preview_nav()
            return

        self._preview_pos = new_pos
        self._update_preview_nav()
        self._on_frame_changed(self._preview_frames[self._preview_pos])

        if self._preview_pos in (0, len(self._preview_frames) - 1):
            self._stop_preview_play()
            self._update_preview_nav()

    def _sync_export_range(self):
        """Show the current preview range as the pending inference interval."""
        if not self.video_core.meta:
            return
        self.timeline.set_inference_range(self._preview_start_idx, self._preview_end_idx)
        self.right_panel.update_export_summary(
            (self._preview_start_idx, self._preview_end_idx),
            annotated_frames=len(self.propagator.annotations) if self.propagator else 0,
            flagged_frames=len(self.propagator.audit_list) if self.propagator else 0,
            output_dir=self._last_export_dir,
        )

    def _on_extend_preview(self):
        """Extend preview range by 0.5 seconds in both directions."""
        if not self.video_core.meta or not self._pending_anchor_frame:
            return
        self._stop_preview_play()
        meta      = self.video_core.meta
        extend_by = max(1, int(0.5 * meta.fps))

        new_start = max(0, self._preview_start_idx - extend_by)
        new_end   = min(meta.total_frames - 1, self._preview_end_idx + extend_by)
        if new_start == self._preview_start_idx and new_end == self._preview_end_idx:
            return
        self._preview_start_idx = new_start
        self._preview_end_idx   = new_end

        self._set_extend_buttons_enabled(False)
        self.lbl_status.setText("延伸 Preview 範圍：擷取 frames 與 features…")

        start_sec = self.video_core.frame_to_time(self._preview_start_idx)
        end_sec   = self.video_core.frame_to_time(self._preview_end_idx)

        def on_extend_prepare_done(frame_paths):
            # Re-propagate the extended range from anchor
            anchor = self._pending_anchor_frame
            bwd_sec = self.video_core.frame_to_time(anchor - self._preview_start_idx) \
                      if anchor > self._preview_start_idx else 0.0
            fwd_sec = self.video_core.frame_to_time(self._preview_end_idx - anchor) \
                      if self._preview_end_idx > anchor else 0.0

            worker = PropagationWorker(
                self.propagator, anchor,
                forward=self._pending_prop_forward,
                backward=self._pending_prop_backward,
                radius_seconds=max(bwd_sec, fwd_sec) + 1.0 / meta.fps,
            )
            worker.progress.connect(
                lambda d, t: self._set_progress_value(d, t)
            )

            def on_ext_prop_done():
                self._show_progress(False)
                self._set_extend_buttons_enabled(True)

                # Rebuild preview frame list from feature cache within new range
                feat_dir = self.video_core.features_dir
                self._preview_frames = sorted(
                    int(p.stem.split("_")[1])
                    for p in feat_dir.glob("frame_*.npy")
                    if self._preview_start_idx
                    <= int(p.stem.split("_")[1])
                    <= self._preview_end_idx
                )
                if not self._preview_frames:
                    self._preview_frames = [anchor]

                # Keep current position at anchor
                try:
                    self._preview_pos = self._preview_frames.index(anchor)
                except ValueError:
                    self._preview_pos = 0

                self._refresh_timeline_states()
                self._sync_export_range()
                self._update_preview_nav()
                self._on_frame_changed(self._preview_frames[self._preview_pos])
                self.lbl_status.setText(
                    f"Preview 範圍已延伸：← {bwd_sec:.1f}s  [anchor]  {fwd_sec:.1f}s →  "
                    f"（共 {len(self._preview_frames)} frames）"
                )

            worker.finished.connect(on_ext_prop_done)
            worker.error.connect(
                lambda e: (logger.error("延伸失敗: %s", e),
                           QMessageBox.critical(self, "延伸失敗", e),
                           self._set_extend_buttons_enabled(True))
            )
            self._worker = worker
            worker.start()

        self._start_prepare_worker(start_sec, end_sec, on_finished=on_extend_prepare_done)

    def _set_extend_buttons_enabled(self, enabled: bool):
        self.btn_extend.setEnabled(enabled)
        self.btn_confirm.setEnabled(enabled)
        self.btn_cancel_preview.setEnabled(enabled)
        self.btn_play_backward.setEnabled(enabled and self._preview_pos > 0)
        self.btn_play_forward.setEnabled(enabled and self._preview_pos < len(self._preview_frames) - 1)

    def _on_confirm_propagation(self):
        """User confirmed — the preview range IS the propagation range.
        Features are already computed, just run the full propagation."""
        self._exit_preview_mode()

        frame_idx  = self._pending_anchor_frame
        meta       = self.video_core.meta
        start_idx  = self._preview_start_idx
        end_idx    = self._preview_end_idx
        self._inference_start_idx = start_idx
        self._inference_end_idx = end_idx

        self.timeline.bar.configure(meta.total_frames, start_idx, end_idx)
        self.timeline.set_inference_range(start_idx, end_idx)

        bwd_sec = self.video_core.frame_to_time(frame_idx - start_idx) \
                  if frame_idx > start_idx else 0.0
        fwd_sec = self.video_core.frame_to_time(end_idx - frame_idx) \
                  if end_idx > frame_idx else 0.0

        n_total = end_idx - start_idx + 1
        status_text = (
            f"開始 inference：frame {start_idx} - {end_idx}，共 {n_total} frames "
            f"（anchor 前 {bwd_sec:.1f}s / 後 {fwd_sec:.1f}s）"
        )
        self.lbl_status.setText(
            f"已確認 — 執行完整傳播，共 {n_total} frames "
            f"（← {bwd_sec:.1f}s  /  {fwd_sec:.1f}s →）…"
        )
        self.lbl_status.setText(status_text)
        self.btn_load_ann.setEnabled(False)

        # Features are already on disk from preview + any extensions;
        # just start propagation directly.
        self._on_full_prepare_done([])

    def _on_full_prepare_done(self, _frame_paths: list):
        frame_idx = self._pending_anchor_frame
        if frame_idx is None:
            self._show_progress(False)
            return

        meta    = self.video_core.meta
        bwd_sec = self.video_core.frame_to_time(frame_idx - self._preview_start_idx) \
                  if frame_idx > self._preview_start_idx else 0.0
        fwd_sec = self.video_core.frame_to_time(self._preview_end_idx - frame_idx) \
                  if self._preview_end_idx > frame_idx else 0.0
        radius  = max(bwd_sec, fwd_sec) + 1.0 / meta.fps

        labels = list({b.label for b in (self._pending_anchor_boxes or [])})
        self.lbl_status.setText(
            f"從 frame {frame_idx} 傳播中  |  labels: {labels}…"
        )
        self._run_propagation(
            frame_idx,
            forward=self._pending_prop_forward,
            backward=self._pending_prop_backward,
            radius_seconds=radius,
        )
        self._pending_anchor_frame = None
        self._pending_anchor_boxes = None

    def _on_cancel_preview(self):
        anchor = self._pending_anchor_frame
        if anchor is not None and self.propagator:
            for f in list(self.propagator.annotations.keys()):
                del self.propagator.annotations[f]
            self._refresh_timeline_states()

        self._pending_anchor_frame = None
        self._pending_anchor_boxes = None
        self._preview_start_idx    = 0
        self._preview_end_idx      = 0
        self._exit_preview_mode()
        self.lbl_status.setText("Preview 已取消 — 請重新點選「Load Annotation」調整 frame #")
        self._show_frame(self._current_frame_idx)

    # ------------------------------------------------------------------
    # Other action handlers
    # ------------------------------------------------------------------

    def _on_open_labelme(self):
        self._open_annotation_program(
            tool_name="LabelMe",
            config_section="labelme",
            fallback_names=("labelme", "labelme.exe"),
            filename_arg="",
            with_output_arg=True,
        )

    def _on_open_anylabeling(self):
        if not self.video_core.meta:
            QMessageBox.warning(self, "X-AnyLabeling", "請先載入影片。")
            return

        try:
            frame_path, label_path = self._ensure_current_annotation_files()
        except Exception as exc:
            QMessageBox.critical(self, "X-AnyLabeling", f"無法準備目前 frame：\n{exc}")
            return

        exe = self._resolve_annotation_exe(
            "X-AnyLabeling",
            "x_anylabeling",
            (
                "xanylabeling",
                "xanylabeling.exe",
                "x-anylabeling",
                "x-anylabeling.exe",
                "anylabeling",
                "anylabeling.exe",
                "X-AnyLabeling.exe",
            ),
        )
        if not exe:
            return

        # Write classes.txt so X-AnyLabeling uses only known labels
        classes_path = label_path.parent / "classes.txt"
        labels = list(self.class_manager.labels)
        if labels:
            classes_path.write_text("\n".join(labels), encoding="utf-8")

        self.label_bridge.watch_label_file(
            label_path,
            lambda boxes, frame_idx=self._current_frame_idx: self._on_external_annotation_saved(frame_idx, boxes),
            self.video_core.meta.width,
            self.video_core.meta.height,
        )

        cmd = self._xany_command_prefix(exe) + [
            "--filename", str(frame_path),
            "--output",   str(label_path.parent),
            "--work-dir", str(label_path.parent / ".xanylabeling"),
            "--nodata",
            "--autosave",
            "--no-auto-update-check",
        ]
        if labels and classes_path.exists():
            cmd += ["--labels", str(classes_path), "--validatelabel", "exact"]

        try:
            subprocess.Popen(cmd, cwd=str(label_path.parent), env=self._annotation_subprocess_env(exe))
        except Exception as exc:
            QMessageBox.critical(
                self,
                "X-AnyLabeling",
                f"無法啟動 X-AnyLabeling：\n{exc}\n\n執行檔：{exe}",
            )
            return

        self.lbl_status.setText(
            f"已用 X-AnyLabeling 開啟 frame {self._current_frame_idx}。"
            f"請在外部工具存檔；DLB 會自動讀回 {label_path.name}。"
        )

    def _open_annotation_program(
        self,
        tool_name: str,
        config_section: str,
        fallback_names: tuple[str, ...],
        filename_arg: str,
        with_output_arg: bool,
    ):
        if not self.video_core.meta:
            QMessageBox.warning(self, tool_name, "請先載入影片。")
            return

        try:
            frame_path, label_path = self._ensure_current_annotation_files()
        except Exception as exc:
            QMessageBox.critical(self, tool_name, f"無法準備目前 frame：\n{exc}")
            return

        exe = self._resolve_annotation_exe(tool_name, config_section, fallback_names)
        if not exe:
            return
        resolved_filename_arg = self._resolve_filename_arg(exe, filename_arg)

        self.label_bridge.watch_label_file(
            label_path,
            lambda boxes, frame_idx=self._current_frame_idx: self._on_external_annotation_saved(frame_idx, boxes),
            self.video_core.meta.width,
            self.video_core.meta.height,
        )

        cmd = [exe]
        if resolved_filename_arg:
            cmd.extend([resolved_filename_arg, str(frame_path)])
        else:
            cmd.append(str(frame_path))
        if with_output_arg:
            # LabelMe 6.x and X-AnyLabeling both accept an output directory.
            cmd.extend(["--output", str(label_path.parent)])

        try:
            subprocess.Popen(cmd, cwd=str(label_path.parent), env=self._annotation_subprocess_env(exe))
        except Exception as exc:
            QMessageBox.critical(
                self,
                tool_name,
                f"無法啟動 {tool_name}：\n{exc}\n\n執行檔：{exe}",
            )
            return

        self.lbl_status.setText(
            f"已用 {tool_name} 開啟 frame {self._current_frame_idx}。"
            f"請在外部工具存檔；DLB 會自動讀回 {label_path.name}。"
        )

    def _on_external_annotation_saved(self, frame_idx: int, boxes: list[LabeledBox]):
        if not self.video_core.meta:
            return
        for box in boxes:
            if not self.class_manager.has_label(box.label):
                next_id = max(
                    (self.class_manager.yolo_id(l) for l in self.class_manager.labels),
                    default=-1,
                ) + 1
                self.class_manager.add(box.label, yolo_id=next_id)
        self.class_manager.save_to_config(CONFIG_PATH)
        self.right_panel.update_classes(self.class_manager)

        self._pending_anchor_frame = frame_idx
        self._pending_anchor_boxes = boxes
        self._pending_prop_forward, self._pending_prop_backward = self._selected_preview_direction()

        if self.propagator:
            self.propagator.set_anchor(frame_idx, boxes)
            self._refresh_timeline_states()

        try:
            frame = self.video_core.read_frame(frame_idx)
        except Exception:
            return
        ann = FrameAnnotation(frame_idx=frame_idx, boxes=boxes, is_anchor=True)
        thresholds = {
            label: self.class_manager.conf_threshold(label)
            for label in self.class_manager.labels
        }
        self.canvas.show_frame(frame, ann, thresholds)
        self.timeline.set_frame_state(frame_idx, "anchor")
        self.right_panel.update_export_summary(
            (self._inference_start_idx, self._inference_end_idx)
            if self._inference_start_idx is not None and self._inference_end_idx is not None
            else None,
            annotated_frames=len(self.propagator.annotations) if self.propagator else 1,
            flagged_frames=len(self.propagator.audit_list) if self.propagator else 0,
            output_dir=self._last_export_dir,
        )
        self.lbl_status.setText(
            f"已讀回 frame {frame_idx} 的外部標註：{len(boxes)} 個 boxes。"
            "請選方向與秒數，按「預覽 tracking + DINO」建立 inference 區間。"
        )

    def _on_preview_current_annotation(self):
        if not self.video_core.meta:
            QMessageBox.warning(self, "Not Ready", "Import a video first.")
            return
        meta = self.video_core.meta
        label_path = self.video_core.label_path(self._current_frame_idx)
        if not label_path.exists():
            QMessageBox.warning(
                self,
                "No Annotation",
                "目前 frame 還沒有 JSON。請先用 LabelMe 或 X-AnyLabeling 標註並存檔。",
            )
            return

        boxes = self.label_bridge.read_boxes(label_path, meta.width, meta.height)
        if not boxes:
            QMessageBox.warning(
                self,
                "No Boxes",
                "目前 frame 的 JSON 沒有 rectangle 標註，無法做 tracking + DINO 預覽。",
            )
            return

        for box in boxes:
            if not self.class_manager.has_label(box.label):
                next_id = max(
                    (self.class_manager.yolo_id(l) for l in self.class_manager.labels),
                    default=-1,
                ) + 1
                self.class_manager.add(box.label, yolo_id=next_id)
        self.class_manager.save_to_config(CONFIG_PATH)
        self.right_panel.update_classes(self.class_manager)

        if self.propagator is None and self.dino_engine is not None:
            self._rebuild_bridges()

        do_forward, do_backward = self._selected_preview_direction()
        self._preview_radius_sec = float(self.spin_preview_sec.value())
        frame_idx = self._current_frame_idx
        self._pending_anchor_frame = frame_idx
        self._pending_anchor_boxes = boxes
        self._pending_prop_forward = do_forward
        self._pending_prop_backward = do_backward

        start_idx = (
            max(0, frame_idx - int(self._preview_radius_sec * meta.fps))
            if do_backward else frame_idx
        )
        end_idx = (
            min(meta.total_frames - 1, frame_idx + int(self._preview_radius_sec * meta.fps))
            if do_forward else frame_idx
        )
        start_sec = self.video_core.frame_to_time(start_idx)
        end_sec = self.video_core.frame_to_time(end_idx)

        self.btn_preview_current.setEnabled(False)
        self.btn_load_ann.setEnabled(False)
        self.lbl_status.setText(
            f"準備 frame {frame_idx} 的 tracking + DINO 預覽：{start_idx} 到 {end_idx}。"
        )

        self._start_prepare_worker(
            start_sec,
            end_sec,
            on_finished=self._on_preview_prepare_done,
        )

    def _selected_preview_direction(self) -> tuple[bool, bool]:
        direction = self.cmb_preview_direction.currentText() if hasattr(self, "cmb_preview_direction") else "雙向"
        if direction == "往後":
            return True, False
        if direction == "往前":
            return False, True
        return True, True

    @staticmethod
    def _resolve_filename_arg(exe: str, filename_arg: str) -> str:
        if filename_arg != "auto":
            return filename_arg
        exe_name = Path(exe).name.lower()
        if exe_name.startswith("xanylabeling") or exe_name.startswith("x-anylabeling"):
            return "--filename"
        return ""

    @staticmethod
    def _project_root() -> Path:
        env_root = os.environ.get("CIM_REPO_ROOT", "")
        if env_root and Path(env_root).exists():
            return Path(env_root)
        if APP_ROOT.name == "LabelMe_Dino_launcher" and APP_ROOT.parent.name == "dist":
            return APP_ROOT.parents[2]
        if APP_ROOT.name == "LabelMe_Dino":
            return APP_ROOT.parent
        return Path.cwd()

    @staticmethod
    def _annotation_subprocess_env(exe: str) -> dict[str, str]:
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        env.pop("PYTHONHOME", None)
        env["PYTHONNOUSERSITE"] = "1"
        scripts_dir = str(Path(exe).resolve().parent)
        env["PATH"] = scripts_dir + os.pathsep + env.get("PATH", "")
        return env

    @staticmethod
    def _xany_command_prefix(exe: str) -> list[str]:
        exe_path = Path(exe)
        if exe_path.name.lower().startswith("xanylabeling"):
            python = exe_path.parent / "python.exe"
            if python.exists():
                return [str(python), "-m", "anylabeling.app"]
        return [exe]

    def _ensure_current_annotation_files(self) -> tuple[Path, Path]:
        assert self.video_core.meta
        frame_path = self.video_core.frame_path(self._current_frame_idx)
        label_path = self.video_core.label_path(self._current_frame_idx)
        meta = self.video_core.meta

        if not frame_path.exists():
            frame = self.video_core.read_frame(self._current_frame_idx)
            cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

        if not label_path.exists():
            boxes = []
            if self.propagator:
                ann = self.propagator.get_annotation(self._current_frame_idx)
                boxes = ann.boxes if ann else []
            self.label_bridge.write_boxes(
                label_path, frame_path, boxes, meta.width, meta.height
            )

        return frame_path.resolve(), label_path.resolve()

    def _resolve_annotation_exe(
        self,
        tool_name: str,
        config_section: str,
        fallback_names: tuple[str, ...],
    ) -> str:
        section = self._cfg.setdefault(config_section, {})
        project_root = self._project_root()

        preferred: list[Path | str | None] = []
        if config_section == "labelme":
            preferred.extend([
                os.environ.get("LABELME_EXE"),
                APP_ROOT / ".venv" / "Scripts" / "labelme.exe",
                project_root / "LabelMe_Dino" / ".venv" / "Scripts" / "labelme.exe",
            ])
        elif config_section == "x_anylabeling":
            preferred.extend([
                os.environ.get("XANYLABELING_EXE"),
                project_root / ".venv-xanylabeling" / "Scripts" / "xanylabeling.exe",
            ])

        preferred.append(section.get("exe_path", ""))
        for candidate in preferred:
            if candidate and Path(candidate).exists():
                section["exe_path"] = str(Path(candidate))
                self._save_config()
                if config_section == "labelme":
                    self._rebuild_bridges()
                return section["exe_path"]

        for name in fallback_names:
            local = Path(sys.executable).parent / name
            if local.exists():
                section["exe_path"] = str(local)
                self._save_config()
                if config_section == "labelme":
                    self._rebuild_bridges()
                return str(local)

        for name in fallback_names:
            found = shutil.which(name)
            if found:
                section["exe_path"] = found
                self._save_config()
                if config_section == "labelme":
                    self._rebuild_bridges()
                return found

        QMessageBox.information(
            self,
            tool_name,
            f"找不到 {tool_name} 的執行檔。\n\n"
            "如果你已安裝，請在下一個視窗選擇它的 .exe。\n"
            "如果尚未安裝，請先安裝 X-AnyLabeling，或把 exe_path 寫入 config.yaml。",
        )
        path, _ = QFileDialog.getOpenFileName(
            self,
            f"選擇 {tool_name} 執行檔",
            "",
            "Executable Files (*.exe);;All Files (*)",
        )
        if not path:
            return ""

        section["exe_path"] = path
        self._save_config()
        if config_section == "labelme":
            self._rebuild_bridges()
        return path

    def _on_threshold_changed(self, label: str, value: float):
        if self.class_manager.has_label(label):
            self.class_manager.update(label, conf_threshold=value)
            self.class_manager.save_to_config(CONFIG_PATH)
            self.right_panel.update_classes(self.class_manager)
            self._show_frame(self._current_frame_idx)

    def _on_edit_classes(self):
        labels_file = self._cfg.get("labelme", {}).get("labels_file", "")
        dlg = ClassEditorDialog(
            self.class_manager, labels_file=labels_file, parent=self
        )
        if dlg.exec_():
            new_lf = dlg.get_labels_file()
            if new_lf and new_lf != labels_file:
                self._cfg.setdefault("labelme", {})["labels_file"] = new_lf
            self.class_manager.save_to_config(CONFIG_PATH)
            self._save_config()
            self._rebuild_bridges()
            self.right_panel.update_classes(self.class_manager)
            self.lbl_status.setText(
                f"Classes: {', '.join(self.class_manager.labels)}"
            )

    def _on_export_yolo(self):
        if not self.video_core.meta:
            QMessageBox.warning(self, "匯出", "請先載入影片。")
            return
        if not self.propagator or not self.propagator.annotations:
            QMessageBox.warning(self, "匯出", "尚未有任何標記，請先執行 Load Annotation。")
            return

        out_dir = QFileDialog.getExistingDirectory(self, "選擇匯出資料夾")
        if not out_dir:
            return

        import cv2 as _cv2
        import shutil as _shutil

        out_path = Path(out_dir)
        meta = self.video_core.meta
        if self._inference_start_idx is None or self._inference_end_idx is None:
            # Fall back to the range of all annotated frames
            annotated_indices = sorted(self.propagator.annotations.keys())
            if not annotated_indices:
                QMessageBox.warning(self, "匯出", "尚未有任何標記。")
                return
            start_idx = annotated_indices[0]
            end_idx = annotated_indices[-1]
        else:
            start_idx = self._inference_start_idx
            end_idx = self._inference_end_idx

        frame_indices = list(range(start_idx, end_idx + 1))

        if not frame_indices:
            QMessageBox.warning(
                self,
                "匯出",
                "Inference 區間沒有可匯出的 frame。\n請先確認傳播是否完成。",
            )
            return

        flagged = len({
            e.frame_idx
            for e in self.propagator.get_audit_list()
            if start_idx <= e.frame_idx <= end_idx
        })
        answer = QMessageBox.question(
            self,
            "確認匯出",
            "將匯出整個 inference 區間內所有 frame。\n"
            "沒有 box 的 frame 也會輸出空 JSON，方便資料集保持連續。\n\n"
            f"Inference 區間：frame {start_idx} - {end_idx}\n"
            f"將匯出：{len(frame_indices)} frames\n"
            f"待檢查：{flagged} frames\n"
            "格式：LabelMe / X-AnyLabeling 相容 JSON + JPG\n\n"
            "要繼續嗎？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        if answer != QMessageBox.Yes:
            return

        exported = skipped = 0
        total = len(frame_indices)
        self._show_progress(True, f"匯出中… 共 {total} frames")

        for done, idx in enumerate(frame_indices):
            ann        = self.propagator.annotations.get(idx)
            frame_path = self.video_core.frame_path(idx)
            json_path  = out_path / f"frame_{idx:06d}.json"
            img_path   = out_path / f"frame_{idx:06d}.jpg"

            # Ensure frame jpg exists (extract from video if not cached)
            if not frame_path.exists():
                try:
                    frame = self.video_core.read_frame(idx)
                    _cv2.imwrite(str(frame_path), frame,
                                 [_cv2.IMWRITE_JPEG_QUALITY, 95])
                except Exception:
                    skipped += 1
                    continue

            # Copy jpg to output folder
            _shutil.copy2(frame_path, img_path)

            # Write JSON next to the image
            boxes = ann.boxes if ann else []
            self.label_bridge.write_boxes(
                json_path, img_path, boxes, meta.width, meta.height
            )

            exported += 1
            self._set_progress_value(done + 1, total, f"Exporting frame {done + 1} / {total}")

        manifest_path = out_path / "manifest.json"
        zip_path = out_path / "export_package.zip"
        manifest = {
            "schema_version": 1,
            "app": "video_annotator",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "source_video": str(self.video_core.video_path) if self.video_core.video_path else None,
            "output_dir": str(out_path),
            "frame_range": {"start": start_idx, "end": end_idx, "count": len(frame_indices)},
            "exported_frames": exported,
            "skipped_frames": skipped,
            "annotated_frames": sum(1 for idx in frame_indices if idx in self.propagator.annotations),
            "flagged_frames": flagged,
            "image_size": {"width": meta.width, "height": meta.height},
            "fps": meta.fps,
            "classes": list(self.class_manager.labels),
            "files": {
                "images": [f"frame_{idx:06d}.jpg" for idx in frame_indices if (out_path / f"frame_{idx:06d}.jpg").exists()],
                "annotations": [f"frame_{idx:06d}.json" for idx in frame_indices if (out_path / f"frame_{idx:06d}.json").exists()],
            },
        }
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for file_path in sorted(out_path.iterdir()):
                if file_path.is_file() and file_path != zip_path:
                    zf.write(file_path, file_path.name)

        self._show_progress(False)
        self._last_export_dir = out_dir
        self.right_panel.update_export_summary(
            (start_idx, end_idx),
            annotated_frames=sum(1 for idx in frame_indices if idx in self.propagator.annotations),
            flagged_frames=flagged,
            output_dir=out_dir,
        )
        QMessageBox.information(
            self, "匯出完成",
            f"已匯出：{exported} 個 frames\n"
            f"略過：  {skipped} 個 frames（無法讀取影像）\n\n"
            f"資料夾：{out_dir}\n\n"
            f"用 LabelMe → File → Open Dir 開啟此資料夾即可看到所有標記。"
        )

    # ------------------------------------------------------------------
    # Frame navigation
    # ------------------------------------------------------------------

    def _on_frame_changed(self, frame_idx: int):
        self._current_frame_idx = frame_idx
        self.timeline.set_current_frame(frame_idx)
        self._show_frame(frame_idx)
        if self.video_core.meta and not self._preview_mode:
            t = self.video_core.frame_to_time(frame_idx)
            mins, secs = divmod(t, 60)
            self.lbl_status.setText(
                f"Frame {frame_idx}  ({int(mins)}:{secs:05.2f})"
            )

    def _on_range_changed(self, start_sec: float, end_sec: float):
        if not self.video_core.meta:
            return
        self._start_prepare_worker(
            start_sec, end_sec,
            on_finished=self._on_range_prepare_done,
        )

    def _show_frame(self, frame_idx: int):
        if not self.video_core.meta:
            return
        try:
            frame = self.video_core.read_frame(frame_idx)
        except Exception:
            return
        ann = self.propagator.get_annotation(frame_idx) if self.propagator else None
        thresholds = {
            label: self.class_manager.conf_threshold(label)
            for label in self.class_manager.labels
        }
        self.canvas.show_frame(frame, ann, thresholds)

    # ------------------------------------------------------------------
    # PrepareWorker (shared by all callers)
    # ------------------------------------------------------------------

    def _start_prepare_worker(
        self,
        start_sec: float,
        end_sec: float,
        on_finished,
    ):
        """Start PrepareWorker; update self.dino_engine when done."""
        self._show_progress(True, "Preparing…")

        worker = PrepareWorker(
            start_sec, end_sec,
            self.video_core,
            self._engine_holder,
        )
        worker.stage.connect(self._on_worker_stage)
        worker.progress.connect(
            lambda d, t: self._set_progress_value(d, t)
        )

        def _finished(frame_paths):
            # Sync engine reference back to main thread
            if self._engine_holder[0] is not None and self.dino_engine is None:
                self.dino_engine = self._engine_holder[0]
                self._rebuild_bridges()
            on_finished(frame_paths)

        worker.finished.connect(_finished)
        def _on_error(e):
            logger.error("Worker Error: %s", e)
            self._show_progress(False)
            self.btn_load_ann.setEnabled(True)
            if hasattr(self, "btn_preview_current"):
                self.btn_preview_current.setEnabled(True)
            QMessageBox.critical(self, "Worker Error", e)

        worker.error.connect(_on_error)
        self._worker = worker
        worker.start()

    def _on_worker_stage(self, msg: str):
        self.lbl_status.setText(msg)
        self.progress_bar.setValue(0)
        if hasattr(self, "progress_overlay_detail"):
            self.progress_overlay_detail.setText(msg)
            self.progress_overlay_bar.setValue(0)
            self.progress_overlay_percent.setText("0%")

    def _on_range_prepare_done(self, frame_paths: list):
        start_sec, end_sec = self.timeline.get_range()
        start_idx = self.video_core.time_to_frame(start_sec)
        end_idx   = self.video_core.time_to_frame(end_sec)
        self.timeline.bar.configure(
            self.video_core.meta.total_frames, start_idx, end_idx
        )
        self._show_progress(False)
        self.lbl_status.setText(
            "Features ready — use Load Annotation to propagate"
        )

    # ------------------------------------------------------------------
    # Propagation
    # ------------------------------------------------------------------

    def _run_propagation(
        self,
        anchor_idx: int,
        forward: bool = True,
        backward: bool = True,
        radius_seconds: float | None = None,
    ):
        if not self.propagator:
            return
        self._show_progress(True, f"Propagating from frame {anchor_idx}…")
        worker = PropagationWorker(
            self.propagator, anchor_idx,
            forward=forward, backward=backward, radius_seconds=radius_seconds,
        )
        worker.progress.connect(
            lambda d, t: self._set_progress_value(d, t)
        )
        worker.finished.connect(self._on_propagation_done)
        worker.error.connect(
            lambda e: (
                logger.error("Propagation Error: %s", e),
                self._show_progress(False),
                QMessageBox.critical(self, "Propagation Error", e),
            )
        )
        self._worker = worker
        worker.start()

    def _on_propagation_done(self):
        self._show_progress(False)
        self.btn_load_ann.setEnabled(True)

        # Write every propagated annotation to disk so YoloExporter can find them
        meta = self.video_core.meta
        for idx, ann in self.propagator.annotations.items():
            if ann.is_anchor:
                continue  # anchor JSON was already copied at load time
            label_path = self.video_core.label_path(idx)
            frame_path = self.video_core.frame_path(idx)
            if frame_path.exists():
                self.label_bridge.write_boxes(
                    label_path, frame_path, ann.boxes, meta.width, meta.height
                )

        audit = self.propagator.get_audit_list()
        n = len(audit)
        n_written = sum(
            1 for ann in self.propagator.annotations.values() if not ann.is_anchor
        )
        self.lbl_status.setText(
            f"Propagation complete — {n_written} frames written"
            + (f", {n} flagged (see Flagged Frames)" if n else ", no issues")
        )
        self._refresh_timeline_states()
        self.right_panel.update_audit(audit)
        if self._inference_start_idx is not None and self._inference_end_idx is not None:
            self.right_panel.update_export_summary(
                (self._inference_start_idx, self._inference_end_idx),
                annotated_frames=len(self.propagator.annotations),
                flagged_frames=len({e.frame_idx for e in audit}),
                output_dir=self._last_export_dir,
            )
        self._show_frame(self._current_frame_idx)

    def _refresh_timeline_states(self):
        if not self.propagator:
            return
        audit_frames = {e.frame_idx for e in self.propagator.audit_list}
        for idx, ann in self.propagator.annotations.items():
            if ann.is_anchor:
                state = "anchor"
            elif idx in audit_frames:
                state = "anomaly"
            else:
                state = "ai"
            self.timeline.set_frame_state(idx, state)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _rebuild_bridges(self):
        exe = self._cfg.get("labelme", {}).get("exe_path", "")
        self.label_bridge = LabelBridge(exe_path=exe)
        if self.dino_engine is not None:
            self.propagator = Propagator(
                video_core=self.video_core,
                dino_engine=self.dino_engine,
                class_manager=self.class_manager,
                radius_seconds=self._cfg.get("propagation", {}).get(
                    "radius_seconds", 3
                ),
                top_k=self._cfg.get("propagation", {}).get("top_k_patches", 20),
                ransac_reproj=self._cfg.get("propagation", {}).get(
                    "ransac_reproj_threshold", 14
                ),
            )

    def _show_progress(self, visible: bool, message: str = ""):
        self.progress_bar.setVisible(visible)
        if hasattr(self, "progress_overlay"):
            self.progress_overlay.setVisible(visible)
        if visible:
            self.progress_bar.setValue(0)
            if hasattr(self, "progress_overlay_bar"):
                title = "Inference 執行中"
                lowered = message.lower()
                if "export" in lowered or "匯出" in message:
                    title = "匯出中"
                elif "prepar" in lowered or "feature" in lowered or "dino" in lowered:
                    title = "準備 inference 資料"
                self.progress_overlay_title.setText(title)
                self.progress_overlay_bar.setValue(0)
                self.progress_overlay_percent.setText("0%")
                self.progress_overlay_detail.setText(message or "Working...")
            if message:
                self.lbl_status.setText(message)
        elif hasattr(self, "progress_overlay_detail"):
            self.progress_overlay_detail.setText("")

    def _set_progress_value(self, done: int, total: int, message: str = ""):
        pct = int(done / total * 100) if total else 0
        pct = max(0, min(100, pct))
        self.progress_bar.setValue(pct)
        if hasattr(self, "progress_overlay_bar"):
            self.progress_overlay_bar.setValue(pct)
            self.progress_overlay_percent.setText(f"{pct}%  ({done} / {total})")
            if message:
                self.progress_overlay_detail.setText(message)
            else:
                self.progress_overlay_detail.setText(f"Processing frame {done} / {total}")

    @staticmethod
    def _load_config() -> dict:
        try:
            with open(CONFIG_PATH, encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        except FileNotFoundError:
            cfg = {}
        env_root = os.environ.get("CIM_REPO_ROOT", "")
        if env_root and Path(env_root).exists():
            project_root = Path(env_root)
        elif APP_ROOT.name == "LabelMe_Dino_launcher" and APP_ROOT.parent.name == "dist":
            project_root = APP_ROOT.parents[2]
        elif APP_ROOT.name == "LabelMe_Dino":
            project_root = APP_ROOT.parent
        else:
            project_root = Path.cwd()

        labelme_candidates = [
            os.environ.get("LABELME_EXE"),
            APP_ROOT / ".venv" / "Scripts" / "labelme.exe",
            project_root / "LabelMe_Dino" / ".venv" / "Scripts" / "labelme.exe",
            Path(sys.executable).parent / "labelme.exe",
            Path(sys.executable).parent / "labelme",
            shutil.which("labelme"),
        ]
        for c in labelme_candidates:
            if c and Path(c).exists():
                cfg.setdefault("labelme", {})["exe_path"] = str(Path(c))
                break
        cfg.setdefault("x_anylabeling", {}).setdefault("exe_path", "")

        xany_candidates = [
            os.environ.get("XANYLABELING_EXE"),
            project_root / ".venv-xanylabeling" / "Scripts" / "xanylabeling.exe",
        ]
        for c in xany_candidates:
            if c and Path(c).exists():
                cfg["x_anylabeling"]["exe_path"] = str(Path(c))
                break

        for name in (
            "xanylabeling",
            "xanylabeling.exe",
            "x-anylabeling",
            "x-anylabeling.exe",
            "anylabeling",
            "anylabeling.exe",
            "X-AnyLabeling.exe",
        ):
            if cfg["x_anylabeling"].get("exe_path"):
                break
            found = shutil.which(name)
            if found:
                cfg["x_anylabeling"]["exe_path"] = found
                break
        return cfg

    def _save_config(self):
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            yaml.dump(self._cfg, f, default_flow_style=False, allow_unicode=True)

    def closeEvent(self, event):
        if self.label_bridge:
            self.label_bridge.close()
        self.video_core.release()
        super().closeEvent(event)
