from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Optional

from src.dino_engine import DinoEngine, INPUT_SIZE, PATCH_SIZE, N_PATCHES
from src.video_core import VideoCore
from src.class_manager import ClassManager


@dataclass
class LabeledBox:
    label: str
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels
    confidence: float = 1.0
    ai_generated: bool = False


@dataclass
class FrameAnnotation:
    frame_idx: int
    boxes: list[LabeledBox] = field(default_factory=list)
    is_anchor: bool = False


@dataclass
class AuditEntry:
    frame_idx: int
    label: str
    confidence: float
    reason: str  # "low_confidence" | "object_lost"


class Propagator:
    def __init__(
        self,
        video_core: VideoCore,
        dino_engine: DinoEngine,
        class_manager: ClassManager,
        radius_seconds: float = 3.0,
        top_k: int = 20,
        ransac_reproj: float = PATCH_SIZE,
    ):
        self.vc = video_core
        self.engine = dino_engine
        self.cm = class_manager
        self.radius_seconds = radius_seconds
        self.top_k = top_k
        self.ransac_reproj = ransac_reproj

        self.annotations: dict[int, FrameAnnotation] = {}
        self.audit_list: list[AuditEntry] = []

    # ------------------------------------------------------------------
    # Anchor management
    # ------------------------------------------------------------------

    def set_anchor(self, frame_idx: int, boxes: list[LabeledBox]):
        self.annotations[frame_idx] = FrameAnnotation(
            frame_idx=frame_idx, boxes=boxes, is_anchor=True
        )

    def get_annotation(self, frame_idx: int) -> Optional[FrameAnnotation]:
        return self.annotations.get(frame_idx)

    def anchor_indices(self) -> list[int]:
        return sorted(idx for idx, ann in self.annotations.items() if ann.is_anchor)

    # ------------------------------------------------------------------
    # Forward / Backward propagation
    # ------------------------------------------------------------------

    def propagate_from_anchor(
        self,
        anchor_idx: int,
        forward: bool = True,
        backward: bool = True,
        radius_seconds: Optional[float] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> list[AuditEntry]:
        assert self.vc.meta, "VideoCore not loaded"

        radius = radius_seconds if radius_seconds is not None else self.radius_seconds
        radius_frames = int(radius * self.vc.meta.fps)
        start_idx = max(0, anchor_idx - radius_frames) if backward else anchor_idx
        end_idx   = min(self.vc.meta.total_frames - 1, anchor_idx + radius_frames) if forward else anchor_idx

        anchor_ann = self.annotations.get(anchor_idx)
        if not anchor_ann:
            raise ValueError(f"No anchor annotation at frame {anchor_idx}")

        # Clear non-anchor AI annotations in range
        for idx in list(self.annotations.keys()):
            if start_idx <= idx <= end_idx:
                ann = self.annotations[idx]
                if not ann.is_anchor:
                    del self.annotations[idx]

        bwd_count = (anchor_idx - start_idx) if backward else 0
        fwd_count = (end_idx - anchor_idx) if forward else 0
        total     = bwd_count + fwd_count
        done      = 0
        new_audits: list[AuditEntry] = []

        # Backward pass
        prev_ann = anchor_ann
        for idx in range(anchor_idx - 1, start_idx - 1, -1):
            audits = self._propagate_step(prev_ann, idx)
            new_audits.extend(audits)
            prev_ann = self.annotations.get(idx, prev_ann)
            done += 1
            if progress_callback:
                progress_callback(done, total)

        # Forward pass
        prev_ann = anchor_ann
        for idx in range(anchor_idx + 1, end_idx + 1):
            audits = self._propagate_step(prev_ann, idx)
            new_audits.extend(audits)
            prev_ann = self.annotations.get(idx, prev_ann)
            done += 1
            if progress_callback:
                progress_callback(done, total)

        self._merge_audit_entries(new_audits)
        return new_audits

    def _propagate_step(
        self,
        src_ann: FrameAnnotation,
        tgt_idx: int,
    ) -> list[AuditEntry]:
        meta = self.vc.meta

        # We need DINO features for confidence scoring
        src_feat_path = self.vc.feature_path(src_ann.frame_idx)
        tgt_feat_path = self.vc.feature_path(tgt_idx)
        if not src_feat_path.exists() or not tgt_feat_path.exists():
            return []

        src_feat = DinoEngine.load(src_feat_path)
        tgt_feat = DinoEngine.load(tgt_feat_path)

        # Read actual frames for optical flow
        try:
            src_frame = self.vc.read_frame(src_ann.frame_idx)
            tgt_frame = self.vc.read_frame(tgt_idx)
            use_of = True
        except Exception:
            use_of = False

        new_boxes: list[LabeledBox] = []
        audits: list[AuditEntry] = []

        for box in src_ann.boxes:
            threshold = self.cm.conf_threshold(box.label)

            if use_of:
                new_bbox, of_conf = _optical_flow_track(
                    src_frame, box.bbox, tgt_frame,
                    meta.width, meta.height,
                )
                # DINO center-patch similarity as a semantic sanity check
                dino_conf = _center_patch_similarity(
                    src_feat, box.bbox, tgt_feat, new_bbox, meta
                )
                # Combined confidence: OF tells us tracking succeeded;
                # DINO tells us the object still looks the same
                conf = of_conf * 0.5 + dino_conf * 0.5
            else:
                # Fall back to pure DINO matching
                new_bbox, conf = self.engine.match_bbox(
                    src_feat, box.bbox, tgt_feat,
                    meta.width, meta.height,
                    top_k=self.top_k,
                    ransac_reproj=self.ransac_reproj,
                )

            new_boxes.append(LabeledBox(
                label=box.label,
                bbox=new_bbox,
                confidence=conf,
                ai_generated=True,
            ))
            if conf < threshold:
                audits.append(AuditEntry(
                    frame_idx=tgt_idx,
                    label=box.label,
                    confidence=conf,
                    reason="object_lost" if conf < 0.2 else "low_confidence",
                ))

        self.annotations[tgt_idx] = FrameAnnotation(
            frame_idx=tgt_idx, boxes=new_boxes, is_anchor=False
        )
        return audits

    # ------------------------------------------------------------------
    # Audit management
    # ------------------------------------------------------------------

    def _merge_audit_entries(self, new_entries: list[AuditEntry]):
        existing_keys = {(e.frame_idx, e.label) for e in self.audit_list}
        for e in new_entries:
            key = (e.frame_idx, e.label)
            if key not in existing_keys:
                self.audit_list.append(e)
                existing_keys.add(key)
        self.audit_list.sort(key=lambda e: e.frame_idx)

    def clear_audit_for_frame(self, frame_idx: int):
        self.audit_list = [e for e in self.audit_list if e.frame_idx != frame_idx]

    def get_audit_list(self) -> list[AuditEntry]:
        return list(self.audit_list)


# ------------------------------------------------------------------
# Optical flow tracking  (module-level helper, no class needed)
# ------------------------------------------------------------------

def _optical_flow_track(
    src_frame: np.ndarray,
    bbox: tuple[float, float, float, float],
    tgt_frame: np.ndarray,
    img_w: int,
    img_h: int,
) -> tuple[tuple[float, float, float, float], float]:
    """
    Track bbox using Lucas-Kanade optical flow with median-translation.

    Key design choices to prevent bbox growth:
    - Tracking points are placed in the INNER 50 % of the bbox (25 % inset on
      every side).  This keeps them away from the bottom edge (road surface)
      and all other background regions that bleed into the bbox.
    - Only the MEDIAN displacement is used — no scale estimation.
      Frame-to-frame translation error is tiny; scale drift from bad inliers
      is the bigger enemy.
    - The original bbox is translated as a rigid body; width and height are
      always preserved exactly.
    """
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1

    # Inner region: 25 % inset on each side
    ix1 = x1 + w * 0.25
    iy1 = y1 + h * 0.25
    ix2 = x2 - w * 0.25
    iy2 = y2 - h * 0.25
    icx = (ix1 + ix2) / 2
    icy = (iy1 + iy2) / 2

    # 9-point grid over the inner region
    pts = np.array([
        [ix1, iy1], [icx, iy1], [ix2, iy1],
        [ix1, icy], [icx, icy], [ix2, icy],
        [ix1, iy2], [icx, iy2], [ix2, iy2],
    ], dtype=np.float32).reshape(-1, 1, 2)

    src_gray = cv2.cvtColor(src_frame, cv2.COLOR_BGR2GRAY)
    tgt_gray = cv2.cvtColor(tgt_frame, cv2.COLOR_BGR2GRAY)

    new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
        src_gray, tgt_gray, pts, None,
        winSize=(31, 31),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )

    good_mask = status.ravel() == 1
    if good_mask.sum() < 3:
        return bbox, 0.1   # object occluded or out of frame

    good_old = pts[good_mask].reshape(-1, 2)
    good_new = new_pts[good_mask].reshape(-1, 2)

    # Median displacement — robust to any remaining background outliers
    dx = float(np.median(good_new[:, 0] - good_old[:, 0]))
    dy = float(np.median(good_new[:, 1] - good_old[:, 1]))

    # Translate original bbox rigidly; size is never changed
    nx1 = float(np.clip(x1 + dx, 0, img_w))
    ny1 = float(np.clip(y1 + dy, 0, img_h))
    nx2 = float(np.clip(x2 + dx, 0, img_w))
    ny2 = float(np.clip(y2 + dy, 0, img_h))

    return (nx1, ny1, nx2, ny2), float(good_mask.mean())


def _center_patch_similarity(
    src_feat: np.ndarray,
    src_bbox: tuple,
    tgt_feat: np.ndarray,
    tgt_bbox: tuple,
    meta,
) -> float:
    """
    Cosine similarity between the center DINO patch of src_bbox and tgt_bbox.
    Used as a semantic confidence check: did the object still look the same?
    """
    def center_vec(feat, bbox):
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2 * INPUT_SIZE / meta.width
        cy = (y1 + y2) / 2 * INPUT_SIZE / meta.height
        px = int(np.clip(cx / PATCH_SIZE, 0, N_PATCHES - 1))
        py = int(np.clip(cy / PATCH_SIZE, 0, N_PATCHES - 1))
        return feat[py, px]

    va = center_vec(src_feat, src_bbox)
    vb = center_vec(tgt_feat, tgt_bbox)
    return float(np.clip(np.dot(va, vb), 0.0, 1.0))
