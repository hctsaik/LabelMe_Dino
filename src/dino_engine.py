from __future__ import annotations

import numpy as np
import torch
import cv2
from pathlib import Path
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

MODEL_ID = "facebook/dinov2-small"
INPUT_SIZE = 518   # ViT-S/14: 518 / 14 = 37 patches per side
PATCH_SIZE = 14
N_PATCHES = INPUT_SIZE // PATCH_SIZE   # 37
FEAT_DIM = 384                          # ViT-S hidden dim


class DinoEngine:
    def __init__(self, device: str = "auto"):
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.processor = AutoImageProcessor.from_pretrained(MODEL_ID)
        self.model = AutoModel.from_pretrained(MODEL_ID).to(self.device)
        self.model.eval()

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def extract(self, image: np.ndarray) -> np.ndarray:
        """
        Extract DINOv2 patch features from a BGR numpy frame.
        Returns float32 array [N_PATCHES, N_PATCHES, FEAT_DIM], L2-normalised.
        """
        pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        pil = pil.resize((INPUT_SIZE, INPUT_SIZE), Image.LANCZOS)

        # do_resize=False / do_center_crop=False: we already resized to INPUT_SIZE manually;
        # letting the processor re-resize would reduce to 224px and give wrong patch count.
        inputs = self.processor(
            images=pil,
            return_tensors="pt",
            do_resize=False,
            do_center_crop=False,
        ).to(self.device)

        with torch.no_grad():
            out = self.model(**inputs)

        # last_hidden_state: [1, 1 + N_PATCHES*N_PATCHES, FEAT_DIM]
        tokens = out.last_hidden_state[0, 1:, :].cpu().numpy()  # skip CLS
        n_tokens = tokens.shape[0]
        grid = int(n_tokens ** 0.5)
        features = tokens.reshape(grid, grid, -1).astype(np.float32)

        norms = np.linalg.norm(features, axis=-1, keepdims=True)
        features /= norms + 1e-8
        return features

    def extract_and_save(self, image: np.ndarray, path: str | Path):
        features = self.extract(image)
        np.save(str(path), features)
        return features

    @staticmethod
    def load(path: str | Path) -> np.ndarray:
        return np.load(str(path))

    # ------------------------------------------------------------------
    # BBox matching
    # ------------------------------------------------------------------

    def match_bbox(
        self,
        src_features: np.ndarray,
        src_bbox: tuple[float, float, float, float],
        tgt_features: np.ndarray,
        img_w: int,
        img_h: int,
        top_k: int = 20,
        ransac_reproj: float = PATCH_SIZE,
    ) -> tuple[tuple[float, float, float, float], float]:
        """
        Propagate src_bbox from src_features into tgt_features.

        Approach:
          1. Map src_bbox → patch indices in the [N_PATCHES, N_PATCHES] grid.
          2. Cosine-similarity match those patches against all target patches.
          3. Keep top-k matches; run RANSAC affine to estimate transform.
          4. Apply transform to src_bbox corners → new bbox.

        Returns (new_bbox, confidence).
        confidence = mean cosine similarity of top-k matched patches.
        """
        px1, py1, px2, py2 = self._bbox_to_patch_coords(src_bbox, img_w, img_h)

        src_region = src_features[py1:py2, px1:px2]   # [ph, pw, D]
        ph, pw, _ = src_region.shape
        if ph == 0 or pw == 0:
            return src_bbox, 0.0

        src_flat = src_region.reshape(-1, FEAT_DIM)               # [ph*pw, D]
        tgt_flat = tgt_features.reshape(-1, FEAT_DIM)             # [37*37, D]

        sim = src_flat @ tgt_flat.T                               # [ph*pw, 37*37]
        best_idx = np.argmax(sim, axis=1)                         # [ph*pw]
        best_sim = sim[np.arange(len(best_idx)), best_idx]

        k = min(top_k, len(best_idx))
        top_sel = np.argsort(best_sim)[-k:]
        confidence = float(np.mean(best_sim[top_sel]))

        src_pts, tgt_pts = self._build_correspondences(
            top_sel, best_idx, px1, py1, pw
        )

        new_bbox = self._ransac_transform(
            src_pts, tgt_pts, src_bbox, img_w, img_h, ransac_reproj
        )
        return new_bbox, confidence

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _bbox_to_patch_coords(
        bbox: tuple[float, float, float, float],
        img_w: int,
        img_h: int,
    ) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = bbox
        sx = INPUT_SIZE / img_w
        sy = INPUT_SIZE / img_h
        n = N_PATCHES
        px1 = max(0, int(x1 * sx / PATCH_SIZE))
        py1 = max(0, int(y1 * sy / PATCH_SIZE))
        px2 = min(n, max(px1 + 1, int(np.ceil(x2 * sx / PATCH_SIZE))))
        py2 = min(n, max(py1 + 1, int(np.ceil(y2 * sy / PATCH_SIZE))))
        return px1, py1, px2, py2

    @staticmethod
    def _build_correspondences(
        top_sel: np.ndarray,
        best_idx: np.ndarray,
        px1: int,
        py1: int,
        pw: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        src_pts, tgt_pts = [], []
        for i in top_sel:
            s_py = py1 + i // pw
            s_px = px1 + i % pw
            src_pts.append([(s_px + 0.5) * PATCH_SIZE, (s_py + 0.5) * PATCH_SIZE])

            t_flat = best_idx[i]
            t_py = t_flat // N_PATCHES
            t_px = t_flat % N_PATCHES
            tgt_pts.append([(t_px + 0.5) * PATCH_SIZE, (t_py + 0.5) * PATCH_SIZE])

        return np.array(src_pts, dtype=np.float32), np.array(tgt_pts, dtype=np.float32)

    @staticmethod
    def _ransac_transform(
        src_pts: np.ndarray,
        tgt_pts: np.ndarray,
        src_bbox: tuple[float, float, float, float],
        img_w: int,
        img_h: int,
        ransac_reproj: float,
    ) -> tuple[float, float, float, float]:
        sx = INPUT_SIZE / img_w
        sy = INPUT_SIZE / img_h
        x1, y1, x2, y2 = src_bbox

        if len(src_pts) >= 3:
            M, _ = cv2.estimateAffinePartial2D(
                src_pts,
                tgt_pts,
                method=cv2.RANSAC,
                ransacReprojThreshold=ransac_reproj,
            )
        else:
            M = None

        if M is None:
            return src_bbox

        corners = np.array(
            [[x1 * sx, y1 * sy],
             [x2 * sx, y1 * sy],
             [x2 * sx, y2 * sy],
             [x1 * sx, y2 * sy]],
            dtype=np.float32,
        )
        ones = np.ones((4, 1), dtype=np.float32)
        new_corners = (M @ np.hstack([corners, ones]).T).T  # [4, 2]

        nx1 = float(np.clip(new_corners[:, 0].min() / sx, 0, img_w))
        ny1 = float(np.clip(new_corners[:, 1].min() / sy, 0, img_h))
        nx2 = float(np.clip(new_corners[:, 0].max() / sx, 0, img_w))
        ny2 = float(np.clip(new_corners[:, 1].max() / sy, 0, img_h))
        return nx1, ny1, nx2, ny2
