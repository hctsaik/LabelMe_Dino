"""ROI segmentation without PyQt. GrabCut is the offline default."""

from __future__ import annotations

import hashlib
import io

import cv2
import numpy as np
from PIL import Image


def propose_masks(image_bgr: np.ndarray, roi: tuple[int, int, int, int]) -> list[dict]:
    x0, y0, x1, y1 = roi
    h, w = image_bgr.shape[:2]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    if x1 - x0 < 8 or y1 - y0 < 8:
        raise ValueError("ROI too small")
    rect = (x0, y0, x1 - x0, y1 - y0)
    out = []
    for i, iters in enumerate((1, 3, 5), start=1):
        mask = np.zeros((h, w), np.uint8)
        bgd = np.zeros((1, 65), np.float64)
        fgd = np.zeros((1, 65), np.float64)
        cv2.grabCut(image_bgr, mask, rect, bgd, fgd, iters, cv2.GC_INIT_WITH_RECT)
        binary = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        if int(binary.sum()) == 0:
            binary[y0:y1, x0:x1] = 255
        png = _png(binary)
        out.append(
            {
                "candidate_id": f"grabcut_{i}_{hashlib.sha256(png).hexdigest()[:10]}",
                "source": "grabcut",
                "score": round(0.7 + 0.08 * i, 3),
                "png": png,
                "xyxy": [x0, y0, x1, y1],
            }
        )
    return out


def _png(mask: np.ndarray) -> bytes:
    im = Image.fromarray(mask)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()
