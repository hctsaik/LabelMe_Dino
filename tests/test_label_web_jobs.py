from __future__ import annotations

import json
import os
import threading
import time
from http.client import HTTPConnection
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from label_web.sam import propose_masks
from label_web.server import bind_state, create_server
import label_web.server as label_server


def _http(httpd, method: str, path: str, body=None):
    host, port = httpd.server_address[:2]
    conn = HTTPConnection(host, port, timeout=30)
    raw = json.dumps(body).encode() if body is not None else None
    headers = {"Content-Type": "application/json"} if raw else {}
    conn.request(method, path, body=raw, headers=headers)
    res = conn.getresponse()
    data = res.read()
    conn.close()
    return res.status, res.getheader("Content-Type"), data


def _tiny_avi(path: Path, n: int = 10) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
    assert writer.isOpened()
    for i in range(n):
        frame = np.zeros((48, 64, 3), np.uint8)
        frame[:] = (20 + i * 8, 40, 80)
        cv2.rectangle(frame, (8, 8), (24, 24), (0, 200, 0), -1)
        writer.write(frame)
    writer.release()


def test_grabcut_not_empty_rectangle():
    img = np.zeros((80, 100, 3), np.uint8)
    img[20:60, 20:70] = (30, 180, 40)
    cands = propose_masks(img, (15, 15, 75, 65))
    assert len(cands) == 3
    assert cands[0]["source"] == "grabcut"
    assert len(cands[0]["png"]) > 50


def test_prepare_video_extracts_frames(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("LABEL_WEB_SKIP_DINO", "1")
    monkeypatch.setenv("LABEL_WEB_DATA", str(tmp_path / "data"))
    video = tmp_path / "clip.avi"
    _tiny_avi(video)
    bind_state(tmp_path / "empty", tmp_path / "data")
    (tmp_path / "empty").mkdir()
    httpd = create_server("127.0.0.1", 0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        st, _, raw = _http(
            httpd,
            "POST",
            "/api/jobs/prepare_video",
            {"video_path": str(video), "start_sec": 0, "end_sec": 0.6},
        )
        assert st == 200
        job_id = json.loads(raw)["job_id"]
        rec = None
        for _ in range(40):
            st, _, raw = _http(httpd, "GET", "/api/jobs/" + job_id)
            rec = json.loads(raw)
            if rec["status"] in {"succeeded", "failed"}:
                break
            time.sleep(0.15)
        assert rec["status"] == "succeeded", rec
        assert rec["result"]["frames"] >= 3
        frames = list((tmp_path / "data" / "video_session" / "frames").glob("frame_*.jpg"))
        assert len(frames) >= 3
        st, _, raw = _http(httpd, "GET", "/api/task")
        task = json.loads(raw)
        assert len(task["images"]) >= 3
    finally:
        httpd.shutdown()
        label_server.STATE = None


def test_sam_propose_http(tmp_path: Path):
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    Image.new("RGB", (80, 60), (20, 80, 20)).save(img_dir / "a.png")
    bind_state(tmp_path, tmp_path / "data")
    httpd = create_server("127.0.0.1", 0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        st, _, raw = _http(
            httpd,
            "POST",
            "/api/sam/propose",
            {"index": 0, "roi": [10, 10, 70, 50]},
        )
        assert st == 200, raw
        cands = json.loads(raw)["candidates"]
        assert len(cands) == 3
        st, ctype, blob = _http(httpd, "GET", "/api/sam/" + cands[0]["candidate_id"] + "/png")
        assert st == 200
        assert ctype == "image/png"
        assert blob[:8] == b"\x89PNG\r\n\x1a\n"
        st, _, raw = _http(httpd, "POST", "/api/sam/accept", {"candidate_id": cands[0]["candidate_id"]})
        assert st == 200
        assert (img_dir / "a.mask.png").is_file()
    finally:
        httpd.shutdown()
        label_server.STATE = None
