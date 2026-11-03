from __future__ import annotations

import json
import threading
from http.client import HTTPConnection
from pathlib import Path

from PIL import Image

import label_web.server as label_server
from label_web.server import bind_state, create_server
from label_web.task import LabelState, is_quiz, sha_for_image


def _handoff(tmp: Path, *, task: str = "fresh", n: int = 2) -> Path:
    images = tmp / "images"
    images.mkdir(parents=True)
    items = []
    for i in range(n):
        sha = f"{i:064x}"
        Image.new("RGB", (64, 48), (40 + i * 80, 80, 120)).save(images / f"{sha}.jpg")
        items.append({"sha256": sha, "image": f"images/{sha}.jpg", "filename": f"orig_{i}.jpg"})
    (tmp / "_handoff.json").write_text(
        json.dumps({"handoff_id": "t1", "task": task, "class_options": ["door", "window"], "items": items}),
        encoding="utf-8",
    )
    (tmp / "classes.txt").write_text("door\nwindow\n", encoding="utf-8")
    return tmp


def _http(httpd, method: str, path: str, body=None):
    host, port = httpd.server_address[:2]
    conn = HTTPConnection(host, port, timeout=10)
    raw = json.dumps(body).encode() if body is not None else None
    headers = {"Content-Type": "application/json"} if raw else {}
    conn.request(method, path, body=raw, headers=headers)
    res = conn.getresponse()
    data = res.read()
    conn.close()
    return res.status, res.getheader("Content-Type"), data


def test_task_sha_and_quiz(tmp_path: Path):
    root = _handoff(tmp_path / "h")
    st = LabelState(root, tmp_path / "data")
    payload = st.task_payload()
    assert payload["images"][0]["sha256"] == "0" * 64
    assert payload["classes"][0] == "door"
    assert is_quiz(root) is False
    quiz = _handoff(tmp_path / "q", task="blind-quiz")
    assert is_quiz(quiz) is True
    assert sha_for_image(root, st.images[0]) == "0" * 64


def test_http_p0_roundtrip(tmp_path: Path):
    root = _handoff(tmp_path / "h")
    bind_state(root, tmp_path / "data")
    httpd = create_server("127.0.0.1", 0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        status, _, raw = _http(httpd, "GET", "/health")
        health = json.loads(raw)
        assert status == 200
        assert health["ok"] is True
        assert health["model_ready"] is False

        status, _, raw = _http(httpd, "GET", "/api/task")
        task = json.loads(raw)
        assert status == 200
        assert len(task["images"]) == 2
        assert task["images"][0]["sha256"] == "0" * 64

        status, ctype, raw = _http(httpd, "GET", "/api/image/0")
        assert status == 200
        assert ctype.startswith("image/")
        assert raw[:2] == b"\xff\xd8"

        box = {
            "shapes": [
                {
                    "label": "door",
                    "points": [[4, 5], [20, 22]],
                    "shape_type": "rectangle",
                }
            ],
            "imageWidth": 64,
            "imageHeight": 48,
        }
        status, _, raw = _http(httpd, "POST", "/api/annotation/0", box)
        assert status == 200
        sidecar = root / "images" / f"{'0'*64}.json"
        assert sidecar.is_file()
        saved = json.loads(sidecar.read_text(encoding="utf-8"))
        assert saved["imagePath"].endswith(".jpg")
        assert saved["shapes"][0]["points"][0] == [4, 5]
        backup = root / "annotations" / f"{'0'*64}.json"
        assert backup.is_file()

        status, _, raw = _http(httpd, "GET", "/")
        assert status == 200
        assert b"<canvas" in raw

        status, _, raw = _http(httpd, "POST", "/api/jobs/prepare_video", {"video_path": "x.mp4"})
        assert status == 400
        assert json.loads(raw)["code"] == "INVALID_INPUT"
    finally:
        httpd.shutdown()
        label_server.STATE = None


def test_blind_quiz_refuses_write(tmp_path: Path):
    root = _handoff(tmp_path / "q", task="blind-quiz")
    bind_state(root, tmp_path / "data")
    httpd = create_server("127.0.0.1", 0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        status, _, raw = _http(
            httpd,
            "POST",
            "/api/annotation/0",
            {"shapes": [{"label": "door", "points": [[0, 0], [1, 1]], "shape_type": "rectangle"}]},
        )
        assert status == 403
        assert json.loads(raw)["code"] == "QUIZ_NO_WRITE"
        assert not (root / "images" / f"{'0'*64}.json").exists()
    finally:
        httpd.shutdown()
        label_server.STATE = None
