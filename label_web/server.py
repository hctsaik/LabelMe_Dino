"""Label Web HTTP worker for NativeApp iframe. No PyQt, no cloud."""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from . import jobs
from .sam import propose_masks
from .task import LabelState

STATIC = Path(__file__).resolve().parent / "static"
STATE_LOCK = threading.Lock()
STATE: LabelState | None = None
SAM_LAST: dict[str, Any] | None = None
SESSION_DIR: Path | None = None


def _err(code: str, message: str) -> dict[str, str]:
    return {"code": code, "message": message}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        return

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, payload: Any) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send(status, data, "application/json; charset=utf-8")

    def _state(self) -> LabelState:
        st = STATE
        if st is None:
            raise RuntimeError("not ready")
        return st

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "content-type")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        try:
            if path == "/health":
                with STATE_LOCK:
                    st = STATE
                    n = len(st.images) if st else 0
                ms = jobs.model_status()
                return self._json(
                    200,
                    {
                        "ok": True,
                        "app": "label-web",
                        "model_ready": ms["model_ready"],
                        "n": n,
                    },
                )
            if path == "/api/status":
                with STATE_LOCK:
                    st = self._state()
                    payload = {
                        "ok": True,
                        "quiz": st.quiz,
                        "n": len(st.images),
                        **jobs.model_status(),
                    }
                return self._json(200, payload)
            if path in {"/", "/index.html"}:
                return self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
            if path.startswith("/static/"):
                rel = path[len("/static/") :]
                fp = (STATIC / rel).resolve()
                if STATIC.resolve() not in fp.parents and fp != STATIC.resolve():
                    return self._json(400, _err("INVALID_INPUT", "bad path"))
                if not fp.is_file():
                    return self._json(404, _err("INVALID_INPUT", "missing static file"))
                ctype, _ = mimetypes.guess_type(str(fp))
                return self._send(200, fp.read_bytes(), ctype or "application/octet-stream")
            if path == "/api/task":
                with STATE_LOCK:
                    return self._json(200, self._state().task_payload())
            if path.startswith("/api/image/"):
                idx = _index(path)
                with STATE_LOCK:
                    img = self._state().image(idx)
                    data = img.read_bytes()
                    name = img.name
                ctype, _ = mimetypes.guess_type(name)
                return self._send(200, data, ctype or "image/jpeg")
            if path.startswith("/api/annotation/"):
                idx = _index(path)
                with STATE_LOCK:
                    st = self._state()
                    img = st.image(idx)
                    sidecar = st.sidecar_path(img)
                    if sidecar.is_file():
                        doc = json.loads(sidecar.read_text(encoding="utf-8"))
                    else:
                        doc = {"shapes": [], "imagePath": img.name}
                return self._json(200, doc)
            if path.startswith("/api/jobs/"):
                job_id = path.rsplit("/", 1)[-1]
                rec = jobs.get_job(job_id)
                if not rec:
                    return self._json(404, _err("INVALID_INPUT", "unknown job"))
                public = dict(rec)
                if rec.get("status") == "succeeded" and rec.get("kind") == "prepare_video":
                    session = (rec.get("result") or {}).get("session_dir")
                    if session:
                        with STATE_LOCK:
                            global SESSION_DIR
                            SESSION_DIR = Path(session)
                            self._state().switch_root(SESSION_DIR)
                            public["task"] = self._state().task_payload()
                return self._json(200, public)
            if path.startswith("/api/sam/") and path.endswith("/png"):
                cand_id = path.split("/")[-2]
                with STATE_LOCK:
                    blob = None
                    if SAM_LAST:
                        for c in SAM_LAST.get("candidates") or []:
                            if c["candidate_id"] == cand_id:
                                blob = c["png"]
                                break
                if not blob:
                    return self._json(404, _err("INVALID_INPUT", "no candidate"))
                return self._send(200, blob, "image/png")
            return self._json(404, _err("INVALID_INPUT", "not found"))
        except (ValueError, IndexError):
            return self._json(404, _err("INVALID_INPUT", "no image"))
        except Exception as exc:  # noqa: BLE001
            return self._json(500, _err("JOB_FAILED", str(exc)))

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8") or "{}")
        except json.JSONDecodeError:
            return self._json(400, _err("INVALID_INPUT", "bad json"))
        if not isinstance(payload, dict):
            payload = {}
        try:
            if path.startswith("/api/annotation/"):
                return self._save_annotation(_index(path), payload)
            if path == "/api/open":
                return self._open(payload)
            if path == "/api/host/pick-folder":
                return self._pick_and_open("folder")
            if path == "/api/host/pick-video":
                return self._pick_video()
            if path == "/api/jobs/prepare_video":
                return self._prepare(payload)
            if path == "/api/jobs/propagate":
                return self._propagate(payload)
            if path == "/api/sam/propose":
                return self._sam_propose(payload)
            if path == "/api/sam/accept":
                return self._sam_accept(payload)
            return self._json(404, _err("INVALID_INPUT", "not found"))
        except (ValueError, IndexError):
            return self._json(404, _err("INVALID_INPUT", "no image"))
        except Exception as exc:  # noqa: BLE001
            return self._json(500, _err("JOB_FAILED", str(exc)))

    def _save_annotation(self, index: int, payload: dict[str, Any]) -> None:
        with STATE_LOCK:
            st = self._state()
            if st.quiz:
                return self._json(
                    403,
                    _err("QUIZ_NO_WRITE", "blind-quiz 禁止把標註寫回 GT"),
                )
            img = st.image(index)
            shapes = payload.get("shapes")
            if shapes is None:
                return self._json(400, _err("INVALID_INPUT", "shapes required"))
            if not isinstance(shapes, list):
                return self._json(400, _err("INVALID_INPUT", "shapes must be a list"))
            doc = {
                "version": "5.0.1",
                "flags": {},
                "shapes": shapes,
                "imagePath": img.name,
                "imageData": None,
                "imageWidth": payload.get("imageWidth"),
                "imageHeight": payload.get("imageHeight"),
            }
            dest = st.sidecar_path(img)
            _atomic_write(dest, doc)
            backup = st.backup_path(img)
            _atomic_write(backup, doc)
        return self._json(200, {"ok": True, "path": dest.name})

    def _open(self, payload: dict[str, Any]) -> None:
        raw = str(payload.get("root") or "").strip()
        if not raw:
            return self._json(400, _err("INVALID_INPUT", "root required"))
        root = Path(raw)
        if not root.is_dir():
            return self._json(400, _err("INVALID_INPUT", "root is not a directory"))
        with STATE_LOCK:
            self._state().switch_root(root)
            task = self._state().task_payload()
        return self._json(200, {"ok": True, "task": task})

    def _pick_and_open(self, _kind: str) -> None:
        chosen = pick_folder_dialog()
        if not chosen:
            return self._json(400, _err("CANCELLED", "沒有選擇資料夾"))
        with STATE_LOCK:
            self._state().switch_root(Path(chosen))
            task = self._state().task_payload()
        return self._json(200, {"ok": True, "task": task, "display_name": Path(chosen).name})

    def _pick_video(self) -> None:
        chosen = pick_video_dialog()
        if not chosen:
            return self._json(400, _err("CANCELLED", "沒有選擇影片"))
        return self._json(200, {"ok": True, "video_path": chosen})

    def _prepare(self, payload: dict[str, Any]) -> None:
        video = str(payload.get("video_path") or "").strip()
        start = float(payload.get("start_sec") or 0)
        end = float(payload.get("end_sec") or 2)
        data = Path(os.environ.get("LABEL_WEB_DATA") or (self._state().data_dir))
        session = data / "video_session"
        try:
            job_id = jobs.start_prepare(video, start, end, session)
        except (FileNotFoundError, ValueError) as exc:
            return self._json(400, _err("INVALID_INPUT", str(exc)))
        return self._json(200, {"job_id": job_id, "status": "queued"})

    def _propagate(self, payload: dict[str, Any]) -> None:
        global SESSION_DIR
        session = SESSION_DIR or (Path(os.environ.get("LABEL_WEB_DATA") or ".") / "video_session")
        if not (session / "frames").is_dir():
            return self._json(400, _err("INVALID_INPUT", "先抽幀（prepare_video）"))
        boxes = payload.get("annotation") or payload.get("shapes") or []
        if isinstance(boxes, dict):
            boxes = boxes.get("shapes") or []
        job_id = jobs.start_propagate(
            session_dir=session,
            anchor_frame=int(payload.get("anchor_frame") or 0),
            boxes=list(boxes),
            radius_seconds=float(payload.get("radius_seconds") or 3),
            classes=list(payload.get("classes") or self._state().classes),
        )
        return self._json(200, {"job_id": job_id, "status": "queued"})

    def _sam_propose(self, payload: dict[str, Any]) -> None:
        global SAM_LAST
        idx = int(payload.get("index") or 0)
        roi = payload.get("roi") or []
        if len(roi) != 4:
            return self._json(400, _err("INVALID_INPUT", "roi must be [x0,y0,x1,y1]"))
        with STATE_LOCK:
            img_path = self._state().image(idx)
        import cv2

        bgr = cv2.imread(str(img_path))
        if bgr is None:
            return self._json(400, _err("INVALID_INPUT", "cannot read image"))
        xyxy = (int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3]))
        cands = propose_masks(bgr, xyxy)
        SAM_LAST = {"index": idx, "candidates": cands}
        return self._json(
            200,
            {
                "candidates": [
                    {"candidate_id": c["candidate_id"], "source": c["source"], "score": c["score"], "xyxy": c["xyxy"]}
                    for c in cands
                ]
            },
        )

    def _sam_accept(self, payload: dict[str, Any]) -> None:
        cand_id = str(payload.get("candidate_id") or "")
        with STATE_LOCK:
            if not SAM_LAST:
                return self._json(400, _err("INVALID_INPUT", "先 propose"))
            chosen = next((c for c in SAM_LAST["candidates"] if c["candidate_id"] == cand_id), None)
            if not chosen:
                return self._json(400, _err("INVALID_INPUT", "unknown candidate"))
            img = self._state().image(int(SAM_LAST["index"]))
            dest = img.with_suffix(".mask.png")
            dest.write_bytes(chosen["png"])
        return self._json(200, {"ok": True, "path": dest.name})


def _index(path: str) -> int:
    return int(path.rstrip("/").rsplit("/", 1)[-1])


def _atomic_write(path: Path, doc: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def bind_state(root: Path, data_dir: Path) -> LabelState:
    global STATE
    STATE = LabelState(root, data_dir)
    return STATE


def pick_folder_dialog() -> str | None:
    if os.name != "nt":
        return None
    script = (
        "Add-Type -AssemblyName System.Windows.Forms; "
        "$d = New-Object System.Windows.Forms.FolderBrowserDialog; "
        "$d.Description = '選擇要標註的圖片資料夾'; "
        "if ($d.ShowDialog() -eq 'OK') { Write-Output $d.SelectedPath }"
    )
    try:
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-STA", "-Command", script],
            capture_output=True, text=True, timeout=300,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    lines = (completed.stdout or "").strip().splitlines()
    return lines[-1] if lines else None


def pick_video_dialog() -> str | None:
    if os.name != "nt":
        return None
    script = (
        "Add-Type -AssemblyName System.Windows.Forms; "
        "$d = New-Object System.Windows.Forms.OpenFileDialog; "
        "$d.Filter = 'Video (*.mp4;*.avi;*.mkv)|*.mp4;*.avi;*.mkv|All|*.*'; "
        "$d.Title = '選擇影片'; "
        "if ($d.ShowDialog() -eq 'OK') { Write-Output $d.FileName }"
    )
    try:
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-STA", "-Command", script],
            capture_output=True, text=True, timeout=300,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    lines = (completed.stdout or "").strip().splitlines()
    return lines[-1] if lines else None


def resolve_root(cli_root: str) -> Path:
    """Never default to the source repo (it contains demo cabinets / animal frames)."""
    task = os.environ.get("LABEL_WEB_TASK_DIR", "").strip()
    if task:
        return Path(task)
    if cli_root:
        return Path(cli_root)
    empty = Path(os.environ.get("LABEL_WEB_DATA") or (Path.home() / ".label-web")) / "empty"
    empty.mkdir(parents=True, exist_ok=True)
    return empty


def create_server(host: str, port: int) -> ThreadingHTTPServer:
    if host not in {"127.0.0.1", "localhost", "::1"}:
        host = "127.0.0.1"
    return ThreadingHTTPServer((host, port), Handler)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="label_web")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--root", default="")
    args = parser.parse_args(argv)
    root = resolve_root(args.root)
    data = Path(os.environ.get("LABEL_WEB_DATA") or (root / "_label_web_data"))
    bind_state(root, data)
    httpd = create_server(args.host, args.port)
    print(f"label-web http://127.0.0.1:{args.port}", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        httpd.shutdown()
