"""Layout E2E: canvas must fill the stage, queue left, inspector right."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest
from playwright.sync_api import sync_playwright

from label_web.server import bind_state, create_server
import label_web.server as label_server

pytestmark = pytest.mark.e2e

SHOT = Path(__file__).resolve().parents[1] / "tmp" / "label_web_layout.png"


@pytest.fixture()
def empty_server(tmp_path: Path):
    root = tmp_path / "empty"
    root.mkdir()
    bind_state(root, tmp_path / "data")
    httpd = create_server("127.0.0.1", 0)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    host, port = httpd.server_address[:2]
    yield f"http://{host}:{port}"
    httpd.shutdown()
    label_server.STATE = None


def test_first_screen_is_empty_not_cabinet(empty_server: str):
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        page.goto(empty_server + "/", wait_until="networkidle")
        empty = page.locator("#empty")
        assert empty.is_visible()
        assert "還沒有要標的圖" in empty.inner_text()
        assert page.get_by_role("button", name="開啟圖片資料夾").is_visible()
        body = page.inner_text("body")
        assert "22b1d19" not in body
        assert "cabinet" not in body.lower()
        browser.close()


@pytest.fixture()
def server(tmp_path: Path):
    root = Path(r"C:\code\claude\LabelMe_Dino\tmp\label_web_evidence")
    if not root.is_dir():
        pytest.skip("no evidence images")
    bind_state(root, tmp_path / "data")
    httpd = create_server("127.0.0.1", 0)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    host, port = httpd.server_address[:2]
    yield f"http://{host}:{port}"
    httpd.shutdown()
    label_server.STATE = None


def test_workspace_layout_fills_stage(server: str):
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        page.goto(server + "/", wait_until="networkidle")
        page.wait_for_selector("#stage-wrap:not([hidden])")
        page.wait_for_timeout(400)
        SHOT.parent.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(SHOT), full_page=False)

        queue = page.locator("#queue").bounding_box()
        stage = page.locator("#stage-wrap").bounding_box()
        canvas = page.locator("#cv").bounding_box()
        inspector = page.locator(".inspector").bounding_box()
        assert queue and stage and canvas and inspector

        vp = page.viewport_size
        assert queue["x"] < stage["x"], "queue must sit on the left of the canvas"
        assert inspector["x"] + inspector["width"] <= vp["width"] + 2, f"inspector clipped off-screen: {inspector} vp={vp}"
        assert inspector["x"] >= stage["x"] + stage["width"] - 4, f"inspector not to the right of stage: {inspector} {stage}"
        assert abs(canvas["width"] - stage["width"]) <= 4, f"canvas must match stage width {canvas} {stage}"
        assert abs(canvas["height"] - stage["height"]) <= 4, f"canvas must match stage height {canvas} {stage}"
        assert stage["width"] > 700, f"stage too narrow: {stage['width']}"
        assert stage["height"] > 500, f"stage too short: {stage['height']}"
        assert canvas["x"] + canvas["width"] <= inspector["x"] + 4, f"canvas overlaps inspector {canvas} {inspector}"
        browser.close()
