import json
import logging
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

import torch  # must be imported before PyQt5 to avoid DLL conflict on Windows
from PyQt5.QtWidgets import QApplication, QMessageBox
from src.gui.main_window import MainWindow

LOG_DIR = Path("logs")


def _purge_old_logs(days: int = 3):
    cutoff = datetime.now().timestamp() - days * 86400
    for f in LOG_DIR.glob("dlb_*.log"):
        if f.stat().st_mtime < cutoff:
            f.unlink()


def _setup_logging():
    LOG_DIR.mkdir(exist_ok=True)
    _purge_old_logs()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"dlb_{stamp}.log"

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    for noisy in ("httpcore", "httpx", "urllib3", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return log_file


def _exception_hook(exc_type, exc_value, exc_tb):
    """Catch all unhandled exceptions, log them, and show a dialog."""
    msg = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    logging.critical("Unhandled exception:\n%s", msg)

    try:
        dlg = QMessageBox()
        dlg.setWindowTitle("Application error")
        dlg.setText("An unexpected error occurred. Please check the logs folder for details.")
        dlg.setDetailedText(msg)
        dlg.exec_()
    except Exception:
        pass

    sys.__excepthook__(exc_type, exc_value, exc_tb)


def _parse_runtime_args(argv: list[str]) -> tuple[list[str], Path | None]:
    cleaned = [argv[0]]
    ready_file: Path | None = None
    i = 1
    while i < len(argv):
        arg = argv[i]
        if arg == "--ready-file" and i + 1 < len(argv):
            ready_file = Path(argv[i + 1])
            i += 2
            continue
        cleaned.append(arg)
        i += 1

    env_ready_file = os.environ.get("LABELME_DINO_READY_FILE", "").strip()
    if ready_file is None and env_ready_file:
        ready_file = Path(env_ready_file)
    return cleaned, ready_file


def _write_ready_file(path: Path, log_file: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "ok": True,
        "app": "video_annotator",
        "pid": os.getpid(),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "log_file": str(log_file.resolve()),
        "cwd": str(Path.cwd()),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logging.info("Ready file written: %s", path)


def main():
    qt_argv, ready_file = _parse_runtime_args(sys.argv)
    log_file = _setup_logging()
    sys.excepthook = _exception_hook
    logging.info("=== video_annotator started ===")
    logging.info("Log file: %s", log_file)

    app = QApplication(qt_argv)
    window = MainWindow()
    window.showMaximized()
    if ready_file is not None:
        _write_ready_file(ready_file, log_file)
    code = app.exec_()
    logging.info("=== App exited with code %d ===", code)
    sys.exit(code)


if __name__ == "__main__":
    main()
