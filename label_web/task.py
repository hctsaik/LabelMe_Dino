from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
SHA_RE = re.compile(r"^[0-9a-fA-F]{64}$")
QUIZ_TASK = "blind-quiz"


def list_images(root: Path) -> list[Path]:
    for sub in ("images", "frames", ""):
        folder = root / sub if sub else root
        if not folder.is_dir():
            continue
        files = sorted(
            p
            for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXT and not p.name.startswith(".")
        )
        if files:
            return files
    return []


def load_classes(root: Path) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    handoff = load_handoff(root)
    for name in handoff.get("class_options") or []:
        text = str(name).strip()
        if text and text not in seen:
            seen.add(text)
            names.append(text)
    classes_file = root / "classes.txt"
    if classes_file.is_file():
        for line in classes_file.read_text(encoding="utf-8").splitlines():
            text = line.strip()
            if text and not text.startswith("#") and text not in seen:
                seen.add(text)
                names.append(text)
    return names or ["object"]


def load_handoff(root: Path) -> dict[str, Any]:
    path = root / "_handoff.json"
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def is_quiz(root: Path) -> bool:
    return str(load_handoff(root).get("task") or "") == QUIZ_TASK


def sha_for_image(root: Path, image: Path) -> str | None:
    if SHA_RE.match(image.stem):
        return image.stem.lower()
    for item in load_handoff(root).get("items") or []:
        rel = str(item.get("image") or "")
        if Path(rel).name == image.name or Path(rel).stem == image.stem:
            sha = str(item.get("sha256") or "").strip()
            return sha.lower() if sha else None
    return None


class LabelState:
    def __init__(self, root: Path, data_dir: Path) -> None:
        self.root = Path(root).resolve()
        self.data_dir = Path(data_dir).resolve()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.reload()

    def reload(self) -> None:
        self.images = list_images(self.root)
        self.classes = load_classes(self.root)
        self.quiz = is_quiz(self.root)

    def switch_root(self, root: Path) -> None:
        self.root = Path(root).resolve()
        self.reload()

    def image(self, index: int) -> Path:
        if index < 0 or index >= len(self.images):
            raise IndexError("image index out of range")
        return self.images[index]

    def sidecar_path(self, image: Path) -> Path:
        return image.with_suffix(".json")

    def backup_path(self, image: Path) -> Path:
        dest = self.root / "annotations" / (image.stem + ".json")
        dest.parent.mkdir(parents=True, exist_ok=True)
        return dest

    def task_payload(self) -> dict[str, Any]:
        images = []
        for i, path in enumerate(self.images):
            images.append(
                {
                    "index": i,
                    "name": path.name,
                    "stem": path.stem,
                    "sha256": sha_for_image(self.root, path),
                }
            )
        return {
            "classes": self.classes,
            "images": images,
            "quiz": self.quiz,
            "root_name": self.root.name,
        }
