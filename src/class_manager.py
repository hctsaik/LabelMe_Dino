from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import yaml


DEFAULT_CONF = 0.6


@dataclass
class ClassDef:
    yolo_id: int
    conf_threshold: float = DEFAULT_CONF


class ClassManager:
    """
    Holds label → (yolo_id, conf_threshold) mapping.
    Persists to / loads from the classes section of config.yaml.
    """

    def __init__(self):
        self._classes: dict[str, ClassDef] = {}

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def add(self, label: str, yolo_id: int, conf_threshold: float = DEFAULT_CONF):
        if label in self._classes:
            raise ValueError(f"Label '{label}' already exists")
        self._classes[label] = ClassDef(yolo_id, conf_threshold)

    def update(self, label: str, **kwargs):
        if label not in self._classes:
            raise KeyError(f"Label '{label}' not found")
        cd = self._classes[label]
        if "yolo_id" in kwargs:
            cd.yolo_id = int(kwargs["yolo_id"])
        if "conf_threshold" in kwargs:
            cd.conf_threshold = float(kwargs["conf_threshold"])

    def remove(self, label: str):
        self._classes.pop(label, None)

    def rename(self, old: str, new: str):
        if old not in self._classes:
            raise KeyError(f"Label '{old}' not found")
        if new in self._classes:
            raise ValueError(f"Label '{new}' already exists")
        self._classes[new] = self._classes.pop(old)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    @property
    def labels(self) -> list[str]:
        return list(self._classes.keys())

    def has_label(self, label: str) -> bool:
        return label in self._classes

    def yolo_id(self, label: str) -> int:
        return self._classes[label].yolo_id

    def conf_threshold(self, label: str) -> float:
        return self._classes.get(label, ClassDef(0)).conf_threshold

    def is_ready(self) -> bool:
        return len(self._classes) > 0

    def validate_label(self, label: str):
        if label not in self._classes:
            raise ValueError(
                f"Label '{label}' not in class list. "
                f"Valid labels: {self.labels}"
            )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            label: {"yolo_id": cd.yolo_id, "conf_threshold": cd.conf_threshold}
            for label, cd in self._classes.items()
        }

    def from_dict(self, data: dict):
        self._classes = {}
        for label, v in data.items():
            self._classes[label] = ClassDef(
                yolo_id=int(v["yolo_id"]),
                conf_threshold=float(v.get("conf_threshold", DEFAULT_CONF)),
            )

    def save_to_config(self, config_path: str | Path):
        path = Path(config_path)
        with open(path) as f:
            cfg = yaml.safe_load(f) or {}
        cfg["classes"] = self.to_dict()
        with open(path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)

    def load_from_config(self, config_path: str | Path):
        path = Path(config_path)
        with open(path) as f:
            cfg = yaml.safe_load(f) or {}
        self.from_dict(cfg.get("classes") or {})

    def export_classes_txt(self, path: str | Path):
        """Write classes.txt sorted by yolo_id."""
        ordered = sorted(self._classes.items(), key=lambda kv: kv[1].yolo_id)
        Path(path).write_text("\n".join(label for label, _ in ordered))
