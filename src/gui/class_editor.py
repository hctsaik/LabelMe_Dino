from __future__ import annotations

from pathlib import Path

from PyQt5.QtWidgets import (
    QDialog, QTableWidget, QTableWidgetItem, QVBoxLayout, QHBoxLayout,
    QPushButton, QHeaderView, QMessageBox, QFileDialog, QLabel,
)
from PyQt5.QtCore import Qt

from src.class_manager import ClassManager


class ClassEditorDialog(QDialog):
    """
    Modal dialog for defining labels, YOLO IDs, and per-class confidence thresholds.
    Labels can be imported from a LabelMe-format labels.txt (one label per line).
    """

    def __init__(self, class_manager: ClassManager, labels_file: str = "", parent=None):
        super().__init__(parent)
        self.cm = class_manager
        self.labels_file = labels_file
        self.setWindowTitle("Class Editor")
        self.setMinimumSize(540, 400)
        self._build_ui()
        self._populate()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # Import row
        import_row = QHBoxLayout()
        self.lbl_file = QLabel(
            Path(self.labels_file).name if self.labels_file else "No labels.txt configured"
        )
        self.lbl_file.setStyleSheet("color: gray; font-size: 11px;")
        self.btn_import = QPushButton("Import from labels.txt")
        import_row.addWidget(QLabel("LabelMe labels:"))
        import_row.addWidget(self.lbl_file, stretch=1)
        import_row.addWidget(self.btn_import)
        layout.addLayout(import_row)

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Label", "YOLO ID", "Conf Threshold"])
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.Stretch)
        hh.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        layout.addWidget(self.table)

        btn_row = QHBoxLayout()
        self.btn_add = QPushButton("Add Row")
        self.btn_remove = QPushButton("Remove Selected")
        btn_row.addWidget(self.btn_add)
        btn_row.addWidget(self.btn_remove)
        btn_row.addStretch()
        self.btn_ok = QPushButton("OK")
        self.btn_ok.setDefault(True)
        self.btn_cancel = QPushButton("Cancel")
        btn_row.addWidget(self.btn_ok)
        btn_row.addWidget(self.btn_cancel)
        layout.addLayout(btn_row)

        self.btn_import.clicked.connect(self._import_labels_file)
        self.btn_add.clicked.connect(self._add_row)
        self.btn_remove.clicked.connect(self._remove_selected)
        self.btn_ok.clicked.connect(self._accept)
        self.btn_cancel.clicked.connect(self.reject)

    def _populate(self):
        # If classes already defined, show them
        if self.cm.labels:
            for label in self.cm.labels:
                self._insert_row(label, self.cm.yolo_id(label), self.cm.conf_threshold(label))
        # Otherwise auto-load from configured labels_file
        elif self.labels_file:
            self._load_from_file(Path(self.labels_file))

    def _insert_row(self, label: str = "", yolo_id: int = 0, conf: float = 0.6):
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem(label))
        self.table.setItem(row, 1, QTableWidgetItem(str(yolo_id)))
        self.table.setItem(row, 2, QTableWidgetItem(str(conf)))

    def _add_row(self):
        next_id = self.table.rowCount()
        self._insert_row("", next_id, 0.6)
        self.table.editItem(self.table.item(self.table.rowCount() - 1, 0))

    def _remove_selected(self):
        rows = sorted(
            {idx.row() for idx in self.table.selectedIndexes()}, reverse=True
        )
        for row in rows:
            self.table.removeRow(row)

    # ------------------------------------------------------------------
    # labels.txt import
    # ------------------------------------------------------------------

    def _import_labels_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select labels.txt", "",
            "Text Files (*.txt);;All Files (*)"
        )
        if not path:
            return
        self.labels_file = path
        self.lbl_file.setText(Path(path).name)
        self._load_from_file(Path(path))

    def _load_from_file(self, path: Path):
        if not path.exists():
            QMessageBox.warning(self, "File Not Found", f"{path} not found.")
            return

        labels = [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")
        ]
        if not labels:
            QMessageBox.warning(self, "Empty File", "No labels found in file.")
            return

        # Keep existing conf thresholds for labels that already exist
        existing_confs = {
            label: self.cm.conf_threshold(label) for label in self.cm.labels
        }

        self.table.setRowCount(0)
        for yolo_id, label in enumerate(labels):
            conf = existing_confs.get(label, 0.6)
            self._insert_row(label, yolo_id, conf)

    # ------------------------------------------------------------------
    # Validation & commit
    # ------------------------------------------------------------------

    def _accept(self):
        seen_labels: set[str] = set()
        seen_ids: set[int] = set()
        rows: list[tuple[str, int, float]] = []

        for row in range(self.table.rowCount()):
            label = self._cell_text(row, 0)
            if not label:
                continue
            try:
                yolo_id = int(self._cell_text(row, 1))
                conf = float(self._cell_text(row, 2))
            except ValueError:
                QMessageBox.warning(
                    self, "Invalid Input",
                    f"Row {row + 1}: YOLO ID must be an integer, Conf must be a float."
                )
                return
            if label in seen_labels:
                QMessageBox.warning(self, "Duplicate", f"Duplicate label: '{label}'")
                return
            if yolo_id in seen_ids:
                QMessageBox.warning(self, "Duplicate", f"Duplicate YOLO ID: {yolo_id}")
                return
            seen_labels.add(label)
            seen_ids.add(yolo_id)
            rows.append((label, yolo_id, conf))

        self.cm.from_dict({
            label: {"yolo_id": yid, "conf_threshold": c}
            for label, yid, c in rows
        })
        self.accept()

    def get_labels_file(self) -> str:
        return self.labels_file

    def _cell_text(self, row: int, col: int) -> str:
        item = self.table.item(row, col)
        return item.text().strip() if item else ""
