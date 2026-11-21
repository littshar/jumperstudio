"""Training history for this project's Jumper dance runs only.

Each row is one run under `D:\jumper_ws\logs\jumper\jumper.dance_crab`: date,
latest iteration/checkpoint and, when enabled in the reward dialog, the saved
reward-weight profile that produced it. Right-click a row to delete the run.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView, QDialog, QHBoxLayout, QHeaderView, QLabel, QMenu,
    QMessageBox, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout,
)

from app.core.jumper_train import latest_run_dir, newest_checkpoint, workspace_path
from app.ui.reward_dialog import show_weight_versions


def _run_rows(show_versions: bool) -> list[dict]:
    ws = workspace_path()
    logs = ws / "logs" / "jumper" / "jumper.dance_crab"
    rows: list[dict] = []
    if not logs.is_dir():
        return rows
    for run in logs.iterdir():
        if not run.is_dir():
            continue
        ckpt = newest_checkpoint(run)
        if ckpt is None:
            continue
        try:
            it = int(ckpt.stem.split("_")[-1])
        except ValueError:
            it = -1
        row = {
            "run": run,
            "ckpt": ckpt.name,
            "iteration": it,
            "mtime": ckpt.stat().st_mtime,
            "profile": "",
        }
        meta = run / "app_run.json"
        if meta.is_file():
            try:
                data = json.loads(meta.read_text(encoding="utf-8"))
                row["profile"] = str(data.get("reward_profile") or "")
                if not row["profile"]:
                    weights = data.get("reward_weights") or {}
                    row["profile"] = (
                        "自定义" if (weights.get("ppo") or weights.get("es"))
                        else "默认")
            except (OSError, ValueError):
                pass
        rows.append(row)
    rows.sort(key=lambda r: r["mtime"], reverse=True)
    return rows


class TrainHistoryDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("训练历史")
        self.setModal(True)
        self.setMinimumSize(640, 360)
        self.show_versions = show_weight_versions()

        self.table = QTableWidget(0, 4 if self.show_versions else 3)
        headers = ["时间", "轮次", "检查点"]
        if self.show_versions:
            headers.append("权重版本")
        self.table.setHorizontalHeaderLabels(headers)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.Stretch)
        if self.show_versions:
            self.table.horizontalHeader().setSectionResizeMode(
                3, QHeaderView.ResizeMode.Stretch)
        self.table.setContextMenuPolicy(
            Qt.ContextMenuPolicy.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._context)
        self._rows: list[dict] = []

        hint = QLabel("只显示本工程的训练记录 · 右键删除")
        hint.setStyleSheet("color:#8b949e;")
        refresh = QPushButton("刷新")
        refresh.clicked.connect(self.reload)
        close = QPushButton("关闭")
        close.setObjectName("accentBtn")
        close.clicked.connect(self.accept)
        btns = QHBoxLayout()
        btns.addWidget(hint, stretch=1)
        btns.addWidget(refresh)
        btns.addWidget(close)
        lay = QVBoxLayout(self)
        lay.addWidget(self.table, stretch=1)
        lay.addLayout(btns)
        self.reload()

    def reload(self) -> None:
        self._rows = _run_rows(self.show_versions)
        self.table.setRowCount(len(self._rows))
        for r, row in enumerate(self._rows):
            when = datetime.fromtimestamp(row["mtime"]).strftime(
                "%Y-%m-%d %H:%M")
            self.table.setItem(r, 0, QTableWidgetItem(when))
            self.table.setItem(
                r, 1, QTableWidgetItem(str(row["iteration"])
                                       if row["iteration"] >= 0 else "—"))
            self.table.setItem(r, 2, QTableWidgetItem(row["ckpt"]))
            if self.show_versions:
                self.table.setItem(
                    r, 3, QTableWidgetItem(row["profile"] or "—"))
        if not self._rows:
            self.table.setRowCount(1)
            item = QTableWidgetItem("暂无训练记录")
            item.setFlags(Qt.ItemFlag.NoItemFlags)
            self.table.setItem(0, 0, item)
            self.table.setSpan(0, 0, 1, self.table.columnCount())

    def _context(self, pos) -> None:
        idx = self.table.indexAt(pos)
        if not idx.isValid() or not self._rows:
            return
        row = idx.row()
        if row < 0 or row >= len(self._rows):
            return
        entry = self._rows[row]
        menu = QMenu(self)
        delete = menu.addAction("删除这次训练")
        action = menu.exec(self.table.mapToGlobal(pos))
        if action is not delete:
            return
        if QMessageBox.question(
                self, "删除训练记录",
                f"删除 {entry['run'].name}？\n检查点和日志会被永久删除。") \
                != QMessageBox.StandardButton.Yes:
            return
        try:
            shutil.rmtree(entry["run"])
        except OSError as exc:
            QMessageBox.warning(self, "删除失败", str(exc))
        self.reload()
