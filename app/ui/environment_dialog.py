"""Training-environment diagnosis, install/repair and uninstall dialog.

First entry to the training page opens this once: it checks the app venv,
robot assets, bundled pretrain, the Jumper workspace, CUDA torch, mjrl/mjlab,
the replay patch and the shortcut, and offers to install what is missing.
Later the same dialog is available from Settings -> Environment.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView, QDialog, QHBoxLayout, QHeaderView, QLabel,
    QMessageBox, QPlainTextEdit, QPushButton, QTableWidget, QTableWidgetItem,
    QVBoxLayout,
)

from app.constants import PROJECT_DIR
from app.core.env_check import (
    Component, diagnose, install_command, uninstall_command,
)


class _DiagnoseWorker(QThread):
    done = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)

    def run(self) -> None:
        try:
            comps, info = diagnose()
            self.done.emit((comps, info))
        except Exception as exc:  # noqa: BLE001
            self.done.emit(([], {"error": f"{type(exc).__name__}: {exc}"}))


class _ManagerWorker(QThread):
    line = Signal(str)
    done = Signal(int)

    def __init__(self, cmd: list[str], parent=None):
        super().__init__(parent)
        self.cmd = cmd

    def run(self) -> None:
        try:
            proc = subprocess.Popen(
                self.cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", bufsize=1,
                cwd=str(PROJECT_DIR),
                creationflags=(getattr(subprocess, "CREATE_NO_WINDOW", 0)
                               if os.name == "nt" else 0))
            assert proc.stdout is not None
            for raw in proc.stdout:
                self.line.emit(raw.rstrip())
            proc.wait()
            self.done.emit(proc.returncode)
        except Exception as exc:  # noqa: BLE001
            self.line.emit(f"{type(exc).__name__}: {exc}")
            self.done.emit(-1)


class EnvironmentDialog(QDialog):
    def __init__(self, parent=None, first_run: bool = False):
        super().__init__(parent)
        self.setWindowTitle("训练环境")
        self.setMinimumSize(720, 480)
        self.first_run = first_run
        self._components: dict[str, Component] = {}
        self._info: dict = {}
        self._diag: _DiagnoseWorker | None = None
        self._manager: _ManagerWorker | None = None

        self.summary = QLabel("正在诊断…")
        self.summary.setWordWrap(True)
        self.summary.setStyleSheet("color:#8b949e;")
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["", "组件", "状态", "说明"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionMode(
            QAbstractItemView.SelectionMode.NoSelection)
        head = self.table.horizontalHeader()
        head.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        head.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        head.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        head.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(2000)
        self.log.setPlaceholderText("安装 / 卸载输出")
        self.log.setStyleSheet(
            "QPlainTextEdit { background:#0b0f14; color:#9fb2c8;"
            " border:1px solid #21262d; border-radius:6px;"
            " font-family:Consolas,'Microsoft YaHei'; }")
        self.log.setMaximumHeight(150)

        self.diag_btn = QPushButton("重新诊断")
        self.diag_btn.clicked.connect(self._diagnose)
        self.install_btn = QPushButton("安装 / 修复")
        self.install_btn.setObjectName("accentBtn")
        self.install_btn.clicked.connect(lambda: self._run_manager("install"))
        self.reinstall_btn = QPushButton("重装")
        self.reinstall_btn.clicked.connect(
            lambda: self._run_manager("install", reinstall=True))
        self.uninstall_btn = QPushButton("卸载")
        self.uninstall_btn.clicked.connect(lambda: self._run_manager("uninstall"))
        self.close_btn = QPushButton("关闭")
        self.close_btn.clicked.connect(self.accept)
        row = QHBoxLayout()
        row.addWidget(self.diag_btn)
        row.addWidget(self.install_btn)
        row.addWidget(self.reinstall_btn)
        row.addWidget(self.uninstall_btn)
        row.addStretch(1)
        row.addWidget(self.close_btn)

        lay = QVBoxLayout(self)
        lay.addWidget(self.summary)
        lay.addWidget(self.table, stretch=1)
        lay.addWidget(self.log)
        lay.addLayout(row)
        self._set_busy(True)
        self._diagnose()

    # ------------------------------------------------------------ diagnose ----
    def _diagnose(self) -> None:
        if self._diag is not None and self._diag.isRunning():
            return
        self._set_busy(True)
        self.summary.setText("正在诊断训练环境…")
        self._diag = _DiagnoseWorker(self)
        self._diag.done.connect(self._on_diagnosed)
        self._diag.start()

    def _on_diagnosed(self, data) -> None:
        comps, info = data
        self._components = {c.key: c for c in comps}
        self._info = info
        self._fill_table(comps)
        if "error" in info:
            self.summary.setText("诊断失败 · " + str(info["error"]))
        else:
            gpu = info.get("gpu", "")
            ready = info.get("ready", False)
            self.summary.setText(
                ("训练环境就绪" if ready else "训练环境缺少组件")
                + f" · GPU {gpu}")
        self._set_busy(False)
        if self.first_run:
            self.first_run = False
            if not info.get("ready", False):
                self._ask_install()

    def _fill_table(self, comps: list[Component]) -> None:
        self.table.setRowCount(len(comps))
        colors = {"ok": "#3fb950", "warn": "#e6b450", "missing": "#e6b450",
                  "error": "#f85149"}
        labels = {"ok": "就绪", "warn": "需修复", "missing": "缺失",
                  "error": "异常"}
        for row, c in enumerate(comps):
            box = QTableWidgetItem()
            box.setFlags(Qt.ItemFlag.ItemIsEnabled
                         | Qt.ItemFlag.ItemIsUserCheckable)
            selected = c.selected
            box.setCheckState(
                Qt.CheckState.Checked if selected else Qt.CheckState.Unchecked)
            self.table.setItem(row, 0, box)
            self.table.setItem(row, 1, QTableWidgetItem(c.title))
            status = QTableWidgetItem(labels.get(c.status, c.status))
            status.setForeground(QColor(colors.get(c.status, "#8b949e")))
            self.table.setItem(row, 2, status)
            self.table.setItem(row, 3, QTableWidgetItem(c.detail))

    def _ask_install(self) -> None:
        # Select the components that are missing/broken and installable.
        for row, c in enumerate(self._components.values()):
            if c.installable and not c.ok:
                item = self.table.item(row, 0)
                if item is not None:
                    item.setCheckState(Qt.CheckState.Checked)
        if QMessageBox.question(
                self, "训练环境",
                "检测到训练环境缺少组件。\n是否自动安装 / 修复选中的部分？",
                QMessageBox.StandardButton.Yes
                | QMessageBox.StandardButton.No) == QMessageBox.StandardButton.Yes:
            self._run_manager("install")

    # ------------------------------------------------------------ actions ----
    def _selected(self, *, install: bool) -> list[str]:
        keys = []
        for row, c in enumerate(self._components.values()):
            item = self.table.item(row, 0)
            if item is None or item.checkState() != Qt.CheckState.Checked:
                continue
            if install and not c.installable:
                continue
            if not install and not c.uninstallable:
                continue
            keys.append(c.key)
        return keys

    def _run_manager(self, action: str, reinstall: bool = False) -> None:
        if self._manager is not None and self._manager.isRunning():
            return
        install = action == "install"
        keys = self._selected(install=install)
        if not keys:
            QMessageBox.information(
                self, "训练环境",
                "请先勾选要安装 / 修复的组件。"
                if install else "请先勾选要卸载的组件。")
            return
        if not install and "app" in keys:
            keys.remove("app")
            QMessageBox.information(
                self, "训练环境",
                "应用虚拟环境需要在关闭 Jumper Studio 后运行\n"
                "uninstall_windows.bat 卸载。")
        if not keys:
            return
        if action == "uninstall":
            if QMessageBox.question(
                    self, "卸载环境",
                    "删除选中组件的虚拟环境 / 缓存？\n"
                    + ", ".join(keys),
                    QMessageBox.StandardButton.Yes
                    | QMessageBox.StandardButton.No) != \
                    QMessageBox.StandardButton.Yes:
                return
            cmd = uninstall_command(keys)
            cmd.append("-Yes")
            self.log.clear()
        else:
            if reinstall and QMessageBox.question(
                    self, "重装环境",
                    "重装会重新创建训练虚拟环境并重新安装依赖。\n继续？",
                    QMessageBox.StandardButton.Yes
                    | QMessageBox.StandardButton.No) != \
                    QMessageBox.StandardButton.Yes:
                return
            cmd = install_command(keys, reinstall=reinstall)
            self.log.clear()
        self._set_busy(True)
        self.summary.setText("正在安装…" if install else "正在卸载…")
        self._manager = _ManagerWorker(cmd, self)
        self._manager.line.connect(self._append_log)
        self._manager.done.connect(
            lambda code: self._on_manager_done(code, action))
        self._manager.start()

    def _append_log(self, text: str) -> None:
        if text:
            self.log.appendPlainText(text)
            bar = self.log.verticalScrollBar()
            bar.setValue(bar.maximum())

    def _on_manager_done(self, code: int, action: str) -> None:
        self.summary.setText(
            ("安装完成" if action == "install" else "卸载完成")
            if code == 0 else
            f"操作失败 · 退出码 {code} · 见下方输出")
        self._set_busy(False)
        self._diagnose()

    def _set_busy(self, busy: bool) -> None:
        for b in (self.diag_btn, self.install_btn, self.reinstall_btn,
                  self.uninstall_btn, self.close_btn):
            b.setEnabled(not busy)
        if busy:
            self.diag_btn.setEnabled(False)

    def closeEvent(self, event) -> None:  # noqa: N802
        if self._manager is not None and self._manager.isRunning():
            QMessageBox.information(self, "训练环境", "安装 / 卸载尚未完成。")
            event.ignore()
            return
        super().closeEvent(event)
