"""Reward-term weight editor, saved profiles, and weight-version toggle.

The PPO dance task and the local ES trial use different reward sets; both are
listed here. Saved profiles are kept in QSettings as JSON and can be reapplied
later. The last switch controls whether the training history shows which
weight profile produced a run (hidden by default).
"""

from __future__ import annotations

import json
import time
from datetime import datetime

from PySide6.QtCore import QSettings, Qt
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QDialog, QHBoxLayout,
    QHeaderView, QInputDialog, QLabel, QMessageBox, QPushButton, QTabWidget,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

KEY_PROFILES = "train/reward_profiles"
KEY_SHOW_VERSIONS = "train/show_weight_versions"

PPO_WEIGHTS = {
    "motion_global_root_pos": 0.5,
    "motion_global_root_ori": 0.5,
    "motion_body_pos": 1.0,
    "motion_body_ori": 1.0,
    "motion_body_lin_vel": 1.0,
    "motion_body_ang_vel": 1.0,
    "motion_joint_pos": 1.0,
    "motion_joint_vel": 0.5,
    "action_rate_l2": -0.1,
    "action_acc_l2": -0.05,
    "power": -0.02,
    "torque_headroom": -2.0,
    "joint_limit": -10.0,
    "self_collisions": -1.0,
    "anchor_height": 1.0,
}

ES_WEIGHTS = {
    "track": 0.28, "stab": 0.14, "hands": 0.08, "smooth": 0.05,
    "survive": 0.10, "yaw": 0.15, "rear": 0.12, "land": 0.08,
}

#: Additional terms the generated Jumper task really evaluates. Adding one
#: writes this weight into the task's REWARD_WEIGHTS; it is not a label only.
PPO_EXTRA_WEIGHTS = {
    "joint_vel_l2": -0.01,
    "joint_acc_l2": -0.005,
    "joint_torques_l2": -0.001,
    "flat_orientation_l2": -0.5,
    "is_alive": 0.5,
    "is_terminated": -1.0,
}


def show_weight_versions() -> bool:
    return QSettings().value(KEY_SHOW_VERSIONS, False, bool)


def _profiles() -> list[dict]:
    raw = QSettings().value(KEY_PROFILES, "", str) or ""
    try:
        data = json.loads(raw)
        return [p for p in data if isinstance(p, dict) and "name" in p]
    except (ValueError, TypeError):
        return []


def _save_profiles(profiles: list[dict]) -> None:
    QSettings().setValue(KEY_PROFILES, json.dumps(profiles[-24:]))


def latest_profile_name() -> str:
    ps = _profiles()
    return str(ps[-1]["name"]) if ps else ""


class _WeightTable(QTableWidget):
    def __init__(self, weights: dict, parent=None):
        super().__init__(0, 2, parent)
        self.setHorizontalHeaderLabels(["奖励项", "权重"])
        self.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch)
        self.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.ResizeToContents)
        self.verticalHeader().setVisible(False)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.setEditTriggers(
            QAbstractItemView.EditTrigger.DoubleClicked
            | QAbstractItemView.EditTrigger.SelectedClicked)
        for name, weight in weights.items():
            self.add_row(name, weight)

    def add_row(self, name: str = "", weight: float = 0.0) -> None:
        r = self.rowCount()
        self.insertRow(r)
        item = QTableWidgetItem(str(name))
        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self.setItem(r, 0, item)
        self.setItem(r, 1, QTableWidgetItem(f"{float(weight):g}"))

    def names(self) -> list[str]:
        return [self.item(r, 0).text() for r in range(self.rowCount())
                if self.item(r, 0) is not None]

    def values(self) -> dict:
        out = {}
        for r in range(self.rowCount()):
            name_item = self.item(r, 0)
            weight_item = self.item(r, 1)
            if name_item is None or weight_item is None:
                continue
            name = name_item.text().strip()
            if not name:
                continue
            try:
                out[name] = float(weight_item.text())
            except ValueError:
                continue
        return out


class RewardDialog(QDialog):
    def __init__(self, weights: dict | None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("奖励权重")
        self.setModal(True)
        self.setMinimumWidth(520)
        weights = weights or {}
        ppo = dict(PPO_WEIGHTS)
        ppo.update(weights.get("ppo") or {})
        es = dict(ES_WEIGHTS)
        es.update(weights.get("es") or {})

        self.ppo_table = _WeightTable(ppo)
        self.es_table = _WeightTable(es)
        tabs = QTabWidget()
        tabs.addTab(self._page(self.ppo_table, PPO_EXTRA_WEIGHTS), "PPO 奖励")
        tabs.addTab(self._page(self.es_table, None), "Smoke 奖励")

        self.version_box = QCheckBox("训练历史显示权重版本")
        self.version_box.setChecked(show_weight_versions())
        self.version_box.setToolTip(
            "在训练历史中显示每次运行使用的奖励权重配置；默认隐藏")

        self.profile_combo = QComboBox()
        self._refresh_profiles()
        save_btn = QPushButton("保存配置")
        save_btn.clicked.connect(self._save_profile)
        apply_btn = QPushButton("应用")
        apply_btn.clicked.connect(self._apply_profile)
        delete_btn = QPushButton("删除")
        delete_btn.clicked.connect(self._delete_profile)
        prof_row = QHBoxLayout()
        prof_row.addWidget(QLabel("历史权重"))
        prof_row.addWidget(self.profile_combo, stretch=1)
        prof_row.addWidget(apply_btn)
        prof_row.addWidget(save_btn)
        prof_row.addWidget(delete_btn)

        hint = QLabel("双击权重编辑 · 新增项来自后端支持列表")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#8b949e;")

        ok = QPushButton("确定")
        ok.setObjectName("accentBtn")
        ok.clicked.connect(self.accept)
        cancel = QPushButton("取消")
        cancel.clicked.connect(self.reject)
        btns = QHBoxLayout()
        btns.addWidget(self.version_box)
        btns.addStretch(1)
        btns.addWidget(cancel)
        btns.addWidget(ok)

        lay = QVBoxLayout(self)
        lay.addWidget(tabs, stretch=1)
        lay.addLayout(prof_row)
        lay.addWidget(hint)
        lay.addLayout(btns)

    @staticmethod
    def _page(table: _WeightTable, extras: dict | None) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(0, 6, 0, 0)
        row = QHBoxLayout()
        if extras:
            combo = QComboBox()
            for name in extras:
                combo.addItem(name, name)
            add = QPushButton("添加奖励")

            def add_extra() -> None:
                name = combo.currentData()
                if not name or name in table.names():
                    return
                table.add_row(name, extras[name])
                combo.removeItem(combo.currentIndex())
                add.setEnabled(combo.count() > 0)

            add.clicked.connect(add_extra)
            row.addWidget(combo, stretch=1)
            row.addWidget(add)
        remove = QPushButton("删除选中")
        remove.clicked.connect(lambda: table.removeRow(table.currentRow())
                               if table.currentRow() >= 0 else None)
        row.addWidget(remove)
        row.addStretch(1)
        lay.addLayout(row)
        lay.addWidget(table)
        return page

    # ---------------------------------------------------------- profiles ----
    def _refresh_profiles(self) -> None:
        self.profile_combo.clear()
        for p in _profiles():
            name = str(p.get("name", "?"))
            ts = float(p.get("ts") or 0)
            when = (datetime.fromtimestamp(ts).strftime("%m-%d %H:%M")
                    if ts > 0 else "")
            self.profile_combo.addItem(f"{name} · {when}" if when else name,
                                       name)

    def _save_profile(self) -> None:
        name, ok = QInputDialog.getText(self, "保存奖励配置", "名称")
        name = (name or "").strip()
        if not ok or not name:
            return
        profiles = [p for p in _profiles() if p.get("name") != name]
        profiles.append({"name": name, "ts": time.time(),
                         "ppo": self.ppo_table.values(),
                         "es": self.es_table.values()})
        _save_profiles(profiles)
        self._refresh_profiles()
        idx = self.profile_combo.findData(name)
        if idx >= 0:
            self.profile_combo.setCurrentIndex(idx)

    def _apply_profile(self) -> None:
        name = self.profile_combo.currentData()
        entry = next((p for p in _profiles() if p.get("name") == name), None)
        if entry is None:
            return
        self.ppo_table.setRowCount(0)
        for k, v in (entry.get("ppo") or {}).items():
            self.ppo_table.add_row(k, v)
        self.es_table.setRowCount(0)
        for k, v in (entry.get("es") or {}).items():
            self.es_table.add_row(k, v)

    def _delete_profile(self) -> None:
        name = self.profile_combo.currentData()
        if not name:
            return
        if QMessageBox.question(
                self, "删除奖励配置", f"删除「{name}」？") \
                != QMessageBox.StandardButton.Yes:
            return
        _save_profiles([p for p in _profiles() if p.get("name") != name])
        self._refresh_profiles()

    # ------------------------------------------------------------ result ----
    def weights(self) -> dict:
        return {"ppo": self.ppo_table.values(), "es": self.es_table.values()}

    def profile_name(self) -> str:
        return str(self.profile_combo.currentData() or "")

    def show_versions(self) -> bool:
        return self.version_box.isChecked()

    def accept(self) -> None:
        QSettings().setValue(KEY_SHOW_VERSIONS, self.version_box.isChecked())
        super().accept()
