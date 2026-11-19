"""Choose the language model that helps the planner (or none: offline rules)."""

from __future__ import annotations

import os
from typing import Any

from qtpy.QtWidgets import QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFormLayout, QLabel, QLineEdit, QPushButton, QVBoxLayout

from vexel.agent import models
from vexel.ui.widgets import heading, muted

PRIVACY = {
    "rules": "完全离线，不发出任何网络请求。",
    "local": "模型在本机或局域网的 Ollama 上运行，数据不离开你的网络。",
    "claude": "数据指纹（数字）发送到 Anthropic。",
    "deepseek": "数据指纹（数字）发送到 DeepSeek（服务器在中国大陆）。",
    "qwen": "数据指纹（数字）发送到阿里云百炼（服务器在中国大陆；海外账号请把地址改为 dashscope-intl）。",
    "gemini": "数据指纹（数字）发送到 Google。",
    "custom": "数据指纹（数字）发送到你填写的服务地址。",
}


class ModelSettingsDialog(QDialog):
    def __init__(self, parent: Any = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("规划模型设置")
        self.resize(620, 400)
        self.cfg = models.load_config(policy=False)  # the person's own settings; the policy is applied when used
        v = QVBoxLayout(self)
        v.addWidget(heading("自动规划时请哪个大模型帮忙"))
        v.addWidget(muted("模型只帮助理解你的目标、在固定范围内建议设置；结果仍由 Vexel 的算子计算，方案由你确认。"
                          "所选模型不可用时（没有密钥、断网、拒答等）自动改用离线规则引擎。"))  # fmt: skip
        form = QFormLayout()
        self.provider = QComboBox()
        for name in models.PROVIDERS:
            self.provider.addItem(models.LABELS[name], name)
        from vexel.policy import load_policy

        self.policy = load_policy()
        for i, name in enumerate(models.PROVIDERS):
            host = models.tier_host(name, self.cfg)
            if host is not None and not self.policy.host_allowed(host):
                self.provider.model().item(i).setEnabled(False)
        current = next((n for n in self.cfg["order"] if n != "rules"), "rules")
        self.provider.setCurrentIndex(max(0, self.provider.findData(current)))
        if self.policy.source is not None:
            banner = QLabel(self.policy.describe())
            banner.setObjectName("notes")
            banner.setWordWrap(True)
            v.addWidget(banner)
        self.model_edit = QLineEdit()
        self.url_edit = QLineEdit()
        self.key_label = QLabel()
        for edit in (self.model_edit, self.url_edit):
            edit.setMinimumWidth(400)
        self.images_check = QCheckBox("同时发送降采样的四宫格诊断图（默认只发数字）")
        form.addRow("模型来源", self.provider)
        form.addRow("模型名", self.model_edit)
        form.addRow("服务地址", self.url_edit)
        form.addRow("API Key", self.key_label)
        form.addRow("", self.images_check)
        v.addLayout(form)
        self.privacy = muted("")
        v.addWidget(self.privacy)
        self.test_btn = QPushButton("测试连接")
        self.test_btn.clicked.connect(self.test)
        v.addWidget(self.test_btn)
        self.result = muted("")
        v.addWidget(self.result)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Save).setText("保存")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        buttons.accepted.connect(self.save)
        buttons.rejected.connect(self.reject)
        v.addWidget(buttons)
        self.provider.currentIndexChanged.connect(self._fill)
        self._fill()

    def name(self) -> str:
        return self.provider.currentData()

    def _fill(self) -> None:
        name = self.name()
        tier = self.cfg.get(name, {})
        has_model = name != "rules"
        self.model_edit.setEnabled(has_model)
        self.model_edit.setText(tier.get("model", ""))
        url_key = "url" if name == "local" else "base_url"
        self.url_edit.setEnabled(name in ("local", *models.COMPATIBLE))
        self.url_edit.setText(tier.get(url_key, ""))
        self.images_check.setEnabled(has_model and self.policy.allow_images)
        self.images_check.setChecked(bool(tier.get("send_images")))
        env = tier.get("api_key_env") or ("ANTHROPIC_API_KEY" if name == "claude" else "")
        if env:
            state = "已设置" if os.environ.get(env) else "未设置"
            extra = "（或已用 ant auth login 登录）" if name == "claude" else ""
            self.key_label.setText(f"环境变量 {env}：{state}{extra}")
        else:
            self.key_label.setText("不需要" if name in ("rules", "local") else "按服务要求设置环境变量，并在配置文件里写 api_key_env")
        keys = "API 密钥只从环境变量读取，Vexel 不保存。" if env else ""
        self.privacy.setText(PRIVACY.get(name, "") + keys)
        self.result.setText("")

    def _collect(self) -> dict[str, Any]:
        cfg = {k: (dict(v) if isinstance(v, dict) else v) for k, v in self.cfg.items()}
        name = self.name()
        cfg["order"] = [name, "rules"] if name != "rules" else ["rules"]
        if name != "rules":
            cfg[name]["model"] = self.model_edit.text().strip()
            cfg[name]["send_images"] = self.images_check.isChecked()
            if name == "local":
                cfg[name]["url"] = self.url_edit.text().strip()
            elif name in models.COMPATIBLE:
                cfg[name]["base_url"] = self.url_edit.text().strip()
        return cfg

    def test(self) -> bool:
        cfg = self._collect()
        tier = models.tiers(cfg)[0]
        host = models.tier_host(tier.name, cfg)
        if host is not None and not self.policy.host_allowed(host):
            self.result.setText(f"不可用：管理员策略禁止访问 {host}")
            return False
        ok, why = tier.available()
        if not ok:
            self.result.setText(f"不可用：{why}")
            return False
        try:
            proposal = tier.propose({"stage": "draft", "goal": "数每个液滴里的细胞", "valid_keys": []})
        except models.ModelUnavailable as e:
            self.result.setText(f"连接失败：{e}")
            return False
        self.result.setText(f"连接成功：{proposal.model or proposal.tier} 理解为 数细胞 = {proposal.cells}")
        return True

    def save(self) -> None:
        self.cfg = self._collect()
        models.save_config(self.cfg)
        self.accept()
