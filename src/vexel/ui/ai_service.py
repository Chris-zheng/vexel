"""帮助 → AI 规划服务…: sign in to the Vexel AI gateway with an invite code, see what is left of
today's requests and tokens and of the organisation's budget, sign out.

The token the service issues is kept in the configuration (``$VEXEL_HOME/config.json``,
readable by this user only). No API key of any model provider is ever on this computer.
"""

from __future__ import annotations

import socket
from typing import Any

from qtpy.QtCore import Qt
from qtpy.QtWidgets import QCheckBox, QDialog, QFormLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QVBoxLayout

from vexel.i18n import tr


class AiServiceDialog(QDialog):
    def __init__(self, parent: Any = None) -> None:
        from vexel.agent import models

        super().__init__(parent)
        self.setWindowTitle(tr("AI 规划服务"))
        self.setMinimumWidth(460)
        self.cfg = models.load_config(policy=False)
        v = QVBoxLayout(self)
        intro = QLabel(tr("自动规划可以请 Vexel 的 AI 规划服务（云端大模型）帮忙选择测量方法和设置。只发送图像的统计数字和你写的目标；"
                          "图片只有在你的组织允许、并在下面勾选时才发送。没有登录时，自动规划只用本机的离线规则。"))  # fmt: skip
        intro.setWordWrap(True)
        v.addWidget(intro)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        self.status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        v.addWidget(self.status)
        form = QFormLayout()
        self.url = QLineEdit(self.cfg["gateway"].get("url", ""))
        self.invite = QLineEdit()
        self.invite.setPlaceholderText("VX-XXXX-XXXX-XXXX-XXXX")
        self.label = QLineEdit(socket.gethostname()[:60])
        self.images = QCheckBox(tr("允许发送缩略图（组织也须允许）"))
        self.images.setChecked(bool(self.cfg["gateway"].get("send_images")))
        form.addRow(tr("服务地址"), self.url)
        form.addRow(tr("邀请码"), self.invite)
        form.addRow(tr("这台电脑的名称"), self.label)
        form.addRow("", self.images)
        v.addLayout(form)
        row = QHBoxLayout()
        self.sign_in_btn = QPushButton(tr("登录"))
        self.sign_in_btn.setObjectName("primary")
        self.sign_in_btn.clicked.connect(self.sign_in)
        self.refresh_btn = QPushButton(tr("刷新额度"))
        self.refresh_btn.clicked.connect(self.refresh)
        self.sign_out_btn = QPushButton(tr("退出登录"))
        self.sign_out_btn.clicked.connect(self.sign_out)
        close = QPushButton(tr("关闭"))
        close.clicked.connect(self.accept)
        for b in (self.sign_in_btn, self.refresh_btn, self.sign_out_btn):
            row.addWidget(b)
        row.addStretch(1)
        row.addWidget(close)
        v.addLayout(row)
        self.images.toggled.connect(lambda on: self._save(send_images=bool(on)))
        self._show()

    def _signed_in(self) -> bool:
        return bool(self.cfg["gateway"].get("token"))

    def _save(self, **fields: Any) -> None:
        from vexel.agent import models

        self.cfg["gateway"].update(fields)
        models.save_config(self.cfg)

    def _show(self, quota: dict[str, Any] | None = None, error: str = "") -> None:
        signed = self._signed_in()
        for w in (self.invite, self.label):
            w.setEnabled(not signed)
        self.sign_in_btn.setVisible(not signed)
        self.refresh_btn.setVisible(signed)
        self.sign_out_btn.setVisible(signed)
        if error:
            self.status.setText(f"<span style='color:#b45309'>{error}</span>")
        elif not signed:
            self.status.setText(tr("未登录：自动规划只用离线规则。向管理员索取邀请码后登录。"))
        elif quota:
            self.status.setText(tr(f"已登录：{quota.get('org', '')}（{quota.get('user', '')}）<br>今天还可以规划 {quota.get('requests_left_today')} 次，"
                                   f"剩余 {quota.get('tokens_left_today'):,} 个 token；本组织本月预算还剩 ${quota.get('budget_left_usd')}"
                                   + ("<br>组织允许发送缩略图" if quota.get("images_allowed") else "<br>组织不允许发送图片：只发送数字")))  # fmt: skip
        else:
            self.status.setText(tr("已登录。"))

    def sign_in(self) -> bool:
        from vexel.agent import models

        code = self.invite.text().strip()
        if not code:
            self._show(error=tr("请输入邀请码"))
            return False
        gw = {**self.cfg["gateway"], "url": self.url.text().strip(), "token": ""}
        try:
            got = models.gateway_request(gw, "/v1/auth/redeem", {"invite": code, "label": self.label.text().strip() or "computer"})
        except models.ModelUnavailable as e:
            self._show(error=tr(f"登录失败：{e}"))
            return False
        self._save(url=gw["url"], token=got.get("token", ""))
        self.invite.clear()
        return self.refresh()

    def refresh(self) -> bool:
        from vexel.agent import models

        try:
            q = models.gateway_request(self.cfg["gateway"], "/v1/quota")
        except models.ModelUnavailable as e:
            self._show(error=tr(f"无法取得额度：{e}"))
            return False
        self._show(q)
        return True

    def sign_out(self) -> None:
        self._save(token="")
        self._show()


__all__ = ["AiServiceDialog"]
