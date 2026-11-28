# Vexel AI 网关：部署说明

网关是桌面版访问云端大模型（DeepSeek、Gemini）的唯一途径：它持有服务商的 API 密钥，按组织和用户限制用量，只接受“规划图像测量”这一种请求。设计见 `src/vexel/gateway/__init__.py`，0.8 计划阶段 3。

## 需要准备

- 一台新加坡的云服务器（例如阿里云或腾讯云新加坡区；2 核 4 GB、Ubuntu 24.04 足够），开放 80、443 端口。
- 一个域名，例如 `ai.你的域名`，A 记录指向这台服务器。
- DeepSeek 与 Gemini 的 API 密钥（至少一个），并在两家后台各设一个每月费用上限。

## 安装（在服务器上，以 root 执行）

```bash
# 1. 运行用户与目录
useradd --system --home /var/lib/vexel --shell /usr/sbin/nologin vexel
mkdir -p /var/lib/vexel /etc/vexel && chown vexel:vexel /var/lib/vexel

# 2. Vexel（带 gateway 依赖）装到 /opt/vexel
curl -LsSf https://astral.sh/uv/install.sh | sh
UV_TOOL_DIR=/opt/vexel-tools UV_TOOL_BIN_DIR=/opt/vexel/bin ~/.local/bin/uv tool install "vexel[gateway] @ <wheel 文件或仓库地址>"

# 3. 密钥与设置
cp gateway.env.example /etc/vexel/gateway.env
chmod 600 /etc/vexel/gateway.env && chown vexel:vexel /etc/vexel/gateway.env
nano /etc/vexel/gateway.env        # 填入 DEEPSEEK_API_KEY、GEMINI_API_KEY 和价格

# 4. 服务
cp vexel-gateway.service /etc/systemd/system/
systemctl daemon-reload && systemctl enable --now vexel-gateway

# 5. HTTPS（Caddy 自动申请证书）
apt install -y caddy
cp Caddyfile /etc/caddy/Caddyfile   # 先把 ai.example.com 改成你的域名
systemctl reload caddy
```

检查：`curl https://ai.你的域名/v1/health` 应返回 `{"ok": true, "providers": [...]}`，列出已配置密钥的服务商（只列名称）。

## 组织与邀请码

在服务器上以 `vexel` 用户运行（数据库位置来自 `/etc/vexel/gateway.env`）：

```bash
sudo -u vexel env $(cat /etc/vexel/gateway.env | xargs) /opt/vexel/bin/vexel gateway org add 某某实验室 --region cn --budget 50
sudo -u vexel env $(cat /etc/vexel/gateway.env | xargs) /opt/vexel/bin/vexel gateway invite 某某实验室 --uses 5 --days 30
sudo -u vexel env $(cat /etc/vexel/gateway.env | xargs) /opt/vexel/bin/vexel gateway usage 某某实验室
```

- `--region cn`：先用 DeepSeek（中国大陆的组织）；`intl`：先用 Gemini。一家不可用时自动换另一家。
- `--budget`：每月预算（美元），用完后该组织的自动规划改用离线规则，直到下个月。
- `--images`：允许该组织发送缩略图（默认不允许，只发数字）。半导体客户保持默认。
- 每个用户每天默认 30 次规划、5 万个 token（`--daily-requests`、`--daily-tokens` 修改）。
- 邀请码只显示一次；每用一次登录一台电脑。`vexel gateway users` 列出用户，`vexel gateway revoke <编号>` 收回某台电脑的权限。

测试者在桌面版里：帮助 → **AI 规划服务…**，填服务地址 `https://ai.你的域名` 和邀请码，点“登录”。

## 运维

- 数据库 `/var/lib/vexel/gateway.db`：每天备份一次即可（例如 `sqlite3 gateway.db ".backup gateway-$(date +%F).db"`）。里面没有图片，也没有用户写的目标原文；邀请码和登录令牌只存哈希。
- 日志：`journalctl -u vexel-gateway`；Caddy 的访问日志不记录请求内容。
- 换密钥：改 `/etc/vexel/gateway.env` 后 `systemctl restart vexel-gateway`。
- 价格：服务商调价后改 `VEXEL_PRICE_*`，预算按新价格计算。

## 本机开发

```bash
uv sync --extra gateway
VEXEL_GATEWAY_MOCK=1 VEXEL_GATEWAY_DB=/tmp/gw.db uv run vexel gateway org add dev
VEXEL_GATEWAY_MOCK=1 VEXEL_GATEWAY_DB=/tmp/gw.db uv run vexel gateway invite dev
VEXEL_GATEWAY_MOCK=1 VEXEL_GATEWAY_DB=/tmp/gw.db uv run vexel gateway serve
```

桌面版里服务地址填 `http://127.0.0.1:8780`。模拟的服务商不需要密钥、不花钱，回答“保持草案”。
