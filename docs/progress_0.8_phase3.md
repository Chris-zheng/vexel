# 0.8 阶段 3（AI 网关与 Agent）进度

更新：2026-11-28。

## 已完成（用模拟的服务商开发与测试）

| 部分 | 内容 | 测试 |
| :-- | :-- | :-- |
| 网关服务 `vexel.gateway` | 组织（地区、月预算、是否允许图片、每人每天的次数与 token）、邀请码、令牌（只存哈希）、用量记录（不含目标原文、答案、图片）、一周缓存 | `tests/test_gateway.py` |
| 规则过滤 | 拒绝滥用、提示词注入、索要配置或密钥、无关请求；上下文里的其他字符串同样检查；拒绝不花 token | `tests/test_gateway.py`、`tests/benchmarks/redteam_eval.py` |
| 提示与校验 | 系统提示、答案格式（含“超出范围”标志）和白名单都在服务端由 Vexel 的配方定义生成；答案逐项校验，白名单外的设置丢弃，说明截断到 160 字 | `tests/test_gateway.py` |
| 用量上限 | 输入 ≤ 3000 token、输出 ≤ 600 token、每个规划 ≤ 4 次调用、每人每天 30 次和 5 万 token、组织月预算；按服务商返回的实际 token 记账 | `tests/test_gateway.py` |
| 服务商 | DeepSeek、Gemini（按地区先后，失败自动换另一家）；模拟服务商 | 真实服务商**未测**（需要密钥） |
| HTTP 接口 | 只有 4 个：health、redeem、quota、plan | `tests/test_gateway.py` |
| 管理命令 | `vexel gateway serve / org / invite / users / revoke / usage` | 手动试过 |
| 部署文件 | `deploy/gateway/`：systemd、Caddy（自动 HTTPS）、环境变量模板、中文部署说明 | 未在服务器上部署 |
| 桌面版 | 帮助 → AI 规划服务…（邀请码登录、额度）；登录后“网关 → 离线规则”，拒绝、额度用完、断网都退回离线规则并写明原因；令牌不进诊断包 | `tests/test_gateway_client.py`（对运行中的网关） |
| Agent 循环 | 试运行没通过时，模型最多修正 3 轮（SEM 与液滴），通过或没有新建议就停 | `tests/test_gateway_client.py` |
| 红队集 | 186 条，分三组；规则过滤单独：应拒绝的 119/123，误拒 1/63；调好之后新写的一组 18/22、误拒 1/11 | `tests/benchmarks/redteam_eval.py` |

## 验证门 3：还差什么

| 要求 | 状态 | 需要 |
| :-- | :-- | :-- |
| 红队集 100% 拒绝（整条链） | 规则过滤单独达不到（新写的一组 18/22）；第二层是模型的“超出范围”标志，加上答案格式本身只允许方法、设置和短说明 | **API 密钥**，然后 `redteam_eval.py --gateway URL --token T` |
| 每个规划的 token 用量 p95 不超过上限 | 输入按构造约 1100 token（估计）、输出上限 600 | 密钥，实测 |
| 启用的服务商在规划评测集上不低于离线规则 | 评测工具已就绪（`plan_eval.py --tier gateway`）；合成用例离线规则 21/21 | **真实图片**（SEM 与液滴各 15–30 张，标好正确的方法）+ 密钥 |
| 断网或额度用完时退回离线规则 | ✓ | — |
| 部署在新加坡 | 部署文件已写 | **服务器与域名** |

## 需要你准备

1. DeepSeek 与 Gemini 的 API 密钥（各在服务商后台设每月费用上限）。放在本机不进仓库的 `.env` 或服务器的 `/etc/vexel/gateway.env`，不要发到对话里。
2. 新加坡的服务器与域名（部署时）。
3. 规划评测用的真实图片及其正确的测量方法（不含客户数据）。
