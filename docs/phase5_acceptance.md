# 第五阶段（智能体）验收记录

日期：2026-11-14。复现：`VEXEL_OFFLINE=1 uv run python tests/benchmarks/agent_acceptance.py`（离线规则引擎）；联网测 Claude：`uv run python tests/benchmarks/agent_acceptance.py --model claude`（需要 Anthropic 凭据）。

## 验收标准（路线图第十节）

> 断网功能完整；联网时配方无需修改即通过质控的比例达到约定值。

## 1. 断网功能完整

设置 `VEXEL_OFFLINE=1`（或没有任何模型可用）时，所有智能体功能都由离线规则引擎完成，不发出任何网络请求：

| 功能 | 断网时 | 测试 |
| :--- | :--- | :--- |
| 探针 `vexel probe`、四宫格诊断图 | 完整（本来就不联网） | `tests/test_agent.py` |
| 规划 `vexel plan`：理解目标、起草、试运行、调整 | 完整（规则引擎） | 同上；下表即离线结果 |
| 配方记忆 | 完整（本地文件） | 同上 |
| MCP 服务 `vexel mcp` | 完整（本地 stdio） | `test_mcp_over_stdio` 在 `VEXEL_OFFLINE=1` 下运行 |
| 桌面界面“自动规划” | 完整 | `test_auto_plan_fills_the_settings` 在 `VEXEL_OFFLINE=1` 下运行 |
| 模型不可用时的降级 | Claude → 本地模型 → 规则，逐级记录原因 | `test_fallback_order_and_offline_switch` |

## 2. 无需人工修改即通过质控的比例

每组数据：规划器按数据和目标起草配方，在前 300 帧试运行（必要时自动调整），然后用规划出的配方分析全部帧，不做任何人工修改。“通过”= 全量运行没有任何质控提示，且（有真值时）液滴数、粒径、频率、细胞数、阴阳性都在验证容差内。

规划层级：rules（实际回答：rules）。机器：arm64，Darwin。

| 数据 | 目标 | 规划出的配方 | 规划状态 | 试运行后调整 | 全量运行无质控提示 | 与真值一致 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| basic | 粒径和生成频率 | droplet_basic | ready | 否 | 是 | 是 |
| small_fast | （未写） | droplet_basic | ready | 否 | 是 | 是 |
| large_slow | measure droplet size | droplet_basic | ready | 否 | 是 | 是 |
| noisy | （未写） | droplet_basic | ready | 否 | 是 | 是 |
| polydisperse | （未写） | droplet_basic | ready | 否 | 是 | 是 |
| cells | 数每个液滴里的细胞 | droplet_encapsulation | ready | 否 | 是 | 是 |
| cells_unasked | （未写） | droplet_encapsulation | ready | 否 | 是 | 是 |
| cells_dense | 包裹率 | droplet_encapsulation | ready | 否 | 是 | 是 |
| fluo | ddPCR 阳性比例 | droplet_fluorescence | ready | 否 | 是 | 是 |
| fluo_unasked | （未写） | droplet_fluorescence | ready | 否 | 是 | 是 |
| fluo_cells | 荧光阳性和每滴细胞数 | droplet_fluorescence_encapsulation | ready | 否 | 是 | 是 |
| vertical | （未写） | droplet_basic | ready | 否 | 是 | – |
| real: DropletMicrofluidics.mov | （未写） | droplet_basic | check | 否 | 否 | – |
| real: WSCNet sequence | 每个液滴里的细胞数 | droplet_encapsulation | check | 是 | 否 | – |

**无需人工修改即通过质控（且与真值一致）：12 / 14**；其中合成数据 12 / 12。

两段真实录像都标为 check（可运行，有质控提示）而不是 ready：.mov 有 1/9 的液滴被标为可疑（上限 10%）；WSCNet 序列的细胞计数帧间一致率偏低（重叠细胞，见用户手册 5.5 节），规划器提高了细胞检测阈值仍未消除。两者都如实报告，没有掩盖。

## 结论与局限

- **断网功能完整：通过。**
- **离线规则引擎**下，12 组合成数据全部无需修改即通过质控并与真值一致；2 段真实录像能运行但有质控提示。
- **联网（Claude）的比例尚未测量**：本机没有 Anthropic 凭据。Claude 层级的请求格式、结构化输出、拒答降级已用替身测试；真实比例需在有凭据的机器上运行上面的命令。路线图要求的“约定值”也尚未与客户约定。
- 合成数据比真实数据理想得多；真实数据只有两段。比例要在客户的真实数据集上再测。
