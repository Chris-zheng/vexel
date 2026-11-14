# 更新记录

## 0.6.0b1（内测版，2026-11）

第六阶段：批处理与实验级统计。

- `vexel batch init` / `vexel batch run` 与桌面界面的“批量分析（多个文件、分组统计）…”：一个配方分析设计表里的每个录像；单个文件出错只记录不中断；中途停止后再运行同一命令从停下处继续。
- 设计表：文件、组、重复、单文件标定与设置、任意元数据列；可从文件夹自动起草（子文件夹名作组与重复）；Excel 存的 UTF-8 / GBK CSV 都能读。
- 分组统计以重复为单位（同组同一重复的文件先平均）：描述统计与 95% 置信区间；两组 Welch t、均值差 CI、Hedges' g、Mann–Whitney U；多组 Welch ANOVA、Kruskal–Wallis、η²、两两 Welch t 加 Holm 校正；设计薄弱时给出提示。
- 批量报告（分组图、表、文件列表、方法、溯源）、`results.csv` / `replicates.csv` / `groups.csv` / `comparisons.csv`。
- `reproduce.py`：在新进程中重跑全部文件、重算统计，并逐个核对结果摘要。
- 英文方法段落：单次报告与批量报告都附英文 Methods，另存 `methods_en.txt`。
- 包裹计数：平滑后合并成一个峰的相邻细胞按斑点大小拆分（默认开启），WSCNet 真实数据上 92% → 96%；可选的小杂质过滤 `inclusions.min_size_fraction`。
- 液滴不动（抽帧、停流、静态阵列）时，探针按尺寸扫描找出液滴、规划器自动设置识别范围并关闭运动门控；`seg.motion_threshold=0` 表示不用运动门控。
- 液滴内计数评估（WSCNet 与 Droplet-Bead 两套真实数据）与“是否需要小模型”的决定条件：docs/eval_cell_counting.md。
- 模型插件接口：入口点 `vexel.plugins`、模型卡（版本、权重 SHA-256、许可证、训练数据、能否商用）、任务约定、溯源记录；`vexel plugins` 命令。见 docs/plugin_interface.md。

## 0.5.0b1（内测版，2026-11）

第一个对外的内测版本，覆盖路线图第一至第五阶段。详见 [用户手册](docs/user_manual.md) 与 [内测说明](docs/closed_testing/README.md)。

**分析（液滴包 S1）**
- 粒径均值、标准差、CV；按流道高度换算压扁液滴的体积。
- 追踪加虚拟计数线，跨帧去重计数；生成频率、间距、速度。
- 包裹计数、空/单/多包裹比例、泊松拟合与 λ 置信区间。
- 单色荧光：每滴荧光强度、自动阴阳判读、阳性比例、每滴拷贝数。
- 质控规则、可疑液滴列表、叠加复核图、单文件 HTML 报告、完整溯源记录。

**流式**
- `vexel replay`、`vexel watch`、`vexel live`：回放或边采集边分析，滚动统计、超限与漂移报警，增量写入 OME-Zarr，采集结束后自动全量终算。

**界面与人工纠错**
- 桌面界面与首次使用引导；画笔训练像素分类器；复核与纠错窗口（轮廓、细胞数、阴阳性、排除），按修改重算。

**智能体**
- `vexel probe`：数据指纹与四宫格诊断图。
- `vexel plan` 与界面里的“自动规划…”：按数据和一句话目标起草配方，在前几百帧试跑并调整，交给你确认。
- 配方记忆：记住在相似数据上通过质控的设置。
- 规划模型可选：Claude、DeepSeek、通义千问、Gemini、任何 OpenAI 兼容服务、本地 Ollama，或完全离线的规则引擎；不可用时自动降级。密钥只从环境变量读取，默认只发数字不发图像。
- `vexel mcp`：MCP 服务，供 AI 助手调用。

**内测支持**
- `vexel bugreport` 与 帮助 → 导出诊断包…：版本、设置、配方、汇总，不含图像和密钥。
- 微流控开发进度评估、内测说明、反馈表、已知问题、组织清单。

已知问题见 [docs/closed_testing/known_issues.md](docs/closed_testing/known_issues.md)。
