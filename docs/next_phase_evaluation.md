# 下阶段开发任务评估

日期：2026-11-18。基础：代码库现状（0.6.0b1，提交 8a58ca8）、[微流控开发进度评估](progress_microfluidics.md)、[液滴内计数评估](eval_cell_counting.md)、路线图 0.0.3，以及对打包、签名、数据来源和同类工具的检索（文末列出来源）。

> **进展（2026-11-18）**：本周三项已完成待验证——跨平台 CI（`.github/workflows/ci.yml`，推送后才会在 GitHub 上第一次运行）；可复现、两地都能完成的安装（`scripts/build_dist.sh`；内测说明 1A/1B）；路线图 0.0.4。

## 一、现状

- **功能**：S1 液滴分析的全部功能已完成，路线图第一至第六阶段都已验收（第六阶段：100 个文件无人值守、导出脚本逐位复现）。模型插件接口已定。
- **真实数据**：只验证过三段公开录像。WSCNet 细胞计数 96%；Droplet-Bead 调参后留出帧 87%。
- **工程**：源码约 13,800 行，测试约 3,900 行（178 个测试，另有 9 个基准）。
- **尚未开始**：内测（需要你执行）；第七阶段（安装包与签名）。

下阶段最大的不确定性不在功能，而在三件事：**能不能顺利装上**、**真实数据上准不准**、**谁来用**。

## 二、新发现的问题（这次检查才看到）

| 问题 | 影响 | 证据 |
| :--- | :--- | :--- |
| **没有持续集成（CI）** | Windows、Linux 从未运行过；每次修改只在一台 Mac 上验证。路线图原定“从第一阶段起在 CI 中出安装包”，没有做到 | 仓库没有 `.github/workflows` |
| **测试者安装时依赖版本不固定** | `pyproject.toml` 只有版本下限，用 .whl 安装时会装上当天最新的 numpy、scikit-image 等，不是我们测过的那一套。任何上游大版本更新都可能让测试者装不上或结果不同 | 依赖全部是 `>=`；`uv.lock` 只对源码安装生效 |
| **中国大陆的安装路径没有验证** | 内测说明里的 uv 安装脚本来自 astral.sh，依赖从 PyPI 下载（约 750 MB 的环境，其中 PySide6 约 350 MB），在大陆可能很慢或失败。路线图待确认事项第 2 条（用户是否在大陆）一直没定 | `.venv` 754 MB；内测说明未提镜像 |
| **路线图与实现不一致**（已更新为 [0.0.4](../vexel_roadmap_0.0.4.md)） | 路线图写的是 napari 界面，实际做的是自研 PySide6 界面；阶段表、风险表都还是旧的。对外沟通和后续规划都会被误导 | `vexel_roadmap_0.0.3.md` 第三、四、十、十一节 |
| 批量报告坐标轴大数字被截断 | 已修复（未提交） | `src/vexel/batch/report.py` |

## 三、候选任务评估

估算按“一人加 AI 辅助”。价值按“对内测和首个客户 beta 的影响”打分。

| # | 任务 | 价值 | 工作量 | 依赖 / 阻塞 | 建议 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | **跨平台 CI**：GitHub Actions 在 macOS、Windows、Linux 上跑测试，并构建 wheel | 高：把“Windows 未测试”变成每次提交都测；内测前最便宜的风险排除 | 2–3 天（预计要修一批 Windows 路径、编码、视频解码问题） | 私有仓库的 Actions 分钟数有限（Windows、macOS 计费更高），需把慢测试分组 | **立即做** |
| 2 | **安装可复现、两地都能完成**：给 wheel 配一份按 `uv.lock` 导出的约束文件；内测说明分大陆与海外两套安装写法（大陆用国内镜像） | 高：直接决定测试者能否装上 | 1 天 | 已确认测试者两地都有 | **立即做** |
| 3 | **路线图更新为 0.0.4** | 中：对内对外口径一致 | 半天 | 无 | **立即做** |
| 4 | **第七阶段原型：未签名安装包**（macOS .app/.dmg，Windows 安装程序），在 CI 里自动构建 | 高：测试者不必装 uv 和 Python；大陆网络问题一并解决 | 1–2 周 | 依赖 1；工具选型见第四节 | 内测期间做，作为第二批测试者的安装方式 |
| 5 | **Excel 导出与报告 PDF** | 中：用户普遍在 Excel 里继续处理；同类学生项目已把 Excel/PDF 导出作为基本功能 | 2 天 | 无 | 内测期间做 |
| 6 | **测试者数据接收流程**：录像加修正记录 → 真值 → 自动加入真实数据回归 | 高：内测的主要产出就是数据；没有流程就会散落 | 3 天 | 有测试者数据 | 内测第一周做 |
| 7 | **计数小模型** | 视数据而定 | 训练管线 1 周，加数据整理 | 需要数据：Gardner 数据集要在 Roboflow 登录下载，许可证未核实；或测试者的修正数据。按[决定条件](eval_cell_counting.md)判断 | 内测后决定；现在只做数据准备 |
| 8 | **签名与公证**（以单位身份） | 中：正式 beta 必需，内测可以不签 | 2–4 天技术工作，外加几周的账号与证书办理 | 单位的 D-U-N-S 编号；Windows 走 Artifact Signing 还是 OV 证书取决于单位注册地，见第四节 | **内测开始时就着手办理** |
| 9 | **联网规划验收**（DeepSeek、通义千问、Claude） | 中：第五阶段验收的未完成项 | 半天（脚本已有） | 需要你提供一个 API 密钥 | 有密钥即可做 |
| 10 | 真实荧光（ddPCR）与粒径参照数据 | 高，但不可控 | – | 需要合作课题组或测试者提供 | 内测招募时专门要 |
| 11 | 平台阶段（ONNX 运行时、SAM、本地训练）与 S2 | 长期价值高 | 数周 | 路线图规定进入 S2 前要有设计伙伴 | S1 结束后再定 |

## 四、第七阶段的技术调研

### 打包工具

| 工具 | 特点 | 适合 Vexel 吗 |
| :--- | :--- | :--- |
| PyInstaller | 最常用；构建快；Qt 官方文档列为满足跨平台要求的工具之一；macOS 签名需开启 hardened runtime 并逐个签名 | **首选**：生态成熟，numpy/scipy/scikit-image/PySide6 都有现成钩子 |
| pyside6-deploy（Nuitka） | Qt 官方工具，把 Python 编译成本地代码；包更小，但构建慢；6.12 起默认改为 standalone 模式 | 备选：若 PyInstaller 体积不可接受再试；对 dask、zarr 等动态导入较多的库风险更高 |
| Briefcase | 生成平台原生安装包 | 未调研到最新情况，暂不考虑 |
| 共性问题 | 都不自带自动更新；两种构建都可能被 Windows Defender 等误报为病毒 | 需要签名降低误报；更新先用“提示有新版本”代替自动更新 |

预计安装包 300–400 MB（PySide6 约占一半），需要裁掉不用的 Qt 模块。

### 签名

- **macOS**：需要 Apple 开发者账号（年费未在本次检索中核实）。公证需要 hardened runtime，Python 应用通常还需要允许未签名可执行内存等例外，用 `notarytool` 提交、`stapler` 装订。技术上有成熟做法，主要是办账号的时间。
- **Windows**：微软的 Trusted Signing 已改名为 Artifact Signing。据社区反馈，2026 年对个人开发者的资格明显收紧，可能只限美国和加拿大，消息不完全一致。另外，2026 年 3 月起公开受信的代码签名证书最长有效期为 458 天。可行路线：以单位（高校或公司）名义申请；或购买商业证书。内测阶段不签名也可以，只是 Windows 会弹 SmartScreen 警告，需要在说明里写明如何放行。

**已决定：以单位身份签名**（2026-11-18）。据此：

- **macOS**：以单位注册 Apple 开发者账号。高校和公司都需要登记在该法人名下的 D-U-N-S 编号，可向邓白氏免费申请，最多约 5 个工作日；注册人若不是单位负责人，还需有人证明其有权代表单位。账号持有人在中国大陆时还需身份验证。中国的公立高校按“教育机构”还是“政府机构”归类，检索中没有找到明确说法，需向 Apple 确认。
- **Windows**：Artifact Signing 的公开信任证书目前据称只对注册在美国、加拿大、欧盟、英国的单位开放，消息来自社区问答，需以微软官方页面为准。**单位注册在其他地区（包括中国大陆）时，用商业的单位验证（OV）代码签名证书**，从 Windows 信任的证书机构购买，签发给单位名称。按现行规则，私钥必须存放在硬件令牌或云端密钥服务里，证书最长 458 天。
- 两边都要几周办理时间，建议内测一开始就着手。

## 五、同类工具的位置（用于内测时的对比提问）

- 免费通用工具：ImageJ/Fiji、CellProfiler（2025 年有专门的无标记液滴管道，但每个样本仍需调整）、Bonsai（2023 年 Lab on a Chip 上发表的实时液滴分析，测粒径、速度、频率）。
- 商业显微软件（ZEISS ZEN、Nikon NIS-Elements）可以做液滴分析，但是通用软件，不针对包裹统计。
- 检索中没有找到同时做**包裹计数与泊松拟合、数字 PCR 判读、以重复为单位的分组统计、完整溯源与复现脚本、在线监控**的工具。这是 Vexel 的差异点，内测问卷应专门验证用户是否在意这些。
- 用户的基本期待：批量处理、导出 Excel 和 PDF。前者已有，后者是任务 5。

## 六、建议的顺序

| 时间 | 做什么 | 完成的标志 |
| :--- | :--- | :--- |
| 本周 | 任务 1、2、3；你开始招募测试者，并开始办理单位的 D-U-N-S 编号与签名证书 | CI 在三个系统上全绿；在 Windows 上能从 wheel 装起来；大陆网络下按新说明装通一次 |
| 内测第 1–2 周 | 发放 wheel；任务 6、5；有密钥就做任务 9 | 收到第一批真实录像，进入回归 |
| 内测第 2–4 周 | 任务 4（未签名安装包，用于第二批测试者）；你开始办签名账号 | macOS 与 Windows 安装包由 CI 自动构建 |
| 内测结束 | 按决定条件判断任务 7；整理反馈，排定修复 | 小模型做或不做的书面结论 |
| 之后 | 签名与公证；S1 结束验收（指标确认单、客户审阅报告、15 分钟可用性） | 路线图第七阶段验收：干净机器一键安装，2–3 个课题组试用 |

**一句话**：下一步最值得投入的不是新功能，而是“装得上、测得到”。先补上跨平台 CI 和可复现安装，再在内测中把安装包做出来，并收集真实数据决定小模型。

## 七、已确认与待确认

已确认（2026-11-18）：

1. **测试者一部分在中国大陆，一部分在海外。** 因此：
   - **安装**：内测说明分两套写法。大陆测试者使用国内 PyPI 镜像（如清华、阿里云镜像，通过 uv 的 `UV_DEFAULT_INDEX` 等设置），uv 本身也给出备用下载方式；海外测试者照常安装。
   - **分发**：安装文件和安装包同时放 GitHub Releases（海外）和一个国内下载点（对象存储或网盘，大陆）。两处文件校验值相同，内测说明里写明 SHA-256。
   - **规划模型**：默认仍是离线规则引擎（两地都能用）。大陆测试者如要用大模型，建议 DeepSeek 或通义千问；海外测试者建议 Claude 或 Gemini。设置界面和隐私说明已经覆盖。
   - **CI**：Windows 安装验证不必依赖身边是否有 Windows 机器，由任务 1 的 CI 完成；但大陆网络下的安装只能由大陆测试者或你本人实测一次。
2. **签名以单位身份进行**，见第四节。

待确认：

1. 用哪个单位签名，单位注册在哪里。这决定 Windows 走 Artifact Signing 还是商业 OV 证书；macOS 需要该单位的 D-U-N-S 编号。
2. 是否提供一个模型 API 密钥（DeepSeek 或其他），用于补测联网规划。
3. 是否同意把路线图更新为 0.0.4（反映自研界面、已完成的阶段和新的风险）。

## 来源

- Qt for Python 部署文档：<https://doc.qt.io/qtforpython-6/deployment/index.html>，<https://doc.qt.io/qtforpython-6/deployment/deployment-pyside6-deploy.html>
- PySide6 与 Nuitka 构建对比（社区）：<https://github.com/Erriez/pyside6-nuitka-test>
- PyInstaller 签名说明：<https://pyinstaller.org/en/v5.1/feature-notes.html>；Python 应用签名与公证实践：<https://haim.dev/posts/2020-08-08-python-macos-app>；Apple 开发者论坛关于逐个签名：<https://developer.apple.com/forums/thread/725061>
- Apple 单位注册与 D-U-N-S：<https://developer.apple.com/support/D-U-N-S/cn/>，<https://developer.apple.com/help/account/membership/identity-verification/>
- Artifact Signing 单位资格（社区问答）：<https://learn.microsoft.com/en-us/answers/questions/2243504/trusted-signing-for-other-countries>，<https://zenn.dev/shm_7ec/articles/signpath-oss-code-signing?locale=en>，产品页 <https://azure.microsoft.com/en-us/products/artifact-signing>
- Azure Trusted Signing / Artifact Signing 资格讨论：<https://learn.microsoft.com/en-us/answers/questions/5810735/cant-create-a-new-trusted-signing-individual-ident>，<https://securityboulevard.com/2026/01/how-to-set-up-azure-trusted-signing-to-sign-an-exe/>
- Gardner 等液滴细胞检测：<https://www.github.com/karl-gardner/droplet_detection>，<https://universe.roboflow.com/karl-gardner-kmk9u/pc3dropletdetection2>，<https://pmc.ncbi.nlm.nih.gov/articles/PMC9706597>
- 同类工具：<https://pubmed.ncbi.nlm.nih.gov/34514234>，<https://www.biorxiv.org/content/10.1101/2025.04.04.647160.full.pdf>，<https://sites.fct.unl.pt/nanotheranostics/publications/open-source-tool-real-time-and-automated-analysis-droplet-based-microf>，<https://digikogu.taltech.ee/en/Download/5d415e34-b331-49b0-a03e-e26bcd4b8a22>
