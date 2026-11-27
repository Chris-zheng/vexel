# 更新记录

## Unreleased — SEM: first acceptance on real images

- Acceptance on 52 Phenom gallery images (data bars read by eye) and three FIB-SEM stacks: `tests/benchmarks/sem_real_eval.py --data <folder>` with the truth in `tests/benchmarks/real/` (images not in the repository).
- Scale bar ends: a JPEG halo along the bar no longer pulls the end ticks inwards (pixel sizes were about 6 % too large on 14 of 52 images).
- Phenom's field width ("FW") is read and cross-checked; field values under a data-bar table header are no longer taken for the scale label; when both the field width and the scale bar are read, the more precise is used and a disagreement beyond their combined uncertainty is flagged.
- ImageJ / Fiji calibrations in TIFF files are read (FIB-SEM stacks).
- Images over 100 MP are refused before their pixels are read, with a message (stitched maps used gigabytes of memory).
- Result: 51 of 52 pixel sizes within 1 % of the field width, the other flagged; recognition names no wrong kind (it names few: 4 of 21).

## Unreleased — SEM in the recipe engine (remote development of 2026-11-27, fixed)

- SEM processing steps, the saturation mask and every SEM measurement run as operators in the recipe engine (`vexel.packs.sem.ops`), with eight built-in recipes (`sem_lines` … `sem_tip`), SEM batch metrics, geometries in `geometries.json`, and an in-memory path for single frames (same result digest as the Zarr path).
- The operators' parameters are generated from the SEM window's own definitions, so names, ranges and defaults match; a test checks that the engine and the window give identical numbers.
- Fixes to the delivered code: the data kinds it used (mask, geometry, height map, ROI) were missing, so no operator module could load; the saturation mask was inverted (measurements used only the saturated pixels); preprocessing steps were called with parameters that do not exist; saturation on 16-bit images used 255 as the limit; the PNG/JPEG reader could not open any file; the API's run endpoint rejected every request.
- `vexel serve` (extra `web`): the engine's HTTP API, local only (127.0.0.1, browsers only from localhost pages, `VEXEL_SERVE_ROOT` limits which files can be run).
- Local model training with hold-out metrics and model cards; tiled inference (`vexel.runtime`). PNG, JPEG, BMP and WebP open as single frames.
- The SEM module stays a preview until the acceptance on real images.

## Unreleased — SEM window: result first

- Opening an image without automatic measurements recognises, plans and tries it at once (switch "打开图片时自动规划", remembered): the trial is drawn on the image and summarised on a card in the panel (image type, method, key numbers, checks, the first reasons) with three choices: accept, adjust (method, settings and region into the panel), do it myself. The plan no longer opens a dialog.
- Method settings live in a collapsed "高级设置" drawer: changed ones are marked with their default in the tooltip, one button restores the defaults; selecting an automatic measurement shows its settings, and a change re-runs it after a moment.
- A calibration banner above the panel: pixel size and source, with "确认" when it waits for a person and "标定…" when there is none.
- The side panel no longer scrolls sideways: long choices and labels shrink or wrap.

## Unreleased — SEM materials measurements and reports

- Particles: solidity, convexity and mean grey level; rule-based classes (agglomerate, irregular / possibly broken, outlier / possibly foreign, normal) with the reason for each, coloured on the image. A round particle split in two by a noisy outline is merged back.
- Pores: porosity (pores cut by the border included) and pore size and shape distributions.
- Fibres: diameter along each fibre's centre line from the grey profile across it (steepest-edge default, not moved by edge brightening), away from crossings, ends and the border; orientation distribution and alignment; gaps between fibres; physical diameters with rectangular pixels. Synthetic fibre mats with ground truth (`vexel.synthetic.sem.fibre_mat`).
- Profile roughness (ISO 21920: Ra, Rq, Rz, Rp, Rv, Rt, Rsk, Rku) on tilt-3D height profiles, with an ISO 16610-21 Gaussian waviness filter (λc setting, `--cutoff`).
- Distribution chart in the SEM window (histogram by number or volume, linear or log, or a scatter plot); pooled statistics over many images in the window and in `vexel sem analyze` (sheet 合并统计, `_pooled.csv`).
- HTML report of an image's record (calibration, processing, annotated image, results, distributions, tilt 3D): export menu and `vexel sem report`.
- Planning: goals such as 孔隙率 or 纤维直径 choose the new recipes, with trial checks and alternatives; the model schema and MCP tools know them.
- Acceptance script `tests/benchmarks/sem_materials_eval.py`.

## Unreleased — user manual

- User manual restructured: a feature list for both modules, a new chapter on starting Vexel (opening the interface, choosing a module, opening the SEM preview, command overview, where data and settings are kept, interface language) and a full SEM chapter; droplet chapters renumbered, SEM limits and FAQ added.
- Launcher: asking for an unknown module, or for the SEM preview without `VEXEL_PREVIEW=1`, now shows the module chooser with an explanation instead of an error.

## Unreleased — SEM window: review and English interface

- English interface for the SEM window (toolbar switch, or `VEXEL_LANG=en`): `vexel.i18n` translates text where it is shown, including messages built with values; `scripts/i18n_extract.py` lists text without a translation; a test opens every tab in English and fails on any Chinese left.
- Review fixes: labels at the image's right edge move left of their point; status messages clear after 10 s; automatic-measurement outlines can be hidden; a processing step's caution is no longer shown twice; clearer profile wording; wider name column.

## Unreleased — SEM M6: tilt 2.5D height maps

- Height maps from two or more SEM views at different stage tilts: rescaling, phase correlation and TV-L1 matching, with a confidence and an uncertainty for every pixel; weighted fusion of several views.
- Levelling, ISO 25178 roughness, height profiles, ISO 5436-1 step heights (two parallel planes); height-map TIFF, STL/OBJ mesh, coloured height map, 3D preview.
- "三维" tab in the SEM window; the reconstruction is kept in the reference image's record; `vexel sem tilt3d`.
- Acceptance script `tests/benchmarks/sem_tilt3d_eval.py`.

## Unreleased — SEM M5 (first half): recognition and automatic planning

- SEM probe: recognises line/space, contact-hole arrays, particles, layer stacks, cross-section profiles, tips or other; never guesses ("unknown" with candidates). Measures noise, stripes, uneven brightness and saturation, and suggests processing that does not harm the measurement.
- SEM planner on the platform agent: recipe, settings, region and processing with reasons, a trial with per-recipe checks and adjustments, recipe memory of accepted plans, optional language-model tiers checked against an allowlist taken from the recipe definitions.
- "自动规划" in the SEM window; `vexel sem plan`, `vexel sem analyze --auto`; MCP tools `sem_calibrate`, `sem_plan`, `sem_analyze`.
- Fixes found on the way: weak layer interfaces next to strong ones, tip radius fitting range, particle minimum size in plans.
- Acceptance script `tests/benchmarks/sem_recognition_eval.py`.

## Unreleased — SEM M4: semiconductor metrology and segmentation

- Automatic measurements on a region, kept in the image's record: line/space CD, pitch, LER/LWR with noise-bias correction; contact holes with array pitch and missing holes; particles with Feret diameters and size distributions; cross-section layer thickness; line/trench profiles with sidewall angles; tip radius and cone angle.
- "自动测量" tab with region selection; Excel sheets and annotated pictures include the results.
- `vexel sem analyze`: many images at once, with repeatability (mean, SD, 3σ) per folder.
- Acceptance script `tests/benchmarks/sem_metrology_eval.py`.

## Unreleased — SEM M3: image processing

- Line levelling, Fourier destriping, background levelling, scan-line alignment, denoising, contrast/CLAHE and sharpening; the chain is kept in the measurement record.
- Saturated areas are marked (never repaired) and left out of the magic wand and line profiles.
- Before/after view, multi-frame alignment and averaging, processed-image export with the source calibration.
- `vexel sem process` and `vexel sem average`; acceptance script `tests/benchmarks/sem_processing_eval.py`.

## 未发布

- **SEM 手动测量**（SEM 计划 M2）：长度、折线、面积（多边形、自由画、魔棒）、角度、分类计数、圆与椭圆拟合、线剖面（阈值、最大梯度、亮边峰值三种边缘定义；线宽、间隔、按自相关与按边缘的周期）；倾斜校正（1/cos θ）；拖动修改、撤销与重做；测量按图像 SHA-256 自动保存并在重新打开时恢复；导出 Excel、CSV、带标注与比例尺的 PNG/SVG、可复现的测量记录。`vexel sem measure`、`vexel sem profile`。合成几何验收：误差均不超过 1 像素对应量，线宽 ≤ 0.5 像素、周期 ≤ 0.2 像素。新增依赖 openpyxl（Excel 导出）。
- **SEM 读取与标定**（SEM 计划 M1）：读取 Thermo Fisher（FEI）、Zeiss、TESCAN、Hitachi、JEOL 的仪器元数据（像素尺寸、成像区高度、电压、放大倍数、视场宽度、工作距离、倾角）；自动识别并裁掉信息栏；测量比例尺（端点刻度、圆头、虚线刻度三种画法）并用文字识别读出标注；与视场宽度或元数据交叉核对；手动标定（点比例尺两端或输入像素尺寸）；经人确认的标定按文件 SHA-256 记录。`vexel sem calibrate`、`vexel sem info`；SEM 窗口显示信息栏、比例尺、来源、核对与确认。在真实的 FEI 与 Zeiss 文件上，信息栏识别与元数据相差 0.18%–0.98%。
- 文字识别为可选组件 `vexel[sem]`（RapidOCR、ONNX Runtime，本机运行、不联网，在独立进程中运行以免与视频库冲突）；离线安装包已包含。
- 合成 SEM 图像生成器（五种厂商布局、带精确真值）与标定验收脚本 `tests/benchmarks/sem_calibration_eval.py`（可直接对真实数据文件夹运行）。
- **模块化平台**（SEM 计划 M0）：微流控液滴与 SEM 作为独立模块运行；`vexel gui` 先显示模块启动器，按文件类型自动进入对应模块；每个窗口有“模块”菜单；`vexel gui --module`、`vexel modules`。SEM 模块为“开发中”（`VEXEL_PREVIEW=1` 可打开预览：识别 Thermo Fisher、Zeiss、TESCAN、Hitachi/JEOL 的元数据）。液滴模块行为不变。设计见 docs/modules.md。
- **组织策略**：管理员可用系统位置的 `policy.json` 锁定离线运行（只允许列出的内部服务）、禁止发送图像；`vexel policy`；规划模型设置窗口与诊断包显示策略。
- **离线安装包**：`scripts/build_offline_bundle.py` 生成含 uv、Python 与全部依赖的安装包，用于无法联网的计算机；在 macOS 上实测离线安装与分析，CI 每周在 Windows、Linux 上实测。
- 跨平台持续集成：Linux、Windows 每次提交运行全部测试，macOS 与回归基准每周；每次都构建 wheel 并按测试者的方式安装试运行（`.github/workflows/ci.yml`）。
- `scripts/build_dist.sh`：生成 wheel、由 `uv.lock` 导出的依赖版本约束文件和 SHA-256 校验值；测试者用 `uv tool install <wheel> -c <constraints>` 安装到与测试相同的版本。
- 内测说明分海外与中国大陆两套安装方式（大陆使用 USTC 与清华镜像）。
- 命令行在无法显示某些字符的终端（如 GBK 编码的 Windows 控制台）上不再因输出而中断。
- 批量报告：大数值坐标轴标注不再被截断。
- 界面截图（docs/screenshots）、下阶段开发任务评估、路线图 0.0.4。

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
