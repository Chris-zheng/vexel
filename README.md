# Vexel

智能体原生的科学图像分析平台：LLM 负责规划，确定性算子负责计算，每个结果可复核、可复现。
使用说明见 [用户手册](docs/user_manual.md)，规划见 [vexel_roadmap_0.0.5.md](vexel_roadmap_0.0.5.md)（SEM 模块见 [开发计划书](docs/sem_module_plan.md)）（下一步见 [下阶段开发任务评估](docs/next_phase_evaluation.md)）。当前版本：**0.7.0b1 内测版**（SEM 模块预览版内测：[SEM 内测说明](docs/closed_testing/sem/README.md)）（[更新记录](CHANGELOG.md)，[内测说明](docs/closed_testing/README.md)，[微流控开发进度评估](docs/progress_microfluidics.md)）。Phase 1–6 完成：引擎与 CLI、液滴包与质控、流式执行（验收见 [docs/phase3_acceptance.md](docs/phase3_acceptance.md)）、界面与人工纠错（见 [docs/corrections.md](docs/corrections.md)；“15 分钟上手”待真人测试，方案见 [docs/usability_test.md](docs/usability_test.md)）、智能体（探针、规划器、配方记忆、可选 Claude / DeepSeek / 通义千问 / Gemini / 本地模型 / 离线规则、MCP 服务；验收见 [docs/phase5_acceptance.md](docs/phase5_acceptance.md)）。批处理与分组统计（`vexel batch`，界面里的“批量分析…”；以重复为统计单位；导出可独立复现的脚本；中英文方法段落；验收见 [docs/phase6_acceptance.md](docs/phase6_acceptance.md)）；模型插件接口已确定（[docs/plugin_interface.md](docs/plugin_interface.md)）。Phase 7 进行中：跨平台 CI（`.github/workflows/ci.yml`）与可复现安装（`scripts/build_dist.sh` 生成 wheel、依赖约束与校验值）已做，安装包与签名待做。界面截图见 [docs/screenshots](docs/screenshots/README.md)。

## 快速开始

桌面界面：

```bash
uv sync                 # 加 --extra agent 安装可选的 Claude SDK 与 MCP 服务
uv run vexel gui
```

命令行：

```bash
uv run vexel synth data/demo                       # 合成液滴视频 + 精确真值
uv run vexel run data/demo/droplets.ome.tif -r droplet_basic -o runs/demo
uv run vexel validate runs/demo --truth data/demo/truth
uv run vexel replay data/demo/droplets.ome.tif -r droplet_basic --speed 0.05   # 按 1/20 倍速在线回放
uv run vexel synth data/cells --cells 1.0                # 液滴内按泊松分布放入细胞
uv run vexel run data/cells/droplets.ome.tif -r droplet_encapsulation -o runs/cells
uv run vexel synth data/fluo --fluo 0.7                  # 加荧光通道（数字检测：阴阳液滴）
uv run vexel run data/fluo/droplets.ome.tif -r droplet_fluorescence -o runs/fluo
uv run pytest                    # 常规测试
uv run pytest -m benchmark       # 回归基准（约 2 分钟）
```

批量与分组统计：

```bash
uv run vexel batch init data/exp01                  # 写出 data/exp01/design.csv，填写 group 与 replicate
uv run vexel batch run data/exp01/design.csv -r droplet_encapsulation -o runs/exp01
python runs/exp01/reproduce.py                      # 独立复现全部结果并核对
```

对自己的数据：

```bash
uv run vexel info  my_video.avi
uv run vexel run   my_video.avi -r droplet_basic --pixel-size 2.0 --frame-interval 0.001
uv run vexel run   my_video.avi -r my_recipe.json --set seg.min_score=0.5
```

没有像素尺寸时结果以像素为单位，`summary.json` 中 `calibrated` 为 `false`。
标定也可以写在旁挂文件里：`<文件名>.vexel.json` 或 `<图像序列目录>/vexel.json`，
内容如 `{"pixel_size": 2.0, "space_unit": "um", "frame_interval": 0.001}`。

## 命令

| 命令 | 作用 |
| :--- | :--- |
| `vexel gui` | 打开桌面界面（模块启动器；`--module` 直接进入某个模块） |
| `vexel report` | 重新生成一次运行的 HTML 报告 |
| `vexel run` | 在录好的数据上执行配方 |
| `vexel probe` | 数据指纹：亮度、噪声、运动、液滴尺寸、细胞、通道角色；可存四宫格诊断图 |
| `vexel plan` | 自动规划：按数据和目标起草配方，在前几百帧试跑并调整，`--run` 接受后分析并记入配方记忆 |
| `vexel mcp` | MCP 服务（stdio）：让 AI 助手调用探针、规划、运行与读取结果 |
| `vexel brush` | 画笔训练：在图上标出液滴和背景，训练像素分类器，写出使用它的配方 |
| `vexel review` | 打开复核与纠错窗口：修正轮廓、细胞数、阴阳性，排除杂质 |
| `vexel correct` | 按修改记录（corrections.json）重算，结果写入 `<结果>/corrected` |
| `vexel live` | 打开在线监控看板：回放录像或监视采集文件夹，实时显示当前帧、滚动统计与报警 |
| `vexel watch` | 监视采集文件夹：每到一张图就分析，增量写入 OME-Zarr，采集结束后自动全量终算 |
| `vexel replay` | 把录像按原始帧率（或倍速）逐帧送入流式执行器：滚动统计、漂移报警、处理帧率与丢帧 |
| `vexel info` | 只读元数据，不读像素 |
| `vexel ops` | 列出算子：作用范围、输入输出类型、参数 |
| `vexel recipes` | 列出内置配方，`--show NAME` 打印 JSON |
| `vexel validate` | 把液滴运行结果与真值比对 |
| `vexel synth` | 生成带真值的合成液滴视频 |
| `vexel convert` | 转为 OME-Zarr 多分辨率金字塔 |
| `vexel batch` | 批量：`init` 从文件夹起草设计表，`run` 用一个配方分析表里每个文件（无人值守、可续跑），按组统计并导出报告、方法段落与复现脚本 |
| `vexel modules` | 列出平台模块（微流控液滴、SEM）；给出文件时判断它属于哪个模块 |
| `vexel sem calibrate` / `vexel sem info` | SEM 图像的像素尺寸（仪器元数据、信息栏比例尺识别或手动）与仪器信息 |
| `vexel sem measure` / `vexel sem profile` | SEM 测量：重新计算并导出一张图的测量（Excel、CSV、标注图）；线剖面的边缘、线宽与周期 |
| `vexel sem process` / `vexel sem average` | SEM 图像处理：逐行校平、去条纹、背景校平、行错位校正、去噪、对比度；饱和区标记；多帧对齐平均 |
| `vexel sem analyze` | SEM 自动测量：线宽与周期、LER/LWR、接触孔、颗粒、截面层厚与侧壁角、尖端；多图重复性 |
| `vexel sem plan` | SEM 自动规划：识别图像类型，选择测量方法、设置、区域和处理，先试运行；`--apply` 加入记录 |
| `vexel sem tilt3d` | SEM 倾转三维：由不同倾角的图重建 2.5D 高度图，粗糙度、剖面、台阶高度，导出 TIFF / STL / OBJ |
| `vexel policy` | 显示当前生效的组织策略（离线运行、允许的内部服务）与实际可用的规划模型 |
| `vexel plugins` | 已安装的插件与模型：算子、许可证、训练数据、权重文件状态 |
| `vexel bugreport` | 导出诊断包（版本、设置、配方、汇总；不含图像和密钥），用于报告问题 |
| `vexel licenses` | 已安装依赖的许可证清单，标出 GPL / 非商用项 |

`batch` 属于后续阶段（Phase 6）。

## 一次运行的产物

```text
runs/demo/
├── recipe.json        # 冻结的配方，所有默认参数已展开
├── summary.json       # 汇总统计
├── tables/*.csv       # 每帧检测、追踪结果、每个液滴一行的表
├── review/*.png       # 分割轮廓叠加图，供人工复核
├── review/flags/      # 可疑液滴的放大裁切图
├── report.html        # 单文件 HTML 报告，可直接发送或打印
├── provenance.json    # 输入哈希、配方哈希、依赖版本、各步耗时与质控指标、结果摘要
├── stream/            # 仅 replay / watch：rolling.csv 滚动统计，alarms.csv 警报，events/ 警报时的画面
├── frames.zarr        # 仅 watch：采集到的帧（OME-Zarr，逐帧追加）
├── final/             # 仅 watch：采集结束后对 frames.zarr 的全量分析
├── corrections.json   # 复核时所做的人工修改
└── corrected/         # 按人工修改重算的结果（格式同上）
```

同一配方加同一数据得到相同的 `result_digest`。

## 结构

```text
src/vexel/
├── core/      SpatialVolume（惰性 TCZYX + 物理间距）、内存水位线、OME-Zarr 缓存
├── readers/   读取器插件：TIFF/OME-TIFF、图像序列、视频、npy、OME-Zarr
├── sources/   文件、回放、监视文件夹数据源；相机仅接口
├── ops/       算子注册表（frame / window / volume 范围声明）与基础算子
├── packs/     领域分析包；droplet/ 含汇总算子、内置配方、真值校验
├── engine/    配方、批式执行器、流式执行器、质控、溯源、人工修正
├── agent/     探针、规划器、配方记忆、模型抽象层、MCP 服务
├── synthetic/ 合成样本生成器
├── ui/        桌面界面与在线监控看板（PySide6，经 qtpy），只调用引擎
└── cli.py
```

新增算子：用 `@register_op` 声明范围、输入输出类型和 pydantic 参数模型，返回 `OpResult`，
并在 `ops/registry.py` 的 `load_builtin_ops` 中导入所在模块。
新增格式：继承 `readers.base.Reader` 并 `register_reader`。

## 当前边界

- 执行器处理 2D 时序（配方选定一个通道和一个 z 平面）；`volume` 范围算子可注册，尚不执行。
- 流式执行器只接受逐帧算子、静态背景、一个追踪步骤和只读表格的算子；数据源有回放和监视文件夹，相机接入推迟到硬件对接阶段。
- `vexel validate` 的默认容差是占位值，待与客户约定后替换。
- 合成生成器放在包内（`vexel.synthetic`）以便 `vexel synth` 使用，而非路线图中的 `tests/synthetic/`。
