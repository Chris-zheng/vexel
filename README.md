# Vexel

智能体原生的科学图像分析平台：LLM 负责规划，确定性算子负责计算，每个结果可复核、可复现。
使用说明见 [用户手册](docs/user_manual.md)，规划见 [vexel_roadmap_0.0.3.md](vexel_roadmap_0.0.3.md)。当前进度：**Phase 1（引擎与 CLI）**。

## 快速开始

桌面界面：

```bash
uv sync
uv run vexel gui
```

命令行：

```bash
uv run vexel synth data/demo                       # 合成液滴视频 + 精确真值
uv run vexel run data/demo/droplets.ome.tif -r droplet_basic -o runs/demo
uv run vexel validate runs/demo --truth data/demo/truth
uv run vexel synth data/cells --cells 1.0                # 液滴内按泊松分布放入细胞
uv run vexel run data/cells/droplets.ome.tif -r droplet_encapsulation -o runs/cells
uv run vexel synth data/fluo --fluo 0.7                  # 加荧光通道（数字检测：阴阳液滴）
uv run vexel run data/fluo/droplets.ome.tif -r droplet_fluorescence -o runs/fluo
uv run pytest                    # 常规测试
uv run pytest -m benchmark       # 回归基准（约 2 分钟）
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
| `vexel gui` | 打开桌面界面 |
| `vexel report` | 重新生成一次运行的 HTML 报告 |
| `vexel run` | 在录好的数据上执行配方 |
| `vexel info` | 只读元数据，不读像素 |
| `vexel ops` | 列出算子：作用范围、输入输出类型、参数 |
| `vexel recipes` | 列出内置配方，`--show NAME` 打印 JSON |
| `vexel validate` | 把液滴运行结果与真值比对 |
| `vexel synth` | 生成带真值的合成液滴视频 |
| `vexel convert` | 转为 OME-Zarr 多分辨率金字塔 |
| `vexel licenses` | 已安装依赖的许可证清单，标出 GPL / 非商用项 |

`batch`、`watch`、`replay` 属于后续阶段。

## 一次运行的产物

```text
runs/demo/
├── recipe.json        # 冻结的配方，所有默认参数已展开
├── summary.json       # 汇总统计
├── tables/*.csv       # 每帧检测、追踪结果、每个液滴一行的表
├── review/*.png       # 分割轮廓叠加图，供人工复核
├── review/flags/      # 可疑液滴的放大裁切图
├── report.html        # 单文件 HTML 报告，可直接发送或打印
└── provenance.json    # 输入哈希、配方哈希、依赖版本、各步耗时与质控指标、结果摘要
```

同一配方加同一数据得到相同的 `result_digest`。

## 结构

```text
src/vexel/
├── core/      SpatialVolume（惰性 TCZYX + 物理间距）、内存水位线、OME-Zarr 缓存
├── readers/   读取器插件：TIFF/OME-TIFF、图像序列、视频、npy、OME-Zarr
├── sources/   文件数据源；相机仅接口
├── ops/       算子注册表（frame / window / volume 范围声明）与基础算子
├── packs/     领域分析包；droplet/ 含汇总算子、内置配方、真值校验
├── engine/    配方、批式执行器、质控、溯源
├── synthetic/ 合成样本生成器
├── ui/        桌面界面（PySide6，经 qtpy），只调用引擎
└── cli.py
```

新增算子：用 `@register_op` 声明范围、输入输出类型和 pydantic 参数模型，返回 `OpResult`，
并在 `ops/registry.py` 的 `load_builtin_ops` 中导入所在模块。
新增格式：继承 `readers.base.Reader` 并 `register_reader`。

## Phase 1 的边界

- 批式执行器处理 2D 时序（配方选定一个通道和一个 z 平面）；`volume` 范围算子可注册，尚不执行。
- 质控规则已可写在配方里（`warn` / `stop`），完整质控门与 HTML 报告在 Phase 2。
- `vexel validate` 的默认容差是占位值，待与客户约定后替换。
- 合成生成器放在包内（`vexel.synthetic`）以便 `vexel synth` 使用，而非路线图中的 `tests/synthetic/`。
