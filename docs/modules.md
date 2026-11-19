# 平台模块（设计说明）

日期：2026-11-19（SEM 计划 M0）。代码：`src/vexel/modules/__init__.py`、`src/vexel/ui/launcher.py`；测试：`tests/test_modules.py` 与 `tests/test_gui.py` 中的启动器测试。

## 为什么要有模块

微流控液滴和 SEM 是两个独立的分析领域，用户、数据、界面都不同，但共用大部分平台能力。模块让两者在同一个程序里**独立运行**：

- 一个模块的界面、依赖或缺陷不影响另一个；
- 新增模块不需要改共用代码。

## 共用与专有

| 共用（平台） | 每个模块自己提供 |
| :--- | :--- |
| 数据底座（`SpatialVolume`）、读取器 | 算子与内置配方（它的 pack：`vexel/packs/<pack>/ops.py`、`recipes/*.json`） |
| 算子注册表、配方引擎、批式与流式执行器 | 桌面窗口 |
| 溯源、质控、复核与修正记录 | 单次运行的报告 |
| 批量运行、设计表、分组统计、复现脚本 | 批量统计的指标表 |
| 插件与模型卡、规划模型与组织策略 | 复核适配器 |
| 诊断包、安装与发布 | 文件识别（这个文件是否属于本模块） |

## 模块表

每个模块是 `vexel.modules.MODULES` 中的一条记录。各部分用 `"包.模块:属性"` 的字符串指定，**用到时才导入**，所以某个模块的依赖缺失或出错，只影响该模块：

```python
Module(
    id="droplet", name="微流控液滴", name_en="Microfluidic droplets", description="…",
    packs=("droplet",),                                    # 算子与配方
    window="vexel.ui.main_window:MainWindow",              # 桌面窗口
    detect="vexel.packs.droplet.detect:score",             # 文件 → 0..1
    report="vexel.report.html_reporter:write_report",      # 单次运行报告
    batch_metrics="vexel.batch.stats:DROPLET_METRICS",     # 分组统计的指标
    review="vexel.packs.droplet.review:adapter",           # 复核适配器
    op_prefixes=("droplet.",),                             # 用到这些算子的配方属于本模块
)
```

共用代码通过注册表找模块，不直接导入任何模块：

- `builtin_recipes()` 汇总所有 pack 的配方；
- 注册表加载所有 pack 的算子；
- `vexel.report.write_report` 按配方所属模块选报告；
- 批量运行按配方所属模块选统计指标；
- 复核窗口按模块选适配器。

一个配方属于哪个模块，由它用到的算子决定，不改配方格式。所以已有配方、结果与溯源记录都不受影响。

## 启动与切换

- `vexel gui`：打开模块启动器（各模块一张卡片）。勾选“下次直接打开上次使用的模块”后跳过启动器。
- `vexel gui <文件>`：各模块的识别函数给文件打分。最高分不低于 0.6 且比第二名高至少 0.2 时，直接打开该模块；否则由人选择。
- `vexel gui --module sem`：直接打开指定模块。
- 每个模块窗口都有“模块”菜单，可以打开其他模块。多个模块可以同时打开，互不影响。
- `vexel modules [文件]`：列出模块；给出文件时显示各模块的打分。
- 开发中的模块（`status="preview"`）在启动器里显示为“开发中”，设置环境变量 `VEXEL_PREVIEW=1` 时可以打开，供开发与内部试用。

## 新增一个模块

1. 在 `vexel/packs/<pack>/` 放算子（`ops.py`）、配方（`recipes/`）和文件识别（`detect.py`）。
2. 写窗口类（无参数构造；有 `open_path(path)` 时启动器会把文件交给它）。
3. 如需要，提供报告、统计指标、复核适配器。
4. 在 `MODULES` 中加一条记录；先设为 `preview`，验收后改为 `ready`。
5. 测试：文件识别、配方归属、窗口能打开。

## M0 的约束

M0 只“搬”不“改”：液滴的算子、配方、窗口、报告文件都没有移动，只是由注册表引用。现有全部测试与回归基准不变地通过，以此保证液滴的行为与结果没有变化。
