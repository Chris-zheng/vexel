# 复核与纠错：设计说明

本文说明人工修正的数据格式、执行方式，以及新的分析包如何接入。用户操作见用户手册 2.8 节。

## 原则

1. **修正是数据，不改原结果。** 修改记录在运行目录的 `corrections.json` 里；按修正重算的结果写入 `corrected/`，原运行不动。
2. **可复现。** 修正由批式执行器在产生目标数据的那一步之后立刻应用，后面的测量、追踪、统计都看到修正后的数据。配方 + 输入 + 修正 → 唯一的结果摘要（`result_digest`），溯源记录里有修正文件的哈希和每条修改是否生效。
3. **用位置指认目标，不用编号。** 每条修改给出帧号和一个点 `at = [y, x]`（像素）。标签编号、轨迹编号在重新分割、重新追踪后会变，位置不会。
4. **修改种类与分析无关。** 所有分析都产出两类东西：每帧的标签图（`labels`），每个目标一行的表（`table`）。修改只针对这两类。
5. **人看过的就是确认过的。** `table.set` 修改的行自动带 `verified = true`；分析包可以据此不再标记可疑、不再按“不稳定”剔除。

## 修改种类

| 种类 | 作用于 | 用途 | 必填 |
| :--- | :--- | :--- | :--- |
| `labels.delete` | 标签图 | 删除误检（杂质、气泡、背景纹理） | `at` |
| `labels.merge` | 标签图 | 一个目标被切成几块 | `points` |
| `labels.paint` | 标签图 | 补画漏检；给了 `at` 则扩展该目标的轮廓 | `shape` |
| `labels.erase` | 标签图 | 修剪轮廓，或沿一条线把粘连目标切开 | `shape` |
| `table.set` | 表 | 修改某个目标的数值或类别 | `at`、`values` |
| `table.exclude` | 表 | 从统计中去掉目标；作用在追踪表上时去掉整条轨迹 | `at` |

`shape` 为 `{"circle": [y, x, r]}` 或 `{"polygon": [[y, x], ...]}`。

点到行的对应：表里有每帧质心（`frame`、`centroid_y_px`、`centroid_x_px`）时，取该帧质心离点最近、且点落在目标内（0.6 倍直径以内，至少 5 像素）的那一行；表是每个轨迹一行（有 `track_id`）时，先在追踪表里这样找到轨迹，再取该轨迹的行。找不到时该条修改记为“没有找到目标”，不报错，结果里写明。

## 文件格式

```json
{
  "schema_version": 1,
  "base": {"run": "/data/exp01_vexel/20261114-101500", "recipe_sha256": "…", "result_digest": "…"},
  "edits": [
    {"kind": "labels.delete", "target": "seg.labels", "frame": 120, "at": [128.0, 301.5], "note": "气泡"},
    {"kind": "labels.paint", "target": "seg.labels", "frame": 120, "shape": {"circle": [130, 610, 15]}},
    {"kind": "table.set", "target": "droplets.droplets", "frame": 88, "at": [127.2, 402.0], "values": {"n_inclusions": 2}},
    {"kind": "table.exclude", "target": "track.table", "frame": 40, "at": [126.0, 210.4]}
  ]
}
```

`author`、`time`、`note` 可选，不影响结果（也不进入修正文件的哈希）。

## 分析包如何接入：复核适配器

通用复核界面只依赖 `vexel.engine.review.ReviewAdapter`：

| 字段 | 含义 | 液滴包 |
| :--- | :--- | :--- |
| `objects` | 每个复核对象一行的表 | `droplets.droplets`（有荧光时 `positivity.droplets`） |
| `labels` | 轮廓工具编辑的标签图 | `seg.labels` |
| `exclude` | “排除”作用的表 | `track.table`（去掉整条轨迹） |
| `flags` | 需要复核的对象（帧、位置、原因） | `flags.table` |
| `fields` | 可人工设定的值（`EditableField`：列名、显示名、作用的表、类型） | 细胞数 `n_inclusions`；阴阳性 `positive_manual` |
| `reasons` | 可疑原因代码 → 显示文字 | 形状不圆、粒径异常、计数不稳、近阈值、帧数少 |

分析包在 `vexel/packs/<包>/review.py` 里提供 `adapter(recipe)`，并登记到 `vexel.packs.review_adapter`。认不出的配方得到一个通用适配器（复核对象取追踪表或第一张表，只能编辑轮廓、排除目标）。

分析包自己负责让人工值生效：例如液滴包的阴阳判读算子读到 `positive_manual` 时用它代替自动判定，泊松统计保留 `verified` 的液滴，可疑液滴列表跳过 `verified` 的液滴。

## 用两个未来场景检验

**荧光细胞分割（S2）。** 配方：背景校正 → 分割（`seg.labels`）→ 测量（每帧每个细胞一行）→ 可选追踪 → 细胞分类。

- 粘连细胞没分开：`labels.erase` 沿接触处画一条线（多边形）切开，或删除后用两个 `labels.paint` 分别补画。
- 一个细胞被分成两块：`labels.merge`。
- 改细胞类型（如有丝分裂期）：`table.set`，`values: {"class_manual": "mitotic"}`，由分类算子读取。
- 不追踪的静态图：目标表就是测量表，点按每帧质心对应，格式不用改。

**工业 2D 缺陷检测（S5）。** 配方：配准 → 与模板作差 → 分割缺陷（`defects.labels`）→ 测量 → 缺陷分类与判定。

- 误报（纹理、灰尘）：`labels.delete`，或对整个缺陷 `table.exclude`。
- 漏检的划痕：`labels.paint` 用多边形描出。
- 改缺陷类别或严重程度：`table.set`，`values: {"class_manual": "scratch", "severity_manual": 2}`，由判定算子读取并重算合格与否。
- 每张图独立：`frame` 就是图像序号。

两个场景都只用到上表的六种修改，差别全在适配器（复核哪张表、哪些字段可改）和各自算子如何读取人工值。

## 尚未做的

- 3D 体数据的编辑（格式上 `frame` 之外需要 `z`；等 3D 执行落地时加 `z` 字段，旧文件保持有效）。
- 修正后只重算受影响的部分：目前总是整段重算；短视频几秒，长视频可能要几分钟。
- 从修正中学习（画笔分类器、按修正调参）。
