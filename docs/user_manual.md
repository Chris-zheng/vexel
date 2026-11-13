# Vexel 用户手册

适用版本：0.1.0（Phase 1：引擎与命令行）

本手册说明当前版本能做什么、怎么用、结果怎么读。当前版本提供桌面界面（第 2 节）和命令行（第 3 节起）两种用法，两者调用同一个分析引擎，默认设置下结果完全相同；
分析对象是**微流控液滴的 2D 明场时序图像**（录好的视频或图像序列）。

---

## 1. 安装

需要 [uv](https://docs.astral.sh/uv/) 和 Python 3.12（uv 会自动准备）。

```bash
git clone https://github.com/Chris-zheng/vexel.git
cd vexel
uv sync
```

确认安装成功：

```bash
uv run vexel --help
```

桌面界面随同安装，不需要额外步骤。

下文所有命令都写成 `uv run vexel ...`。如果已激活虚拟环境（`source .venv/bin/activate`），可以直接写 `vexel ...`。

---

## 2. 桌面界面

不习惯命令行时，用桌面界面完成全部常规分析。

```bash
uv run vexel gui
```

也可以启动时直接打开数据：`uv run vexel gui data/demo/droplets.ome.tif`。没有数据时，先用 `uv run vexel synth data/demo` 生成一段合成视频来熟悉界面。

窗口分左右两栏：左侧按 1、2、3 三步操作，右侧显示结果。

### 2.1 第一步：数据

点 **选择文件…** 选视频或 TIFF；图像序列和 OME-Zarr 用 **图像序列文件夹…**。也可以把文件直接拖进窗口。

载入后，文件名下方显示帧数、画面尺寸、数据类型和通道数。只读元数据，大文件也立即显示。多通道数据在第二步里选择各项分析用哪个通道。

### 2.2 第二步：设置

| 项 | 说明 |
| :--- | :--- |
| 像素尺寸 (µm) | 每个像素对应的实际长度 |
| 帧间隔 (ms) | 相邻两帧的采集时间间隔 |
| 流动方向 | 液滴在画面中左右流动选“水平”，上下流动选“竖直” |
| 液滴通道 | 仅多通道数据。在哪个通道里分割液滴，选液滴轮廓最清楚的通道（通常是明场） |
| 包裹计数 | 勾选后统计每个液滴里的细胞数，给出空滴、单包裹、多包裹比例并与泊松分布比较（见 5.5 节） |
| 细胞所在通道 | 仅多通道数据且勾选包裹计数。在哪个通道里数细胞，例如细胞经荧光染色时选荧光通道 |
| 细胞外观 | 勾选包裹计数后出现。明场下细胞通常比液滴内部暗；荧光通道下细胞比液滴内部亮 |
| 荧光 | 勾选后测每个液滴的荧光强度，判为阳性或阴性，给出阳性比例和每滴平均拷贝数（见 5.6 节） |
| 荧光通道 | 仅多通道数据且勾选荧光。测荧光强度用的通道 |

界面里的通道从 1 开始编号；配方和命令行里的通道号从 0 开始，界面的“通道 2”即配方里的 `1`。通道名称（如 brightfield、fluorescence）取自文件元数据，没有时只显示编号。

文件里有标定信息时会自动填入，并提示“已从文件读取标定，请核对”。

**界面上显示什么就用什么。** 字段留空表示不标定，即使文件里有标定信息也不使用：像素尺寸留空时粒径以像素为单位，帧间隔留空时频率以“次/帧”为单位。

**打开视频时务必核对帧间隔。** 视频里记录的通常是回放帧率而不是相机采集帧率，界面会对此给出提示。

### 2.3 高级设置

点 **高级设置** 展开。默认值与所选内置配方一致（见 2.6 节）；鼠标悬停在每一项上有说明。改坏了点 **恢复默认**。

| 项 | 对应配方参数 | 什么时候调 |
| :--- | :--- | :--- |
| 最低对比度 | `seg.min_threshold` | 漏检时调低；空白处出现误检时调高 |
| 平滑 | `seg.smooth_sigma` | 轮廓毛糙时调高；粒径偏大时调低 |
| 最小面积 | `seg.min_area` | 杂质、噪点被当成液滴时调高 |
| 粘连分离距离 | `seg.min_distance` | 一个液滴被切成几块时调高；相邻液滴连成一个时调低 |
| 最大帧间位移 | `track.max_distance_px` | 同一液滴被数成多个时调高；不同液滴被连成一条轨迹时调低 |
| 计数线位置 | `droplets.count_line` | 计数线处有遮挡或流道结构时移开 |
| 细胞直径 | `inclusions.cell_diameter_px` | 仅包裹计数。按实际细胞在图像中的像素直径填写；两个相邻细胞被数成一个时调小 |
| 细胞最低对比度 | `inclusions.min_contrast` | 仅包裹计数。默认“自动”，按每帧估计的噪声设阈值；漏数细胞时填一个较小的值，空液滴被数出细胞时填一个较大的值 |
| 阳性阈值 | `positivity.threshold` | 仅荧光。默认“自动”；填入数值即改为固定阈值（等同于 `positivity.method=fixed`）。自动判读提示只发现一群液滴、或两群重叠时，用阴性和阳性对照确定阈值后填在这里 |

各项的详细含义与调整思路见第 7 节。

### 2.4 第三步：分析

- **快速预览**：只分析前 200 帧，几秒内出结果，用来检查设置是否合适。预览结果放在临时目录，关闭程序后删除，不计算输入文件指纹。
- **开始分析**：分析全部帧。结果保存在数据所在文件夹下的 `<文件名>_vexel/<日期-时间>/`；该文件夹不可写时保存到 `~/Vexel/<文件名>_vexel/<日期-时间>/`。

运行中左下角显示进度和当前步骤，可以随时 **取消**；取消后不留下不完整的结果。

建议流程：先快速预览，看复核图里的轮廓是否贴合；不合适就调高级设置再预览；满意后再开始分析。

### 2.5 结果

右侧从上到下：

- **四个关键数字**：液滴数、平均粒径、粒径 CV、生成频率，每项下方附一行补充（轨迹数、标准差、粒径范围、平均间距）。各项含义见 5.1 节。
- **包裹统计**（勾选包裹计数时）：平均每滴细胞数 λ 及其 95% 置信区间；空滴、单包裹、多包裹的实测比例，下方附泊松预期。各项含义见 5.5 节。
- **荧光判读**（勾选荧光时）：阳性液滴比例及 95% 置信区间；阳性数 / 总数与所用阈值；每滴平均拷贝数及其区间；阴阳分离度 D 与阈值的确定方式。各项含义见 5.6 节。
- **黄色提示条**（只在需要时出现）：预览结果、未标定、没有检测到液滴、质控规则提示等。出现时应先看复核图确认结果。
- **复核图**：均匀抽取的 4 帧，红线为液滴轮廓；包裹计数时黄圈为检出的细胞。勾选荧光时，底图为荧光通道。用 **上一张 / 下一张** 翻看；点图片用系统看图软件打开原图。检查要点见 5.3 节。
- **粒径分布**：每个液滴一个计数，虚线为均值。
- **包裹数分布**（勾选包裹计数时）：每滴含 0、1、2… 个细胞的液滴比例，蓝色为实测，灰色为泊松预期。
- **粒径与荧光**（勾选荧光时）：每个液滴一个点，蓝色为阳性、灰色为阴性，虚线为阈值。阴阳两群应上下分开；阈值附近点多时，判读结果需谨慎对待。

右上角 **打开结果文件夹** 打开保存目录，里面的表格、溯源记录与命令行运行的完全相同（见第 5 节）。

要查看以前的结果：在尚未载入数据时，点右侧的 **打开已有结果…**，选择一次运行的结果文件夹。

### 2.6 界面与命令行的对应关系

默认设置下，界面运行与下面的命令使用同一个配方，`provenance.json` 中的配方哈希相同：

```bash
uv run vexel run <数据> -r droplet_basic --pixel-size <像素尺寸> --frame-interval <帧间隔(秒)>
```

界面按勾选的分析内容选用内置配方：

| 包裹计数 | 荧光 | 配方 |
| :--- | :--- | :--- |
| 否 | 否 | `droplet_basic` |
| 是 | 否 | `droplet_encapsulation` |
| 否 | 是 | `droplet_fluorescence` |
| 是 | 是 | `droplet_fluorescence_encapsulation` |

其他设置对应的命令行参数：细胞外观选“比液滴内部亮”为 `--set inclusions.polarity=bright`；液滴通道、细胞所在通道、荧光通道分别为 `--set input.channel=N`、`--set input.channels.cells=N`、`--set input.channels.fluo=N`（N 从 0 起）。

在高级设置里改过的参数，等同于命令行的 `--set`。实际使用的全部参数都记录在结果文件夹的 `recipe.json` 中。

---

## 3. 五分钟上手（命令行）

没有自己的数据时，先用合成数据走一遍完整流程。

**第一步：生成一段带真值的合成液滴视频。**

```bash
uv run vexel synth data/demo
```

得到 `data/demo/droplets.ome.tif`（400 帧，256×768，已带像素尺寸和帧间隔）和 `data/demo/truth/`（真值）。

**第二步：用内置配方分析。**

```bash
uv run vexel run data/demo/droplets.ome.tif -r droplet_basic -o runs/demo
```

终端会依次打印五个步骤，然后是汇总结果：

```text
droplets.summary
  n_droplets               40
  diameter_mean            60.6
  diameter_std             1.886
  diameter_cv_percent      3.112
  frequency                100.1
  velocity_mean            2e+04
  spacing_mean             199.8
  length_unit              um
  ...
result digest  7a64a303...
output         runs/demo
```

**第三步：与真值比对。**

```bash
uv run vexel validate runs/demo --truth data/demo/truth
```

```text
metric                    measured       truth       error     tolerance  result
n_droplets                      40          40           0             0  PASS
diameter_mean                 60.6       60.25       0.57%            2%  PASS
diameter_cv_percent          3.112       3.099      0.0136             1  PASS
frequency                    100.1       100.1       0.00%            2%  PASS

validation PASSED
```

**第四步：看复核图。** 打开 `runs/demo/review/` 里的 PNG，红色轮廓应贴合每个液滴的外缘。

---

## 4. 分析自己的数据

### 4.1 支持的输入

| 输入 | 说明 |
| :--- | :--- |
| `.tif` / `.tiff` | 多页 TIFF、OME-TIFF、ImageJ TIFF，可含多个通道。会读取文件里的像素尺寸、帧间隔和通道名称 |
| 图像序列目录 | 目录内的 `.tif` `.png` `.jpg` `.bmp`，每个文件一帧，尺寸必须一致，按文件名中的数字自然排序 |
| `.avi` `.mp4` `.mov` `.mkv` `.m4v` | 视频。帧间隔取自视频帧率；按 R、G、B 三个通道读入。彩色相机拍的荧光通常在其中一个颜色通道（如绿色荧光在 G，即通道 1） |
| `.npy` | 2 维按 YX、3 维按 TYX、4 维按 TCYX、5 维按 TCZYX 解释 |
| OME-Zarr 目录 | 例如 `vexel convert` 的输出 |

各通道分别存成不同文件（如明场一个视频、荧光另一个视频）的数据目前不支持，需要先合并成一个多通道文件。

### 4.2 先看元数据

```bash
uv run vexel info my_video.avi
```

```text
source   my_video.avi
reader   video
shape    {'T': 5000, 'C': 3, 'Z': 1, 'Y': 512, 'X': 1024}  uint8  (7864.3 MB)
spacing  y=1 x=1 z=1 px;  t=0.04 s
note     no pixel size found: results will be in pixels (pass --pixel-size to calibrate)
```

这一步不读像素，大文件也是立刻返回。重点看两处：帧数和尺寸是否符合预期；`spacing` 一行是否已有物理单位。

**注意：** 高速相机导出的视频，文件里的帧率通常是回放帧率（例如 25 fps），不是采集帧率。这种情况必须用 `--frame-interval` 指定真实的采集帧间隔，否则生成频率和速度都是错的。

### 4.3 标定

没有像素尺寸时，粒径以像素为单位输出，`summary.json` 中 `calibrated` 为 `false`。提供标定有三种方式，优先级从高到低：

1. 命令行参数：

   ```bash
   uv run vexel run my_video.avi -r droplet_basic --pixel-size 2.0 --frame-interval 0.001
   ```

   `--pixel-size` 的单位默认是 um，可用 `--unit` 改；`--frame-interval` 的单位是秒。

2. 旁挂文件。单个文件用 `<文件名>.vexel.json`（如 `my_video.avi.vexel.json`），图像序列目录用目录内的 `vexel.json`：

   ```json
   {"pixel_size": 2.0, "space_unit": "um", "frame_interval": 0.001}
   ```

3. 文件自带的元数据（OME-TIFF、ImageJ TIFF、视频帧率）。

### 4.4 运行

```bash
uv run vexel run my_video.avi -r droplet_basic --pixel-size 2.0 --frame-interval 0.001 -o runs/exp01
```

常用选项：

| 选项 | 作用 |
| :--- | :--- |
| `-r, --recipe` | 配方文件路径，或内置配方名 |
| `-o, --out` | 输出目录。省略时为 `runs/<输入名>-<配方名>-<时间戳>` |
| `--set 步骤.参数=值` | 临时覆盖配方参数，可重复使用 |
| `--max-frames N` | 只处理前 N 帧。调参时用它快速试 |
| `--keep-intermediates` | 保留每一步的中间数组（`scratch/` 下的 Zarr） |
| `--no-input-hash` | 不计算输入文件哈希。文件很大时可省时间，但溯源记录里就没有输入指纹 |
| `--overwrite` | 覆盖输出目录里的上一次运行 |
| `--workers N` | 逐帧算子使用的线程数，默认不超过 8 |

输出目录非空时默认拒绝运行。加了 `--overwrite` 也只会覆盖 Vexel 自己的运行目录，不会删除其他内容。

---

## 5. 读懂结果

```text
runs/exp01/
├── recipe.json        冻结的配方，所有默认参数已展开
├── summary.json       汇总统计
├── tables/
│   ├── measure.table.csv        每帧每个检测目标一行
│   ├── track.table.csv          同上，多一列 track_id
│   └── droplets.droplets.csv    每个液滴一行（已去重）
├── review/*.png       分割轮廓叠加图
└── provenance.json    溯源记录
```

### 5.1 summary.json

| 字段 | 含义 |
| :--- | :--- |
| `n_droplets` | 液滴数。同一液滴跨多帧只计一次，以轨迹越过计数线为准 |
| `diameter_mean` / `_std` / `_median` / `_min` / `_max` | 粒径统计，单位见 `length_unit` |
| `diameter_cv_percent` | 粒径变异系数（标准差 ÷ 均值 × 100，样本标准差） |
| `frequency` | 生成频率 =（液滴数 − 1）÷（首末液滴越线的时间差），单位见 `frequency_unit` |
| `velocity_mean` | 沿流动方向的平均速度，单位见 `velocity_unit` |
| `spacing_mean` | 相邻液滴的平均中心间距 = 速度 ÷ 频率 |
| `count_line_px` | 计数线的像素位置 |
| `n_tracks_total` | 追踪到的轨迹总数 |
| `n_tracks_not_crossing` | 没有越过计数线的轨迹数。视频开头已过线、结尾未到线的液滴属于此类，属正常现象 |
| `n_tracks_too_short` | 因帧数太少被忽略的轨迹数 |
| `n_tracks_unsized` | 越线但没有一帧完整出现在画面内、因而无法测径的轨迹数 |
| `calibrated` | 是否有像素尺寸。`false` 时长度单位为 px |

粒径是**等效面积直径**：与液滴分割区域面积相等的圆的直径。每个液滴取它完整出现在画面内的各帧的中位数。

**这是 2D 投影直径。** 被流道压扁的液滴，体积换算尚未实现。

### 5.2 表格

`droplets.droplets.csv`（每个液滴一行）：

| 列 | 含义 |
| :--- | :--- |
| `track_id` | 轨迹编号 |
| `n_frames`、`first_frame`、`last_frame` | 被检测到的帧数与首末帧 |
| `crossing_frame`、`crossing_time` | 越过计数线的时刻（帧内插值） |
| `diameter`、`diameter_px` | 粒径（物理单位 / 像素） |
| `circularity` | 圆度 4πA/P²，1 为正圆 |
| `velocity` | 沿流动方向的速度 |
| `cross_position` | 垂直于流动方向的位置 |
| `n_inclusions` | 液滴里的细胞数（仅 `droplet_encapsulation`）。取该液滴各完整帧计数的众数；无法计数时为 −1 |
| `inclusion_agreement` | 与众数一致的帧所占比例（仅 `droplet_encapsulation`）。低于 0.8 说明各帧计数不稳定，应在复核图里核对 |

使用荧光配方时，每个液滴还有：

| 列 | 含义 |
| :--- | :--- |
| `fluo_mean` | 液滴内部（避开边缘 `edge_margin_px` 个像素）扣除背景后的平均荧光，取各完整帧的中位数。阴阳判读默认用这一列 |
| `fluo_median`、`fluo_p95` | 液滴内部扣除背景后的中位数与第 95 百分位 |
| `fluo_integrated` | 整个液滴范围内扣除背景后的荧光总和 |
| `positive` | 是否判为阳性（在 `positivity.droplets.csv` 中） |

荧光配方的每液滴结果以 `positivity.droplets.csv` 为准：它是 `droplets.droplets.csv` 加上 `positive` 列。

`encapsulation.distribution.csv`（仅包裹计数配方，每个细胞数 k 一行）：`k`、`observed`（含 k 个细胞的液滴数）、`observed_fraction`、`poisson_fraction`（泊松概率）、`poisson_expected`（泊松预期液滴数）。

`measure.table.csv` 与 `track.table.csv`（每帧每个目标一行）：带 `_px` 后缀的列是像素单位，不带的是物理单位。`touches_border` 为真表示目标碰到画面边缘，测径时不采用。

### 5.3 复核图

`review/` 下默认有 4 张，均匀取自整段视频。**每次分析新类型的数据都应该看这几张图。** 需要检查：

- 轮廓是否贴合液滴外缘，没有明显偏内或偏外；
- 是否有漏检的液滴；
- 相邻液滴是否被连成一个，或一个液滴被切成几块；
- 流道壁、杂质、气泡是否被误当成液滴。

包裹计数时，图中检出的细胞以黄圈标出。另需检查：

- 每个细胞是否都有黄圈，空液滴里是否出现多余的黄圈；
- 紧挨着的两个细胞是否被各自圈出；
- 贴近液滴边缘的细胞是否漏圈（见第 11 节的限制）。

### 5.4 溯源记录

`provenance.json` 记录：运行状态、配方哈希、输入文件的 SHA-256 与标定、Python 与各依赖的版本、每一步的耗时与质控指标、各输出文件的哈希，以及总的 `result_digest`。

**同一配方加同一数据，`result_digest` 必须相同。** 两次运行结果是否一致，比较这个值即可。

### 5.5 包裹统计

使用 `droplet_encapsulation` 配方（或在界面中勾选包裹计数）时，`summary.json` 多出 `encapsulation.summary`。统计对象与 5.1 节相同：越过计数线、每个只计一次的液滴。

| 字段 | 含义 |
| :--- | :--- |
| `n_droplets` | 参与包裹统计的液滴数 |
| `n_inclusions_total` | 细胞总数 |
| `lambda` | 平均每滴细胞数 λ（泊松参数的最大似然估计） |
| `lambda_ci_low` / `_high` | λ 的 95% 精确置信区间（按细胞总数的泊松区间） |
| `frac_empty` / `frac_single` / `frac_multi` | 实测的空滴（0 个）、单包裹（1 个）、多包裹（≥2 个）比例 |
| `poisson_empty` / `_single` / `_multi` | 同一 λ 下泊松分布的预期比例：e^−λ、λe^−λ、其余 |
| `single_among_occupied` | 有细胞的液滴中单包裹所占比例，即单细胞包裹的纯度 |
| `variance`、`dispersion_index` | 计数的样本方差，离散指数 D = 方差 ÷ λ。泊松分布时 D ≈ 1 |
| `chi2`、`chi2_dof`、`chi2_p` | 实测分布对泊松分布的卡方拟合优度检验。预期数不足 5 的相邻类别会合并；类别太少时为空 |
| `n_low_agreement` | 各帧计数一致率低于 0.8 的液滴数 |
| `n_excluded` | 无法计数（液滴内部区域太小）而未计入的液滴数 |

**怎么读：**

- 细胞随机上样时，包裹数应服从泊松分布。`chi2_p` 小于 0.05 表示实测分布与泊松分布差异显著。
- D 明显大于 1：细胞聚集、成团或上样浓度随时间漂移。D 明显小于 1：排布比随机更规则（例如惯性有序包裹）。
- `n_low_agreement` 不为 0 时，这些液滴的计数可能不准，先在复核图中核对，再看统计结论。

**计数方法：** 每一帧里，每个液滴先向内收缩 `edge_margin_px` 个像素以避开深色轮廓；用液滴内部的中位灰度作为该液滴的背景；平滑后比背景暗（或亮）超过阈值的局部极值记为一个细胞。阈值默认自动：从液滴内部估计噪声水平，取平滑后噪声的 6 倍（`noise_factor`），因此 8 位和 16 位图像都适用。每个液滴最终取它完整出现在画面内的各帧计数的众数。

### 5.6 荧光阴阳判读

使用 `droplet_fluorescence` 或 `droplet_fluorescence_encapsulation`（或在界面中勾选荧光）时，`summary.json` 多出 `positivity.summary`。统计对象同样是越过计数线、每个只计一次的液滴。

| 字段 | 含义 |
| :--- | :--- |
| `n_droplets`、`n_positive` | 参与判读的液滴数与阳性数 |
| `positive_fraction` | 阳性比例 p |
| `positive_fraction_ci_low` / `_high` | p 的 95% Wilson 置信区间 |
| `copies_per_droplet` | 每滴平均拷贝数 λ = −ln(1 − p)，即数字 PCR 等数字检测的泊松校正。全部阳性时无法估计，为空 |
| `copies_ci_low` / `_high` | 由 p 的置信区间换算得到的 λ 区间 |
| `threshold` | 阳性阈值（`fluo_mean` 高于它为阳性） |
| `threshold_method` | 阈值的确定方式：`mixture` 自动分群、`outlier` 离群值规则、`fixed` 固定值、`otsu` 大津法 |
| `populations` | 自动判读认定的群数：2 为阴阳两群，1 为只有一群 |
| `separation_d` | 阴阳两群的分离度（Ashman's D）。D ≥ 4 时两群分得很开；2–4 有重叠；小于 2 基本分不开 |
| `dip_ratio` | 两群之间密度低谷的深度（低谷高度 ÷ 较矮的峰高）。越小越清楚 |
| `negative_mean`、`positive_mean` | 两群的平均荧光 |
| `saturated` | 是否全部为阳性 |

**自动阈值怎么定（`method: auto`）：** 只有下面三项同时成立，才认定存在阴阳两群，并取两群概率相等处为阈值：

1. 两成分高斯混合模型的 BIC 比单成分低 10 以上（`min_bic_gain`）；
2. 两群之间的密度确有低谷，`dip_ratio` 小于 0.6（`max_dip_ratio`）；
3. 分离度 D 不小于 3（`min_separation`）。

否则视为只有一群，用离群值规则判读：高于中位数加 6 倍稳健标准差（`outlier_k`）的液滴才判为阳性。这样全阴性样本（如无模板对照）不会被硬拆出假阳性。代价是：如果所有液滴都是阳性，或阴阳两群严重重叠，自动判读会全部判为阴性，结果页和 `populations = 1` 会提示。此时应该用阴性和阳性对照确定阈值，改用固定阈值。

**在合成数据上的表现：**

| 条件 | 结果 |
| :--- | :--- |
| 阴阳分得开（阳性约为阴性的 4 倍以上，液滴间差异 10%），阳性比例 1%–97% | 每个液滴都判对 |
| 同上，荧光噪声加大到 10 个灰度级 | 每个液滴都判对 |
| 阳性的液滴间差异 30%（两群边缘重叠） | 200 个液滴错 2 个（1%） |
| 全阴性，液滴间差异 10% 或 30% | 无假阳性 |
| 阳性只比阴性亮 1.6 倍且差异 20%（两群严重重叠） | 自动判读认定只有一群，全部判阴性并给出提示；需要固定阈值 |

另用随机数检验了自动判据：对正态、伽马、对数正态等单峰分布，200 个液滴以上时误拆成两群的比例不超过 2%；分得开的两群，200 个液滴以上检出率 95%–100%。液滴少于约 40 个时，自动判读更倾向于“只有一群”。

---

## 6. 配方

配方是一个 JSON 文件，描述分析的每一步。它在运行前被校验并冻结，运行中不会改变。

### 6.1 查看内置配方

```bash
uv run vexel recipes
uv run vexel recipes --show droplet_basic
```

内置四个配方：

| 配方 | 内容 |
| :--- | :--- |
| `droplet_basic` | 液滴粒径、计数与生成频率 |
| `droplet_encapsulation` | 再加包裹计数与泊松统计 |
| `droplet_fluorescence` | 再加荧光强度与阴阳判读（不含包裹计数） |
| `droplet_fluorescence_encapsulation` | 以上全部 |

四个配方中同名的步骤参数完全相同，只是组合不同。

`droplet_basic` 的五个步骤：

| 步骤 id | 算子 | 作用 |
| :--- | :--- | :--- |
| `bg` | `preprocess.static_background` | 用时间中值估计静态背景（流道壁、不均匀照明）并扣除 |
| `seg` | `segment.threshold_watershed` | 阈值分割，填充环形轮廓，分开粘连目标 |
| `measure` | `measure.regionprops` | 测量每个目标的位置、面积、粒径、圆度 |
| `track` | `track.link` | 把逐帧检测连成轨迹 |
| `droplets` | `droplet.summary` | 按计数线去重，输出每液滴表和汇总统计 |

`droplet_encapsulation` 多两步，并让 `droplets` 步骤接收细胞计数：

| 步骤 id | 算子 | 作用 |
| :--- | :--- | :--- |
| `inclusions` | `droplet.inclusions` | 逐帧数每个液滴里的细胞，并输出细胞标记供复核图使用 |
| `encapsulation` | `droplet.poisson` | 汇总包裹数分布，与泊松分布比较 |

细胞在 `input.channels.cells` 通道里数，默认与分割通道相同（0）。

`droplet_fluorescence` 在 `droplet_basic` 基础上多两步：

| 步骤 id | 算子 | 作用 |
| :--- | :--- | :--- |
| `intensity` | `droplet.intensity` | 在荧光通道（`input.channels.fluo`，默认 1）逐帧测每个液滴的荧光 |
| `positivity` | `droplet.positivity` | 判读每个液滴的阴阳性，汇总阳性比例与每滴拷贝数 |

### 6.2 配方结构

```json
{
  "schema_version": 1,
  "name": "my_recipe",
  "description": "说明文字",
  "seed": 0,
  "input": {"channel": 0, "z": 0},
  "steps": [
    {
      "id": "seg",
      "op": "segment.threshold_watershed",
      "inputs": {"image": "source"},
      "params": {"min_area": 100},
      "qc": [{"metric": "n_objects.max", "min": 1, "on_fail": "warn"}]
    }
  ],
  "review": {"image": "source", "labels": "seg.labels", "n_frames": 4}
}
```

- `input`：分析哪个通道、哪个 z 平面。`channel` 是 `source`；`channels` 给其他通道起名，例如 `"channels": {"fluo": 1}` 后，步骤里用 `"source.fluo"` 引用通道 1。这样可以在一个通道分割、在另一个通道测量。通道号从 0 起。
- `steps`：按顺序执行。`id` 只能用小写字母、数字、下划线，且以字母开头。
- `inputs`：每个输入接到哪里。`"source"` 是原始图像，其余写成 `"步骤id.输出名"`，只能引用前面的步骤。
- `params`：省略的参数取默认值。写错参数名会报错，不会被悄悄忽略。
- `qc`：可选的质控规则，见 6.4。
- `review`：可选，指定复核图用哪个图像（`image`）和哪个分割结果（`labels`，红线）；`marks` 可再指定一组标记（黄圈），如检出的细胞。

运行前会检查：算子是否存在、接线的数据类型是否匹配（图像 / 标签 / 表格 / 汇总）、参数是否合法。

### 6.3 修改参数

临时修改用 `--set`：

```bash
uv run vexel run my_video.avi -r droplet_basic --set seg.min_area=200 --set droplets.count_line=0.4
```

需要长期使用时，导出内置配方再编辑：

```bash
uv run vexel recipes --show droplet_basic > my_recipe.json
uv run vexel run my_video.avi -r my_recipe.json
```

通道也用 `--set` 改，不必编辑配方：

```bash
uv run vexel run my_video.avi -r droplet_fluorescence --set input.channels.fluo=1
uv run vexel run my_stack.tif -r droplet_fluorescence_encapsulation --set input.channels.cells=1 --set inclusions.polarity=bright
```

实际使用的全部参数（含默认值）都写在输出目录的 `recipe.json` 里。

### 6.4 质控规则

每个算子会输出质控指标，记录在 `provenance.json` 的 `steps[].qc` 中。可以在配方里给指标设上下限：

```json
"qc": [{"metric": "foreground_fraction.max", "max": 0.6, "on_fail": "warn"}]
```

- 逐帧算子的指标按时间聚合，写成 `指标名.mean`、`指标名.min`、`指标名.max`。
- 跨帧算子的指标直接用指标名。
- `on_fail` 为 `warn` 时只提示并继续；为 `stop` 时中止运行，已完成的结果和溯源记录仍会保存，命令返回码为 3。

各算子的指标：

| 算子 | 指标 |
| :--- | :--- |
| `preprocess.static_background` | `frames_sampled`、`background_mean`、`residual_mean` |
| `segment.threshold_watershed` | `threshold`、`n_objects`、`foreground_fraction`（各带 `.mean/.min/.max`） |
| `measure.regionprops` | `n_objects`（带 `.mean/.min/.max`） |
| `track.link` | `n_tracks`、`mean_track_length` |
| `droplet.summary` | `n_droplets`、`diameter_cv_percent`、`circularity_median` |
| `droplet.inclusions` | `n_inclusions`、`contrast_threshold`（各带 `.mean/.min/.max`） |
| `droplet.poisson` | `n_droplets`、`dispersion_index`、`chi2_p`、`low_agreement_fraction` |
| `droplet.intensity` | `background`、`droplet_mean`（各带 `.mean/.min/.max`） |
| `droplet.positivity` | `n_droplets`、`positive_fraction`、`separation_d` |
| `stats.describe` | `n` |

`droplet_encapsulation` 默认对 `low_agreement_fraction` 设了上限 0.1：计数不稳定的液滴超过 10% 时给出警告。荧光配方默认要求 `separation_d` 不小于 2，否则警告阴阳两群分不开。

---

## 7. 调参指南

先用 `--max-frames 200` 快速试，看复核图，再跑全量。

| 现象 | 调整 |
| :--- | :--- |
| 没有液滴的帧里出现大片误检 | 调高 `seg.min_threshold`（自动阈值的下限，单位是扣背景后的灰度差） |
| 液滴漏检 | 调低 `seg.min_threshold`；或调低 `seg.min_area` |
| 轮廓比液滴偏大 | 调低 `seg.smooth_sigma`（0 为不平滑） |
| 轮廓毛糙、噪点被当成目标 | 调高 `seg.smooth_sigma` 或 `seg.opening_radius`；调高 `seg.min_area` |
| 液滴轮廓不闭合，只分割出一段弧 | 调高 `seg.smooth_sigma`；或调低 `seg.min_threshold` |
| 相邻液滴被连成一个 | 确认 `seg.split_touching` 为 `true`；调低 `seg.min_distance` |
| 一个液滴被切成几块 | 调高 `seg.min_distance`（应略小于液滴半径的像素数） |
| 同一液滴被数成多个（`n_tracks_total` 远大于实际） | 调高 `track.max_distance_px`；它应大于液滴每帧的位移 |
| 不同液滴被连成一条轨迹 | 调低 `track.max_distance_px`；它应小于相邻液滴间距的一半 |
| 每帧位移大于液滴间距的一半 | 设置 `track.expected_shift_px` 为每帧的预期位移，例如 `--set 'track.expected_shift_px=[0,25]'` 表示每帧沿 x 移动 25 像素 |
| 液滴沿竖直方向流动 | `--set droplets.flow_axis=y` |
| 计数线处有遮挡或流道结构 | 调整 `droplets.count_line`（0 到 1 之间，沿流动方向的比例位置） |
| 背景扣除后液滴残影明显 | 液滴过密，同一像素超过一半时间被液滴覆盖，时间中值不再是背景。调高 `bg.max_frames`；仍不行则该方法不适用 |
| 细胞漏数（复核图中细胞没有黄圈） | 设定 `inclusions.min_contrast` 为一个较小的固定值（参考 `provenance.json` 中 `contrast_threshold.mean` 的自动值，从它的一半试起）；或调低 `inclusions.noise_factor` |
| 空液滴里数出细胞（噪点、杂质被圈出） | 设定 `inclusions.min_contrast` 为一个较大的固定值；或调高 `inclusions.noise_factor` |
| 两个紧挨的细胞被数成一个 | 调小 `inclusions.cell_diameter_px`；或设定 `inclusions.min_separation_px` |
| 一个细胞被数成两个 | 调大 `inclusions.cell_diameter_px` |
| 贴近液滴边缘的细胞漏数 | 调小 `inclusions.edge_margin_px`；但过小会把液滴的深色轮廓误数成细胞 |
| 细胞比液滴内部亮（荧光） | `--set inclusions.polarity=bright` |
| 荧光提示“只发现一群液滴”，但样本应含阳性 | 用阳性和阴性对照确定阈值，`--set positivity.threshold=<值> --set positivity.method=fixed`（界面：高级设置 › 阳性阈值） |
| 全阴性对照里出现少量阳性 | 多为荧光离群液滴（气泡、杂质）。核对复核图；必要时用固定阈值 |
| 荧光阴阳两群重叠（D 小于 4） | 检查荧光通道是否选对、曝光是否合适；必要时调大 `intensity.edge_margin_px` 避开液滴边缘 |
| 荧光背景不均匀 | `intensity.background` 取液滴外像素的中位数作背景；背景很不均匀时可设为 `none`，并用固定阈值 |

**关于粒径偏差：** 阈值法测得的边界位置受平滑量和阈值影响。在合成数据上，`smooth_sigma` 为 1.0 时粒径均值偏大约 2.75%，为 0.5 时约 0.6%。对绝对粒径有要求时，应该用已知尺寸的样品或人工标注做一次校准（见第 9 节），不要直接相信默认参数。

---

## 8. 算子参考

完整且最新的列表以命令输出为准：

```bash
uv run vexel ops
```

**作用范围**决定算子一次需要看到多少数据：`frame` 逐帧独立处理；`window` 需要时间轴上的多帧。

| 算子 | 范围 | 输入 → 输出 | 说明 |
| :--- | :--- | :--- | :--- |
| `preprocess.static_background` | window | 图像 → 图像 | 时间中值背景扣除。`mode`：`absdiff`（默认，亮暗目标都变亮）、`subtract`、`divide` |
| `preprocess.background_correction` | frame | 图像 → 图像 | 单帧背景校正：顶帽或大尺度高斯。适用于没有稳定静态背景的数据 |
| `preprocess.gaussian` | frame | 图像 → 图像 | 高斯平滑 |
| `preprocess.median` | frame | 图像 → 图像 | 中值去噪 |
| `segment.threshold_watershed` | frame | 图像 → 标签 | 阈值（`otsu` / `triangle` / `fixed`）+ 填洞 + 开运算 + 分水岭分离 |
| `measure.regionprops` | frame | 标签（+ 可选强度图像）→ 表格 | 位置、面积、等效直径、长短轴、偏心率、圆度、平均强度 |
| `track.link` | window | 表格 → 表格 | 线性指派追踪，增加 `track_id` 列 |
| `stats.describe` | window | 表格 → 汇总 | 某一列的 n、均值、标准差、CV、中位数、范围 |
| `droplet.summary` | window | 表格（+ 可选细胞计数表、荧光表）→ 表格 + 汇总 | 计数线去重、粒径统计、生成频率、速度、间距；有细胞计数表时为每个液滴给出细胞数，有荧光表时给出荧光 |
| `droplet.inclusions` | frame | 图像 + 标签 → 表格 + 标签 | 数每个液滴内的细胞（暗点或亮点），输出每帧计数和细胞标记 |
| `droplet.poisson` | window | 表格 → 表格 + 汇总 | 包裹数分布、λ、空滴/单包裹/多包裹比例、泊松预期、离散指数、卡方检验 |
| `droplet.intensity` | frame | 图像 + 标签 → 表格 | 每个液滴扣除背景后的平均、中位、第 95 百分位与积分荧光 |
| `droplet.positivity` | window | 表格 → 表格 + 汇总 | 阴阳判读（自动分群 / 离群值规则 / 大津法 / 固定阈值）、阳性比例、每滴拷贝数 |

---

## 9. 用真值验证

`vexel validate` 把一次运行的汇总结果与真值比对，所有检查项都在容差内才算通过。基本检查四项：液滴数、粒径均值、粒径 CV、生成频率。运行做了包裹计数且真值带细胞数时，另加五项：逐液滴细胞数错误率、λ、空滴比例、单包裹比例、多包裹比例。运行做了荧光判读且真值带阴阳性时，再加两项：逐液滴判错率、阳性比例。

```bash
uv run vexel validate runs/exp01 --truth path/to/truth
```

| 选项 | 默认 | 含义 |
| :--- | :--- | :--- |
| `--count-tol` | 0 | 液滴数允许的绝对误差 |
| `--diameter-tol` | 0.02 | 粒径均值允许的相对误差 |
| `--cv-tol` | 1.0 | CV 允许的绝对误差，百分点 |
| `--frequency-tol` | 0.02 | 生成频率允许的相对误差 |
| `--mismatch-tol` | 0.05 | 细胞数数错的液滴允许占的比例 |
| `--lambda-tol` | 0.05 | λ 允许的相对误差 |
| `--fraction-tol` | 0.03 | 空滴、单包裹、多包裹比例允许的绝对误差 |
| `--positive-mismatch-tol` | 0.02 | 阴阳判错的液滴允许占的比例 |
| `--positive-fraction-tol` | 0.03 | 阳性比例允许的绝对误差 |

**默认容差是占位值**，尚未与实际使用方约定。

### 给真实数据做标注

真值目录需要三个文件，格式与合成数据的 `truth/` 相同：

`droplets.csv`，每个液滴一行：

```text
droplet_id,diameter_um
0,58.2
1,60.1
```

要验证包裹计数，再加一列 `n_cells`（该液滴里的细胞数）；要验证荧光判读，再加一列 `positive`（`True` / `False`）。例如：

```text
droplet_id,diameter_um,n_cells
0,58.2,1
1,60.1,0
```

`positions.csv`，每个液滴在每一帧的中心 x 坐标（像素）：

```text
frame,droplet_id,x_px
0,0,120.5
1,0,131.0
```

`config.json`：

```json
{"frame_interval_s": 0.001}
```

比对时会用与分析相同的计数线和帧范围，从 `positions.csv` 算出哪些液滴越线，再比较这些液滴的数量、粒径均值、CV 和频率。细胞数按液滴逐个比对：测得的液滴与真值液滴按越线时刻配对（相差不超过 1 帧），再看细胞数是否相同。

限制：运行必须已标定为 um；目前只支持沿 x 方向流动的数据。

### 生成不同条件的合成数据

```bash
uv run vexel synth data/noisy --frames 1000 --noise 8 --diameter 40 --cv 5 --frequency 180 --seed 3
```

| 选项 | 默认 | 含义 |
| :--- | :--- | :--- |
| `--frames` | 400 | 帧数 |
| `--diameter` | 60 | 平均粒径，um |
| `--cv` | 3 | 粒径 CV，% |
| `--frequency` | 100 | 生成频率，Hz |
| `--noise` | 3 | 相机噪声标准差，灰度级 |
| `--cells` | 0 | 平均每滴细胞数 λ，按泊松分布放入细胞；0 为不放细胞 |
| `--fluo` | 不加 | 加一个荧光通道，数值为每滴平均拷贝数；含 1 个及以上拷贝的液滴为阳性 |
| `--cells-in` | brightfield | 细胞出现在哪个通道：`brightfield`（明场暗点）或 `fluorescence`（荧光亮点，需要 `--fluo`） |
| `--seed` | 0 | 随机种子 |

合成视频固定为 256×768、2 um/像素、1 ms/帧、流速 20 mm/s。加 `--fluo` 时为两通道 OME-TIFF（通道 0 明场、通道 1 荧光）：油相背景 8，阴性液滴 25，阳性液滴 120，液滴间差异 10%，噪声 3 个灰度级；明场通道与不加荧光时逐像素相同。细胞为直径 10 µm（5 像素）的深色圆点，比液滴内部暗 50 个灰度级，随液滴平移，彼此至少相隔 1 个像素。它是理想化的：液滴为正圆、匀速、不粘连、轮廓清晰；细胞形状一致、不转动、不贴边。**在合成数据上通过不代表在真实数据上准确。**

在合成数据上的包裹计数准确度（每组 40 个液滴，共 14 组）：λ 从 0.3 到 3、噪声 3 到 12 个灰度级、细胞对比度低至 15 个灰度级时，逐液滴细胞数全部正确；只有细胞对比度 15 且噪声 6 时（信噪比约 2.5），40 个液滴中有 3 个数错，且这 3 个的帧间一致率都低于 0.4。

---

## 10. 其他命令

**转换为 OME-Zarr 金字塔**（便于大文件的后续读取和在其他软件中查看）：

```bash
uv run vexel convert my_video.avi my_video.zarr --pixel-size 2.0 --frame-interval 0.001
```

**依赖许可证清单**（`CHECK` 标记 GPL 或非商用条款，`UNKNOWN` 表示包未声明）：

```bash
uv run vexel licenses
```

**命令返回码：**

| 返回码 | 含义 |
| :--- | :--- |
| 0 | 成功 |
| 1 | `validate` 未通过 |
| 2 | 输入、配方或参数错误 |
| 3 | `run` 被质控规则中止 |

---

## 11. 当前版本的限制

- 桌面界面只覆盖单个文件的分析与结果查看；HTML 报告、批量处理、目录监视、实时回放、在图上手工纠错都还没有。
- 桌面界面只能使用两个内置配方 `droplet_basic` 与 `droplet_encapsulation`；自定义配方需用命令行。
- 只处理 2D 时序。可以同时读多个通道，但只分析一个 z 平面；3D 算子尚不执行。
- 分割只有阈值加分水岭一种方法。对比度低、轮廓不闭合、严重粘连的液滴效果有限。
- 粒径是 2D 投影的等效直径，没有压扁液滴的体积换算。
- 各通道分别存成不同文件的数据不支持，需要先合并成一个多通道文件。
- 荧光阴阳判读只有一个荧光通道；多色（如双色探针）判读尚未实现。
- 没有体积换算，因此只给出每滴拷贝数，不给出浓度（拷贝/µL）。
- 自动阴阳判读无法区分“全部阴性”和“全部阳性”；只有一群时一律按阴性处理并提示。
- 包裹计数要求细胞在液滴内部可分辨。紧贴液滴边缘（`edge_margin_px` 以内）的细胞会漏数；成团、重叠的细胞会少数；液滴内部亮度很不均匀（如明场下的透镜效应）时，自动阈值偏保守。
- 准确度只在合成数据上验证过，尚未在真实液滴数据上验证。
- 视频按 RGB 三通道解码，灰度视频会多占内存。
- 每个处理块最多占用约 2 GB 内存（系统内存的 10%，上限 2 GB）。
- 只在 macOS（Apple Silicon）上运行过，Windows 与 Linux 未测试。

---

## 12. 常见问题

**`error: ... is not empty; choose another output directory or pass overwrite`**
输出目录已有内容。换一个 `-o`，或加 `--overwrite`。

**`error: no reader for ...`**
文件扩展名不在支持列表内（见 4.1）。

**`error: step 'seg': invalid parameters for ...`**
参数名或取值不对。用 `uv run vexel ops` 查看该算子的参数。

**`n_droplets` 为 0 或很少**
先看复核图。没有轮廓：分割没检出，按第 7 节调 `seg` 的参数。有轮廓：看 `summary.json`，`n_tracks_not_crossing` 很大说明轨迹没越过计数线，检查 `droplets.flow_axis` 和 `droplets.count_line`；`n_tracks_total` 远大于实际液滴数说明追踪断了，调 `track.max_distance_px`。

**`frequency` 明显不对**
检查帧间隔。`uv run vexel info` 显示的 `t=` 是否是真实的采集帧间隔（见 4.2 的注意事项）。

**结果单位是 px**
没有像素尺寸，见 4.3。

**`run` 提示 QC warning**
某项质控指标超出配方设定的范围。运行仍会完成，但结果需要对照复核图确认。

**大文件运行很慢才开始**
开始前的元数据读取是即时的；结束时的 `hashing input` 需要完整读一遍输入文件。可用 `--no-input-hash` 跳过。
