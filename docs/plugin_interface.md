# 模型插件接口（设计说明）

日期：2026-11-14。状态：**接口已定、已实现并有测试；还没有正式插件**。SAM、深度学习分割、本地训练会在后续的平台阶段按此接口接入；微流控若需要“数细胞的小模型”，也走这里。代码：`src/vexel/plugins.py`；测试：`tests/test_plugins.py`。

## 为什么现在定

现有的算子注册表、配方、流式执行器、溯源记录都假定算子是确定性的经典算法。训练好的模型多了三件事：**权重文件**（会变、要校验）、**训练数据的许可证**（决定结果能否商用）、**输入假设**（像素尺寸、通道、归一化）。先把这三件事的位置定下来，以后接 SAM 或深度模型时只是“多装一个插件”，不必改引擎、配方格式和已有结果。

## 插件是什么

一个普通的 Python 包，在 `pyproject.toml` 里声明入口点：

```toml
[project.entry-points."vexel.plugins"]
droplet_cells = "vexel_droplet_cells:register"
```

Vexel 第一次查找算子时调用每个插件的 `register()`。它做两件事：

```python
from vexel.ops.registry import Kind, Params, Scope, register_op, OpResult
from vexel.plugins import Dataset, ModelCard, load_model, register_model

CARD = ModelCard(
    id="droplet-cells", version="1.0", task="count.inclusions", framework="onnx",
    license="Apache-2.0", file="droplet_cells.onnx", sha256="…",
    url="https://…/droplet_cells-1.0.onnx",
    training_data=(Dataset("Gardner et al. droplet_detection", "待核实", "https://github.com/karl-gardner/droplet_detection"),),
    input={"channels": 1, "pixel_size_um": 2.0, "normalise": "per-droplet z-score"},
    citation="…",
)

class Params_(Params):
    model: str = "droplet-cells@1.0"

def register():
    register_model(CARD, loader=lambda card: OnnxCounter(card))   # loader returns an object with .card and .predict(...)

    @register_op("droplet.inclusions_model", scope=Scope.FRAME,
                 inputs={"image": Kind.IMAGE, "labels": Kind.LABELS},
                 outputs={"table": Kind.TABLE, "labels": Kind.LABELS}, params=Params_)
    def inclusions_model(params, ctx, *, image, labels):
        model = load_model(params.model)        # verified, loaded once per process
        ...
        return OpResult({"table": table, "labels": marks}, qc={...})
```

插件加载失败（缺依赖、代码出错）只在 `vexel plugins` 里报告，**不影响 Vexel 启动和其他算子**。

## 三条规则

1. **配方里写死模型版本。** 用到模型的算子参数里有 `model` 字段，值为 `id@version`，不允许省略版本。执行器把每个用到模型的步骤的模型卡（编号、任务、框架、权重 SHA-256、许可证、训练数据、能否商用）写进 `provenance.json` 的该步骤；单次报告和批量报告的“方法”段落自动写出模型与训练数据。
2. **权重要校验，不偷偷下载。** 权重文件放在 `~/.vexel/models/<id>/<version>/<file>`（`VEXEL_HOME` 可改），或模型卡里写绝对路径。`load_model` 校验 SHA-256，不符就拒绝运行。缺文件时报错信息和 `vexel plugins` 都写明应放在哪里、从哪里下载。
3. **声明训练数据。** 模型卡列出训练数据集及其许可证。权重或任何一个训练集是非商用（NC、research only 等）时，`commercial_use` 为 `False`；许可证写“待核实”或空时为 `None`。`vexel plugins` 把这一点列出来，对应路线图第十二节“记录每个模型用了哪些数据集训练”。

## 任务：输入输出的约定

每个模型卡声明一种任务；执行器在运行时检查使用它的算子是否有该任务要求的输入和输出。约定的目的是**模型算子可以原样替换配方里的经典算子，后面的步骤不用改**：

| 任务 | 输入 | 输出 | 可替换的现有算子 |
| :--- | :--- | :--- | :--- |
| `segment.instances` | `image` | `labels`（实例，0 为背景）；可选 `probability` | `segment.circles`、`segment.threshold_watershed`、`segment.pixel_classifier` |
| `segment.classes` | `image` | `labels`（每像素类别号，0 为背景） | – （多类画笔、组织分区） |
| `count.inclusions` | `image`、`labels`（液滴） | `table`（`frame`、`label`、`n_inclusions`，可选 `confidence`）；`labels`（计到的位置） | `droplet.inclusions` |
| `classify.objects` | `image`、`labels` | `table`（`frame`、`label`、`class`、`confidence`） | – （液滴分类、细胞表型） |

范围规则不变：`frame` 范围的算子可以进流式执行器，所以要在线使用的模型必须逐帧工作；需要看整段时间或整个体的模型用 `window` / `volume` 范围。

## 与已有功能的关系

- **画笔像素分类器**（`segment.pixel_classifier`）是本机训练的模型，已经按“配方里写训练文件路径 + SHA-256”的方式记录；以后的本地训练会产出带模型卡的权重，走同一套接口。
- **复核与纠错**：模型结果和经典结果一样可以修正，修正记录照常保存；这些修正正是将来训练和评估模型的标注来源。
- **自动规划**：规划器目前只在内置配方里选择。插件算子注册后会出现在 `vexel ops` 与 MCP 的 `list_operators` 中；让规划器使用它们需要另外为每个任务写选择规则，留到平台阶段。
- **批量与复现**：批量运行把用到的模型记进每个文件的溯源；`reproduce.py` 需要同一版本的插件和同一份权重（由 SHA-256 保证）。

## 命令

```bash
uv run vexel plugins          # 已安装的插件、注册的算子与模型、许可证、权重文件状态
uv run vexel plugins --json
```

有插件加载失败时返回码为 1。

## 尚未做（平台阶段）

- ONNX Runtime / PyTorch 的通用加载器与设备选择（CPU、CUDA、Apple MPS），推理分块与批量。
- 模型下载命令（`vexel plugins fetch`，显式由用户执行，下载后校验）。
- SAM 点击式分割在复核窗口中的交互。
- 本地训练（勾选类别 → 训练 → 生成模型卡与权重 → 在留出数据上评估后才允许用于分析）。
- 规划器对插件算子的选择规则。
