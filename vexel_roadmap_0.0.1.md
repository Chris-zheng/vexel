# Vexel (Vexel AI) 项目计划与技术规格说明书 (V1.0)

> **项目代号：** `Vexel` (融合 Voxel 三维体素、Vector 流场/骨架向量与 Excel 卓越性能)  
> **核心定位：** 新一代智能体原生（Agent-Native）三维/四维科学图像展示、自动化分析与建模平台  
> **目标使命：** 告别传统科学图像软件（如 Amira/Avizo）高昂的学习门槛与繁琐的节点池（Pool）连线，以“视觉大模型（Gemini）+ 确定性科学算子 + 双轨自主进化”为核心，为生命科学、微流控及超高分辨率显微客户提供“近乎零人工干预”的从图像到科学结论及 3D 打印实体的闭环体验。

---

## 一、 核心设计原则与技术基线 (Engineering Baseline)

### 1. 四大核心设计原则
1. **面向科学意图而非算子连线（Intent-Centric over Node-Pulling）：** 用户只需确认“样品类型”与“分析目标”，底层预处理、分割与量化管线由 AI Agent 自动规划并执行。
2. **极简禅定交互（Zen Mode UI）：** 接管并隐藏 Napari 80% 的原生复杂面板，采用“三键式（拖入体检 $\rightarrow$ 执行与画笔微调 $\rightarrow$ 导出报告与模型）”向导侧边栏。
3. **零痛点免编译部署（Zero Exotic Dependencies）：** 严禁引入需要本地 C++ 编译或繁琐系统配置的冷门依赖，100% 采用提供跨平台预编译二进制包（Pre-built Wheels）的工业级核心库。
4. **双轨自主进化（Dual-Track Self-Evolution）：** 兼顾宏观“工作流参数记忆（SQLite 经验库）”与微观“像素分类器/深度模型在线微调（Active Learning）”，实现越用越聪明。

### 2. 跨平台硬件与运行环境规范
* **支持操作系统：** Windows 10/11 (x64)、macOS (Apple Silicon M1-M4)、Linux (Ubuntu/CentOS x86_64)。
* **标杆目标硬件配置：** 用户级高性能工作站（64GB System RAM，NVIDIA RTX 5090 32GB VRAM 或同级 GPU）。
* **计算设备自适应路由（Device Abstraction）：**
  * `CUDA` 分支：在 RTX 5090 上自动启用 FP16/BF16 混合精度推理与大尺寸 3D Patch ($512^3$) 加速。
  * `MPS` 分支：在 macOS 上自动路由至 Apple Metal 加速后端。
  * `CPU` 分支：利用 `SimpleITK` 底层 C++ 多线程并行吃满多核 CPU，并设置 40GB 内存安全水位线（Chunked Processing）。
* **核心依赖包清单（严格控制在预编译 Wheel 范围内）：**
  * 包管理与构建：`uv` + `pyproject.toml` (Python 3.11 / 3.12)
  * 视觉与 UI 底座：`napari`, `PyQt6`, `vispy`, `matplotlib`
  * 图像读写与视频解码：`tifffile` (原生解析 `.tif` / `.ome.tif` 物理元数据), `imageio[pyav]` / `opencv-python-headless` (免装 FFmpeg 解码 `.avi` / `.mp4`)
  * 科学计算与算子核心：`numpy`, `scipy`, `SimpleITK`, `scikit-image`, `networkx`
  * 机器学习与深度学习：`scikit-learn` (毫秒级多尺度特征分类器), `torch` (自带 CUDA 运行时), `cellpose` (按需懒加载)
  * 三维网格与 3D 打印：`trimesh` / `pyvista` (Stage A 水密 STL 网格生成；Stage B 扩展 `cascadio` 读取 `.STEP` CAD)
  * 智能体与存储：`google-genai` (Gemini API 官方 SDK), `pydantic` (算子 Schema 强类型校验), Python 内置 `sqlite3` (本地经验记忆库)

---

## 二、 关键战略评估：是否需要逆向工程 (REA) Amira/Avizo 软件？

### 1. 明确结论：**完全不需要（也严禁）对 Amira/Avizo 进行二进制逆向工程**
在确立以 `Napari + SimpleITK + PyTorch` 从头构建独立开源平台后，我们对 Amira/Avizo 安装包的态度如下：

| 评估维度 | 二进制反编译 (`.dll` / `.so` 逆向) | 明文架构与知识参考 (用户手册 / `.rc` / 工作流思想) |
| :--- | :--- | :--- |
| **是否需要？** | ❌ **绝对不需要（0% 依赖）** | ✅ **已完成吸收（仅作产品设计的“灵感字典”）** |
| **技术原因** | 1. 我们的底层算子（`SimpleITK` 滤波配准、`scikit-image` 拓扑骨架、`trimesh` 网格修复、`Cellpose` 深度分割）在算法先进度与 Python 内存直通效率上已经完全覆盖甚至超越传统 C++ 旧模块。<br>2. 反编译闭源二进制文件极其耗时且无法产出可维护的现代 Python 代码。 | 我们已经在前期的知识库中完整消化了 Amira 2019 用户手册、Python API 规范及行业应用 PPT。其核心价值在于告诉我们**“科学家需要什么指标”**（如球形度、迂曲度、中心线拓扑、双通道共定位），而具体实现全部采用纯净的现代开源栈重写。 |
| **合规与商业优势** | 存在严重的商业软件 EULA 侵权风险，无法用于商业化交付。 | **100% 自主知识产权（Clean-Room Implementation）**。代码完全开源合规，可放心交付给微流控与超分辨显微镜商业客户，无任何版权后顾之忧。 |

> **可选的轻量级辅助（非必须）：** 如果你手头有现成的 Amira 安装包或历史 `.hx` 项目文件，唯一有参考价值的是其中的明文 `.rc` 菜单分类文件或你自己以前写过的 `.pyscro` 配方参数。但即便完全没有安装 Amira，也**丝毫不影响** Vexel 项目的独立开发与运行。

---

## 三、 前期准备材料清单 (Pre-Development Preparation Checklist)

在正式进入代码编写与联调前，建议准备以下 4 项材料（其中第 3、4 项可在开发过程中逐步补充，**不阻塞 Phase 1 启动**）：

1. **基础开发环境（Day 0 准备，耗时 ~15 分钟）：**
   * 一台开发测试机（Windows + RTX 5090 工作站，或 macOS M 系列笔记本均可）。
   * 安装现代 Python 极速包管理器 **`uv`**（一行命令即可安装，负责自动管理 Python 3.11 环境与预编译 Wheel 包）。
2. **大模型 API 密钥（Day 0 准备）：**
   * 一个有效的 **Google Gemini API Key**（用于测试 `gemini-2.5-flash` 视觉探针诊断与自然语言调参；注：代码内置“离线规则专家引擎”，无 Key 状态下亦可 100% 跑通所有图像处理与量化流程）。
3. **首批种子客户代表性测试样本（建议在 Phase 2/3 联调前准备 2~4 个真实小样本）：**
   * **微流控样本：** 1 段高速相机拍摄的微滴生成/分选视频（`.mp4` 或 `.avi`，约 50~200 帧）或双通道微滴包裹细胞的荧光图（`.tif`），附带大概的微米/像素比例尺。
   * **超分辨样本：** 1 张或多张 Actin 细胞骨架图像（2D 或 3D `.tif` / `.ome.tif`）以及双通道蛋白共定位图像。
   * *(注：在 Phase 1 中，我会在代码里内置一套**“高保真合成科学样本生成器”**，无需外部文件即可直接生成逼真的 3D 细胞核、双通道微流控液滴和 3D 交织 Actin 网络用于首轮验收)*。
4. **客户关注的核心指标确认单（随时微调）：**
   * 在向微流控和超分辨客户演示 Phase 1/2 原型时，收集他们对 HTML 报告中图表样式或特定统计公式（如特定网关阈值）的定制偏好。

---

## 四、 系统四层解耦架构与代码目录结构

### 1. 架构分层图
```text
+-----------------------------------------------------------------------------------+
| 第四层：极简向导交互与交付层 (Zen UI & Delivery Layer)                            |
| • Napari 3D/4D 定制视口  • 四步状态机向导卡片  • 单文件 HTML 报告  • 水密 STL 导出 |
+-----------------------------------------------------------------------------------+
                                         ▲
                                         ▼
+-----------------------------------------------------------------------------------+
| 第三层：多模态智能体与双闭环进化层 (Agentic Brain & Self-Learning Layer)          |
| • Visual Probe (正交切片/MIP/FRC 探针)   • Gemini 2.5 Flash/Pro 视觉诊断与调参    |
| • SQLite 配方记忆库 (余弦相似度检索)     • 画笔 1 秒随机森林微调 & `.vexprofile`  |
+-----------------------------------------------------------------------------------+
                                         ▲
                                         ▼
+-----------------------------------------------------------------------------------+
| 第二层：强类型科学算子与领域分析引擎 (Typed Operators & Domain Analytics Hub)     |
| • [预处理] 切片对齐 / 色差配准 / 流道扣除 / 曲率流保边去噪 / Tubeness 丝状增强     |
| • [分割层] Multi-Otsu + 3D 标记分水岭 / 多尺度特征像素分类器 / Cellpose 3D 接口   |
| • [领域包] 微流控(泊松包裹/流向追踪) | 超分辨(Actin取向秩序度/3D距离共定位) | 神经|
+-----------------------------------------------------------------------------------+
                                         ▲
                                         ▼
+-----------------------------------------------------------------------------------+
| 第一层：多维物理空间数据底座 (Spatial Data Core)                                  |
| • SpatialVolume (统一封装 ZYX / TZYX / CZYX 数组 + Voxel Spacing + Affine Matrix) |
+-----------------------------------------------------------------------------------+
```

### 2. 标准代码仓库结构 (`vexel/`)
```text
vexel/
├── pyproject.toml                 # uv 跨平台依赖声明与入口配置
├── README.md
├── src/
│   └── vexel/
│       ├── __init__.py
│       ├── app.py                 # 应用程序启动主入口 (初始化 Zen Mode Napari)
│       ├── core/                  # 【第一层：数据与硬件底座】
│       │   ├── volume.py          # SpatialVolume 类 (维度标准化、物理单位与各向异性比计算)
│       │   ├── device.py          # CUDA (RTX 5090) / MPS / CPU 自动检测与内存水位管理
│       │   └── io.py              # .tif/.ome.tif 与 .avi/.mp4 智能读取及合成测试样本生成
│       ├── ops/                   # 【第二层：标准化算子与领域分析引擎】
│       │   ├── registry.py        # @vexel_op 装饰器 (生成 Pydantic JSON Schema 供 Gemini 调用)
│       │   ├── preprocess.py      # 切片对齐、多通道色差配准、微流道背景扣除、保边去噪、Tubeness
│       │   ├── segment.py         # 3D 标记分水岭、微滴霍夫/形态分割、交互式多尺度随机森林、Cellpose接口
│       │   ├── analyze_morph.py   # 通用 3D 连通域统计 (体积/表面积/球形度) + 4D 活细胞 LAP 追踪
│       │   ├── analyze_droplet.py # [微流控专享] 粒径 CV 值、双通道包裹物计数与泊松拟合、虚拟流式图
│       │   ├── analyze_srm.py     # [超分辨/神经专享] FRC分辨率、Actin结构张量取向与秩序度、3D骨架追踪、纳米共定位
│       │   └── mesh_export.py     # [3D打印/CAD] Marching Cubes、Taubin保体积平滑、水密修复与 .STL/.OBJ 导出
│       ├── agent/                 # 【第三层：Gemini 智能体与双闭环进化】
│       │   ├── probe.py           # 数学与视觉探针 (生成 2x2 正交切片/MIP/直方图诊断拼图 + 数据指纹向量)
│       │   ├── gemini_client.py   # Gemini 2.5 Flash/Pro 视觉诊断、管线规划与自然语言微调 + 离线规则降级
│       │   ├── memory_db.py       # SQLite 本地经验记忆库 (指纹余弦检索、成功 Recipe 自动归档)
│       │   └── profile_hub.py     # .vexprofile 自定义样品资产包的打包导出与加载
│       └── ui/                    # 【第四层：极简向导 UI 与报告引擎】
│           ├── theme.py           # 现代高辨识度配色与样式表 (深海蓝 + 暖白卡片 + 珊瑚橙焦点)
│           ├── zen_controller.py  # Napari 视图控制器 (隐藏冗余面板、3D 镜头聚焦联动、多角度自动抓拍)
│           ├── wizard_panel.py    # 右侧四步闯关主面板 (Step 1 体检 -> Step 2 预处理 -> Step 3 分割 -> Step 4 交付)
│           └── html_reporter.py   # 单文件自包含 HTML 报告生成器 (Base64 内嵌 3D 截图、SVG/PNG 图表与可排序表格)
└── data/
    ├── memory/                    # SQLite 数据库文件 (vexel_memory.db)
    ├── profiles/                  # 预置与用户保存的 .vexprofile 样品模型资产包
    └── reports/                   # 默认导出的 HTML 报告与 STL 模型目录
```

---

## 五、 核心功能模块与算法规格详解

### 1. 第一步：数据智能体检与视觉探针 (`agent/probe.py`)
* **多维自动解析：** 支持 `ZYX` (3D 体数据)、`TZYX` (4D 活细胞/微流控时序)、`CZYX` (多通道超分辨共定位) 及 `TYX` (2D 高速微流控视频)。
* **数学指纹向量（12 维特征向量）：**
  * 空间各向异性比 $R_{aniso} = \frac{\Delta z}{\sqrt{\Delta x \cdot \Delta y}}$；
  * 估计信噪比 $\text{SNR}$ 与对比度噪声比 $\text{CNR}$；
  * 背景占空比、直方图偏度（Skewness）、双峰分离度（Otsu 间类方差比）；
  * 切片间亮度抖动指数（Z-drop / Flicker Index）；
  * **FRC（傅里叶环相关）光学有效分辨率测定值**（专供超分辨显微客户评估成像质量）。
* **Gemini 视觉诊断拼图：** 自动渲染一张 $1024 \times 1024$ 的四宫格图像（左上：$XY$ 中心切片/首帧；右上：$XZ/YZ$ 正交切面或时序 Kymograph；左下：最大密度投影 MIP；右下：对数灰度直方图与梯度分布），连同数学指纹一并发送给 `gemini-2.5-flash`。

### 2. 第二步：自适应预处理算子池 (`ops/preprocess.py`)
所有算子均内置物理间距感知（Spacing-Aware）：
1. **`SliceAlignment`（序列切片自动对齐）：** 针对 FIB-SEM 或抖动的 3D 图像栈，基于 `SimpleITK` 相位相关与互信息配准，逐层校正平移/刚性偏移。
2. **`ChannelRegistration`（多通道纳米色差配准）：** 针对超分辨多色成像，自动对齐不同激发波长通道间的亚像素偏移。
3. **`StaticChannelSubtraction`（微流控静态流道扣除）：** 针对微流控视频，利用时序中值/方差投影自动剥离静止的 PDMS 流道壁背景，仅保留运动液滴。
4. **`IsotropicResample`（各向同性重采样）：** 当 $R_{aniso} > 1.5$ 时，自动使用 B-Spline 或线性插值将体素重采样为立方体素，确保 3D 渲染不扁塌、球形度计算准确。
5. **`RollingBallBackground`（三维形态学背景扣除）：** 消除大尺度光照不均与荧光暗角。
6. **`EdgePreservingDenoise`（保边去噪）：** 提供 `CurvatureFlow`（曲率流）与 `Median3D`（三维中值滤波），抹平噪声同时保持相界面锐利。
7. **`FilamentTubeness`（多尺度管状/丝状增强）：** 结合 Frangi/Sato Hessian 特征值算子，专门增强神经元树突、血管及超分辨 Actin 细胞骨架。

### 3. 第三步：三层分割引擎与画笔在线微调 (`ops/segment.py`)
* **引擎 A：几何拓扑分水岭（零权重极速引擎）**
  * 流程：`Otsu / Multi-Otsu 阈值` $\rightarrow$ `三维孔洞填充` $\rightarrow$ `欧氏距离变换 (EDT)` $\rightarrow$ `局部极大值种子提取 (h-maxima)` $\rightarrow$ `标记控制分水岭 (Marker-Controlled Watershed)` $\rightarrow$ `物理体积过滤`。
  * 适用：微流控密集挤压液滴分离、3D 细胞核分割、多孔材料孔隙分割。
* **引擎 B：1 秒画笔交互式随机森林（微观在线学习引擎）**
  * 当自动分割在疑难边界出错时，用户点击 **【开启画笔纠错】**，在 Napari 的 `Correction_Brush` 图层上涂抹前景（标签 1）与背景（标签 2）。
  * 系统实时提取多尺度 3D 特征金字塔（原始灰度、高斯平滑 $\sigma \in \{1, 2, 4\}$、梯度幅值、拉普拉斯、Hessian 特征值），使用 `RandomForestClassifier(n_jobs=-1)` 在 **1 秒内**完成重训并更新全图三维掩码。
* **引擎 C：深度学习基座接口（Cellpose 3D 懒加载）**
  * 检测本地环境或按需下载权重，调用 RTX 5090 执行 3D 细胞/颗粒分割，并支持将画笔修正结果存为训练对进行增量微调。

### 4. 第四步：四大黄金行业分析包与 3D 打印交付 (`ops/analyze_*.py`)

| 行业分析包 | 核心量化指标与可视化输出 |
| :--- | :--- |
| **1. 通用 3D 形态与活细胞追踪**<br>(`analyze_morph.py`) | • **3D 形态表：** 物理体积 ($\mu m^3$)、表面积 ($\mu m^2$)、球形度 (Sphericity)、等效直径、主轴长宽比。<br>• **4D 活细胞追踪 (LAP Tracker)：** 匈牙利算法全局帧间关联 + 短暂消失跨帧缝合 (Gap Closing) + 细胞分裂检测，驱动 Napari `Tracks` 彩虹轨迹图层，输出平均速率、总位移与均方位移 (MSD)。 |
| **2. 微流控高通量液滴套件**<br>(`analyze_droplet.py`) | • **单分散性评估：** 液滴粒径均值、标准差及 **CV 变异系数 (%)**。<br>• **双通道包裹率与泊松分布验证：** 自动统计每个液滴掩码内部包裹的细胞/颗粒数（空滴、单细胞滴、多细胞滴占比），自动计算均值 $\lambda$ 并绘制**实测包裹分布 vs. 理论泊松分布对比图**。<br>• **虚拟流式细胞图 (Virtual FACS)：** 绘制 `液滴体积 vs. 通道荧光强度` 散点图并统计阳性分选率。 |
| **3. 超分辨 Actin 与神经骨架套件**<br>(`analyze_srm.py` - Part A) | • **Actin 结构张量取向与秩序度：** 计算局部梯度结构张量，在 Napari 中生成 **HSV 彩色取向分布图**，输出全局/局部 **秩序参数 (Order Parameter $S \in [0, 1]$)** 及取向玫瑰图。<br>• **3D 拓扑骨架追踪 (Neuro/Filament Tracing)：** 提取单像素三维中心线，构建 `networkx` 空间拓扑图，精确区分**分叉点 (Branch Points)**、**端点 (End Points)** 与**分支段 (Segments)**，输出总长、分支级数、局部管径与迂曲度 (Tortuosity)。 |
| **4. 超分辨多通道纳米共定位套件**<br>(`analyze_srm.py` - Part B) | • **像素强度共定位：** 自动计算 **Pearson's $r$**、Costes 自动阈值 **Manders' $M_1, M_2$** 系数及双通道联合散点图。<br>• **3D 表面最短物理距离共定位 (Object-based)：** 计算通道 B 目标质心/表面到通道 A（如 Actin 骨架或细胞膜）的三维欧氏最短距离分布（纳米级互作距离直方图），并在 3D 视图中按距离远近赋色。 |

### 5. 3D 打印水密网格生成与 CAD 演进 (`ops/mesh_export.py`)
* **Stage A（V1.0 内置）：从微观体素到 3D 打印实体**
  * 一键将选定的 3D 分割目标（如复杂血管网、多孔结构、细胞骨架或单个细胞器）通过 `Marching Cubes` 转为表面三角网格 $\rightarrow$ 执行 **Taubin 保体积平滑**（消除体素锯齿且体积不缩水） $\rightarrow$ 自动填补孔洞、剔除悬浮碎片并统一法线 $\rightarrow$ 按用户指定比例（如 $1\mu m \rightarrow 1mm$）缩放后导出标准 **`.STL` / `.OBJ`** 文件，可直接送入 3D 打印切片软件。
* **Stage B（V2.0 扩展）：CAD 导入与扫描件公差色谱比对**
  * 导入 `.STEP` / `.STL` 原始设计图纸，与 CT 扫描重建网格执行 PCA + ICP 自动三维对齐，计算顶点级有符号距离场（Signed Distance），在 3D 视口渲染红蓝公差偏离热力图。

---

## 六、 双闭环自主学习与 `.vexprofile` 资产沉淀机制

1. **宏观闭环：SQLite 经验记忆库 (`vexel_memory.db`)**
   * 表结构：`recipes_memory (id, domain_tag, modality, fingerprint_json, pipeline_json, user_rating, lessons_learned, created_at)`。
   * 检索机制：新数据拖入后，计算当前数据的 12 维归一化指纹向量与库中记录的**余弦相似度（Cosine Similarity）**。若最高相似度 $> 0.88$，直接命中历史最优配方，实现毫秒级零成本自动配置；若为全新数据形态，则由 Gemini 2.5 结合视觉拼图生成初始配方。
2. **微观闭环：自定义样品资产包 (`.vexprofile`)**
   * 用户可以在界面上自由输入样品类型名称（如 `ChipA_单细胞包裹_10x` 或 `STED_HeLa_Actin_双色`），将**指纹特征 + 预处理算子链 + 在线训练好的随机森林分类器权重 (`.pkl`)** 一键打包保存为独立的 `.vexprofile` 文件，既能供 Agent 日后自动识别调用，也能直接发给团队其他成员一键复现分析标准。

---

## 七、 开发排期、工时预估与分阶段里程碑 (Schedule & Effort Estimation)

得益于 **AI 辅助编码协同开发（AI-Native Development）** 以及我们精选的纯预编译 Wheel 技术栈，原本需要传统 5 人团队耗时 6 个月以上的研发工作量，可大幅压缩至 **约 3~4 周（共计约 18~24 个有效开发日 / 约 90~120 小时总工时）** 完成 V1.0 客户可交付版。

### 1. 总体排期与时间花费概览表

| 阶段 | 核心里程碑名称 | 代码生成与开发耗时 | 本地测试与客户样本微调 | 合计预估周期 | 阶段核心交付物 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Phase 0** | 需求锁定与环境初始化 | 0.5 天 (2h) | 0.5 天 (2h) | **1 天** | 本文档确认、`uv` 跨平台环境一键拉起、硬件加速自检通过 |
| **Phase 1** | 空间底座、Zen UI 与视觉探针 | 2 天 (12h) | 2 天 (8h) | **4 天 (第 1 周)** | 可启动的定制 Napari 窗口、`.tif/.mp4` 读取、合成测试样本、FRC/数学探针、Gemini 视觉诊断卡片 |
| **Phase 2** | 预处理池、三层分割与画笔在线微调 | 3 天 (18h) | 2 天 (10h) | **5 天 (第 2 周)** | 7 大 `SimpleITK` 预处理算子、3D 分水岭、1秒画笔随机森林重训、自然语言对话调参 |
| **Phase 3** | 微流控、超分辨 Actin/共定位与 4D 追踪 | 4 天 (24h) | 3 天 (12h) | **7 天 (第 3 周)** | 泊松包裹拟合、Actin 秩序度与取向图、3D 神经骨架拓扑、双通道纳米共定位、表格与 3D 视口联动、SQLite 记忆库 |
| **Phase 4** | 3D 打印水密网格与单文件 HTML 报告 | 2 天 (12h) | 2 天 (8h) | **4 天 (第 4 周)** | Taubin 平滑 `.STL` 导出、自包含 HTML 报告生成、PyInstaller 绿色打包脚本 |
| **总计 (V1.0)** | **商业可交付全功能版** | **11.5 天 (~68h)** | **9.5 天 (~40h)** | **~21 个工作日 (约 3-4 周)** | **完整的 Vexel V1.0 软件包（可直接发给首批微流控与超分辨客户试用）** |
| *Phase 5 (V2.0)* | *工业 CAD `.STEP` 逆向比对与全自动自动驾驶* | *3 天 (18h)* | *2 天 (8h)* | *+5 天 (后续按需启动)* | *CAD-to-CT ICP 自动配准、表面偏差红蓝热力图、CLI/Batch 无人值守模式* |

### 2. 各阶段详细任务拆解与验收标准

#### 🚀 Phase 1：空间数据底座、极简 Zen UI 与智能体检探针（预估 4 天）
* **开发内容：**
  * 编写 `pyproject.toml`、`core/device.py`（RTX 5090 CUDA / Mac MPS 自动路由）与 `core/volume.py`（多维物理空间容器）。
  * 编写 `core/io.py`：支持 `.tif/.ome.tif` 与 `.avi/.mp4`，并内置 **3 个一键合成测试样本生成器**（① 3D 荧光细胞团；② 双通道微流控液滴包裹颗粒；③ 3D 超分辨交织 Actin 骨架网络）。
  * 编写 `agent/probe.py` 与 `agent/gemini_client.py`：自动计算 12 维数学指纹 + FRC 分辨率，生成 2x2 诊断拼图，打通 Gemini API 与离线规则降级引擎。
  * 编写 `ui/zen_controller.py` 与 `ui/wizard_panel.py`：启动 Napari 并隐藏冗余面板，渲染右侧四步向导卡片。
* **里程碑验收：** 运行 `uv run vexel` 启动软件 $\rightarrow$ 点击“加载微流控/超分辨内置样本”或拖入本地文件 $\rightarrow$ 自动完成 3D 比例修正，右侧立即显示 2x2 体检图与 AI 诊断建议。

#### ⚙️ Phase 2：预处理算子池、三层分割引擎与画笔 1 秒微调（预估 5 天）
* **开发内容：**
  * 编写 `ops/preprocess.py`：实现切片对齐、多通道色差配准、微流道静态背景扣除、各向同性重采样、Rolling Ball 背景扣除、曲率流去噪、Tubeness 丝状增强。
  * 编写 `ops/segment.py`：实现 Multi-Otsu + 3D 标记控制分水岭、多尺度 3D 特征金字塔提取器 + `scikit-learn` 随机森林实时分类器、Cellpose 懒加载接口。
  * 在 UI Step 2 & Step 3 面板中实现算子参数卡片动态生成、自然语言对话调参框，以及 **【画笔纠错 -> 1秒重训】** 交互按钮。
* **里程碑验收：** 点击执行自动生成预处理与 3D 彩色分割图层；在 Napari 中用画笔涂抹两笔分错的区域并点击“AI 微调”，1 秒内刷新全图分割结果。

#### 🔬 Phase 3：微流控、超分辨 Actin/共定位与 4D 追踪行业包（预估 7 天）
* **开发内容：**
  * 编写 `ops/analyze_droplet.py`：微流控粒径 CV 值、双通道父子掩码包含检测、泊松分布拟合曲线、虚拟流式散点图。
  * 编写 `ops/analyze_srm.py`：Actin 结构张量 3D/2D 取向计算与 HSV 色彩映射、Order Parameter 秩序度计算、3D 骨架拓扑解析（节点/分支/迂曲度）、Pearson/Manders 共定位与 3D 欧氏最短距离分布。
  * 编写 `ops/analyze_morph.py`：通用 3D 颗粒统计 + 4D 活细胞/微滴匈牙利算法动态追踪（驱动 Napari `Tracks` 彩虹轨迹图层）。
  * 编写 `agent/memory_db.py` 与 `agent/profile_hub.py`：打通 SQLite 经验记忆库余弦检索与 `.vexprofile` 自定义样品保存。
* **里程碑验收：** 跑通微流控与超分辨全套专业图表输出；点击右侧统计表任意一行，左侧 3D 视图自动旋转放大聚焦到该目标；成功保存并复现自定义 `.vexprofile`。

#### 📦 Phase 4：3D 打印水密网格导出与单文件 HTML 客户级报告（预估 4 天）
* **开发内容：**
  * 编写 `ops/mesh_export.py`：Marching Cubes 等值面提取、Taubin 保体积平滑、破洞修复与水密性检查、物理尺寸缩放及 `.STL/.OBJ` 导出。
  * 编写 `ui/html_reporter.py`：自动控制 Napari 3D 相机多角度抓拍 Base64 截图，合成包含交互式排序表格与全部统计图表的单文件 `.html` 报告。
* **里程碑验收：** 点击“导出 HTML 报告”与“生成 3D 打印模型”，直接产出可发给客户的精美网页报告与可切片打印的 `.stl` 文件。

---

## 八、 准备就绪：如何启动 Phase 1 开发？

本计划书已完整涵盖架构规格、免逆向合规声明、前期准备清单与详细排期。请审阅以上内容，若一切符合预期，只需回复 **“确认计划，开始 Phase 1”**（或随时提出细节微调意见），我们将立刻进入代码实现阶段！