# Vexel Web Workbench 开发文档与架构规格

> **最后更新**：2026-11-26（Sprint C / P2 全部收口完成）  
> **适用范围**：根目录 Web 全栈工作台（`src/` + `server.ts` + `deploy/gpu-worker/`），已完成 Phase 1–4 落地及 P0+P1+P2 全量收口（24 维少样本 DL 与 WebGPU 探测、模型 IndexedDB 持久化、`onnxruntime-web` 离线推理、多页 OME-TIFF 与 OME-Zarr 多尺度金字塔桥接、纯像素 PNG 重编码打包、SQLite 多用户鉴权/配额与费用熔断台账、Modal/RunPod 生产 GPU LoRA Worker 部署包与 23 项 Vitest 自动化验证）。  
> **缺口核对与后续排期**：请同步查阅 [docs/GAP_AUDIT_AND_ROADMAP.md](./docs/GAP_AUDIT_AND_ROADMAP.md)。

---

## 1. 系统总体架构

Vexel Web Workbench 采用 **“浏览器端确定性计算 + Web Worker 异步加速 + WebGL2 硬件体渲染 + 服务端域安全代理与声明式训练网关”** 四层结构：

```text
+------------------------------------------------------------------------------+
| Layer 4: UI 视图层 (React 19 + Tailwind CSS v4 + D3 + Three.js)              |
|  - TopBar 模式导航: 工作台 / 3D 实验室 / 模型中心 / 批量处理 / 工具箱         |
|  - 2D 画布: InteractiveCanvas (缩放平移、滚轮调阈值点选、ROI、标注、复核)     |
|  - 3D 视口: WebGL3DViewport (真 3D 光线步进/正交切片 + 2.5D PBR 高度场)       |
|  - 分析面板: InspectorPanel / Studio3DView / ModelHub / NcRatioHistogramPanel |
+------------------------------------------------------------------------------+
| Layer 3: 异步计算与引擎层 (src/engine/* + Dedicated Web Workers)              |
|  - 资源预算门控: resourceBudget.ts (2D 像素与 3D 体素四档分级保护)             |
|  - 2D 少样本 DL: deepFewShotCore.ts + deepTrainWorker.ts + mlRuntime.ts      |
|  - 3D 体分析: volume3d.ts + volumeAnalysisCore.ts + volumeAnalysisWorker.ts  |
|  - 2D 确定性计量: imageProcessor.ts + microscopyDecoder.ts + semanticMapper  |
+------------------------------------------------------------------------------+
| Layer 2: 安全网关与 AI/GPU 代理层 (server.ts, Express on Port 3000)          |
|  - AI 规划网关: /api/v1/ai/* (限流 + 越狱/偏题拦截 + 算子白名单清洗)          |
|  - GPU 训练网关: /api/v1/train/* (纯声明式参数校验 + 沙箱脚本 + ONNX 元数据)  |
|  - SSRF 防护: 屏蔽云元数据地址 (169.254.169.254 / metadata.google.internal)   |
+------------------------------------------------------------------------------+
| Layer 1: 外部算力与模型生态 (可选)                                            |
|  - LLM 规划: Gemini / Claude / DeepSeek / OpenAI 兼容 / 本地 Ollama / 离线规则|
|  - GPU 微调: 本地 CUDA / Modal / RunPod / AutoDL / Vertex AI / SageMaker     |
+------------------------------------------------------------------------------+
```

---

## 2. 技术栈与目录结构

### 核心依赖（`package.json`）
- **前端框架**：React `19.x`、TypeScript、Vite、Tailwind CSS `4.x`
- **AI 推理与 3D 可视化**：`onnxruntime-web`（浏览器端离线 WASM 推理，静态资源挂载于同源 `/ort/`）、`three`（WebGL2 `Data3DTexture` + `RoomEnvironment` + `OrbitControls`）、`d3`、`lucide-react`
- **科学格式与导出**：纯 TypeScript 科学 TIFF / OME-TIFF / Gatan DM3-DM4 解码器、`xlsx`（Excel 导出）
- **后端服务与测试**：`express`、`tsx`、`@google/genai`、`dotenv`、`vitest`

### 源码模块清单

#### 引擎与安全层（`src/engine/` & `src/server/`）
| 模块 | 核心职责 |
| :--- | :--- |
| `resourceBudget.ts` | 2D（`evaluate2DBudget`）与 3D（`evaluate3DBudget`）内存/显存四档评估器，输出瓦片大小、重叠 halo 与降采样建议 |
| `deepFewShotCore.ts` | 浏览器端少样本深度学习引擎：24 通道多尺度特征金字塔（含 Hessian 曲率 $\lambda_1/\lambda_2$、45°/135° 定向 Gabor、宽视野光照校正、形态学 Top-Hat/Bottom-Hat 及外部 WebGPU/ONNX 嵌入融合槽位，向下兼容 16D 模型）、空间分块留出集、瓶颈残差 MLP、512×512 羽化瓦片推理 |
| `webgpuProbe.ts` | 浏览器 `navigator.gpu`（WebGPU）与 WebGL2 硬件能力探测器，输出 3D 纹理/缓冲上限及 ONNX 执行后端建议 |
| `deepTrainWorker.ts` | 专用 Web Worker，承载 `deepFewShotCore.ts` 的 24ch 特征缓存、训练迭代进度推送与中止信号 |
| `mlRuntime.ts` | 主线程训练/推理调度封装（Worker 优先 + 不可用时主线程回退）与传统像素分类器兼容层 |
| `modelStore.ts` | 浏览器 IndexedDB（`vexel_model_registry_v1`）模型持久化库，支持自训练少样本模型与二进制 `.onnx` 模型落盘恢复、`.vexmodel.json` 导出/导入与 SHA-256 防篡改校验 |
| `onnxRuntime.ts` | 基于 `onnxruntime-web` 的离线 ONNX 推理引擎：50 MB 大小门控、图结构与张量维度检查、真实 SHA-256 计算、同源 `/ort/` WASM 加载、NCHW/NHWC 自适应与余弦羽化瓦片推理 |
| `volume3d.ts` | 内置合成 3D 肾细胞-细胞核共聚焦 Z-Stack（128×128×48，5 枚独立细胞+暗核，各向异性标定）、多页 TIFF/OME-TIFF 与多切片图像序列加载器（自动 LOD 降采样与 P0.5–P99.5 对比度拉伸）、2.5D 面粗糙度计算（$S_a, S_q, S_z$） |
| `volumeAnalysisCore.ts` | 三维 Otsu 阈值、6 邻域外部泛洪封闭空腔填充、26 邻域 2-Pass Union-Find 三维连通域（紧凑 `Uint16Array`）、嵌套三维暗细胞核分割、各向异性物理体积 / Crofton 表面积 / 球形度 / 等效直径 / 3×3 协方差 Jacobi 旋转不变 PCA 主轴 `[L1,L2,L3]`（含伸长率与扁平率）/ 3D N/C 体积核质比计算 |
| `volumeAnalysisWorker.ts` / `volumeAnalysisRunner.ts` | 专用 Web Worker 与前端 Promise 调度器（Transferable Buffer 回传、3D CSV/Excel 导出含 PCA 指标） |
| `imageProcessor.ts` / `operators.ts` | 2D 预处理算子、自动分割、边界吸附魔棒与敏感度阈值控制、人工纠错重算、`model.segment`（支持本地少样本 DL 与真实 `onnx-wasm` 概率图直通） |
| `microscopyDecoder.ts` | 纯 TS 2D/3D 科学显微格式解码器：支持单页与多页 8/16/32-bit TIFF / OME-TIFF（UTF-8 OME-XML & ImageJ XY/Z 物理间距统一解析）、SEM 厂商私有标签与 Gatan DM3/DM4 |
| `src/server/securityGuardrails.ts` | 服务端纯函数安全护栏：AI 域意图拦截器、算子白名单清洗、声明式 GPU 训练校验、滑动窗口限流器、私网/元数据地址阻断、GPU 回调 HMAC-SHA256 验签 |
| `src/server/imageSanitizer.ts` | 纯 Node.js PNG 魔数/CRC32/尺寸与防解压炸弹校验、扫描线解码与最小规范 PNG 重编码（剥离所有 EXIF/`tEXt`/尾部多态载荷）、PKZIP 安全训练集打包器 |
| `src/server/governanceStore.ts` | Node 22 原生 SQLite（WAL 模式）多用户鉴权（`scrypt` 加盐哈希 + 首次管理员认领锁定 + RBAC）、持久化配额与日/月 USD 费用熔断台账 |
| `src/server/omeZarrBridge.ts` | OME-NGFF v0.4 / Zarr v3 多尺度金字塔解析器、浏览器 3D 预算最优层选择、受限本地目录路径边界校验与公开 HTTPS 安全探测 |
| `deploy/gpu-worker/*` | 生产云端/本地 GPU LoRA 隔离执行包：`Dockerfile`、`train_worker.py`（`timm` + LoRA 注入 + ONNX 导出 + HMAC 回调）、`modal_app.py`、`runpod_handler.py` |
| `planner.ts` / `specimens.ts` / `recipes.ts` / `models.ts` / `semanticMapper.ts` / `exporter.ts` | AI 规划与离线回退、内置样本/配方/模型卡、字段语义映射与 CSV/Excel/JSON 导出 |

#### 核心 UI 组件（`src/components/`）
| 组件 | 核心职责 |
| :--- | :--- |
| `TopBar.tsx` | 顶栏导航（工作台 / 3D 实验室 / 模型中心 / 批量处理 / 工具箱）、语言与暗黑主题切换 |
| `InteractiveCanvas.tsx` | 2D 主画布：缩放平移、比例尺标定、ROI 框选、涂抹画笔、滚轮灵敏度/阈值调节自动选区、低置信高亮与点击复核 |
| `InspectorPanel.tsx` | 4 阶段工作台右侧面板（标定与裁边 → 配方与预处理 → 少样本模型微调含实时进度条/Loss-mIoU 曲线/Abort → 结果与复核含 `2D 指标 / 3D 体计量` 统一切换） |
| `Studio3DView.tsx` | 独立 3D 实验室：Z-Stack 加载/内置数据、3D 资源预算条、WebGL3DViewport、3D 体分析触发与三维结果表（含 PCA 伸长率）/3D N/C 分布/导出，并与工作台共享 3D 分析结果 |
| `WebGL3DViewport.tsx` | WebGL2 硬件双模视口：真 3D 光线步进（MIP/Alpha/Iso）+ 正交三平面切片 + 真实 3D 体素标签纹理 `uLabelVolume` 叠加 + 3D 目标高亮拾取；2.5D PBR 高度场 + 2D 分割掩膜投影 + X/Y 双向实时剖切 + `156²/256²/384²` 网格切换 + 粗糙度 HUD；软件渲染降级与 `webglcontextlost` 恢复 |
| `Virtual3DPreviewWidget.tsx` | 工作台内嵌轻量 3D 卡片，支持 2.5D 快速预览及一键跳转至完整 `3D 实验室` |
| `ModelHub.tsx` | 模型中心：浏览器端少样本 DL 训练器（实时 Loss / Val-mIoU 曲线与中断）、在线/本地 GPU LoRA 声明式任务控制台（含图像/掩膜纯像素清洗上传、`.zip` 安全训练集下载、脚本下载、审计日志）、本地 ONNX/JSON 模型卡注册 |
| `NcRatioHistogramPanel.tsx` | 基于 D3 的交互式核质比直方图（同时支持 2D 面积 N/C 与 3D 体积 N/C 模式及点击联动筛选） |
| `AgentCommandBar.tsx` | 智能体自然语言指令栏、多供应商切换、域拦截反馈展示 |
| `BatchView.tsx` / `ToolsPanel.tsx` / `PipelineDrawer.tsx` | 批处理队列、辅助工具箱与确定性流程抽屉 |

---

## 3. 2D / 3D 浏览器资源预算与尺寸建议

`src/engine/resourceBudget.ts` 将计算规模显式划分为四档（`optimal` / `tiled_safe` / `lod_guarded` / `offload_gpu_required`）：

### 3.1 2D 图像尺寸建议
| 档位 | 分辨率区间 | 像素量 | 估算峰值内存 | 系统策略 |
| :--- | :--- | :--- | :--- | :--- |
| `optimal` | $\le 2048 \times 2048$ | $\le 4\text{ MP}$ | ~70–90 MB | 浏览器 Worker 实时少样本训练与全图推理 |
| `tiled_safe` | $\le 4096 \times 4096$ | $\le 16\text{ MP}$ | ~220 MB | 自动启用 `512×512`（`64px` overlap）羽化瓦片流式推理 |
| `lod_guarded` | $\le 8192 \times 8192$ | $\le 64\text{ MP}$ | ~600 MB | 建议框选局部 ROI 训练，结合分块流式推理（浏览器软上限） |
| `offload_gpu_required` | $> 8192 \times 8192$ | $> 64\text{ MP}$ | $> 600$ MB | 建议外移至本地工作站或云端 GPU（OME-TIFF / OME-Zarr 分块） |

### 3.2 3D 体数据（Z-Stack）尺寸建议
| 档位 | 体素尺寸区间 | 体素量 | 估算峰值内存 | 系统策略 |
| :--- | :--- | :--- | :--- | :--- |
| `optimal` | $\le 512 \times 512 \times 64$ | $\le 16.8\text{M}$ | ~90–115 MB | WebGL2 原生全分辨率 `Data3DTexture` 光线步进 + 26 邻域体分析 |
| `tiled_safe` | $\le 512 \times 512 \times 128$ | $\le 33.5\text{M}$ | ~160–200 MB | 紧凑 `Uint16` 标签 + 自适应步长渲染 |
| `lod_guarded` | $\le 1024 \times 1024 \times 64$ | $\le 67.1\text{M}$ | ~300–350 MB | `loadZStackFromFiles` 自动降采样至 $\le 512 \times 512 \times 128$ 并按比例同步校正 `voxelSizeXY` / `voxelSizeZ` |
| `offload_gpu_required` | $> 67.1\text{M}$ 体素 | $> 67.1\text{M}$ | $> 350$ MB | 浏览器强制降采样预览；全精度分析需交由本地/云端 GPU 后端 |

---

## 4. 后端 API 与安全防滥用契约（`server.ts`）

所有 API 路由运行于 Express（端口 `3000`），**密钥仅保存在服务端环境变量，前端永不接触密钥**。

| 端点 | 方法 | 限流 | 核心职责与安全校验 |
| :--- | :--- | :--- | :--- |
| `/api/v1/ai/providers` | `GET` | — | 返回各 LLM 供应商就绪状态（仅布尔值，不暴露 Key） |
| `/api/v1/ai/plan` | `POST` | 30 次/分/IP | 1) 长度 $\le 1000$ 字符校验；2) `checkImageDomainIntent` 拦截越狱/提示词注入/非图像闲聊任务（返回 `400`，不消耗云端 Token）；3) `sanitizePlanProposal` 按 `ALLOWED_PREPROCESS_OPS` / `ALLOWED_ALL_OPS` 白名单清洗输出；4) 无 Key 时安全离线回退 |
| `/api/v1/ai/inspect-import` | `POST` | 30 次/分/IP | 导入显微图像自动视觉识别（模态、信息栏位置、比例尺文本） |
| `/api/v1/train/presets` | `GET` | — | 返回受控 GPU 服务商预设、允许的骨干网络（`dinov2_vit_s14`、`convnext_tiny`、`sam_vit_b_lora`、`unet_resnet34`）、LoRA 秩（`4/8/16`）、损失函数（`focal_dice`/`ce_dice`/`tversky`）及最大超时上限（900s） |
| `/api/v1/train/jobs` | `GET` | — | 返回近期训练任务审计日志 |
| `/api/v1/train/jobs` | `POST` | 6 次/分/IP，单 IP 并发=1 | 1) 拒绝任何含 `FORBIDDEN_CODE_FIELDS` 的请求；2) 白名单枚举与前置标注像素门禁校验；3) 接收样本与掩膜 PNG 时执行纯像素解码重编码（校验魔数/CRC32/防解压炸弹，剥离 EXIF/`tEXt`/尾部多态字节并校验宽高一致），打包为含 `image.png + mask.png + manifest.json + train_vexel_lora.py` 的 `.zip` 数据包；4) 如配置 `VEXEL_GPU_TRAIN_ENDPOINT` 则校验非元数据内网地址后仅转发纯像素清洗后载荷 |
| `/api/v1/train/import-onnx` | `POST` | — | 校验外部导出 ONNX 模型元数据、任务类型与 SHA-256，生成标准化 `ModelCard` |
| `/api/v1/train/callback` | `POST` | — | 校验 `X-Vexel-Signature`（HMAC-SHA256），将云端 GPU Worker 实际耗时、费用、ONNX SHA-256 与验证 mIoU 回写 SQLite 台账 |
| `/api/v1/auth/*` | `GET/POST` | 20 次/分/IP | 首次管理员认领（`/bootstrap-admin`，完成后自动锁定）、研究员注册（`/register`）、登录（`/login`）与状态检查（`/status`, `/me`） |
| `/api/v1/governance/*` | `GET/POST` | — | 查询配额/预算摘要（`/summary`）；管理员调整日/月 USD 上限与触发/解除成本熔断（`/budget`）、查看 SQLite 持久化审计台账（`/ledger`） |
| `/api/v1/zarr/inspect` | `POST` | — | OME-Zarr v0.4/v3 多尺度金字塔解析与浏览器 3D 预算最优层推荐（支持 JSON、公开 HTTPS 与受限本地目录） |

### 环境变量（见 `.env.example`）
- LLM 代理：`GEMINI_API_KEY`、`ANTHROPIC_API_KEY`、`DEEPSEEK_API_KEY`、`OPENAI_API_KEY`、`OPENAI_BASE_URL`、`OLLAMA_BASE_URL`
- 在线 GPU 训练桥接与签名回调：`VEXEL_GPU_TRAIN_ENDPOINT`、`VEXEL_GPU_TRAIN_TOKEN`、`VEXEL_TRAIN_MAX_WALL_CLOCK_SEC`、`VEXEL_GPU_CALLBACK_SECRET`
- 多用户治理与 OME-Zarr：`VEXEL_ADMIN_PASSCODE`、`VEXEL_DB_PATH`、`VEXEL_ZARR_ROOT`

---

## 5. 本地工作站与云端 GPU 选型基线

- **本地推荐配置**：
  - 科研主力档（推荐）：RTX 4090 (24 GB) 或 RTX 5090 (32 GB) + 64 GB RAM + 2 TB NVMe（兼顾 2D LoRA 微调与大规模 3D Z-Stack 预处理）。
  - 入门实验室档：RTX 4070 Ti Super (16 GB) + 32 GB RAM。
  - 高隐私/大规模体电镜档：RTX 6000 Ada / Pro 6000 (48–96 GB) + 128 GB RAM。
- **云端 Serverless GPU 首选**：
  - 海外/全局弹性：**Modal Serverless GPU**（按秒计费、空闲 0 成本、容器秒级冷启动）或 **RunPod Serverless**。
  - 中国大陆低延迟：**AutoDL / 算力云**。
  - 企业合规与 VPC 隔离：**Google Cloud Vertex AI / Cloud Run GPU** 或 **AWS SageMaker**。
- **浏览器 3D 渲染路线**：当前以 **Three.js + WebGL2** 覆盖 98%+ 终端兼容；渲染核心与场景编排解耦，待 WebGPU 在 Safari/Linux 全面稳定后平滑切换至 `WebGPURenderer` + WGSL Compute Shader。

---

## 6. 自动化测试与构建验证命令

```bash
npm run lint   # TypeScript 严格静态类型检查 (tsc --noEmit)
npm test       # Vitest 自动化验证套件 (tests/engine.test.ts, tests/onnx.test.ts, tests/security.test.ts)
npm run build  # Vite 生产构建（含 src/assets 与 /ort ONNX WASM 资源同步拷贝）
```
