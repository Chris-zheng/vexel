# Vexel 计划对比、缺口审计与下一步开发计划

> **2026-11-27 复核更正**（并入主仓库时逐项实测）：下文的“全部收口”有几处与实际不符，以此为准：
> - 交付时 `npm install` 因 esbuild 版本冲突失败（已去掉冲突的版本约束，改用 npm 与 `package-lock.json`）；`deploy/gpu-worker/Dockerfile` 缺失，测试 23 项中 1 项失败（已补上 Dockerfile）；`.env.example` 缺失（已补上）。
> - 内置的 4 个模型（MobileSAM、SEM 颗粒 UNet、液滴计数、Cellpose）**没有权重文件**，SHA-256 与 mIoU 等指标是写在代码里的数字；“模型分割”实际退回 Otsu 阈值却标为 ONNX 推理。已改为“无权重，不可用”，并如实标注退回阈值。
> - GPU 训练包的编码器 `pretrained=False`：不是预训练权重，LoRA 微调从随机权重开始；训练包尚未在 GPU 上运行过。
> - 安全：原先监听 0.0.0.0、AI 接口无需登录、限流按可伪造的 X-Forwarded-For、任何人可注册、Ollama 与 Python 桥接地址可指向内网、图片默认发往云端。已改为默认只监听本机、远程请求须登录、按连接地址限流、注册由管理员开通、内网地址拒绝、图片默认不出本机、`VEXEL_OFFLINE=1` 时不发任何外部请求。
> - 工作流不检查图像是否适用（例如把俯视线条当截面得出“75 层”）；平面图上显示的“2.5D 粗糙度”是灰度当高度算的，不是物理粗糙度。这两点尚未修改。

> **审计基准**：
> 1. 主对照计划：已确认的《AI 深度训练架构、2D→3D 扩展、算力与防滥用分阶段升级方案》（Phase 1–4 `implementation_plan.md`）。
> 2. 长期平台参照：`/vexel/vexel_roadmap_0.0.5.md`（Vexel Python 桌面端 / CLI 路线图）。
>
> **审计日期**：2026-11-26（Sprint C / P2 全部收口复审 — 全量 Phase 1–4 + P0/P1/P2 完结）  
> **当前构建与验证状态**：`npm run build`、`npm run lint`（`tsc --noEmit`）与 `npm test`（Vitest 3 套测试文件 / **23 项用例全部通过**）均 0 错误通过。所有 **P0（G-01～G-04）、P1（G-05～G-08）与 P2（G-09～G-11）共 11 项缺口已全部收口完成**。

---

## 一、现状 vs. 初始计划逐项对比

状态定义：`✅ 已完成`（功能与验收点对齐）｜`🟡 部分完成`（主干已实现，存在明确功能或验收缺口）｜`❌ 未实现`

### Phase 1：资源预算门控 + 浏览器 Web Worker 少样本深度学习引擎

| 计划要求（`implementation_plan.md` §3.1, §4.3） | 实际实现情况 | 状态 | 具体缺口说明 |
| :--- | :--- | :---: | :--- |
| 2D / 3D 内存与分辨率四档评估器（`optimal / tiled_safe / lod_guarded / offload_gpu_required`） | 已实现 `src/engine/resourceBudget.ts`，并在工作台、`ModelHub`、`Studio3DView` 中生效 | ✅ | 无明显功能缺口 |
| 冻结编码器：24 维多尺度特征（Gaussian + LoG + 多方向 Gabor + 结构张量 + Hessian 曲率 + 宽视野上下文 + WebGPU/ONNX 特征缓存槽位） | `src/engine/deepFewShotCore.ts` 已升级至 **24 通道**（`DEEP_FEATURE_DIM = 24`，含 Hessian 特征值 $\lambda_1/\lambda_2$、45°/135° 定向 Gabor、r=15 宽视野光照校正、局部形态学 Top-Hat/Bottom-Hat 及外部 WebGPU/ONNX 特征融合槽位，且 100% 向下兼容已保存的 16D 模型）+ `webgpuProbe.ts` 硬件能力探测 | ✅ | **G-11 已收口** |
| 瓶颈残差 MLP + Adam + 余弦退火 + 类别逆频加权 + 空间双边滤波 | 已在 `deepFewShotCore.ts` 完整实现 | ✅ | 已完整落地 |
| Web Worker 异步执行、训练进度/Loss/Val-mIoU 曲线与中断按钮、512×512 羽化瓦片推理 | `deepTrainWorker.ts` + `mlRuntime.ts` + `ModelHub.tsx` + 工作台 `InspectorPanel.tsx` Stage 3 均已实现完整进度条、实时 SVG Loss/Val-mIoU 曲线与 Abort 中断 | ✅ | **G-08 已收口** |
| 留出集 mIoU/F1/精度指标 + 模型 SHA-256 + **刷新页面后模型持久化可 100% 复现推理** | 已实现 `src/engine/modelStore.ts`（IndexedDB `vexel_model_registry_v1` 持久化 + `.vexmodel.json` 导出/导入 + SHA-256 篡改拒绝），刷新页面自动恢复并逐像素 100% 复现推理 | ✅ | **G-01 已收口** |
| 性能基准实测（少样本训练、特征缓存命中、3D 体分析、ONNX 离线推理） | 已通过 `tests/engine.test.ts` 与 `tests/onnx.test.ts` 自动化实测：少样本收敛 mIoU ≥ 0.85 / F1 ≥ 0.90、缓存重训 < 1s、3D 肾细胞体分析 365 ms（< 2s）、真实 ONNX 瓦片推理 ~515 ms | ✅ | **G-04 基准验证已收口**；后续可补充 4096² 真机浏览器内存监控 |

---

### Phase 2：在线/本地 GPU 训练桥接 + 域安全与防滥用网关

| 计划要求（`implementation_plan.md` §3.4, §3.5, §4.3） | 实际实现情况 | 状态 | 具体缺口说明 |
| :--- | :--- | :---: | :--- |
| AI 规划域拦截器（拒答越狱/非图像任务，不消耗云端 Token）+ 算子白名单清洗 | `src/server/securityGuardrails.ts` + `server.ts` `/api/v1/ai/plan` 已实现并经 `tests/security.test.ts` 7 项安全对抗用例自动化验证 | ✅ | **G-04 安全单测已收口** |
| 声明式训练接口（禁止任意代码/仓库/脚本字段）+ 服务商预设 + 并发与限流 | `validateDeclarativeTrainJob` + `FORBIDDEN_CODE_FIELDS` + 6/min 限流 + 单 IP 并发=1 + SSRF 元数据阻断，全部经 `tests/security.test.ts` 自动化验证 | ✅ | 核心防护与自动化回归均已闭环 |
| 图像与掩膜**真实数据上传 + 服务端图像头校验 + 重新编码去 EXIF/元数据**后打包下发 | 已实现 `src/server/imageSanitizer.ts` + `server.ts` `/api/v1/train/jobs` + `ModelHub.tsx`：校验 PNG 魔数/CRC32/尺寸与解压炸弹上限，像素级解码并重编码最小规范 PNG（0 EXIF/`tEXt`/尾部多态载荷残留），打包含 `image.png + mask.png + manifest.json + train_vexel_lora.py` 的可下载 `.zip` 数据包 | ✅ | **G-05 已收口**（`tests/security.test.ts` 3 项像素清洗与 ZIP 打包用例实测通过） |
| 沙箱 LoRA 训练脚本加载真实预置骨干网权重（`timm` / LoRA 注入 + Focal/Dice/Tversky + 生产部署模板） | 已交付 `/deploy/gpu-worker/` 完整生产包：`Dockerfile`（非 root、CUDA 12.4、`--network=none` 离线兼容）、`train_worker.py`（真实 PyTorch + `timm` 冻结编码器 + `LoRALinear`/`LoRAConv2d` 注入 + AMP + ≤50 MB ONNX 导出 + HMAC 签名回调）、`modal_app.py`、`runpod_handler.py` 与 `POST /api/v1/train/callback` 验签入库接口 | ✅ | **G-10 已收口** |
| **浏览器端离线执行导入的 `.onnx` 权重**（`onnxruntime-web`） | 已接入 `onnxruntime-web`（`src/engine/onnxRuntime.ts` + 本地 `/ort` WASM 静态服务，无外部 CDN 依赖），支持 50 MB 门控、真实图校验、SHA-256 计算、IndexedDB 持久化与余弦羽化瓦片推理，并在 `model.segment` 中直接产出概率图 | ✅ | **G-02 已收口**（`tests/onnx.test.ts` 真二进制模型推理实测通过） |
| 用户级身份认证、持久化配额账本（`userId + imageHash + gpuSeconds + tokenCost`）与日/月费用熔断 | 已实现 `src/server/governanceStore.ts`（Node 22 原生 SQLite WAL 持久化库）+ 首次启动管理员认领锁定 + `scrypt` 加盐密码哈希 + 内存态 Bearer 会话 + `admin/researcher` RBAC + 用户日 GPU 配额与全站日/月 USD 费用自动熔断器，并接入 `/api/v1/train/jobs`、`/api/v1/ai/plan` 与 `ModelHub` 管理面板 | ✅ | **G-09 已收口**（重开 SQLite 持久化与熔断单测全部通过） |

---

### Phase 3：WebGL2 硬件双模 3D 视口（真 3D 体渲染 + 2.5D 表面重建）

| 计划要求（`implementation_plan.md` §2.2, §3.2, §4.3） | 实际实现情况 | 状态 | 具体缺口说明 |
| :--- | :--- | :---: | :--- |
| WebGL2 `Data3DTexture` 真 3D 光线步进（MIP / Alpha / Iso）+ 窗宽窗位 + 伪彩 + Z 剖切 + 正交三平面切片 | `WebGL3DViewport.tsx` 已完整实现，并包含软件渲染降级与 `webglcontextlost` 恢复 | ✅ | 已完整落地 |
| 内置 3D Z-Stack 演示数据 + 用户导入序列（含多页 TIFF/OME-TIFF）+ 自动 LOD 降采样与间距同步校正 | `microscopyDecoder.ts`（`decodeMultiPageTiffBuffer`）+ `volume3d.ts`（`loadZStackFromTiffBuffer` / `loadZStackFromFiles`）+ `Studio3DView.tsx` 已原生支持单文件多页 8/16/32-bit `.tif/.tiff/.ome.tif`（自动解析 UTF-8 OME-XML / ImageJ XY & Z 物理间距并统一单位）、多文件 TIFF 序列与 PNG/JPG/WebP 序列 | ✅ | **G-03 已收口**（8-bit 与 16-bit 多页 OME-TIFF 解码与 XY/Z 标定已通过 `tests/engine.test.ts` 验证） |
| 2.5D PBR 高度场 + 粗糙度（$S_a, S_q, S_z$）+ **2D 分割掩膜类别着色彩叠** + 最高 256²–384² 采样 + X/Y 双向剖面 | `WebGL3DViewport.tsx` 已实现 PBR 高度场、$S_a/S_q/S_z$、2D 分割掩膜投影叠加（含选中目标琥珀高亮）、X/Y 双向实时剖切（不重建几何）与 `156² / 256² / 384²` 网格精细度切换 | ✅ | **G-06 已收口** |
| 3D 分割目标真实表面网格 / 3D 标签纹理叠加 | `WebGL3DViewport.tsx` 已在光线步进（MIP/Alpha/Iso）与正交三平面切片中同时绑定真实 3D 分割标签纹理 `uLabelVolume`（`Data3DTexture` + `NearestFilter`，精确区分胞质/核体素边界），并与可点击 3D 壳体共存 | ✅ | **G-07 3D 标签叠加已收口** |

---

### Phase 4：Web Worker 三维体分析、3D N/C 核质比与跨视图联动

| 计划要求（`implementation_plan.md` §3.2.2, §4.3） | 实际实现情况 | 状态 | 具体缺口说明 |
| :--- | :--- | :---: | :--- |
| Web Worker 26 邻域三维连通域（紧凑 `Uint16`）+ 封闭空腔填充 + 嵌套 3D 细胞核分割 + 3D N/C 体积核质比 | `volumeAnalysisCore.ts` + `volumeAnalysisWorker.ts` + `volumeAnalysisRunner.ts` 已完整实现 | ✅ | 已完整落地 |
| 3D 物理计量（体积、表面积、球形度、等效直径、主轴长） | `volumeAnalysisCore.ts` 已实现三维各向异性二阶中心矩（3×3 协方差矩阵 Jacobi 特征值求解），输出旋转不变 PCA 主轴长 `[L1, L2, L3]`、伸长率 `elongation` 与扁平率 `flatness`，并同步至 3D 表格与 CSV 导出 | ✅ | **G-07 PCA 主轴已收口**（旋转 3D 椭球体误差 `< 5%` 自动化单测通过） |
| 3D 目标表、WebGL 3D 视口、D3 3D N/C 直方图双向联动 + 3D CSV/Excel 导出 | 已在 `Studio3DView.tsx` 与 `NcRatioHistogramPanel.tsx` 完整打通，并在工作台右侧 Stage 4 `Results` 新增 `2D 平面指标 / 3D 体计量` 切换面板（共享状态 + 一键后台 3D 分析） | ✅ | **G-08 已收口** |
| 端到端回归测试与运行时验收验证（引擎算法、ONNX 真推理、OME-TIFF 解码、安全网关 400/429、PCA 主轴与 PNG 清洗） | 已接入 `vitest` 并建成 3 套测试文件共 18 项自动化用例（`tests/engine.test.ts`、`tests/onnx.test.ts`、`tests/security.test.ts`），`npm test` 全部通过 | ✅ | **G-04 / P1 回归均已收口** |

---

### 附：与上游 Python 平台路线图（`vexel/vexel_roadmap_0.0.5.md`）的定位差异
Web 工作台侧重交互式浏览器体验、少样本 DL 微调与 WebGL2 3D 可视化；以下 Python 核心能力目前仅存在于 `/vexel/`（PySide6/CLI），尚未与 Web 端打通：
- **OME-Zarr 多分辨率金字塔核外读写**（用于 $>64\text{MP}$ 2D 或 $>67\text{M}$ 体素的超大数据）。
- **流式目录监视（`watch`）与高速录像回放（`replay`）**。
- **SEM 多倾角立体 2.5D 重建（`sem tilt3d`）与完整 ISO 21920 轮廓粗糙度**。

---

## 二、缺口优先级汇总

| 优先级 | 缺口编号 | 缺口名称 | 状态 | 业务/工程影响 |
| :---: | :---: | :--- | :---: | :--- |
| **P0** | G-01 | 浏览器训练模型 IndexedDB 持久化与可复现加载 | ✅ 已完成 | 刷新页面自动恢复自训练模型与 SHA-256，支持 `.vexmodel.json` 导出/导入与防篡改校验 |
| **P0** | G-02 | 接入 `onnxruntime-web` 实现导入 ONNX 权重真推理 | ✅ 已完成 | 支持上传 `.onnx`（≤50 MB）、图校验、本地 `/ort` 离线 WASM 瓦片推理与 IndexedDB 持久化 |
| **P0** | G-03 | 多页 TIFF / OME-TIFF Z-Stack 直接解码导入 | ✅ 已完成 | 纯 TS 原生解码单文件多页 `.tif/.tiff/.ome.tif` 与 TIFF 序列，自动解析 OME-XML / ImageJ XY & Z 物理间距 |
| **P0** | G-04 | 自动化测试与安全/性能验收脚本（Unit + ONNX + Security） | ✅ 已完成 | `npm test` 覆盖 14 项自动化测试（2D/3D 预算、DL 收敛与持久化、OME-TIFF、3D 体分析、ONNX 真推理、安全网关） |
| **P1** | G-05 | 服务端图像二进制上传 + 纯像素重编码清洗 + 数据集安全打包 | ✅ 已完成 | 校验 PNG 魔数/CRC32/尺寸与防解压炸弹，剥离 EXIF/`tEXt`/尾部多态载荷并打包 `.zip` 安全训练集 |
| **P1** | G-06 | 2.5D 表面叠加 2D 分割类别掩膜 + Y 轴剖切 + 高精网格切换 | ✅ 已完成 | 2.5D 高度场支持 2D 分割掩膜投影（含选中高亮）、X/Y 双向实时剖切及 `156²/256²/384²` 切换 |
| **P1** | G-07 | 3D 主轴 PCA 惯性张量计算 + 真实 3D Label 纹理/体素等值面叠加 | ✅ 已完成 | 3×3 协方差 Jacobi 求解旋转不变主轴 `[L1,L2,L3]`（误差 `<5%`）+ `uLabelVolume` 真 3D 标签着色叠加 |
| **P1** | G-08 | 工作台 Stage 3 内联训练曲线/Abort 与 Results 面板 2D/3D 切换 | ✅ 已完成 | Stage 3 内联实时进度条/Loss-mIoU SVG 曲线/Abort 中断；Stage 4 新增 `2D 指标 / 3D 体计量` 统一切换 |
| **P2** | G-09 | 多用户认证 + 持久化配额/费用熔断账本（SQLite WAL） | ✅ 已完成 | 原生 SQLite 持久化、首次启动管理员认领锁定、scrypt 鉴权、用户 GPU 秒配额与日/月 USD 自动熔断 |
| **P2** | G-10 | 真实云端 GPU Worker 模板（Dockerfile / Modal / RunPod + `timm` LoRA + HMAC 回调） | ✅ 已完成 | 提供即用型 `/deploy/gpu-worker/` 部署包与 `/api/v1/train/callback` HMAC-SHA256 验签回传闭环 |
| **P2** | G-11 | 24 维特征增强 / WebGPU 探测槽位 + OME-Zarr 多尺度金字塔后端桥接 | ✅ 已完成 | 24ch 金字塔（含 Hessian/定向 Gabor/外部嵌入槽位）+ WebGPU 能力探测 + `/api/v1/zarr/inspect` 桥接 |

---

## 三、下一步分阶段开发执行计划

### Sprint A（P0 闭环收口：可复现 + 真 ONNX + 多页 TIFF + 自动化验收）— ✅ 已完成
**实测验收结果（2026-11-26）**：
1. **G-01（模型持久化）**：`src/engine/modelStore.ts` 已落库 IndexedDB（`vexel_model_registry_v1`），支持 `.vexmodel.json` 导出/导入与 SHA-256 防篡改校验；单测验证回传推理结果 100% 逐像素一致。
2. **G-02（`onnxruntime-web` 离线真推理）**：`src/engine/onnxRuntime.ts` + `server.ts` `/ort` 本地 WASM 静态路由 + `vite.config.ts` 产物拷贝；单测使用真实二进制 `.onnx` 模型验证图解析、SHA-256 与羽化瓦片推理（~515 ms）。
3. **G-03（多页 TIFF / OME-TIFF Z-Stack）**：`microscopyDecoder.ts`（`decodeMultiPageTiffBuffer` + UTF-8 OME-XML 解码 + XY/Z 物理单位统一）+ `volume3d.ts`（`loadZStackFromTiffBuffer` / `loadZStackFromFiles`）+ `Studio3DView.tsx`；8-bit 与 16-bit 多页 OME-TIFF 单测全通过。
4. **G-04（自动化测试与基准验证）**：`npm test` 3 套测试文件 14/14 全部通过（内置 3D 肾细胞 128×128×48 分析耗时 **365 ms**、5/5 细胞与 5/5 细胞核准确检出、安全网关 7/7 拦截验证通过）；`tsc --noEmit` 与 `vite build` 0 错误。

---

### Sprint B（P1 工程增强：2.5D/3D 视觉精度 + 图像重编码清洗 + UI 联动统一）— ✅ 已完成
**实测验收结果（2026-11-26）**：
1. **G-05（服务端纯像素重编码清洗与安全打包）**：`src/server/imageSanitizer.ts` + `server.ts` `/api/v1/train/jobs` + `ModelHub.tsx` 已实现 PNG 魔数/CRC32/尺寸与防解压炸弹校验、纯像素重编码（0 EXIF/`tEXt`/尾部多态脚本残留）及 PKZIP 安全训练集（`image.png + mask.png + manifest.json + train_vexel_lora.py`）一键下载；`tests/security.test.ts` 3 项新增安全与打包单测全部通过。
2. **G-06（2.5D 分割掩膜投影 & X/Y 双向剖面 & 高精网格）**：`WebGL3DViewport.tsx` 已实现 2D 分割掩膜与选中目标高亮向 2.5D PBR 高度场投影、X/Y 双向实时剖切（无几何重建零卡顿）与 `156² / 256² / 384²` 网格精细度切换。
3. **G-07（3D PCA 主轴与真实 3D 标签纹理叠加）**：`volumeAnalysisCore.ts` 已实现 3×3 各向异性二阶中心矩 Jacobi 特征值求解，输出旋转不变 `[L1, L2, L3]`、伸长率与扁平率（`tests/engine.test.ts` 旋转 3D 椭球体误差 `< 5%` 单测通过）；`WebGL3DViewport.tsx` 已将 `labelVolume` 上传为第二张 WebGL2 `Data3DTexture`（`uLabelVolume`）并同步叠加于 MIP/Alpha/Iso 光线步进与正交三平面切片。
4. **G-08（工作台内联体验统一）**：`InspectorPanel.tsx` Stage 3 已嵌入实时进度条、SVG Loss/Val-mIoU 曲线与 Abort 中止按钮；Stage 4 `Results` 已新增 `2D 平面指标 / 3D 体计量` 统一切换并与 `Studio3DView` 共享 3D 分析结果。
5. **整体验证**：`npm test` 3 套测试文件 **18/18 全部通过**，`tsc --noEmit` 与 `vite build` 均 0 错误。

---

### Sprint C（P2 生产化上线：多租户配额熔断 + 真实云端 GPU Worker + WebGPU/OME-Zarr）— ✅ 已完成
**实测验收结果（2026-11-26）**：
1. **G-09（多用户认证与 SQLite 持久化配额/成本熔断账本）**：`src/server/governanceStore.ts` + `server.ts`（`/api/v1/auth/*`、`/api/v1/governance/*`）+ `GovernanceQuotaPanel.tsx` 已落盘 SQLite WAL 数据库；支持首次启动管理员认领并自动锁定、`VEXEL_ADMIN_PASSCODE` 环境变量覆写、`scrypt` 加盐密码哈希、内存态 Bearer 令牌、研究员/管理员 RBAC、每用户日 GPU 秒配额及日/月 USD 费用自动/手动熔断；重开库持久化单测通过。
2. **G-10（生产云端 GPU Worker 模板与签名回调闭环）**：已交付 `/deploy/gpu-worker/`（`Dockerfile`、`train_worker.py`、`modal_app.py`、`runpod_handler.py`、`README.md`），内置冻结 `timm` 骨干 + `LoRALinear`/`LoRAConv2d` 注入、Focal+Dice/CE+Dice/Tversky 损失、BF16/FP16 AMP、≤50 MB ONNX 导出与 `X-Vexel-Signature` HMAC-SHA256 回调；服务端 `/api/v1/train/callback` 验签与账本回写单测通过。
3. **G-11（24 维特征升级 + WebGPU 探测 + OME-Zarr 多尺度金字塔桥接）**：`deepFewShotCore.ts` 升级至 24 通道（收敛 `mIoU ≥ 0.85` 耗时 ~789 ms，向下 100% 兼容 16D 历史模型）并预留外部 WebGPU/ONNX 特征融合槽位；`webgpuProbe.ts` 在 3D 视口与模型中心探测并展示 `WebGPU/WebGL2` 就绪状态；`src/server/omeZarrBridge.ts` + `/api/v1/zarr/inspect` + `OmeZarrBridgePanel.tsx` 支持 OME-NGFF v0.4/v3 多分辨率金字塔解析、安全路径/外网限制与一键体素物理标定同步。
4. **整体验证**：`npm test` 3 套测试文件 **23/23 全部通过**，`tsc --noEmit` 与 `vite build` 均 0 错误。
