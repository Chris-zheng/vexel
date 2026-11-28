# Vexel Scientific Image Analysis Workbench (Web Edition)

智能体原生（Agent-Native）的显微与工业科学图像分析平台：**AI 负责规划与少样本学习，确定性算子负责计量，每个结果可复核、可追溯、可复现**。

> 注：本仓库根目录为 **Vexel Web Workbench**（React 19 + TypeScript + Three.js WebGL2 + D3 + Express 安全网关）；子目录 `/vexel/` 保留了 Vexel Python 桌面端 / CLI 核心工程及其历史路线图（`0.6.0b1` / `vexel_roadmap_0.0.5.md`）。

---

> **状态说明（2026-11-27，并入主仓库 `web/` 时核对）**
> - 计量结果**未经验收**：经过合成图与真实图片验收的计量引擎是 Python 版（仓库根目录）。Web 版的数字只作预览，正式结果以 Python 引擎为准。
> - 模型中心里的内置模型**只有模型卡，没有权重文件**，显示为“无权重，不可用”；工作流里的“模型分割”在没有权重时退回阈值分割，并如实标注。
> - 默认只在本机（127.0.0.1）监听；开放到网络时，其他电脑必须登录才能使用 AI、训练与数据接口。图片默认不发给云端模型（见 `.env.example`）。
> - 工作流与图像不符时，结果页最上方会提示（内置示例按其类型判断；上传图片只按大类判断）。
> - **连接 Python 引擎**：`uv run --extra web --extra sem vexel serve`，在引擎设置里选“本地 Python 引擎”并测试连接；SEM 线条、接触孔、颗粒、纤维、多层、弯曲多层与层级嵌套（细胞核/细胞）工作流的结果页会出现“Python 引擎（已验证）”卡片，给出验收过的数字和与浏览器预览的差异；上传的图片以原始文件（全分辨率、原始位深、仪器元数据）交给 Python，Python 的测量轮廓以虚线画在图上。详见主仓库 `docs/user_manual.md` 11.11。
> - 浏览器的“弯曲多层”方法在合成的倾斜/弯曲层叠上层数与厚度都不对（`tests/benchmarks/web_layer_parity.py`）；多层截面以 Python 结果为准。
> - 上传图片的默认像素尺寸（SEM 无比例尺时 2 nm/px）不是标定，标定来源记为“无”，不会交给 Python 引擎。

## 核心能力一览

1. **端到端 2D 科学工作台（4 阶段闭环）**
   - **标定与裁边**：自动/手动像素标定、比例尺交互对齐、SEM 底部信息栏检测与裁切。
   - **确定性工作流与 AI 规划**：内置细胞/核质比、微流控液滴、SEM 线宽/孔洞/颗粒/纤维/分层、PCB 走线检查等多领域配方；支持 Gemini / Claude / DeepSeek / OpenAI 兼容 / 本地 Ollama 多模型编排及完全离线降级。
   - **人机协同微调（Few-Shot DL + 滚轮阈值点选）**：
     - 点选测量与点选加入 Ground Truth 时，支持**鼠标中键滚轮实时调节灵敏度/容差**以扩大或缩小自动边界。
     - **浏览器 Web Worker 少样本深度微调**：冻结 16 通道多尺度特征金字塔 + 瓶颈残差 MLP + 空间双边滤波，带实时 Loss / Val-mIoU 曲线与中断控制，推理支持 512×512 羽化瓦片流式分块。
     - **在线/本地 GPU LoRA 训练桥接**：仅声明式白名单参数下发，自动生成沙箱化 `train_vexel_lora.py` 与签名清单，并支持本地模型/ONNX 权重卡注册。
   - **复核、计量与可复现导出**：目标表与画布双向联动、低置信高亮与人工纠错版本快照、D3 N/C 核质比直方图、CSV / Excel / 报告导出。

2. **WebGL2 双模 3D 实验室与三维体分析**
   - **真 3D 体渲染（Z-Stack）**：基于 Three.js + WebGL2 `Data3DTexture` 的光线步进（MIP / Alpha 合成 / 等值面）、三平面正交切片（XY / XZ / YZ）、Z 轴剖切与科学伪彩映射；内置 3D 肾细胞-细胞核共聚焦示例栈，并支持导入多张图像序列。
   - **2D 伪 3D 表面重建（2.5D）**：基于 PBR 高度场位移网格、环境光照、X 轴实时剖面与面粗糙度（$S_a, S_q, S_z$）计算。
   - **Web Worker 三维体计量**：26 邻域三维连通域（紧凑 `Uint16` 标签）+ 封闭空腔填充 + 嵌套三维细胞核识别，输出各向异性校准后的物理体积、Crofton 表面积、球形度、等效直径及 **3D N/C 体积核质比**，并与 3D 视口、结果表及 D3 分布图双向联动。

3. **域边界与防滥用安全网关**
   - AI 规划端：滑动窗口限流 + 提示词注入/越狱与非图像任务前置拦截 + 算子白名单清洗。
   - GPU 训练端：禁用任意代码/脚本/仓库 URL 字段，仅允许预设骨干网、LoRA 秩、损失函数和超时上限，并校验图像分辨率、类别数与最小像素数。

---

## 快速开始

### 环境要求
- Node.js 20+
- npm 10+

### 安装与启动

```bash
# 1. 安装依赖
npm install

# 2. 复制环境变量模板（按需填入后端 API Key；无 Key 时自动离线降级）
cp .env.example .env

# 3. 启动全栈开发服务器（Express API + Vite 前端，端口 3000）
npm run dev
```

访问 `http://localhost:3000`。

### 构建与校验

```bash
npm run lint     # TypeScript 类型检查 (tsc --noEmit)
npm run build    # Vite 生产打包
npm start        # 生产模式运行 server.ts
```

---

## 开发文档索引

| 文档 | 内容说明 |
| :--- | :--- |
| [DEVELOPMENT.md](./DEVELOPMENT.md) | 完整架构、模块清单、API 契约、2D/3D 资源预算矩阵、安全防滥用机制与算力配置建议 |
| [docs/GAP_AUDIT_AND_ROADMAP.md](./docs/GAP_AUDIT_AND_ROADMAP.md) | **现状 vs. 初始计划逐项对比、缺口清单（P0/P1/P2）与下一步开发执行计划** |
| [vexel/vexel_roadmap_0.0.5.md](./vexel/vexel_roadmap_0.0.5.md) | Vexel Python 桌面端 / CLI 平台总路线图（V0.0.5） |
| [vexel/CHANGELOG.md](./vexel/CHANGELOG.md) | Vexel Python 核心历史更新日志 |
