import type {
  DetectionObject,
  HumanCorrection,
  OperatorSpec,
  OverlayShape,
  PlanProposal,
  RoiRect,
  RunResult,
  Specimen,
  StepPreview,
  SummaryMetric,
  UnifiedRecipe,
} from '../types/vexel';
import { traceComponentPixelContours } from './contourTrace';
import {
  applyCoherenceEnhance,
  applyLocalContrastNorm,
  detectLowConfidenceSegments,
} from './curvedLayerEnhance';
import { predictLocalModel, type TrainedLocalModel } from './mlRuntime';
import { runMultilayerAnalysis } from './multilayerMetrology';
import {
  runConstrainedSubRegionSegmentation,
  runHierarchicalRatioMetrology,
  runParentRegionSegmentation,
  type ParentSegmentationOutput,
  type SubRegionSegmentationOutput,
} from './hierarchicalSeg';
import { fastSha256Hex } from './samples';
import {
  computeEdgeRefinementField,
  extractSeparatedComponentsWithOcclusion,
  segmentSemBrightEdgeParticles,
  type EdgeRefinementMode,
} from './semParticles';

// ---------------------------------------------------------------------------
// Unified Operator Registry (mirrors Python vexel.ops.registry 1:1)
// ---------------------------------------------------------------------------

export const OPERATOR_CATALOG: OperatorSpec[] = [
  {
    op: 'sem.line_level',
    category: 'preprocess',
    modality: 'sem',
    labelZh: '逐行校平 (Line Level)',
    labelEn: 'Scan-Line Levelling',
    descZh: '消除电镜逐行扫描基线跳变，不改变水平方向几何边缘。',
    descEn: 'Corrects per-row scan-line offset jumps without distorting vertical edges.',
    inputs: ['image'],
    outputs: ['image'],
    params: [
      {
        key: 'strength',
        labelZh: '校平强度',
        labelEn: 'Strength',
        type: 'number',
        defaultValue: 0.85,
        min: 0.1,
        max: 1.0,
        step: 0.05,
      },
    ],
  },
  {
    op: 'sem.destripe',
    category: 'preprocess',
    modality: 'sem',
    labelZh: '去扫描条纹 (Destripe)',
    labelEn: 'Destripe Scan Lines',
    descZh: '抑制沿扫描方向的周期性水平条纹噪声。',
    descEn: 'Attenuates periodic horizontal scan-line stripe artefacts.',
    inputs: ['image'],
    outputs: ['image'],
    params: [
      {
        key: 'strength',
        labelZh: '去条纹强度',
        labelEn: 'Attenuation',
        type: 'number',
        defaultValue: 0.8,
        min: 0.1,
        max: 1.0,
        step: 0.05,
      },
    ],
  },
  {
    op: 'sem.background',
    category: 'preprocess',
    modality: 'shared',
    labelZh: '背景校平 (Background Level)',
    labelEn: 'Background Levelling',
    descZh: '扣除大尺度光照或背底梯度，提升弱对比目标分割稳定性。',
    descEn: 'Removes large-scale uneven illumination gradient across the field of view.',
    inputs: ['image'],
    outputs: ['image'],
    params: [
      {
        key: 'strength',
        labelZh: '梯度扣除比例',
        labelEn: 'Flattening',
        type: 'number',
        defaultValue: 0.9,
        min: 0.2,
        max: 1.0,
        step: 0.05,
      },
    ],
  },
  {
    op: 'sem.denoise',
    category: 'preprocess',
    modality: 'shared',
    labelZh: '高斯去噪 (Denoise)',
    labelEn: 'Gaussian Denoise',
    descZh: '轻度平滑高频散粒噪声（注意：测 LER/LWR 粗糙度前不应开启）。',
    descEn: 'Mild smoothing of shot noise (do not enable before LER/LWR roughness metrology).',
    inputs: ['image'],
    outputs: ['image'],
    params: [
      {
        key: 'sigma',
        labelZh: '平滑半径 σ (px)',
        labelEn: 'Sigma (px)',
        type: 'number',
        defaultValue: 1.0,
        min: 0.5,
        max: 3.0,
        step: 0.25,
        unit: 'px',
      },
    ],
  },
  {
    op: 'sem.local_contrast',
    category: 'preprocess',
    modality: 'shared',
    labelZh: '局部对比度归一化 (LCN/CLAHE)',
    labelEn: 'Local Contrast Normalization (LCN)',
    descZh: '按局部窗口均值与标准差自适应均衡对比度，拉齐暗区与亮区条纹反差，专门解决光照不均与弱对比层片。',
    descEn: 'Adaptive local z-score contrast normalization using integral images to equalize dark and bright low-contrast regions without amplifying flat noise.',
    inputs: ['image'],
    outputs: ['image'],
    params: [
      {
        key: 'windowPx',
        labelZh: '局部窗口尺寸',
        labelEn: 'Local Window Size',
        type: 'number',
        defaultValue: 24,
        min: 8,
        max: 96,
        step: 4,
        unit: 'px',
      },
      {
        key: 'clipLimit',
        labelZh: '对比度限幅 (σ)',
        labelEn: 'Contrast Clip Limit',
        type: 'number',
        defaultValue: 2.5,
        min: 1.2,
        max: 5.0,
        step: 0.2,
      },
      {
        key: 'blend',
        labelZh: '融合强度',
        labelEn: 'Blend Strength',
        type: 'number',
        defaultValue: 0.88,
        min: 0.2,
        max: 1.0,
        step: 0.05,
      },
    ],
  },
  {
    op: 'sem.coherence_enhance',
    category: 'preprocess',
    modality: 'shared',
    labelZh: '各向异性沿层增强 (Coherence Filter)',
    labelEn: 'Anisotropic Coherence Layer Filter',
    descZh: '基于结构张量估计每个像素的弯曲层切线方向，仅沿层切向平滑去噪并续连断裂层，同时在法向保持甚至锐化薄层界面。',
    descEn: 'Structure-tensor oriented streamline filter that denoises & reconnects curved broken lamellae strictly along local tangents while keeping normal boundaries sharp.',
    inputs: ['image'],
    outputs: ['image'],
    params: [
      {
        key: 'alongSigma',
        labelZh: '沿层切向平滑 σ',
        labelEn: 'Along-Layer Smooth σ',
        type: 'number',
        defaultValue: 3.2,
        min: 1.0,
        max: 8.0,
        step: 0.4,
        unit: 'px',
      },
      {
        key: 'tensorSigma',
        labelZh: '结构张量尺度 σ',
        labelEn: 'Structure Tensor σ',
        type: 'number',
        defaultValue: 3.0,
        min: 1.5,
        max: 7.0,
        step: 0.5,
        unit: 'px',
      },
      {
        key: 'iterations',
        labelZh: '迭代增强次数',
        labelEn: 'Iterations',
        type: 'number',
        defaultValue: 2,
        min: 1,
        max: 4,
        step: 1,
      },
      {
        key: 'sharpen',
        labelZh: '法向界面锐化',
        labelEn: 'Normal Edge Sharpen',
        type: 'number',
        defaultValue: 0.45,
        min: 0,
        max: 1.2,
        step: 0.15,
      },
    ],
  },
  {
    op: 'generic.intensity_seg',
    category: 'segment',
    modality: 'shared',
    labelZh: '通用灰度阈值分割 (Intensity Segmentation)',
    labelEn: 'Intensity-Based Segmentation',
    descZh: '面向未识别或通用图像的默认强度阈值分割算子，支持 Otsu 自动、局部自适应与手动阈值，输出连通域轮廓、面积占比与等效直径分布。',
    descEn: 'Default intensity-based segmentation for unrecognized or general structures using Otsu, local adaptive, or manual thresholding with morphological cleanup.',
    inputs: ['image'],
    outputs: ['labels', 'table', 'summary', 'geometry'],
    params: [
      {
        key: 'method',
        labelZh: '阈值算法',
        labelEn: 'Threshold Method',
        type: 'choice',
        defaultValue: 'otsu',
        choices: [
          { value: 'otsu', labelZh: 'Otsu 自动全局阈值', labelEn: 'Otsu Auto Global' },
          { value: 'adaptive', labelZh: '局部自适应阈值', labelEn: 'Local Adaptive Mean' },
          { value: 'manual', labelZh: '手动固定阈值', labelEn: 'Manual Fixed Threshold' },
        ],
      },
      {
        key: 'polarity',
        labelZh: '前景极性',
        labelEn: 'Foreground Polarity',
        type: 'choice',
        defaultValue: 'auto',
        choices: [
          { value: 'auto', labelZh: '自动判别前景极性', labelEn: 'Auto Detect' },
          { value: 'bright', labelZh: '亮目标（暗背底）', labelEn: 'Bright on Dark' },
          { value: 'dark', labelZh: '暗目标（亮背底）', labelEn: 'Dark on Bright' },
        ],
      },
      {
        key: 'thresholdOffset',
        labelZh: '阈值灵敏度偏移',
        labelEn: 'Threshold Offset',
        type: 'number',
        defaultValue: 0,
        min: -80,
        max: 80,
        step: 2,
      },
      {
        key: 'manualThreshold',
        labelZh: '手动灰度阈值 (Manual)',
        labelEn: 'Manual Threshold',
        type: 'number',
        defaultValue: 128,
        min: 5,
        max: 250,
        step: 5,
      },
      {
        key: 'minAreaPx',
        labelZh: '最小目标面积',
        labelEn: 'Min Region Area',
        type: 'number',
        defaultValue: 36,
        min: 8,
        max: 500,
        step: 8,
        unit: 'px²',
      },
      {
        key: 'morphClean',
        labelZh: '3×3 形态学开运算去噪',
        labelEn: '3×3 Morphological Cleanup',
        type: 'boolean',
        defaultValue: true,
      },
    ],
  },
  {
    op: 'model.segment',
    category: 'model',
    modality: 'shared',
    labelZh: 'Trained Model 智能分割',
    labelEn: 'Trained Model Segmentation',
    descZh: '调用模型中心已校验的 ONNX 或本地微调模型执行分块推理，支持遮挡处理策略与边缘曲率/重叠置信度边界拟合优化。',
    descEn: 'Runs a verified ONNX or locally fine-tuned ModelCard with tiled inference, occlusion handling, and curvature + overlap-confidence boundary optimization.',
    inputs: ['image'],
    outputs: ['labels', 'probability'],
    params: [
      {
        key: 'probThreshold',
        labelZh: '置信概率阈值',
        labelEn: 'Probability Threshold',
        type: 'number',
        defaultValue: 0.5,
        min: 0.15,
        max: 0.92,
        step: 0.05,
      },
      {
        key: 'occlusionHandling',
        labelZh: '遮挡处理 (Occlusion)',
        labelEn: 'Occlusion Handling',
        type: 'choice',
        defaultValue: 'dashed',
        choices: [
          { value: 'dashed', labelZh: '以虚线边框显示被遮挡颗粒', labelEn: 'Show Occluded with Dashed Border' },
          { value: 'hide', labelZh: '隐藏被遮挡颗粒', labelEn: 'Hide Occluded Particles' },
        ],
      },
      {
        key: 'curvatureBoundaryFit',
        labelZh: '边缘曲率与重叠置信度边界优化',
        labelEn: 'Curvature & Overlap Boundary Fit',
        type: 'boolean',
        defaultValue: true,
        helpZh: '分析轮廓有符号曲率凹角与邻接重叠置信度，沿亮边梯度亚像素贴合边界。',
        helpEn: 'Analyzes signed edge curvature cusps & overlap confidence to snap overlapping seams onto peak SE gradients.',
      },
      {
        key: 'overlapConfidenceThreshold',
        labelZh: '重叠置信度阈值',
        labelEn: 'Overlap Confidence Threshold',
        type: 'number',
        defaultValue: 0.45,
        min: 0.2,
        max: 0.85,
        step: 0.05,
      },
      {
        key: 'tiled',
        labelZh: '启用 512px 分块推理',
        labelEn: '512px Tiled Inference',
        type: 'boolean',
        defaultValue: true,
      },
    ],
  },
  {
    op: 'droplet.analyze',
    category: 'metrology',
    modality: 'droplet',
    labelZh: '液滴分割与包裹/荧光计量',
    labelEn: 'Droplet Detection, Encapsulation & Fluorescence',
    descZh: '提取液滴粒径 CV、压扁体积、内部细胞泊松包裹率及荧光阳性比例，并输出 QC 可疑列表。',
    descEn: 'Measures droplet diameter CV, flattened volume, Poisson cell encapsulation, and fluorescence positivity.',
    inputs: ['image'],
    outputs: ['table', 'summary', 'geometry'],
    params: [
      {
        key: 'minDiameterPx',
        labelZh: '最小液滴直径',
        labelEn: 'Min Diameter',
        type: 'number',
        defaultValue: 18,
        min: 8,
        max: 120,
        step: 2,
        unit: 'px',
      },
      {
        key: 'countCells',
        labelZh: '统计液滴内细胞数',
        labelEn: 'Count Encapsulated Cells',
        type: 'boolean',
        defaultValue: true,
      },
      {
        key: 'cellContrast',
        labelZh: '细胞最低对比度',
        labelEn: 'Cell Min Contrast',
        type: 'number',
        defaultValue: 36,
        min: 12,
        max: 120,
        step: 2,
      },
      {
        key: 'measureFluo',
        labelZh: '荧光阴阳判读',
        labelEn: 'Fluorescence Positivity',
        type: 'boolean',
        defaultValue: false,
      },
      {
        key: 'fluoThreshold',
        labelZh: '阳性阈值 (0=自动 Otsu)',
        labelEn: 'Positive Threshold (0=Auto)',
        type: 'number',
        defaultValue: 0,
        min: 0,
        max: 240,
        step: 5,
      },
    ],
  },
  {
    op: 'sem.lines',
    category: 'metrology',
    modality: 'sem',
    labelZh: 'SEM 线条 CD 与 LER/LWR 计量',
    labelEn: 'SEM Line/Space CD & LER/LWR',
    descZh: '逐行亚像素提取左右边缘，计算 CD、Pitch 与扣除噪声偏差的 3σ LER/LWR。',
    descEn: 'Sub-pixel per-row edge extraction for CD, pitch, and noise-bias-corrected 3σ LER/LWR.',
    inputs: ['image'],
    outputs: ['table', 'summary', 'geometry'],
    params: [
      {
        key: 'method',
        labelZh: '边缘定义',
        labelEn: 'Edge Definition',
        type: 'choice',
        defaultValue: 'threshold',
        choices: [
          { value: 'threshold', labelZh: '阈值法 (50%)', labelEn: 'Threshold (50%)' },
          { value: 'gradient', labelZh: '最大梯度', labelEn: 'Max Gradient' },
          { value: 'peak', labelZh: '亮边峰值', labelEn: 'Bright Edge Peak' },
        ],
      },
      {
        key: 'level',
        labelZh: '阈值比例',
        labelEn: 'Threshold Level',
        type: 'number',
        defaultValue: 0.5,
        min: 0.2,
        max: 0.85,
        step: 0.05,
      },
      {
        key: 'EdgeRefinement',
        labelZh: '边缘锐化 (EdgeRefinement)',
        labelEn: 'Edge Refinement (Sobel / Canny)',
        type: 'choice',
        defaultValue: 'off',
        choices: [
          { value: 'off', labelZh: '关闭（原始梯度/阈值）', labelEn: 'Off (Raw Profile)' },
          { value: 'sobel_3x3', labelZh: 'Sobel 3×3（标准锐化）', labelEn: 'Sobel 3×3 (Standard)' },
          { value: 'sobel_5x5', labelZh: 'Sobel 5×5（强抗噪低对比）', labelEn: 'Sobel 5×5 (Low-Contrast)' },
          { value: 'canny_mild', labelZh: 'Canny NMS（标准双阈值）', labelEn: 'Canny NMS (Balanced)' },
          { value: 'canny_strong', labelZh: 'Canny NMS（高敏弱边界）', labelEn: 'Canny NMS (High Sensitivity)' },
        ],
        helpZh: '通过 Sobel 3×3/5×5 卷积或 Canny 非极大值抑制与双阈值滞后追踪，在低对比度线条边缘实现亚像素级精确定位。',
        helpEn: 'Uses Sobel 3×3/5×5 or Canny NMS + double-threshold hysteresis for pixel-precise localization on low-contrast line edges.',
      },
    ],
  },
  {
    op: 'sem.holes',
    category: 'metrology',
    modality: 'sem',
    labelZh: 'SEM 接触孔阵列与缺孔检测',
    labelEn: 'SEM Contact Holes & Lattice',
    descZh: '测孔径、圆度、3σ 均匀性、二维阵列周期并自动定位缺失孔。',
    descEn: 'Measures hole diameter, roundness, 3σ uniformity, 2D lattice pitch, and missing vias.',
    inputs: ['image'],
    outputs: ['table', 'summary', 'geometry'],
    params: [
      {
        key: 'level',
        labelZh: '边缘阈值比例',
        labelEn: 'Edge Level',
        type: 'number',
        defaultValue: 0.5,
        min: 0.25,
        max: 0.8,
        step: 0.05,
      },
      {
        key: 'minDiameterPx',
        labelZh: '最小孔径',
        labelEn: 'Min Hole Diameter',
        type: 'number',
        defaultValue: 12,
        min: 4,
        max: 80,
        step: 2,
        unit: 'px',
      },
    ],
  },
  {
    op: 'sem.particles',
    category: 'metrology',
    modality: 'sem',
    labelZh: 'SEM 颗粒分割与形貌分类',
    labelEn: 'SEM Particles & Classification',
    descZh: '支持标准连通域与「SEM 亮边叠加分水岭」，可分离边界发亮的多层堆叠颗粒，自动识别被上层遮挡颗粒并输出 d10/d50/d90 粒度分布。',
    descEn: 'Supports standard or SE bright-edge watershed separation for stacked SEM particles, detects partially occluded lower particles, and computes d10/d50/d90.',
    inputs: ['image'],
    outputs: ['table', 'summary', 'geometry'],
    params: [
      {
        key: 'separationMode',
        labelZh: '边界分离算法',
        labelEn: 'Boundary Separation',
        type: 'choice',
        defaultValue: 'standard',
        choices: [
          { value: 'standard', labelZh: '标准阈值连通域（稀疏颗粒）', labelEn: 'Standard Threshold (Dispersed)' },
          { value: 'sem_edge_watershed', labelZh: 'SEM 亮边叠加分水岭（密集/堆叠）', labelEn: 'SEM Bright-Edge Watershed (Stacked)' },
        ],
      },
      {
        key: 'feature',
        labelZh: '目标亮度',
        labelEn: 'Feature Polarity',
        type: 'choice',
        defaultValue: 'bright',
        choices: [
          { value: 'bright', labelZh: '亮颗粒（暗背底）', labelEn: 'Bright on dark' },
          { value: 'dark', labelZh: '暗孔隙（亮背底）', labelEn: 'Dark on bright' },
        ],
      },
      {
        key: 'minDiameterPx',
        labelZh: '最小颗粒直径',
        labelEn: 'Min Diameter',
        type: 'number',
        defaultValue: 10,
        min: 4,
        max: 80,
        step: 2,
        unit: 'px',
      },
      {
        key: 'edgeBarrierStrength',
        labelZh: 'SEM 亮边屏障权重',
        labelEn: 'Bright-Edge Barrier Weight',
        type: 'number',
        defaultValue: 0.65,
        min: 0.1,
        max: 1.0,
        step: 0.05,
        helpZh: '利用二次电子亮边与梯度作为分水岭坝体，防止上层球体与下层晶体合并。',
        helpEn: 'Uses SE bright rims + gradient as watershed dams to separate stacked upper/lower objects.',
      },
      {
        key: 'EdgeRefinement',
        labelZh: '边缘锐化 (EdgeRefinement)',
        labelEn: 'Edge Refinement (Sobel / Canny)',
        type: 'choice',
        defaultValue: 'off',
        choices: [
          { value: 'off', labelZh: '关闭（默认梯度）', labelEn: 'Off (Default)' },
          { value: 'sobel_3x3', labelZh: 'Sobel 3×3（标准锐化）', labelEn: 'Sobel 3×3 (Standard)' },
          { value: 'sobel_5x5', labelZh: 'Sobel 5×5（强抗噪低对比）', labelEn: 'Sobel 5×5 (Low-Contrast)' },
          { value: 'canny_mild', labelZh: 'Canny NMS（标准双阈值）', labelEn: 'Canny NMS (Balanced)' },
          { value: 'canny_strong', labelZh: 'Canny NMS（高敏弱边界）', labelEn: 'Canny NMS (High Sensitivity)' },
        ],
        helpZh: '强化低对比度颗粒边缘的分水岭坝体，并沿轮廓法向进行 Sobel/Canny 亚像素吸附定位。',
        helpEn: 'Sharpens watershed edge barriers and snaps particle contours along normals onto Sobel/Canny sub-pixel edge ridges.',
      },
      {
        key: 'splitSensitivity',
        labelZh: '叠加拆分灵敏度',
        labelEn: 'Overlap Split Sensitivity',
        type: 'number',
        defaultValue: 0.55,
        min: 0.15,
        max: 0.95,
        step: 0.05,
      },
      {
        key: 'occlusionSensitivity',
        labelZh: '被遮挡颗粒判别灵敏度',
        labelEn: 'Occlusion Detection Sensitivity',
        type: 'number',
        defaultValue: 0.5,
        min: 0.1,
        max: 0.95,
        step: 0.05,
        helpZh: '结合边缘曲率凹角、凸包凹陷接触、邻接压盖与阴影落差判别被上层物体挡住的颗粒。',
        helpEn: 'Combines signed edge curvature cusps, convex-hull concave contact, neighbor overlap, and SE shadow cues.',
      },
      {
        key: 'curvatureBoundaryFit',
        labelZh: '边缘曲率与重叠置信度边界优化',
        labelEn: 'Curvature & Overlap Boundary Fit',
        type: 'boolean',
        defaultValue: true,
      },
      {
        key: 'overlapConfidenceThreshold',
        labelZh: '重叠置信度阈值',
        labelEn: 'Overlap Confidence Threshold',
        type: 'number',
        defaultValue: 0.45,
        min: 0.2,
        max: 0.85,
        step: 0.05,
      },
      {
        key: 'excludeBorder',
        labelZh: '剔除接触图像边缘颗粒',
        labelEn: 'Exclude Border-Touching',
        type: 'boolean',
        defaultValue: true,
      },
      {
        key: 'outlierZ',
        labelZh: '异常离群阈值 (σ)',
        labelEn: 'Outlier Z-Threshold',
        type: 'number',
        defaultValue: 2.2,
        min: 1.5,
        max: 4.5,
        step: 0.1,
      },
    ],
  },
  {
    op: 'sem.fibres',
    category: 'metrology',
    modality: 'sem',
    labelZh: 'SEM 纤维中心线直径与取向',
    labelEn: 'SEM Fibre Diameter & Alignment',
    descZh: '先分割每条直线纤维并拟合中心线方向，再沿每条纤维中心线的法向（垂直方向）定位双侧边缘并计算亚像素直径。',
    descEn: 'Segments each straight fibre, fits its centre-line orientation, and finds sub-pixel edges strictly perpendicular to the fibre axis.',
    inputs: ['image'],
    outputs: ['table', 'summary', 'geometry'],
    params: [
      {
        key: 'feature',
        labelZh: '纤维极性',
        labelEn: 'Fibre Polarity',
        type: 'choice',
        defaultValue: 'bright',
        choices: [
          { value: 'bright', labelZh: '亮纤维（暗背底）', labelEn: 'Bright on dark' },
          { value: 'dark', labelZh: '暗纤维（亮背底）', labelEn: 'Dark on bright' },
        ],
      },
      {
        key: 'thresholdBias',
        labelZh: '分割灵敏度',
        labelEn: 'Threshold Bias',
        type: 'number',
        defaultValue: 0.45,
        min: 0.2,
        max: 0.85,
        step: 0.05,
      },
      {
        key: 'edgeMethod',
        labelZh: '垂直边缘检测法',
        labelEn: 'Perpendicular Edge Method',
        type: 'choice',
        defaultValue: 'gradient',
        choices: [
          { value: 'gradient', labelZh: '法向最大梯度峰值', labelEn: 'Max Normal Gradient' },
          { value: 'threshold', labelZh: '50% 半高阈值插值', labelEn: '50% Threshold Crossing' },
        ],
      },
      {
        key: 'sampleSpacingPx',
        labelZh: '沿中心线采样间距',
        labelEn: 'Centreline Sample Step',
        type: 'number',
        defaultValue: 24,
        min: 10,
        max: 60,
        step: 2,
        unit: 'px',
      },
    ],
  },
  {
    op: 'cells.analyze',
    category: 'metrology',
    modality: 'droplet',
    labelZh: '细胞分割、计数与形态分析',
    labelEn: 'Cell Segmentation & Morphometry',
    descZh: '自动识别明场或荧光细胞轮廓，输出细胞计数、等效直径、面积分布与融合度 (Confluency)。',
    descEn: 'Segments brightfield or fluorescent cells for count, equivalent diameter, area, and confluency.',
    inputs: ['image'],
    outputs: ['table', 'summary', 'geometry'],
    params: [
      {
        key: 'polarity',
        labelZh: '目标极性',
        labelEn: 'Cell Polarity',
        type: 'choice',
        defaultValue: 'auto',
        choices: [
          { value: 'auto', labelZh: '自动判别', labelEn: 'Auto' },
          { value: 'bright', labelZh: '亮细胞（暗背底/荧光）', labelEn: 'Bright on dark' },
          { value: 'dark', labelZh: '暗细胞（明场）', labelEn: 'Dark on bright' },
        ],
      },
      {
        key: 'minDiameterPx',
        labelZh: '最小细胞直径',
        labelEn: 'Min Cell Diameter',
        type: 'number',
        defaultValue: 10,
        min: 4,
        max: 80,
        step: 2,
        unit: 'px',
      },
      {
        key: 'sensitivity',
        labelZh: '分割灵敏度偏移',
        labelEn: 'Threshold Bias',
        type: 'number',
        defaultValue: 0,
        min: -35,
        max: 35,
        step: 5,
      },
    ],
  },
  {
    op: 'pcb.analyze',
    category: 'metrology',
    modality: 'sem',
    labelZh: 'PCB 走线与焊盘计量检测',
    labelEn: 'PCB Trace & Pad Inspection',
    descZh: '分离阻焊层背底并提取铜走线、焊盘及过孔结构，统计焊盘数量、面积、最小/平均线宽与铜覆盖率。',
    descEn: 'Separates solder mask from copper traces/pads to measure pad count, trace width, and copper coverage.',
    inputs: ['image'],
    outputs: ['table', 'summary', 'geometry'],
    params: [
      {
        key: 'minAreaPx',
        labelZh: '最小焊盘/走线面积',
        labelEn: 'Min Feature Area',
        type: 'number',
        defaultValue: 40,
        min: 10,
        max: 400,
        step: 10,
        unit: 'px²',
      },
      {
        key: 'contrastBias',
        labelZh: '铜箔阈值偏移',
        labelEn: 'Copper Threshold Bias',
        type: 'number',
        defaultValue: 0,
        min: -35,
        max: 35,
        step: 5,
      },
    ],
  },
  {
    op: 'sem.multilayer',
    category: 'metrology',
    modality: 'sem',
    labelZh: '多层截面 · 中心层数与层厚测量',
    labelEn: 'Multilayer Stack · Count & Thickness',
    descZh: '自动估计分层方向与微小倾角，在中心测量带做倾角补偿投影并亚像素提取层间界面，输出中心层数、各层厚度、总厚与灰度剖面。',
    descEn: 'Estimates stack tilt via structure tensor, extracts a tilt-compensated center ROI intensity profile, and measures sub-pixel layer count & per-layer thickness.',
    inputs: ['image'],
    outputs: ['table', 'summary', 'geometry'],
    params: [
      {
        key: 'geometryMode',
        labelZh: '层几何形态',
        labelEn: 'Layer Geometry',
        type: 'choice',
        defaultValue: 'straight',
        choices: [
          { value: 'straight', labelZh: '平直/微倾层（倾角补偿直线带）', labelEn: 'Straight / Tilted Layers' },
          { value: 'curved', labelZh: '弯曲/同心弧层（沿局部切线追踪）', labelEn: 'Curved / Arc Layers (Streamline)' },
        ],
      },
      {
        key: 'orientation',
        labelZh: '分层方向',
        labelEn: 'Stack Orientation',
        type: 'choice',
        defaultValue: 'auto',
        choices: [
          { value: 'auto', labelZh: '自动识别（含倾角补偿）', labelEn: 'Auto (with tilt compensation)' },
          { value: 'horizontal', labelZh: '水平层（上下堆叠）', labelEn: 'Horizontal layers (vertical stack)' },
          { value: 'vertical', labelZh: '垂直层（左右堆叠）', labelEn: 'Vertical layers (horizontal stack)' },
        ],
      },
      {
        key: 'detectionMode',
        labelZh: '界面检测模式',
        labelEn: 'Interface Detection Mode',
        type: 'choice',
        defaultValue: 'gradient',
        choices: [
          { value: 'gradient', labelZh: '阶跃界面梯度法（通用多层膜）', labelEn: 'Step Gradient (General Multilayer)' },
          { value: 'alternating', labelZh: '明暗交替法（周期亮暗层）', labelEn: 'Alternating Bright/Dark' },
          { value: 'dark_seams', labelZh: '暗缝分隔法（暗界线/沟槽）', labelEn: 'Dark Seam Valleys' },
          { value: 'bright_seams', labelZh: '亮缝分隔法（亮界线/亮脊）', labelEn: 'Bright Seam Ridges' },
        ],
      },
      {
        key: 'sensitivity',
        labelZh: '界面检测灵敏度',
        labelEn: 'Interface Sensitivity',
        type: 'number',
        defaultValue: 0.48,
        min: 0.1,
        max: 0.95,
        step: 0.05,
      },
      {
        key: 'minLayerPx',
        labelZh: '最小单层厚度',
        labelEn: 'Min Layer Thickness',
        type: 'number',
        defaultValue: 5,
        min: 2,
        max: 80,
        step: 1,
        unit: 'px',
      },
      {
        key: 'smoothSigma',
        labelZh: '剖面平滑 σ',
        labelEn: 'Profile Smooth σ',
        type: 'number',
        defaultValue: 1.25,
        min: 0.5,
        max: 5.0,
        step: 0.25,
        unit: 'px',
      },
      {
        key: 'roiWidthPct',
        labelZh: '中心测量带宽度',
        labelEn: 'Center Band Width',
        type: 'number',
        defaultValue: 36,
        min: 5,
        max: 90,
        step: 5,
        unit: '%',
      },
      {
        key: 'roiCenterOffsetPct',
        labelZh: '中心测量带偏移',
        labelEn: 'Center Band Offset',
        type: 'number',
        defaultValue: 0,
        min: -40,
        max: 40,
        step: 2,
        unit: '%',
      },
      {
        key: 'stackStartPct',
        labelZh: '堆叠起始范围',
        labelEn: 'Stack Start',
        type: 'number',
        defaultValue: 4,
        min: 0,
        max: 85,
        step: 2,
        unit: '%',
      },
      {
        key: 'stackEndPct',
        labelZh: '堆叠结束范围',
        labelEn: 'Stack End',
        type: 'number',
        defaultValue: 96,
        min: 15,
        max: 100,
        step: 2,
        unit: '%',
      },
      {
        key: 'autoTrimExterior',
        labelZh: '仅计中心界面间层（剔除外部基体）',
        labelEn: 'Trim Exterior Background/Substrate',
        type: 'boolean',
        defaultValue: true,
      },
    ],
  },
  {
    op: 'filter.objects',
    category: 'filter',
    modality: 'shared',
    labelZh: '目标属性过滤器 (Filter)',
    labelEn: 'Object Attribute Filter',
    descZh: '按物理尺寸范围与最低圆度自动过滤杂质或噪点目标。',
    descEn: 'Filters detected objects by physical diameter range and minimum circularity.',
    inputs: ['table'],
    outputs: ['table'],
    params: [
      {
        key: 'minDiameterPhys',
        labelZh: '最小物理直径 (0=不限)',
        labelEn: 'Min Physical Diameter',
        type: 'number',
        defaultValue: 0,
        min: 0,
        max: 500,
        step: 5,
      },
      {
        key: 'minCircularity',
        labelZh: '最低圆度 (0=不限)',
        labelEn: 'Min Circularity',
        type: 'number',
        defaultValue: 0,
        min: 0,
        max: 0.95,
        step: 0.05,
      },
    ],
  },
  {
    op: 'region.parent_boundary',
    category: 'segment',
    modality: 'shared',
    labelZh: '一级母区分割 (细胞/外边界)',
    labelEn: 'Stage 1 · Parent Region Boundary',
    descZh: '先分割图像中的一级母区（如肾切片中的闭合暗膜圆形细胞），输出母区实例掩膜 parent_mask 供下游域内子区分割严格约束使用。',
    descEn: 'Segments primary parent regions (e.g., closed dark-membrane cells in tissue sections) and outputs parent_mask to constrain downstream sub-region segmentation.',
    inputs: ['image'],
    outputs: ['parent_mask', 'geometry'],
    params: [
      {
        key: 'mode',
        labelZh: '母区边界模式',
        labelEn: 'Boundary Mode',
        type: 'choice',
        defaultValue: 'dark_membrane',
        choices: [
          { value: 'dark_membrane', labelZh: '闭合暗膜环 (极坐标 DP 贴合)', labelEn: 'Dark Membrane Ring (Polar DP)' },
          { value: 'intensity_blobs', labelZh: '灰度连通域 (Otsu + 填洞)', labelEn: 'Intensity Blobs (Otsu + Fill)' },
          { value: 'upstream_model', labelZh: '上游 AI 模型掩膜', labelEn: 'Upstream Model Mask' },
        ],
      },
      {
        key: 'minRadiusPct',
        labelZh: '最小母区半径占比',
        labelEn: 'Min Radius (% Short Edge)',
        type: 'number',
        defaultValue: 14,
        min: 4,
        max: 38,
        step: 1,
        unit: '%',
      },
      {
        key: 'maxRadiusPct',
        labelZh: '最大母区半径占比',
        labelEn: 'Max Radius (% Short Edge)',
        type: 'number',
        defaultValue: 44,
        min: 10,
        max: 52,
        step: 1,
        unit: '%',
      },
      {
        key: 'membraneSensitivity',
        labelZh: '膜边界检测灵敏度',
        labelEn: 'Membrane Sensitivity',
        type: 'number',
        defaultValue: 0.58,
        min: 0.2,
        max: 0.9,
        step: 0.02,
      },
      {
        key: 'minAreaPx',
        labelZh: '最小母区面积',
        labelEn: 'Min Parent Area',
        type: 'number',
        defaultValue: 800,
        min: 100,
        max: 50000,
        step: 100,
        unit: 'px²',
      },
    ],
  },
  {
    op: 'region.subregion_seg',
    category: 'segment',
    modality: 'shared',
    labelZh: '二级域内子区分割 (内部细胞核)',
    labelEn: 'Stage 2 · Constrained Sub-Region',
    descZh: '严格限定在每个一级 parent_mask 内部执行局部鲁棒统计分割（向内缩进剔除母区膜环干扰），完全屏蔽母区外的暗色杂物/溶酶体。',
    descEn: 'Segments nested sub-regions strictly inside each parent_mask (with inward margin excluding the parent membrane) so exterior dark structures are completely ignored.',
    inputs: ['image', 'parent_mask'],
    outputs: ['subregion_mask', 'parent_mask', 'geometry'],
    params: [
      {
        key: 'polarity',
        labelZh: '子区灰度极性',
        labelEn: 'Sub-Region Polarity',
        type: 'choice',
        defaultValue: 'dark',
        choices: [
          { value: 'dark', labelZh: '暗色子区 (如深染细胞核)', labelEn: 'Dark Sub-Region (e.g. Nucleus)' },
          { value: 'bright', labelZh: '亮色子区 (如荧光/空泡)', labelEn: 'Bright Sub-Region (e.g. Vacuole)' },
        ],
      },
      {
        key: 'contrastZ',
        labelZh: '域内鲁棒对比度阈值 (MAD σ)',
        labelEn: 'Interior Contrast Threshold (Z)',
        type: 'number',
        defaultValue: 2.1,
        min: 0.8,
        max: 4.2,
        step: 0.1,
      },
      {
        key: 'smoothSigma',
        labelZh: '域内纹理平滑半径',
        labelEn: 'Texture Smoothing',
        type: 'number',
        defaultValue: 3.5,
        min: 0.5,
        max: 7.5,
        step: 0.5,
        unit: 'px',
      },
      {
        key: 'borderMarginPx',
        labelZh: '内缩排除膜环宽度',
        labelEn: 'Membrane Exclusion Margin',
        type: 'number',
        defaultValue: 12,
        min: 0,
        max: 36,
        step: 1,
        unit: 'px',
      },
      {
        key: 'minAreaRatioPct',
        labelZh: '最小子区面积占比',
        labelEn: 'Min Area Ratio',
        type: 'number',
        defaultValue: 0.4,
        min: 0.1,
        max: 20,
        step: 0.1,
        unit: '%',
      },
      {
        key: 'maxAreaRatioPct',
        labelZh: '最大子区面积占比',
        labelEn: 'Max Area Ratio',
        type: 'number',
        defaultValue: 65,
        min: 10,
        max: 90,
        step: 5,
        unit: '%',
      },
      {
        key: 'maxSubregionsPerParent',
        labelZh: '单母区保留子区数 (0=全部)',
        labelEn: 'Max Sub-Regions / Parent (0=All)',
        type: 'number',
        defaultValue: 1,
        min: 0,
        max: 10,
        step: 1,
      },
    ],
  },
  {
    op: 'region.hierarchy_ratio',
    category: 'metrology',
    modality: 'shared',
    labelZh: '层级嵌套面积比计量 (Sub/Parent)',
    labelEn: 'Stage 3 · Hierarchical Area Ratio',
    descZh: '按父子归属配对每个母区与内部子区，计算逐细胞核/细胞面积占比 (%)、平均与总面积占比并生成双色分层轮廓叠加。',
    descEn: 'Pairs each parent region with its nested sub-regions to quantify per-cell Sub/Parent area ratio (%), summary stats, and dual-color overlays.',
    inputs: ['parent_mask', 'subregion_mask'],
    outputs: ['table', 'summary', 'geometry'],
    params: [
      {
        key: 'minRatioPct',
        labelZh: '质控最低面积占比',
        labelEn: 'QC Min Ratio',
        type: 'number',
        defaultValue: 0,
        min: 0,
        max: 80,
        step: 1,
        unit: '%',
      },
      {
        key: 'maxRatioPct',
        labelZh: '质控最高面积占比',
        labelEn: 'QC Max Ratio',
        type: 'number',
        defaultValue: 100,
        min: 5,
        max: 100,
        step: 1,
        unit: '%',
      },
    ],
  },
];

export const BUILTIN_RECIPES: UnifiedRecipe[] = [
  {
    schemaVersion: 1,
    id: 'droplet_encapsulation',
    name: 'droplet_encapsulation',
    nameZh: '微流控液滴 · 粒径与细胞包裹统计',
    nameEn: 'Droplets · Size & Cell Encapsulation',
    modality: 'droplet',
    descriptionZh: '自动分割液滴、测粒径 CV 与压扁体积、计数液滴内细胞并拟合泊松分布。',
    descriptionEn: 'Segments droplets, measures diameter CV & volume, counts encapsulated cells with Poisson fit.',
    seed: 0,
    steps: [
      { id: 'bg', op: 'sem.background', enabled: false, params: { strength: 0.8 } },
      {
        id: 'measure',
        op: 'droplet.analyze',
        enabled: true,
        params: { minDiameterPx: 18, countCells: true, cellContrast: 36, measureFluo: false, fluoThreshold: 0 },
      },
      { id: 'filt', op: 'filter.objects', enabled: false, params: { minDiameterPhys: 0, minCircularity: 0 } },
    ],
    qcRules: [
      { id: 'qc_cv', metric: 'diameter_cv', max: 12.0, onFail: 'warn' },
      { id: 'qc_flag', metric: 'flagged_pct', max: 20.0, onFail: 'warn' },
    ],
  },
  {
    schemaVersion: 1,
    id: 'droplet_fluorescence',
    name: 'droplet_fluorescence',
    nameZh: '微流控液滴 · 数字荧光阴阳判读 (ddPCR)',
    nameEn: 'Droplets · Digital Fluorescence (ddPCR)',
    modality: 'droplet',
    descriptionZh: '明场分割液滴轮廓，联动荧光通道执行自动 Otsu 阴阳判读与拷贝数换算。',
    descriptionEn: 'Segments droplets in brightfield and calls positive/negative fluorescence with copies/droplet.',
    seed: 0,
    steps: [
      {
        id: 'measure',
        op: 'droplet.analyze',
        enabled: true,
        params: { minDiameterPx: 18, countCells: false, cellContrast: 36, measureFluo: true, fluoThreshold: 0 },
      },
    ],
    qcRules: [{ id: 'qc_cv', metric: 'diameter_cv', max: 12.0, onFail: 'warn' }],
  },
  {
    schemaVersion: 1,
    id: 'sem_lines',
    name: 'sem_lines',
    nameZh: 'SEM 半导体 · 线条 CD 与 LER/LWR 计量',
    nameEn: 'SEM · Line/Space CD & LER/LWR',
    modality: 'sem',
    descriptionZh: '逐行校平去条纹后提取亚像素线条边缘，输出 CD、Pitch 与扣除噪声偏差的 3σ LER/LWR。',
    descriptionEn: 'Levels scan stripes and extracts sub-pixel line edges for CD, pitch, and unbiased 3σ LER/LWR.',
    seed: 0,
    steps: [
      { id: 'level', op: 'sem.line_level', enabled: true, params: { strength: 0.85 } },
      { id: 'destripe', op: 'sem.destripe', enabled: true, params: { strength: 0.8 } },
      { id: 'measure', op: 'sem.lines', enabled: true, params: { method: 'threshold', level: 0.5, EdgeRefinement: 'off' } },
    ],
    qcRules: [{ id: 'qc_ler', metric: 'ler_3sigma_unbiased', max: 18.0, onFail: 'warn' }],
  },
  {
    schemaVersion: 1,
    id: 'sem_holes',
    name: 'sem_holes',
    nameZh: 'SEM 半导体 · 接触孔阵列与缺孔检测',
    nameEn: 'SEM · Contact Hole Array & Missing Vias',
    modality: 'sem',
    descriptionZh: '背景校平后测接触孔直径、圆度、3σ 均匀性及阵列周期，自动标记缺失孔位置。',
    descriptionEn: 'Levels background, measures contact-hole diameter uniformity & lattice pitch, flags missing vias.',
    seed: 0,
    steps: [
      { id: 'bg', op: 'sem.background', enabled: true, params: { strength: 0.85 } },
      { id: 'measure', op: 'sem.holes', enabled: true, params: { level: 0.5, minDiameterPx: 12 } },
    ],
    qcRules: [{ id: 'qc_miss', metric: 'missing_holes', max: 0, onFail: 'warn' }],
  },
  {
    schemaVersion: 1,
    id: 'sem_particles',
    name: 'sem_particles',
    nameZh: 'SEM 材料 · 颗粒分割、粒度分布与形貌分类',
    nameEn: 'SEM · Particle Size Distribution & Classification',
    modality: 'sem',
    descriptionZh: '背景校平并分割纳米颗粒，计算 d10/d50/d90（数量/体积）并分类正常/团聚/不规则/异物。',
    descriptionEn: 'Segments nanoparticles, computes d10/d50/d90 (number/volume) and classifies morphology.',
    seed: 0,
    steps: [
      { id: 'bg', op: 'sem.background', enabled: true, params: { strength: 0.9 } },
      { id: 'model', op: 'model.segment', enabled: false, modelRef: 'vexel-sem-particles-unet@1.2', params: { probThreshold: 0.5, tiled: true } },
      {
        id: 'measure',
        op: 'sem.particles',
        enabled: true,
        params: {
          separationMode: 'standard',
          feature: 'bright',
          minDiameterPx: 10,
          edgeBarrierStrength: 0.65,
          EdgeRefinement: 'off',
          splitSensitivity: 0.55,
          occlusionSensitivity: 0.5,
          excludeBorder: true,
          outlierZ: 2.2,
        },
      },
      { id: 'filt', op: 'filter.objects', enabled: false, params: { minDiameterPhys: 0, minCircularity: 0 } },
    ],
    qcRules: [{ id: 'qc_out', metric: 'class_outlier', max: 0, onFail: 'warn' }],
  },
  {
    schemaVersion: 1,
    id: 'sem_stacked_particles',
    name: 'sem_stacked_particles',
    nameZh: 'SEM 亮边叠加颗粒 · 边缘屏障分水岭与遮挡分离',
    nameEn: 'SEM · Bright-Edge Stacked Particles & Occlusion',
    modality: 'sem',
    descriptionZh: '专为二次电子边界发亮、多层堆叠压盖的 SEM 颗粒/晶体图像设计：结合亮边梯度坝体与标记分水岭拆分叠加边界，自动检出被上层遮挡颗粒（可在图层 Overlay 隐藏或异色高亮）。',
    descriptionEn: 'Built for bright-edge SE-SEM stacked particles/crystals: bright-rim gradient barrier watershed separation + automatic detection of partially occluded lower particles.',
    seed: 0,
    steps: [
      { id: 'denoise', op: 'sem.denoise', enabled: true, params: { sigma: 0.75 } },
      {
        id: 'measure',
        op: 'sem.particles',
        enabled: true,
        params: {
          separationMode: 'sem_edge_watershed',
          feature: 'bright',
          minDiameterPx: 16,
          edgeBarrierStrength: 0.7,
          EdgeRefinement: 'sobel_3x3',
          splitSensitivity: 0.58,
          occlusionSensitivity: 0.52,
          excludeBorder: false,
          outlierZ: 2.6,
        },
      },
      { id: 'filt', op: 'filter.objects', enabled: false, params: { minDiameterPhys: 0, minCircularity: 0 } },
    ],
    qcRules: [],
  },
  {
    schemaVersion: 1,
    id: 'nested_cell_nucleus',
    name: 'nested_cell_nucleus',
    nameZh: '层级嵌套分割 · 细胞边界 → 内部暗色细胞核与面积占比',
    nameEn: 'Hierarchical · Cell Boundary → Nested Nucleus & Area Ratio',
    modality: 'droplet',
    descriptionZh: '标准三段式父子层级工作流：先由闭合暗膜检测 + 极坐标 DP 提取细胞外边界（parent_mask），再严格在每个细胞域内分割暗色细胞核（subregion_mask，自动屏蔽胞外杂物与膜环），最后输出逐细胞核/细胞面积占比 (%)。',
    descriptionEn: 'Composable 3-stage parent-to-subregion workflow: detects outer cell boundaries (parent_mask), constrains dark nucleus segmentation strictly inside each cell (subregion_mask), and computes per-cell nucleus-to-cell area ratios (%).',
    seed: 0,
    steps: [
      { id: 'denoise', op: 'sem.denoise', enabled: true, params: { sigma: 1.0 } },
      {
        id: 'parent_seg',
        op: 'region.parent_boundary',
        enabled: true,
        params: {
          mode: 'dark_membrane',
          minRadiusPct: 14,
          maxRadiusPct: 44,
          membraneSensitivity: 0.58,
          minAreaPx: 800,
        },
      },
      {
        id: 'sub_seg',
        op: 'region.subregion_seg',
        enabled: true,
        params: {
          polarity: 'dark',
          contrastZ: 2.1,
          smoothSigma: 3.5,
          borderMarginPx: 12,
          minAreaRatioPct: 0.4,
          maxAreaRatioPct: 65,
          maxSubregionsPerParent: 1,
        },
      },
      {
        id: 'ratio',
        op: 'region.hierarchy_ratio',
        enabled: true,
        params: {
          minRatioPct: 0,
          maxRatioPct: 100,
        },
      },
    ],
    qcRules: [],
  },
  {
    schemaVersion: 1,
    id: 'cell_segmentation',
    name: 'cell_segmentation',
    nameZh: '细胞显微 · 细胞分割、计数与融合度分析',
    nameEn: 'Cells · Segmentation, Counting & Confluency',
    modality: 'droplet',
    descriptionZh: '自动区分明场/荧光细胞极性，输出细胞总数、平均直径、圆度分布与视场融合度。',
    descriptionEn: 'Auto-detects brightfield/fluorescence polarity for cell count, size distribution, and confluency.',
    seed: 0,
    steps: [
      { id: 'bg', op: 'sem.background', enabled: true, params: { strength: 0.65 } },
      { id: 'measure', op: 'cells.analyze', enabled: true, params: { polarity: 'auto', minDiameterPx: 10, sensitivity: 0 } },
      { id: 'filt', op: 'filter.objects', enabled: false, params: { minDiameterPhys: 0, minCircularity: 0 } },
    ],
    qcRules: [],
  },
  {
    schemaVersion: 1,
    id: 'pcb_inspection',
    name: 'pcb_inspection',
    nameZh: 'PCB 板检测 · 铜走线线宽与焊盘统计',
    nameEn: 'PCB · Trace Width & Solder Pad Inspection',
    modality: 'sem',
    descriptionZh: '分离阻焊层并检测 PCB 焊盘与铜走线，量化走线宽度、焊盘数量、面积与铜覆盖率。',
    descriptionEn: 'Extracts copper traces and solder pads from PCB imagery for pad count, trace width, and coverage.',
    seed: 0,
    steps: [
      { id: 'bg', op: 'sem.background', enabled: true, params: { strength: 0.5 } },
      { id: 'measure', op: 'pcb.analyze', enabled: true, params: { minAreaPx: 40, contrastBias: 0 } },
    ],
    qcRules: [],
  },
  {
    schemaVersion: 1,
    id: 'sem_fibres',
    name: 'sem_fibres',
    nameZh: 'SEM 材料 · 纤维中心线直径与取向分布',
    nameEn: 'SEM · Fibre Diameter & Alignment',
    modality: 'sem',
    descriptionZh: '分割每条直线纤维并提取中心线方向，沿中心线垂直方向搜索双侧边缘计算直径与取向分布。',
    descriptionEn: 'Segments each straight fibre, fits its centre-line direction, and measures diameters perpendicular to each fibre axis.',
    seed: 0,
    steps: [
      { id: 'bg', op: 'sem.background', enabled: true, params: { strength: 0.7 } },
      {
        id: 'measure',
        op: 'sem.fibres',
        enabled: true,
        params: {
          feature: 'bright',
          thresholdBias: 0.45,
          edgeMethod: 'gradient',
          sampleSpacingPx: 24,
        },
      },
    ],
    qcRules: [],
  },
  {
    schemaVersion: 1,
    id: 'sem_multilayer',
    name: 'sem_multilayer',
    nameZh: '多层截面 · 中心层数与层厚测量',
    nameEn: 'Cross-Section · Center Layer Count & Thickness',
    modality: 'sem',
    descriptionZh: '针对多层薄膜、涂层、超晶格或层状断面图像：自动补偿样本倾角并在中心区域提取亚像素层间界面，统计中心层数、逐层厚度与总厚。',
    descriptionEn: 'Measures center multilayer stack count and per-layer thickness via tilt-compensated ROI projection and sub-pixel interface detection.',
    seed: 0,
    steps: [
      { id: 'bg', op: 'sem.background', enabled: false, params: { strength: 0.6 } },
      { id: 'denoise', op: 'sem.denoise', enabled: true, params: { sigma: 0.8 } },
      {
        id: 'measure',
        op: 'sem.multilayer',
        enabled: true,
        params: {
          geometryMode: 'straight',
          orientation: 'auto',
          detectionMode: 'gradient',
          sensitivity: 0.48,
          minLayerPx: 5,
          smoothSigma: 1.25,
          roiWidthPct: 36,
          roiCenterOffsetPct: 0,
          stackStartPct: 4,
          stackEndPct: 96,
          autoTrimExterior: true,
        },
      },
    ],
    qcRules: [],
  },
  {
    schemaVersion: 1,
    id: 'sem_curved_multilayer',
    name: 'sem_curved_multilayer',
    nameZh: '弯曲/弱对比多层 · 各向异性增强与层厚测量',
    nameEn: 'Curved / Uneven Multilayer · Coherence & Thickness',
    modality: 'sem',
    descriptionZh: '专为非直线、明暗不均、高噪声弱对比层片设计：先执行局部对比度归一化 (LCN) 与结构张量沿层各向异性增强，再沿弯曲层切线追踪界面并辅以 2D FFT 主周期验证。建议配合手动 ROI 避开裂纹与多畴界。',
    descriptionEn: 'Built for curved, low-contrast, noisy lamellae: Local Contrast Normalization + Structure-Tensor Coherence Filter + Curved Iso-Layer Tracking + 2D FFT Period Check.',
    seed: 0,
    steps: [
      { id: 'lcn', op: 'sem.local_contrast', enabled: true, params: { windowPx: 24, clipLimit: 2.6, blend: 0.9 } },
      {
        id: 'coh',
        op: 'sem.coherence_enhance',
        enabled: true,
        params: { alongSigma: 3.6, tensorSigma: 3.2, iterations: 2, sharpen: 0.5 },
      },
      {
        id: 'measure',
        op: 'sem.multilayer',
        enabled: true,
        params: {
          geometryMode: 'curved',
          orientation: 'auto',
          detectionMode: 'alternating',
          sensitivity: 0.56,
          minLayerPx: 3,
          smoothSigma: 1.0,
          roiWidthPct: 24,
          roiCenterOffsetPct: 0,
          stackStartPct: 4,
          stackEndPct: 96,
          autoTrimExterior: true,
        },
      },
    ],
    qcRules: [],
  },
  {
    schemaVersion: 1,
    id: 'generic_intensity_seg',
    name: 'generic_intensity_seg',
    nameZh: '通用灰度阈值分割 · 默认未知特征工作流',
    nameEn: 'Default · Intensity-Based Segmentation',
    modality: 'droplet',
    descriptionZh: '当图像无法匹配为已有结构工作流时自动启用：执行轻度去噪与默认灰度强度阈值分割（Otsu/自适应/手动），输出连通域轮廓、面积占比与等效尺寸统计。',
    descriptionEn: 'Default fallback when an image cannot be recognized as an existing structure workflow: runs intensity-based segmentation (Otsu/adaptive/manual) for region contours, area fraction, and size distribution.',
    seed: 0,
    steps: [
      { id: 'bg', op: 'sem.background', enabled: false, params: { strength: 0.7 } },
      { id: 'denoise', op: 'sem.denoise', enabled: true, params: { sigma: 1.0 } },
      {
        id: 'seg',
        op: 'generic.intensity_seg',
        enabled: true,
        params: {
          method: 'otsu',
          polarity: 'auto',
          thresholdOffset: 0,
          manualThreshold: 128,
          minAreaPx: 36,
          morphClean: true,
        },
      },
      { id: 'filt', op: 'filter.objects', enabled: false, params: { minDiameterPhys: 0, minCircularity: 0 } },
    ],
    qcRules: [],
  },
];

export const DEFAULT_INTENSITY_RECIPE_ID = 'generic_intensity_seg';
export const UNRECOGNIZED_MANUAL_RECIPE_ID = '__unrecognized__';

export const UNRECOGNIZED_MANUAL_RECIPE: UnifiedRecipe = {
  schemaVersion: 1,
  id: UNRECOGNIZED_MANUAL_RECIPE_ID,
  name: 'manual_unrecognized',
  nameZh: '未能识别 — 手工标注 / 手动测量模式',
  nameEn: 'Unrecognized — Manual Annotation / Measurement',
  modality: 'droplet',
  descriptionZh: '当前图像未匹配到内置场景，请使用左侧画笔标注、点选分割或标尺/剖面工具进行手动分析，或从上方下拉菜单选择工作流。',
  descriptionEn: 'Image did not match a built-in specimen class. Annotate or measure manually with the canvas tools, or pick a workflow above.',
  seed: 0,
  steps: [],
  qcRules: [],
};

// ---------------------------------------------------------------------------
// Pixel-Level Preprocessing Kernels
// ---------------------------------------------------------------------------

function applyLineLevel(src: Float32Array, w: number, h: number, cropRows: number, strength: number): Float32Array {
  const out = new Float32Array(src);
  const rowMeds = new Float32Array(cropRows);
  const rowBuf = new Float32Array(w);
  let globalSum = 0;
  for (let y = 0; y < cropRows; y++) {
    for (let x = 0; x < w; x++) rowBuf[x] = src[y * w + x];
    rowBuf.sort();
    // Use 30th percentile (background floor) so vertical bright lines don't bias row baseline
    rowMeds[y] = rowBuf[Math.floor(w * 0.25)];
    globalSum += rowMeds[y];
  }
  const globalMed = globalSum / Math.max(1, cropRows);
  for (let y = 0; y < cropRows; y++) {
    const delta = (rowMeds[y] - globalMed) * strength;
    for (let x = 0; x < w; x++) {
      out[y * w + x] = Math.max(0, Math.min(255, src[y * w + x] - delta));
    }
  }
  return out;
}

function applyDestripe(src: Float32Array, w: number, h: number, cropRows: number, strength: number): Float32Array {
  const out = new Float32Array(src);
  const rowMeans = new Float32Array(cropRows);
  for (let y = 0; y < cropRows; y++) {
    let s = 0;
    for (let x = 0; x < w; x++) s += src[y * w + x];
    rowMeans[y] = s / w;
  }
  // Smooth row means with radius-5 window to isolate high-frequency horizontal stripes
  for (let y = 0; y < cropRows; y++) {
    let local = 0;
    let cnt = 0;
    for (let dy = -5; dy <= 5; dy++) {
      const ny = y + dy;
      if (ny >= 0 && ny < cropRows) {
        local += rowMeans[ny];
        cnt++;
      }
    }
    const stripe = (rowMeans[y] - local / cnt) * strength;
    for (let x = 0; x < w; x++) {
      out[y * w + x] = Math.max(0, Math.min(255, src[y * w + x] - stripe));
    }
  }
  return out;
}

function applyBackgroundLevel(src: Float32Array, w: number, h: number, cropRows: number, strength: number): Float32Array {
  // Fit a 2D plane a*x + b*y + c on downsampled pixels and flatten illumination gradient
  let sx = 0, sy = 0, sv = 0, n = 0;
  for (let y = 0; y < cropRows; y += 4) {
    for (let x = 0; x < w; x += 4) {
      sx += x;
      sy += y;
      sv += src[y * w + x];
      n++;
    }
  }
  const mx = sx / n, my = sy / n, mv = sv / n;
  let sxx = 0, syy = 0, sxv = 0, syv = 0;
  for (let y = 0; y < cropRows; y += 4) {
    for (let x = 0; x < w; x += 4) {
      const dx = x - mx, dy = y - my, dv = src[y * w + x] - mv;
      sxx += dx * dx;
      syy += dy * dy;
      sxv += dx * dv;
      syv += dy * dv;
    }
  }
  const ax = sxv / Math.max(1, sxx);
  const ay = syv / Math.max(1, syy);
  const out = new Float32Array(src);
  for (let y = 0; y < cropRows; y++) {
    for (let x = 0; x < w; x++) {
      const planeDelta = (ax * (x - mx) + ay * (y - my)) * strength;
      out[y * w + x] = Math.max(0, Math.min(255, src[y * w + x] - planeDelta));
    }
  }
  return out;
}

function applyGaussianDenoise(src: Float32Array, w: number, h: number, cropRows: number, sigma: number): Float32Array {
  const rad = Math.max(1, Math.min(5, Math.round(sigma * 2)));
  const kernel: number[] = [];
  let kSum = 0;
  for (let i = -rad; i <= rad; i++) {
    const val = Math.exp(-(i * i) / (2 * sigma * sigma));
    kernel.push(val);
    kSum += val;
  }
  for (let i = 0; i < kernel.length; i++) kernel[i] /= kSum;

  const tmp = new Float32Array(src);
  for (let y = 0; y < cropRows; y++) {
    for (let x = 0; x < w; x++) {
      let acc = 0;
      for (let k = -rad; k <= rad; k++) {
        const cx = Math.max(0, Math.min(w - 1, x + k));
        acc += src[y * w + cx] * kernel[k + rad];
      }
      tmp[y * w + x] = acc;
    }
  }
  const out = new Float32Array(src);
  for (let y = 0; y < cropRows; y++) {
    for (let x = 0; x < w; x++) {
      let acc = 0;
      for (let k = -rad; k <= rad; k++) {
        const cy = Math.max(0, Math.min(cropRows - 1, y + k));
        acc += tmp[cy * w + x] * kernel[k + rad];
      }
      out[y * w + x] = acc;
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// Connected Component Labelling & Shape Descriptors
// ---------------------------------------------------------------------------

interface RawComponent {
  id: number;
  areaPx: number;
  cx: number;
  cy: number;
  minX: number;
  maxX: number;
  minY: number;
  maxY: number;
  eqDiameterPx: number;
  circularity: number;
  solidity: number;
  aspectRatio: number;
  meanGray: number;
  touchingBorder: boolean;
  contour: Array<[number, number]>;
  holeContours: Array<Array<[number, number]>>;
}

function connectedComponents(
  binary: Uint8Array,
  gray: Float32Array,
  w: number,
  cropRows: number,
  minAreaPx: number,
  maxAreaPx = 1e9
): RawComponent[] {
  const labels = new Int32Array(w * cropRows);
  const queue = new Int32Array(w * cropRows);
  const comps: RawComponent[] = [];
  let nextId = 1;

  for (let y = 0; y < cropRows; y++) {
    for (let x = 0; x < w; x++) {
      const start = y * w + x;
      if (!binary[start] || labels[start] !== 0) continue;

      let head = 0;
      let tail = 0;
      queue[tail++] = start;
      labels[start] = nextId;

      let area = 0;
      let sumX = 0, sumY = 0, sumG = 0;
      let minX = x, maxX = x, minY = y, maxY = y;
      let perimeter = 0;
      let touchingBorder = false;

      while (head < tail) {
        const idx = queue[head++];
        const px = idx % w;
        const py = (idx - px) / w;
        area++;
        sumX += px;
        sumY += py;
        sumG += gray[idx];
        if (px < minX) minX = px;
        if (px > maxX) maxX = px;
        if (py < minY) minY = py;
        if (py > maxY) maxY = py;
        if (px <= 1 || px >= w - 2 || py <= 1 || py >= cropRows - 2) touchingBorder = true;

        let edgeCount = 0;
        if (px > 0 && binary[idx - 1]) {
          if (!labels[idx - 1]) { labels[idx - 1] = nextId; queue[tail++] = idx - 1; }
        } else edgeCount++;
        if (px + 1 < w && binary[idx + 1]) {
          if (!labels[idx + 1]) { labels[idx + 1] = nextId; queue[tail++] = idx + 1; }
        } else edgeCount++;
        if (py > 0 && binary[idx - w]) {
          if (!labels[idx - w]) { labels[idx - w] = nextId; queue[tail++] = idx - w; }
        } else edgeCount++;
        if (py + 1 < cropRows && binary[idx + w]) {
          if (!labels[idx + w]) { labels[idx + w] = nextId; queue[tail++] = idx + w; }
        } else edgeCount++;

        if (edgeCount > 0) {
          perimeter += edgeCount === 1 ? 1 : edgeCount === 2 ? 1.414 : 2;
        }
      }

      if (area >= minAreaPx && area <= maxAreaPx) {
        const cx = sumX / area + 0.5;
        const cy = sumY / area + 0.5;
        const eqD = Math.sqrt((4 * area) / Math.PI);
        const circ = Math.min(1, (4 * Math.PI * area) / Math.max(1, perimeter * perimeter));
        const bw = Math.max(1, maxX - minX + 1);
        const bh = Math.max(1, maxY - minY + 1);
        const aspectRatio = Math.max(bw, bh) / Math.min(bw, bh);
        const bboxArea = bw * bh;
        // Estimate solidity comparing fill vs inscribed ellipse fill (π/4 ≈ 0.785)
        const solidity = Math.min(1, area / Math.max(1, bboxArea * 0.785));

        // Trace exact pixel-edge (crack-code) outer boundary & internal holes
        const traced = traceComponentPixelContours(
          labels,
          nextId,
          w,
          cropRows,
          minX,
          maxX,
          minY,
          maxY
        );

        comps.push({
          id: comps.length + 1,
          areaPx: area,
          cx,
          cy,
          minX,
          maxX,
          minY,
          maxY,
          eqDiameterPx: eqD,
          circularity: circ,
          solidity,
          aspectRatio,
          meanGray: sumG / area,
          touchingBorder,
          contour: traced.outer,
          holeContours: traced.holes,
        });
      }
      nextId++;
    }
  }
  return comps;
}

function otsuThreshold(values: number[]): number {
  if (values.length < 2) return 128;
  let minV = Infinity;
  let maxV = -Infinity;
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    if (v < minV) minV = v;
    if (v > maxV) maxV = v;
  }
  if (maxV - minV < 1e-6) return minV;
  const bins = 64;
  const hist = new Float64Array(bins);
  for (let i = 0; i < values.length; i++) {
    const b = Math.min(bins - 1, Math.floor(((values[i] - minV) / (maxV - minV)) * bins));
    hist[b]++;
  }
  const total = values.length;
  let sumAll = 0;
  for (let i = 0; i < bins; i++) sumAll += i * hist[i];
  let sumB = 0, wB = 0, maxBetween = -1, bestBin = bins >> 1;
  for (let i = 0; i < bins; i++) {
    wB += hist[i];
    if (wB === 0) continue;
    const wF = total - wB;
    if (wF === 0) break;
    sumB += i * hist[i];
    const mB = sumB / wB;
    const mF = (sumAll - sumB) / wF;
    const between = wB * wF * (mB - mF) ** 2;
    if (between > maxBetween) {
      maxBetween = between;
      bestBin = i;
    }
  }
  return minV + ((bestBin + 0.5) / bins) * (maxV - minV);
}

function meanStd(arr: number[]): { mean: number; sd: number } {
  if (arr.length === 0) return { mean: 0, sd: 0 };
  const mean = arr.reduce((a, b) => a + b, 0) / arr.length;
  if (arr.length === 1) return { mean, sd: 0 };
  const v = arr.reduce((a, b) => a + (b - mean) ** 2, 0) / (arr.length - 1);
  return { mean, sd: Math.sqrt(v) };
}

function quantile(sorted: number[], q: number): number {
  if (sorted.length === 0) return 0;
  const pos = (sorted.length - 1) * q;
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}

// ---------------------------------------------------------------------------
// Metrology Operators (Microfluidics + SEM)
// ---------------------------------------------------------------------------

function runDropletAnalysis(
  specimen: Specimen,
  gray: Float32Array,
  probMap: Float32Array | undefined,
  params: Record<string, number | boolean | string>,
  corrections: HumanCorrection[],
  probThreshold = 0.5
): {
  objects: DetectionObject[];
  shapes: OverlayShape[];
  summary: SummaryMetric[];
  summaryMap: Record<string, number | string | null>;
  warnings: string[];
  histogram: RunResult['histogram'];
  poissonChart?: RunResult['poissonChart'];
} {
  const { width, height, calibration } = specimen;
  const pxSize = calibration.pixelSize;
  const unit = calibration.spaceUnit;
  const minDiaPx = Number(params.minDiameterPx ?? 18);
  const countCells = Boolean(params.countCells);
  const cellContrast = Number(params.cellContrast ?? 36);
  const measureFluo = Boolean(params.measureFluo) && Boolean(specimen.fluoGray);
  const fluoThreshParam = Number(params.fluoThreshold ?? 0);

  // Detect channel walls (top/bottom dark bands)
  let wallTop = 0;
  let wallBot = height;
  for (let y = 0; y < Math.floor(height * 0.25); y++) {
    let m = 0;
    for (let x = 0; x < width; x++) m += gray[y * width + x];
    if (m / width < 100) wallTop = y + 4;
  }
  for (let y = height - 1; y >= Math.floor(height * 0.75); y--) {
    let m = 0;
    for (let x = 0; x < width; x++) m += gray[y * width + x];
    if (m / width < 100) wallBot = y - 4;
  }

  // Segment dark droplet rings (or use trained model probability map if bound)
  const binary = new Uint8Array(width * height);
  if (probMap) {
    for (let y = wallTop; y < wallBot; y++) {
      for (let x = 0; x < width; x++) {
        if (probMap[y * width + x] >= probThreshold) binary[y * width + x] = 1;
      }
    }
  } else {
    // Compute interior fluid median and threshold dark droplet optical rings
    const samples: number[] = [];
    for (let y = wallTop + 2; y < wallBot - 2; y += 3) {
      for (let x = 0; x < width; x += 4) samples.push(gray[y * width + x]);
    }
    samples.sort((a, b) => a - b);
    const bgLevel = quantile(samples, 0.75);
    const ringThresh = bgLevel - 52;
    for (let y = wallTop + 2; y < wallBot - 2; y++) {
      for (let x = 2; x < width - 2; x++) {
        if (gray[y * width + x] < ringThresh) binary[y * width + x] = 1;
      }
    }
    // Fill droplet interiors via bounding-box scanline fill on connected ring components
  }

  const rawRings = connectedComponents(binary, gray, width, height, Math.PI * (minDiaPx * 0.35) ** 2, width * height * 0.25);
  const filtered = rawRings.filter((c) => {
    const bw = c.maxX - c.minX + 1;
    const bh = c.maxY - c.minY + 1;
    const dia = 0.5 * (bw + bh);
    return dia >= minDiaPx && c.aspectRatio < 1.65 && !c.touchingBorder;
  });

  const objects: DetectionObject[] = [];
  const shapes: OverlayShape[] = [];
  const warnings: string[] = [];

  // Precompute physical diameters to detect size outliers
  const rawDiasPhys = filtered.map((c) => 0.5 * (c.maxX - c.minX + c.maxY - c.minY) * pxSize);
  const { mean: meanRawD, sd: sdRawD } = meanStd(rawDiasPhys);

  // Measure fluorescence per droplet if enabled
  const fluoMeans: number[] = [];
  if (measureFluo && specimen.fluoGray) {
    for (const c of filtered) {
      const r = 0.42 * (c.maxX - c.minX + c.maxY - c.minY) * 0.5;
      let s = 0, cnt = 0;
      for (let y = Math.floor(c.cy - r); y <= Math.ceil(c.cy + r); y++) {
        for (let x = Math.floor(c.cx - r); x <= Math.ceil(c.cx + r); x++) {
          if (x >= 0 && x < width && y >= 0 && y < height && Math.hypot(x - c.cx, y - c.cy) <= r) {
            s += specimen.fluoGray[y * width + x];
            cnt++;
          }
        }
      }
      fluoMeans.push(s / Math.max(1, cnt));
    }
  }
  const autoFluoThresh = fluoThreshParam > 0 ? fluoThreshParam : fluoMeans.length >= 2 ? otsuThreshold(fluoMeans) : 90;

  const corrForSpec = corrections.filter((c) => c.specimenId === specimen.id);

  filtered.forEach((c, idx) => {
    const objId = idx + 1;
    const radiusPx = 0.25 * (c.maxX - c.minX + c.maxY - c.minY);
    const diaPhys = radiusPx * 2 * pxSize;

    // Count dark cells inside droplet interior (radius * 0.68 to avoid dark outer ring)
    let nCells = 0;
    const cellMarks: Array<[number, number]> = [];
    if (countCells) {
      const innerR = Math.max(4, radiusPx * 0.66);
      const innerPixels: number[] = [];
      for (let y = Math.floor(c.cy - innerR); y <= Math.ceil(c.cy + innerR); y++) {
        for (let x = Math.floor(c.cx - innerR); x <= Math.ceil(c.cx + innerR); x++) {
          if (Math.hypot(x - c.cx, y - c.cy) <= innerR) innerPixels.push(gray[y * width + x]);
        }
      }
      innerPixels.sort((a, b) => a - b);
      const interiorMed = quantile(innerPixels, 0.72);
      const cellMask = new Uint8Array(width * height);
      for (let y = Math.floor(c.cy - innerR); y <= Math.ceil(c.cy + innerR); y++) {
        for (let x = Math.floor(c.cx - innerR); x <= Math.ceil(c.cx + innerR); x++) {
          if (Math.hypot(x - c.cx, y - c.cy) <= innerR && interiorMed - gray[y * width + x] >= cellContrast) {
            cellMask[y * width + x] = 1;
          }
        }
      }
      const spots = connectedComponents(cellMask, gray, width, height, 6, Math.PI * innerR * innerR * 0.7);
      nCells = spots.length;
      for (const sp of spots) cellMarks.push([sp.cx, sp.cy]);
    }

    const fluoVal = measureFluo ? fluoMeans[idx] : undefined;
    let isPositive = measureFluo && fluoVal !== undefined ? fluoVal >= autoFluoThresh : undefined;
    let excluded = false;

    // Apply human review corrections deterministically
    for (const corr of corrForSpec) {
      if (corr.objectId !== objId) continue;
      if (corr.field === 'cells') nCells = Number(corr.newValue);
      if (corr.field === 'positive') isPositive = Boolean(corr.newValue);
      if (corr.field === 'excluded') excluded = Boolean(corr.newValue);
    }

    const isSizeOutlier = sdRawD > 0.5 && Math.abs(diaPhys - meanRawD) > 2.0 * sdRawD;
    const isNearThresh = measureFluo && fluoVal !== undefined && Math.abs(fluoVal - autoFluoThresh) < 12;
    const flagged = isSizeOutlier || isNearThresh || (countCells && nCells >= 3);
    const flagReason = isSizeOutlier
      ? '粒径异常 (Size outlier)'
      : isNearThresh
      ? '近阳性阈值 (Near threshold)'
      : countCells && nCells >= 3
      ? '高包裹数建议复核 (High cell count)'
      : undefined;

    objects.push({
      id: objId,
      kind: 'droplet',
      x: c.cx,
      y: c.cy,
      diameter: Number(diaPhys.toFixed(2)),
      area: Number((Math.PI * (diaPhys / 2) ** 2).toFixed(1)),
      circularity: Number(Math.min(0.99, Math.max(0.75, 1 / c.aspectRatio)).toFixed(3)),
      cells: countCells ? nCells : undefined,
      fluoIntensity: fluoVal !== undefined ? Number(fluoVal.toFixed(1)) : undefined,
      positive: isPositive,
      cls: measureFluo ? (isPositive ? 'positive' : 'negative') : 'normal',
      flagged,
      flagReason,
      excluded,
      confidence: flagged ? 0.78 : 0.96,
    });

    if (!excluded) {
      const strokeColor = measureFluo
        ? isPositive
          ? '#10B981'
          : '#64748B'
        : flagged
        ? '#F59E0B'
        : '#06B6D4';
      shapes.push({
        id: `drop_${objId}`,
        kind: 'polygon',
        points: c.contour,
        color: strokeColor,
        label: countCells ? `#${objId} (${nCells})` : measureFluo ? `#${objId} ${isPositive ? '+' : '−'}` : `#${objId}`,
        objectId: objId,
        layer: flagged ? 'flags' : 'segmentation',
      });
      cellMarks.forEach(([mx, my], mIdx) => {
        shapes.push({
          id: `cell_${objId}_${mIdx}`,
          kind: 'circle',
          points: [
            [mx, my],
            [3.8, 3.8],
          ],
          color: '#FACC15',
          objectId: objId,
          layer: 'marks',
        });
      });
    }
  });

  const active = objects.filter((o) => !o.excluded);
  const dias = active.map((o) => o.diameter ?? 0);
  const { mean: dMean, sd: dSd } = meanStd(dias);
  const cv = dMean > 0 ? (dSd / dMean) * 100 : 0;

  // Volume calculation: oblate flattened pancake when D > channelHeight, else sphere (pL = 10^-3 µm³ * 10^-3 -> µm³/1000)
  const hUm = calibration.channelHeightUm;
  const volsPl = dias.map((d) => {
    if (unit !== 'µm') return 0;
    if (hUm && d > hUm) {
      const r = d / 2;
      const flatR = Math.max(0, r - hUm / 2);
      const volUm3 = Math.PI * flatR * flatR * hUm + (Math.PI * Math.PI * flatR * hUm * hUm) / 4 + (Math.PI * hUm ** 3) / 6;
      return volUm3 / 1000;
    }
    return ((Math.PI / 6) * d ** 3) / 1000;
  });
  const volMean = meanStd(volsPl).mean;
  const flaggedCount = active.filter((o) => o.flagged).length;
  const flaggedPct = active.length > 0 ? (flaggedCount / active.length) * 100 : 0;

  const summary: SummaryMetric[] = [
    {
      key: 'n_droplets',
      labelZh: '有效液滴数',
      labelEn: 'Droplets Counted',
      value: active.length,
      unit: '个',
      detailZh: `可疑待复核 ${flaggedCount} 个`,
      detailEn: `${flaggedCount} flagged for review`,
      highlight: true,
    },
    {
      key: 'diameter_mean',
      labelZh: '平均粒径',
      labelEn: 'Mean Diameter',
      value: Number(dMean.toFixed(2)),
      unit,
      detailZh: `SD ${dSd.toFixed(2)} ${unit} · 体积 ${volMean.toFixed(1)} pL`,
      detailEn: `SD ${dSd.toFixed(2)} ${unit} · Vol ${volMean.toFixed(1)} pL`,
      highlight: true,
    },
    {
      key: 'diameter_cv',
      labelZh: '粒径 CV',
      labelEn: 'Diameter CV',
      value: Number(cv.toFixed(2)),
      unit: '%',
      detailZh: cv < 5 ? '高度单分散' : '存在多分散波动',
      detailEn: cv < 5 ? 'Highly monodisperse' : 'Elevated polydispersity',
    },
  ];

  const summaryMap: Record<string, number | string | null> = {
    n_droplets: active.length,
    diameter_mean: Number(dMean.toFixed(3)),
    diameter_sd: Number(dSd.toFixed(3)),
    diameter_cv: Number(cv.toFixed(2)),
    volume_mean_pl: Number(volMean.toFixed(2)),
    flagged_pct: Number(flaggedPct.toFixed(1)),
  };

  let poissonChart: RunResult['poissonChart'];
  if (countCells && active.length > 0) {
    const countsArr = active.map((o) => o.cells ?? 0);
    const lambda = countsArr.reduce((a, b) => a + b, 0) / active.length;
    const emptyPct = (countsArr.filter((c) => c === 0).length / active.length) * 100;
    const singlePct = (countsArr.filter((c) => c === 1).length / active.length) * 100;
    const multiPct = (countsArr.filter((c) => c >= 2).length / active.length) * 100;
    const expEmpty = Math.exp(-lambda) * 100;
    const expSingle = lambda * Math.exp(-lambda) * 100;

    summary.push(
      {
        key: 'lambda',
        labelZh: '平均每滴细胞数 λ',
        labelEn: 'Cells / Droplet (λ)',
        value: Number(lambda.toFixed(3)),
        unit: 'cells/drop',
        detailZh: `单包裹 ${singlePct.toFixed(1)}% (泊松预期 ${expSingle.toFixed(1)}%)`,
        detailEn: `Single ${singlePct.toFixed(1)}% (Poisson ${expSingle.toFixed(1)}%)`,
        highlight: true,
      },
      {
        key: 'frac_empty',
        labelZh: '空滴比例',
        labelEn: 'Empty Fraction',
        value: Number(emptyPct.toFixed(1)),
        unit: '%',
        detailZh: `泊松预期 ${expEmpty.toFixed(1)}%`,
        detailEn: `Poisson ${expEmpty.toFixed(1)}%`,
      },
      {
        key: 'frac_multi',
        labelZh: '多包裹比例 (≥2)',
        labelEn: 'Multi-Cell (≥2)',
        value: Number(multiPct.toFixed(1)),
        unit: '%',
      }
    );
    summaryMap.lambda = Number(lambda.toFixed(4));
    summaryMap.frac_single = Number((singlePct / 100).toFixed(4));

    const maxK = Math.max(4, ...countsArr);
    const ks: number[] = [];
    const obs: number[] = [];
    const exp: number[] = [];
    let fact = 1;
    for (let k = 0; k <= maxK; k++) {
      if (k > 1) fact *= k;
      ks.push(k);
      obs.push(Number(((countsArr.filter((c) => c === k).length / active.length) * 100).toFixed(1)));
      exp.push(Number((((Math.pow(lambda, k) * Math.exp(-lambda)) / fact) * 100).toFixed(1)));
    }
    poissonChart = { counts: ks, observedPct: obs, expectedPct: exp, lambda };
  }

  if (measureFluo && active.length > 0) {
    const posCount = active.filter((o) => o.positive).length;
    const posFrac = posCount / active.length;
    const copiesPerDrop = posFrac < 0.999 ? -Math.log(1 - posFrac) : 6.9;
    summary.push(
      {
        key: 'positive_fraction',
        labelZh: '荧光阳性比例',
        labelEn: 'Positive Fraction',
        value: Number((posFrac * 100).toFixed(1)),
        unit: '%',
        detailZh: `阳性 ${posCount} / ${active.length} · 阈值 ${autoFluoThresh.toFixed(1)}`,
        detailEn: `${posCount} / ${active.length} positive · Thresh ${autoFluoThresh.toFixed(1)}`,
        highlight: true,
      },
      {
        key: 'copies_per_droplet',
        labelZh: '每滴拷贝数 (CPD)',
        labelEn: 'Copies / Droplet',
        value: Number(copiesPerDrop.toFixed(3)),
        unit: 'copies',
      }
    );
    summaryMap.positive_fraction = Number(posFrac.toFixed(4));
    summaryMap.copies_per_droplet = Number(copiesPerDrop.toFixed(4));
  }

  if (!calibration.channelHeightUm) {
    warnings.push('未填写流道高度，体积按球形换算；若液滴被流道压扁请补充流道高度。');
  }
  if (flaggedCount > 0) {
    warnings.push(`检出 ${flaggedCount} 个可疑液滴（粒径异常或高包裹数），建议在“复核与纠错”标签页核查。`);
  }

  return {
    objects,
    shapes,
    summary,
    summaryMap,
    warnings,
    histogram: {
      labelZh: '液滴粒径分布',
      labelEn: 'Droplet Diameter Distribution',
      unit,
      values: dias,
      mean: dMean,
    },
    poissonChart,
  };
}

function runSemLinesAnalysis(
  specimen: Specimen,
  gray: Float32Array,
  params: Record<string, number | boolean | string>,
  hasDenoise: boolean
): {
  objects: DetectionObject[];
  shapes: OverlayShape[];
  summary: SummaryMetric[];
  summaryMap: Record<string, number | string | null>;
  warnings: string[];
  profileChart: RunResult['profileChart'];
  histogram: RunResult['histogram'];
} {
  const { width, height, calibration } = specimen;
  const cropRows = calibration.cropRows ?? height;
  const pxSize = calibration.pixelSize;
  const unit = calibration.spaceUnit;
  const level = Number(params.level ?? 0.5);
  const method = String(params.method ?? 'threshold');
  const edgeRefinement = String(params.EdgeRefinement ?? 'off') as EdgeRefinementMode;
  const warnings: string[] = [];

  if (hasDenoise) {
    warnings.push('工作流在粗糙度测量前启用了去噪算子：可能人为压低 LER/LWR 数值，建议关闭 sem.denoise。');
  }

  const edgeField =
    edgeRefinement !== 'off'
      ? computeEdgeRefinementField(gray, width, cropRows, edgeRefinement)
      : null;

  const parabolicSubpixel = (fPrev: number, fCurr: number, fNext: number, xCenter: number): number => {
    const denom = fPrev - 2 * fCurr + fNext;
    if (denom < -1e-6) {
      const delta = Math.max(-0.5, Math.min(0.5, (0.5 * (fPrev - fNext)) / denom));
      return xCenter + delta;
    }
    return xCenter;
  };

  // Average column profile to find line intervals
  const colProfile = new Float64Array(width);
  for (let y = 10; y < cropRows - 10; y++) {
    for (let x = 0; x < width; x++) colProfile[x] += gray[y * width + x];
  }
  const nRows = Math.max(1, cropRows - 20);
  let pMin = Infinity, pMax = -Infinity;
  for (let x = 0; x < width; x++) {
    colProfile[x] /= nRows;
    if (colProfile[x] < pMin) pMin = colProfile[x];
    if (colProfile[x] > pMax) pMax = colProfile[x];
  }
  const dynRange = Math.max(8, pMax - pMin);
  const thresh = pMin + dynRange * level;

  // Locate rising/falling edge pairs
  const intervals: Array<{ xL: number; xR: number }> = [];
  let inLine = colProfile[0] >= thresh;
  let curStart = 0;
  for (let x = 1; x < width; x++) {
    const high = colProfile[x] >= thresh;
    if (!inLine && high) {
      curStart = x;
      inLine = true;
    } else if (inLine && !high) {
      if (x - curStart >= 8) intervals.push({ xL: curStart, xR: x });
      inLine = false;
    }
  }

  const objects: DetectionObject[] = [];
  const shapes: OverlayShape[] = [];
  const cds: number[] = [];
  const lersRaw: number[] = [];
  const lersUnbiased: number[] = [];
  const lwrsUnbiased: number[] = [];
  const centers: number[] = [];

  const yStep = Math.max(2, Math.floor((cropRows - 24) / 70));
  const isCanny = edgeRefinement === 'canny_mild' || edgeRefinement === 'canny_strong';

  intervals.forEach((iv, idx) => {
    const lineId = idx + 1;
    const leftPts: Array<[number, number]> = [];
    const rightPts: Array<[number, number]> = [];
    const xsL: number[] = [];
    const xsR: number[] = [];
    const ws: number[] = [];

    for (let y = 12; y < cropRows - 12; y += yStep) {
      const rowOff = y * width;
      const searchL0 = Math.max(2, iv.xL - 8);
      const searchL1 = Math.min(width - 3, iv.xL + 8);
      const searchR0 = Math.max(2, iv.xR - 8);
      const searchR1 = Math.min(width - 3, iv.xR + 8);

      // Compute refined Sobel/Canny sub-pixel candidate when EdgeRefinement is enabled
      let refinedLx: number | null = null;
      let refinedRx: number | null = null;
      if (edgeField) {
        let bestLx = -1;
        let bestLScore = -1;
        let bestRx = -1;
        let bestRScore = -1;

        if (isCanny) {
          for (let x = searchL0; x <= searchL1; x++) {
            const i = rowOff + x;
            if (edgeField.cannyMask[i] > 0.5 && edgeField.gx[i] > 0) {
              const score = edgeField.edgeResponse[i] * (0.65 + 0.35 * (edgeField.gx[i] / (edgeField.mag[i] + 1e-4)));
              if (score > bestLScore) {
                bestLScore = score;
                bestLx = x;
              }
            }
          }
          for (let x = searchR0; x <= searchR1; x++) {
            const i = rowOff + x;
            if (edgeField.cannyMask[i] > 0.5 && edgeField.gx[i] < 0) {
              const score = edgeField.edgeResponse[i] * (0.65 + 0.35 * (-edgeField.gx[i] / (edgeField.mag[i] + 1e-4)));
              if (score > bestRScore) {
                bestRScore = score;
                bestRx = x;
              }
            }
          }
        }

        if (bestLx < 0) {
          let maxGx = 0;
          for (let x = searchL0; x <= searchL1; x++) {
            const gxVal = edgeField.gx[rowOff + x];
            if (gxVal > maxGx) {
              maxGx = gxVal;
              bestLx = x;
            }
          }
        }
        if (bestRx < 0) {
          let maxNegGx = 0;
          for (let x = searchR0; x <= searchR1; x++) {
            const negGxVal = -edgeField.gx[rowOff + x];
            if (negGxVal > maxNegGx) {
              maxNegGx = negGxVal;
              bestRx = x;
            }
          }
        }

        if (bestLx >= 1 && bestLx < width - 1) {
          refinedLx = parabolicSubpixel(
            edgeField.gx[rowOff + bestLx - 1],
            edgeField.gx[rowOff + bestLx],
            edgeField.gx[rowOff + bestLx + 1],
            bestLx
          );
        }
        if (bestRx >= 1 && bestRx < width - 1) {
          refinedRx = parabolicSubpixel(
            -edgeField.gx[rowOff + bestRx - 1],
            -edgeField.gx[rowOff + bestRx],
            -edgeField.gx[rowOff + bestRx + 1],
            bestRx
          );
        }
      }

      let lx = iv.xL;
      let rx = iv.xR;
      if (method === 'gradient') {
        if (refinedLx !== null && refinedRx !== null) {
          lx = refinedLx;
          rx = refinedRx;
        } else {
          let maxG = -1;
          for (let x = searchL0; x <= searchL1; x++) {
            const g = gray[rowOff + x + 1] - gray[rowOff + x - 1];
            if (g > maxG) { maxG = g; lx = x; }
          }
          let minG = 1;
          for (let x = searchR0; x <= searchR1; x++) {
            const g = gray[rowOff + x + 1] - gray[rowOff + x - 1];
            if (g < minG) { minG = g; rx = x; }
          }
        }
      } else if (method === 'peak') {
        let bestL = -1;
        let bestLxInt = iv.xL;
        for (let x = searchL0; x <= searchL1; x++) {
          const boost = edgeField ? edgeField.edgeResponse[rowOff + x] * dynRange * 0.25 : 0;
          const v = gray[rowOff + x] + boost;
          if (v > bestL) { bestL = v; bestLxInt = x; }
        }
        let bestR = -1;
        let bestRxInt = iv.xR;
        for (let x = searchR0; x <= searchR1; x++) {
          const boost = edgeField ? edgeField.edgeResponse[rowOff + x] * dynRange * 0.25 : 0;
          const v = gray[rowOff + x] + boost;
          if (v > bestR) { bestR = v; bestRxInt = x; }
        }
        if (edgeField && bestLxInt >= 1 && bestLxInt < width - 1) {
          const sL = (x: number) => gray[rowOff + x] + edgeField.edgeResponse[rowOff + x] * dynRange * 0.25;
          lx = parabolicSubpixel(sL(bestLxInt - 1), sL(bestLxInt), sL(bestLxInt + 1), bestLxInt);
        } else {
          lx = bestLxInt;
        }
        if (edgeField && bestRxInt >= 1 && bestRxInt < width - 1) {
          const sR = (x: number) => gray[rowOff + x] + edgeField.edgeResponse[rowOff + x] * dynRange * 0.25;
          rx = parabolicSubpixel(sR(bestRxInt - 1), sR(bestRxInt), sR(bestRxInt + 1), bestRxInt);
        } else {
          rx = bestRxInt;
        }
      } else {
        // Sub-pixel threshold crossing (with adaptive low-contrast Sobel/Canny refinement)
        let foundL = false;
        let slopeL = dynRange;
        for (let x = searchL0; x < searchL1; x++) {
          const v0 = gray[rowOff + x];
          const v1 = gray[rowOff + x + 1];
          if (v0 <= thresh && v1 >= thresh && v1 > v0) {
            lx = x + (thresh - v0) / (v1 - v0);
            slopeL = v1 - v0;
            foundL = true;
            break;
          }
        }
        let foundR = false;
        let slopeR = dynRange;
        for (let x = searchR0; x < searchR1; x++) {
          const v0 = gray[rowOff + x];
          const v1 = gray[rowOff + x + 1];
          if (v0 >= thresh && v1 <= thresh && v0 > v1) {
            rx = x + (v0 - thresh) / (v0 - v1);
            slopeR = v0 - v1;
            foundR = true;
            break;
          }
        }
        if (edgeField) {
          if (refinedLx !== null) {
            if (!foundL) {
              lx = refinedLx;
            } else if (Math.abs(refinedLx - lx) <= 3.5) {
              const lowContrastFactor = Math.max(0, Math.min(1, 1 - slopeL / (dynRange * 0.35)));
              const wRefine = (isCanny ? 0.42 : 0.32) + lowContrastFactor * 0.42;
              lx = lx * (1 - wRefine) + refinedLx * wRefine;
            }
          }
          if (refinedRx !== null) {
            if (!foundR) {
              rx = refinedRx;
            } else if (Math.abs(refinedRx - rx) <= 3.5) {
              const lowContrastFactor = Math.max(0, Math.min(1, 1 - slopeR / (dynRange * 0.35)));
              const wRefine = (isCanny ? 0.42 : 0.32) + lowContrastFactor * 0.42;
              rx = rx * (1 - wRefine) + refinedRx * wRefine;
            }
          }
        }
      }

      if (rx > lx + 4) {
        xsL.push(lx);
        xsR.push(rx);
        ws.push(rx - lx);
        leftPts.push([lx, y]);
        rightPts.push([rx, y]);
      }
    }

    if (ws.length >= 6) {
      const cdPx = meanStd(ws).mean;
      const sdL = meanStd(xsL).sd;
      const sdR = meanStd(xsR).sd;
      const sdW = meanStd(ws).sd;
      // Estimate high-frequency noise variance via adjacent-row difference variance / 2
      let diffL2 = 0, diffR2 = 0;
      for (let i = 1; i < xsL.length; i++) {
        diffL2 += (xsL[i] - xsL[i - 1]) ** 2;
        diffR2 += (xsR[i] - xsR[i - 1]) ** 2;
      }
      const noiseVarL = diffL2 / (2 * Math.max(1, xsL.length - 1));
      const noiseVarR = diffR2 / (2 * Math.max(1, xsR.length - 1));
      const unbiasedSdL = Math.sqrt(Math.max(0, sdL * sdL - noiseVarL * 0.55));
      const unbiasedSdR = Math.sqrt(Math.max(0, sdR * sdR - noiseVarR * 0.55));
      const unbiasedSdW = Math.sqrt(Math.max(0, sdW * sdW - (noiseVarL + noiseVarR) * 0.45));

      const lerRaw3 = 3 * 0.5 * (sdL + sdR) * pxSize;
      const lerUnb3 = 3 * 0.5 * (unbiasedSdL + unbiasedSdR) * pxSize;
      const lwrUnb3 = 3 * unbiasedSdW * pxSize;
      const cdPhys = cdPx * pxSize;
      const cx = 0.5 * (meanStd(xsL).mean + meanStd(xsR).mean);

      cds.push(cdPhys);
      lersRaw.push(lerRaw3);
      lersUnbiased.push(lerUnb3);
      lwrsUnbiased.push(lwrUnb3);
      centers.push(cx);

      objects.push({
        id: lineId,
        kind: 'line',
        x: cx,
        y: cropRows * 0.5,
        width: Number(cdPhys.toFixed(2)),
        ler3Sigma: Number(lerUnb3.toFixed(2)),
        lwr3Sigma: Number(lwrUnb3.toFixed(2)),
        cls: 'normal',
        confidence: 0.97,
      });

      shapes.push(
        {
          id: `line_l_${lineId}`,
          kind: 'polyline',
          points: leftPts,
          color: '#06B6D4',
          label: `L${lineId}`,
          objectId: lineId,
          layer: 'edges',
        },
        {
          id: `line_r_${lineId}`,
          kind: 'polyline',
          points: rightPts,
          color: '#F472B6',
          objectId: lineId,
          layer: 'edges',
        }
      );
    }
  });

  const pitches: number[] = [];
  for (let i = 1; i < centers.length; i++) pitches.push((centers[i] - centers[i - 1]) * pxSize);
  const cdStats = meanStd(cds);
  const pitchMean = meanStd(pitches).mean;
  const lerRawMean = meanStd(lersRaw).mean;
  const lerUnbMean = meanStd(lersUnbiased).mean;
  const lwrUnbMean = meanStd(lwrsUnbiased).mean;

  const summary: SummaryMetric[] = [
    {
      key: 'cd_mean',
      labelZh: '平均线宽 CD',
      labelEn: 'Mean Line CD',
      value: Number(cdStats.mean.toFixed(2)),
      unit,
      detailZh: `线间 SD ${cdStats.sd.toFixed(2)} ${unit} · 共 ${cds.length} 条线`,
      detailEn: `SD ${cdStats.sd.toFixed(2)} ${unit} · ${cds.length} lines`,
      highlight: true,
    },
    {
      key: 'pitch',
      labelZh: '阵列周期 Pitch',
      labelEn: 'Pitch',
      value: Number(pitchMean.toFixed(2)),
      unit,
      detailZh: `间隔 ${(Math.max(0, pitchMean - cdStats.mean)).toFixed(2)} ${unit}`,
      detailEn: `Space ${(Math.max(0, pitchMean - cdStats.mean)).toFixed(2)} ${unit}`,
      highlight: true,
    },
    {
      key: 'ler_3sigma_unbiased',
      labelZh: 'LER 3σ（扣除噪声）',
      labelEn: 'LER 3σ (Unbiased)',
      value: Number(lerUnbMean.toFixed(2)),
      unit,
      detailZh: `含噪声原始值 ${lerRawMean.toFixed(2)} ${unit}`,
      detailEn: `Biased raw ${lerRawMean.toFixed(2)} ${unit}`,
      highlight: true,
    },
    {
      key: 'lwr_3sigma_unbiased',
      labelZh: 'LWR 3σ（扣除噪声）',
      labelEn: 'LWR 3σ (Unbiased)',
      value: Number(lwrUnbMean.toFixed(2)),
      unit,
      detailZh: `边缘定义: ${method}${edgeRefinement !== 'off' ? ` · 锐化: ${edgeRefinement}` : ''}`,
      detailEn: `Edge method: ${method}${edgeRefinement !== 'off' ? ` · Refine: ${edgeRefinement}` : ''}`,
    },
  ];

  const profilePositions: number[] = [];
  const profileInts: number[] = [];
  for (let x = 0; x < width; x += 2) {
    profilePositions.push(Number((x * pxSize).toFixed(1)));
    profileInts.push(Number(colProfile[x].toFixed(1)));
  }
  const edgePositions = intervals.flatMap((iv) => [iv.xL * pxSize, iv.xR * pxSize]);

  return {
    objects,
    shapes,
    summary,
    summaryMap: {
      lines: cds.length,
      cd_mean: Number(cdStats.mean.toFixed(3)),
      cd_sd: Number(cdStats.sd.toFixed(3)),
      pitch: Number(pitchMean.toFixed(3)),
      ler_3sigma_unbiased: Number(lerUnbMean.toFixed(3)),
      lwr_3sigma_unbiased: Number(lwrUnbMean.toFixed(3)),
      edge_refinement: edgeRefinement,
    },
    warnings,
    profileChart: { positions: profilePositions, intensities: profileInts, edges: edgePositions, unit },
    histogram: {
      labelZh: '各线 CD 分布',
      labelEn: 'Per-Line CD Distribution',
      unit,
      values: cds,
      mean: cdStats.mean,
    },
  };
}

function runSemHolesAnalysis(
  specimen: Specimen,
  gray: Float32Array,
  params: Record<string, number | boolean | string>,
  probMap?: Float32Array,
  probThreshold = 0.5
): {
  objects: DetectionObject[];
  shapes: OverlayShape[];
  summary: SummaryMetric[];
  summaryMap: Record<string, number | string | null>;
  warnings: string[];
  histogram: RunResult['histogram'];
} {
  const { width, height, calibration } = specimen;
  const cropRows = calibration.cropRows ?? height;
  const pxSize = calibration.pixelSize;
  const unit = calibration.spaceUnit;
  const level = Number(params.level ?? 0.5);
  const minDiaPx = Number(params.minDiameterPx ?? 12);

  const binary = new Uint8Array(width * cropRows);
  if (probMap) {
    for (let i = 0; i < width * cropRows; i++) {
      if (probMap[i] >= probThreshold) binary[i] = 1;
    }
  } else {
    const vals: number[] = [];
    for (let y = 0; y < cropRows; y += 3) for (let x = 0; x < width; x += 3) vals.push(gray[y * width + x]);
    vals.sort((a, b) => a - b);
    const lo = quantile(vals, 0.08);
    const hi = quantile(vals, 0.85);
    const thresh = lo + (hi - lo) * level;

    for (let i = 0; i < width * cropRows; i++) {
      if (gray[i] < thresh) binary[i] = 1;
    }
  }

  const comps = connectedComponents(binary, gray, width, cropRows, Math.PI * (minDiaPx / 2) ** 2, width * cropRows * 0.08).filter(
    (c) => !c.touchingBorder && c.eqDiameterPx >= minDiaPx
  );

  const objects: DetectionObject[] = [];
  const shapes: OverlayShape[] = [];
  const dias: number[] = [];

  comps.forEach((c, idx) => {
    const id = idx + 1;
    const diaPhys = c.eqDiameterPx * pxSize;
    dias.push(diaPhys);
    const merged = c.aspectRatio > 1.38;
    objects.push({
      id,
      kind: 'hole',
      x: c.cx,
      y: c.cy,
      diameter: Number(diaPhys.toFixed(2)),
      circularity: Number(Math.min(0.99, 1 / c.aspectRatio).toFixed(3)),
      aspectRatio: Number(c.aspectRatio.toFixed(2)),
      cls: merged ? 'agglomerate' : 'normal',
      flagged: merged,
      flagReason: merged ? '疑似双孔粘连 (Merged via)' : undefined,
      confidence: merged ? 0.72 : 0.96,
    });
    shapes.push({
      id: `hole_${id}`,
      kind: 'polygon',
      points: c.contour,
      holes: c.holeContours.length > 0 ? c.holeContours : undefined,
      color: merged ? '#F59E0B' : '#10B981',
      fill: merged ? 'rgba(245, 158, 11, 0.14)' : 'rgba(16, 185, 129, 0.14)',
      objectId: id,
      layer: 'segmentation',
    });
  });

  // 2D lattice pitch & missing-hole grid check
  const xs = comps.map((c) => c.cx).sort((a, b) => a - b);
  const ys = comps.map((c) => c.cy).sort((a, b) => a - b);
  const cluster1D = (arr: number[], tol: number): number[] => {
    const groups: number[][] = [];
    for (const v of arr) {
      const last = groups[groups.length - 1];
      if (last && v - last[last.length - 1] < tol) last.push(v);
      else groups.push([v]);
    }
    return groups.filter((g) => g.length >= 2).map((g) => meanStd(g).mean);
  };
  const colCenters = cluster1D(xs, 26);
  const rowCenters = cluster1D(ys, 26);
  const pitchX = colCenters.length >= 2 ? meanStd(colCenters.slice(1).map((v, i) => v - colCenters[i])).mean * pxSize : 0;
  const pitchY = rowCenters.length >= 2 ? meanStd(rowCenters.slice(1).map((v, i) => v - rowCenters[i])).mean * pxSize : 0;

  let missingCount = 0;
  for (const ry of rowCenters) {
    for (const cx of colCenters) {
      const hit = comps.some((c) => Math.hypot(c.cx - cx, c.cy - ry) < 24);
      if (!hit) {
        missingCount++;
        const mid = 1000 + missingCount;
        objects.push({
          id: mid,
          kind: 'hole',
          x: cx,
          y: ry,
          diameter: 0,
          cls: 'missing',
          flagged: true,
          flagReason: '阵列缺失孔 (Missing contact hole)',
          confidence: 0.93,
        });
        shapes.push({
          id: `missing_${missingCount}`,
          kind: 'circle',
          points: [
            [cx, ry],
            [18, 18],
          ],
          color: '#EF4444',
          dashed: true,
          label: `MISS ${missingCount}`,
          objectId: mid,
          layer: 'flags',
        });
      }
    }
  }

  const dStats = meanStd(dias);
  const warnings: string[] = [];
  if (missingCount > 0) warnings.push(`阵列检出 ${missingCount} 处缺失接触孔，已在画布用红色虚线圆标出。`);
  if (!calibration.confirmed) warnings.push('当前标定来自信息栏比例尺识别，尚未经人工确认。');

  return {
    objects,
    shapes,
    summary: [
      { key: 'holes', labelZh: '检出接触孔数', labelEn: 'Detected Holes', value: comps.length, unit: '个', highlight: true },
      {
        key: 'diameter_mean',
        labelZh: '平均孔径',
        labelEn: 'Mean Diameter',
        value: Number(dStats.mean.toFixed(2)),
        unit,
        detailZh: `3σ 均匀性 ${(dStats.sd * 3).toFixed(2)} ${unit}`,
        detailEn: `3σ uniformity ${(dStats.sd * 3).toFixed(2)} ${unit}`,
        highlight: true,
      },
      {
        key: 'pitch_x',
        labelZh: '阵列周期 (X / Y)',
        labelEn: 'Lattice Pitch (X / Y)',
        value: `${pitchX.toFixed(1)} / ${pitchY.toFixed(1)}`,
        unit,
      },
      {
        key: 'missing_holes',
        labelZh: '缺失孔缺陷',
        labelEn: 'Missing Holes',
        value: missingCount,
        unit: '个',
        highlight: missingCount > 0,
      },
    ],
    summaryMap: {
      holes: comps.length,
      diameter_mean: Number(dStats.mean.toFixed(3)),
      diameter_3sigma: Number((dStats.sd * 3).toFixed(3)),
      pitch_a: Number(pitchX.toFixed(2)),
      missing_holes: missingCount,
    },
    warnings,
    histogram: {
      labelZh: '接触孔孔径分布',
      labelEn: 'Contact Hole Diameter Distribution',
      unit,
      values: dias,
      mean: dStats.mean,
    },
  };
}

function runSemParticlesAnalysis(
  specimen: Specimen,
  gray: Float32Array,
  probMap: Float32Array | undefined,
  params: Record<string, number | boolean | string>,
  corrections: HumanCorrection[],
  probThreshold = 0.5
): {
  objects: DetectionObject[];
  shapes: OverlayShape[];
  summary: SummaryMetric[];
  summaryMap: Record<string, number | string | null>;
  warnings: string[];
  histogram: RunResult['histogram'];
} {
  const { width, height, calibration } = specimen;
  const cropRows = calibration.cropRows ?? height;
  const pxSize = calibration.pixelSize;
  const unit = calibration.spaceUnit;
  const separationMode = String(params.separationMode ?? 'standard');
  const isBright = String(params.feature ?? 'bright') === 'bright';
  const minDiaPx = Number(params.minDiameterPx ?? 10);
  const edgeBarrierStrength = Number(params.edgeBarrierStrength ?? 0.65);
  const edgeRefinement = String(params.EdgeRefinement ?? 'off') as EdgeRefinementMode;
  const splitSensitivity = Number(params.splitSensitivity ?? 0.55);
  const occlusionSensitivity = Number(params.occlusionSensitivity ?? 0.5);
  const curvatureBoundaryFit =
    params.curvatureBoundaryFit !== undefined ? Boolean(params.curvatureBoundaryFit) : true;
  const overlapConfidenceThreshold =
    params.overlapConfidenceThreshold !== undefined
      ? Number(params.overlapConfidenceThreshold)
      : undefined;
  const excludeBorder = params.excludeBorder !== undefined ? Boolean(params.excludeBorder) : true;
  const outlierZ = Number(params.outlierZ ?? 2.2);

  type ParticleComp = {
    areaPx: number;
    cx: number;
    cy: number;
    eqDiameterPx: number;
    circularity: number;
    solidity: number;
    aspectRatio: number;
    meanGray: number;
    touchingBorder: boolean;
    occluded: boolean;
    occlusionScore: number;
    overlapConfidence: number;
    concaveCuspCount: number;
    boundaryFitScore: number;
    occludedReason?: string;
    contour: Array<[number, number]>;
    holeContours: Array<Array<[number, number]>>;
  };

  let comps: ParticleComp[];

  if (separationMode === 'sem_edge_watershed') {
    comps = segmentSemBrightEdgeParticles(gray, width, cropRows, {
      isBright,
      minDiameterPx: minDiaPx,
      edgeBarrierStrength,
      edgeRefinement,
      splitSensitivity,
      occlusionSensitivity,
      excludeBorder,
      curvatureBoundaryFit,
      overlapConfidenceThreshold,
      probMap,
      probThreshold,
    });
  } else {
    const binary = new Uint8Array(width * cropRows);
    if (probMap) {
      for (let i = 0; i < width * cropRows; i++) {
        if (probMap[i] >= probThreshold) binary[i] = 1;
      }
    } else {
      const vals: number[] = [];
      for (let i = 0; i < width * cropRows; i += 4) vals.push(gray[i]);
      const thresh = otsuThreshold(vals);
      for (let i = 0; i < width * cropRows; i++) {
        binary[i] = (isBright ? gray[i] >= thresh : gray[i] <= thresh) ? 1 : 0;
      }
    }

    // Label 4-connected regions so standard mode also gets true convex-hull solidity, curvature fit & occlusion analysis
    const labels = new Int32Array(width * cropRows);
    const queue = new Int32Array(width * cropRows);
    let nextLabel = 1;
    for (let y = 0; y < cropRows; y++) {
      const row = y * width;
      for (let x = 0; x < width; x++) {
        const start = row + x;
        if (!binary[start] || labels[start] !== 0) continue;
        let head = 0;
        let tail = 0;
        queue[tail++] = start;
        labels[start] = nextLabel;
        while (head < tail) {
          const idx = queue[head++];
          const px = idx % width;
          const py = (idx - px) / width;
          if (px > 0 && binary[idx - 1] && !labels[idx - 1]) {
            labels[idx - 1] = nextLabel;
            queue[tail++] = idx - 1;
          }
          if (px + 1 < width && binary[idx + 1] && !labels[idx + 1]) {
            labels[idx + 1] = nextLabel;
            queue[tail++] = idx + 1;
          }
          if (py > 0 && binary[idx - width] && !labels[idx - width]) {
            labels[idx - width] = nextLabel;
            queue[tail++] = idx - width;
          }
          if (py + 1 < cropRows && binary[idx + width] && !labels[idx + width]) {
            labels[idx + width] = nextLabel;
            queue[tail++] = idx + width;
          }
        }
        nextLabel++;
      }
    }

    comps = extractSeparatedComponentsWithOcclusion(labels, gray, width, cropRows, {
      minDiameterPx: minDiaPx,
      occlusionSensitivity,
      excludeBorder,
      curvatureBoundaryFit,
      overlapConfidenceThreshold,
      edgeRefinement,
    });
  }

  const diasAll = comps.map((c) => c.eqDiameterPx * pxSize);
  const graysAll = comps.map((c) => c.meanGray);
  const sortedDias = [...diasAll].sort((a, b) => a - b);
  const medD = quantile(sortedDias, 0.5);
  const madD = Math.max(
    1.5 * pxSize,
    quantile(
      sortedDias.map((d) => Math.abs(d - medD)).sort((a, b) => a - b),
      0.5
    ) * 1.4826
  );
  const { mean: meanG, sd: sdG } = meanStd(graysAll);

  const CLASS_COLOR: Record<string, string> = {
    normal: '#10B981',
    agglomerate: '#F59E0B',
    irregular: '#EC4899',
    outlier: '#EF4444',
    occluded: '#C084FC',
  };

  const corrForSpec = corrections.filter((c) => c.specimenId === specimen.id);
  const objects: DetectionObject[] = [];
  const shapes: OverlayShape[] = [];

  comps.forEach((c, idx) => {
    const id = idx + 1;
    const dPhys = c.eqDiameterPx * pxSize;
    const zSize = Math.abs(dPhys - medD) / Math.max(1e-3, madD);
    const zGray = sdG > 3 ? Math.abs(c.meanGray - meanG) / sdG : 0;

    let occluded = c.occluded;
    let cls: DetectionObject['cls'] = 'normal';
    let reason = '形态与尺寸均在主分布区间内';

    if (occluded) {
      cls = 'occluded';
      reason = c.occludedReason ?? '疑似被上层颗粒部分遮挡（局部边界被压盖）';
    } else if (zSize > outlierZ * 1.5 || zGray > outlierZ) {
      cls = 'outlier';
      reason = zGray > outlierZ ? '灰度显著异常（疑似异物污染）' : '等效直径远超中位值（离群大颗粒）';
    } else if (c.aspectRatio > 1.75) {
      cls = 'irregular';
      reason = `长宽比 ${c.aspectRatio.toFixed(2)}（疑似破碎或长条碎片）`;
    } else if (c.aspectRatio > 1.32 || c.solidity < 0.87) {
      cls = 'agglomerate';
      reason = `轮廓凹陷/粘连（凸包实心度 ${c.solidity.toFixed(2)}，疑似团聚体）`;
    }

    let excluded = false;
    for (const corr of corrForSpec) {
      if (corr.objectId !== id) continue;
      if (corr.field === 'cls') {
        cls = corr.newValue as DetectionObject['cls'];
        occluded = cls === 'occluded';
      }
      if (corr.field === 'excluded') excluded = Boolean(corr.newValue);
    }

    const flagged = cls !== 'normal';
    const baseColor = CLASS_COLOR[cls ?? 'normal'] ?? '#10B981';

    objects.push({
      id,
      kind: 'particle',
      x: c.cx,
      y: c.cy,
      diameter: Number(dPhys.toFixed(1)),
      area: Number((c.areaPx * pxSize * pxSize).toFixed(1)),
      circularity: Number(c.circularity.toFixed(3)),
      solidity: Number(c.solidity.toFixed(3)),
      aspectRatio: Number(c.aspectRatio.toFixed(2)),
      cls,
      occluded,
      occlusionScore: c.occlusionScore,
      overlapConfidence: c.overlapConfidence,
      concaveCuspCount: c.concaveCuspCount,
      boundaryFitScore: c.boundaryFitScore,
      occludedReason: occluded ? c.occludedReason ?? reason : undefined,
      reason,
      flagged,
      flagReason: flagged ? reason : undefined,
      excluded,
      confidence: occluded ? Number(Math.max(0.55, 0.88 - c.occlusionScore * 0.25).toFixed(2)) : cls === 'normal' ? 0.95 : 0.84,
    });

    if (!excluded) {
      shapes.push({
        id: `part_${id}`,
        kind: 'polygon',
        points: c.contour,
        holes: c.holeContours.length > 0 ? c.holeContours : undefined,
        color: baseColor,
        fill: `${baseColor}24`,
        dashed: occluded,
        label: flagged ? `#${id} ${cls}` : `#${id}`,
        objectId: id,
        occluded,
        layer: flagged ? 'flags' : 'segmentation',
      });
    }
  });

  const active = objects.filter((o) => !o.excluded);
  const unoccluded = active.filter((o) => !o.occluded);
  const sizingPool = unoccluded.length >= 3 ? unoccluded : active;
  const activeDias = sizingPool.map((o) => o.diameter ?? 0).sort((a, b) => a - b);
  const dStats = meanStd(activeDias);
  const d10 = quantile(activeDias, 0.1);
  const d50 = quantile(activeDias, 0.5);
  const d90 = quantile(activeDias, 0.9);
  const nOcc = active.filter((o) => o.occluded).length;
  const nAgg = active.filter((o) => o.cls === 'agglomerate').length;
  const nIrr = active.filter((o) => o.cls === 'irregular').length;
  const nOut = active.filter((o) => o.cls === 'outlier').length;
  const totalAreaPx = comps.reduce((a, b) => a + b.areaPx, 0);
  const areaFrac = (totalAreaPx / Math.max(1, width * cropRows)) * 100;

  const warnings: string[] = [];
  if (nOcc > 0) {
    warnings.push(
      `检出 ${nOcc} 个疑似被上层物体部分遮挡的颗粒（粒度统计已优先基于完整未遮挡颗粒计算；可在画布右上角「图层 Overlay」切换隐藏或异色边框显示）。`
    );
  }
  if (nOut > 0) warnings.push(`检出 ${nOut} 个疑似异物/超大异常颗粒，建议人工复核。`);
  if (nAgg > 0) warnings.push(`检出 ${nAgg} 个团聚粘连体，可切换至「SEM 亮边叠加分水岭」或模型中心进一步拆分。`);

  return {
    objects,
    shapes,
    summary: [
      {
        key: 'particles',
        labelZh: '有效颗粒数',
        labelEn: 'Particles',
        value: active.length,
        unit: '个',
        detailZh: nOcc > 0 ? `完整 ${unoccluded.length} · 被遮挡 ${nOcc} · 面积 ${areaFrac.toFixed(1)}%` : `面积分数 ${areaFrac.toFixed(1)}%`,
        detailEn: nOcc > 0 ? `Complete ${unoccluded.length} · Occluded ${nOcc} · Area ${areaFrac.toFixed(1)}%` : `Area fraction ${areaFrac.toFixed(1)}%`,
        highlight: true,
      },
      {
        key: 'd50_n',
        labelZh: '中位径 d50 (数量)',
        labelEn: 'Median d50 (Number)',
        value: Number(d50.toFixed(1)),
        unit,
        detailZh: `d10 ${d10.toFixed(1)} · d90 ${d90.toFixed(1)} ${unit}`,
        detailEn: `d10 ${d10.toFixed(1)} · d90 ${d90.toFixed(1)} ${unit}`,
        highlight: true,
      },
      {
        key: 'diameter_mean',
        labelZh: '平均等效直径',
        labelEn: 'Mean Eq. Diameter',
        value: Number(dStats.mean.toFixed(1)),
        unit,
        detailZh: `SD ${dStats.sd.toFixed(1)} ${unit}${nOcc > 0 && unoccluded.length >= 3 ? '（已排除遮挡）' : ''}${edgeRefinement !== 'off' ? ` · 锐化: ${edgeRefinement}` : ''}`,
        detailEn: `SD ${dStats.sd.toFixed(1)} ${unit}${nOcc > 0 && unoccluded.length >= 3 ? ' (unoccluded)' : ''}${edgeRefinement !== 'off' ? ` · Refine: ${edgeRefinement}` : ''}`,
      },
      {
        key: 'class_summary',
        labelZh: '遮挡与异常形貌',
        labelEn: 'Occluded & Morphology Flags',
        value: nOcc + nAgg + nIrr + nOut,
        unit: '个',
        detailZh: `遮挡 ${nOcc} · 团聚 ${nAgg} · 不规则 ${nIrr} · 异物 ${nOut}`,
        detailEn: `Occl ${nOcc} · Agg ${nAgg} · Irreg ${nIrr} · Out ${nOut}`,
        highlight: nOut > 0 || nOcc > 0,
      },
    ],
    summaryMap: {
      particles: active.length,
      particles_complete: unoccluded.length,
      diameter_mean: Number(dStats.mean.toFixed(2)),
      d10_n: Number(d10.toFixed(2)),
      d50_n: Number(d50.toFixed(2)),
      d90_n: Number(d90.toFixed(2)),
      area_fraction: Number((areaFrac / 100).toFixed(4)),
      class_occluded: nOcc,
      class_agglomerate: nAgg,
      class_irregular: nIrr,
      class_outlier: nOut,
      edge_refinement: edgeRefinement,
    },
    warnings,
    histogram: {
      labelZh: nOcc > 0 ? '完整颗粒等效直径分布' : '颗粒等效直径分布',
      labelEn: nOcc > 0 ? 'Unoccluded Particle Equivalent Diameter Distribution' : 'Particle Equivalent Diameter Distribution',
      unit,
      values: activeDias,
      mean: dStats.mean,
    },
  };
}

/** Exact O(W*H) 2D Euclidean Distance Transform to nearest 0-pixel (Felzenszwalb-Huttenlocher). */
function computeEuclideanDistanceTransform(binary: Uint8Array, w: number, h: number): Float32Array {
  const INF = 1e12;
  const maxDim = Math.max(w, h);
  const f = new Float64Array(maxDim);
  const v = new Int32Array(maxDim);
  const z = new Float64Array(maxDim + 1);
  const tmp = new Float64Array(w * h);
  const dist = new Float32Array(w * h);

  const dt1d = (len: number, outArr: Float64Array, outOffset: number, outStride: number) => {
    let k = 0;
    v[0] = 0;
    z[0] = -INF;
    z[1] = INF;
    for (let q = 1; q < len; q++) {
      let s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k]);
      while (s <= z[k]) {
        k--;
        s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k]);
      }
      k++;
      v[k] = q;
      z[k] = s;
      z[k + 1] = INF;
    }
    k = 0;
    for (let q = 0; q < len; q++) {
      while (z[k + 1] < q) k++;
      const d = q - v[k];
      outArr[outOffset + q * outStride] = d * d + f[v[k]];
    }
  };

  // Pass 1: along columns
  for (let x = 0; x < w; x++) {
    for (let y = 0; y < h; y++) {
      f[y] = binary[y * w + x] ? INF : 0;
    }
    dt1d(h, tmp, x, w);
  }
  // Pass 2: along rows
  const rowOut = new Float64Array(w);
  for (let y = 0; y < h; y++) {
    const off = y * w;
    for (let x = 0; x < w; x++) f[x] = tmp[off + x];
    dt1d(w, rowOut, 0, 1);
    for (let x = 0; x < w; x++) dist[off + x] = Math.sqrt(rowOut[x]);
  }
  return dist;
}

/** Zhang-Suen topological thinning producing a 1-pixel-wide medial skeleton. */
function zhangSuenThinning(binary: Uint8Array, w: number, h: number): Uint8Array {
  const img = new Uint8Array(binary);
  const toClear: number[] = [];

  for (let iter = 0; iter < 42; iter++) {
    let changed = false;

    for (let pass = 0; pass < 2; pass++) {
      toClear.length = 0;
      for (let y = 1; y < h - 1; y++) {
        const off = y * w;
        for (let x = 1; x < w - 1; x++) {
          const idx = off + x;
          if (!img[idx]) continue;

          const p2 = img[idx - w];
          const p3 = img[idx - w + 1];
          const p4 = img[idx + 1];
          const p5 = img[idx + w + 1];
          const p6 = img[idx + w];
          const p7 = img[idx + w - 1];
          const p8 = img[idx - 1];
          const p9 = img[idx - w - 1];

          const B = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
          if (B < 2 || B > 6) continue;

          const A =
            (!p2 && p3 ? 1 : 0) +
            (!p3 && p4 ? 1 : 0) +
            (!p4 && p5 ? 1 : 0) +
            (!p5 && p6 ? 1 : 0) +
            (!p6 && p7 ? 1 : 0) +
            (!p7 && p8 ? 1 : 0) +
            (!p8 && p9 ? 1 : 0) +
            (!p9 && p2 ? 1 : 0);
          if (A !== 1) continue;

          if (pass === 0) {
            if (p2 * p4 * p6 !== 0) continue;
            if (p4 * p6 * p8 !== 0) continue;
          } else {
            if (p2 * p4 * p8 !== 0) continue;
            if (p2 * p6 * p8 !== 0) continue;
          }
          toClear.push(idx);
        }
      }
      if (toClear.length > 0) {
        changed = true;
        for (let i = 0; i < toClear.length; i++) img[toClear[i]] = 0;
      }
    }
    if (!changed) break;
  }

  // Final pass: remove redundant 8-connected L-corner pixels on diagonal lines so diagonals are strictly 1px thin
  for (let y = 1; y < h - 1; y++) {
    const off = y * w;
    for (let x = 1; x < w - 1; x++) {
      const idx = off + x;
      if (!img[idx]) continue;
      const n = img[idx - w];
      const e = img[idx + 1];
      const s = img[idx + w];
      const we = img[idx - 1];
      const ne = img[idx - w + 1];
      const se = img[idx + w + 1];
      const sw = img[idx + w - 1];
      const nw = img[idx - w - 1];
      // Remove corner pixel if its two orthogonal neighbors are connected diagonally via itself and no other branch relies on it
      if (n && e && !sw && !s && !we) img[idx] = 0;
      else if (e && s && !nw && !n && !we) img[idx] = 0;
      else if (s && we && !ne && !n && !e) img[idx] = 0;
      else if (we && n && !se && !s && !e) img[idx] = 0;
    }
  }

  return img;
}

function sampleGrayBilinear(gray: Float32Array, w: number, h: number, x: number, y: number): number {
  const cx = Math.max(0, Math.min(w - 1.001, x));
  const cy = Math.max(0, Math.min(h - 1.001, y));
  const x0 = Math.floor(cx);
  const y0 = Math.floor(cy);
  const fx = cx - x0;
  const fy = cy - y0;
  const i00 = y0 * w + x0;
  const v00 = gray[i00];
  const v10 = gray[i00 + 1];
  const v01 = gray[i00 + w];
  const v11 = gray[i00 + w + 1];
  return (v00 * (1 - fx) + v10 * fx) * (1 - fy) + (v01 * (1 - fx) + v11 * fx) * fy;
}

interface StraightFibreSegment {
  points: Array<[number, number]>;
  mx: number;
  my: number;
  tx: number;
  ty: number;
  nx: number;
  ny: number;
  tMin: number;
  tMax: number;
  length: number;
  angleDeg: number;
  medianHalfW: number;
}

function fitLinePCA(points: Array<[number, number]>, distMap: Float32Array, w: number): StraightFibreSegment {
  const n = points.length;
  let sx = 0;
  let sy = 0;
  for (let i = 0; i < n; i++) {
    sx += points[i][0];
    sy += points[i][1];
  }
  const mx = sx / Math.max(1, n);
  const my = sy / Math.max(1, n);

  let sxx = 0;
  let syy = 0;
  let sxy = 0;
  for (let i = 0; i < n; i++) {
    const dx = points[i][0] - mx;
    const dy = points[i][1] - my;
    sxx += dx * dx;
    syy += dy * dy;
    sxy += dx * dy;
  }

  // Principal eigenvector of 2x2 covariance matrix
  const theta = 0.5 * Math.atan2(2 * sxy, sxx - syy);
  let tx = Math.cos(theta);
  let ty = Math.sin(theta);
  // Canonical orientation in [0, 180) degrees (ty >= 0, or if ty === 0 then tx >= 0)
  if (ty < -1e-6 || (Math.abs(ty) <= 1e-6 && tx < 0)) {
    tx = -tx;
    ty = -ty;
  }
  const nx = -ty;
  const ny = tx;

  let tMin = Infinity;
  let tMax = -Infinity;
  const halfWs: number[] = [];
  for (let i = 0; i < n; i++) {
    const proj = (points[i][0] - mx) * tx + (points[i][1] - my) * ty;
    if (proj < tMin) tMin = proj;
    if (proj > tMax) tMax = proj;
    const ix = Math.round(points[i][0]);
    const iy = Math.round(points[i][1]);
    halfWs.push(distMap[iy * w + ix]);
  }
  halfWs.sort((a, b) => a - b);
  const medianHalfW = halfWs.length > 0 ? halfWs[Math.floor(halfWs.length * 0.5)] : 4;
  let angleDeg = (Math.atan2(ty, tx) * 180) / Math.PI;
  if (angleDeg < 0) angleDeg += 180;
  if (angleDeg >= 180) angleDeg -= 180;

  return {
    points,
    mx,
    my,
    tx,
    ty,
    nx,
    ny,
    tMin,
    tMax,
    length: Math.max(0, tMax - tMin),
    angleDeg,
    medianHalfW,
  };
}

/** Order an 8-connected skeleton chain and split gently/sharply bent chains into straight pieces. */
function splitChainIntoStraightSegments(
  chain: Array<[number, number]>,
  distMap: Float32Array,
  w: number,
  maxDeviationPx = 3.6
): StraightFibreSegment[] {
  if (chain.length < 6) return [];
  const fit = fitLinePCA(chain, distMap, w);
  if (fit.length < 5) return [];

  // Order points along principal projection so splitting produces clean halves
  const ordered = chain
    .map((pt) => ({
      pt,
      t: (pt[0] - fit.mx) * fit.tx + (pt[1] - fit.my) * fit.ty,
      perp: Math.abs((pt[0] - fit.mx) * fit.nx + (pt[1] - fit.my) * fit.ny),
    }))
    .sort((a, b) => a.t - b.t);

  let worstIdx = -1;
  let worstPerp = 0;
  for (let i = 4; i < ordered.length - 4; i++) {
    if (ordered[i].perp > worstPerp) {
      worstPerp = ordered[i].perp;
      worstIdx = i;
    }
  }

  if (worstPerp > maxDeviationPx && worstIdx >= 6 && ordered.length - worstIdx >= 6) {
    const leftPts = ordered.slice(0, worstIdx).map((o) => o.pt);
    const rightPts = ordered.slice(worstIdx).map((o) => o.pt);
    return [
      ...splitChainIntoStraightSegments(leftPts, distMap, w, maxDeviationPx),
      ...splitChainIntoStraightSegments(rightPts, distMap, w, maxDeviationPx),
    ];
  }

  return [fit];
}

function runSemFibresAnalysis(
  specimen: Specimen,
  gray: Float32Array,
  params: Record<string, number | boolean | string>,
  corrections: HumanCorrection[] = []
): {
  objects: DetectionObject[];
  shapes: OverlayShape[];
  summary: SummaryMetric[];
  summaryMap: Record<string, number | string | null>;
  warnings: string[];
  histogram: RunResult['histogram'];
  profileChart?: RunResult['profileChart'];
} {
  const { width, height, calibration } = specimen;
  const cropRows = calibration.cropRows ?? height;
  const pxSize = calibration.pixelSize;
  const unit = calibration.spaceUnit;
  const feature = String(params.feature ?? 'bright');
  const bias = Number(params.thresholdBias ?? 0.45);
  const edgeMethod = String(params.edgeMethod ?? 'gradient');
  const sampleSpacing = Math.max(8, Number(params.sampleSpacingPx ?? 24));
  const brightFibres = feature !== 'dark';

  // Step 1: Segment fibres into binary foreground mask (invert signal if dark fibres)
  const signal = new Float32Array(width * cropRows);
  const vals: number[] = [];
  for (let i = 0; i < width * cropRows; i++) {
    const s = brightFibres ? gray[i] : 255 - gray[i];
    signal[i] = s;
    if (i % 4 === 0) vals.push(s);
  }
  vals.sort((a, b) => a - b);
  const qLow = quantile(vals, 0.25);
  const qHigh = quantile(vals, 0.96);
  const thresh = qLow + (qHigh - qLow) * bias;

  const rawFg = new Uint8Array(width * cropRows);
  for (let i = 0; i < width * cropRows; i++) {
    if (signal[i] >= thresh) rawFg[i] = 1;
  }

  // Light 3x3 speckle removal so thinning produces clean fibre centerlines
  const fg = new Uint8Array(width * cropRows);
  let fgCount = 0;
  for (let y = 1; y < cropRows - 1; y++) {
    const off = y * width;
    for (let x = 1; x < width - 1; x++) {
      const idx = off + x;
      const sum =
        rawFg[idx - width - 1] + rawFg[idx - width] + rawFg[idx - width + 1] +
        rawFg[idx - 1] + rawFg[idx] + rawFg[idx + 1] +
        rawFg[idx + width - 1] + rawFg[idx + width] + rawFg[idx + width + 1];
      if ((rawFg[idx] && sum >= 4) || (!rawFg[idx] && sum >= 7)) {
        fg[idx] = 1;
        fgCount++;
      }
    }
  }

  // Step 2: Distance transform + Zhang-Suen medial axis thinning
  const distMap = computeEuclideanDistanceTransform(fg, width, cropRows);
  const skel = zhangSuenThinning(fg, width, cropRows);

  // Step 3: Detect fibre crossing junctions on the skeleton and mask crossing zones.
  // Near-orthogonal X-crossings often thin to a 2x2 pixel nexus where each individual pixel
  // has 3x3 transitions = 2 even though 3-4 branches radiate out. Counting distinct foreground
  // arcs along a Chebyshev ring of radius r=2 (and r=3 for thicker fibres) catches every T/Y/X
  // crossing while producing zero false junctions on straight or diagonal 1-px lines.
  const countRingBranches = (cx: number, cy: number, r: number): number => {
    if (cx < r || cx >= width - r || cy < r || cy >= cropRows - r) return 0;
    const ring: number[] = [];
    for (let x = cx - r; x <= cx + r; x++) ring.push(skel[(cy - r) * width + x]);
    for (let y = cy - r + 1; y <= cy + r; y++) ring.push(skel[y * width + (cx + r)]);
    for (let x = cx + r - 1; x >= cx - r; x--) ring.push(skel[(cy + r) * width + x]);
    for (let y = cy + r - 1; y >= cy - r + 1; y--) ring.push(skel[y * width + (cx - r)]);
    let branches = 0;
    const m = ring.length;
    for (let i = 0; i < m; i++) {
      if (!ring[i] && ring[(i + 1) % m]) branches++;
    }
    return branches;
  };

  const nearJunction = new Uint8Array(width * cropRows);
  const junctionPts: Array<[number, number, number]> = [];
  for (let y = 2; y < cropRows - 2; y++) {
    const off = y * width;
    for (let x = 2; x < width - 2; x++) {
      const idx = off + x;
      if (!skel[idx]) continue;
      // Count 8-connected skeleton transitions and degree
      const n0 = skel[idx - width];
      const n1 = skel[idx - width + 1];
      const n2 = skel[idx + 1];
      const n3 = skel[idx + width + 1];
      const n4 = skel[idx + width];
      const n5 = skel[idx + width - 1];
      const n6 = skel[idx - 1];
      const n7 = skel[idx - width - 1];
      const deg = n0 + n1 + n2 + n3 + n4 + n5 + n6 + n7;
      const trans =
        (!n0 && n1 ? 1 : 0) + (!n1 && n2 ? 1 : 0) + (!n2 && n3 ? 1 : 0) + (!n3 && n4 ? 1 : 0) +
        (!n4 && n5 ? 1 : 0) + (!n5 && n6 ? 1 : 0) + (!n6 && n7 ? 1 : 0) + (!n7 && n0 ? 1 : 0);

      let isJunction = trans >= 3;
      if (!isJunction && deg >= 3) {
        if (countRingBranches(x, y, 2) >= 3) {
          isJunction = true;
        } else if (distMap[idx] >= 4.0 && countRingBranches(x, y, 3) >= 3) {
          isJunction = true;
        }
      }
      if (isJunction) {
        junctionPts.push([x, y, Math.max(5, Math.ceil(distMap[idx] * 1.25) + 2)]);
      }
    }
  }
  for (const [jx, jy, rad] of junctionPts) {
    const r2 = rad * rad;
    const y0 = Math.max(0, jy - rad);
    const y1 = Math.min(cropRows - 1, jy + rad);
    const x0 = Math.max(0, jx - rad);
    const x1 = Math.min(width - 1, jx + rad);
    for (let y = y0; y <= y1; y++) {
      const dy = y - jy;
      for (let x = x0; x <= x1; x++) {
        const dx = x - jx;
        if (dx * dx + dy * dy <= r2) {
          nearJunction[y * width + x] = 1;
        }
      }
    }
  }

  // Step 4: Extract non-junction skeleton branches & split curved pieces into straight segments
  const visited = new Uint8Array(width * cropRows);
  const rawSegments: StraightFibreSegment[] = [];
  const queue = new Int32Array(width * cropRows);

  for (let y = 2; y < cropRows - 2; y++) {
    for (let x = 2; x < width - 2; x++) {
      const start = y * width + x;
      if (!skel[start] || nearJunction[start] || visited[start]) continue;

      let head = 0;
      let tail = 0;
      queue[tail++] = start;
      visited[start] = 1;
      const pts: Array<[number, number]> = [];

      while (head < tail) {
        const idx = queue[head++];
        const px = idx % width;
        const py = (idx - px) / width;
        pts.push([px, py]);

        for (let dy = -1; dy <= 1; dy++) {
          const ny = py + dy;
          if (ny < 1 || ny >= cropRows - 1) continue;
          for (let dx = -1; dx <= 1; dx++) {
            if (dx === 0 && dy === 0) continue;
            const nx = px + dx;
            if (nx < 1 || nx >= width - 1) continue;
            const nIdx = ny * width + nx;
            if (skel[nIdx] && !nearJunction[nIdx] && !visited[nIdx]) {
              visited[nIdx] = 1;
              queue[tail++] = nIdx;
            }
          }
        }
      }

      if (pts.length >= 6) {
        rawSegments.push(...splitChainIntoStraightSegments(pts, distMap, width, 3.6));
      }
    }
  }

  // Step 5: Merge collinear straight segments across fibre crossings into individual straight fibres
  rawSegments.sort((a, b) => b.length - a.length);
  const used = new Uint8Array(rawSegments.length);
  const fibres: StraightFibreSegment[] = [];

  const COS_MAX_ANGLE = Math.cos((10 * Math.PI) / 180); // ±10° collinearity tolerance
  for (let i = 0; i < rawSegments.length; i++) {
    if (used[i]) continue;
    used[i] = 1;
    let current = rawSegments[i];
    let mergedAny = true;

    while (mergedAny) {
      mergedAny = false;
      for (let j = 0; j < rawSegments.length; j++) {
        if (used[j]) continue;
        const cand = rawSegments[j];
        const dot = Math.abs(current.tx * cand.tx + current.ty * cand.ty);
        if (dot < COS_MAX_ANGLE) continue;

        const perpCandToCur = Math.abs(
          (cand.mx - current.mx) * current.nx + (cand.my - current.my) * current.ny
        );
        const perpCurToCand = Math.abs(
          (current.mx - cand.mx) * cand.nx + (current.my - cand.my) * cand.ny
        );
        const perpTol = Math.max(4.0, Math.min(9.5, 0.7 * Math.max(current.medianHalfW, cand.medianHalfW)));
        if (perpCandToCur > perpTol || perpCurToCand > perpTol * 1.35) continue;

        // Check longitudinal gap along current's tangent axis AND verify the bridge crosses real foreground (crossing zone)
        const projA = (cand.mx + cand.tMin * cand.tx - current.mx) * current.tx +
                      (cand.my + cand.tMin * cand.ty - current.my) * current.ty;
        const projB = (cand.mx + cand.tMax * cand.tx - current.mx) * current.tx +
                      (cand.my + cand.tMax * cand.ty - current.my) * current.ty;
        const cMin = Math.min(projA, projB);
        const cMax = Math.max(projA, projB);
        const gap = Math.max(0, cMin - current.tMax, current.tMin - cMax);
        const maxAllowedGap = Math.max(180, current.medianHalfW * 24);
        if (gap > maxAllowedGap) continue;

        if (gap > 2) {
          const tBridgeStart = cMin > current.tMax ? current.tMax : cMax;
          const tBridgeEnd = cMin > current.tMax ? cMin : current.tMin;
          const steps = Math.max(6, Math.ceil(gap / 2));
          let fgHits = 0;
          for (let s = 1; s < steps; s++) {
            const tb = tBridgeStart + ((tBridgeEnd - tBridgeStart) * s) / steps;
            const bx = Math.round(current.mx + tb * current.tx);
            const by = Math.round(current.my + tb * current.ty);
            if (bx >= 0 && bx < width && by >= 0 && by < cropRows && fg[by * width + bx]) {
              fgHits++;
            }
          }
          if (fgHits / Math.max(1, steps - 1) < 0.7) continue;
        }

        used[j] = 1;
        current = fitLinePCA([...current.points, ...cand.points], distMap, width);
        mergedAny = true;
      }
    }

    if (current.length >= 22) {
      fibres.push(current);
    }
  }

  // Step 6: Along each segmented straight fibre's centerline, cast strictly PERPENDICULAR rays
  // along (nx, ny) = (-ty, tx) to detect both sub-pixel fibre edges and measure true diameter!
  const corrMap = new Map<number, HumanCorrection[]>();
  for (const c of corrections) {
    if (c.specimenId !== specimen.id) continue;
    const arr = corrMap.get(c.objectId) ?? [];
    arr.push(c);
    corrMap.set(c.objectId, arr);
  }

  const objects: DetectionObject[] = [];
  const shapes: OverlayShape[] = [];
  const allPerpDiameters: number[] = [];
  const fibreAngles: number[] = [];
  const fibreMeanDiameters: number[] = [];
  let representativeProfile: RunResult['profileChart'];

  fibres.forEach((fib) => {
    const { mx, my, tx, ty, nx, ny, tMin, tMax, medianHalfW, angleDeg } = fib;

    interface PerpCaliper {
      cx: number;
      cy: number;
      left: [number, number];
      right: [number, number];
      dPx: number;
      dPhys: number;
      profilePos: number[];
      profileInt: number[];
      edgeLeftPos: number;
      edgeRightPos: number;
    }

    const calipers: PerpCaliper[] = [];
    const maxSearch = Math.min(44, Math.max(12, Math.ceil(medianHalfW * 2.5) + 4));
    const stepS = 0.5;
    const numHalfSteps = Math.round(maxSearch / stepS);

    for (let t = tMin + 8; t <= tMax - 8; t += sampleSpacing) {
      const cx = mx + t * tx;
      const cy = my + t * ty;
      const ix = Math.round(cx);
      const iy = Math.round(cy);

      if (ix < 8 || ix >= width - 8 || iy < 8 || iy >= cropRows - 8) continue;
      if (!fg[iy * width + ix] || nearJunction[iy * width + ix]) continue;

      // Also verify that points ±3px along the centerline are away from junctions
      const fx1 = Math.round(cx + 3 * tx);
      const fy1 = Math.round(cy + 3 * ty);
      const fx2 = Math.round(cx - 3 * tx);
      const fy2 = Math.round(cy - 3 * ty);
      if (
        fx1 < 2 || fx1 >= width - 2 || fy1 < 2 || fy1 >= cropRows - 2 ||
        fx2 < 2 || fx2 >= width - 2 || fy2 < 2 || fy2 >= cropRows - 2 ||
        nearJunction[fy1 * width + fx1] || nearJunction[fy2 * width + fx2]
      ) {
        continue;
      }

      const localR = Math.max(2.0, distMap[iy * width + ix]);
      // Skip if local thickness inflated abruptly due to an unflagged grazing crossing
      if (localR > medianHalfW * 1.6 + 3.0) continue;

      // Sample intensity profile strictly along the perpendicular normal (nx, ny)
      const totalSamples = numHalfSteps * 2 + 1;
      const prof = new Float64Array(totalSamples);
      const posPx = new Float64Array(totalSamples);
      let outOfBounds = false;

      for (let k = -numHalfSteps; k <= numHalfSteps; k++) {
        const s = k * stepS;
        const sx = cx + s * nx;
        const sy = cy + s * ny;
        if (sx < 1 || sx >= width - 2 || sy < 1 || sy >= cropRows - 2) {
          outOfBounds = true;
          break;
        }
        prof[k + numHalfSteps] = sampleGrayBilinear(signal, width, cropRows, sx, sy);
        posPx[k + numHalfSteps] = s;
      }
      if (outOfBounds) continue;

      // Local core intensity near centerline (median of inner core avoids SEM edge-brightening peak bias)
      const centerIdx = numHalfSteps;
      const coreHalfIdx = Math.max(2, Math.round((localR * 0.55) / stepS));
      const coreSamples: number[] = [];
      for (let k = -coreHalfIdx; k <= coreHalfIdx; k++) {
        coreSamples.push(prof[centerIdx + k]);
      }
      coreSamples.sort((a, b) => a - b);
      const coreVal = coreSamples[Math.floor(coreSamples.length * 0.5)];

      const outerSamples: number[] = [];
      const outerStart = Math.min(numHalfSteps - 2, Math.round((localR * 1.25) / stepS));
      for (let k = outerStart; k <= numHalfSteps; k++) {
        outerSamples.push(prof[centerIdx + k]);
        outerSamples.push(prof[centerIdx - k]);
      }
      outerSamples.sort((a, b) => a - b);
      const bgVal = outerSamples.length > 0 ? outerSamples[Math.floor(outerSamples.length * 0.2)] : qLow;
      if (coreVal - bgVal < 20) continue;

      const midLevel = bgVal + 0.5 * (coreVal - bgVal);

      // Search outward along +normal and -normal for sub-pixel fibre edges
      const findEdge = (dir: 1 | -1): number | null => {
        if (edgeMethod === 'gradient') {
          const minIdx = Math.max(2, Math.round((localR * 0.45) / stepS));
          const maxIdx = Math.min(numHalfSteps - 2, Math.round((localR * 1.85 + 3) / stepS));
          let bestK = -1;
          let bestGrad = -1;
          for (let k = minIdx; k <= maxIdx; k++) {
            const idxC = centerIdx + dir * k;
            // Outward drop (positive when bright fibre drops to dark background)
            const g = prof[idxC - dir] - prof[idxC + dir];
            if (g > bestGrad) {
              bestGrad = g;
              bestK = k;
            }
          }
          if (bestK < 0 || bestGrad < 6) return null;
          // Parabolic sub-pixel refinement on outward gradient
          const gPrev = prof[centerIdx + dir * (bestK - 1) - dir] - prof[centerIdx + dir * (bestK - 1) + dir];
          const gCurr = bestGrad;
          const gNext = prof[centerIdx + dir * (bestK + 1) - dir] - prof[centerIdx + dir * (bestK + 1) + dir];
          const denom = gPrev - 2 * gCurr + gNext;
          const delta = Math.abs(denom) > 1e-6 ? Math.max(-0.5, Math.min(0.5, 0.5 * (gPrev - gNext) / denom)) : 0;
          return (bestK + delta) * stepS;
        }

        // Default: 50% half-height threshold crossing with linear sub-pixel interpolation
        const maxIdx = Math.min(numHalfSteps - 1, Math.round((localR * 1.9 + 4) / stepS));
        for (let k = 1; k <= maxIdx; k++) {
          const vPrev = prof[centerIdx + dir * (k - 1)];
          const vCurr = prof[centerIdx + dir * k];
          if (vPrev >= midLevel && vCurr < midLevel) {
            const frac = (vPrev - midLevel) / Math.max(1e-6, vPrev - vCurr);
            return (k - 1 + Math.max(0, Math.min(1, frac))) * stepS;
          }
        }
        return null;
      };

      const sPos = findEdge(1);
      const sNeg = findEdge(-1);
      if (sPos === null || sNeg === null) continue;
      if (sPos < 1.5 || sNeg < 1.5) continue;
      if (Math.abs(sPos - sNeg) > Math.max(4.5, localR * 0.75)) continue;

      const dPx = sPos + sNeg;
      if (dPx < 3.5 || dPx > Math.max(40, medianHalfW * 3.4 + 6)) continue;

      const leftPt: [number, number] = [cx - sNeg * nx, cy - sNeg * ny];
      const rightPt: [number, number] = [cx + sPos * nx, cy + sPos * ny];

      const profilePos: number[] = [];
      const profileInt: number[] = [];
      for (let k = 0; k < totalSamples; k++) {
        profilePos.push(Number(((posPx[k] + maxSearch) * pxSize).toFixed(2)));
        profileInt.push(Number(prof[k].toFixed(1)));
      }

      calipers.push({
        cx,
        cy,
        left: leftPt,
        right: rightPt,
        dPx,
        dPhys: dPx * pxSize,
        profilePos,
        profileInt,
        edgeLeftPos: Number(((maxSearch - sNeg) * pxSize).toFixed(2)),
        edgeRightPos: Number(((maxSearch + sPos) * pxSize).toFixed(2)),
      });
    }

    if (calipers.length === 0 || (fib.length > 60 && calipers.length < 2)) return;

    const fibreId = objects.length + 1;
    let excluded = false;
    for (const cr of corrMap.get(fibreId) ?? []) {
      if (cr.field === 'excluded') excluded = Boolean(cr.newValue);
    }

    // Robust MAD outlier rejection per fibre to drop any residual crossing-touching caliper
    const sortedD = calipers.map((c) => c.dPhys).sort((a, b) => a - b);
    const medD = sortedD[Math.floor(sortedD.length / 2)];
    const absDevs = sortedD.map((d) => Math.abs(d - medD)).sort((a, b) => a - b);
    const mad = absDevs[Math.floor(absDevs.length / 2)];
    const tol = Math.max(2.8 * mad, medD * 0.24);
    const validCalipers = calipers.filter((c) => Math.abs(c.dPhys - medD) <= tol);
    const usedCalipers = validCalipers.length > 0 ? validCalipers : calipers;

    const fibDias = usedCalipers.map((c) => c.dPhys);
    const fStats = meanStd(fibDias);
    const fibLenPhys = (tMax - tMin) * pxSize;

    // Select a clean representative perpendicular cross-section profile for the Inspector chart
    if (!representativeProfile && usedCalipers.length > 0) {
      const midCal = usedCalipers[Math.floor(usedCalipers.length / 2)];
      representativeProfile = {
        positions: midCal.profilePos,
        intensities: midCal.profileInt,
        edges: [midCal.edgeLeftPos, midCal.edgeRightPos],
        unit,
      };
    }

    const startPt: [number, number] = [mx + tMin * tx, my + tMin * ty];
    const endPt: [number, number] = [mx + tMax * tx, my + tMax * ty];
    const midX = 0.5 * (startPt[0] + endPt[0]);
    const midY = 0.5 * (startPt[1] + endPt[1]);

    objects.push({
      id: fibreId,
      kind: 'fibre',
      x: midX,
      y: midY,
      diameter: Number(fStats.mean.toFixed(1)),
      width: Number(fibLenPhys.toFixed(1)),
      orientationDeg: Number(angleDeg.toFixed(1)),
      cls: fStats.sd > fStats.mean * 0.2 ? 'irregular' : 'normal',
      flagged: fStats.sd > fStats.mean * 0.2,
      flagReason:
        fStats.sd > fStats.mean * 0.2
          ? `High diameter variation (SD ${fStats.sd.toFixed(1)} ${unit})`
          : undefined,
      reason: `${usedCalipers.length} ⊥ samples · SD ${fStats.sd.toFixed(1)} ${unit}`,
      excluded,
      confidence: usedCalipers.length >= 4 ? 0.95 : 0.85,
    });

    if (!excluded) {
      fibreMeanDiameters.push(fStats.mean);
      fibreAngles.push(angleDeg);
      for (const d of fibDias) allPerpDiameters.push(d);
    }

    // Visual Overlay 1: Fitted straight fibre centerline (amber dashed) with fibre ID, diameter & orientation
    shapes.push({
      id: `fib_axis_${fibreId}`,
      kind: 'polyline',
      points: [startPt, endPt],
      color: '#F59E0B',
      dashed: true,
      label: `F#${fibreId} · ${fStats.mean.toFixed(0)}${unit} · ${angleDeg.toFixed(0)}°`,
      objectId: fibreId,
      layer: 'roi',
    });

    // Visual Overlay 2: Strictly perpendicular edge-to-edge diameter calipers (cyan) + edge points
    usedCalipers.forEach((cal, cIdx) => {
      shapes.push({
        id: `fib_perp_${fibreId}_${cIdx}`,
        kind: 'polyline',
        points: [cal.left, cal.right],
        color: '#06B6D4',
        objectId: fibreId,
        layer: 'edges',
      });
      shapes.push({
        id: `fib_el_${fibreId}_${cIdx}`,
        kind: 'point',
        points: [cal.left],
        color: '#38BDF8',
        objectId: fibreId,
        layer: 'marks',
      });
      shapes.push({
        id: `fib_er_${fibreId}_${cIdx}`,
        kind: 'point',
        points: [cal.right],
        color: '#38BDF8',
        objectId: fibreId,
        layer: 'marks',
      });
    });
  });

  const dStats = meanStd(allPerpDiameters);
  const dCv = dStats.mean > 0 ? (dStats.sd / dStats.mean) * 100 : 0;
  const porosityPct = (1 - fgCount / Math.max(1, width * cropRows)) * 100;

  // Nematic orientation alignment order parameter S = |<e^(i*2*theta)>| in [0, 1]
  let cos2 = 0;
  let sin2 = 0;
  for (const a of fibreAngles) {
    const rad = (a * Math.PI) / 180;
    cos2 += Math.cos(2 * rad);
    sin2 += Math.sin(2 * rad);
  }
  const alignment = fibreAngles.length > 0 ? Math.hypot(cos2, sin2) / fibreAngles.length : 0;
  let principalAngle =
    fibreAngles.length > 0 ? (0.5 * Math.atan2(sin2, cos2) * 180) / Math.PI : 0;
  if (principalAngle < 0) principalAngle += 180;

  return {
    objects,
    shapes,
    summary: [
      {
        key: 'fibre_count',
        labelZh: '分割直线纤维数',
        labelEn: 'Segmented Fibres',
        value: fibreMeanDiameters.length,
        unit: 'fibres',
        detailZh: `共 ${allPerpDiameters.length} 组垂直边缘截面测点`,
        detailEn: `${allPerpDiameters.length} perpendicular edge calipers`,
        highlight: true,
      },
      {
        key: 'diameter_mean',
        labelZh: '平均法向纤维直径',
        labelEn: 'Mean Normal Diameter',
        value: Number(dStats.mean.toFixed(1)),
        unit,
        detailZh: `SD ${dStats.sd.toFixed(1)} ${unit} · CV ${dCv.toFixed(1)}%`,
        detailEn: `SD ${dStats.sd.toFixed(1)} ${unit} · CV ${dCv.toFixed(1)}%`,
        highlight: true,
      },
      {
        key: 'porosity',
        labelZh: '表面孔隙率',
        labelEn: 'Surface Porosity',
        value: Number(porosityPct.toFixed(1)),
        unit: '%',
        highlight: true,
      },
      {
        key: 'alignment',
        labelZh: '纤维取向一致度',
        labelEn: 'Orientation Alignment',
        value: Number(alignment.toFixed(3)),
        unit: '0–1',
        detailZh:
          alignment < 0.35
            ? `各向同性随机网络 (主方向 ${principalAngle.toFixed(0)}°)`
            : `存在明显择优取向 (主方向 ${principalAngle.toFixed(0)}°)`,
        detailEn:
          alignment < 0.35
            ? `Isotropic mat (mean ${principalAngle.toFixed(0)}°)`
            : `Aligned network (mean ${principalAngle.toFixed(0)}°)`,
      },
    ],
    summaryMap: {
      fibre_count: fibreMeanDiameters.length,
      diameter_mean: Number(dStats.mean.toFixed(2)),
      diameter_sd: Number(dStats.sd.toFixed(2)),
      diameter_cv: Number(dCv.toFixed(2)),
      porosity: Number((porosityPct / 100).toFixed(4)),
      alignment: Number(alignment.toFixed(4)),
      principal_angle_deg: Number(principalAngle.toFixed(1)),
      samples: allPerpDiameters.length,
    },
    warnings:
      fibreMeanDiameters.length === 0
        ? ['未分割出有效直线纤维，请尝试调整分割灵敏度或切换亮/暗纤维极性。']
        : [],
    histogram: {
      labelZh: '法向纤维直径分布',
      labelEn: 'Perpendicular Fibre Diameter Distribution',
      unit,
      values: allPerpDiameters,
      mean: dStats.mean,
    },
    profileChart: representativeProfile,
  };
}

function runCellAnalysis(
  specimen: Specimen,
  gray: Float32Array,
  probMap: Float32Array | undefined,
  params: Record<string, number | boolean | string>,
  corrections: HumanCorrection[],
  probThreshold = 0.5
): {
  objects: DetectionObject[];
  shapes: OverlayShape[];
  summary: SummaryMetric[];
  summaryMap: Record<string, number | string | null>;
  warnings: string[];
  histogram?: RunResult['histogram'];
} {
  const { width, calibration } = specimen;
  const cropRows = calibration.cropRows ?? specimen.height;
  const px = calibration.pixelSize;
  const unit = calibration.spaceUnit;
  const polarityParam = String(params.polarity ?? 'auto');
  const minDiaPx = Number(params.minDiameterPx ?? 10);
  const sensitivity = Number(params.sensitivity ?? 0);

  const vals: number[] = [];
  let sum = 0;
  for (let i = 0; i < width * cropRows; i += 2) {
    vals.push(gray[i]);
    sum += gray[i];
  }
  const meanG = sum / Math.max(1, vals.length);
  const th = otsuThreshold(vals) + sensitivity;
  const brightOnDark = polarityParam === 'bright' || (polarityParam === 'auto' && meanG < 115);

  const binary = new Uint8Array(width * cropRows);
  let fgPx = 0;
  for (let i = 0; i < width * cropRows; i++) {
    const fg = probMap
      ? probMap[i] >= probThreshold
      : brightOnDark
      ? gray[i] >= th
      : gray[i] <= th;
    if (fg) {
      binary[i] = 1;
      fgPx++;
    }
  }

  const minAreaPx = Math.max(12, Math.PI * Math.pow(minDiaPx / 2, 2));
  const maxAreaPx = width * cropRows * 0.18;
  const comps = connectedComponents(binary, gray, width, cropRows, minAreaPx, maxAreaPx);

  const corrMap = new Map<number, HumanCorrection[]>();
  for (const c of corrections) {
    if (c.specimenId !== specimen.id) continue;
    const arr = corrMap.get(c.objectId) ?? [];
    arr.push(c);
    corrMap.set(c.objectId, arr);
  }

  const objects: DetectionObject[] = [];
  const shapes: OverlayShape[] = [];
  const dias: number[] = [];
  const areas: number[] = [];
  const circs: number[] = [];

  for (const c of comps) {
    const dPhys = c.eqDiameterPx * px;
    const areaPhys = c.areaPx * px * px;
    let excluded = false;
    let cls: DetectionObject['cls'] = c.solidity < 0.78 ? 'agglomerate' : c.circularity < 0.65 ? 'irregular' : 'normal';
    const objCorrs = corrMap.get(c.id) ?? [];
    for (const cr of objCorrs) {
      if (cr.field === 'excluded') excluded = Boolean(cr.newValue);
      if (cr.field === 'cls') cls = cr.newValue as DetectionObject['cls'];
    }
    const flagged = cls === 'agglomerate' || cls === 'irregular' || c.touchingBorder;
    const flagReason =
      cls === 'agglomerate'
        ? 'Cluster / overlapping cells'
        : c.touchingBorder
        ? 'Border-touching cell'
        : cls === 'irregular'
        ? 'Irregular cell morphology'
        : undefined;

    objects.push({
      id: c.id,
      kind: 'cell',
      x: c.cx,
      y: c.cy,
      diameter: Number(dPhys.toFixed(2)),
      area: Number(areaPhys.toFixed(2)),
      circularity: Number(c.circularity.toFixed(3)),
      solidity: Number(c.solidity.toFixed(3)),
      aspectRatio: Number(c.aspectRatio.toFixed(2)),
      cls,
      flagged,
      flagReason,
      excluded,
      confidence: flagged ? 0.78 : 0.93,
    });

    if (!excluded) {
      dias.push(dPhys);
      areas.push(areaPhys);
      circs.push(c.circularity);
    }

    const color =
      cls === 'agglomerate' ? '#F59E0B' : cls === 'irregular' ? '#A78BFA' : '#10B981';
    shapes.push({
      id: `cell_${c.id}`,
      kind: 'polygon',
      points: c.contour,
      holes: c.holeContours.length > 0 ? c.holeContours : undefined,
      color,
      fill: `${color}22`,
      label: `#${c.id}`,
      objectId: c.id,
      layer: 'segmentation',
    });
  }

  const dStats = meanStd(dias);
  const dCv = dStats.mean > 0 ? (dStats.sd / dStats.mean) * 100 : 0;
  const aStats = meanStd(areas);
  const cStats = meanStd(circs);
  const confluencyPct = (fgPx / Math.max(1, width * cropRows)) * 100;

  return {
    objects,
    shapes,
    summary: [
      {
        key: 'cell_count',
        labelZh: '检出细胞总数',
        labelEn: 'Detected Cells',
        value: dias.length,
        unit: 'cells',
        detailZh: `含 ${objects.filter((o) => !o.excluded && o.cls === 'agglomerate').length} 个团聚/重叠细胞簇`,
        detailEn: `${objects.filter((o) => !o.excluded && o.cls === 'agglomerate').length} clustered/overlapping`,
        highlight: true,
      },
      {
        key: 'diameter_mean',
        labelZh: '平均等效直径',
        labelEn: 'Mean Eq. Diameter',
        value: Number(dStats.mean.toFixed(2)),
        unit,
        detailZh: `SD ${dStats.sd.toFixed(2)} ${unit} · CV ${dCv.toFixed(1)}%`,
        detailEn: `SD ${dStats.sd.toFixed(2)} ${unit} · CV ${dCv.toFixed(1)}%`,
        highlight: true,
      },
      {
        key: 'confluency_pct',
        labelZh: '视场细胞融合度',
        labelEn: 'Confluency',
        value: Number(confluencyPct.toFixed(1)),
        unit: '%',
        highlight: true,
      },
      {
        key: 'area_mean',
        labelZh: '平均细胞面积',
        labelEn: 'Mean Cell Area',
        value: Number(aStats.mean.toFixed(1)),
        unit: `${unit}²`,
      },
      {
        key: 'circularity_mean',
        labelZh: '平均圆度',
        labelEn: 'Mean Circularity',
        value: Number(cStats.mean.toFixed(3)),
        unit: '0–1',
      },
    ],
    summaryMap: {
      cell_count: dias.length,
      diameter_mean: Number(dStats.mean.toFixed(2)),
      confluency_pct: Number(confluencyPct.toFixed(1)),
      area_mean: Number(aStats.mean.toFixed(2)),
      circularity_mean: Number(cStats.mean.toFixed(3)),
    },
    warnings: [],
    histogram: {
      labelZh: '细胞等效直径分布',
      labelEn: 'Cell Diameter Distribution',
      unit,
      values: dias,
      mean: dStats.mean,
    },
  };
}

function runPcbAnalysis(
  specimen: Specimen,
  gray: Float32Array,
  params: Record<string, number | boolean | string>,
  corrections: HumanCorrection[],
  probMap?: Float32Array,
  probThreshold = 0.5
): {
  objects: DetectionObject[];
  shapes: OverlayShape[];
  summary: SummaryMetric[];
  summaryMap: Record<string, number | string | null>;
  warnings: string[];
  histogram?: RunResult['histogram'];
} {
  const { width, rgba, calibration } = specimen;
  const cropRows = calibration.cropRows ?? specimen.height;
  const px = calibration.pixelSize;
  const unit = calibration.spaceUnit;
  const minAreaPx = Number(params.minAreaPx ?? 40);
  const contrastBias = Number(params.contrastBias ?? 0);

  // Build copper/pad enhancement map (boosts warm metallic pads & bright traces over green/blue solder mask)
  const score = new Float32Array(width * cropRows);
  const vals: number[] = [];
  for (let i = 0; i < width * cropRows; i++) {
    const r = rgba[i * 4];
    const g = rgba[i * 4 + 1];
    const b = rgba[i * 4 + 2];
    const maskSuppress = Math.max(0, g - Math.max(r, b)) * 0.75 + Math.max(0, b - Math.max(r, g)) * 0.55;
    const s = Math.max(0, Math.min(255, gray[i] + Math.max(0, r - g) * 0.4 - maskSuppress));
    score[i] = s;
    if (i % 2 === 0) vals.push(s);
  }
  const binary = new Uint8Array(width * cropRows);
  let copperPx = 0;
  if (probMap) {
    for (let i = 0; i < width * cropRows; i++) {
      if (probMap[i] >= probThreshold) {
        binary[i] = 1;
        copperPx++;
      }
    }
  } else {
    const th = otsuThreshold(vals) + contrastBias;
    for (let i = 0; i < width * cropRows; i++) {
      if (score[i] >= th) {
        binary[i] = 1;
        copperPx++;
      }
    }
  }

  const comps = connectedComponents(binary, score, width, cropRows, minAreaPx, width * cropRows * 0.35);

  const corrMap = new Map<number, HumanCorrection[]>();
  for (const c of corrections) {
    if (c.specimenId !== specimen.id) continue;
    const arr = corrMap.get(c.objectId) ?? [];
    arr.push(c);
    corrMap.set(c.objectId, arr);
  }

  const objects: DetectionObject[] = [];
  const shapes: OverlayShape[] = [];
  const traceWidths: number[] = [];
  const padDias: number[] = [];
  let padCount = 0;
  let traceCount = 0;

  for (const c of comps) {
    const bw = Math.max(1, c.maxX - c.minX + 1);
    const bh = Math.max(1, c.maxY - c.minY + 1);
    const isTrace = c.aspectRatio >= 2.2 || (c.circularity < 0.45 && Math.max(bw, bh) > 22);
    const estWidthPx = isTrace
      ? Math.max(2, c.areaPx / Math.max(1, Math.max(bw, bh)))
      : c.eqDiameterPx;
    const widthPhys = estWidthPx * px;
    const areaPhys = c.areaPx * px * px;

    let excluded = false;
    for (const cr of corrMap.get(c.id) ?? []) {
      if (cr.field === 'excluded') excluded = Boolean(cr.newValue);
    }

    const kind: DetectionObject['kind'] = isTrace ? 'trace' : 'pad';
    const flagged = isTrace && estWidthPx < 4;
    objects.push({
      id: c.id,
      kind,
      x: c.cx,
      y: c.cy,
      width: Number(widthPhys.toFixed(3)),
      diameter: !isTrace ? Number((c.eqDiameterPx * px).toFixed(3)) : undefined,
      area: Number(areaPhys.toFixed(3)),
      circularity: Number(c.circularity.toFixed(3)),
      aspectRatio: Number(c.aspectRatio.toFixed(2)),
      cls: flagged ? 'irregular' : 'normal',
      flagged,
      flagReason: flagged ? 'Narrow trace risk (<4 px)' : undefined,
      excluded,
      confidence: 0.9,
    });

    if (!excluded) {
      if (isTrace) {
        traceCount++;
        traceWidths.push(widthPhys);
      } else {
        padCount++;
        padDias.push(c.eqDiameterPx * px);
      }
    }

    const color = isTrace ? '#06B6D4' : '#F59E0B';
    shapes.push({
      id: `pcb_${c.id}`,
      kind: 'polygon',
      points: c.contour,
      holes: c.holeContours.length > 0 ? c.holeContours : undefined,
      color,
      fill: isTrace ? 'rgba(6, 182, 212, 0.16)' : 'rgba(245, 158, 11, 0.14)',
      label: isTrace ? `T#${c.id}` : `P#${c.id}`,
      objectId: c.id,
      layer: 'segmentation',
    });
  }

  const widthPool = traceWidths.length > 0 ? traceWidths : padDias;
  const twStats = meanStd(widthPool);
  let minTw = widthPool.length > 0 ? Infinity : 0;
  for (let i = 0; i < widthPool.length; i++) {
    if (widthPool[i] < minTw) minTw = widthPool[i];
  }
  const copperPct = (copperPx / Math.max(1, width * cropRows)) * 100;

  return {
    objects,
    shapes,
    summary: [
      {
        key: 'pad_count',
        labelZh: '检出焊盘数',
        labelEn: 'Detected Pads',
        value: padCount,
        unit: 'pads',
        highlight: true,
      },
      {
        key: 'trace_count',
        labelZh: '检出铜走线数',
        labelEn: 'Detected Traces',
        value: traceCount,
        unit: 'traces',
        highlight: true,
      },
      {
        key: 'trace_width_mean',
        labelZh: '平均走线/特征宽度',
        labelEn: 'Mean Trace Width',
        value: Number(twStats.mean.toFixed(3)),
        unit,
        detailZh: `最小线宽 ${minTw.toFixed(3)} ${unit} · SD ${twStats.sd.toFixed(3)}`,
        detailEn: `Min ${minTw.toFixed(3)} ${unit} · SD ${twStats.sd.toFixed(3)}`,
        highlight: true,
      },
      {
        key: 'copper_coverage_pct',
        labelZh: '铜箔覆盖率',
        labelEn: 'Copper Coverage',
        value: Number(copperPct.toFixed(1)),
        unit: '%',
      },
    ],
    summaryMap: {
      pad_count: padCount,
      trace_count: traceCount,
      trace_width_mean: Number(twStats.mean.toFixed(3)),
      trace_width_min: Number(minTw.toFixed(3)),
      copper_coverage_pct: Number(copperPct.toFixed(1)),
    },
    warnings: [],
    histogram: {
      labelZh: '走线宽度与焊盘尺寸分布',
      labelEn: 'Trace Width / Pad Size Distribution',
      unit,
      values: traceWidths.length > 0 ? traceWidths : padDias,
      mean: twStats.mean,
    },
  };
}

function runIntensitySegmentation(
  specimen: Specimen,
  gray: Float32Array,
  probMap: Float32Array | undefined,
  params: Record<string, number | boolean | string>,
  corrections: HumanCorrection[],
  probThreshold = 0.5
): {
  objects: DetectionObject[];
  shapes: OverlayShape[];
  summary: SummaryMetric[];
  summaryMap: Record<string, number | string | null>;
  warnings: string[];
  histogram?: RunResult['histogram'];
  binary: Uint8Array;
  maskPreview: Float32Array;
  qcLabel: string;
} {
  const { width, height, calibration } = specimen;
  const cropRows = calibration.cropRows ?? height;
  const px = calibration.pixelSize;
  const unit = calibration.spaceUnit;
  const method = String(params.method ?? 'otsu');
  const polarityParam = String(params.polarity ?? 'auto');
  const thresholdOffset = Number(params.thresholdOffset ?? 0);
  const manualThreshold = Number(params.manualThreshold ?? 128);
  const minAreaPx = Math.max(8, Number(params.minAreaPx ?? 36));
  const morphClean = params.morphClean !== false;

  const totalPx = width * cropRows;
  const sampleVals: number[] = [];
  for (let i = 0; i < totalPx; i += 2) sampleVals.push(gray[i]);
  const otsuT = otsuThreshold(sampleVals);
  const globalT = Math.max(1, Math.min(254, (method === 'manual' ? manualThreshold : otsuT) + thresholdOffset));

  // Auto-detect polarity by checking border median vs global Otsu threshold
  let brightOnDark = true;
  if (polarityParam === 'bright') {
    brightOnDark = true;
  } else if (polarityParam === 'dark') {
    brightOnDark = false;
  } else {
    let borderSum = 0;
    let borderCnt = 0;
    for (let x = 0; x < width; x += 2) {
      borderSum += gray[x] + gray[(cropRows - 1) * width + x];
      borderCnt += 2;
    }
    for (let y = 0; y < cropRows; y += 2) {
      borderSum += gray[y * width] + gray[y * width + (width - 1)];
      borderCnt += 2;
    }
    const borderMean = borderSum / Math.max(1, borderCnt);
    let aboveCnt = 0;
    for (let i = 0; i < totalPx; i += 2) {
      if (gray[i] >= otsuT) aboveCnt++;
    }
    const aboveFrac = aboveCnt / Math.max(1, sampleVals.length);
    if (aboveFrac >= 0.03 && aboveFrac <= 0.45) {
      brightOnDark = true;
    } else if (aboveFrac >= 0.55 && aboveFrac <= 0.97) {
      brightOnDark = false;
    } else {
      brightOnDark = borderMean < otsuT;
    }
  }

  let binary = new Uint8Array(totalPx);

  if (probMap) {
    for (let i = 0; i < totalPx; i++) {
      if (probMap[i] >= probThreshold) binary[i] = 1;
    }
  } else if (method === 'adaptive') {
    // Fast integral-image local mean adaptive threshold (radius 15px)
    const integral = new Float64Array((width + 1) * (cropRows + 1));
    const stride = width + 1;
    for (let y = 0; y < cropRows; y++) {
      let rowAcc = 0;
      const gOff = y * width;
      const iOff = (y + 1) * stride;
      const pOff = y * stride;
      for (let x = 0; x < width; x++) {
        rowAcc += gray[gOff + x];
        integral[iOff + x + 1] = integral[pOff + x + 1] + rowAcc;
      }
    }
    const r = 15;
    const bias = 8 + thresholdOffset * 0.5;
    for (let y = 0; y < cropRows; y++) {
      const y0 = Math.max(0, y - r);
      const y1 = Math.min(cropRows - 1, y + r);
      for (let x = 0; x < width; x++) {
        const x0 = Math.max(0, x - r);
        const x1 = Math.min(width - 1, x + r);
        const area = (x1 - x0 + 1) * (y1 - y0 + 1);
        const sum =
          integral[(y1 + 1) * stride + (x1 + 1)] -
          integral[y0 * stride + (x1 + 1)] -
          integral[(y1 + 1) * stride + x0] +
          integral[y0 * stride + x0];
        const localMean = sum / Math.max(1, area);
        const v = gray[y * width + x];
        const isFg = brightOnDark ? v > localMean + bias : v < localMean - bias;
        if (isFg) binary[y * width + x] = 1;
      }
    }
  } else {
    for (let i = 0; i < totalPx; i++) {
      const isFg = brightOnDark ? gray[i] >= globalT : gray[i] <= globalT;
      if (isFg) binary[i] = 1;
    }
  }

  // Optional 3x3 morphological opening (erode -> dilate) to remove 1-px isolated noise
  if (morphClean) {
    const eroded = new Uint8Array(totalPx);
    for (let y = 1; y < cropRows - 1; y++) {
      const row = y * width;
      for (let x = 1; x < width - 1; x++) {
        const idx = row + x;
        if (
          binary[idx] &&
          binary[idx - 1] &&
          binary[idx + 1] &&
          binary[idx - width] &&
          binary[idx + width]
        ) {
          eroded[idx] = 1;
        }
      }
    }
    const opened = new Uint8Array(totalPx);
    for (let y = 1; y < cropRows - 1; y++) {
      const row = y * width;
      for (let x = 1; x < width - 1; x++) {
        const idx = row + x;
        if (
          eroded[idx] ||
          eroded[idx - 1] ||
          eroded[idx + 1] ||
          eroded[idx - width] ||
          eroded[idx + width]
        ) {
          opened[idx] = 1;
        }
      }
    }
    binary = opened;
  }

  const maxAreaPx = totalPx * 0.45;
  const comps = connectedComponents(binary, gray, width, cropRows, minAreaPx, maxAreaPx);

  const corrMap = new Map<number, HumanCorrection[]>();
  for (const c of corrections) {
    if (c.specimenId !== specimen.id) continue;
    const arr = corrMap.get(c.objectId) ?? [];
    arr.push(c);
    corrMap.set(c.objectId, arr);
  }

  const objects: DetectionObject[] = [];
  const shapes: OverlayShape[] = [];
  const dias: number[] = [];
  let fgAreaPx = 0;

  for (const c of comps) {
    const dPhys = c.eqDiameterPx * px;
    const areaPhys = c.areaPx * px * px;
    let excluded = false;
    for (const cr of corrMap.get(c.id) ?? []) {
      if (cr.field === 'excluded') excluded = Boolean(cr.newValue);
    }
    const flagged = c.touchingBorder || c.solidity < 0.72;
    const flagReason = c.touchingBorder
      ? 'Border-touching region'
      : c.solidity < 0.72
      ? 'Concave / merged region'
      : undefined;

    if (!excluded) {
      dias.push(dPhys);
      fgAreaPx += c.areaPx;
    }

    objects.push({
      id: c.id,
      kind: 'region',
      x: c.cx,
      y: c.cy,
      diameter: Number(dPhys.toFixed(2)),
      area: Number(areaPhys.toFixed(2)),
      circularity: Number(c.circularity.toFixed(3)),
      solidity: Number(c.solidity.toFixed(3)),
      aspectRatio: Number(c.aspectRatio.toFixed(2)),
      cls: c.solidity < 0.72 ? 'irregular' : 'normal',
      reason: `I=${c.meanGray.toFixed(0)} · A=${areaPhys.toFixed(1)} ${unit}²`,
      flagged,
      flagReason,
      excluded,
      confidence: c.touchingBorder ? 0.78 : 0.88,
    });

    const color = flagged ? '#F59E0B' : '#06B6D4';
    if (c.contour.length >= 3) {
      shapes.push({
        id: `seg_poly_${c.id}`,
        kind: 'polygon',
        points: c.contour,
        holes: c.holeContours.length > 0 ? c.holeContours : undefined,
        color,
        fill: flagged ? 'rgba(245, 158, 11, 0.14)' : 'rgba(6, 182, 212, 0.14)',
        label: `#${c.id}`,
        objectId: c.id,
        layer: 'segmentation',
      });
    } else {
      shapes.push({
        id: `seg_box_${c.id}`,
        kind: 'box',
        points: [
          [c.minX, c.minY],
          [c.maxX, c.maxY],
        ],
        color,
        label: `#${c.id}`,
        objectId: c.id,
        layer: 'segmentation',
      });
    }
  }

  const dStats = meanStd(dias);
  const areaFracPct = (fgAreaPx / Math.max(1, totalPx)) * 100;
  const methodTag = method === 'adaptive' ? 'Adaptive' : method === 'manual' ? `Manual T=${Math.round(globalT)}` : `Otsu T=${Math.round(globalT)}`;
  const polTag = brightOnDark ? 'Bright' : 'Dark';

  const maskPreview = new Float32Array(width * height);
  for (let i = 0; i < totalPx; i++) {
    maskPreview[i] = binary[i] ? 235 : Math.round(gray[i] * 0.35);
  }

  return {
    objects,
    shapes,
    summary: [
      {
        key: 'region_count',
        labelZh: '强度分割区域数',
        labelEn: 'Segmented Regions',
        value: dias.length,
        unit: 'regions',
        detailZh: `${methodTag} · ${brightOnDark ? '亮目标' : '暗目标'}`,
        detailEn: `${methodTag} · ${polTag} foreground`,
        highlight: true,
      },
      {
        key: 'area_fraction_pct',
        labelZh: '前景面积占比',
        labelEn: 'Foreground Area Fraction',
        value: Number(areaFracPct.toFixed(2)),
        unit: '%',
        highlight: true,
      },
      {
        key: 'eq_diameter_mean',
        labelZh: '平均等效直径',
        labelEn: 'Mean Eq. Diameter',
        value: dias.length > 0 ? Number(dStats.mean.toFixed(2)) : 0,
        unit,
        detailZh: `SD ${dStats.sd.toFixed(2)} ${unit}`,
        detailEn: `SD ${dStats.sd.toFixed(2)} ${unit}`,
        highlight: true,
      },
      {
        key: 'applied_threshold',
        labelZh: '生效灰度阈值',
        labelEn: 'Applied Threshold',
        value: method === 'adaptive' ? 'Local' : Math.round(globalT),
        unit: method === 'adaptive' ? '±15px' : 'GV',
        detailZh: morphClean ? '已启用 3×3 开运算去噪' : '未启用形态学去噪',
        detailEn: morphClean ? '3×3 morph open ON' : 'Morph cleanup OFF',
      },
    ],
    summaryMap: {
      region_count: dias.length,
      area_fraction_pct: Number(areaFracPct.toFixed(2)),
      eq_diameter_mean: Number(dStats.mean.toFixed(2)),
      eq_diameter_sd: Number(dStats.sd.toFixed(2)),
      applied_threshold: Math.round(globalT),
    },
    warnings:
      dias.length === 0
        ? ['默认灰度阈值未检出有效目标，请在参数面板调整阈值偏移、切换前景极性或改用局部自适应阈值。']
        : [],
    histogram:
      dias.length > 0
        ? {
            labelZh: '分割区域等效直径分布',
            labelEn: 'Segmented Region Equivalent Diameter',
            unit,
            values: dias,
            mean: dStats.mean,
          }
        : undefined,
    binary,
    maskPreview,
    qcLabel: `${methodTag} · ${dias.length} regions`,
  };
}

// ---------------------------------------------------------------------------
// ROI Sub-Specimen Slicing & Full-Frame Coordinate Remapping Helpers
// ---------------------------------------------------------------------------

function normalizeRoi(
  roi: RoiRect | null | undefined,
  fullW: number,
  cropRows: number
): RoiRect | null {
  if (!roi) return null;
  const x = Math.max(0, Math.min(fullW - 16, Math.round(roi.x)));
  const y = Math.max(0, Math.min(cropRows - 16, Math.round(roi.y)));
  const width = Math.max(16, Math.min(fullW - x, Math.round(roi.width)));
  const height = Math.max(16, Math.min(cropRows - y, Math.round(roi.height)));
  if (width >= fullW && height >= cropRows && x === 0 && y === 0) return null;
  return { x, y, width, height };
}

/**
 * Computes the bounding box enclosing all active (non-excluded) segmented results
 * (`result.shapes` plus fallback to `result.objects`) expanded by `paddingRatio`
 * (default 10% on each side) and strictly clamped within the specimen's working FOV.
 */
export function computeSegmentedResultsBoundingBox(
  result: RunResult | null | undefined,
  specimen: Specimen,
  paddingRatio = 0.1
): RoiRect | null {
  if (!result) return null;

  const fullW = Math.max(1, specimen.width);
  const cropRows = Math.max(
    1,
    Math.min(specimen.height, specimen.calibration.cropRows ?? specimen.height)
  );

  const activeObjects = result.objects.filter((o) => !o.excluded);
  const activeIds = new Set(activeObjects.map((o) => o.id));
  const coveredObjectIds = new Set<number>();

  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  let hasGeometry = false;

  const expandPoint = (px: number, py: number) => {
    if (!Number.isFinite(px) || !Number.isFinite(py)) return;
    if (px < minX) minX = px;
    if (py < minY) minY = py;
    if (px > maxX) maxX = px;
    if (py > maxY) maxY = py;
    hasGeometry = true;
  };

  for (const shape of result.shapes) {
    if (shape.layer === 'roi' || shape.layer === 'manual' || shape.layer === 'flags') {
      continue;
    }
    if (shape.objectId !== undefined) {
      if (activeObjects.length > 0 && !activeIds.has(shape.objectId)) {
        continue;
      }
      coveredObjectIds.add(shape.objectId);
    }
    if (!shape.points || shape.points.length === 0) continue;

    if (shape.kind === 'circle') {
      const [cx, cy] = shape.points[0];
      const rx = shape.points[1] ? Math.abs(shape.points[1][0]) : 4;
      const ry = shape.points[1] ? Math.abs(shape.points[1][1] ?? rx) : rx;
      expandPoint(cx - rx, cy - ry);
      expandPoint(cx + rx, cy + ry);
    } else {
      for (const [px, py] of shape.points) {
        expandPoint(px, py);
      }
    }
  }

  // Fallback for active detected objects that do not have a linked geometry shape
  const pxSize = specimen.calibration.pixelSize > 0 ? specimen.calibration.pixelSize : 1;
  for (const obj of activeObjects) {
    if (coveredObjectIds.has(obj.id)) continue;
    const physSpan = obj.diameter ?? obj.width ?? obj.thickness ?? 0;
    const rPx = physSpan > 0 ? Math.max(4, physSpan / (2 * pxSize)) : 4;
    expandPoint(obj.x - rPx, obj.y - rPx);
    expandPoint(obj.x + rPx, obj.y + rPx);
  }

  if (!hasGeometry || !Number.isFinite(minX) || !Number.isFinite(minY)) {
    return null;
  }

  const clampedMinX = Math.max(0, Math.min(fullW, minX));
  const clampedMinY = Math.max(0, Math.min(cropRows, minY));
  const clampedMaxX = Math.max(clampedMinX, Math.min(fullW, maxX));
  const clampedMaxY = Math.max(clampedMinY, Math.min(cropRows, maxY));

  const rawW = Math.max(1, clampedMaxX - clampedMinX);
  const rawH = Math.max(1, clampedMaxY - clampedMinY);
  const padX = rawW * Math.max(0, paddingRatio);
  const padY = rawH * Math.max(0, paddingRatio);

  let x0 = Math.max(0, Math.floor(clampedMinX - padX));
  let y0 = Math.max(0, Math.floor(clampedMinY - padY));
  let x1 = Math.min(fullW, Math.ceil(clampedMaxX + padX));
  let y1 = Math.min(cropRows, Math.ceil(clampedMaxY + padY));

  const minSizeX = Math.min(16, fullW);
  const minSizeY = Math.min(16, cropRows);
  if (x1 - x0 < minSizeX) {
    const cx = (x0 + x1) / 2;
    x0 = Math.max(0, Math.min(fullW - minSizeX, Math.round(cx - minSizeX / 2)));
    x1 = x0 + minSizeX;
  }
  if (y1 - y0 < minSizeY) {
    const cy = (y0 + y1) / 2;
    y0 = Math.max(0, Math.min(cropRows - minSizeY, Math.round(cy - minSizeY / 2)));
    y1 = y0 + minSizeY;
  }

  return {
    x: x0,
    y: y0,
    width: Math.max(1, x1 - x0),
    height: Math.max(1, y1 - y0),
  };
}

function sliceSpecimenToRoi(specimen: Specimen, roi: RoiRect): Specimen {
  const { width: fullW, gray, rgba, fluoGray, fluoRgba } = specimen;
  const { x: rx, y: ry, width: rw, height: rh } = roi;
  const roiGray = new Float32Array(rw * rh);
  const roiRgba = new Uint8ClampedArray(rw * rh * 4);
  const roiFluoGray = fluoGray ? new Float32Array(rw * rh) : undefined;
  const roiFluoRgba = fluoRgba ? new Uint8ClampedArray(rw * rh * 4) : undefined;

  for (let y = 0; y < rh; y++) {
    const srcRow = (ry + y) * fullW + rx;
    const dstRow = y * rw;
    for (let x = 0; x < rw; x++) {
      const si = srcRow + x;
      const di = dstRow + x;
      roiGray[di] = gray[si];
      roiRgba[di * 4] = rgba[si * 4];
      roiRgba[di * 4 + 1] = rgba[si * 4 + 1];
      roiRgba[di * 4 + 2] = rgba[si * 4 + 2];
      roiRgba[di * 4 + 3] = rgba[si * 4 + 3];
      if (roiFluoGray && fluoGray) roiFluoGray[di] = fluoGray[si];
      if (roiFluoRgba && fluoRgba) {
        roiFluoRgba[di * 4] = fluoRgba[si * 4];
        roiFluoRgba[di * 4 + 1] = fluoRgba[si * 4 + 1];
        roiFluoRgba[di * 4 + 2] = fluoRgba[si * 4 + 2];
        roiFluoRgba[di * 4 + 3] = fluoRgba[si * 4 + 3];
      }
    }
  }

  return {
    ...specimen,
    width: rw,
    height: rh,
    gray: roiGray,
    rgba: roiRgba,
    fluoGray: roiFluoGray,
    fluoRgba: roiFluoRgba,
    calibration: {
      ...specimen.calibration,
      cropRows: rh,
    },
  };
}

function compositeGrayFromRoi(
  baseFullGray: Float32Array,
  roiGray: Float32Array,
  fullW: number,
  fullH: number,
  roi: RoiRect
): Float32Array {
  const out = new Float32Array(baseFullGray);
  const { x: rx, y: ry, width: rw, height: rh } = roi;
  for (let y = 0; y < rh; y++) {
    const dy = ry + y;
    if (dy >= fullH) break;
    const dstRow = dy * fullW + rx;
    const srcRow = y * rw;
    for (let x = 0; x < rw; x++) {
      if (rx + x >= fullW) break;
      out[dstRow + x] = roiGray[srcRow + x];
    }
  }
  return out;
}

function compositeProbFromRoi(
  roiProb: Float32Array,
  fullW: number,
  fullH: number,
  roi: RoiRect
): Float32Array {
  const out = new Float32Array(fullW * fullH);
  const { x: rx, y: ry, width: rw, height: rh } = roi;
  for (let y = 0; y < rh; y++) {
    const dy = ry + y;
    if (dy >= fullH) break;
    const dstRow = dy * fullW + rx;
    const srcRow = y * rw;
    for (let x = 0; x < rw; x++) {
      if (rx + x >= fullW) break;
      out[dstRow + x] = roiProb[srcRow + x];
    }
  }
  return out;
}

function remapShapeFromRoi(s: OverlayShape, rx: number, ry: number): OverlayShape {
  if (s.kind === 'circle' && s.points.length >= 2) {
    return {
      ...s,
      points: [
        [s.points[0][0] + rx, s.points[0][1] + ry],
        s.points[1],
        ...s.points.slice(2).map(([px, py]): [number, number] => [px + rx, py + ry]),
      ],
    };
  }
  return {
    ...s,
    points: s.points.map(([px, py]): [number, number] => [px + rx, py + ry]),
    holes: s.holes?.map((loop) => loop.map(([px, py]): [number, number] => [px + rx, py + ry])),
  };
}

// ---------------------------------------------------------------------------
// Unified Recipe Executor (Microfluidics + SEM + Trained Models + Filters)
// ---------------------------------------------------------------------------

export const RAW_SOURCE_STEP_ID = '__raw_source__';
export const AUTO_CROP_STEP_ID = '__auto_crop_databar__';

export function hasAutoInfoBarCrop(specimen: Specimen): boolean {
  const hasImportedCrop = Boolean(
    specimen.importReport?.scaleBarCropped && specimen.importReport.croppedBarPx > 0
  );
  const hasDatabarCrop = Boolean(
    specimen.calibration.cropRows && specimen.calibration.cropRows < specimen.height
  );
  return hasImportedCrop || hasDatabarCrop;
}

export function executeRecipe(
  specimen: Specimen,
  recipe: UnifiedRecipe,
  options: {
    corrections?: HumanCorrection[];
    localModels?: Map<string, TrainedLocalModel>;
    precomputedProbMaps?: Map<string, Float32Array>;
    roi?: RoiRect | null;
  } = {}
): RunResult {
  const tStart = performance.now();
  const fullCropRows = specimen.calibration.cropRows ?? specimen.height;
  const activeRoi = normalizeRoi(options.roi, specimen.width, fullCropRows);

  // When a manual ROI is set, slice a tight sub-specimen so EVERY step (preprocess, model inference,
  // segmentation, metrology) runs strictly inside the ROI (faster & immune to out-of-ROI artifacts).
  const workSpec = activeRoi ? sliceSpecimenToRoi(specimen, activeRoi) : specimen;
  const workW = workSpec.width;
  const workH = workSpec.height;
  const workCropRows = workSpec.calibration.cropRows ?? workH;

  let workingGray: Float32Array = new Float32Array(workSpec.gray);
  let activeProbMap: Float32Array | undefined;
  let activeProbThreshold = 0.5;
  let hasImageTransform = false;
  let preModelWorkingGray: Float32Array | undefined;

  const toFullGray = (buf: Float32Array): Float32Array =>
    activeRoi
      ? compositeGrayFromRoi(specimen.gray, buf, specimen.width, specimen.height, activeRoi)
      : new Float32Array(buf);
  const toFullProb = (buf: Float32Array): Float32Array =>
    activeRoi
      ? compositeProbFromRoi(buf, specimen.width, specimen.height, activeRoi)
      : new Float32Array(buf);

  const stepPreviews: StepPreview[] = [];
  const stepLogs: RunResult['provenance']['steps'] = [];
  const modelRefs: string[] = [];
  const warnings: string[] = [];

  // Step 1 (Automatic): If info/scale bar was auto-calibrated & cropped, expose it explicitly right after "0. Original Image"
  if (hasAutoInfoBarCrop(specimen)) {
    const croppedPx =
      specimen.importReport?.scaleBarCropped && specimen.importReport.croppedBarPx > 0
        ? specimen.importReport.croppedBarPx
        : specimen.height - fullCropRows;
    const calTag = `${specimen.calibration.pixelSize} ${specimen.calibration.spaceUnit}/px`;
    const cropQc = `calibrated=${calTag} · cropped=${croppedPx}px`;
    stepPreviews.push({
      stepId: AUTO_CROP_STEP_ID,
      op: 'ingest.crop_databar',
      labelZh: '信息栏裁切',
      labelEn: 'Data Bar Crop',
      durationMs: 0.4,
      qcSummary: cropQc,
    });
    stepLogs.push({ id: AUTO_CROP_STEP_ID, op: 'ingest.crop_databar', durationMs: 0.4, qc: cropQc });
  }

  let objects: DetectionObject[] = [];
  let shapes: OverlayShape[] = [];
  let summary: SummaryMetric[] = [];
  let summaryMap: Record<string, number | string | null> = {};
  let histogram: RunResult['histogram'];
  let poissonChart: RunResult['poissonChart'];
  let profileChart: RunResult['profileChart'];

  const hasDenoise = recipe.steps.some((s) => s.enabled && s.op === 'sem.denoise');
  const corrections = options.corrections ?? [];
  let modelCurvatureFit: boolean | undefined;
  let modelOverlapThreshold: number | undefined;
  let activeParentResult: ParentSegmentationOutput | null = null;
  let activeSubRegionResult: SubRegionSegmentationOutput | null = null;
  const hasHierarchyMetrology = recipe.steps.some((s) => s.enabled && s.op === 'region.hierarchy_ratio');

  // Helper: if an upstream model.segment produced a probability map, build a smooth 0..255 image
  // for gradient/profile-based downstream operators so every step strictly consumes the previous step's output.
  const getDownstreamGray = (): Float32Array => {
    if (!activeProbMap) return workingGray;
    const out = new Float32Array(workW * workH);
    for (let i = 0; i < workW * workCropRows; i++) {
      out[i] = activeProbMap[i] * 255;
    }
    return out;
  };

  for (const step of recipe.steps) {
    if (!step.enabled) continue;
    const spec = OPERATOR_CATALOG.find((o) => o.op === step.op);
    const ts = performance.now();
    let qcStr = 'OK';

    if (step.op === 'sem.line_level') {
      workingGray = applyLineLevel(workingGray, workW, workH, workCropRows, Number(step.params.strength ?? 0.85));
      hasImageTransform = true;
      qcStr = `strength=${step.params.strength ?? 0.85}`;
    } else if (step.op === 'sem.destripe') {
      workingGray = applyDestripe(workingGray, workW, workH, workCropRows, Number(step.params.strength ?? 0.8));
      hasImageTransform = true;
      qcStr = `attenuation=${step.params.strength ?? 0.8}`;
    } else if (step.op === 'sem.background') {
      workingGray = applyBackgroundLevel(workingGray, workW, workH, workCropRows, Number(step.params.strength ?? 0.9));
      hasImageTransform = true;
      qcStr = `flattening=${step.params.strength ?? 0.9}`;
    } else if (step.op === 'sem.denoise') {
      workingGray = applyGaussianDenoise(workingGray, workW, workH, workCropRows, Number(step.params.sigma ?? 1.0));
      hasImageTransform = true;
      qcStr = `sigma=${step.params.sigma ?? 1.0}px`;
    } else if (step.op === 'sem.local_contrast') {
      const win = Number(step.params.windowPx ?? 24);
      const clip = Number(step.params.clipLimit ?? 2.5);
      const blend = Number(step.params.blend ?? 0.88);
      workingGray = applyLocalContrastNorm(workingGray, workW, workH, workCropRows, {
        windowPx: win,
        clipLimit: clip,
        blend,
      });
      hasImageTransform = true;
      qcStr = `win=${win}px · clip=${clip}σ`;
    } else if (step.op === 'sem.coherence_enhance') {
      const alongSigma = Number(step.params.alongSigma ?? 3.2);
      const tensorSigma = Number(step.params.tensorSigma ?? 3.0);
      const iterations = Number(step.params.iterations ?? 2);
      const sharpen = Number(step.params.sharpen ?? 0.45);
      workingGray = applyCoherenceEnhance(workingGray, workW, workH, workCropRows, {
        alongSigma,
        tensorSigma,
        iterations,
        sharpen,
      });
      hasImageTransform = true;
      qcStr = `along=${alongSigma}px · iter=${iterations}`;
    } else if (step.op === 'model.segment') {
      // Snapshot cumulative preprocessed image right before model inference for consistent local model training
      preModelWorkingGray = new Float32Array(workingGray);
      const ref = step.modelRef || 'vexel-sem-particles-unet@1.2';
      activeProbThreshold = Number(step.params.probThreshold ?? step.params.threshold ?? 0.5);
      if (step.params.curvatureBoundaryFit !== undefined) {
        modelCurvatureFit = Boolean(step.params.curvatureBoundaryFit);
      }
      if (step.params.overlapConfidenceThreshold !== undefined) {
        modelOverlapThreshold = Number(step.params.overlapConfidenceThreshold);
      }
      modelRefs.push(ref);
      const precomputedProb = options.precomputedProbMaps?.get(ref);
      const localModel = options.localModels?.get(ref);
      if (precomputedProb && precomputedProb.length === workW * workH) {
        activeProbMap = precomputedProb;
        qcStr = `onnx-wasm=${ref} · p≥${activeProbThreshold.toFixed(2)}`;
      } else if (localModel) {
        activeProbMap = predictLocalModel(localModel, workingGray, workW, workH, workCropRows);
        qcStr = `local=${ref} · p≥${activeProbThreshold.toFixed(2)} · IoU ${(localModel.metrics.meanIoU * 100).toFixed(1)}%`;
      } else {
        // No weights for this model: an Otsu threshold, said plainly (never presented as model inference)
        const vals: number[] = [];
        for (let i = 0; i < workW * workCropRows; i += 4) vals.push(workingGray[i]);
        const th = otsuThreshold(vals);
        const prob = new Float32Array(workW * workH);
        const invert = workSpec.subType === 'sem_holes' || workSpec.modality === 'droplet';
        for (let i = 0; i < workW * workCropRows; i++) {
          const z = (invert ? th - workingGray[i] : workingGray[i] - th) / 22;
          prob[i] = 1 / (1 + Math.exp(-z));
        }
        activeProbMap = prob;
        qcStr = `no-weights=${ref} → Otsu threshold (no model inference)`;
      }
      hasImageTransform = true;
    } else if (step.op === 'droplet.analyze') {
      const r = runDropletAnalysis(workSpec, workingGray, activeProbMap, step.params, corrections, activeProbThreshold);
      objects = r.objects;
      shapes = r.shapes;
      summary = r.summary;
      summaryMap = r.summaryMap;
      warnings.push(...r.warnings);
      histogram = r.histogram;
      poissonChart = r.poissonChart;
      qcStr = `n=${objects.length}`;
    } else if (step.op === 'sem.lines') {
      const r = runSemLinesAnalysis(workSpec, getDownstreamGray(), step.params, hasDenoise);
      objects = r.objects;
      shapes = r.shapes;
      summary = r.summary;
      summaryMap = r.summaryMap;
      warnings.push(...r.warnings);
      profileChart = r.profileChart;
      histogram = r.histogram;
      const lineRef = String(step.params.EdgeRefinement ?? 'off');
      qcStr = `lines=${objects.length}${lineRef !== 'off' ? ` · ${lineRef}` : ''}`;
    } else if (step.op === 'sem.holes') {
      const r = runSemHolesAnalysis(workSpec, workingGray, step.params, activeProbMap, activeProbThreshold);
      objects = r.objects;
      shapes = r.shapes;
      summary = r.summary;
      summaryMap = r.summaryMap;
      warnings.push(...r.warnings);
      histogram = r.histogram;
      qcStr = `holes=${objects.filter((o) => o.cls !== 'missing').length}`;
    } else if (step.op === 'sem.particles') {
      const mergedParams: Record<string, number | boolean | string> = {
        ...step.params,
        ...(modelCurvatureFit !== undefined ? { curvatureBoundaryFit: modelCurvatureFit } : {}),
        ...(modelOverlapThreshold !== undefined
          ? { overlapConfidenceThreshold: modelOverlapThreshold }
          : {}),
      };
      const r = runSemParticlesAnalysis(
        workSpec,
        workingGray,
        activeProbMap,
        mergedParams,
        corrections,
        activeProbThreshold
      );
      objects = r.objects;
      shapes = r.shapes;
      summary = r.summary;
      summaryMap = r.summaryMap;
      warnings.push(...r.warnings);
      histogram = r.histogram;
      const partRef = String(mergedParams.EdgeRefinement ?? 'off');
      qcStr = `particles=${objects.length}${partRef !== 'off' ? ` · ${partRef}` : ''}`;
    } else if (step.op === 'sem.fibres') {
      const r = runSemFibresAnalysis(workSpec, getDownstreamGray(), step.params, corrections);
      objects = r.objects;
      shapes = r.shapes;
      summary = r.summary;
      summaryMap = r.summaryMap;
      warnings.push(...r.warnings);
      histogram = r.histogram;
      profileChart = r.profileChart;
      qcStr = `fibres=${objects.length}`;
    } else if (step.op === 'cells.analyze') {
      const r = runCellAnalysis(workSpec, workingGray, activeProbMap, step.params, corrections, activeProbThreshold);
      objects = r.objects;
      shapes = r.shapes;
      summary = r.summary;
      summaryMap = r.summaryMap;
      warnings.push(...r.warnings);
      histogram = r.histogram;
      qcStr = `cells=${objects.length}`;
    } else if (step.op === 'pcb.analyze') {
      const r = runPcbAnalysis(workSpec, workingGray, step.params, corrections, activeProbMap, activeProbThreshold);
      objects = r.objects;
      shapes = r.shapes;
      summary = r.summary;
      summaryMap = r.summaryMap;
      warnings.push(...r.warnings);
      histogram = r.histogram;
      qcStr = `features=${objects.length}`;
    } else if (step.op === 'sem.multilayer') {
      const r = runMultilayerAnalysis(workSpec, getDownstreamGray(), step.params, corrections);
      objects = r.objects;
      shapes = r.shapes;
      summary = r.summary;
      summaryMap = r.summaryMap;
      warnings.push(...r.warnings);
      histogram = r.histogram;
      profileChart = r.profileChart;
      qcStr = r.qcLabel;
    } else if (step.op === 'generic.intensity_seg') {
      const r = runIntensitySegmentation(workSpec, workingGray, activeProbMap, step.params, corrections, activeProbThreshold);
      objects = r.objects;
      shapes = r.shapes;
      summary = r.summary;
      summaryMap = r.summaryMap;
      warnings.push(...r.warnings);
      histogram = r.histogram;
      qcStr = r.qcLabel;

      // Preserve continuous preprocessed grayscale for local pixel-classifier training if no model step ran yet
      if (!preModelWorkingGray) {
        preModelWorkingGray = new Float32Array(workingGray);
      }

      // Propagate segmented binary image forward so ANY downstream step uses this step's output image
      const nextGray = new Float32Array(workW * workH);
      const nextProb = new Float32Array(workW * workH);
      for (let i = 0; i < workW * workCropRows; i++) {
        const fg = r.binary[i] ? 1 : 0;
        nextGray[i] = fg ? 255 : 0;
        nextProb[i] = fg;
      }
      workingGray = nextGray;
      activeProbMap = nextProb;
      activeProbThreshold = 0.5;
      hasImageTransform = true;

      const dur = Number((performance.now() - ts).toFixed(1));
      stepPreviews.push({
        stepId: step.id,
        op: step.op,
        labelZh: spec?.labelZh ?? step.op,
        labelEn: spec?.labelEn ?? step.op,
        durationMs: dur,
        gray: toFullGray(r.maskPreview),
        qcSummary: qcStr,
      });
      stepLogs.push({ id: step.id, op: step.op, durationMs: dur, qc: qcStr });
      continue;
    } else if (step.op === 'region.parent_boundary') {
      activeParentResult = runParentRegionSegmentation(
        workingGray,
        workW,
        workCropRows,
        step.params,
        activeProbMap,
        activeProbThreshold
      );
      activeSubRegionResult = null;
      qcStr = activeParentResult.qcLabel;

      // If no downstream hierarchy metrology runs, still expose parent boundaries & areas as objects/shapes
      if (!hasHierarchyMetrology) {
        const fallback = runHierarchicalRatioMetrology(
          workSpec,
          activeParentResult,
          null,
          { minRatioPct: 0, maxRatioPct: 100 },
          corrections
        );
        objects = fallback.objects;
        shapes = fallback.shapes;
        summary = fallback.summary;
        summaryMap = fallback.summaryMap;
        warnings.push(...fallback.warnings);
        histogram = fallback.histogram;
      }

      const dur = Number((performance.now() - ts).toFixed(1));
      stepPreviews.push({
        stepId: step.id,
        op: step.op,
        labelZh: spec?.labelZh ?? step.op,
        labelEn: spec?.labelEn ?? step.op,
        durationMs: dur,
        gray: toFullGray(activeParentResult.previewGray),
        qcSummary: qcStr,
      });
      stepLogs.push({ id: step.id, op: step.op, durationMs: dur, qc: qcStr });
      continue;
    } else if (step.op === 'region.subregion_seg') {
      if (!activeParentResult) {
        activeParentResult = runParentRegionSegmentation(
          workingGray,
          workW,
          workCropRows,
          { mode: 'dark_membrane' },
          activeProbMap,
          activeProbThreshold
        );
      }
      activeSubRegionResult = runConstrainedSubRegionSegmentation(
        workingGray,
        workW,
        workCropRows,
        activeParentResult,
        step.params
      );
      qcStr = activeSubRegionResult.qcLabel;

      if (!hasHierarchyMetrology) {
        const fallback = runHierarchicalRatioMetrology(
          workSpec,
          activeParentResult,
          activeSubRegionResult,
          { minRatioPct: 0, maxRatioPct: 100 },
          corrections
        );
        objects = fallback.objects;
        shapes = fallback.shapes;
        summary = fallback.summary;
        summaryMap = fallback.summaryMap;
        warnings.push(...fallback.warnings);
        histogram = fallback.histogram;
      }

      const dur = Number((performance.now() - ts).toFixed(1));
      stepPreviews.push({
        stepId: step.id,
        op: step.op,
        labelZh: spec?.labelZh ?? step.op,
        labelEn: spec?.labelEn ?? step.op,
        durationMs: dur,
        gray: toFullGray(activeSubRegionResult.previewGray),
        qcSummary: qcStr,
      });
      stepLogs.push({ id: step.id, op: step.op, durationMs: dur, qc: qcStr });
      continue;
    } else if (step.op === 'region.hierarchy_ratio') {
      if (!activeParentResult) {
        activeParentResult = runParentRegionSegmentation(
          workingGray,
          workW,
          workCropRows,
          { mode: 'dark_membrane' },
          activeProbMap,
          activeProbThreshold
        );
      }
      if (!activeSubRegionResult) {
        activeSubRegionResult = runConstrainedSubRegionSegmentation(
          workingGray,
          workW,
          workCropRows,
          activeParentResult,
          { polarity: 'dark' }
        );
      }
      const r = runHierarchicalRatioMetrology(
        workSpec,
        activeParentResult,
        activeSubRegionResult,
        step.params,
        corrections
      );
      objects = r.objects;
      shapes = r.shapes;
      summary = r.summary;
      summaryMap = r.summaryMap;
      warnings.push(...r.warnings);
      histogram = r.histogram;
      qcStr = r.qcLabel;
    } else if (step.op === 'filter.objects') {
      const minD = Number(step.params.minDiameterPhys ?? 0);
      const minC = Number(step.params.minCircularity ?? 0);
      const before = objects.length;
      const keptIds = new Set<number>();
      objects = objects.filter((o) => {
        const passD = minD <= 0 || (o.diameter ?? o.width ?? Infinity) >= minD;
        const passC = minC <= 0 || (o.circularity ?? 1) >= minC;
        if (passD && passC) {
          keptIds.add(o.id);
          return true;
        }
        return false;
      });
      shapes = shapes.filter((s) => s.objectId === undefined || keptIds.has(s.objectId));
      qcStr = `kept ${objects.length}/${before}`;
    }

    const dur = Number((performance.now() - ts).toFixed(1));
    stepPreviews.push({
      stepId: step.id,
      op: step.op,
      labelZh: spec?.labelZh ?? step.op,
      labelEn: spec?.labelEn ?? step.op,
      durationMs: dur,
      gray: hasImageTransform ? toFullGray(workingGray) : undefined,
      probMap: step.op === 'model.segment' && activeProbMap ? toFullProb(activeProbMap) : undefined,
      qcSummary: qcStr,
    });
    stepLogs.push({ id: step.id, op: step.op, durationMs: dur, qc: qcStr });
  }

  // Scan for low-contrast / low-confidence segments (in working FOV, remapped to full frame)
  const lowConfidenceSegments = detectLowConfidenceSegments(
    preModelWorkingGray ?? workingGray,
    workW,
    workCropRows,
    objects,
    shapes,
    activeRoi ? { x: activeRoi.x, y: activeRoi.y } : undefined
  );

  // Remap ROI-local object & shape coordinates back to full specimen pixel coordinates
  if (activeRoi) {
    const { x: rx, y: ry } = activeRoi;
    objects = objects.map((o) => ({ ...o, x: o.x + rx, y: o.y + ry }));
    shapes = shapes.map((s) => remapShapeFromRoi(s, rx, ry));
  }

  // Evaluate QC rules
  for (const rule of recipe.qcRules) {
    const val = summaryMap[rule.metric];
    if (typeof val === 'number') {
      if (rule.max !== undefined && val > rule.max) {
        warnings.push(`QC 提醒: ${rule.metric} = ${val} 超过上限 ${rule.max}`);
      }
      if (rule.min !== undefined && val < rule.min) {
        warnings.push(`QC 提醒: ${rule.metric} = ${val} 低于下限 ${rule.min}`);
      }
    }
  }

  const recipeDigest = fastSha256Hex(JSON.stringify(recipe));
  const resultDigest = fastSha256Hex(
    JSON.stringify({ specimen: specimen.sha256, recipeDigest, roi: activeRoi, summaryMap, n: objects.length })
  );
  const durationMs = Number((performance.now() - tStart).toFixed(1));
  const finalProcessedGray = hasImageTransform ? toFullGray(workingGray) : undefined;
  const finalPreModelGray = preModelWorkingGray
    ? toFullGray(preModelWorkingGray)
    : finalProcessedGray;

  return {
    recipe: JSON.parse(JSON.stringify(recipe)),
    recipeDigest,
    resultDigest,
    executedAt: new Date().toISOString(),
    durationMs,
    status: warnings.length > 0 ? 'warning' : 'completed',
    summary,
    summaryMap,
    objects,
    shapes,
    stepPreviews,
    warnings,
    roi: activeRoi,
    processedGray: finalProcessedGray,
    preModelGray: finalPreModelGray,
    lowConfidenceSegments,
    histogram,
    poissonChart,
    profileChart,
    provenance: {
      engine: 'vexel-unified-2.0',
      engineVersion: '2.0.0',
      inputName: specimen.name,
      inputSha256: specimen.sha256,
      recipeId: recipe.id,
      recipeDigest,
      resultDigest,
      calibrated: specimen.calibration.spaceUnit !== 'px',
      pixelSize: specimen.calibration.pixelSize,
      spaceUnit: specimen.calibration.spaceUnit,
      modelRefs,
      correctionsApplied: corrections.filter((c) => c.specimenId === specimen.id).length,
      steps: stepLogs,
    },
  };
}

// ---------------------------------------------------------------------------
// Deterministic Result-First Fingerprint Probe & Automatic Planner
// ---------------------------------------------------------------------------

export function probeSpecimenFingerprint(specimen: Specimen) {
  const cropRows = specimen.calibration.cropRows ?? specimen.height;
  const n = specimen.width * cropRows;
  let sum = 0, sat = 0;
  for (let i = 0; i < n; i++) {
    const v = specimen.gray[i];
    sum += v;
    if (v <= 1 || v >= 254) sat++;
  }
  const mean = sum / Math.max(1, n);
  let gx = 0, gy = 0;
  for (let y = 1; y < cropRows - 1; y += 2) {
    for (let x = 1; x < specimen.width - 1; x += 2) {
      gx += Math.abs(specimen.gray[y * specimen.width + x + 1] - specimen.gray[y * specimen.width + x - 1]);
      gy += Math.abs(specimen.gray[(y + 1) * specimen.width + x] - specimen.gray[(y - 1) * specimen.width + x]);
    }
  }
  const anisotropy = Math.abs(gx - gy) / Math.max(1, gx + gy);
  return {
    name: specimen.name,
    modality: specimen.modality,
    subType: specimen.subType,
    shape: [specimen.height, specimen.width],
    cropRows,
    channels: specimen.channels,
    hasFluorescence: Boolean(specimen.fluoGray),
    vendor: specimen.calibration.vendor ?? null,
    pixelSize: specimen.calibration.pixelSize,
    spaceUnit: specimen.calibration.spaceUnit,
    meanLuminance: Number(mean.toFixed(1)),
    saturatedFraction: Number((sat / Math.max(1, n)).toFixed(4)),
    edgeAnisotropy: Number(anisotropy.toFixed(3)),
  };
}

export function buildAutoPlan(
  specimen: Specimen,
  goal: string,
  localModels: Map<string, TrainedLocalModel>,
  aiOverride?: {
    planner: PlanProposal['planner'];
    model: string;
    fallbackReason?: string;
    proposal?: {
      recipeId?: string;
      modelRef?: string | null;
      addPreprocess?: string[];
      removePreprocess?: string[];
      paramOverrides?: Array<{ stepOp: string; key: string; value: string }>;
      reasons?: string[];
    } | null;
  },
  options?: { runTrial?: boolean }
): PlanProposal {
  const runTrial = options?.runTrial ?? true;
  const lower = goal.toLowerCase();

  // Default Intensity-Based Segmentation Fallback: if an uploaded image cannot be recognized as an
  // existing domain structure workflow, suggest generic_intensity_seg (and only run trial if runTrial=true).
  if (
    !goal.trim() &&
    !aiOverride?.proposal?.recipeId &&
    (specimen.subType === 'unknown' || (specimen.importReport && !specimen.importReport.recognized))
  ) {
    const defaultSegBase =
      BUILTIN_RECIPES.find((r) => r.id === DEFAULT_INTENSITY_RECIPE_ID) ?? UNRECOGNIZED_MANUAL_RECIPE;
    const defaultSegRecipe: UnifiedRecipe = JSON.parse(JSON.stringify(defaultSegBase));
    const previewResult = runTrial
      ? executeRecipe(specimen, defaultSegRecipe, { localModels })
      : undefined;
    const rep = specimen.importReport;
    const checks: PlanProposal['checks'] = [
      {
        mark: '!',
        textZh: runTrial
          ? '未匹配为特定结构工作流，已自动启用默认「通用灰度阈值分割」'
          : '未匹配为特定结构工作流（半自动待确认，尚未自动执行）',
        textEn: runTrial
          ? 'Feature unrecognized — auto-switched to default Intensity-Based Segmentation'
          : 'Feature unrecognized — semi-auto staging (not yet executed)',
      },
    ];
    if (rep?.scaleBarCropped) {
      checks.push({
        mark: '✓',
        textZh: `已裁切 ${rep.croppedBarPx} px 信息栏/标尺区域${rep.scaleBarPx ? `（标尺 ${rep.scaleBarPx} px）` : ''}`,
        textEn: `Cropped ${rep.croppedBarPx} px info/scale bar${rep.scaleBarPx ? ` (${rep.scaleBarPx} px bar)` : ''}`,
      });
    } else if (rep?.cropStatus === 'pending' && (rep.proposedCropBarPx ?? 0) > 0) {
      checks.push({
        mark: '!',
        textZh: `检测到约 ${rep.proposedCropBarPx} px 信息栏/标尺，待确认裁切`,
        textEn: `Detected ~${rep.proposedCropBarPx} px info/scale bar — awaiting crop confirmation`,
      });
    } else {
      checks.push({
        mark: '!',
        textZh: '未检测到标准标尺栏，可手动标定像素尺度',
        textEn: 'No instrument scale bar detected; calibrate pixel size manually if needed',
      });
    }
    if (previewResult) {
      checks.push({
        mark: previewResult.objects.length > 0 ? '✓' : '!',
        textZh:
          previewResult.objects.length > 0
            ? `默认强度分割检出 ${previewResult.objects.length} 个连通区域（可微调阈值或切换手工标注）`
            : '默认阈值未检出目标，请在参数面板调整阈值/极性或使用左侧手工工具',
        textEn:
          previewResult.objects.length > 0
            ? `Default intensity segmentation detected ${previewResult.objects.length} regions (tune threshold or annotate manually)`
            : 'No regions detected at default threshold — adjust threshold/polarity or use manual tools',
      });
    } else {
      checks.push({
        mark: '!',
        textZh: '请先确认信息栏裁切、物理标定并选定工作流后再启动执行',
        textEn: 'Confirm info-bar crop, calibration, and workflow selection before running',
      });
    }

    return {
      status: runTrial ? 'check' : 'needs_user',
      planner: rep?.source === 'vision' ? 'gemini' : 'rules',
      modelName: rep?.source === 'vision' ? 'gemini-4-hydrogen' : 'offline-deterministic-rules',
      modality: specimen.modality,
      recipeId: defaultSegRecipe.id,
      recipeNameZh: defaultSegRecipe.nameZh,
      recipeNameEn: defaultSegRecipe.nameEn,
      confidence: rep?.confidence ?? 0.35,
      checks,
      reasonsZh: [
        rep?.reasonZh ?? '当前图像未匹配到微流控、细胞、PCB 或 SEM 专用结构工作流，参考推荐默认灰度强度阈值分割工作流。',
        '请按步骤确认信息栏裁切与物理标定，选定工作流或调整参数后点击「执行分析」。',
      ],
      reasonsEn: [
        rep?.reasonEn ?? 'Feature did not match existing domain workflows; suggesting default Intensity-Based Segmentation.',
        'Confirm info-bar crop and physical calibration, choose a workflow or tune parameters, then click Run.',
      ],
      suggestedRecipe: defaultSegRecipe,
      previewResult,
    };
  }

  let chosenId = 'sem_particles';

  if (aiOverride?.proposal?.recipeId && BUILTIN_RECIPES.some((r) => r.id === aiOverride.proposal!.recipeId)) {
    chosenId = aiOverride.proposal.recipeId;
  } else if (/细胞核|核质比|面积比|面积占比|子区|嵌套|肾|sub-?region|nested|hierarch|nuclei.*ratio|nucleus.*ratio|kidney/i.test(lower)) {
    chosenId = 'nested_cell_nucleus';
  } else if (/弯曲|弧|同心|不均匀|弱对比|各向异性|curved|arc|uneven|coherence|concentric/i.test(lower)) {
    chosenId = 'sem_curved_multilayer';
  } else if (/层数|层厚|多层|膜厚|截面|涂层|堆叠|lamella|multilayer|layer|thickness|stack|coating|cross.?section/i.test(lower)) {
    chosenId = 'sem_multilayer';
  } else if (/阈值|灰度|通用分割|intensity|threshold|otsu/i.test(lower)) {
    chosenId = DEFAULT_INTENSITY_RECIPE_ID;
  } else if (/pcb|焊盘|走线|电路板|trace|pad|solder/i.test(lower)) {
    chosenId = 'pcb_inspection';
  } else if (/荧光|阳性|ddpcr|fluo|positive/i.test(lower)) {
    chosenId = 'droplet_fluorescence';
  } else if (/液滴|微流控|微滴|泊松|droplet|poisson/i.test(lower)) {
    chosenId = 'droplet_encapsulation';
  } else if (/细胞|细胞核|融合度|cell|nuclei|confluency/i.test(lower)) {
    chosenId =
      specimen.subType === 'droplet_encapsulation'
        ? 'droplet_encapsulation'
        : specimen.subType === 'kidney_nested_cells'
        ? 'nested_cell_nucleus'
        : 'cell_segmentation';
  } else if (/线宽|线条|粗糙度|ler|lwr|\bcd\b|line|pitch/i.test(lower)) {
    chosenId = 'sem_lines';
  } else if (/接触孔|缺孔|孔阵列|via|hole/i.test(lower)) {
    chosenId = 'sem_holes';
  } else if (/纤维|纺丝|fibre|fiber|align/i.test(lower)) {
    chosenId = 'sem_fibres';
  } else if (/颗粒|粒径|团聚|异物|particle|defect/i.test(lower)) {
    chosenId = 'sem_particles';
  } else if (specimen.importReport?.suggestedRecipeId && BUILTIN_RECIPES.some((r) => r.id === specimen.importReport!.suggestedRecipeId)) {
    chosenId = specimen.importReport.suggestedRecipeId;
  } else {
    // Auto-select from specimen probe
    const map: Record<Specimen['subType'], string> = {
      droplet_encapsulation: 'droplet_encapsulation',
      droplet_fluorescence: 'droplet_fluorescence',
      cells_count: 'cell_segmentation',
      kidney_nested_cells: 'nested_cell_nucleus',
      pcb_traces: 'pcb_inspection',
      sem_lines: 'sem_lines',
      sem_holes: 'sem_holes',
      sem_particles: 'sem_particles',
      sem_pores: 'sem_particles',
      sem_fibres: 'sem_fibres',
      sem_multilayer: 'sem_multilayer',
      unknown: DEFAULT_INTENSITY_RECIPE_ID,
      uploaded: DEFAULT_INTENSITY_RECIPE_ID,
    };
    chosenId = map[specimen.subType];
  }

  const base = BUILTIN_RECIPES.find((r) => r.id === chosenId) ?? BUILTIN_RECIPES[0];
  const suggestedRecipe: UnifiedRecipe = JSON.parse(JSON.stringify(base));

  // Apply AI proposal overrides safely against operator catalog
  const prop = aiOverride?.proposal;
  if (prop) {
    for (const addOp of prop.addPreprocess ?? []) {
      const existing = suggestedRecipe.steps.find((s) => s.op === addOp);
      if (existing) {
        existing.enabled = true;
      } else {
        const spec = OPERATOR_CATALOG.find((o) => o.op === addOp);
        if (spec && !(chosenId === 'sem_lines' && addOp === 'sem.denoise')) {
          const defaults: Record<string, number | boolean | string> = {};
          for (const p of spec.params) defaults[p.key] = p.defaultValue;
          suggestedRecipe.steps.unshift({ id: `ai_${addOp.replace('.', '_')}`, op: addOp, enabled: true, params: defaults });
        }
      }
    }
    for (const remOp of prop.removePreprocess ?? []) {
      const existing = suggestedRecipe.steps.find((s) => s.op === remOp);
      if (existing) existing.enabled = false;
    }
    if (prop.modelRef) {
      const mStep = suggestedRecipe.steps.find((s) => s.op === 'model.segment');
      if (mStep) {
        mStep.enabled = true;
        mStep.modelRef = prop.modelRef;
      }
    }
    for (const ov of prop.paramOverrides ?? []) {
      const st = suggestedRecipe.steps.find((s) => s.op === ov.stepOp);
      const spec = OPERATOR_CATALOG.find((o) => o.op === ov.stepOp);
      const pDef = spec?.params.find((p) => p.key === ov.key);
      if (st && pDef) {
        if (pDef.type === 'number') {
          const num = Number(ov.value);
          if (Number.isFinite(num)) {
            st.params[ov.key] = Math.max(pDef.min ?? -1e9, Math.min(pDef.max ?? 1e9, num));
          }
        } else if (pDef.type === 'boolean') {
          st.params[ov.key] = ov.value === 'true';
        } else if (!pDef.choices || pDef.choices.some((c) => c.value === ov.value)) {
          st.params[ov.key] = ov.value;
        }
      }
    }
  }

  // Run instant trial execution ONLY when runTrial is true (disabled for semi-auto single-image import)
  const previewResult = runTrial
    ? executeRecipe(specimen, suggestedRecipe, { localModels })
    : undefined;

  const checks: PlanProposal['checks'] = [];
  if (specimen.importReport?.scaleBarCropped) {
    checks.push({
      mark: '✓',
      textZh: `已确认裁切 ${specimen.importReport.croppedBarPx} px 信息栏/标尺${
        specimen.importReport.scaleBarPx ? `（标尺 ${specimen.importReport.scaleBarPx} px）` : ''
      }`,
      textEn: `Cropped ${specimen.importReport.croppedBarPx} px info/scale bar${
        specimen.importReport.scaleBarPx ? ` (${specimen.importReport.scaleBarPx} px bar)` : ''
      }`,
    });
  } else if (
    specimen.importReport?.cropStatus === 'pending' &&
    (specimen.importReport.proposedCropBarPx ?? 0) > 0
  ) {
    checks.push({
      mark: '!',
      textZh: `检测到约 ${specimen.importReport.proposedCropBarPx} px 信息栏区域（已在画布框出，待确认裁切）`,
      textEn: `Detected ~${specimen.importReport.proposedCropBarPx} px info bar (outlined on canvas, awaiting confirmation)`,
    });
  }
  if (specimen.calibration.confirmed) {
    checks.push({
      mark: '✓',
      textZh: `物理标定已确认 (${specimen.calibration.pixelSize} ${specimen.calibration.spaceUnit}/px)`,
      textEn: `Calibrated (${specimen.calibration.pixelSize} ${specimen.calibration.spaceUnit}/px)`,
    });
  } else {
    checks.push({
      mark: '!',
      textZh: '标定待人工确认（请核对像素尺度或标尺数值后确认）',
      textEn: 'Calibration awaits user confirmation',
    });
  }
  if (previewResult) {
    if (previewResult.objects.length > 0) {
      checks.push({
        mark: '✓',
        textZh: `试运行检出 ${previewResult.objects.length} 个目标 (${previewResult.durationMs} ms)`,
        textEn: `Trial detected ${previewResult.objects.length} objects (${previewResult.durationMs} ms)`,
      });
    } else {
      checks.push({
        mark: '✗',
        textZh: '试运行未检出目标，建议调整阈值或启用画笔训练',
        textEn: 'No objects detected in trial; adjust parameters or train a brush model',
      });
    }
    if (previewResult.warnings.length > 0) {
      checks.push({
        mark: '!',
        textZh: previewResult.warnings[0],
        textEn: `${previewResult.warnings.length} QC advisory note(s) generated`,
      });
    }
  } else {
    checks.push({
      mark: '!',
      textZh: `半自动待确认：参考推荐「${suggestedRecipe.nameZh}」，选定或调参后点击执行`,
      textEn: `Semi-auto pending: suggested "${suggestedRecipe.nameEn}" — confirm or tune before running`,
    });
  }

  const sceneZh =
    specimen.importReport?.sceneLabelZh ??
    (specimen.modality === 'sem' ? 'SEM 扫描电镜' : '微流控液滴');
  const sceneEn =
    specimen.importReport?.sceneLabelEn ??
    (specimen.modality === 'sem' ? 'SEM' : 'Microfluidics');

  const defaultReasonsZh = [
    specimen.importReport?.reasonZh ??
      `参考识别为「${sceneZh}」，建议工作流：${suggestedRecipe.nameZh}`,
    specimen.importReport?.scaleBarCropped
      ? `已确认裁切 ${specimen.importReport.croppedBarPx} px 信息栏/标尺（原高 ${specimen.importReport.originalHeight} px → 有效视场 ${specimen.height} px）`
      : specimen.importReport?.cropStatus === 'pending' && (specimen.importReport.proposedCropBarPx ?? 0) > 0
      ? `已自动框出 ${specimen.importReport.proposedCropBarPx} px 信息栏/标尺区域，等待用户确认后再执行裁切`
      : `图像有效视场 ${specimen.width}×${specimen.height} px`,
    chosenId === 'sem_lines'
      ? '已配置逐行校平与去条纹，并禁止在 LER/LWR 粗糙度测量前执行去噪'
      : '全部中间步骤与物理测量值均由确定性算子计算并锁定 SHA-256 摘要',
  ];
  const defaultReasonsEn = [
    specimen.importReport?.reasonEn ??
      `Tentatively classified as "${sceneEn}"; suggested workflow: ${suggestedRecipe.nameEn}`,
    specimen.importReport?.scaleBarCropped
      ? `Confirmed ${specimen.importReport.croppedBarPx} px info/scale bar crop (${specimen.importReport.originalHeight} px → ${specimen.height} px working FOV)`
      : specimen.importReport?.cropStatus === 'pending' && (specimen.importReport.proposedCropBarPx ?? 0) > 0
      ? `Outlined ${specimen.importReport.proposedCropBarPx} px info/scale bar on canvas — awaiting user confirmation`
      : `Active field of view: ${specimen.width}×${specimen.height} px`,
    chosenId === 'sem_lines'
      ? 'Configured scan-line levelling & destriping; denoising disabled to protect unbiased LER/LWR'
      : 'All intermediate steps and physical metrics computed deterministically with SHA-256 digest',
  ];

  const status: PlanProposal['status'] = !previewResult
    ? 'needs_user'
    : previewResult.objects.length === 0
    ? 'needs_user'
    : previewResult.warnings.length > 0 || !specimen.calibration.confirmed
    ? 'check'
    : 'ready';

  return {
    status,
    planner: aiOverride?.planner ?? (specimen.importReport?.source === 'vision' ? 'gemini' : 'rules'),
    modelName:
      aiOverride?.model ??
      (specimen.importReport?.source === 'vision' ? 'gemini-4-hydrogen' : 'offline-deterministic-rules'),
    modality: specimen.modality,
    recipeId: suggestedRecipe.id,
    recipeNameZh: suggestedRecipe.nameZh,
    recipeNameEn: suggestedRecipe.nameEn,
    confidence:
      specimen.importReport?.confidence ?? (previewResult && previewResult.objects.length > 0 ? 0.94 : 0.45),
    checks,
    reasonsZh: prop?.reasons?.length ? prop.reasons : defaultReasonsZh,
    reasonsEn: prop?.reasons?.length ? prop.reasons : defaultReasonsEn,
    suggestedRecipe,
    previewResult,
    fallbackNotice: aiOverride?.fallbackReason,
  };
}
