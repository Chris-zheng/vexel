import type { ResourceBudgetReport, ResourceBudgetTier } from '../types/vexel';

/**
 * Evaluates 2D image memory footprint and processing tier based on the approved
 * Vexel browser resource budget matrix:
 * - <= 2048×2048 (4 MP): optimal real-time interactive
 * - <= 4096×4096 (16 MP): tiled_safe (512×512 feathered tiles in Web Worker)
 * - <= 8192×8192 (64 MP): lod_guarded (pyramid display + tiled ROI analysis)
 * - > 8192×8192: offload_gpu_required (local/cloud OME-TIFF/Zarr backend)
 */
export function evaluate2DBudget(
  width: number,
  height: number,
  task: '2d_train' | '2d_infer' = '2d_train'
): ResourceBudgetReport {
  const safeW = Math.max(1, Math.round(width));
  const safeH = Math.max(1, Math.round(height));
  const pixels = safeW * safeH;
  // Raw RGBA (4 bytes) + Float32 gray (4 bytes)
  const rawMB = (pixels * 8) / (1024 * 1024);

  // Tiled 512×512 (64px halo => 640×640) × 16 float channels (~26 MB workspace) + image buffers
  const tileWorkspaceMB = task === '2d_train' ? 42 : 28;
  const fullPeakMB = rawMB + tileWorkspaceMB + (pixels > 4_200_000 ? 48 : 18);

  let tier: ResourceBudgetTier = 'optimal';
  let tileSize = 512;
  let overlapPx = 64;
  let downsampleFactor = 1;
  let recommendationZh = '最佳实时交互区间：浏览器 Web Worker 少样本深度微调与全图推理流畅运行。';
  let recommendationEn = 'Optimal real-time zone: Browser Web Worker few-shot DL fine-tuning and inference run smoothly.';

  if (pixels <= 4_194_304) {
    tier = 'optimal';
    tileSize = 512;
    overlapPx = 64;
    downsampleFactor = 1;
  } else if (pixels <= 16_777_216) {
    tier = 'tiled_safe';
    tileSize = 512;
    overlapPx = 64;
    downsampleFactor = 1;
    recommendationZh = '高精安全上限（≤4096²）：已自动启用 512×512 羽化瓦片流式推理，防止内存峰值溢出。';
    recommendationEn = 'High-res safe zone (≤4096²): Automatic 512×512 feathered tiled streaming keeps memory bounded.';
  } else if (pixels <= 67_108_864) {
    tier = 'lod_guarded';
    tileSize = 512;
    overlapPx = 64;
    downsampleFactor = 2;
    recommendationZh = '接近浏览器硬天花板（≤8192²）：建议框选局部 ROI 进行少样本微调，并开启分块流式推理。';
    recommendationEn = 'Near browser hard ceiling (≤8192²): Scope training to a manual ROI and use tiled streaming inference.';
  } else {
    tier = 'offload_gpu_required';
    tileSize = 512;
    overlapPx = 64;
    downsampleFactor = 4;
    recommendationZh = '超出浏览器安全线（>64 MP）：建议切换本地工作站或云端 GPU 后端进行 OME-TIFF/Zarr 分块分析。';
    recommendationEn = 'Exceeds browser safe memory (>64 MP): Offload to Local/Cloud GPU backend with OME-TIFF/Zarr chunking.';
  }

  return {
    task,
    dimensions: { width: safeW, height: safeH, depth: 1 },
    elementCount: pixels,
    rawMemoryMB: Number(rawMB.toFixed(1)),
    estimatedPeakMB: Number(fullPeakMB.toFixed(1)),
    tier,
    recommendedTileSize: tileSize,
    recommendedOverlapPx: overlapPx,
    recommendedDownsampleFactor: downsampleFactor,
    recommendationZh,
    recommendationEn,
  };
}

/**
 * Evaluates 3D Z-Stack volume memory budget and WebGL2 Data3DTexture LOD limits:
 * - <= 512×512×64 (16.8M voxels): optimal 60fps native 3D raymarching + <1.2s CCA
 * - <= 512×512×128 (33.5M voxels): tiled_safe (Uint16 labels + adaptive ray steps)
 * - <= 1024×1024×64 (67.1M voxels): lod_guarded (auto-downsample render texture to <=512×512×128)
 * - > 67.1M voxels: offload_gpu_required (OME-Zarr out-of-core GPU worker)
 */
export function evaluate3DBudget(
  width: number,
  height: number,
  depth: number,
  task: '3d_render' | '3d_analyze' = '3d_analyze'
): ResourceBudgetReport {
  const safeW = Math.max(1, Math.round(width));
  const safeH = Math.max(1, Math.round(height));
  const safeD = Math.max(1, Math.round(depth));
  const voxels = safeW * safeH * safeD;

  // Compact Uint8 volume (1B/vx) + Uint16 labels (2B/vx) + GPU Data3DTexture upload + scratch
  const rawMB = voxels / (1024 * 1024);
  const labelMB = (voxels * 2) / (1024 * 1024);
  const gpuTexMB = voxels / (1024 * 1024);
  const peakMB = rawMB + (task === '3d_analyze' ? labelMB * 1.45 + 32 : gpuTexMB * 1.5 + 28);

  let tier: ResourceBudgetTier = 'optimal';
  let downsampleFactor = 1;
  let recommendationZh = '最佳 3D 交互区间：WebGL2 原生全分辨率 Data3DTexture 光线步进与秒级三维连通域分析。';
  let recommendationEn = 'Optimal 3D zone: Native full-resolution WebGL2 Data3DTexture raymarching & sub-second 3D metrology.';

  if (voxels <= 16_777_216 && safeW <= 512 && safeH <= 512 && safeD <= 64) {
    tier = 'optimal';
    downsampleFactor = 1;
  } else if (voxels <= 33_554_432 && safeW <= 512 && safeH <= 512 && safeD <= 128) {
    tier = 'tiled_safe';
    downsampleFactor = 1;
    recommendationZh = '3D 高精安全上限（≤512×512×128）：采用紧凑 Uint16 标签与自适应步长渲染，内存稳定受控。';
    recommendationEn = 'High-detail 3D safe limit (≤512×512×128): Uses compact Uint16 labels & adaptive raymarch steps.';
  } else if (voxels <= 67_108_864) {
    tier = 'lod_guarded';
    downsampleFactor = 2;
    recommendationZh = '触及浏览器 3D 硬上限：自动启用显示 LOD 2× 降采样（纹理≤512×512×128）与 Z-Slab 分段分析，保障零崩页。';
    recommendationEn = 'Near browser 3D ceiling: Auto-enabled 2× display LOD (≤512×512×128) + Z-slab streaming analysis.';
  } else {
    tier = 'offload_gpu_required';
    downsampleFactor = Math.max(2, Math.ceil(Math.cbrt(voxels / 16_777_216)));
    recommendationZh = '超出浏览器体数据上限（>67M 体素）：已强制降采样用于浏览器预览，全精度分析建议外移至本地/云端 OME-Zarr GPU 后端。';
    recommendationEn = 'Exceeds browser volume cap (>67M voxels): Downsampled for preview; offload full-res analysis to OME-Zarr GPU backend.';
  }

  return {
    task,
    dimensions: { width: safeW, height: safeH, depth: safeD },
    elementCount: voxels,
    rawMemoryMB: Number(rawMB.toFixed(1)),
    estimatedPeakMB: Number(peakMB.toFixed(1)),
    tier,
    recommendedTileSize: 512,
    recommendedOverlapPx: 32,
    recommendedDownsampleFactor: downsampleFactor,
    recommendationZh,
    recommendationEn,
  };
}
