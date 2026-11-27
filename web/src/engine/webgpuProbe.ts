/**
 * Browser WebGPU / WebGL2 hardware acceleration capability probe.
 * Safely detects `navigator.gpu` availability, adapter limits, and recommends
 * optimal execution providers for ONNX visual encoders & 3D volume rendering.
 */

export interface HardwareAccelProfile {
  webgpuSupported: boolean;
  webgl2Supported: boolean;
  adapterName: string | null;
  maxTextureDimension3D: number;
  maxStorageBufferMB: number;
  recommended3DBackend: 'webgpu-ready' | 'webgl2' | 'cpu-fallback';
  recommendedOnnxProviders: Array<'webgpu' | 'wasm'>;
  summaryZh: string;
  summaryEn: string;
}

let cachedProfile: HardwareAccelProfile | null = null;

function probeWebGL2(): { ok: boolean; max3dTex: number } {
  if (typeof document === 'undefined') return { ok: false, max3dTex: 0 };
  try {
    const canvas = document.createElement('canvas');
    const gl = canvas.getContext('webgl2');
    if (!gl) return { ok: false, max3dTex: 0 };
    const max3dTex = Number(gl.getParameter(gl.MAX_3D_TEXTURE_SIZE) || 2048);
    gl.getExtension('WEBGL_lose_context')?.loseContext();
    return { ok: true, max3dTex };
  } catch {
    return { ok: false, max3dTex: 0 };
  }
}

export async function probeHardwareAcceleration(): Promise<HardwareAccelProfile> {
  if (cachedProfile) return cachedProfile;

  const gl2 = probeWebGL2();
  let webgpuSupported = false;
  let adapterName: string | null = null;
  let maxTex3D = gl2.max3dTex || 2048;
  let maxStorageMB = 128;

  if (typeof navigator !== 'undefined' && 'gpu' in navigator) {
    try {
      const gpuObj = (
        navigator as unknown as {
          gpu?: {
            requestAdapter: () => Promise<{
              limits?: { maxTextureDimension3D?: number; maxStorageBufferBindingSize?: number };
              info?: { description?: string; vendor?: string; architecture?: string };
            } | null>;
          };
        }
      ).gpu;
      const adapter = await gpuObj?.requestAdapter();
      if (adapter) {
        webgpuSupported = true;
        const info = adapter.info;
        adapterName =
          info?.description ||
          [info?.vendor, info?.architecture].filter(Boolean).join(' ') ||
          'WebGPU Adapter';
        if (adapter.limits?.maxTextureDimension3D) {
          maxTex3D = adapter.limits.maxTextureDimension3D;
        }
        if (adapter.limits?.maxStorageBufferBindingSize) {
          maxStorageMB = Math.round(adapter.limits.maxStorageBufferBindingSize / (1024 * 1024));
        }
      }
    } catch {
      webgpuSupported = false;
    }
  }

  const recommended3DBackend: HardwareAccelProfile['recommended3DBackend'] = webgpuSupported
    ? 'webgpu-ready'
    : gl2.ok
    ? 'webgl2'
    : 'cpu-fallback';

  const recommendedOnnxProviders: Array<'webgpu' | 'wasm'> = webgpuSupported
    ? ['webgpu', 'wasm']
    : ['wasm'];

  cachedProfile = {
    webgpuSupported,
    webgl2Supported: gl2.ok,
    adapterName,
    maxTextureDimension3D: maxTex3D,
    maxStorageBufferMB: maxStorageMB,
    recommended3DBackend,
    recommendedOnnxProviders,
    summaryZh: webgpuSupported
      ? `WebGPU 就绪 (${adapterName} · 3D 纹理上限 ${maxTex3D} · 存储缓冲 ${maxStorageMB} MB)，ONNX 编码器支持 WebGPU/WASM 双加速`
      : gl2.ok
      ? `WebGL2 硬件加速就绪 (3D 纹理上限 ${maxTex3D})，ONNX 编码器运行于离线 WASM SIMD`
      : '仅 CPU 回退模式',
    summaryEn: webgpuSupported
      ? `WebGPU ready (${adapterName} · 3D Tex ${maxTex3D} · Storage ${maxStorageMB} MB); ONNX supports WebGPU/WASM`
      : gl2.ok
      ? `WebGL2 accelerated (3D Tex ${maxTex3D}); ONNX runs on offline WASM SIMD`
      : 'CPU fallback mode',
  };

  return cachedProfile;
}
