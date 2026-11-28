import type {
  Calibration,
  DetectedSceneType,
  ImportRecognitionReport,
  Specimen,
} from '../types/vexel';
import {
  decodeGatanDmBuffer,
  decodeTiffBuffer,
  detectProprietaryMicroscopyContainer,
  isNativeScientificBinary,
} from './microscopyDecoder';

/** Deterministic PRNG (Mulberry32) so synthetic ground-truth specimens are 100% reproducible. */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function gaussNoise(rng: () => number): number {
  const u1 = Math.max(1e-6, rng());
  const u2 = rng();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

export function fastSha256Hex(data: Uint8Array | Uint8ClampedArray | string): string {
  // Deterministic FNV-1a 256-bit-style hex digest for synchronous client IDs (real SHA-256 async available in exporters)
  const bytes = typeof data === 'string' ? new TextEncoder().encode(data) : data;
  const h = [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19];
  const step = Math.max(1, Math.floor(bytes.length / 8192));
  for (let i = 0; i < bytes.length; i += step) {
    const b = bytes[i];
    for (let k = 0; k < 8; k++) {
      h[k] = Math.imul(h[k] ^ (b + k * 31 + (i & 255)), 0x01000193) >>> 0;
    }
  }
  return h.map((v) => v.toString(16).padStart(8, '0')).join('');
}

function grayToRgba(gray: Float32Array, width: number, height: number, tint?: [number, number, number]): Uint8ClampedArray {
  const rgba = new Uint8ClampedArray(width * height * 4);
  const tr = tint ? tint[0] : 1;
  const tg = tint ? tint[1] : 1;
  const tb = tint ? tint[2] : 1;
  for (let i = 0; i < width * height; i++) {
    const v = Math.max(0, Math.min(255, Math.round(gray[i])));
    rgba[i * 4] = Math.min(255, Math.round(v * tr));
    rgba[i * 4 + 1] = Math.min(255, Math.round(v * tg));
    rgba[i * 4 + 2] = Math.min(255, Math.round(v * tb));
    rgba[i * 4 + 3] = 255;
  }
  return rgba;
}

function drawDataBar(
  gray: Float32Array,
  width: number,
  height: number,
  barRows: number,
  scalePx: number
): void {
  const y0 = height - barRows;
  for (let y = y0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      gray[y * width + x] = y === y0 ? 90 : 12;
    }
  }
  // Draw crisp white scale bar in lower-right of data bar
  const sx1 = width - 28;
  const sx0 = Math.max(width - 28 - scalePx, 40);
  const sy = y0 + Math.floor(barRows * 0.56);
  for (let y = sy - 2; y <= sy + 2; y++) {
    for (let x = sx0; x <= sx1; x++) {
      gray[y * width + x] = 242;
    }
  }
  // End ticks
  for (let y = sy - 6; y <= sy + 6; y++) {
    for (const tx of [sx0, sx0 + 1, sx1 - 1, sx1]) {
      if (tx >= 0 && tx < width && y >= y0 && y < height) {
        gray[y * width + tx] = 242;
      }
    }
  }
}

export function buildDropletEncapsulationSpecimen(seed = 101, suffix = ''): Specimen {
  const width = 768;
  const height = 320;
  const rng = mulberry32(seed);
  const gray = new Float32Array(width * height);

  // Microfluidic channel walls (dark bands top/bottom, bright laminar fluid interior)
  const wallTop = 44;
  const wallBot = height - 44;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let base = 224 - 10 * Math.abs((y - height / 2) / (height / 2)) + 4 * Math.sin(x / 110);
      if (y < wallTop || y >= wallBot) {
        base = 24 + 6 * rng();
      } else if (Math.abs(y - wallTop) < 5 || Math.abs(y - wallBot) < 5) {
        base = 88;
      }
      gray[y * width + x] = Math.max(0, Math.min(255, base + gaussNoise(rng) * 3.8));
    }
  }

  // Place 9 droplets across two rows or staggered channel flow with realistic dark optical ring + bright interior + dark encapsulated cells
  const cellCounts = [0, 1, 2, 1, 0, 3, 1, 0, 2, 1, 0, 1];
  const centers: Array<{ cx: number; cy: number; r: number; cells: number; odd?: boolean }> = [];
  for (let i = 0; i < 10; i++) {
    const row = i % 2;
    const cx = 68 + i * 68 + (rng() - 0.5) * 8;
    const cy = row === 0 ? 114 + (rng() - 0.5) * 14 : 206 + (rng() - 0.5) * 14;
    const r = i === 8 ? 22.5 : 15.2 + (rng() - 0.5) * 1.4; // #8 is a slightly oversized droplet for QC flag realism
    centers.push({ cx, cy, r, cells: cellCounts[(i + seed) % cellCounts.length], odd: i === 8 });
  }

  for (const d of centers) {
    const rOuter = d.r + 3.5;
    const x0 = Math.max(0, Math.floor(d.cx - rOuter));
    const x1 = Math.min(width - 1, Math.ceil(d.cx + rOuter));
    const y0 = Math.max(wallTop + 2, Math.floor(d.cy - rOuter));
    const y1 = Math.min(wallBot - 2, Math.ceil(d.cy + rOuter));
    for (let y = y0; y <= y1; y++) {
      for (let x = x0; x <= x1; x++) {
        const dist = Math.hypot(x - d.cx, y - d.cy);
        if (dist <= d.r + 2.6) {
          const ring = Math.exp(-Math.pow((dist - (d.r - 1.1)) / 2.2, 2));
          let val = 236 - 178 * ring + gaussNoise(rng) * 2.5;
          if (dist < d.r - 3.2) {
            val = 238 - 14 * (dist / d.r) + gaussNoise(rng) * 2.2;
          }
          gray[y * width + x] = Math.max(8, Math.min(252, val));
        }
      }
    }
    // Paint dark cells inside droplet
    for (let c = 0; c < d.cells; c++) {
      const ang = (c / Math.max(1, d.cells)) * Math.PI * 2 + seed * 0.7;
      const rad = d.cells === 1 ? 2.5 * rng() : d.r * 0.36;
      const px = d.cx + Math.cos(ang) * rad;
      const py = d.cy + Math.sin(ang) * rad;
      const cr = 3.1 + rng() * 0.7;
      for (let y = Math.floor(py - cr - 2); y <= Math.ceil(py + cr + 2); y++) {
        for (let x = Math.floor(px - cr - 2); x <= Math.ceil(px + cr + 2); x++) {
          if (x < 0 || x >= width || y < 0 || y >= height) continue;
          const dist = Math.hypot(x - px, y - py);
          if (dist < cr + 1.5) {
            const spot = Math.exp(-Math.pow(dist / cr, 2) * 1.4);
            gray[y * width + x] = Math.max(15, gray[y * width + x] - 132 * spot);
          }
        }
      }
    }
  }

  const rgba = grayToRgba(gray, width, height);
  return {
    id: `droplet_encap_${seed}${suffix}`,
    name: `droplets_encap_60um${suffix}.ome.tif`,
    modality: 'droplet',
    subType: 'droplet_encapsulation',
    width,
    height,
    frames: 600,
    channels: 1,
    rgba,
    gray,
    calibration: {
      pixelSize: 2.0,
      spaceUnit: 'µm',
      frameIntervalMs: 1.0,
      channelHeightUm: 50,
      source: 'metadata',
      confirmed: true,
    },
    sha256: fastSha256Hex(rgba),
    descriptionZh: '微流控单分散液滴细胞包裹明场样本（60 µm 液滴，泊松 λ ≈ 1.0，含大粒径异常滴）',
    descriptionEn: 'Brightfield microfluidic droplet cell encapsulation sample (60 µm droplets, Poisson λ ≈ 1.0)',
  };
}

export function buildDropletFluorescenceSpecimen(seed = 202, suffix = ''): Specimen {
  const width = 768;
  const height = 320;
  const rng = mulberry32(seed);
  const gray = new Float32Array(width * height);
  const fluoGray = new Float32Array(width * height);

  const wallTop = 44;
  const wallBot = height - 44;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let base = y < wallTop || y >= wallBot ? 24 : 220;
      gray[y * width + x] = Math.max(0, Math.min(255, base + gaussNoise(rng) * 3.5));
      fluoGray[y * width + x] = Math.max(0, Math.min(255, 16 + Math.abs(gaussNoise(rng) * 4.5)));
    }
  }

  const positiveFlags = [true, false, true, true, false, false, true, false, true, false];
  for (let i = 0; i < 10; i++) {
    const row = i % 2;
    const cx = 68 + i * 68 + (rng() - 0.5) * 6;
    const cy = row === 0 ? 116 + (rng() - 0.5) * 10 : 204 + (rng() - 0.5) * 10;
    const r = 15.0 + (rng() - 0.5) * 1.1;
    const isPos = positiveFlags[(i + seed) % positiveFlags.length];
    const fluoLevel = isPos ? 185 + rng() * 35 : 30 + rng() * 14;

    for (let y = Math.floor(cy - r - 4); y <= Math.ceil(cy + r + 4); y++) {
      for (let x = Math.floor(cx - r - 4); x <= Math.ceil(cx + r + 4); x++) {
        if (x < 0 || x >= width || y < wallTop + 2 || y >= wallBot - 2) continue;
        const dist = Math.hypot(x - cx, y - cy);
        if (dist <= r + 2.5) {
          const ring = Math.exp(-Math.pow((dist - (r - 1.1)) / 2.1, 2));
          gray[y * width + x] = Math.max(12, Math.min(250, 232 - 170 * ring + gaussNoise(rng) * 2.5));
        }
        if (dist <= r - 1.5) {
          const glow = Math.cos((dist / r) * (Math.PI / 2.2));
          fluoGray[y * width + x] = Math.min(255, fluoGray[y * width + x] + fluoLevel * glow);
        }
      }
    }
  }

  const rgba = grayToRgba(gray, width, height);
  const fluoRgba = grayToRgba(fluoGray, width, height, [0.15, 1.0, 0.45]);
  return {
    id: `droplet_fluo_${seed}${suffix}`,
    name: `ddpcr_fluo_assay${suffix}.ome.tif`,
    modality: 'droplet',
    subType: 'droplet_fluorescence',
    width,
    height,
    frames: 400,
    channels: 2,
    rgba,
    fluoRgba,
    gray,
    fluoGray,
    calibration: {
      pixelSize: 2.0,
      spaceUnit: 'µm',
      frameIntervalMs: 1.0,
      channelHeightUm: 50,
      source: 'metadata',
      confirmed: true,
    },
    sha256: fastSha256Hex(rgba),
    descriptionZh: '数字 PCR 双通道荧光液滴样本（明场轮廓 + 绿色荧光通道，阴阳双峰分布）',
    descriptionEn: 'Dual-channel digital PCR fluorescence droplets (brightfield + GFP channel, bimodal intensity)',
  };
}

export function buildSemLinesSpecimen(seed = 303, suffix = ''): Specimen {
  const width = 640;
  const height = 480;
  const barRows = 44;
  const imgH = height - barRows;
  const rng = mulberry32(seed);
  const gray = new Float32Array(width * height);

  // Correlated line-edge roughness (LER) random walks per line edge
  const nLines = 7;
  const pitch = 74;
  const baseCD = 32;
  const startX = 68;
  const leftEdges: Float32Array[] = [];
  const rightEdges: Float32Array[] = [];

  for (let l = 0; l < nLines; l++) {
    const le = new Float32Array(imgH);
    const re = new Float32Array(imgH);
    let wl = 0;
    let wr = 0;
    const cx = startX + l * pitch + (rng() - 0.5) * 1.8;
    const cd = baseCD + (rng() - 0.5) * 1.4;
    for (let y = 0; y < imgH; y++) {
      wl = 0.88 * wl + gaussNoise(rng) * 0.55;
      wr = 0.88 * wr + gaussNoise(rng) * 0.55;
      le[y] = cx - cd / 2 + wl;
      re[y] = cx + cd / 2 + wr;
    }
    leftEdges.push(le);
    rightEdges.push(re);
  }

  // Row-level scan stripe offsets + subtle illumination gradient
  const rowOffsets = new Float32Array(imgH);
  for (let y = 0; y < imgH; y++) {
    rowOffsets[y] = (rng() - 0.5) * 10 + 4 * Math.sin(y * 0.42);
  }

  for (let y = 0; y < imgH; y++) {
    for (let x = 0; x < width; x++) {
      let val = 54 + 14 * (x / width) + rowOffsets[y];
      for (let l = 0; l < nLines; l++) {
        const lx = leftEdges[l][y];
        const rx = rightEdges[l][y];
        if (x >= lx - 5 && x <= rx + 5) {
          // Smooth sigmoidal top + classic SEM secondary-electron bright edge blooming
          const sL = 1 / (1 + Math.exp(-(x - lx) * 1.35));
          const sR = 1 / (1 + Math.exp((x - rx) * 1.35));
          const top = sL * sR;
          const bloomL = Math.exp(-Math.pow((x - lx) / 2.2, 2));
          const bloomR = Math.exp(-Math.pow((x - rx) / 2.2, 2));
          val = Math.max(val, 54 + 118 * top + 52 * (bloomL + bloomR) + rowOffsets[y]);
        }
      }
      gray[y * width + x] = Math.max(2, Math.min(253, val + gaussNoise(rng) * 7.5));
    }
  }

  drawDataBar(gray, width, height, barRows, 100);
  const rgba = grayToRgba(gray, width, height);
  return {
    id: `sem_lines_${seed}${suffix}`,
    name: `fei_linespace_32nm${suffix}.tif`,
    modality: 'sem',
    subType: 'sem_lines',
    width,
    height,
    frames: 1,
    channels: 1,
    rgba,
    gray,
    calibration: {
      pixelSize: 1.5,
      spaceUnit: 'nm',
      source: 'metadata',
      confirmed: true,
      vendor: 'Thermo Fisher (FEI)',
      acceleratingVoltageKv: 2.0,
      workingDistanceMm: 4.2,
      magnification: '150,000x',
      cropRows: imgH,
    },
    sha256: fastSha256Hex(rgba),
    descriptionZh: '半导体光刻线条/间隔 CD-SEM 图像（FEI 元数据，含真实边缘粗糙度与扫描条纹）',
    descriptionEn: 'Semiconductor line/space CD-SEM micrograph (FEI metadata, correlated LER & scan stripes)',
  };
}

export function buildSemHolesSpecimen(seed = 404, suffix = ''): Specimen {
  const width = 640;
  const height = 480;
  const barRows = 44;
  const imgH = height - barRows;
  const rng = mulberry32(seed);
  const gray = new Float32Array(width * height);

  for (let y = 0; y < imgH; y++) {
    for (let x = 0; x < width; x++) {
      const bg = 178 - 16 * (y / imgH) + 10 * Math.cos(x / 180) + gaussNoise(rng) * 6.5;
      gray[y * width + x] = Math.max(20, Math.min(248, bg));
    }
  }

  const cols = 7;
  const rows = 5;
  const pitchX = 76;
  const pitchY = 74;
  const ox = 86;
  const oy = 66;
  const missingSet = new Set(['2,3', '3,5']);

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      if (missingSet.has(`${r},${c}`)) continue;
      const cx = ox + c * pitchX + (rng() - 0.5) * 2.2;
      const cy = oy + r * pitchY + (rng() - 0.5) * 2.2;
      const rad = 18.0 + (rng() - 0.5) * 1.6;
      for (let y = Math.floor(cy - rad - 6); y <= Math.ceil(cy + rad + 6); y++) {
        for (let x = Math.floor(cx - rad - 6); x <= Math.ceil(cx + rad + 6); x++) {
          if (x < 0 || x >= width || y < 0 || y >= imgH) continue;
          const dist = Math.hypot(x - cx, y - cy);
          const edgeRim = Math.exp(-Math.pow((dist - (rad + 1.5)) / 2.2, 2));
          const hole = 1 / (1 + Math.exp((dist - rad) * 1.4));
          const val = gray[y * width + x] - 136 * hole + 38 * edgeRim;
          gray[y * width + x] = Math.max(5, Math.min(252, val));
        }
      }
    }
  }

  drawDataBar(gray, width, height, barRows, 100);
  const rgba = grayToRgba(gray, width, height);
  return {
    id: `sem_holes_${seed}${suffix}`,
    name: `zeiss_via_array_80nm${suffix}.tif`,
    modality: 'sem',
    subType: 'sem_holes',
    width,
    height,
    frames: 1,
    channels: 1,
    rgba,
    gray,
    calibration: {
      pixelSize: 2.0,
      spaceUnit: 'nm',
      source: 'databar',
      confirmed: false, // Pending human confirmation! Showcases CalibrationBanner workflow
      vendor: 'Zeiss',
      acceleratingVoltageKv: 3.0,
      workingDistanceMm: 5.1,
      magnification: '100,000x',
      cropRows: imgH,
    },
    sha256: fastSha256Hex(rgba),
    descriptionZh: '半导体接触孔阵列 SEM 图像（Zeiss 信息栏比例尺待确认，含 2 处缺失孔缺陷）',
    descriptionEn: 'Semiconductor contact-hole array SEM (Zeiss data-bar calibration pending, 2 missing vias)',
  };
}

export function buildSemParticlesSpecimen(seed = 505, suffix = ''): Specimen {
  const width = 640;
  const height = 480;
  const barRows = 44;
  const imgH = height - barRows;
  const rng = mulberry32(seed);
  const gray = new Float32Array(width * height);

  // Dark textured carbon substrate with uneven illumination
  for (let y = 0; y < imgH; y++) {
    for (let x = 0; x < width; x++) {
      const grad = 44 + 26 * (x / width) - 12 * (y / imgH);
      gray[y * width + x] = Math.max(8, Math.min(240, grad + gaussNoise(rng) * 6.0));
    }
  }

  interface ParticleSpec {
    cx: number;
    cy: number;
    rx: number;
    ry: number;
    ang: number;
    bright: number;
  }

  const plist: ParticleSpec[] = [
    // Normal dispersed nanoparticles
    { cx: 90, cy: 86, rx: 16, ry: 15, ang: 0.2, bright: 175 },
    { cx: 195, cy: 76, rx: 14, ry: 13.5, ang: 0.5, bright: 170 },
    { cx: 318, cy: 92, rx: 17, ry: 16, ang: 1.0, bright: 180 },
    { cx: 455, cy: 84, rx: 15, ry: 14.5, ang: 0.1, bright: 172 },
    { cx: 562, cy: 112, rx: 16, ry: 15, ang: 0.3, bright: 176 },
    { cx: 112, cy: 210, rx: 18, ry: 17, ang: 0.8, bright: 178 },
    { cx: 245, cy: 288, rx: 15, ry: 14, ang: 0.4, bright: 169 },
    { cx: 486, cy: 278, rx: 17, ry: 16.5, ang: 0.7, bright: 175 },
    { cx: 560, cy: 356, rx: 14, ry: 13.5, ang: 0.2, bright: 172 },
    { cx: 120, cy: 352, rx: 16, ry: 15, ang: 0.1, bright: 174 },
    { cx: 342, cy: 368, rx: 15.5, ry: 14.5, ang: 0.9, bright: 171 },
    // Agglomerate (two touching fused spheres)
    { cx: 256, cy: 178, rx: 16, ry: 15.5, ang: 0, bright: 180 },
    { cx: 282, cy: 186, rx: 15, ry: 14.5, ang: 0, bright: 180 },
    // Irregular / elongated shard
    { cx: 412, cy: 195, rx: 28, ry: 11, ang: 0.55, bright: 176 },
    // Outlier (oversized bright foreign contaminant)
    { cx: 362, cy: 274, rx: 32, ry: 29, ang: 0.2, bright: 238 },
  ];

  for (const p of plist) {
    const rMax = Math.max(p.rx, p.ry) + 6;
    const ca = Math.cos(p.ang);
    const sa = Math.sin(p.ang);
    for (let y = Math.floor(p.cy - rMax); y <= Math.ceil(p.cy + rMax); y++) {
      for (let x = Math.floor(p.cx - rMax); x <= Math.ceil(p.cx + rMax); x++) {
        if (x < 0 || x >= width || y < 0 || y >= imgH) continue;
        const dx = x - p.cx;
        const dy = y - p.cy;
        const u = (dx * ca + dy * sa) / p.rx;
        const v = (-dx * sa + dy * ca) / p.ry;
        const d2 = u * u + v * v;
        if (d2 <= 1.25) {
          const d = Math.sqrt(d2);
          const body = 1 / (1 + Math.exp((d - 1.0) * 7.5));
          const shade = 0.72 + 0.28 * Math.max(0, 1 - d2 * 0.65);
          const val = Math.max(gray[y * width + x], 46 + (p.bright - 46) * body * shade + gaussNoise(rng) * 3.5);
          gray[y * width + x] = Math.min(252, val);
        }
      }
    }
  }

  drawDataBar(gray, width, height, barRows, 100);
  const rgba = grayToRgba(gray, width, height);
  return {
    id: `sem_particles_${seed}${suffix}`,
    name: `tescan_nanoparticles${suffix}.tif`,
    modality: 'sem',
    subType: 'sem_particles',
    width,
    height,
    frames: 1,
    channels: 1,
    rgba,
    gray,
    calibration: {
      pixelSize: 5.0,
      spaceUnit: 'nm',
      source: 'metadata',
      confirmed: true,
      vendor: 'TESCAN',
      acceleratingVoltageKv: 10.0,
      workingDistanceMm: 8.5,
      magnification: '50,000x',
      cropRows: imgH,
    },
    sha256: fastSha256Hex(rgba),
    descriptionZh: '材料纳米颗粒 SEM 图像（含正常颗粒、团聚体、不规则碎片与大尺寸异物）',
    descriptionEn: 'Materials nanoparticle SEM (normal spheres, agglomerate pair, irregular shard, bright outlier)',
  };
}

export function buildSemFibresSpecimen(seed = 606, suffix = ''): Specimen {
  const width = 640;
  const height = 480;
  const barRows = 44;
  const imgH = height - barRows;
  const rng = mulberry32(seed);
  const gray = new Float32Array(width * height);

  for (let i = 0; i < imgH * width; i++) {
    gray[i] = Math.max(6, Math.min(240, 34 + gaussNoise(rng) * 5.0));
  }

  const fibres = [
    { x0: 20, y0: 70, x1: 620, y1: 155, halfW: 6.5 },
    { x0: 40, y0: 380, x1: 610, y1: 95, halfW: 5.5 },
    { x0: 110, y0: 20, x1: 280, y1: 420, halfW: 7.2 },
    { x0: 360, y0: 15, x1: 520, y1: 425, halfW: 6.0 },
    { x0: 10, y0: 240, x1: 630, y1: 310, halfW: 5.8 },
    { x0: 210, y0: 410, x1: 590, y1: 210, halfW: 6.8 },
  ];

  for (const f of fibres) {
    const dx = f.x1 - f.x0;
    const dy = f.y1 - f.y0;
    const len2 = dx * dx + dy * dy;
    for (let y = 0; y < imgH; y++) {
      for (let x = 0; x < width; x++) {
        const t = Math.max(0, Math.min(1, ((x - f.x0) * dx + (y - f.y0) * dy) / len2));
        const px = f.x0 + t * dx;
        const py = f.y0 + t * dy;
        const dist = Math.hypot(x - px, y - py);
        if (dist <= f.halfW + 3.5) {
          const core = 1 / (1 + Math.exp((dist - f.halfW) * 1.5));
          const edgeBright = Math.exp(-Math.pow((dist - (f.halfW - 0.8)) / 1.5, 2));
          const val = 38 + 148 * core + 34 * edgeBright + gaussNoise(rng) * 3.0;
          gray[y * width + x] = Math.max(gray[y * width + x], Math.min(250, val));
        }
      }
    }
  }

  drawDataBar(gray, width, height, barRows, 100);
  const rgba = grayToRgba(gray, width, height);
  return {
    id: `sem_fibres_${seed}${suffix}`,
    name: `hitachi_electrospun_fibres${suffix}.tif`,
    modality: 'sem',
    subType: 'sem_fibres',
    width,
    height,
    frames: 1,
    channels: 1,
    rgba,
    gray,
    calibration: {
      pixelSize: 8.0,
      spaceUnit: 'nm',
      source: 'metadata',
      confirmed: true,
      vendor: 'Hitachi',
      acceleratingVoltageKv: 5.0,
      workingDistanceMm: 7.4,
      magnification: '25,000x',
      cropRows: imgH,
    },
    sha256: fastSha256Hex(rgba),
    descriptionZh: '静电纺丝无纺布纤维 SEM 图像（含多交叉点纤维网络，测中心线直径与取向）',
    descriptionEn: 'Electrospun polymer fibre mat SEM (crossing fibre network for centre-line diameter & orientation)',
  };
}

export function getBuiltinSpecimens(): Specimen[] {
  return [
    buildDropletEncapsulationSpecimen(101),
    buildDropletFluorescenceSpecimen(202),
    buildSemLinesSpecimen(303),
    buildSemHolesSpecimen(404),
    buildSemParticlesSpecimen(505),
    buildSemFibresSpecimen(606),
  ];
}

const SCENE_LABELS: Record<DetectedSceneType, { zh: string; en: string }> = {
  microfluidics_droplet: { zh: '微流控 / 微滴图像', en: 'Microfluidics / Droplets' },
  cells: { zh: '细胞显微图像', en: 'Biological Cells' },
  pcb_board: { zh: 'PCB 电路板图像', en: 'PCB Circuit Board' },
  sem_metrology: { zh: 'SEM 扫描电镜图像', en: 'SEM Micrograph' },
  unknown: { zh: '未能识别图像类型', en: 'Unrecognized Image' },
};

interface BarScanResult {
  hasDataBar: boolean;
  barPosition: 'bottom' | 'top' | 'none';
  barHeightPx: number;
  scaleBarDetected: boolean;
  scaleBarPx: number | null;
  scaleBarCropRows: number; // rows cropped from bottom (or top) to remove info bar / scale bar
}

/**
 * Scan bottom/top strips for an instrument data bar and locate crisp horizontal scale-bar segments.
 */
function detectInfoBarAndScaleBar(gray: Float32Array, width: number, height: number): BarScanResult {
  const rowStats = (y: number) => {
    let sum = 0;
    let sumSq = 0;
    const off = y * width;
    for (let x = 0; x < width; x++) {
      const v = gray[off + x];
      sum += v;
      sumSq += v * v;
    }
    const mean = sum / width;
    const sd = Math.sqrt(Math.max(0, sumSq / width - mean * mean));
    return { mean, sd };
  };

  // 1. Check bottom info bar (dark or bright strip with uniform background)
  let darkBottom = 0;
  let brightBottom = 0;
  const maxBottomScan = Math.floor(height * 0.24);
  for (let i = 0; i < maxBottomScan; i++) {
    const { mean, sd } = rowStats(height - 1 - i);
    if (mean < 46 && sd < 62) {
      if (brightBottom === 0) darkBottom++;
      else break;
    } else if (mean > 224 && sd < 52) {
      if (darkBottom === 0) brightBottom++;
      else break;
    } else {
      break;
    }
  }

  // 2. Check top info bar if no bottom bar found
  let darkTop = 0;
  let brightTop = 0;
  if (darkBottom < 14 && brightBottom < 14) {
    const maxTopScan = Math.floor(height * 0.14);
    for (let i = 0; i < maxTopScan; i++) {
      const { mean, sd } = rowStats(i);
      if (mean < 46 && sd < 62) {
        if (brightTop === 0) darkTop++;
        else break;
      } else if (mean > 224 && sd < 52) {
        if (darkTop === 0) brightTop++;
        else break;
      } else {
        break;
      }
    }
  }

  let hasDataBar = false;
  let barPosition: 'bottom' | 'top' | 'none' = 'none';
  let barHeightPx = 0;
  let barIsDark = true;

  if (darkBottom >= 14) {
    hasDataBar = true;
    barPosition = 'bottom';
    barHeightPx = darkBottom;
    barIsDark = true;
  } else if (brightBottom >= 14) {
    hasDataBar = true;
    barPosition = 'bottom';
    barHeightPx = brightBottom;
    barIsDark = false;
  } else if (darkTop >= 14) {
    hasDataBar = true;
    barPosition = 'top';
    barHeightPx = darkTop;
    barIsDark = true;
  } else if (brightTop >= 14) {
    hasDataBar = true;
    barPosition = 'top';
    barHeightPx = brightTop;
    barIsDark = false;
  }

  // 3. Detect horizontal scale bar segment inside the data bar OR in bottom corners
  const findHorizontalScaleBar = (yStart: number, yEnd: number, darkBg: boolean): { lenPx: number; yRow: number } | null => {
    const minLen = Math.max(22, Math.round(width * 0.045));
    const maxLen = Math.round(width * 0.42);
    const candidates: Array<{ y: number; x0: number; x1: number; len: number }> = [];

    for (let y = Math.max(1, yStart); y < Math.min(height - 1, yEnd); y++) {
      const off = y * width;
      let runStart = -1;
      for (let x = 0; x <= width; x++) {
        const isBarPx =
          x < width && (darkBg ? gray[off + x] > 195 : gray[off + x] < 55);
        if (isBarPx && runStart === -1) {
          runStart = x;
        } else if (!isBarPx && runStart !== -1) {
          const len = x - runStart;
          if (len >= minLen && len <= maxLen) {
            // Confirm strong contrast just above or below the horizontal bar
            let contrastHits = 0;
            const checkAbove = Math.max(0, y - 5);
            const checkBelow = Math.min(height - 1, y + 5);
            const samples = 8;
            for (let s = 0; s < samples; s++) {
              const sx = runStart + Math.floor(((s + 0.5) * len) / samples);
              const vBar = gray[off + sx];
              const vUp = gray[checkAbove * width + sx];
              const vDn = gray[checkBelow * width + sx];
              if (Math.abs(vBar - vUp) > 65 || Math.abs(vBar - vDn) > 65) {
                contrastHits++;
              }
            }
            if (contrastHits >= 6) {
              candidates.push({ y, x0: runStart, x1: x, len });
            }
          }
          runStart = -1;
        }
      }
    }

    if (candidates.length === 0) return null;

    // Group vertically adjacent matching runs (thickness 1..12 px) and pick cleanest bar
    let best: { lenPx: number; yRow: number; score: number } | null = null;
    for (const c of candidates) {
      let thickness = 1;
      for (const o of candidates) {
        if (
          o !== c &&
          Math.abs(o.y - c.y) <= 10 &&
          Math.abs(o.x0 - c.x0) <= 4 &&
          Math.abs(o.len - c.len) <= 4
        ) {
          thickness++;
        }
      }
      if (thickness >= 1 && thickness <= 14) {
        const score = thickness * 12 + c.len * 0.25;
        if (!best || score > best.score) {
          best = { lenPx: c.len, yRow: c.y, score };
        }
      }
    }
    return best ? { lenPx: best.lenPx, yRow: best.yRow } : null;
  };

  let scaleBarDetected = false;
  let scaleBarPx: number | null = null;
  let scaleBarCropRows = hasDataBar ? barHeightPx : 0;

  if (hasDataBar) {
    const y0 = barPosition === 'bottom' ? height - barHeightPx : 0;
    const y1 = barPosition === 'bottom' ? height : barHeightPx;
    const found = findHorizontalScaleBar(y0, y1, barIsDark);
    if (found) {
      scaleBarDetected = true;
      scaleBarPx = found.lenPx;
    }
  } else {
    // No full-width info bar: check bottom 15% for an embedded corner scale bar
    const yStart = Math.floor(height * 0.85);
    let bottomMean = 0;
    let cnt = 0;
    for (let y = yStart; y < height; y += 2) {
      for (let x = 0; x < width; x += 2) {
        bottomMean += gray[y * width + x];
        cnt++;
      }
    }
    bottomMean /= Math.max(1, cnt);
    const found =
      findHorizontalScaleBar(yStart, height - 2, bottomMean < 135) ??
      findHorizontalScaleBar(yStart, height - 2, bottomMean >= 135);
    if (found) {
      scaleBarDetected = true;
      scaleBarPx = found.lenPx;
      // Crop just above the embedded corner scale bar (leaving safe margin for printed label above bar)
      const proposedCrop = height - Math.max(0, found.yRow - 18);
      if (proposedCrop >= 16 && proposedCrop <= Math.floor(height * 0.2)) {
        scaleBarCropRows = proposedCrop;
        barPosition = 'bottom';
      }
    }
  }

  return {
    hasDataBar,
    barPosition,
    barHeightPx,
    scaleBarDetected,
    scaleBarPx,
    scaleBarCropRows,
  };
}

/**
 * Deterministic local classifier for uploaded scientific / industrial images.
 * Honestly returns `unknown` (recognized: false) when image features do not match
 * microfluidics/droplets, biological cells, PCB board, or SEM micrographs.
 */
function classifyImportedImageLocal(
  fileName: string,
  rgba: Uint8ClampedArray,
  gray: Float32Array,
  width: number,
  height: number,
  barInfo: BarScanResult
): {
  recognized: boolean;
  sceneType: DetectedSceneType;
  subType: Specimen['subType'];
  modality: Specimen['modality'];
  confidence: number;
  suggestedRecipeId: string | null;
  reasonZh: string;
  reasonEn: string;
} {
  const n = width * height;
  let chromaSum = 0;
  let greenSolderMaskPx = 0;
  let blueSolderMaskPx = 0;
  let copperGoldPx = 0;
  let darkBgPx = 0;
  let brightFluoPx = 0;
  let graySum = 0;

  for (let i = 0; i < n; i++) {
    const r = rgba[i * 4];
    const g = rgba[i * 4 + 1];
    const b = rgba[i * 4 + 2];
    const v = gray[i];
    graySum += v;
    chromaSum += Math.abs(r - g) + Math.abs(g - b);

    if (g > r + 14 && g > b + 10 && g > 40 && g < 195) greenSolderMaskPx++;
    if (b > r + 18 && b > g + 6 && b > 45 && b < 190) blueSolderMaskPx++;
    if (r > 155 && g > 120 && r > b + 28) copperGoldPx++;
    if (v < 34) darkBgPx++;
    if (v > 120 && (g > r + 24 || b > r + 24 || r > g + 30)) brightFluoPx++;
  }

  const meanChroma = chromaSum / Math.max(1, n);
  const meanGray = graySum / Math.max(1, n);
  const greenFrac = greenSolderMaskPx / Math.max(1, n);
  const blueFrac = blueSolderMaskPx / Math.max(1, n);
  const copperFrac = copperGoldPx / Math.max(1, n);
  const darkFrac = darkBgPx / Math.max(1, n);
  const fluoFrac = brightFluoPx / Math.max(1, n);

  // Measure horizontal vs vertical edge strength (Manhattan bias for PCB / semiconductor lines)
  let gxSum = 0;
  let gySum = 0;
  let strongEdgePx = 0;
  let sampleCnt = 0;
  for (let y = 2; y < height - 2; y += 2) {
    for (let x = 2; x < width - 2; x += 2) {
      const gx = Math.abs(gray[y * width + x + 1] - gray[y * width + x - 1]);
      const gy = Math.abs(gray[(y + 1) * width + x] - gray[(y - 1) * width + x]);
      gxSum += gx;
      gySum += gy;
      if (gx + gy > 42) strongEdgePx++;
      sampleCnt++;
    }
  }
  const edgeDensity = strongEdgePx / Math.max(1, sampleCnt);
  const anisotropy = Math.abs(gxSum - gySum) / Math.max(1, gxSum + gySum);

  // Check for horizontal microfluidic channel walls (two strong dark horizontal bands near top & bottom)
  let topDarkRow = false;
  let botDarkRow = false;
  for (let y = Math.floor(height * 0.04); y < Math.floor(height * 0.22); y++) {
    let rm = 0;
    for (let x = 0; x < width; x++) rm += gray[y * width + x];
    if (rm / width < meanGray - 36) topDarkRow = true;
  }
  for (let y = Math.floor(height * 0.78); y < Math.floor(height * 0.96); y++) {
    let rm = 0;
    for (let x = 0; x < width; x++) rm += gray[y * width + x];
    if (rm / width < meanGray - 36) botDarkRow = true;
  }
  const hasChannelWalls = topDarkRow && botDarkRow && meanGray > 95;

  const lower = fileName.toLowerCase();

  // 1. Explicit filename hints (high-precision domain cues)
  if (/pcb|board|circuit|solder|pad|trace|gerber|电路板|焊盘/.test(lower)) {
    return {
      recognized: true,
      sceneType: 'pcb_board',
      subType: 'pcb_traces',
      modality: 'sem',
      confidence: 0.9,
      suggestedRecipeId: 'pcb_inspection',
      reasonZh: '识别为 PCB 电路板图像，自动推荐「PCB 走线与焊盘检测」工作流。',
      reasonEn: 'Identified as a PCB board image; recommended PCB Traces & Pads Inspection workflow.',
    };
  }
  if (/fluo|ddpcr|pcr|荧光液滴|阳性液滴/.test(lower)) {
    return {
      recognized: true,
      sceneType: 'microfluidics_droplet',
      subType: 'droplet_fluorescence',
      modality: 'droplet',
      confidence: 0.9,
      suggestedRecipeId: 'droplet_fluorescence',
      reasonZh: '识别为荧光微滴 (ddPCR) 图像，自动推荐「数字荧光阴阳判读」工作流。',
      reasonEn: 'Identified as fluorescence droplets (ddPCR); recommended Digital Fluorescence workflow.',
    };
  }
  if (/drop|microfluid|emulsion|channel|微流控|微滴|液滴/.test(lower)) {
    return {
      recognized: true,
      sceneType: 'microfluidics_droplet',
      subType: 'droplet_encapsulation',
      modality: 'droplet',
      confidence: 0.9,
      suggestedRecipeId: 'droplet_encapsulation',
      reasonZh: '识别为微流控液滴图像，自动推荐「微流控液滴粒径与包裹统计」工作流。',
      reasonEn: 'Identified as microfluidic droplets; recommended Droplet Size & Encapsulation workflow.',
    };
  }
  if (/cell|nuclei|dapi|hela|tissue|spheroid|organoid|细胞|细胞核/.test(lower)) {
    return {
      recognized: true,
      sceneType: 'cells',
      subType: 'cells_count',
      modality: 'droplet',
      confidence: 0.89,
      suggestedRecipeId: 'cell_segmentation',
      reasonZh: '识别为细胞显微图像，自动推荐「细胞分割、计数与形态分析」工作流。',
      reasonEn: 'Identified as biological cells; recommended Cell Segmentation & Morphometry workflow.',
    };
  }
  if (/sem|fei|zeiss|tescan|hitachi|jeol|electron|电镜|电镜图/.test(lower)) {
    let recipeId = 'sem_particles';
    let subType: Specimen['subType'] = 'sem_particles';
    if (/line|cd|ler|lwr|pitch|grating|线条/.test(lower)) {
      recipeId = 'sem_lines';
      subType = 'sem_lines';
    } else if (/hole|via|array|接触孔|阵列/.test(lower)) {
      recipeId = 'sem_holes';
      subType = 'sem_holes';
    } else if (/fibre|fiber|electrospun|纤维/.test(lower)) {
      recipeId = 'sem_fibres';
      subType = 'sem_fibres';
    } else if (/layer|multilayer|stack|cross.?section|film|coating|lamella|多层|层厚|膜厚|截面|涂层/.test(lower)) {
      recipeId = 'sem_multilayer';
      subType = 'sem_multilayer';
    }
    return {
      recognized: true,
      sceneType: 'sem_metrology',
      subType,
      modality: 'sem',
      confidence: 0.91,
      suggestedRecipeId: recipeId,
      reasonZh: '识别为 SEM 扫描电镜图像，已根据特征匹配标准电镜计量工作流。',
      reasonEn: 'Identified as an SEM micrograph; matched to the corresponding SEM metrology workflow.',
    };
  }

  // 2. Pure visual heuristics (without filename hints)
  // 2a. PCB board: strong green or blue solder mask with copper/metallic pads
  if ((greenFrac > 0.28 || blueFrac > 0.32) && (copperFrac > 0.03 || edgeDensity > 0.14)) {
    return {
      recognized: true,
      sceneType: 'pcb_board',
      subType: 'pcb_traces',
      modality: 'sem',
      confidence: 0.84,
      suggestedRecipeId: 'pcb_inspection',
      reasonZh: '检测到典型 PCB 阻焊层色域与高对比走线/焊盘结构，推荐「PCB 检测」工作流。',
      reasonEn: 'Detected characteristic PCB solder-mask colour profile and trace/pad contrast.',
    };
  }

  // 2b. SEM micrograph: near-monochrome + instrument data bar or scale bar
  if (meanChroma < 5.5 && (barInfo.hasDataBar || barInfo.scaleBarDetected)) {
    const isLines = anisotropy > 0.32;
    return {
      recognized: true,
      sceneType: 'sem_metrology',
      subType: isLines ? 'sem_lines' : 'sem_particles',
      modality: 'sem',
      confidence: 0.86,
      suggestedRecipeId: isLines ? 'sem_lines' : 'sem_particles',
      reasonZh: isLines
        ? '检测到灰度电镜特征、仪器信息栏及高各向异性条纹，推荐「SEM 线条 CD/LER」工作流。'
        : '检测到灰度电镜特征与仪器标尺信息栏，推荐「SEM 颗粒分割与粒度分布」工作流。',
      reasonEn: isLines
        ? 'Monochrome micrograph with instrument bar and directional lines; recommended SEM Lines CD/LER.'
        : 'Monochrome micrograph with instrument bar/scale bar; recommended SEM Particle Metrology.',
    };
  }

  // 2c. Microfluidics channel walls + brightfield droplets
  if (hasChannelWalls && meanChroma < 16) {
    return {
      recognized: true,
      sceneType: 'microfluidics_droplet',
      subType: 'droplet_encapsulation',
      modality: 'droplet',
      confidence: 0.82,
      suggestedRecipeId: 'droplet_encapsulation',
      reasonZh: '检测到上下平行微流控流道壁与明场视场，推荐「微流控液滴」工作流。',
      reasonEn: 'Detected parallel microfluidic channel walls in brightfield; recommended Droplets workflow.',
    };
  }

  // 2d. Fluorescence biological cells (very dark background + distinct coloured glowing cell regions)
  if (darkFrac > 0.45 && fluoFrac > 0.025 && fluoFrac < 0.48) {
    return {
      recognized: true,
      sceneType: 'cells',
      subType: 'cells_count',
      modality: 'droplet',
      confidence: 0.8,
      suggestedRecipeId: 'cell_segmentation',
      reasonZh: '检测到暗背底荧光细胞目标分布，推荐「细胞分割与计数」工作流。',
      reasonEn: 'Detected dark-field fluorescent cell structures; recommended Cell Segmentation workflow.',
    };
  }

  // 3. Honest fallback: Unrecognized image — do NOT force a false scientific workflow
  return {
    recognized: false,
    sceneType: 'unknown',
    subType: 'unknown',
    modality: 'droplet',
    confidence: 0.28,
    suggestedRecipeId: null,
    reasonZh:
      '未能自动识别为微流控、微滴、细胞、PCB 板或 SEM 电镜图像，请手工标注/手动测量，或从上方下拉菜单手动选择工作流。',
    reasonEn:
      'Could not automatically classify as microfluidics, droplets, cells, PCB, or SEM. Please annotate/measure manually or select a workflow above.',
  };
}

export interface DecodedUploadBundle {
  specimen: Specimen;
  /** Downsampled original (pre-crop) JPEG data URL so server Vision OCR can read the info bar & scale bar label. */
  originalPreviewBase64: string;
}

/**
 * Decode an uploaded browser File (PNG/JPG/BMP/WebP) using the 3-step import workflow:
 * 1) Inspect & classify image (microfluidics/droplets, cells, PCB board, SEM, or unrecognized)
 * 2) Check for instrument info bar & scale bar, calibrate, and physically crop the scale bar / info bar
 * 3) Attach structured ImportRecognitionReport with suggested workflow (or null when unrecognized)
 */
export async function decodeUploadedImage(
  file: File,
  options?: { autoCrop?: boolean }
): Promise<DecodedUploadBundle> {
  const autoCrop = options?.autoCrop ?? false;
  const maxDim = 960;
  let origWidth = 0;
  let origHeight = 0;
  let origRgba: Uint8ClampedArray;
  let origGray: Float32Array;
  let originalPreviewBase64 = '';
  let fileFormat = file.type ? file.type.replace('image/', '').toUpperCase() : 'RASTER';
  let bitDepth = 8;
  let vendorMetaCal: Partial<Calibration> | undefined;
  let vendorMetadataFound = false;
  let downscaleFactor = 1;
  let sourceWidth = 0;
  let sourceHeight = 0;

  // Peek first 256 bytes for magic-header format routing (TIFF/OME-TIFF/SEM-TIFF/DM3/DM4/CZI/ND2/LIF/etc.)
  const headerSlice = await file.slice(0, 256).arrayBuffer();
  const headerBytes = new Uint8Array(headerSlice);

  const proprietary = detectProprietaryMicroscopyContainer(file.name, headerBytes);
  const sciKind = isNativeScientificBinary(file.name, headerBytes);

  if (proprietary && !sciKind) {
    throw new Error(`${proprietary.format}: ${proprietary.guidance}`);
  }

  if (sciKind) {
    const fullBuf = await file.arrayBuffer();
    const decoded =
      sciKind === 'tiff'
        ? await decodeTiffBuffer(fullBuf)
        : decodeGatanDmBuffer(fullBuf);

    fileFormat = decoded.formatLabel;
    bitDepth = decoded.bitDepth;
    vendorMetaCal = decoded.metadataCalibration;
    vendorMetadataFound = decoded.vendorMetadataFound;

    const scale = Math.min(1, maxDim / Math.max(decoded.width, decoded.height));
    downscaleFactor = scale;
    sourceWidth = decoded.width;
    sourceHeight = decoded.height;
    origWidth = Math.max(32, Math.round(decoded.width * scale));
    origHeight = Math.max(32, Math.round(decoded.height * scale));

    const srcCanvas = document.createElement('canvas');
    srcCanvas.width = decoded.width;
    srcCanvas.height = decoded.height;
    srcCanvas
      .getContext('2d')!
      .putImageData(
        new ImageData(new Uint8ClampedArray(decoded.rgba), decoded.width, decoded.height),
        0,
        0
      );

    const canvas = document.createElement('canvas');
    canvas.width = origWidth;
    canvas.height = origHeight;
    const ctx = canvas.getContext('2d')!;
    ctx.drawImage(srcCanvas, 0, 0, origWidth, origHeight);

    const prevScale = Math.min(1, 640 / Math.max(origWidth, origHeight));
    const prevCanvas = document.createElement('canvas');
    prevCanvas.width = Math.max(32, Math.round(origWidth * prevScale));
    prevCanvas.height = Math.max(32, Math.round(origHeight * prevScale));
    prevCanvas.getContext('2d')!.drawImage(canvas, 0, 0, prevCanvas.width, prevCanvas.height);
    originalPreviewBase64 = prevCanvas.toDataURL('image/jpeg', 0.85);

    if (origWidth === decoded.width && origHeight === decoded.height) {
      origRgba = decoded.rgba;
      origGray = decoded.gray;
    } else {
      const data = ctx.getImageData(0, 0, origWidth, origHeight);
      origRgba = new Uint8ClampedArray(data.data);
      origGray = new Float32Array(origWidth * origHeight);
      for (let i = 0; i < origWidth * origHeight; i++) {
        origGray[i] =
          0.299 * origRgba[i * 4] + 0.587 * origRgba[i * 4 + 1] + 0.114 * origRgba[i * 4 + 2];
      }
    }
  } else {
    const url = URL.createObjectURL(file);
    try {
      const img = await new Promise<HTMLImageElement>((resolve, reject) => {
        const el = new Image();
        el.onload = () => resolve(el);
        el.onerror = () =>
          reject(
            new Error(
              `Unable to decode image ${file.name}. Supported native formats: TIFF/OME-TIFF/SEM-TIFF, Gatan DM3/DM4, PNG, JPEG, BMP, WebP.`
            )
          );
        el.src = url;
      });
      const scale = Math.min(1, maxDim / Math.max(img.width, img.height));
      downscaleFactor = scale;
      sourceWidth = img.naturalWidth || img.width;
      sourceHeight = img.naturalHeight || img.height;
      origWidth = Math.max(32, Math.round(img.width * scale));
      origHeight = Math.max(32, Math.round(img.height * scale));

      const canvas = document.createElement('canvas');
      canvas.width = origWidth;
      canvas.height = origHeight;
      const ctx = canvas.getContext('2d')!;
      ctx.drawImage(img, 0, 0, origWidth, origHeight);

      const prevScale = Math.min(1, 640 / Math.max(origWidth, origHeight));
      const prevCanvas = document.createElement('canvas');
      prevCanvas.width = Math.max(32, Math.round(origWidth * prevScale));
      prevCanvas.height = Math.max(32, Math.round(origHeight * prevScale));
      prevCanvas.getContext('2d')!.drawImage(canvas, 0, 0, prevCanvas.width, prevCanvas.height);
      originalPreviewBase64 = prevCanvas.toDataURL('image/jpeg', 0.85);

      const data = ctx.getImageData(0, 0, origWidth, origHeight);
      origRgba = new Uint8ClampedArray(data.data);
      origGray = new Float32Array(origWidth * origHeight);
      for (let i = 0; i < origWidth * origHeight; i++) {
        origGray[i] =
          0.299 * origRgba[i * 4] + 0.587 * origRgba[i * 4 + 1] + 0.114 * origRgba[i * 4 + 2];
      }
    } finally {
      URL.revokeObjectURL(url);
    }
  }

  let chromaSum = 0;
  for (let i = 0; i < origWidth * origHeight; i++) {
    const r = origRgba[i * 4];
    const g = origRgba[i * 4 + 1];
    const b = origRgba[i * 4 + 2];
    chromaSum += Math.abs(r - g) + Math.abs(g - b);
  }
  const meanChroma = chromaSum / Math.max(1, origWidth * origHeight);

  // Step 2: Detect info bar & scale bar.
  // In semi-auto mode (autoCrop = false), do NOT physically crop yet — keep original image and outline proposed crop for user confirmation.
  const barInfo = detectInfoBarAndScaleBar(origGray, origWidth, origHeight);
  const detectedCropPx =
    barInfo.scaleBarCropRows >= 14 && origHeight - barInfo.scaleBarCropRows >= 48
      ? barInfo.scaleBarCropRows
      : 0;
  const previewCroppedHeight = origHeight - detectedCropPx;
  const previewOffsetRows = barInfo.barPosition === 'top' ? detectedCropPx : 0;

  // Slice preview FOV for local feature hint only
  let clsRgba = origRgba;
  let clsGray = origGray;
  if (detectedCropPx > 0) {
    clsRgba = origRgba.subarray(
      previewOffsetRows * origWidth * 4,
      (previewOffsetRows + previewCroppedHeight) * origWidth * 4
    );
    clsGray = origGray.subarray(
      previewOffsetRows * origWidth,
      (previewOffsetRows + previewCroppedHeight) * origWidth
    );
  }

  // Only physically apply crop immediately when autoCrop is explicitly enabled (e.g., Batch Processing)
  const appliedCropPx = autoCrop ? detectedCropPx : 0;
  const workingHeight = origHeight - appliedCropPx;
  let rgba: Uint8ClampedArray;
  let gray: Float32Array;
  if (appliedCropPx > 0) {
    rgba = new Uint8ClampedArray(origWidth * workingHeight * 4);
    gray = new Float32Array(origWidth * workingHeight);
    const startRow = barInfo.barPosition === 'top' ? appliedCropPx : 0;
    rgba.set(
      origRgba.subarray(startRow * origWidth * 4, (startRow + workingHeight) * origWidth * 4)
    );
    gray.set(origGray.subarray(startRow * origWidth, (startRow + workingHeight) * origWidth));
  } else {
    rgba = origRgba;
    gray = origGray;
  }

  // Step 1: Tentative local image recognition hint on clean FOV (suggestion only; never auto-executed in semi-auto mode)
  const cls = classifyImportedImageLocal(
    file.name,
    clsRgba,
    clsGray,
    origWidth,
    previewCroppedHeight,
    barInfo
  );

  // Derive physical calibration proposal: prioritize embedded microscopy vendor metadata tags (FEI/Zeiss/TESCAN/OME-TIFF/DM3/DM4)
  let pixelSize = 1.0;
  let spaceUnit: Calibration['spaceUnit'] = 'px';
  let scaleBarValue: number | null = null;
  let scaleBarUnit: Calibration['spaceUnit'] | null = null;
  // 'databar' only once the bar's number has been read (OCR, applyVisionInspectionToSpecimen): a detected
  // bar with an assumed label, or a default pixel size, is not a calibration
  let calSource: Calibration['source'] = 'none';
  // In semi-auto mode, always wait for user confirmation before locking calibration
  let calConfirmed = false;

  if (vendorMetaCal?.pixelSize && vendorMetaCal.pixelSize > 0 && vendorMetaCal.spaceUnit) {
    // Adjust pixelSize if image was downsampled to maxDim so physical measurements remain exact
    pixelSize = Number((vendorMetaCal.pixelSize / Math.max(1e-6, downscaleFactor)).toPrecision(4));
    spaceUnit = vendorMetaCal.spaceUnit;
    calSource = 'metadata';
    calConfirmed = autoCrop;
  } else if (barInfo.scaleBarDetected && barInfo.scaleBarPx && barInfo.scaleBarPx >= 16) {
    if (cls.sceneType === 'sem_metrology') {
      scaleBarValue = 500;
      scaleBarUnit = 'nm';
    } else if (cls.sceneType === 'pcb_board') {
      scaleBarValue = 1;
      scaleBarUnit = 'mm';
    } else {
      scaleBarValue = 100;
      scaleBarUnit = 'µm';
    }
    pixelSize = Number((scaleBarValue / barInfo.scaleBarPx).toPrecision(3));
    spaceUnit = scaleBarUnit;
  } else if (cls.sceneType === 'sem_metrology') {
    pixelSize = 2.0;
    spaceUnit = 'nm';
  } else if (cls.sceneType === 'pcb_board') {
    pixelSize = 0.025;
    spaceUnit = 'mm';
  } else if (cls.sceneType === 'microfluidics_droplet' || cls.sceneType === 'cells') {
    pixelSize = 1.0;
    spaceUnit = 'µm';
  }

  const labels = SCENE_LABELS[cls.sceneType];
  const importReport: ImportRecognitionReport = {
    recognized: cls.recognized,
    sceneType: cls.sceneType,
    sceneLabelZh: labels.zh,
    sceneLabelEn: labels.en,
    confidence: cls.confidence,
    source: 'local',
    reasonZh: cls.reasonZh,
    reasonEn: cls.reasonEn,
    hasDataBar: barInfo.hasDataBar,
    barPosition: barInfo.barPosition,
    croppedBarPx: appliedCropPx,
    proposedCropBarPx: detectedCropPx,
    cropStatus: autoCrop ? (appliedCropPx > 0 ? 'confirmed' : 'skipped') : 'pending',
    originalWidth: origWidth,
    originalHeight: origHeight,
    scaleBarDetected: barInfo.scaleBarDetected,
    scaleBarPx: barInfo.scaleBarPx,
    scaleBarValue,
    scaleBarUnit,
    scaleBarCropped: appliedCropPx > 0,
    suggestedRecipeId: cls.suggestedRecipeId,
    fileFormat,
    bitDepth,
    vendorMetadataFound,
  };

  const specimen: Specimen = {
    id: `upload_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`,
    name: file.name,
    modality: cls.modality,
    subType: cls.subType,
    width: origWidth,
    height: workingHeight,
    frames: 1,
    channels: meanChroma > 8 ? 3 : 1,
    rgba,
    gray,
    rawRgba: appliedCropPx > 0 ? origRgba : undefined,
    rawWidth: appliedCropPx > 0 ? origWidth : undefined,
    rawHeight: appliedCropPx > 0 ? origHeight : undefined,
    rawBarPosition: appliedCropPx > 0 ? barInfo.barPosition : undefined,
    rawCroppedBarPx: appliedCropPx > 0 ? appliedCropPx : undefined,
    calibration: {
      pixelSize,
      spaceUnit,
      source: calSource,
      confirmed: calConfirmed,
      vendor: vendorMetaCal?.vendor,
      acceleratingVoltageKv: vendorMetaCal?.acceleratingVoltageKv,
      workingDistanceMm: vendorMetaCal?.workingDistanceMm,
      magnification: vendorMetaCal?.magnification,
      cropRows: workingHeight,
    },
    importReport,
    sourceFile: file,
    sourceWidth,
    sourceHeight,
    sha256: fastSha256Hex(rgba),
    descriptionZh: autoCrop
      ? cls.recognized
        ? `用户导入 ${fileFormat}（${bitDepth}-bit，${origWidth}×${workingHeight} px，识别为「${labels.zh}」${vendorMetadataFound ? '，已读取厂商标定元数据' : appliedCropPx > 0 ? `，已裁切 ${appliedCropPx} px 标尺/信息栏` : ''}）`
        : `用户导入 ${fileFormat}（${bitDepth}-bit，${origWidth}×${workingHeight} px，未识别出特定结构）`
      : `用户导入 ${fileFormat}（${bitDepth}-bit，${origWidth}×${origHeight} px，半自动待确认：等待确认信息栏裁切、标定与工作流）`,
    descriptionEn: autoCrop
      ? cls.recognized
        ? `Imported ${fileFormat} (${bitDepth}-bit, ${origWidth}×${workingHeight} px, classified as ${labels.en}${vendorMetadataFound ? ', vendor metadata calibrated' : appliedCropPx > 0 ? `, cropped ${appliedCropPx} px bar` : ''})`
        : `Imported ${fileFormat} (${bitDepth}-bit, ${origWidth}×${workingHeight} px, unrecognized)`
      : `Imported ${fileFormat} (${bitDepth}-bit, ${origWidth}×${origHeight} px — semi-auto staging: awaiting crop, calibration & workflow confirmation)`,
  };

  return { specimen, originalPreviewBase64 };
}

/**
 * Apply or revert user-confirmed info-bar cropping on a specimen without losing the raw uncropped source.
 * Passing `cropPx <= 0` or `barPosition === 'none'` restores the full uncropped image and marks cropStatus as 'skipped'.
 */
export function applyInfoBarCropToSpecimen(
  specimen: Specimen,
  cropPxInput: number,
  barPosition: 'bottom' | 'top' | 'none'
): Specimen {
  const origRgba = specimen.rawRgba ?? specimen.rgba;
  const origWidth = specimen.rawWidth ?? specimen.width;
  const origHeight = specimen.rawHeight ?? specimen.height;

  const maxAllowed = Math.max(0, origHeight - 48);
  const clampedCropPx =
    barPosition === 'none'
      ? 0
      : Math.max(0, Math.min(maxAllowed, Math.round(cropPxInput)));

  const buildGray = (buf: Uint8ClampedArray, w: number, h: number): Float32Array => {
    const out = new Float32Array(w * h);
    for (let i = 0; i < w * h; i++) {
      out[i] = 0.299 * buf[i * 4] + 0.587 * buf[i * 4 + 1] + 0.114 * buf[i * 4 + 2];
    }
    return out;
  };

  const prevReport = specimen.importReport;

  if (clampedCropPx > 0) {
    const newH = origHeight - clampedCropPx;
    const startRow = barPosition === 'top' ? clampedCropPx : 0;
    const nextRgba = new Uint8ClampedArray(origWidth * newH * 4);
    nextRgba.set(
      origRgba.subarray(startRow * origWidth * 4, (startRow + newH) * origWidth * 4)
    );
    const nextGray = buildGray(nextRgba, origWidth, newH);
    const nextReport: ImportRecognitionReport | undefined = prevReport
      ? {
          ...prevReport,
          barPosition,
          croppedBarPx: clampedCropPx,
          proposedCropBarPx: clampedCropPx,
          scaleBarCropped: true,
          cropStatus: 'confirmed',
        }
      : undefined;

    return {
      ...specimen,
      width: origWidth,
      height: newH,
      rgba: nextRgba,
      gray: nextGray,
      rawRgba: origRgba,
      rawWidth: origWidth,
      rawHeight: origHeight,
      rawBarPosition: barPosition,
      rawCroppedBarPx: clampedCropPx,
      calibration: {
        ...specimen.calibration,
        cropRows: newH,
      },
      importReport: nextReport,
      sha256: fastSha256Hex(nextRgba),
    };
  }

  // Restore full uncropped image (skipped or reverted)
  const fullGray = buildGray(origRgba, origWidth, origHeight);
  const nextReport: ImportRecognitionReport | undefined = prevReport
    ? {
        ...prevReport,
        croppedBarPx: 0,
        scaleBarCropped: false,
        cropStatus: 'skipped',
      }
    : undefined;

  return {
    ...specimen,
    width: origWidth,
    height: origHeight,
    rgba: origRgba,
    gray: fullGray,
    rawRgba: undefined,
    rawWidth: undefined,
    rawHeight: undefined,
    rawBarPosition: undefined,
    rawCroppedBarPx: undefined,
    calibration: {
      ...specimen.calibration,
      cropRows: origHeight,
    },
    importReport: nextReport,
    sha256: fastSha256Hex(origRgba),
  };
}

export interface VisionInspectionPayload {
  recognized?: boolean;
  sceneType?: string;
  confidence?: number;
  hasDataBar?: boolean;
  scaleBarValue?: number | null;
  scaleBarUnit?: string | null;
  suggestedRecipeId?: string | null;
  reasonZh?: string;
  reasonEn?: string;
}

const VALID_SCENE_TYPES: DetectedSceneType[] = [
  'microfluidics_droplet',
  'cells',
  'pcb_board',
  'sem_metrology',
  'unknown',
];

const VALID_RECIPE_IDS = new Set([
  'droplet_encapsulation',
  'droplet_fluorescence',
  'cell_segmentation',
  'pcb_inspection',
  'sem_lines',
  'sem_holes',
  'sem_particles',
  'sem_fibres',
  'sem_multilayer',
]);

/**
 * Merge server-side Vision classification & scale bar OCR back into an imported Specimen.
 */
export function applyVisionInspectionToSpecimen(
  specimen: Specimen,
  vision: VisionInspectionPayload
): Specimen {
  const prev = specimen.importReport;
  if (!prev) return specimen;

  const sceneType: DetectedSceneType = VALID_SCENE_TYPES.includes(vision.sceneType as DetectedSceneType)
    ? (vision.sceneType as DetectedSceneType)
    : prev.sceneType;
  const recognized = Boolean(vision.recognized) && sceneType !== 'unknown';
  const labels = SCENE_LABELS[sceneType];

  let suggestedRecipeId: string | null = null;
  if (recognized) {
    if (vision.suggestedRecipeId && VALID_RECIPE_IDS.has(vision.suggestedRecipeId)) {
      suggestedRecipeId = vision.suggestedRecipeId;
    } else if (sceneType === 'microfluidics_droplet') {
      suggestedRecipeId = 'droplet_encapsulation';
    } else if (sceneType === 'cells') {
      suggestedRecipeId = 'cell_segmentation';
    } else if (sceneType === 'pcb_board') {
      suggestedRecipeId = 'pcb_inspection';
    } else if (sceneType === 'sem_metrology') {
      suggestedRecipeId = 'sem_particles';
    }
  }

  let subType: Specimen['subType'] = 'unknown';
  let modality: Specimen['modality'] = 'droplet';
  if (suggestedRecipeId === 'droplet_fluorescence') {
    subType = 'droplet_fluorescence';
    modality = 'droplet';
  } else if (suggestedRecipeId === 'droplet_encapsulation') {
    subType = 'droplet_encapsulation';
    modality = 'droplet';
  } else if (suggestedRecipeId === 'cell_segmentation') {
    subType = 'cells_count';
    modality = 'droplet';
  } else if (suggestedRecipeId === 'pcb_inspection') {
    subType = 'pcb_traces';
    modality = 'sem';
  } else if (suggestedRecipeId === 'sem_lines') {
    subType = 'sem_lines';
    modality = 'sem';
  } else if (suggestedRecipeId === 'sem_holes') {
    subType = 'sem_holes';
    modality = 'sem';
  } else if (suggestedRecipeId === 'sem_fibres') {
    subType = 'sem_fibres';
    modality = 'sem';
  } else if (suggestedRecipeId === 'sem_multilayer') {
    subType = 'sem_multilayer';
    modality = 'sem';
  } else if (suggestedRecipeId === 'sem_particles') {
    subType = 'sem_particles';
    modality = 'sem';
  }

  // Refine calibration if Vision OCR read the printed scale bar value & unit
  const normUnit = (u?: string | null): Calibration['spaceUnit'] | null => {
    if (!u) return null;
    const s = u.trim().toLowerCase();
    if (s === 'nm') return 'nm';
    if (s === 'um' || s === 'µm' || s === 'μm') return 'µm';
    if (s === 'mm') return 'mm';
    return null;
  };
  const ocrVal =
    typeof vision.scaleBarValue === 'number' && Number.isFinite(vision.scaleBarValue) && vision.scaleBarValue > 0
      ? vision.scaleBarValue
      : prev.scaleBarValue;
  const ocrUnit = normUnit(vision.scaleBarUnit) ?? prev.scaleBarUnit;

  let nextCalibration: Calibration = { ...specimen.calibration };
  // Never overwrite a calibration that the user has already explicitly confirmed
  if (
    !specimen.calibration.confirmed &&
    ocrVal &&
    ocrUnit &&
    prev.scaleBarPx &&
    prev.scaleBarPx >= 12
  ) {
    nextCalibration = {
      ...nextCalibration,
      pixelSize: Number((ocrVal / prev.scaleBarPx).toPrecision(4)),
      spaceUnit: ocrUnit,
      source: 'databar',
      confirmed: false,
    };
  }

  const nextReport: ImportRecognitionReport = {
    ...prev,
    recognized,
    sceneType,
    sceneLabelZh: labels.zh,
    sceneLabelEn: labels.en,
    confidence:
      typeof vision.confidence === 'number' && Number.isFinite(vision.confidence)
        ? Math.max(0, Math.min(1, vision.confidence))
        : prev.confidence,
    source: 'vision',
    reasonZh: vision.reasonZh || prev.reasonZh,
    reasonEn: vision.reasonEn || prev.reasonEn,
    hasDataBar: prev.hasDataBar || Boolean(vision.hasDataBar),
    scaleBarValue: ocrVal,
    scaleBarUnit: ocrUnit,
    suggestedRecipeId,
  };

  return {
    ...specimen,
    modality,
    subType,
    calibration: nextCalibration,
    importReport: nextReport,
    descriptionZh: recognized
      ? `用户导入图像（${specimen.width}×${specimen.height} px，AI 识别为「${labels.zh}」${prev.croppedBarPx > 0 ? `，已裁切 ${prev.croppedBarPx} px 标尺/信息栏` : ''}）`
      : `用户导入图像（${specimen.width}×${specimen.height} px，未能识别图像类型，请手工标注或手动测量）`,
    descriptionEn: recognized
      ? `Imported (${specimen.width}×${specimen.height} px, AI classified as ${labels.en}${prev.croppedBarPx > 0 ? `, cropped ${prev.croppedBarPx} px bar` : ''})`
      : `Imported (${specimen.width}×${specimen.height} px, unrecognized — manual annotation or measurement recommended)`,
  };
}

export interface ScaleBarSnapResult {
  snapped: boolean;
  p1: [number, number];
  p2: [number, number];
  lengthPx: number;
  polarity?: 'bright' | 'dark';
  confidence?: number;
}

/**
 * Magnetically snaps a user's rough manual calibration stroke (`p1 -> p2`) onto the exact
 * sub-pixel start and end edges of the nearest high-contrast scale bar inside a local corridor.
 * Supports bright-on-dark, dark-on-bright, haloed bars, and end-tick bars (both horizontal & vertical).
 */
export function snapManualLineToScaleBar(
  gray: Float32Array,
  width: number,
  height: number,
  p1: [number, number],
  p2: [number, number]
): ScaleBarSnapResult {
  const dx = p2[0] - p1[0];
  const dy = p2[1] - p1[1];
  const userLen = Math.hypot(dx, dy);
  if (userLen < 8 || width < 16 || height < 16) {
    return { snapped: false, p1, p2, lengthPx: Number(userLen.toFixed(1)) };
  }

  const isHorizontal = Math.abs(dx) >= Math.abs(dy);
  // Work in axis-aligned coordinates: u = along bar, v = perpendicular to bar
  const uLenImg = isHorizontal ? width : height;
  const vLenImg = isHorizontal ? height : width;
  const getGrayUV = (u: number, v: number): number => {
    const cu = Math.max(0, Math.min(uLenImg - 1, u));
    const cv = Math.max(0, Math.min(vLenImg - 1, v));
    return isHorizontal ? gray[cv * width + cu] : gray[cu * width + cv];
  };

  const u1 = isHorizontal ? p1[0] : p1[1];
  const u2 = isHorizontal ? p2[0] : p2[1];
  const v1 = isHorizontal ? p1[1] : p1[0];
  const v2 = isHorizontal ? p2[1] : p2[0];
  const uMinUser = Math.min(u1, u2);
  const uMaxUser = Math.max(u1, u2);
  const userSpan = Math.max(8, uMaxUser - uMinUser);
  const vMidUser = (v1 + v2) * 0.5;

  // Search corridor: extend +-35% (18..72 px) along axis, +-18 px perpendicular
  const uPad = Math.max(18, Math.min(72, Math.round(userSpan * 0.35)));
  const vPad = 18;
  const uStart = Math.max(1, Math.floor(uMinUser - uPad));
  const uEnd = Math.min(uLenImg - 2, Math.ceil(uMaxUser + uPad));
  const vStart = Math.max(1, Math.floor(vMidUser - vPad));
  const vEnd = Math.min(vLenImg - 2, Math.ceil(vMidUser + vPad));

  if (uEnd - uStart < 10 || vEnd - vStart < 3) {
    return { snapped: false, p1, p2, lengthPx: Number(userLen.toFixed(1)) };
  }

  // Compute local intensity percentiles inside the search corridor for adaptive thresholding
  const corridorSamples: number[] = [];
  for (let v = vStart; v <= vEnd; v++) {
    for (let u = uStart; u <= uEnd; u += 2) {
      corridorSamples.push(getGrayUV(u, v));
    }
  }
  corridorSamples.sort((a, b) => a - b);
  if (corridorSamples.length < 16) {
    return { snapped: false, p1, p2, lengthPx: Number(userLen.toFixed(1)) };
  }
  const q = (p: number) =>
    corridorSamples[Math.max(0, Math.min(corridorSamples.length - 1, Math.floor(p * (corridorSamples.length - 1))))];
  const p10 = q(0.1);
  const p50 = q(0.5);
  const p90 = q(0.9);
  const dyn = p90 - p10;
  if (dyn < 38) {
    return { snapped: false, p1, p2, lengthPx: Number(userLen.toFixed(1)) };
  }

  interface RunCand {
    v: number;
    u0: number;
    u1: number;
    len: number;
    polarity: 'bright' | 'dark';
    contrast: number;
    uniformity: number;
    score: number;
  }

  const cands: RunCand[] = [];
  const minAcceptLen = Math.max(8, userSpan * 0.48);
  const maxAcceptLen = Math.min(uLenImg * 0.65, userSpan * 1.75 + 24);

  const scanPolarity = (polarity: 'bright' | 'dark') => {
    const isBright = polarity === 'bright';
    const thresh = isBright
      ? Math.max(p50 + dyn * 0.32, p90 - Math.min(35, dyn * 0.25))
      : Math.min(p50 - dyn * 0.32, p10 + Math.min(35, dyn * 0.25));

    for (let v = vStart; v <= vEnd; v++) {
      let runStart = -1;
      for (let u = uStart; u <= uEnd + 1; u++) {
        const hit =
          u <= uEnd && (isBright ? getGrayUV(u, v) >= thresh : getGrayUV(u, v) <= thresh);
        if (hit && runStart === -1) {
          runStart = u;
        } else if (!hit && runStart !== -1) {
          const runEnd = u - 1;
          const len = runEnd - runStart + 1;
          if (len >= minAcceptLen && len <= maxAcceptLen) {
            // Overlap with user's drawn span
            const ov = Math.max(0, Math.min(runEnd, uMaxUser) - Math.max(runStart, uMinUser));
            const ovRatio = ov / Math.max(1, Math.min(len, userSpan));
            if (ovRatio >= 0.35) {
              // Check crisp end-edge contrast immediately outside left & right ends
              let leftOut = 0;
              let rightOut = 0;
              let lCnt = 0;
              let rCnt = 0;
              for (let k = 2; k <= 5; k++) {
                if (runStart - k >= 0) {
                  leftOut += getGrayUV(runStart - k, v);
                  lCnt++;
                }
                if (runEnd + k < uLenImg) {
                  rightOut += getGrayUV(runEnd + k, v);
                  rCnt++;
                }
              }
              if (lCnt > 0 && rCnt > 0) {
                leftOut /= lCnt;
                rightOut /= rCnt;
                let barSum = 0;
                let barSumSq = 0;
                const samples = Math.min(len, 24);
                for (let s = 0; s < samples; s++) {
                  const su = runStart + Math.floor(((s + 0.5) * len) / samples);
                  const gv = getGrayUV(su, v);
                  barSum += gv;
                  barSumSq += gv * gv;
                }
                const barMean = barSum / samples;
                const barSd = Math.sqrt(Math.max(0, barSumSq / samples - barMean * barMean));
                const leftStep = isBright ? barMean - leftOut : leftOut - barMean;
                const rightStep = isBright ? barMean - rightOut : rightOut - barMean;
                const minEndStep = Math.min(leftStep, rightStep);

                // Also verify contrast above or below the bar (distinguishes a bar from a solid block)
                let vertContrast = 0;
                for (const dv of [-5, -3, 3, 5]) {
                  const vv = v + dv;
                  if (vv < 0 || vv >= vLenImg) continue;
                  let bandSum = 0;
                  for (let s = 0; s < 6; s++) {
                    const su = runStart + Math.floor(((s + 0.5) * len) / 6);
                    bandSum += getGrayUV(su, vv);
                  }
                  const bandMean = bandSum / 6;
                  const diff = isBright ? barMean - bandMean : bandMean - barMean;
                  if (diff > vertContrast) vertContrast = diff;
                }

                if (minEndStep >= 34 && vertContrast >= 30 && barSd < 52) {
                  const lenFit = Math.exp(-Math.pow((len - userSpan) / Math.max(16, userSpan * 0.45), 2));
                  const vDistFit = Math.exp(-Math.pow((v - vMidUser) / 11, 2));
                  const uniformFit = Math.max(0.2, 1 - barSd / 60);
                  const score =
                    (minEndStep / 255) * 0.42 +
                    (vertContrast / 255) * 0.2 +
                    lenFit * 0.18 +
                    vDistFit * 0.12 +
                    uniformFit * 0.08;
                  cands.push({
                    v,
                    u0: runStart,
                    u1: runEnd,
                    len,
                    polarity,
                    contrast: minEndStep,
                    uniformity: uniformFit,
                    score,
                  });
                }
              }
            }
          }
          runStart = -1;
        }
      }
    }
  };

  scanPolarity('bright');
  scanPolarity('dark');

  if (cands.length === 0) {
    return { snapped: false, p1, p2, lengthPx: Number(userLen.toFixed(1)) };
  }

  // Group vertically adjacent matching runs of the same polarity (bar thickness 1..18 px)
  let bestGroup: RunCand[] = [];
  let bestGroupScore = -1;
  const used = new Uint8Array(cands.length);

  for (let i = 0; i < cands.length; i++) {
    if (used[i]) continue;
    const seed = cands[i];
    const group: RunCand[] = [seed];
    used[i] = 1;
    for (let j = i + 1; j < cands.length; j++) {
      if (used[j]) continue;
      const o = cands[j];
      if (
        o.polarity === seed.polarity &&
        Math.abs(o.u0 - seed.u0) <= 3 &&
        Math.abs(o.u1 - seed.u1) <= 3 &&
        Math.abs(o.v - seed.v) <= 16
      ) {
        group.push(o);
        used[j] = 1;
      }
    }
    const thickness = group.length;
    if (thickness > 20) continue; // too tall to be a scale bar
    const meanScore = group.reduce((a, b) => a + b.score, 0) / thickness;
    const thicknessBonus = Math.min(1.35, 0.92 + Math.min(thickness, 8) * 0.06);
    const totalScore = meanScore * thicknessBonus;
    if (totalScore > bestGroupScore) {
      bestGroupScore = totalScore;
      bestGroup = group;
    }
  }

  if (bestGroup.length === 0 || bestGroupScore < 0.24) {
    return { snapped: false, p1, p2, lengthPx: Number(userLen.toFixed(1)) };
  }

  const polarity = bestGroup[0].polarity;
  const isBright = polarity === 'bright';
  const vMinBar = Math.min(...bestGroup.map((g) => g.v));
  const vMaxBar = Math.max(...bestGroup.map((g) => g.v));
  const vCenter = (vMinBar + vMaxBar) * 0.5;
  const u0Med = Math.round(bestGroup.reduce((a, b) => a + b.u0, 0) / bestGroup.length);
  const u1Med = Math.round(bestGroup.reduce((a, b) => a + b.u1, 0) / bestGroup.length);

  // Average 1D profile across the bar thickness rows for sub-pixel edge refinement
  const profU0 = Math.max(0, u0Med - 6);
  const profU1 = Math.min(uLenImg - 1, u1Med + 6);
  const profLen = profU1 - profU0 + 1;
  const prof = new Float32Array(profLen);
  const rowCount = vMaxBar - vMinBar + 1;
  for (let u = profU0; u <= profU1; u++) {
    let s = 0;
    for (let v = vMinBar; v <= vMaxBar; v++) s += getGrayUV(u, v);
    prof[u - profU0] = s / rowCount;
  }

  // Interior mean and outer background mean
  const innerA = Math.min(profLen - 1, Math.max(0, u0Med - profU0 + 2));
  const innerB = Math.min(profLen - 1, Math.max(innerA, u1Med - profU0 - 2));
  let barAvg = 0;
  for (let k = innerA; k <= innerB; k++) barAvg += prof[k];
  barAvg /= Math.max(1, innerB - innerA + 1);

  const bgLeft = (prof[0] + prof[Math.min(1, profLen - 1)] + prof[Math.min(2, profLen - 1)]) / 3;
  const bgRight =
    (prof[profLen - 1] + prof[Math.max(0, profLen - 2)] + prof[Math.max(0, profLen - 3)]) / 3;

  const midLeft = (barAvg + bgLeft) * 0.5;
  const midRight = (barAvg + bgRight) * 0.5;

  // Sub-pixel left edge crossing
  let subLeft = u0Med - 0.5;
  for (let k = 0; k < Math.min(profLen - 1, innerA + 2); k++) {
    const a = prof[k];
    const b = prof[k + 1];
    if (isBright ? a <= midLeft && b >= midLeft && b > a : a >= midLeft && b <= midLeft && a > b) {
      const frac = (midLeft - a) / (b - a);
      subLeft = profU0 + k + frac;
      break;
    }
  }

  // Sub-pixel right edge crossing
  let subRight = u1Med + 0.5;
  for (let k = profLen - 1; k > Math.max(0, innerB - 2); k--) {
    const a = prof[k - 1];
    const b = prof[k];
    if (isBright ? a >= midRight && b <= midRight && a > b : a <= midRight && b >= midRight && b > a) {
      const frac = (midRight - a) / (b - a);
      subRight = profU0 + (k - 1) + frac;
      break;
    }
  }

  const snappedSpan = subRight - subLeft;
  if (!Number.isFinite(snappedSpan) || snappedSpan < 6) {
    return { snapped: false, p1, p2, lengthPx: Number(userLen.toFixed(1)) };
  }

  // Preserve original user drag direction (left->right vs right->left)
  const forward = u2 >= u1;
  const startU = forward ? subLeft : subRight;
  const endU = forward ? subRight : subLeft;

  const snappedP1: [number, number] = isHorizontal
    ? [Number(startU.toFixed(2)), Number(vCenter.toFixed(2))]
    : [Number(vCenter.toFixed(2)), Number(startU.toFixed(2))];
  const snappedP2: [number, number] = isHorizontal
    ? [Number(endU.toFixed(2)), Number(vCenter.toFixed(2))]
    : [Number(vCenter.toFixed(2)), Number(endU.toFixed(2))];

  return {
    snapped: true,
    p1: snappedP1,
    p2: snappedP2,
    lengthPx: Number(snappedSpan.toFixed(1)),
    polarity,
    confidence: Number(Math.min(0.99, bestGroupScore).toFixed(2)),
  };
}

/**
 * Load the real TEM kidney tissue section asset (`/assets/aistudio/image_0423f6bd.png`)
 * into a Vexel Specimen pre-tagged for the composable Hierarchical Parent→Sub-Region workflow.
 */
export function loadKidneySectionSpecimen(
  imageUrl = '/assets/aistudio/image_0423f6bd.png'
): Promise<Specimen> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => {
      const width = img.naturalWidth || img.width;
      const height = img.naturalHeight || img.height;
      if (width < 8 || height < 8) {
        reject(new Error('Invalid kidney specimen dimensions'));
        return;
      }
      const canvas = document.createElement('canvas');
      canvas.width = width;
      canvas.height = height;
      const ctx = canvas.getContext('2d');
      if (!ctx) {
        reject(new Error('Canvas 2D context unavailable'));
        return;
      }
      ctx.drawImage(img, 0, 0, width, height);
      const imgData = ctx.getImageData(0, 0, width, height);
      const rgba = new Uint8ClampedArray(imgData.data);
      const gray = new Float32Array(width * height);
      for (let i = 0; i < width * height; i++) {
        gray[i] = 0.299 * rgba[i * 4] + 0.587 * rgba[i * 4 + 1] + 0.114 * rgba[i * 4 + 2];
      }
      const sampleStr = Array.from(rgba.subarray(0, Math.min(2048, rgba.length))).join(',');
      const sha256 = fastSha256Hex(`kidney_section_${width}x${height}_${sampleStr}`);
      resolve({
        id: 'sample_kidney_nested_cells',
        name: 'kidney_section_nested_cells.png',
        modality: 'droplet',
        subType: 'kidney_nested_cells',
        width,
        height,
        frames: 1,
        channels: 1,
        rgba,
        gray,
        calibration: {
          pixelSize: 1,
          spaceUnit: 'px',
          source: 'none',
          confirmed: true,
        },
        importReport: {
          recognized: true,
          sceneType: 'cells',
          sceneLabelZh: '肾切片（TEM）· 细胞核（核膜）与核仁',
          sceneLabelEn: 'Kidney Section (TEM) · Nuclei (nuclear envelope) & Nucleoli',
          confidence: 0.96,
          source: 'local',
          reasonZh: '两个由核膜围成的圆形细胞核，核内的暗色团块是核仁；匹配「层级嵌套分割」工作流。本图不含完整的细胞边界，所以面积比在这里是核仁/细胞核，不是核/细胞或核质比。',
          reasonEn: 'Two round nuclei bounded by their nuclear envelopes, with dark nucleoli inside; matched to the Hierarchical workflow. The image holds no complete cell boundary, so the area ratio here is nucleolus/nucleus, not nucleus/cell or N/C.',
          hasDataBar: false,
          barPosition: 'none',
          croppedBarPx: 0,
          originalWidth: width,
          originalHeight: height,
          scaleBarDetected: false,
          scaleBarPx: null,
          scaleBarValue: null,
          scaleBarUnit: null,
          scaleBarCropped: false,
          suggestedRecipeId: 'nested_cell_nucleus',
          fileFormat: 'PNG',
          bitDepth: 8,
        },
        sha256,
        descriptionZh: '肾组织切片透射电镜图：两个由核膜围成的大圆形细胞核，核内的暗色团块是核仁；细胞本身延伸到图外（中间的竖线是相邻细胞的细胞膜）。用于演示「先母区外边界 → 再域内受限子区分割 → 子区/母区面积占比」层级工作流：在这张图上测得的是核仁/细胞核面积比。',
        descriptionEn: 'Kidney tissue section (TEM): two large round nuclei bounded by their nuclear envelopes, with dark nucleoli inside; the cells extend beyond the image (the vertical line in the middle is the membrane between neighbouring cells). Demonstrates parent-to-subregion segmentation and area ratios: on this image the ratio is nucleolus/nucleus.',
      });
    };
    img.onerror = () => reject(new Error(`Failed to load specimen image: ${imageUrl}`));
    img.src = imageUrl;
  });
}



