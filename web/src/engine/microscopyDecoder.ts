import type { Calibration } from '../types/vexel';

/**
 * Pure-TypeScript Scientific Microscopy & Electron Microscopy Decoder.
 *
 * Natively decodes in browser (and Node.js):
 *  - Baseline & Scientific TIFF / OME-TIFF / ImageJ TIFF / Zeiss LSM TIFF (8-bit, 16-bit, 32-bit float;
 *    Uncompressed, PackBits, LZW, Deflate/Zlib; Grayscale & RGB; Little-Endian II & Big-Endian MM)
 *  - Vendor SEM TIFF embedded physical calibration tags:
 *      * Carl Zeiss SmartSEM (Tag 34118): AP_IMAGE_PIXEL_SIZE, AP_EHT, AP_WD, AP_MAG
 *      * Thermo Fisher / FEI Helios, Quanta, Scios, Apreo, Talos (Tag 34682): PixelWidth, HV, WD, HFW
 *      * TESCAN (Tag 50431 / ImageDescription): PixelSizeX, HV, WD
 *      * JEOL & Hitachi SEM TIFF ASCII headers & OME-XML / ImageJ Resolution tags
 *  - Gatan DigitalMicrograph (.dm3 / .dm4) TEM / STEM micrographs with physical nm/µm calibration
 *
 * Proprietary multi-dimensional vendor containers (.czi, .nd2, .lif, .oir, .vsi, .ser, .emi, .dcm)
 * are inspected by magic signature and routed with actionable Python engine / OME-TIFF guidance.
 */

export interface MicroscopyFormatSpec {
  id: string;
  extensions: string[];
  vendor: string;
  field: 'light' | 'electron' | 'both';
  supportTier: 'native_web' | 'python_bridge';
  calibrationExtraction: string;
  noteZh: string;
  noteEn: string;
}

export const MICROSCOPY_FORMAT_REGISTRY: MicroscopyFormatSpec[] = [
  {
    id: 'ome_tiff',
    extensions: ['.tif', '.tiff', '.ome.tif', '.ome.tiff'],
    vendor: 'OME / ImageJ / Fiji / Leica / Nikon / Olympus / Zeiss Export',
    field: 'both',
    supportTier: 'native_web',
    calibrationExtraction: 'OME-XML PhysicalSizeX + ImageJ Resolution',
    noteZh: '原生支持 8/16/32-bit 科学 TIFF（无压缩/LZW/PackBits/Deflate），自动百分位拉伸并读取 OME-XML/ImageJ 像素尺度',
    noteEn: 'Native 8/16/32-bit TIFF (Uncompressed/LZW/PackBits/Deflate) with auto-windowing & OME-XML/ImageJ pixel size',
  },
  {
    id: 'sem_tiff',
    extensions: ['.tif', '.tiff'],
    vendor: 'Thermo Fisher (FEI) · Zeiss SmartSEM · TESCAN · JEOL · Hitachi',
    field: 'electron',
    supportTier: 'native_web',
    calibrationExtraction: 'Tags 34682 (FEI), 34118 (Zeiss), 50431 (TESCAN), 270 (JEOL/Hitachi)',
    noteZh: '原生解析主流 SEM 厂商私有 TIFF 标签，自动提取 PixelWidth、加速电压 (kV)、工作距离 (WD) 与放大倍率',
    noteEn: 'Native SEM private TIFF tag parser extracting PixelWidth, kV, WD, and magnification across major EM vendors',
  },
  {
    id: 'gatan_dm',
    extensions: ['.dm3', '.dm4'],
    vendor: 'Gatan DigitalMicrograph (TEM / STEM / EELS)',
    field: 'electron',
    supportTier: 'native_web',
    calibrationExtraction: 'DM3/DM4 Calibrations.Dimension.0.Scale & Units',
    noteZh: '原生解析 Gatan DM3/DM4 二进制标签树，支持 8/16/32-bit 与浮点灰度及 nm/µm 标定读取',
    noteEn: 'Native binary tag-tree decoder for Gatan DM3/DM4 TEM/STEM images with nm/µm calibration',
  },
  {
    id: 'standard_raster',
    extensions: ['.png', '.jpg', '.jpeg', '.bmp', '.webp'],
    vendor: 'Universal Optical / SEM Export / Camera',
    field: 'both',
    supportTier: 'native_web',
    calibrationExtraction: 'Auto info-bar / scale-bar detection + Vision OCR',
    noteZh: '原生浏览器解码，并自动运行信息栏与标尺检测、标定及物理裁切',
    noteEn: 'Native browser decode with automatic data-bar & scale-bar detection, calibration, and crop',
  },
  {
    id: 'zeiss_czi_lsm',
    extensions: ['.czi', '.lsm'],
    vendor: 'Carl Zeiss (ZEN / Confocal)',
    field: 'light',
    supportTier: 'python_bridge',
    calibrationExtraction: '.lsm TIFF-based native preview; .czi via Python bridge (bioio / czifile)',
    noteZh: '.lsm (基于 TIFF) 可直接 Web 解码；.czi 需连接本地 Python 引擎 (vexel serve + czifile/bioio) 或在 ZEN 导出 OME-TIFF',
    noteEn: '.lsm (TIFF-based) decodes natively; .czi requires local Python engine (czifile/bioio) or OME-TIFF export from ZEN',
  },
  {
    id: 'nikon_nd2',
    extensions: ['.nd2'],
    vendor: 'Nikon NIS-Elements',
    field: 'light',
    supportTier: 'python_bridge',
    calibrationExtraction: 'Python bridge (nd2 / bioio) or NIS-Elements OME-TIFF export',
    noteZh: '多维荧光/时间序列 .nd2 需本地 Python 引擎 (nd2/bioio) 或在 NIS-Elements 导出 OME-TIFF',
    noteEn: 'Multi-dimensional .nd2 requires local Python bridge (nd2/bioio) or OME-TIFF export from NIS-Elements',
  },
  {
    id: 'leica_lif',
    extensions: ['.lif'],
    vendor: 'Leica LAS X',
    field: 'light',
    supportTier: 'python_bridge',
    calibrationExtraction: 'Python bridge (readlif / bioio) or LAS X OME-TIFF export',
    noteZh: '多序列容器 .lif 需本地 Python 引擎 (readlif/bioio) 或在 LAS X 导出 OME-TIFF',
    noteEn: 'Multi-series .lif container requires local Python bridge (readlif/bioio) or OME-TIFF export from LAS X',
  },
  {
    id: 'olympus_oir',
    extensions: ['.oir', '.oib', '.vsi'],
    vendor: 'Olympus / Evident (cellSens / FLUOVIEW)',
    field: 'light',
    supportTier: 'python_bridge',
    calibrationExtraction: 'Python bridge (bioio) or cellSens OME-TIFF export',
    noteZh: '专有容器 .oir/.oib/.vsi 需本地 Python 引擎 (bioio) 或导出 OME-TIFF',
    noteEn: 'Proprietary .oir/.oib/.vsi requires local Python bridge (bioio) or OME-TIFF export',
  },
  {
    id: 'fei_ser_emi',
    extensions: ['.ser', '.emi'],
    vendor: 'Thermo Fisher / FEI TIA / Velox (TEM / STEM)',
    field: 'electron',
    supportTier: 'python_bridge',
    calibrationExtraction: 'Python bridge (rosettasciio / hyperspy) or Velox TIFF export',
    noteZh: '电镜 .ser/.emi 序列需本地 Python 引擎 (rosettasciio) 或在 Velox 导出含标签的 16-bit TIFF',
    noteEn: 'TEM .ser/.emi requires local Python bridge (rosettasciio) or tagged 16-bit TIFF export from Velox',
  },
];

export const ACCEPTED_MICROSCOPY_EXTENSIONS =
  '.tif,.tiff,.ome.tif,.ome.tiff,.lsm,.dm3,.dm4,.czi,.nd2,.lif,.oir,.oib,.vsi,.ser,.emi,.dcm,.png,.jpg,.jpeg,.bmp,.webp,image/*';

export interface DecodedMicroscopyRaster {
  width: number;
  height: number;
  channels: number;
  bitDepth: number;
  formatLabel: string;
  rgba: Uint8ClampedArray;
  gray: Float32Array;
  metadataCalibration?: Partial<Calibration>;
  vendorMetadataFound: boolean;
}

// ============================================================================
// Helper: Auto-contrast 16-bit / 32-bit float scientific grayscale -> 0..255
// ============================================================================

function normalizeScientificGray(raw: Float32Array): Uint8ClampedArray {
  const n = raw.length;
  const step = Math.max(1, Math.floor(n / 40000));
  const sample: number[] = [];
  for (let i = 0; i < n; i += step) {
    const v = raw[i];
    if (Number.isFinite(v)) sample.push(v);
  }
  sample.sort((a, b) => a - b);
  const lo = sample.length > 0 ? sample[Math.floor(sample.length * 0.005)] : 0;
  const hi =
    sample.length > 0
      ? sample[Math.min(sample.length - 1, Math.floor(sample.length * 0.995))]
      : 255;
  const span = Math.max(1e-6, hi - lo);
  const out = new Uint8ClampedArray(n);
  for (let i = 0; i < n; i++) {
    const scaled = ((raw[i] - lo) / span) * 255;
    out[i] = scaled < 0 ? 0 : scaled > 255 ? 255 : Math.round(scaled);
  }
  return out;
}

// ============================================================================
// Unit normalization helpers -> Calibration['spaceUnit'] + pixelSize
// ============================================================================

function normalizeMetersToCalibration(metersPerPx: number): {
  pixelSize: number;
  spaceUnit: Calibration['spaceUnit'];
} {
  if (!Number.isFinite(metersPerPx) || metersPerPx <= 0) {
    return { pixelSize: 1, spaceUnit: 'px' };
  }
  if (metersPerPx < 1e-6) {
    return { pixelSize: Number((metersPerPx * 1e9).toPrecision(4)), spaceUnit: 'nm' };
  }
  if (metersPerPx < 1e-3) {
    return { pixelSize: Number((metersPerPx * 1e6).toPrecision(4)), spaceUnit: 'µm' };
  }
  return { pixelSize: Number((metersPerPx * 1e3).toPrecision(4)), spaceUnit: 'mm' };
}

function normalizeUnitValue(
  value: number,
  rawUnit: string
): { pixelSize: number; spaceUnit: Calibration['spaceUnit'] } | null {
  if (!Number.isFinite(value) || value <= 0) return null;
  const u = rawUnit.trim().toLowerCase();
  if (u === 'm' || u === 'meter' || u === 'meters') {
    return normalizeMetersToCalibration(value);
  }
  if (u === 'cm') {
    return normalizeMetersToCalibration(value * 1e-2);
  }
  if (u === 'mm') {
    return normalizeMetersToCalibration(value * 1e-3);
  }
  if (u === 'um' || u === 'µm' || u === 'μm' || u === 'micron' || u === 'microns' || u === '\\u00b5m') {
    return normalizeMetersToCalibration(value * 1e-6);
  }
  if (u === 'nm' || u === 'nanometer' || u === 'nanometers') {
    return normalizeMetersToCalibration(value * 1e-9);
  }
  if (u === 'pm') {
    return normalizeMetersToCalibration(value * 1e-12);
  }
  if (u === 'a' || u === 'å' || u === 'angstrom') {
    return normalizeMetersToCalibration(value * 1e-10);
  }
  return null;
}

// ============================================================================
// Compression Decoders for Scientific TIFF (PackBits, LZW, Deflate/Zlib)
// ============================================================================

function decodePackBits(src: Uint8Array, expectedBytes: number): Uint8Array {
  const out = new Uint8Array(expectedBytes);
  let sp = 0;
  let dp = 0;
  while (sp < src.length && dp < expectedBytes) {
    const n = (src[sp++] << 24) >> 24; // signed int8
    if (n >= 0) {
      const count = n + 1;
      for (let i = 0; i < count && sp < src.length && dp < expectedBytes; i++) {
        out[dp++] = src[sp++];
      }
    } else if (n !== -128) {
      const count = 1 - n;
      const val = src[sp++] ?? 0;
      for (let i = 0; i < count && dp < expectedBytes; i++) {
        out[dp++] = val;
      }
    }
  }
  return out;
}

function decodeTiffLzw(src: Uint8Array, expectedBytes: number): Uint8Array {
  // MSB-first variable-width TIFF LZW (9..12 bits, Clear=256, EOI=257)
  const out = new Uint8Array(expectedBytes);
  let outPos = 0;

  let bitBuf = 0;
  let bitCount = 0;
  let bytePos = 0;
  let codeSize = 9;
  let nextCode = 258;

  const prefix = new Int32Array(4096);
  const suffix = new Uint8Array(4096);
  const stack = new Uint8Array(4096);

  const resetTable = () => {
    codeSize = 9;
    nextCode = 258;
  };

  const readCode = (): number => {
    while (bitCount < codeSize) {
      if (bytePos >= src.length) return 257;
      bitBuf = (bitBuf << 8) | src[bytePos++];
      bitCount += 8;
    }
    bitCount -= codeSize;
    return (bitBuf >>> bitCount) & ((1 << codeSize) - 1);
  };

  const emitString = (code: number): number => {
    let sp = 0;
    let cur = code;
    while (cur >= 258 && sp < 4096) {
      stack[sp++] = suffix[cur];
      cur = prefix[cur];
    }
    const firstByte = cur & 0xff;
    stack[sp++] = firstByte;
    while (sp > 0 && outPos < expectedBytes) {
      out[outPos++] = stack[--sp];
    }
    return firstByte;
  };

  resetTable();
  let oldCode = -1;

  while (outPos < expectedBytes) {
    const code = readCode();
    if (code === 257) break;
    if (code === 256) {
      resetTable();
      oldCode = -1;
      continue;
    }

    if (oldCode === -1) {
      if (code < 256 && outPos < expectedBytes) {
        out[outPos++] = code;
      }
      oldCode = code;
      continue;
    }

    let firstChar = 0;
    if (code < nextCode) {
      firstChar = emitString(code);
    } else if (code === nextCode) {
      firstChar = emitString(oldCode);
      if (outPos < expectedBytes) out[outPos++] = firstChar;
    } else {
      break;
    }

    if (nextCode < 4096) {
      prefix[nextCode] = oldCode;
      suffix[nextCode] = firstChar;
      nextCode++;
      if (nextCode === 511) codeSize = 10;
      else if (nextCode === 1023) codeSize = 11;
      else if (nextCode === 2047) codeSize = 12;
    }
    oldCode = code;
  }

  return out;
}

async function decodeDeflateChunk(src: Uint8Array, expectedBytes: number): Promise<Uint8Array> {
  if (typeof DecompressionStream === 'undefined') {
    throw new Error('Deflate-compressed TIFF requires DecompressionStream support.');
  }
  const tryFormat = async (fmt: 'deflate' | 'deflate-raw'): Promise<Uint8Array> => {
    const ds = new DecompressionStream(fmt);
    const writer = ds.writable.getWriter();
    const copy = new Uint8Array(src.byteLength);
    copy.set(src);
    await writer.write(copy);
    await writer.close();
    const ab = await new Response(ds.readable).arrayBuffer();
    const arr = new Uint8Array(ab);
    if (arr.length >= expectedBytes) return arr.subarray(0, expectedBytes);
    const padded = new Uint8Array(expectedBytes);
    padded.set(arr);
    return padded;
  };
  try {
    return await tryFormat('deflate');
  } catch {
    return await tryFormat('deflate-raw');
  }
}

// ============================================================================
// Vendor SEM / OME-TIFF / ImageJ Metadata Calibration Parsers
// ============================================================================

function parseFeiHeliosTag(text: string): Partial<Calibration> | null {
  const pwMatch = text.match(/PixelWidth\s*=\s*([0-9.eE+-]+)/i);
  if (!pwMatch) return null;
  const meters = parseFloat(pwMatch[1]);
  if (!Number.isFinite(meters) || meters <= 0) return null;
  const cal = normalizeMetersToCalibration(meters);

  const hvMatch = text.match(/(?:^|\n)\s*HV\s*=\s*([0-9.eE+-]+)/i);
  const wdMatch = text.match(/(?:^|\n)\s*WorkingDistance\s*=\s*([0-9.eE+-]+)/i) ?? text.match(/(?:^|\n)\s*WD\s*=\s*([0-9.eE+-]+)/i);
  const magMatch = text.match(/(?:^|\n)\s*Magnification\s*=\s*([0-9.eE+-]+)/i);
  const hvRaw = hvMatch ? parseFloat(hvMatch[1]) : NaN;
  const wdRaw = wdMatch ? parseFloat(wdMatch[1]) : NaN;

  return {
    pixelSize: cal.pixelSize,
    spaceUnit: cal.spaceUnit,
    source: 'metadata',
    confirmed: true,
    vendor: 'Thermo Fisher / FEI SEM',
    acceleratingVoltageKv: Number.isFinite(hvRaw) ? Number((hvRaw > 100 ? hvRaw / 1000 : hvRaw).toFixed(2)) : undefined,
    workingDistanceMm: Number.isFinite(wdRaw) ? Number((wdRaw < 1 ? wdRaw * 1000 : wdRaw).toFixed(2)) : undefined,
    magnification: magMatch ? `${Math.round(parseFloat(magMatch[1]))}x` : undefined,
  };
}

function parseZeissSmartSemTag(text: string): Partial<Calibration> | null {
  // Zeiss Tag 34118 stores key lines followed by "Label = Value Unit"
  const pxMatch =
    text.match(/AP_IMAGE_PIXEL_SIZE[\s\r\n]+[^\r\n]*?=\s*([0-9.eE+-]+)\s*([a-zA-Zµμ]+)/i) ??
    text.match(/AP_PIXEL_SIZE[\s\r\n]+[^\r\n]*?=\s*([0-9.eE+-]+)\s*([a-zA-Zµμ]+)/i) ??
    text.match(/Image Pixel Size\s*=\s*([0-9.eE+-]+)\s*([a-zA-Zµμ]+)/i);
  if (!pxMatch) return null;
  const cal = normalizeUnitValue(parseFloat(pxMatch[1]), pxMatch[2]);
  if (!cal) return null;

  const ehtMatch = text.match(/(?:AP_EHT[\s\r\n]+[^\r\n]*?|EHT)\s*=\s*([0-9.eE+-]+)\s*(kV|V)/i);
  const wdMatch = text.match(/(?:AP_WD[\s\r\n]+[^\r\n]*?|WD)\s*=\s*([0-9.eE+-]+)\s*(mm|m)/i);
  const magMatch = text.match(/(?:AP_MAG[\s\r\n]+[^\r\n]*?|Mag)\s*=\s*([^\r\n]+)/i);

  let kv: number | undefined;
  if (ehtMatch) {
    const v = parseFloat(ehtMatch[1]);
    kv = ehtMatch[2].toLowerCase() === 'v' ? Number((v / 1000).toFixed(2)) : Number(v.toFixed(2));
  }
  let wd: number | undefined;
  if (wdMatch) {
    const v = parseFloat(wdMatch[1]);
    wd = wdMatch[2].toLowerCase() === 'm' ? Number((v * 1000).toFixed(2)) : Number(v.toFixed(2));
  }

  return {
    pixelSize: cal.pixelSize,
    spaceUnit: cal.spaceUnit,
    source: 'metadata',
    confirmed: true,
    vendor: 'Carl Zeiss SmartSEM',
    acceleratingVoltageKv: kv,
    workingDistanceMm: wd,
    magnification: magMatch ? magMatch[1].trim() : undefined,
  };
}

function parseTescanTag(text: string): Partial<Calibration> | null {
  const pxMatch = text.match(/PixelSizeX\s*=\s*([0-9.eE+-]+)/i);
  if (!pxMatch) return null;
  const cal = normalizeMetersToCalibration(parseFloat(pxMatch[1]));
  const hvMatch = text.match(/HV\s*=\s*([0-9.eE+-]+)/i);
  const wdMatch = text.match(/WD\s*=\s*([0-9.eE+-]+)/i);
  const hvRaw = hvMatch ? parseFloat(hvMatch[1]) : NaN;
  const wdRaw = wdMatch ? parseFloat(wdMatch[1]) : NaN;
  return {
    pixelSize: cal.pixelSize,
    spaceUnit: cal.spaceUnit,
    source: 'metadata',
    confirmed: true,
    vendor: 'TESCAN SEM',
    acceleratingVoltageKv: Number.isFinite(hvRaw) ? Number((hvRaw > 100 ? hvRaw / 1000 : hvRaw).toFixed(2)) : undefined,
    workingDistanceMm: Number.isFinite(wdRaw) ? Number((wdRaw < 1 ? wdRaw * 1000 : wdRaw).toFixed(2)) : undefined,
  };
}

function parseOmeXml(text: string): Partial<Calibration> | null {
  const sizeMatch = text.match(/PhysicalSizeX\s*=\s*["']([0-9.eE+-]+)["']/i);
  if (!sizeMatch) return null;
  const unitMatch = text.match(/PhysicalSizeXUnit\s*=\s*["']([^"']+)["']/i);
  const cal = normalizeUnitValue(parseFloat(sizeMatch[1]), unitMatch ? unitMatch[1] : 'µm');
  if (!cal) return null;
  return {
    pixelSize: cal.pixelSize,
    spaceUnit: cal.spaceUnit,
    source: 'metadata',
    confirmed: true,
    vendor: 'OME-TIFF',
  };
}

function parseJeolOrHitachiText(text: string): Partial<Calibration> | null {
  // Hitachi: PixelSize=2.45 (nm) or DataSize / PixelWidth
  const hitachiPx = text.match(/PixelSize\s*=\s*([0-9.eE+-]+)\s*(nm|um|µm|mm)?/i);
  if (hitachiPx) {
    const val = parseFloat(hitachiPx[1]);
    const unit = hitachiPx[2] || (val > 0.01 ? 'nm' : 'm');
    const cal = normalizeUnitValue(val, unit);
    if (cal) {
      return {
        pixelSize: cal.pixelSize,
        spaceUnit: cal.spaceUnit,
        source: 'metadata',
        confirmed: true,
        vendor: /JEOL/i.test(text) ? 'JEOL SEM' : 'Hitachi SEM',
      };
    }
  }
  return null;
}

function parseImageJCalibration(
  imageDescription: string,
  xResolution?: number,
  resolutionUnit?: number
): Partial<Calibration> | null {
  if (!xResolution || !Number.isFinite(xResolution) || xResolution <= 0) return null;
  const unitMatch = imageDescription.match(/unit\s*=\s*([^\r\n]+)/i);
  if (unitMatch) {
    const cal = normalizeUnitValue(1 / xResolution, unitMatch[1].trim());
    if (cal) {
      return {
        pixelSize: cal.pixelSize,
        spaceUnit: cal.spaceUnit,
        source: 'metadata',
        confirmed: true,
        vendor: 'ImageJ / Fiji TIFF',
      };
    }
  }
  // Standard TIFF ResolutionUnit: 2 = inch, 3 = centimeter
  if (resolutionUnit === 3 && xResolution > 100) {
    const cal = normalizeMetersToCalibration(0.01 / xResolution);
    return {
      pixelSize: cal.pixelSize,
      spaceUnit: cal.spaceUnit,
      source: 'metadata',
      confirmed: true,
      vendor: 'Calibrated TIFF',
    };
  }
  return null;
}

// ============================================================================
// Pure-TypeScript Scientific TIFF / OME-TIFF / SEM-TIFF Decoder
// ============================================================================

const TIFF_TYPE_SIZES: Record<number, number> = {
  1: 1, // BYTE
  2: 1, // ASCII
  3: 2, // SHORT
  4: 4, // LONG
  5: 8, // RATIONAL
  6: 1, // SBYTE
  7: 1, // UNDEFINED
  8: 2, // SSHORT
  9: 4, // SLONG
  10: 8, // SRATIONAL
  11: 4, // FLOAT
  12: 8, // DOUBLE
};

interface IfdEntry {
  tag: number;
  type: number;
  count: number;
  valueOffset: number;
}

export interface DecodedMultiPageTiff {
  width: number;
  height: number;
  depth: number;
  totalPagesInFile: number;
  bitDepth: number;
  formatLabel: string;
  slices: Float32Array[];
  voxelSizeXY?: number;
  voxelSizeZ?: number;
  spaceUnit?: Calibration['spaceUnit'];
  vendorMetadataFound: boolean;
}

function parseZSpacingFromDescription(
  imageDesc: string,
  fallbackUnit?: Calibration['spaceUnit']
): { voxelSizeZ?: number; spaceUnit?: Calibration['spaceUnit'] } {
  if (!imageDesc) return {};
  // 1) OME-XML PhysicalSizeZ="0.75" PhysicalSizeZUnit="µm"
  const omeVal = imageDesc.match(/PhysicalSizeZ\s*=\s*["']([\d.eE+-]+)["']/i);
  if (omeVal) {
    const val = Number(omeVal[1]);
    if (Number.isFinite(val) && val > 0) {
      const omeUnit = imageDesc.match(/PhysicalSizeZUnit\s*=\s*["']([^"']+)["']/i)?.[1] ?? '';
      let unit: Calibration['spaceUnit'] = fallbackUnit ?? 'µm';
      if (/nm/i.test(omeUnit)) unit = 'nm';
      else if (/mm/i.test(omeUnit)) unit = 'mm';
      else if (/[µu]m|micron/i.test(omeUnit)) unit = 'µm';
      return { voxelSizeZ: val, spaceUnit: unit };
    }
  }
  // 2) ImageJ / Fiji stack header: spacing=0.75\nunit=micron
  const ijSpacing = imageDesc.match(/(?:^|\n)\s*spacing\s*=\s*([\d.eE+-]+)/i);
  if (ijSpacing) {
    const val = Number(ijSpacing[1]);
    if (Number.isFinite(val) && val > 0) {
      const ijUnit = imageDesc.match(/(?:^|\n)\s*unit\s*=\s*([^\n\r]+)/i)?.[1]?.trim() ?? '';
      let unit: Calibration['spaceUnit'] = fallbackUnit ?? 'µm';
      if (/nm/i.test(ijUnit)) unit = 'nm';
      else if (/mm/i.test(ijUnit)) unit = 'mm';
      else if (/[µu]m|micron/i.test(ijUnit)) unit = 'µm';
      return { voxelSizeZ: val, spaceUnit: unit };
    }
  }
  return {};
}

async function decodeTiffIfdAt(
  buffer: ArrayBuffer,
  view: DataView,
  bytes: Uint8Array,
  littleEndian: boolean,
  ifdOffset: number,
  parseCalibration: boolean
): Promise<DecodedMicroscopyRaster & { nextIfdOffset: number; imageDesc: string }> {
  if (ifdOffset <= 0 || ifdOffset + 2 > buffer.byteLength) {
    throw new Error('Invalid TIFF IFD offset.');
  }

  const entryCount = view.getUint16(ifdOffset, littleEndian);
  const tags = new Map<number, IfdEntry>();
  for (let i = 0; i < entryCount; i++) {
    const base = ifdOffset + 2 + i * 12;
    if (base + 12 > buffer.byteLength) break;
    const tag = view.getUint16(base, littleEndian);
    const type = view.getUint16(base + 2, littleEndian);
    const count = view.getUint32(base + 4, littleEndian);
    const typeSize = TIFF_TYPE_SIZES[type] ?? 1;
    const totalBytes = typeSize * count;
    const valueOffset = totalBytes <= 4 ? base + 8 : view.getUint32(base + 8, littleEndian);
    tags.set(tag, { tag, type, count, valueOffset });
  }

  const nextIfdPtrPos = ifdOffset + 2 + entryCount * 12;
  const nextIfdOffset =
    nextIfdPtrPos + 4 <= buffer.byteLength ? view.getUint32(nextIfdPtrPos, littleEndian) : 0;

  const readNumbers = (tagId: number): number[] => {
    const e = tags.get(tagId);
    if (!e) return [];
    const out: number[] = [];
    const size = TIFF_TYPE_SIZES[e.type] ?? 1;
    for (let i = 0; i < e.count; i++) {
      const off = e.valueOffset + i * size;
      if (off + size > buffer.byteLength) break;
      switch (e.type) {
        case 1:
        case 7:
          out.push(view.getUint8(off));
          break;
        case 3:
          out.push(view.getUint16(off, littleEndian));
          break;
        case 4:
          out.push(view.getUint32(off, littleEndian));
          break;
        case 5: {
          const num = view.getUint32(off, littleEndian);
          const den = view.getUint32(off + 4, littleEndian);
          out.push(den !== 0 ? num / den : 0);
          break;
        }
        case 6:
          out.push(view.getInt8(off));
          break;
        case 8:
          out.push(view.getInt16(off, littleEndian));
          break;
        case 9:
          out.push(view.getInt32(off, littleEndian));
          break;
        case 11:
          out.push(view.getFloat32(off, littleEndian));
          break;
        case 12:
          out.push(view.getFloat64(off, littleEndian));
          break;
        default:
          out.push(view.getUint8(off));
      }
    }
    return out;
  };

  const readScalar = (tagId: number, fallback: number): number => {
    const arr = readNumbers(tagId);
    return arr.length > 0 ? arr[0] : fallback;
  };

  const readAscii = (tagId: number): string => {
    const e = tags.get(tagId);
    if (!e || e.count <= 0) return '';
    const end = Math.min(buffer.byteLength, e.valueOffset + e.count);
    if (e.valueOffset >= end) return '';
    const slice = bytes.subarray(e.valueOffset, end);
    try {
      return new TextDecoder('utf-8', { fatal: true }).decode(slice).replace(/\0+$/, '');
    } catch {
      return new TextDecoder('latin1').decode(slice).replace(/\0+$/, '');
    }
  };

  const width = readScalar(256, 0);
  const height = readScalar(257, 0);
  if (width <= 0 || height <= 0) {
    throw new Error('TIFF missing valid ImageWidth / ImageLength tags.');
  }

  const bitsPerSampleArr = readNumbers(258);
  const bitsPerSample = bitsPerSampleArr[0] ?? 8;
  const compression = readScalar(259, 1);
  const photometric = readScalar(262, 1); // 0=WhiteIsZero, 1=BlackIsZero, 2=RGB
  const samplesPerPixel = readScalar(277, Math.max(1, bitsPerSampleArr.length));
  const rowsPerStrip = readScalar(278, height);
  const planarConfig = readScalar(284, 1); // 1=Chunky, 2=Planar
  const predictor = readScalar(317, 1);
  const sampleFormat = readScalar(339, 1); // 1=uint, 2=int, 3=float

  const stripOffsets = readNumbers(273);
  const stripByteCounts = readNumbers(279);
  const tileWidth = readScalar(322, 0);
  const tileHeight = readScalar(323, 0);
  const tileOffsets = readNumbers(324);
  const tileByteCounts = readNumbers(325);

  const bytesPerSample = Math.max(1, Math.ceil(bitsPerSample / 8));
  const pixelStrideBytes = samplesPerPixel * bytesPerSample;
  const rawImageBytes = new Uint8Array(width * height * pixelStrideBytes);

  const decompressBlock = async (slice: Uint8Array, expected: number): Promise<Uint8Array> => {
    if (compression === 1) return slice;
    if (compression === 32773) return decodePackBits(slice, expected);
    if (compression === 5) return decodeTiffLzw(slice, expected);
    if (compression === 8 || compression === 32946) return decodeDeflateChunk(slice, expected);
    throw new Error(`Unsupported TIFF compression scheme (${compression}).`);
  };

  const undoPredictor2 = (buf: Uint8Array, rowW: number, rowH: number, spp: number) => {
    if (predictor !== 2) return;
    if (bytesPerSample === 1) {
      for (let y = 0; y < rowH; y++) {
        const rowOff = y * rowW * spp;
        for (let x = spp; x < rowW * spp; x++) {
          buf[rowOff + x] = (buf[rowOff + x] + buf[rowOff + x - spp]) & 0xff;
        }
      }
    } else if (bytesPerSample === 2) {
      const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
      for (let y = 0; y < rowH; y++) {
        const rowPx = y * rowW * spp;
        for (let x = spp; x < rowW * spp; x++) {
          const prev = dv.getUint16((rowPx + x - spp) * 2, littleEndian);
          const cur = dv.getUint16((rowPx + x) * 2, littleEndian);
          dv.setUint16((rowPx + x) * 2, (cur + prev) & 0xffff, littleEndian);
        }
      }
    }
  };

  if (tileWidth > 0 && tileHeight > 0 && tileOffsets.length > 0) {
    const tilesAcross = Math.ceil(width / tileWidth);
    const tilesDown = Math.ceil(height / tileHeight);
    const tileRowBytes = tileWidth * pixelStrideBytes;
    const expectedTileBytes = tileRowBytes * tileHeight;

    for (let ty = 0; ty < tilesDown; ty++) {
      for (let tx = 0; tx < tilesAcross; tx++) {
        const idx = ty * tilesAcross + tx;
        const off = tileOffsets[idx];
        const len = tileByteCounts[idx] ?? expectedTileBytes;
        if (off === undefined || off + len > buffer.byteLength) continue;
        const decoded = await decompressBlock(bytes.subarray(off, off + len), expectedTileBytes);
        undoPredictor2(decoded, tileWidth, tileHeight, samplesPerPixel);

        const copyW = Math.min(tileWidth, width - tx * tileWidth);
        const copyH = Math.min(tileHeight, height - ty * tileHeight);
        for (let row = 0; row < copyH; row++) {
          const srcStart = row * tileRowBytes;
          const dstStart = ((ty * tileHeight + row) * width + tx * tileWidth) * pixelStrideBytes;
          rawImageBytes.set(decoded.subarray(srcStart, srcStart + copyW * pixelStrideBytes), dstStart);
        }
      }
    }
  } else if (stripOffsets.length > 0) {
    if (planarConfig === 2 && samplesPerPixel > 1) {
      const stripsPerPlane = Math.ceil(height / rowsPerStrip);
      for (let c = 0; c < samplesPerPixel; c++) {
        let curRow = 0;
        for (let s = 0; s < stripsPerPlane && curRow < height; s++) {
          const idx = c * stripsPerPlane + s;
          const off = stripOffsets[idx];
          const rowsInStrip = Math.min(rowsPerStrip, height - curRow);
          const expected = rowsInStrip * width * bytesPerSample;
          const len = stripByteCounts[idx] ?? expected;
          if (off === undefined || off + len > buffer.byteLength) continue;
          const decoded = await decompressBlock(bytes.subarray(off, off + len), expected);
          undoPredictor2(decoded, width, rowsInStrip, 1);
          for (let r = 0; r < rowsInStrip; r++) {
            for (let x = 0; x < width; x++) {
              const srcBase = (r * width + x) * bytesPerSample;
              const dstBase = ((curRow + r) * width + x) * pixelStrideBytes + c * bytesPerSample;
              for (let b = 0; b < bytesPerSample; b++) {
                rawImageBytes[dstBase + b] = decoded[srcBase + b] ?? 0;
              }
            }
          }
          curRow += rowsInStrip;
        }
      }
    } else {
      let curRow = 0;
      for (let s = 0; s < stripOffsets.length && curRow < height; s++) {
        const off = stripOffsets[s];
        const rowsInStrip = Math.min(rowsPerStrip, height - curRow);
        const expected = rowsInStrip * width * pixelStrideBytes;
        const len = stripByteCounts[s] ?? expected;
        if (off === undefined || off + len > buffer.byteLength) continue;
        const decoded = await decompressBlock(bytes.subarray(off, off + len), expected);
        undoPredictor2(decoded, width, rowsInStrip, samplesPerPixel);
        const dstStart = curRow * width * pixelStrideBytes;
        const copyLen = Math.min(expected, decoded.length, rawImageBytes.length - dstStart);
        rawImageBytes.set(decoded.subarray(0, copyLen), dstStart);
        curRow += rowsInStrip;
      }
    }
  } else {
    throw new Error('TIFF contains neither StripOffsets nor TileOffsets.');
  }

  // Convert decoded samples into RGBA + grayscale Float32Array
  const n = width * height;
  const rawView = new DataView(rawImageBytes.buffer, rawImageBytes.byteOffset, rawImageBytes.byteLength);
  const rgba = new Uint8ClampedArray(n * 4);
  const gray = new Float32Array(n);

  const readSampleValue = (byteOffset: number): number => {
    if (bitsPerSample === 8) {
      return sampleFormat === 2 ? rawView.getInt8(byteOffset) : rawView.getUint8(byteOffset);
    }
    if (bitsPerSample === 16) {
      return sampleFormat === 2
        ? rawView.getInt16(byteOffset, littleEndian)
        : rawView.getUint16(byteOffset, littleEndian);
    }
    if (bitsPerSample === 32) {
      if (sampleFormat === 3) return rawView.getFloat32(byteOffset, littleEndian);
      if (sampleFormat === 2) return rawView.getInt32(byteOffset, littleEndian);
      return rawView.getUint32(byteOffset, littleEndian);
    }
    return rawView.getUint8(byteOffset);
  };

  if (samplesPerPixel >= 3 && photometric === 2) {
    const chRaw = [new Float32Array(n), new Float32Array(n), new Float32Array(n)];
    for (let i = 0; i < n; i++) {
      const base = i * pixelStrideBytes;
      chRaw[0][i] = readSampleValue(base);
      chRaw[1][i] = readSampleValue(base + bytesPerSample);
      chRaw[2][i] = readSampleValue(base + bytesPerSample * 2);
    }
    const r8 = bitsPerSample === 8 ? null : normalizeScientificGray(chRaw[0]);
    const g8 = bitsPerSample === 8 ? null : normalizeScientificGray(chRaw[1]);
    const b8 = bitsPerSample === 8 ? null : normalizeScientificGray(chRaw[2]);

    for (let i = 0; i < n; i++) {
      const rv = r8 ? r8[i] : Math.max(0, Math.min(255, Math.round(chRaw[0][i])));
      const gv = g8 ? g8[i] : Math.max(0, Math.min(255, Math.round(chRaw[1][i])));
      const bv = b8 ? b8[i] : Math.max(0, Math.min(255, Math.round(chRaw[2][i])));
      rgba[i * 4] = rv;
      rgba[i * 4 + 1] = gv;
      rgba[i * 4 + 2] = bv;
      rgba[i * 4 + 3] = 255;
      gray[i] = 0.299 * rv + 0.587 * gv + 0.114 * bv;
    }
  } else {
    const rawGray = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      rawGray[i] = readSampleValue(i * pixelStrideBytes);
    }
    const norm =
      bitsPerSample === 8 && sampleFormat === 1
        ? Uint8ClampedArray.from(rawGray)
        : normalizeScientificGray(rawGray);
    for (let i = 0; i < n; i++) {
      const v = photometric === 0 ? 255 - norm[i] : norm[i];
      rgba[i * 4] = v;
      rgba[i * 4 + 1] = v;
      rgba[i * 4 + 2] = v;
      rgba[i * 4 + 3] = 255;
      gray[i] = v;
    }
  }

  const imageDesc = readAscii(270);
  let metadataCalibration: Partial<Calibration> | undefined;
  let formatLabel = 'Scientific TIFF';

  if (parseCalibration) {
    const feiText = readAscii(34682);
    const zeissText = readAscii(34118);
    const tescanText = readAscii(50431);
    const xRes = readScalar(282, 0);
    const resUnit = readScalar(296, 1);

    if (feiText) {
      const cal = parseFeiHeliosTag(feiText);
      if (cal) {
        metadataCalibration = cal;
        formatLabel = 'Thermo Fisher / FEI SEM TIFF';
      }
    }
    if (!metadataCalibration && zeissText) {
      const cal = parseZeissSmartSemTag(zeissText);
      if (cal) {
        metadataCalibration = cal;
        formatLabel = 'Carl Zeiss SmartSEM TIFF';
      }
    }
    if (!metadataCalibration && (tescanText || /PixelSizeX\s*=/.test(imageDesc))) {
      const cal = parseTescanTag(tescanText || imageDesc);
      if (cal) {
        metadataCalibration = cal;
        formatLabel = 'TESCAN SEM TIFF';
      }
    }
    if (!metadataCalibration && /OME/i.test(imageDesc)) {
      const cal = parseOmeXml(imageDesc);
      if (cal) {
        metadataCalibration = cal;
        formatLabel = 'OME-TIFF';
      }
    }
    if (!metadataCalibration && imageDesc) {
      const cal = parseJeolOrHitachiText(imageDesc);
      if (cal) {
        metadataCalibration = cal;
        formatLabel = `${cal.vendor} TIFF`;
      }
    }
    if (!metadataCalibration) {
      const cal = parseImageJCalibration(imageDesc, xRes, resUnit);
      if (cal) {
        metadataCalibration = cal;
        formatLabel = cal.vendor === 'ImageJ / Fiji TIFF' ? 'ImageJ TIFF' : 'Calibrated TIFF';
      }
    }
  }

  return {
    width,
    height,
    channels: samplesPerPixel >= 3 && photometric === 2 ? 3 : 1,
    bitDepth: bitsPerSample,
    formatLabel,
    rgba,
    gray,
    metadataCalibration,
    vendorMetadataFound: Boolean(metadataCalibration),
    nextIfdOffset,
    imageDesc,
  };
}

function parseTiffHeader(buffer: ArrayBuffer): {
  view: DataView;
  bytes: Uint8Array;
  littleEndian: boolean;
  firstIfdOffset: number;
} {
  const view = new DataView(buffer);
  const bytes = new Uint8Array(buffer);
  if (buffer.byteLength < 8) {
    throw new Error('Invalid TIFF buffer (too small).');
  }
  const b0 = view.getUint8(0);
  const b1 = view.getUint8(1);
  let littleEndian = true;
  if (b0 === 0x49 && b1 === 0x49) {
    littleEndian = true;
  } else if (b0 === 0x4d && b1 === 0x4d) {
    littleEndian = false;
  } else {
    throw new Error('Invalid TIFF byte-order header (expected II or MM).');
  }

  const magic = view.getUint16(2, littleEndian);
  if (magic === 43) {
    throw new Error(
      'BigTIFF (>4GB) container detected. Connect the local Python engine (vexel serve with tifffile) to stream BigTIFF planes.'
    );
  }
  if (magic !== 42) {
    throw new Error(`Unsupported TIFF magic number (${magic}).`);
  }

  const firstIfdOffset = view.getUint32(4, littleEndian);
  if (firstIfdOffset <= 0 || firstIfdOffset + 2 > buffer.byteLength) {
    throw new Error('Invalid TIFF IFD offset.');
  }
  return { view, bytes, littleEndian, firstIfdOffset };
}

export async function decodeTiffBuffer(buffer: ArrayBuffer): Promise<DecodedMicroscopyRaster> {
  const { view, bytes, littleEndian, firstIfdOffset } = parseTiffHeader(buffer);
  const page0 = await decodeTiffIfdAt(buffer, view, bytes, littleEndian, firstIfdOffset, true);
  return {
    width: page0.width,
    height: page0.height,
    channels: page0.channels,
    bitDepth: page0.bitDepth,
    formatLabel: page0.formatLabel,
    rgba: page0.rgba,
    gray: page0.gray,
    metadataCalibration: page0.metadataCalibration,
    vendorMetadataFound: page0.vendorMetadataFound,
  };
}

/**
 * Decode a Multi-Page TIFF / OME-TIFF / ImageJ Z-Stack buffer in pure TypeScript (browser & Node).
 * Walks the IFD linked list, extracts XY/Z physical spacing from OME-XML or ImageJ headers,
 * and skips mismatched thumbnail pages.
 */
export async function decodeMultiPageTiffBuffer(
  buffer: ArrayBuffer,
  options: { maxPages?: number } = {}
): Promise<DecodedMultiPageTiff> {
  const { view, bytes, littleEndian, firstIfdOffset } = parseTiffHeader(buffer);
  const maxPages = Math.max(1, options.maxPages ?? 512);

  // First pass: walk IFD offsets safely (guarding against cycles)
  const visited = new Set<number>();
  const ifdOffsets: number[] = [];
  let cur = firstIfdOffset;
  while (cur > 0 && cur + 2 <= buffer.byteLength && !visited.has(cur) && ifdOffsets.length < 2048) {
    visited.add(cur);
    ifdOffsets.push(cur);
    const entryCount = view.getUint16(cur, littleEndian);
    const nextPtr = cur + 2 + entryCount * 12;
    cur = nextPtr + 4 <= buffer.byteLength ? view.getUint32(nextPtr, littleEndian) : 0;
  }

  const totalPagesInFile = ifdOffsets.length;
  const pageStride = totalPagesInFile > maxPages ? Math.ceil(totalPagesInFile / maxPages) : 1;

  const firstPage = await decodeTiffIfdAt(buffer, view, bytes, littleEndian, ifdOffsets[0], true);
  const slices: Float32Array[] = [firstPage.gray];

  for (let i = pageStride; i < ifdOffsets.length; i += pageStride) {
    const p = await decodeTiffIfdAt(buffer, view, bytes, littleEndian, ifdOffsets[i], false);
    if (p.width === firstPage.width && p.height === firstPage.height) {
      slices.push(p.gray);
    }
  }

  const zCal = parseZSpacingFromDescription(
    firstPage.imageDesc,
    firstPage.metadataCalibration?.spaceUnit
  );

  const unitToMeters = (u: Calibration['spaceUnit'] | undefined): number | null => {
    if (u === 'nm') return 1e-9;
    if (u === 'µm') return 1e-6;
    if (u === 'mm') return 1e-3;
    return null;
  };

  let spaceUnit: Calibration['spaceUnit'] | undefined =
    zCal.spaceUnit ?? firstPage.metadataCalibration?.spaceUnit;
  let voxelSizeXY = firstPage.metadataCalibration?.pixelSize;
  let voxelSizeZ =
    zCal.voxelSizeZ !== undefined ? Number((zCal.voxelSizeZ * pageStride).toFixed(4)) : undefined;

  // Harmonize XY and Z into the same physical unit (e.g. OME-XML XY=0.3 µm auto-scaled to 300 nm vs Z=0.9 µm)
  const xyUnit = firstPage.metadataCalibration?.spaceUnit;
  const zUnit = zCal.spaceUnit;
  if (
    voxelSizeXY !== undefined &&
    xyUnit &&
    spaceUnit &&
    xyUnit !== spaceUnit &&
    unitToMeters(xyUnit) !== null &&
    unitToMeters(spaceUnit) !== null
  ) {
    const xyMeters = voxelSizeXY * unitToMeters(xyUnit)!;
    voxelSizeXY = Number((xyMeters / unitToMeters(spaceUnit)!).toPrecision(5));
  }
  if (
    voxelSizeZ !== undefined &&
    zUnit &&
    spaceUnit &&
    zUnit !== spaceUnit &&
    unitToMeters(zUnit) !== null &&
    unitToMeters(spaceUnit) !== null
  ) {
    const zMeters = voxelSizeZ * unitToMeters(zUnit)!;
    voxelSizeZ = Number((zMeters / unitToMeters(spaceUnit)!).toPrecision(5));
  }

  return {
    width: firstPage.width,
    height: firstPage.height,
    depth: slices.length,
    totalPagesInFile,
    bitDepth: firstPage.bitDepth,
    formatLabel: firstPage.formatLabel,
    slices,
    voxelSizeXY,
    voxelSizeZ,
    spaceUnit,
    vendorMetadataFound: firstPage.vendorMetadataFound || voxelSizeZ !== undefined,
  };
}

// ============================================================================
// Pure-TypeScript Gatan DigitalMicrograph (.dm3 / .dm4) TEM/STEM Decoder
// ============================================================================

interface DmCandidateImage {
  width: number;
  height: number;
  dataType: number;
  byteOffset: number;
  byteLength: number;
  scale?: number;
  unit?: string;
}

const DM_DATA_TYPE_BYTES: Record<number, number> = {
  1: 2, // int16
  2: 4, // float32
  3: 8, // complex64
  6: 1, // uint8
  7: 4, // int32
  8: 4, // rgb4
  9: 1, // int8
  10: 2, // uint16
  11: 4, // uint32
  12: 8, // float64
};

export function decodeGatanDmBuffer(buffer: ArrayBuffer): DecodedMicroscopyRaster {
  const view = new DataView(buffer);
  const bytes = new Uint8Array(buffer);
  if (buffer.byteLength < 24) {
    throw new Error('Invalid Gatan DM3/DM4 file (too small).');
  }

  const version = view.getUint32(0, false);
  if (version !== 3 && version !== 4) {
    throw new Error(`Unsupported Gatan DigitalMicrograph version (${version}); expected DM3 or DM4.`);
  }
  const isDm4 = version === 4;

  let offset = 4;
  const readSizeField = (): number => {
    if (isDm4) {
      const hi = view.getUint32(offset, false);
      const lo = view.getUint32(offset + 4, false);
      offset += 8;
      return hi * 4294967296 + lo;
    }
    const v = view.getUint32(offset, false);
    offset += 4;
    return v;
  };

  readSizeField(); // root byte size
  const byteOrder = view.getUint32(offset, false);
  offset += 4;
  const le = byteOrder === 1;

  const candidates: DmCandidateImage[] = [];

  const dmTypeSize = (t: number): number => {
    switch (t) {
      case 2:
      case 4:
        return 2;
      case 3:
      case 5:
      case 6:
        return 4;
      case 7:
      case 11:
      case 12:
        return 8;
      case 8:
      case 9:
      case 10:
        return 1;
      default:
        return 1;
    }
  };

  const readDmScalar = (t: number, pos: number): number => {
    switch (t) {
      case 2:
        return view.getInt16(pos, le);
      case 3:
        return view.getInt32(pos, le);
      case 4:
        return view.getUint16(pos, le);
      case 5:
        return view.getUint32(pos, le);
      case 6:
        return view.getFloat32(pos, le);
      case 7:
        return view.getFloat64(pos, le);
      case 8:
      case 9:
        return view.getUint8(pos);
      case 10:
        return view.getInt8(pos);
      case 11:
      case 12: {
        const lo = view.getUint32(pos, le);
        const hi = view.getUint32(pos + 4, le);
        return le ? hi * 4294967296 + lo : lo * 4294967296 + hi;
      }
      default:
        return 0;
    }
  };

  interface ActiveImageCtx {
    width?: number;
    height?: number;
    dataType?: number;
    dataOffset?: number;
    dataLength?: number;
    scale?: number;
    unit?: string;
  }

  const parseTagGroup = (path: string, imgCtx: ActiveImageCtx | null) => {
    if (offset + 2 > buffer.byteLength) return;
    offset += 2; // isSorted, isOpen
    const nTags = readSizeField();

    for (let i = 0; i < nTags && offset < buffer.byteLength; i++) {
      const kind = view.getUint8(offset++);
      const nameLen = view.getUint16(offset, false);
      offset += 2;
      const name =
        nameLen > 0
          ? new TextDecoder('latin1').decode(bytes.subarray(offset, offset + nameLen))
          : String(i);
      offset += nameLen;

      const fullPath = path ? `${path}.${name}` : name;

      if (isDm4) {
        readSizeField(); // tag byte length
      }

      if (kind === 20) {
        // TagGroup
        const isNewImageEntry = path === 'ImageList';
        const childCtx: ActiveImageCtx | null = isNewImageEntry ? {} : imgCtx;
        parseTagGroup(fullPath, childCtx);
        if (
          isNewImageEntry &&
          childCtx &&
          childCtx.width &&
          childCtx.height &&
          childCtx.dataOffset !== undefined &&
          childCtx.dataLength
        ) {
          candidates.push({
            width: childCtx.width,
            height: childCtx.height,
            dataType: childCtx.dataType ?? 10,
            byteOffset: childCtx.dataOffset,
            byteLength: childCtx.dataLength,
            scale: childCtx.scale,
            unit: childCtx.unit,
          });
        }
      } else if (kind === 21) {
        // TagEntry: '%%%%' marker
        offset += 4;
        const nInfo = readSizeField();
        const info: number[] = [];
        for (let k = 0; k < nInfo; k++) {
          info.push(readSizeField());
        }

        const primaryType = info[0];
        if (primaryType === 20) {
          // Array
          const elemType = info[1];
          const count = info[2] ?? 0;
          if (elemType === 15) {
            // Array of structs
            const nFields = info[3] ?? 0;
            let structBytes = 0;
            for (let f = 0; f < nFields; f++) {
              structBytes += dmTypeSize(info[5 + f * 2]);
            }
            const arrCount = info[4 + nFields * 2] ?? 0;
            offset += arrCount * structBytes;
          } else {
            const elemBytes = dmTypeSize(elemType);
            const totalBytes = count * elemBytes;
            if (imgCtx && fullPath.endsWith('ImageData.Data')) {
              imgCtx.dataOffset = offset;
              imgCtx.dataLength = totalBytes;
            } else if (
              imgCtx &&
              (fullPath.endsWith('Calibrations.Dimension.0.Units') ||
                fullPath.endsWith('Dimension.0.Units')) &&
              count > 0 &&
              count < 64
            ) {
              // UTF-16 or uint16 string
              let uStr = '';
              for (let c = 0; c < count; c++) {
                const code =
                  elemBytes === 2
                    ? view.getUint16(offset + c * 2, le)
                    : view.getUint8(offset + c);
                if (code > 0) uStr += String.fromCharCode(code);
              }
              if (uStr.trim()) imgCtx.unit = uStr.trim();
            }
            offset += totalBytes;
          }
        } else if (primaryType === 15) {
          // Struct
          const nFields = info[2] ?? 0;
          for (let f = 0; f < nFields; f++) {
            offset += dmTypeSize(info[4 + f * 2]);
          }
        } else if (primaryType === 18) {
          // String
          const strLen = info[1] ?? 0;
          offset += strLen;
        } else {
          // Scalar value
          const val = readDmScalar(primaryType, offset);
          offset += dmTypeSize(primaryType);
          if (imgCtx) {
            if (fullPath.endsWith('ImageData.Dimensions.0')) imgCtx.width = Math.round(val);
            else if (fullPath.endsWith('ImageData.Dimensions.1')) imgCtx.height = Math.round(val);
            else if (fullPath.endsWith('ImageData.DataType')) imgCtx.dataType = Math.round(val);
            else if (
              fullPath.endsWith('Calibrations.Dimension.0.Scale') ||
              fullPath.endsWith('Dimension.0.Scale')
            ) {
              imgCtx.scale = val;
            }
          }
        }
      } else {
        break;
      }
    }
  };

  parseTagGroup('', null);

  if (candidates.length === 0) {
    throw new Error(`No 2D image payload found inside Gatan DM${version} file.`);
  }

  // Pick the largest 2D image (skip small thumbnail previews)
  candidates.sort((a, b) => b.width * b.height - a.width * a.height);
  const best = candidates[0];
  const n = best.width * best.height;
  const raw = new Float32Array(n);
  const elemBytes = DM_DATA_TYPE_BYTES[best.dataType] ?? 2;
  const bitDepth = elemBytes * 8;

  for (let i = 0; i < n; i++) {
    const pos = best.byteOffset + i * elemBytes;
    if (pos + elemBytes > buffer.byteLength) break;
    switch (best.dataType) {
      case 6:
        raw[i] = view.getUint8(pos);
        break;
      case 9:
        raw[i] = view.getInt8(pos);
        break;
      case 1:
        raw[i] = view.getInt16(pos, le);
        break;
      case 10:
        raw[i] = view.getUint16(pos, le);
        break;
      case 7:
        raw[i] = view.getInt32(pos, le);
        break;
      case 11:
        raw[i] = view.getUint32(pos, le);
        break;
      case 2:
        raw[i] = view.getFloat32(pos, le);
        break;
      case 12:
        raw[i] = view.getFloat64(pos, le);
        break;
      default:
        raw[i] = view.getUint16(pos, le);
    }
  }

  const norm = normalizeScientificGray(raw);
  const rgba = new Uint8ClampedArray(n * 4);
  const gray = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const v = norm[i];
    rgba[i * 4] = v;
    rgba[i * 4 + 1] = v;
    rgba[i * 4 + 2] = v;
    rgba[i * 4 + 3] = 255;
    gray[i] = v;
  }

  let metadataCalibration: Partial<Calibration> | undefined;
  if (best.scale && best.scale > 0) {
    const cal = normalizeUnitValue(best.scale, best.unit ?? 'nm');
    if (cal) {
      metadataCalibration = {
        pixelSize: cal.pixelSize,
        spaceUnit: cal.spaceUnit,
        source: 'metadata',
        confirmed: true,
        vendor: `Gatan DM${version}`,
      };
    }
  }

  return {
    width: best.width,
    height: best.height,
    channels: 1,
    bitDepth,
    formatLabel: `Gatan DigitalMicrograph DM${version}`,
    rgba,
    gray,
    metadataCalibration,
    vendorMetadataFound: Boolean(metadataCalibration),
  };
}

// ============================================================================
// Proprietary Multi-Dimensional Container Sniffer (CZI, ND2, LIF, OIR, SER)
// ============================================================================

export function detectProprietaryMicroscopyContainer(
  fileName: string,
  headerBytes: Uint8Array
): { format: string; vendor: string; guidance: string } | null {
  const lower = fileName.toLowerCase();
  const ascii16 = new TextDecoder('ascii', { fatal: false }).decode(headerBytes.subarray(0, 16));

  if (ascii16.startsWith('ZISRAWFILE') || lower.endsWith('.czi')) {
    return {
      format: 'Zeiss CZI (.czi)',
      vendor: 'Carl Zeiss ZEN',
      guidance:
        'Zeiss .czi container detected. Connect the local Python engine (`uv run vexel serve` with `czifile`/`bioio`) for native multi-channel Z-stack reading, or export as OME-TIFF from Zeiss ZEN.',
    };
  }
  if (
    (headerBytes.length >= 4 &&
      headerBytes[0] === 0x0b &&
      headerBytes[1] === 0xce &&
      headerBytes[2] === 0xda &&
      headerBytes[3] === 0x0a) ||
    lower.endsWith('.nd2')
  ) {
    return {
      format: 'Nikon ND2 (.nd2)',
      vendor: 'Nikon NIS-Elements',
      guidance:
        'Nikon .nd2 container detected. Connect the local Python engine (`vexel serve` with `nd2`/`bioio`) or export as OME-TIFF from NIS-Elements.',
    };
  }
  if (lower.endsWith('.lif')) {
    return {
      format: 'Leica LIF (.lif)',
      vendor: 'Leica LAS X',
      guidance:
        'Leica .lif multi-series file detected. Connect the local Python engine (`vexel serve` with `readlif`/`bioio`) or export as OME-TIFF from LAS X.',
    };
  }
  if (lower.endsWith('.oir') || lower.endsWith('.oib') || lower.endsWith('.vsi')) {
    return {
      format: 'Olympus / Evident (.oir/.oib/.vsi)',
      vendor: 'Olympus / Evident cellSens / FLUOVIEW',
      guidance:
        'Olympus/Evident proprietary container detected. Connect the local Python engine (`vexel serve` with `bioio`) or export as OME-TIFF.',
    };
  }
  if (lower.endsWith('.ser') || lower.endsWith('.emi')) {
    return {
      format: 'FEI / Thermo Fisher TIA (.ser/.emi)',
      vendor: 'Thermo Fisher / FEI Velox / TIA',
      guidance:
        'FEI/Thermo TEM .ser/.emi file detected. Connect the local Python engine (`vexel serve` with `rosettasciio`) or export as tagged 16-bit TIFF from Velox.',
    };
  }
  if (
    lower.endsWith('.dcm') ||
    (headerBytes.length >= 132 &&
      new TextDecoder('ascii').decode(headerBytes.subarray(128, 132)) === 'DICM')
  ) {
    return {
      format: 'DICOM (.dcm)',
      vendor: 'Medical / Bio-Imaging DICOM',
      guidance:
        'DICOM container detected. Connect the local Python engine (`vexel serve` with `pydicom`) or export as 16-bit TIFF.',
    };
  }
  return null;
}

export function isNativeScientificBinary(
  fileName: string,
  headerBytes: Uint8Array
): 'tiff' | 'dm' | null {
  const lower = fileName.toLowerCase();
  const isTiffMagic =
    headerBytes.length >= 4 &&
    ((headerBytes[0] === 0x49 &&
      headerBytes[1] === 0x49 &&
      headerBytes[2] === 42 &&
      headerBytes[3] === 0) ||
      (headerBytes[0] === 0x4d &&
        headerBytes[1] === 0x4d &&
        headerBytes[2] === 0 &&
        headerBytes[3] === 42));

  if (isTiffMagic || lower.endsWith('.tif') || lower.endsWith('.tiff') || lower.endsWith('.lsm')) {
    return 'tiff';
  }
  if (lower.endsWith('.dm3') || lower.endsWith('.dm4')) {
    return 'dm';
  }
  return null;
}
