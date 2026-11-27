import { describe, expect, it } from 'vitest';
import { evaluate2DBudget, evaluate3DBudget } from '../src/engine/resourceBudget';
import {
  DEEP_FEATURE_DIM,
  extractMultiScaleFeatures16,
  extractMultiScaleFeatures24,
  predictDeepFewShotTiled,
  trainDeepFewShotHead,
} from '../src/engine/deepFewShotCore';
import { parseOmeZarrMetadata } from '../src/server/omeZarrBridge';
import {
  computeLocalModelChecksum,
  exportLocalModelJson,
  parseImportedLocalModelJson,
  saveLocalModel,
  loadAllLocalModels,
  deleteLocalModel,
} from '../src/engine/modelStore';
import { generateSyntheticKidney3DStack, loadZStackFromTiffBuffer } from '../src/engine/volume3d';
import { analyzeVolume3D } from '../src/engine/volumeAnalysisCore';
import type { TrainedLocalModel } from '../src/engine/mlRuntime';
import { fastSha256Hex } from '../src/engine/samples';

/**
 * Helper: build a valid uncompressed multi-page OME-TIFF binary buffer in pure JS
 * (Little-Endian II, magic 42, 8-bit or 16-bit grayscale, linked IFD chain, OME-XML XY/Z spacing).
 */
function buildSyntheticMultiPageOmeTiff(opts: {
  width: number;
  height: number;
  depth: number;
  bitsPerSample?: 8 | 16;
  physicalSizeX?: number;
  physicalSizeZ?: number;
}): ArrayBuffer {
  const { width, height, depth } = opts;
  const bps = opts.bitsPerSample ?? 8;
  const bytesPerSample = bps / 8;
  const planeBytes = width * height * bytesPerSample;
  const omeXml = `<?xml version="1.0" encoding="UTF-8"?><OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06"><Image ID="Image:0"><Pixels PhysicalSizeX="${opts.physicalSizeX ?? 0.25}" PhysicalSizeXUnit="µm" PhysicalSizeZ="${opts.physicalSizeZ ?? 0.75}" PhysicalSizeZUnit="µm" SizeX="${width}" SizeY="${height}" SizeZ="${depth}"/></Image></OME>\0`;
  const omeBytes = new TextEncoder().encode(omeXml);

  // Layout: [Header 8B] [OME-XML] [for each page: IFD (2 + N*12 + 4) + pixel data]
  const omeOffset = 8;
  let cursor = omeOffset + omeBytes.length;
  if (cursor % 2 !== 0) cursor++;

  const pageIfdOffsets: number[] = [];
  const pageDataOffsets: number[] = [];
  for (let z = 0; z < depth; z++) {
    const tagCount = z === 0 ? 10 : 9;
    const ifdSize = 2 + tagCount * 12 + 4;
    pageIfdOffsets.push(cursor);
    cursor += ifdSize;
    pageDataOffsets.push(cursor);
    cursor += planeBytes;
    if (cursor % 2 !== 0) cursor++;
  }

  const buf = new ArrayBuffer(cursor);
  const view = new DataView(buf);
  const u8 = new Uint8Array(buf);

  // Header: II, 42, first IFD offset
  u8[0] = 0x49;
  u8[1] = 0x49;
  view.setUint16(2, 42, true);
  view.setUint32(4, pageIfdOffsets[0], true);
  u8.set(omeBytes, omeOffset);

  for (let z = 0; z < depth; z++) {
    const entries: Array<[number, number, number, number]> = [
      [256, 4, 1, width], // ImageWidth
      [257, 4, 1, height], // ImageLength
      [258, 3, 1, bps], // BitsPerSample
      [259, 3, 1, 1], // Compression = None
      [262, 3, 1, 1], // Photometric = BlackIsZero
      [273, 4, 1, pageDataOffsets[z]], // StripOffsets
      [277, 3, 1, 1], // SamplesPerPixel
      [278, 4, 1, height], // RowsPerStrip
      [279, 4, 1, planeBytes], // StripByteCounts
    ];
    if (z === 0) {
      entries.push([270, 2, omeBytes.length, omeOffset]); // ImageDescription (OME-XML)
    }
    entries.sort((a, b) => a[0] - b[0]);

    let p = pageIfdOffsets[z];
    view.setUint16(p, entries.length, true);
    p += 2;
    for (const [tag, type, count, val] of entries) {
      view.setUint16(p, tag, true);
      view.setUint16(p + 2, type, true);
      view.setUint32(p + 4, count, true);
      if (type === 3 && count === 1) {
        view.setUint16(p + 8, val, true);
      } else {
        view.setUint32(p + 8, val, true);
      }
      p += 12;
    }
    view.setUint32(p, z + 1 < depth ? pageIfdOffsets[z + 1] : 0, true);

    // Fill slice pixels with a bright 3D sphere centered at (w/2, h/2, d/2)
    const dOff = pageDataOffsets[z];
    const cx = width / 2;
    const cy = height / 2;
    const cz = depth / 2;
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const r = Math.hypot((x - cx) / (width * 0.25), (y - cy) / (height * 0.25), (z - cz) / (depth * 0.35));
        const val = r <= 1 ? (bps === 16 ? 52000 : 210) : (bps === 16 ? 4000 : 25);
        const idx = y * width + x;
        if (bps === 16) {
          view.setUint16(dOff + idx * 2, val, true);
        } else {
          u8[dOff + idx] = val;
        }
      }
    }
  }

  return buf;
}

describe('Phase 1: 2D & 3D Resource Budget Evaluator', () => {
  it('classifies 2D image dimensions into all 4 safety tiers accurately', () => {
    expect(evaluate2DBudget(1024, 1024).tier).toBe('optimal');
    expect(evaluate2DBudget(2048, 2048).tier).toBe('optimal');
    expect(evaluate2DBudget(4096, 4096).tier).toBe('tiled_safe');
    expect(evaluate2DBudget(8192, 8192).tier).toBe('lod_guarded');
    expect(evaluate2DBudget(12000, 12000).tier).toBe('offload_gpu_required');
  });

  it('classifies 3D Z-Stack volumes into all 4 safety tiers accurately', () => {
    expect(evaluate3DBudget(128, 128, 48).tier).toBe('optimal');
    expect(evaluate3DBudget(512, 512, 64).tier).toBe('optimal');
    expect(evaluate3DBudget(512, 512, 128).tier).toBe('tiled_safe');
    expect(evaluate3DBudget(1024, 1024, 64).tier).toBe('lod_guarded');
    expect(evaluate3DBudget(1024, 1024, 128).tier).toBe('offload_gpu_required');
  });
});

describe('Phase 1: Few-Shot Deep Learning Engine & Model Persistence', () => {
  it('converges on a dual-class specimen, runs fast with cached features, and reproduces inference 100% after export/import round-trip', async () => {
    const width = 192;
    const height = 192;
    const gray = new Float32Array(width * height);
    const mask = new Uint8Array(width * height);

    // Build synthetic bright cellular blobs (class 1) on dark background (class 2)
    const centers = [
      [48, 48, 22],
      [140, 52, 24],
      [54, 142, 20],
      [138, 138, 25],
    ];
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const idx = y * width + x;
        let inside = false;
        for (const [cx, cy, r] of centers) {
          if ((x - cx) ** 2 + (y - cy) ** 2 <= r * r) {
            inside = true;
            break;
          }
        }
        const noise = ((x * 31 + y * 17) % 13) - 6;
        gray[idx] = inside ? 205 + noise : 42 + noise;
        // Sparse ground-truth brush annotations across all 4 spatial quadrants
        for (const [cx, cy, r] of centers) {
          if ((x - cx) ** 2 + (y - cy) ** 2 <= (r * 0.55) ** 2) {
            mask[idx] = 1;
          }
        }
        if (x < 18 || x > width - 19 || (Math.abs(x - 96) < 8 && y > 20 && y < height - 20)) {
          mask[idx] = 2;
        }
      }
    }

    const t0 = performance.now();
    const feats = extractMultiScaleFeatures16(gray, width, height);
    const outcome = trainDeepFewShotHead(
      feats,
      mask,
      width,
      ['Cell', 'Background'],
      {
        hiddenDim: 24,
        epochs: 28,
        learningRate: 0.03,
        hardNegativeMining: true,
        bilateralSmooth: true,
      },
      false
    );
    const totalMs = performance.now() - t0;

    // Cached-feature incremental re-train timing check
    const tCache0 = performance.now();
    const cachedOutcome = trainDeepFewShotHead(
      feats,
      mask,
      width,
      ['Cell', 'Background'],
      {
        hiddenDim: 24,
        epochs: 28,
        learningRate: 0.03,
        hardNegativeMining: true,
        bilateralSmooth: true,
      },
      true
    );
    const cacheTrainMs = performance.now() - tCache0;

    expect(outcome.metrics.meanIoU).toBeGreaterThanOrEqual(0.85);
    expect(outcome.metrics.macroF1).toBeGreaterThanOrEqual(0.9);
    expect(totalMs).toBeLessThan(3000);
    expect(cacheTrainMs).toBeLessThan(1000);
    expect(cachedOutcome.cacheHit).toBe(true);

    // Predict before persistence/export
    const probBefore = predictDeepFewShotTiled(outcome.weights, gray, width, height, height, 128, 16);

    const trainedModel: TrainedLocalModel = {
      card: {
        id: 'test_kidney_dl',
        version: '1.0',
        ref: 'test_kidney_dl@1.0',
        nameZh: '测试少样本模型',
        nameEn: 'Test Few-Shot Model',
        task: 'segment.instances',
        framework: 'onnx',
        modality: 'sem',
        license: 'Local-User-Owned',
        commercialUse: true,
        trainingData: [{ name: 'Synthetic Unit Specimen', license: 'Local-User-Owned' }],
        sha256: fastSha256Hex(JSON.stringify(outcome.weights)),
        tiledInference: true,
        tileSize: 128,
        fileStatus: 'local-trained',
        metrics: {
          meanIoU: outcome.metrics.meanIoU,
          macroF1: outcome.metrics.macroF1,
          accuracy: outcome.metrics.accuracy,
        },
        descriptionZh: 'Test',
        descriptionEn: 'Test',
        isLocalTrained: true,
        classes: ['Cell', 'Background'],
      },
      classes: ['Cell', 'Background'],
      deepHead: outcome.weights,
      classStats: [],
      stumps: [],
      metrics: outcome.metrics,
    };

    // Persistence round-trip (modelStore + portable JSON SHA-256 verification)
    await saveLocalModel(trainedModel);
    const restoredList = await loadAllLocalModels();
    expect(restoredList.some((m) => m.card.ref === 'test_kidney_dl@1.0')).toBe(true);

    const exportedJson = exportLocalModelJson(trainedModel);
    const imported = parseImportedLocalModelJson(exportedJson);
    expect(computeLocalModelChecksum(imported)).toBe(computeLocalModelChecksum(trainedModel));

    // Tampered checksum must be rejected
    const tampered = exportedJson.replace('"sha256": "', '"sha256": "deadbeef');
    expect(() => parseImportedLocalModelJson(tampered)).toThrow(/SHA-256/i);

    // 100% pixel-identical reproducible inference after round-trip
    const probAfter = predictDeepFewShotTiled(imported.deepHead!, gray, width, height, height, 128, 16);
    expect(probAfter.length).toBe(probBefore.length);
    for (let i = 0; i < probBefore.length; i++) {
      expect(probAfter[i]).toBe(probBefore[i]);
    }

    await deleteLocalModel('test_kidney_dl@1.0');
  });
});

describe('Phase 3 & 4: Multi-Page OME-TIFF Z-Stack & 3D Volumetric Analysis', () => {
  it('decodes 8-bit and 16-bit Multi-Page OME-TIFF Z-Stacks with calibrated XY/Z spacing', async () => {
    const tiff8 = buildSyntheticMultiPageOmeTiff({
      width: 64,
      height: 64,
      depth: 8,
      bitsPerSample: 8,
      physicalSizeX: 0.3,
      physicalSizeZ: 0.9,
    });
    const res8 = await loadZStackFromTiffBuffer(tiff8, 'confocal_8bit.ome.tif');
    expect(res8.volume.width).toBe(64);
    expect(res8.volume.height).toBe(64);
    expect(res8.volume.depth).toBe(8);
    expect(res8.volume.voxelSizeXY).toBeCloseTo(0.3, 3);
    expect(res8.volume.voxelSizeZ).toBeCloseTo(0.9, 3);
    expect(res8.volume.spaceUnit).toBe('µm');
    expect(res8.volume.data.length).toBe(64 * 64 * 8);

    const tiff16 = buildSyntheticMultiPageOmeTiff({
      width: 48,
      height: 48,
      depth: 6,
      bitsPerSample: 16,
      physicalSizeX: 0.2,
      physicalSizeZ: 0.6,
    });
    const res16 = await loadZStackFromTiffBuffer(tiff16, 'confocal_16bit.ome.tif');
    expect(res16.volume.width).toBe(48);
    expect(res16.volume.height).toBe(48);
    expect(res16.volume.depth).toBe(6);
    expect(res16.volume.voxelSizeXY).toBeCloseTo(0.2, 3);
    expect(res16.volume.voxelSizeZ).toBeCloseTo(0.6, 3);
    // Center voxel should be significantly brighter than corner background
    const centerIdx = 3 * 48 * 48 + 24 * 48 + 24;
    expect(res16.volume.data[centerIdx]).toBeGreaterThan(180);
    expect(res16.volume.data[0]).toBeLessThan(40);
  });

  it('analyzes built-in 3D Kidney Z-Stack (128×128×48) in < 2s with exact cell/nucleus counts and calibrated 3D N/C ratio', () => {
    const vol = generateSyntheticKidney3DStack();
    const t0 = performance.now();
    const res = analyzeVolume3D({
      volumeId: vol.id,
      data: vol.data,
      width: vol.width,
      height: vol.height,
      depth: vol.depth,
      voxelSizeXY: vol.voxelSizeXY,
      voxelSizeZ: vol.voxelSizeZ,
      spaceUnit: vol.spaceUnit,
      cellThreshold: 0,
      nucleusDarkRatio: 0.65,
      minCellVoxels: 45,
    });
    const elapsedMs = performance.now() - t0;

    expect(elapsedMs).toBeLessThan(2000);
    expect(res.cellCount).toBe(5);
    expect(res.nucleusCount).toBe(5);
    expect(res.meanCellVolume).toBeGreaterThan(500);
    expect(res.meanSphericity).toBeGreaterThan(0.65);
    expect(res.meanSphericity).toBeLessThanOrEqual(1.0);
    expect(res.meanNcVolumePct).toBeGreaterThan(10);
    expect(res.meanNcVolumePct).toBeLessThan(40);
    expect(res.labelVolume.length).toBe(vol.width * vol.height * vol.depth);
  });

  it('computes rotation-invariant 3D PCA principal axis lengths within < 5% error on a rotated anisotropic 3D ellipsoid (G-07)', () => {
    const width = 96;
    const height = 96;
    const depth = 64;
    const voxelSizeXY = 0.5;
    const voxelSizeZ = 0.75;
    const data = new Uint8Array(width * height * depth).fill(15);

    // Physical semi-axes (in µm): a = 15, b = 9, c = 5 -> full diameters [30, 18, 10] µm
    const a = 15;
    const b = 9;
    const c = 5;
    const cx = (width / 2) * voxelSizeXY;
    const cy = (height / 2) * voxelSizeXY;
    const cz = (depth / 2) * voxelSizeZ;

    // Apply compound 3D rotation: 35° around Z then 25° around Y
    const thZ = (35 * Math.PI) / 180;
    const thY = (25 * Math.PI) / 180;
    const cosZ = Math.cos(thZ);
    const sinZ = Math.sin(thZ);
    const cosY = Math.cos(thY);
    const sinY = Math.sin(thY);

    for (let z = 0; z < depth; z++) {
      const pz = z * voxelSizeZ - cz;
      for (let y = 0; y < height; y++) {
        const py = y * voxelSizeXY - cy;
        for (let x = 0; x < width; x++) {
          const px = x * voxelSizeXY - cx;
          // Inverse rotate into ellipsoid local frame
          const x1 = cosZ * px + sinZ * py;
          const y1 = -sinZ * px + cosZ * py;
          const z1 = pz;
          const xr = cosY * x1 - sinY * z1;
          const yr = y1;
          const zr = sinY * x1 + cosY * z1;

          if ((xr * xr) / (a * a) + (yr * yr) / (b * b) + (zr * zr) / (c * c) <= 1.0) {
            data[z * width * height + y * width + x] = 215;
          }
        }
      }
    }

    const res = analyzeVolume3D({
      volumeId: 'rotated_ellipsoid_pca_test',
      data,
      width,
      height,
      depth,
      voxelSizeXY,
      voxelSizeZ,
      spaceUnit: 'µm',
      cellThreshold: 120,
      nucleusDarkRatio: 0.4,
      minCellVoxels: 100,
    });

    expect(res.cellCount).toBe(1);
    const obj = res.objects[0];
    const [L1, L2, L3] = obj.axesPhys;
    const expected = [2 * a, 2 * b, 2 * c]; // [30, 18, 10]

    expect(Math.abs(L1 - expected[0]) / expected[0]).toBeLessThan(0.05);
    expect(Math.abs(L2 - expected[1]) / expected[1]).toBeLessThan(0.05);
    expect(Math.abs(L3 - expected[2]) / expected[2]).toBeLessThan(0.05);
    expect(obj.elongation).toBeDefined();
    expect(obj.flatness).toBeDefined();
    expect(Math.abs(obj.elongation! - a / b) / (a / b)).toBeLessThan(0.05);
    expect(Math.abs(obj.flatness! - b / c) / (b / c)).toBeLessThan(0.05);
  });
});

describe('P2 (G-11): 24-Channel Multi-Scale Pyramid + External Embedding Slot & OME-Zarr Pyramid Bridge', () => {
  it('extracts 24 finite channels, blends external WebGPU/ONNX embeddings, and converges with mIoU >= 0.85', () => {
    expect(DEEP_FEATURE_DIM).toBe(24);
    const width = 160;
    const height = 160;
    const gray = new Float32Array(width * height);
    const mask = new Uint8Array(width * height);

    const centers = [
      [42, 42, 20],
      [118, 46, 22],
      [46, 118, 19],
      [116, 116, 21],
    ];
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const idx = y * width + x;
        let inside = false;
        for (const [cx, cy, r] of centers) {
          if ((x - cx) ** 2 + (y - cy) ** 2 <= r * r) {
            inside = true;
            break;
          }
        }
        // Add subtle directional shading ramp to stress wide-context r=15 channel
        const shading = (x / width) * 18;
        gray[idx] = (inside ? 195 : 48) + shading;
        for (const [cx, cy, r] of centers) {
          if ((x - cx) ** 2 + (y - cy) ** 2 <= (r * 0.55) ** 2) mask[idx] = 1;
        }
        if (x < 14 || x > width - 15 || (Math.abs(x - 80) < 7 && y > 16 && y < height - 16)) {
          mask[idx] = 2;
        }
      }
    }

    const t0 = performance.now();
    const feats24 = extractMultiScaleFeatures24(gray, width, height);
    expect(feats24.length).toBe(width * height * 24);
    for (let i = 0; i < 24 * 64; i++) {
      expect(Number.isFinite(feats24[i])).toBe(true);
    }

    // Verify optional external embedding fusion slot modifies channels 16..23 cleanly
    const ext = new Float32Array(width * height * 4).fill(2.0);
    const featsFused = extractMultiScaleFeatures24(gray, width, height, undefined, {
      data: ext,
      channels: 4,
    });
    expect(featsFused[16]).not.toBe(feats24[16]);

    const outcome = trainDeepFewShotHead(
      feats24,
      mask,
      width,
      ['Target', 'Background'],
      {
        hiddenDim: 24,
        epochs: 28,
        learningRate: 0.03,
        hardNegativeMining: true,
        bilateralSmooth: true,
      },
      false
    );
    const elapsedMs = performance.now() - t0;

    expect(outcome.weights.featureDim).toBe(24);
    expect(outcome.metrics.meanIoU).toBeGreaterThanOrEqual(0.85);
    expect(elapsedMs).toBeLessThan(3000);

    const probMap = predictDeepFewShotTiled(outcome.weights, gray, width, height, height, 128, 16);
    expect(probMap.length).toBe(width * height);
    expect(probMap[42 * width + 42]).toBeGreaterThan(0.7);
    expect(probMap[4 * width + 4]).toBeLessThan(0.3);
  });

  it('parses OME-NGFF v0.4 multiscale metadata and auto-selects optimal browser 3D pyramid level', () => {
    const meta = {
      multiscales: [
        {
          version: '0.4',
          name: 'Large_LightSheet_Organoid',
          axes: [
            { name: 'z', type: 'space', unit: 'micrometer' },
            { name: 'y', type: 'space', unit: 'micrometer' },
            { name: 'x', type: 'space', unit: 'micrometer' },
          ],
          datasets: [
            { path: '0', coordinateTransformations: [{ type: 'scale', scale: [0.5, 0.16, 0.16] }] },
            { path: '1', coordinateTransformations: [{ type: 'scale', scale: [0.5, 0.32, 0.32] }] },
            { path: '2', coordinateTransformations: [{ type: 'scale', scale: [1.0, 0.64, 0.64] }] },
            { path: '3', coordinateTransformations: [{ type: 'scale', scale: [2.0, 1.28, 1.28] }] },
          ],
        },
      ],
    };

    const result = parseOmeZarrMetadata(meta, {
      shapesByPath: {
        '0': [512, 4096, 4096],
        '1': [512, 2048, 2048],
        '2': [256, 1024, 1024],
        '3': [128, 512, 512],
      },
    });

    expect(result.outOfCoreRequired).toBe(true);
    expect(result.levels.length).toBe(4);
    expect(result.levels[0].fitsBrowser3DBudget).toBe(false);
    expect(result.recommendedLevel.path).toBe('3');
    expect(result.recommendedLevel.fitsBrowser3DBudget).toBe(true);
    expect(result.recommendedLevel.voxelSizeXY).toBeCloseTo(1.28, 3);
    expect(result.recommendedLevel.voxelSizeZ).toBeCloseTo(2.0, 3);
    expect(result.recommendedLevel.spaceUnit).toBe('µm');
  });
});
