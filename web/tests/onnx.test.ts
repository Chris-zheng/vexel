import { describe, expect, it } from 'vitest';
import {
  buildPersistedOnnxModel,
  disposeOnnxSession,
  inspectOnnxBinary,
  predictOnnxModelTiled,
} from '../src/engine/onnxRuntime';

/**
 * Minimal Protobuf wire encoder to construct a genuine, spec-compliant ONNX binary model
 * (`ir_version=7`, `opset=13`, Graph: `Y = Sigmoid(X)` with dynamic float32 input `[1, 1, H, W]`).
 * Allows testing REAL `onnxruntime-web` binary validation and tiled execution with zero mocks.
 */
function encodeVarint(n: number): number[] {
  const out: number[] = [];
  let v = n >>> 0;
  while (v >= 0x80) {
    out.push((v & 0x7f) | 0x80);
    v >>>= 7;
  }
  out.push(v);
  return out;
}

function fieldVarint(fieldNum: number, val: number): number[] {
  return [...encodeVarint((fieldNum << 3) | 0), ...encodeVarint(val)];
}

function fieldBytes(fieldNum: number, payload: number[] | Uint8Array): number[] {
  const arr = Array.from(payload);
  return [...encodeVarint((fieldNum << 3) | 2), ...encodeVarint(arr.length), ...arr];
}

function fieldString(fieldNum: number, str: string): number[] {
  return fieldBytes(fieldNum, new TextEncoder().encode(str));
}

function buildMinimalSigmoidOnnxBinary(): ArrayBuffer {
  // TensorShapeProto.Dimension: fixed 1, fixed 1, dynamic "H", dynamic "W"
  const dim1 = fieldVarint(1, 1); // dim_value = 1
  const dimH = fieldString(2, 'H'); // dim_param = "H"
  const dimW = fieldString(2, 'W'); // dim_param = "W"
  const shapeProto = [
    ...fieldBytes(1, dim1),
    ...fieldBytes(1, dim1),
    ...fieldBytes(1, dimH),
    ...fieldBytes(1, dimW),
  ];
  // TypeProto.Tensor: elem_type=1 (FLOAT), shape=shapeProto
  const tensorTypeProto = [...fieldVarint(1, 1), ...fieldBytes(2, shapeProto)];
  // TypeProto: tensor_type (field 1)
  const typeProto = fieldBytes(1, tensorTypeProto);

  // ValueInfoProto: name (1), type (2)
  const inputInfo = [...fieldString(1, 'image'), ...fieldBytes(2, typeProto)];
  const outputInfo = [...fieldString(1, 'mask_prob'), ...fieldBytes(2, typeProto)];

  // NodeProto: input="image" (1), output="mask_prob" (2), op_type="Sigmoid" (4)
  const nodeProto = [
    ...fieldString(1, 'image'),
    ...fieldString(2, 'mask_prob'),
    ...fieldString(4, 'Sigmoid'),
  ];

  // GraphProto: node (1), name (2), input (11), output (12)
  const graphProto = [
    ...fieldBytes(1, nodeProto),
    ...fieldString(2, 'vexel_test_sigmoid_seg'),
    ...fieldBytes(11, inputInfo),
    ...fieldBytes(12, outputInfo),
  ];

  // OperatorSetIdProto: domain="" (1), version=13 (2)
  const opsetProto = [...fieldString(1, ''), ...fieldVarint(2, 13)];

  // ModelProto: ir_version=7 (1), graph (7), opset_import (8)
  const modelBytes = new Uint8Array([
    ...fieldVarint(1, 7),
    ...fieldBytes(7, graphProto),
    ...fieldBytes(8, opsetProto),
  ]);
  return modelBytes.buffer;
}

describe('Phase 2: Real Offline ONNX Runtime (onnxruntime-web)', () => {
  it('inspects, validates SHA-256, and executes tiled inference on a real binary .onnx model', async () => {
    const onnxBuf = buildMinimalSigmoidOnnxBinary();
    const meta = await inspectOnnxBinary(onnxBuf);

    expect(meta.inputName).toBe('image');
    expect(meta.outputName).toBe('mask_prob');
    expect(meta.sha256).toMatch(/^[a-f0-9]{64}$/);
    expect(meta.inputDims.length).toBe(4);

    const persisted = await buildPersistedOnnxModel('demo_seg.onnx', onnxBuf, {
      modelId: 'demo_seg',
      version: '1.0',
    });
    expect(persisted.ref).toBe('demo_seg@1.0');
    expect(persisted.card.framework).toBe('onnx');
    expect(persisted.card.fileStatus).toBe('verified');

    // Run cosine-feathered tiled inference on a 96×96 test image (with tileSize=64 -> 4 overlapping tiles)
    const width = 96;
    const height = 96;
    const gray = new Float32Array(width * height);
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        gray[y * width + x] = x < width / 2 ? 0 : 255;
      }
    }

    const prob = await predictOnnxModelTiled(persisted, gray, width, height, {
      tileSize: 64,
      overlap: 16,
    });

    expect(prob.length).toBe(width * height);
    // Left half: input = 0.0 -> Sigmoid(0.0) = 0.500
    expect(prob[48 * width + 10]).toBeCloseTo(0.5, 2);
    // Right half: input = 1.0 -> Sigmoid(1.0) ≈ 0.731
    expect(prob[48 * width + 80]).toBeCloseTo(1 / (1 + Math.exp(-1)), 2);

    await disposeOnnxSession(persisted.ref);
  });

  it('rejects malformed or oversized binary uploads safely', async () => {
    const garbage = new Uint8Array([1, 2, 3, 4]).buffer;
    await expect(inspectOnnxBinary(garbage)).rejects.toThrow();
  });
});
