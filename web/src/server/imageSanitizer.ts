import crypto from 'crypto';
import zlib from 'zlib';

/**
 * Pure-Node.js Security Image Sanitizer & Canonical Dataset Packager (G-05).
 *
 * Defense-in-depth second line of defense for online GPU training:
 *  1. Strictly verifies PNG magic header and chunk CRC32 checksums.
 *  2. Enforces dimension bounds & decompression-bomb memory caps (`maxOutputLength`).
 *  3. Fully decodes PNG scanline filters (None/Sub/Up/Average/Paeth) down to pure pixel bytes.
 *  4. Strips ALL ancillary/metadata chunks (`eXIf`, `tEXt`, `zTXt`, `iTXt`, `tIME`, custom chunks)
 *     and any trailing polyglot bytes appended after `IEND`.
 *  5. Re-encodes a canonical minimal PNG containing strictly `IHDR`, fresh `IDAT`, and `IEND`.
 *  6. Packages `image.png`, `mask.png`, `manifest.json`, and `train_vexel_lora.py` into a
 *     standard PKZIP archive (zero external binary dependencies).
 */

const PNG_SIGNATURE = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);

// Precomputed standard IEEE 802.3 CRC32 lookup table (shared by PNG chunks & ZIP archive)
const CRC32_TABLE = (() => {
  const table = new Uint32Array(256);
  for (let i = 0; i < 256; i++) {
    let c = i;
    for (let k = 0; k < 8; k++) {
      c = (c & 1) !== 0 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    }
    table[i] = c >>> 0;
  }
  return table;
})();

export function crc32(buf: Uint8Array, start = 0, end = buf.length): number {
  let c = 0xffffffff;
  for (let i = start; i < end; i++) {
    c = CRC32_TABLE[(c ^ buf[i]) & 0xff] ^ (c >>> 8);
  }
  return (c ^ 0xffffffff) >>> 0;
}

function paethPredictor(a: number, b: number, c: number): number {
  const p = a + b - c;
  const pa = Math.abs(p - a);
  const pb = Math.abs(p - b);
  const pc = Math.abs(p - c);
  if (pa <= pb && pa <= pc) return a;
  if (pb <= pc) return b;
  return c;
}

function makePngChunk(type: string, data: Buffer): Buffer {
  const typeBuf = Buffer.from(type, 'ascii');
  const lenBuf = Buffer.alloc(4);
  lenBuf.writeUInt32BE(data.length, 0);
  const crcPayload = Buffer.concat([typeBuf, data]);
  const crcBuf = Buffer.alloc(4);
  crcBuf.writeUInt32BE(crc32(crcPayload), 0);
  return Buffer.concat([lenBuf, typeBuf, data, crcBuf]);
}

/** Encode raw unfiltered 8-bit pixel buffer (colorType: 0=Gray, 2=RGB, 6=RGBA) into canonical minimal PNG. */
export function encodeCanonicalPng(
  pixels: Uint8Array,
  width: number,
  height: number,
  colorType: 0 | 2 | 6
): Buffer {
  const bpp = colorType === 0 ? 1 : colorType === 2 ? 3 : 4;
  const rowBytes = width * bpp;
  if (pixels.length !== rowBytes * height) {
    throw new Error('Pixel buffer size mismatch for canonical PNG encoding.');
  }

  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(width, 0);
  ihdr.writeUInt32BE(height, 4);
  ihdr[8] = 8; // bitDepth = 8
  ihdr[9] = colorType;
  ihdr[10] = 0; // compression = deflate
  ihdr[11] = 0; // filter = adaptive
  ihdr[12] = 0; // interlace = none

  // Prepend filter byte 0 (None) to each scanline
  const rawScanlines = Buffer.alloc((rowBytes + 1) * height);
  for (let y = 0; y < height; y++) {
    const dst = y * (rowBytes + 1);
    rawScanlines[dst] = 0;
    rawScanlines.set(pixels.subarray(y * rowBytes, (y + 1) * rowBytes), dst + 1);
  }

  const compressed = zlib.deflateSync(rawScanlines, { level: 6 });
  return Buffer.concat([
    PNG_SIGNATURE,
    makePngChunk('IHDR', ihdr),
    makePngChunk('IDAT', compressed),
    makePngChunk('IEND', Buffer.alloc(0)),
  ]);
}

export interface SanitizedPngResult {
  cleanPng: Buffer;
  width: number;
  height: number;
  channels: 1 | 3 | 4;
  colorType: 0 | 2 | 6;
  rawPixels: Uint8Array;
  originalBytes: number;
  sanitizedBytes: number;
  sanitizedSha256: string;
  strippedChunks: string[];
  trailingBytesStripped: number;
}

export interface SanitizePngOptions {
  minDim?: number;
  maxDim?: number;
  maxPixels?: number;
  maxInputBytes?: number;
}

/**
 * Decode an untrusted PNG buffer down to raw pixels and re-encode a clean canonical PNG
 * containing zero ancillary metadata chunks (`eXIf`, `tEXt`, `zTXt`, `iTXt`, etc.) and zero trailing bytes.
 */
export function sanitizePngBuffer(
  input: Buffer,
  options: SanitizePngOptions = {}
): SanitizedPngResult {
  const minDim = options.minDim ?? 16;
  const maxDim = options.maxDim ?? 4096;
  const maxPixels = options.maxPixels ?? 4096 * 4096;
  const maxInputBytes = options.maxInputBytes ?? 14 * 1024 * 1024;

  if (!Buffer.isBuffer(input) || input.length < 33) {
    throw new Error('Invalid PNG payload: buffer too small.');
  }
  if (input.length > maxInputBytes) {
    throw new Error(
      `PNG payload exceeds security cap (${(maxInputBytes / (1024 * 1024)).toFixed(0)} MB).`
    );
  }
  if (!input.subarray(0, 8).equals(PNG_SIGNATURE)) {
    throw new Error('Invalid image header: expected canonical PNG magic signature.');
  }

  let offset = 8;
  let width = 0;
  let height = 0;
  let bitDepth = 0;
  let colorType = -1;
  let interlace = 0;
  let sawIhdr = false;
  let sawIend = false;
  const idatParts: Buffer[] = [];
  const strippedChunks: string[] = [];

  while (offset + 12 <= input.length) {
    const length = input.readUInt32BE(offset);
    const type = input.subarray(offset + 4, offset + 8).toString('ascii');
    const dataStart = offset + 8;
    const dataEnd = dataStart + length;
    const crcOffset = dataEnd;

    if (length > 0x7fffffff || crcOffset + 4 > input.length) {
      throw new Error(`Malformed PNG chunk (${type}) exceeds buffer bounds.`);
    }

    const expectedCrc = input.readUInt32BE(crcOffset);
    const actualCrc = crc32(input, offset + 4, dataEnd);
    if (expectedCrc !== actualCrc) {
      throw new Error(`Corrupted PNG chunk CRC32 in ${type}.`);
    }

    const chunkData = input.subarray(dataStart, dataEnd);

    if (type === 'IHDR') {
      if (sawIhdr || length !== 13) {
        throw new Error('Invalid PNG IHDR chunk.');
      }
      sawIhdr = true;
      width = chunkData.readUInt32BE(0);
      height = chunkData.readUInt32BE(4);
      bitDepth = chunkData[8];
      colorType = chunkData[9];
      const compression = chunkData[10];
      const filterMethod = chunkData[11];
      interlace = chunkData[12];

      if (width < minDim || height < minDim || width > maxDim || height > maxDim) {
        throw new Error(
          `PNG dimensions ${width}×${height} outside allowed range [${minDim}, ${maxDim}].`
        );
      }
      if (width * height > maxPixels) {
        throw new Error(`PNG pixel count ${width * height} exceeds allowed cap (${maxPixels}).`);
      }
      if (bitDepth !== 8) {
        throw new Error(`Unsupported PNG bit depth ${bitDepth}; expected 8-bit.`);
      }
      if (colorType !== 0 && colorType !== 2 && colorType !== 6) {
        throw new Error(`Unsupported PNG color type ${colorType}; expected Gray(0), RGB(2), or RGBA(6).`);
      }
      if (compression !== 0 || filterMethod !== 0 || interlace !== 0) {
        throw new Error('Unsupported PNG compression, filter, or interlace mode.');
      }
    } else if (type === 'IDAT') {
      if (!sawIhdr) throw new Error('PNG IDAT appeared before IHDR.');
      idatParts.push(chunkData);
    } else if (type === 'IEND') {
      sawIend = true;
      offset = crcOffset + 4;
      break;
    } else {
      // Strictly strip every ancillary / non-pixel metadata chunk (eXIf, tEXt, zTXt, iTXt, etc.)
      strippedChunks.push(type);
    }

    offset = crcOffset + 4;
  }

  if (!sawIhdr || !sawIend || idatParts.length === 0) {
    throw new Error('Incomplete PNG stream (missing IHDR, IDAT, or IEND).');
  }

  const trailingBytesStripped = Math.max(0, input.length - offset);
  const channels = (colorType === 0 ? 1 : colorType === 2 ? 3 : 4) as 1 | 3 | 4;
  const rowBytes = width * channels;
  const expectedInflatedBytes = (rowBytes + 1) * height;

  let inflated: Buffer;
  try {
    inflated = zlib.inflateSync(Buffer.concat(idatParts), {
      maxOutputLength: expectedInflatedBytes,
    });
  } catch {
    throw new Error('PNG IDAT decompression failed or triggered decompression-bomb protection.');
  }

  if (inflated.length !== expectedInflatedBytes) {
    throw new Error(
      `Decoded PNG scanline length (${inflated.length}) does not match expected (${expectedInflatedBytes}).`
    );
  }

  // Reconstruct unfiltered raw pixel matrix
  const rawPixels = new Uint8Array(rowBytes * height);
  for (let y = 0; y < height; y++) {
    const srcRowStart = y * (rowBytes + 1);
    const filterType = inflated[srcRowStart];
    const dstRowStart = y * rowBytes;
    const prevRowStart = (y - 1) * rowBytes;

    for (let x = 0; x < rowBytes; x++) {
      const raw = inflated[srcRowStart + 1 + x];
      const a = x >= channels ? rawPixels[dstRowStart + x - channels] : 0;
      const b = y > 0 ? rawPixels[prevRowStart + x] : 0;
      const c = y > 0 && x >= channels ? rawPixels[prevRowStart + x - channels] : 0;

      let recon = raw;
      if (filterType === 0) {
        recon = raw;
      } else if (filterType === 1) {
        recon = (raw + a) & 0xff;
      } else if (filterType === 2) {
        recon = (raw + b) & 0xff;
      } else if (filterType === 3) {
        recon = (raw + ((a + b) >>> 1)) & 0xff;
      } else if (filterType === 4) {
        recon = (raw + paethPredictor(a, b, c)) & 0xff;
      } else {
        throw new Error(`Invalid PNG scanline filter type (${filterType}).`);
      }
      rawPixels[dstRowStart + x] = recon;
    }
  }

  const cleanPng = encodeCanonicalPng(rawPixels, width, height, colorType as 0 | 2 | 6);
  const sanitizedSha256 = crypto.createHash('sha256').update(cleanPng).digest('hex');

  return {
    cleanPng,
    width,
    height,
    channels,
    colorType: colorType as 0 | 2 | 6,
    rawPixels,
    originalBytes: input.length,
    sanitizedBytes: cleanPng.length,
    sanitizedSha256,
    strippedChunks,
    trailingBytesStripped,
  };
}

/** Decode a base64 or `data:image/png;base64,...` string safely into a Buffer. */
export function decodeBase64PngString(raw: string, maxEncodedChars = 20 * 1024 * 1024): Buffer {
  if (typeof raw !== 'string' || raw.length === 0 || raw.length > maxEncodedChars) {
    throw new Error('Invalid or oversized base64 PNG payload.');
  }
  const cleaned = raw.replace(/^data:image\/png;base64,/i, '').trim();
  if (!/^[A-Za-z0-9+/=\s]+$/.test(cleaned)) {
    throw new Error('Malformed base64 characters in PNG upload.');
  }
  return Buffer.from(cleaned, 'base64');
}

/**
 * Pure-Node.js canonical PKZIP archive builder (STORE mode + IEEE CRC32).
 * Zero external packages; fully compatible with Python `zipfile`, macOS, Linux, and Windows.
 */
export function buildZipArchive(entries: Array<{ name: string; data: Buffer }>): Buffer {
  const localParts: Buffer[] = [];
  const centralParts: Buffer[] = [];
  let offset = 0;

  for (const entry of entries) {
    // Reject path traversal in zip entry names
    const safeName = entry.name.replace(/\\/g, '/').replace(/^\/+/, '').replace(/\.\.\//g, '');
    const nameBuf = Buffer.from(safeName, 'utf8');
    const data = entry.data;
    const sum = crc32(data);

    // Local File Header (30 bytes + name)
    const localHeader = Buffer.alloc(30);
    localHeader.writeUInt32LE(0x04034b50, 0); // signature
    localHeader.writeUInt16LE(20, 4); // version needed (2.0)
    localHeader.writeUInt16LE(0, 6); // flags
    localHeader.writeUInt16LE(0, 8); // compression = 0 (STORE)
    localHeader.writeUInt16LE(0, 10); // mod time
    localHeader.writeUInt16LE(0x5421, 12); // mod date (deterministic)
    localHeader.writeUInt32LE(sum, 14);
    localHeader.writeUInt32LE(data.length, 18);
    localHeader.writeUInt32LE(data.length, 22);
    localHeader.writeUInt16LE(nameBuf.length, 26);
    localHeader.writeUInt16LE(0, 28); // extra len

    localParts.push(localHeader, nameBuf, data);

    // Central Directory Header (46 bytes + name)
    const centralHeader = Buffer.alloc(46);
    centralHeader.writeUInt32LE(0x02014b50, 0);
    centralHeader.writeUInt16LE(20, 4); // version made by
    centralHeader.writeUInt16LE(20, 6); // version needed
    centralHeader.writeUInt16LE(0, 8); // flags
    centralHeader.writeUInt16LE(0, 10); // compression = STORE
    centralHeader.writeUInt16LE(0, 12); // mod time
    centralHeader.writeUInt16LE(0x5421, 14); // mod date
    centralHeader.writeUInt32LE(sum, 16);
    centralHeader.writeUInt32LE(data.length, 20);
    centralHeader.writeUInt32LE(data.length, 24);
    centralHeader.writeUInt16LE(nameBuf.length, 28);
    centralHeader.writeUInt16LE(0, 30); // extra len
    centralHeader.writeUInt16LE(0, 32); // comment len
    centralHeader.writeUInt16LE(0, 34); // disk start
    centralHeader.writeUInt16LE(0, 36); // internal attrs
    centralHeader.writeUInt32LE(0, 38); // external attrs
    centralHeader.writeUInt32LE(offset, 42); // relative offset of local header

    centralParts.push(centralHeader, nameBuf);
    offset += localHeader.length + nameBuf.length + data.length;
  }

  const centralDir = Buffer.concat(centralParts);
  const eocd = Buffer.alloc(22);
  eocd.writeUInt32LE(0x06054b50, 0); // EOCD signature
  eocd.writeUInt16LE(0, 4); // disk number
  eocd.writeUInt16LE(0, 6); // start disk
  eocd.writeUInt16LE(entries.length, 8);
  eocd.writeUInt16LE(entries.length, 10);
  eocd.writeUInt32LE(centralDir.length, 12);
  eocd.writeUInt32LE(offset, 16);
  eocd.writeUInt16LE(0, 20); // comment len

  return Buffer.concat([...localParts, centralDir, eocd]);
}
