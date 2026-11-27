/**
 * The browser engine's multilayer measurement on stacks written by
 * tests/benchmarks/web_layer_parity.py (raw float32 images + cases.json), one JSON line per
 * engine and case. Run through that script: python tests/benchmarks/web_layer_parity.py
 */
import { readFileSync } from 'node:fs';
import { runMultilayerAnalysis } from '../src/engine/multilayerMetrology';
const dir = process.argv[2];
const cases = JSON.parse(readFileSync(`${dir}/cases.json`, 'utf8'));
for (const [name, c] of Object.entries<any>(cases)) {
  const buf = readFileSync(`${dir}/${name}.f32`);
  const gray = new Float32Array(buf.buffer, buf.byteOffset, c.w * c.h);
  const spec: any = { id: name, width: c.w, height: c.h, gray, calibration: { pixelSize: 1, spaceUnit: 'px', cropRows: c.h } };
  for (const geometryMode of ['straight', 'curved']) {
    const r = runMultilayerAnalysis(spec, gray, { geometryMode });
    const th = r.objects.filter((o: any) => o.kind === 'layer').map((o: any) => o.thickness);
    const err = th.length === c.truth.length ? Math.max(...th.map((t: number, i: number) => Math.abs(t - c.truth[i]))) : null;
    console.log(JSON.stringify({ name, engine: `web-${geometryMode}`, layers: th.length, thickness: th, maxErr: err }));
  }
  const pe = Math.max(...c.python.map((t: number, i: number) => Math.abs(t - c.truth[i])));
  console.log(JSON.stringify({ name, engine: 'python', layers: c.python.length, thickness: c.python.map((t: number) => +t.toFixed(2)), maxErr: +pe.toFixed(3) }));
}
