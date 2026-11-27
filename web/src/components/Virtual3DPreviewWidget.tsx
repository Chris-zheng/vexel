import { useEffect, useMemo, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { Box, Maximize2, RotateCcw } from 'lucide-react';
import type {
  Lang,
  OccludedDisplayMode,
  RoiRect,
  RunResult,
  Specimen,
} from '../types/vexel';
import { computeSurfaceRoughness } from '../engine/volume3d';

interface Virtual3DPreviewWidgetProps {
  specimen: Specimen;
  result: RunResult;
  lang: Lang;
  selectedObjectId: number | null;
  occludedDisplayMode: OccludedDisplayMode;
  roi: RoiRect | null;
  onOpen3DStudio?: () => void;
}

type ShadingStyle = 'sem' | 'colormap' | 'wireframe';

function elevationRgb(t: number): [number, number, number] {
  const v = Math.max(0, Math.min(1, t));
  const stops: Array<[number, number, number, number]> = [
    [0.0, 0.05, 0.08, 0.17],
    [0.25, 0.06, 0.46, 0.56],
    [0.52, 0.07, 0.72, 0.5],
    [0.78, 0.96, 0.62, 0.05],
    [1.0, 0.99, 0.95, 0.78],
  ];
  for (let i = 0; i < stops.length - 1; i++) {
    const a = stops[i];
    const b = stops[i + 1];
    if (v <= b[0]) {
      const f = (v - a[0]) / Math.max(1e-6, b[0] - a[0]);
      return [
        a[1] + (b[1] - a[1]) * f,
        a[2] + (b[2] - a[2]) * f,
        a[3] + (b[3] - a[3]) * f,
      ];
    }
  }
  return [0.99, 0.95, 0.78];
}

export function Virtual3DPreviewWidget({
  specimen,
  result,
  lang,
  occludedDisplayMode,
  roi,
  onOpen3DStudio,
}: Virtual3DPreviewWidgetProps) {
  const mountRef = useRef<HTMLDivElement>(null);
  const [shading, setShading] = useState<ShadingStyle>('sem');
  const [zScale, setZScale] = useState(1.25);

  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);
  const meshGroupRef = useRef<THREE.Group | null>(null);
  const requestRenderRef = useRef<() => void>(() => {});

  const fieldStats = useMemo(() => {
    const fullW = specimen.width;
    const fullH = specimen.calibration.cropRows ?? specimen.height;
    const rx = roi ? Math.max(0, Math.min(fullW - 1, Math.round(roi.x))) : 0;
    const ry = roi ? Math.max(0, Math.min(fullH - 1, Math.round(roi.y))) : 0;
    const fovW = roi ? Math.max(16, Math.min(fullW - rx, Math.round(roi.width))) : fullW;
    const fovH = roi ? Math.max(16, Math.min(fullH - ry, Math.round(roi.height))) : fullH;
    const srcGray = result.processedGray ?? specimen.gray;

    const gw = 72;
    const gh = Math.max(36, Math.min(72, Math.round((gw * fovH) / Math.max(1, fovW))));
    const heights = new Float32Array(gw * gh);
    let minV = Infinity;
    let maxV = -Infinity;

    for (let y = 0; y < gh; y++) {
      const sy = Math.min(fullH - 1, ry + Math.floor(((y + 0.5) / gh) * fovH));
      for (let x = 0; x < gw; x++) {
        const sx = Math.min(fullW - 1, rx + Math.floor(((x + 0.5) / gw) * fovW));
        const v = srcGray[sy * fullW + sx] ?? 0;
        heights[y * gw + x] = v;
        if (v < minV) minV = v;
        if (v > maxV) maxV = v;
      }
    }
    const span = Math.max(10, maxV - minV);
    for (let i = 0; i < heights.length; i++) {
      heights[i] = (heights[i] - minV) / span;
    }
    const rough = computeSurfaceRoughness(heights, zScale * 10);
    const activeObjs = result.objects.filter((o) => !o.excluded);
    const occludedCount = activeObjs.filter((o) => o.occluded).length;

    return {
      gw,
      gh,
      fovW,
      fovH,
      heights,
      rough,
      completeCount: activeObjs.length - occludedCount,
      occludedCount,
    };
  }, [specimen, result, roi, zScale]);

  // Initialize compact on-demand WebGL2 renderer
  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;
    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'low-power' });
    } catch {
      return;
    }
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setSize(mount.clientWidth || 300, 196);
    renderer.setClearColor(0x070b14, 1);

    mount.innerHTML = '';
    mount.appendChild(renderer.domElement);
    renderer.domElement.style.display = 'block';
    renderer.domElement.style.width = '100%';
    renderer.domElement.style.height = '196px';

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(42, (mount.clientWidth || 300) / 196, 0.1, 30);
    camera.position.set(1.7, 1.55, 1.9);

    scene.add(new THREE.HemisphereLight(0xdbeafe, 0x0f172a, 0.8));
    const dir = new THREE.DirectionalLight(0xffffff, 1.35);
    dir.position.set(2.5, 4.0, 2.2);
    scene.add(dir);

    const grid = new THREE.GridHelper(2.6, 10, 0x1e3a5f, 0x111c33);
    grid.position.y = -0.45;
    scene.add(grid);

    const meshGroup = new THREE.Group();
    scene.add(meshGroup);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.09;
    controls.enablePan = false;
    controls.minDistance = 1.1;
    controls.maxDistance = 5.5;

    rendererRef.current = renderer;
    cameraRef.current = camera;
    controlsRef.current = controls;
    meshGroupRef.current = meshGroup;

    let rafId: number | null = null;
    let damp = 0;
    const renderFrame = () => {
      rafId = null;
      const changed = controls.update();
      renderer.render(scene, camera);
      if (changed || damp > 0) {
        if (damp > 0) damp--;
        rafId = requestAnimationFrame(renderFrame);
      }
    };
    const requestRender = (extra = 2) => {
      damp = Math.max(damp, extra);
      if (rafId === null) rafId = requestAnimationFrame(renderFrame);
    };
    requestRenderRef.current = () => requestRender(3);

    const onChange = () => requestRender(6);
    controls.addEventListener('change', onChange);

    const ro = new ResizeObserver(() => {
      const w = Math.max(180, mount.clientWidth);
      camera.aspect = w / 196;
      camera.updateProjectionMatrix();
      renderer.setSize(w, 196);
      requestRender(2);
    });
    ro.observe(mount);
    requestRender(3);

    return () => {
      ro.disconnect();
      if (rafId !== null) cancelAnimationFrame(rafId);
      controls.removeEventListener('change', onChange);
      controls.dispose();
      renderer.dispose();
      rendererRef.current = null;
    };
  }, []);

  // Rebuild heightfield mesh when specimen/result/shading/zScale changes
  useEffect(() => {
    const group = meshGroupRef.current;
    if (!group) return;
    while (group.children.length > 0) {
      const ch = group.children.pop()!;
      if (ch instanceof THREE.Mesh) {
        ch.geometry.dispose();
        if (Array.isArray(ch.material)) ch.material.forEach((m) => m.dispose());
        else ch.material.dispose();
      }
    }

    const { gw, gh, heights } = fieldStats;
    const aspect = gw / Math.max(1, gh);
    const wWorld = aspect >= 1 ? 2.1 : 2.1 * aspect;
    const hWorld = aspect >= 1 ? 2.1 / aspect : 2.1;

    const geo = new THREE.PlaneGeometry(wWorld, hWorld, gw - 1, gh - 1);
    geo.rotateX(-Math.PI / 2);

    const pos = geo.attributes.position as THREE.BufferAttribute;
    const cols = new Float32Array(pos.count * 3);
    const zAmp = 0.45 * zScale;

    for (let i = 0; i < pos.count; i++) {
      const h = heights[Math.min(heights.length - 1, i)] ?? 0;
      pos.setY(i, (h - 0.3) * zAmp);
      if (shading === 'colormap') {
        const [r, g, b] = elevationRgb(h);
        cols[i * 3] = r;
        cols[i * 3 + 1] = g;
        cols[i * 3 + 2] = b;
      } else {
        const v = 0.2 + h * 0.76;
        cols[i * 3] = v * 0.92;
        cols[i * 3 + 1] = v * 0.98;
        cols[i * 3 + 2] = Math.min(1, v * 1.05);
      }
    }
    geo.setAttribute('color', new THREE.BufferAttribute(cols, 3));
    geo.computeVertexNormals();

    const mat = new THREE.MeshStandardMaterial({
      vertexColors: true,
      roughness: 0.42,
      metalness: shading === 'sem' ? 0.3 : 0.08,
      wireframe: shading === 'wireframe',
      side: THREE.DoubleSide,
    });
    group.add(new THREE.Mesh(geo, mat));
    requestRenderRef.current();
  }, [fieldStats, shading, zScale]);

  const physW = (fieldStats.fovW * specimen.calibration.pixelSize).toFixed(1);
  const physH = (fieldStats.fovH * specimen.calibration.pixelSize).toFixed(1);

  return (
    <div className="rounded-md border border-slate-200 dark:border-slate-800 bg-slate-50/70 dark:bg-[#0A0F1A] overflow-hidden">
      {/* Widget Header */}
      <div className="px-2.5 py-2 border-b border-slate-200 dark:border-slate-800 flex items-center justify-between gap-2">
        <div className="flex items-center gap-1.5 min-w-0">
          <Box className="w-3.5 h-3.5 text-cyan-500 shrink-0" />
          <span className="text-[11.5px] font-semibold text-slate-900 dark:text-slate-100 truncate">
            {lang === 'zh' ? '灰度地形图（可视化，不是高度）' : 'Grey-level relief (view only, not height)'}
          </span>
        </div>

        {onOpen3DStudio && (
          <button
            type="button"
            onClick={onOpen3DStudio}
            className="h-6 px-2 rounded border border-cyan-500/50 bg-cyan-500/12 hover:bg-cyan-500/20 text-blue-700 dark:text-cyan-300 text-[10.5px] font-medium flex items-center gap-1 shrink-0 cursor-pointer"
          >
            <Maximize2 className="w-3 h-3" />
            <span>{lang === 'zh' ? '打开 3D 实验室' : 'Open 3D Lab'}</span>
          </button>
        )}
      </div>

      {/* Interactive Three.js WebGL2 Canvas */}
      <div className="relative bg-[#070B14]">
        <div ref={mountRef} className="w-full h-[196px] cursor-grab active:cursor-grabbing" />

        <div className="pointer-events-none absolute left-2 top-2 px-1.5 py-0.5 rounded bg-[#070B14] border border-slate-800 font-mono text-[9.5px] text-slate-300 tabular-nums">
          {physW}×{physH} {specimen.calibration.spaceUnit}
          {roi ? ' (ROI)' : ''}
        </div>

        <button
          type="button"
          onClick={() => {
            if (cameraRef.current && controlsRef.current) {
              cameraRef.current.position.set(1.7, 1.55, 1.9);
              controlsRef.current.target.set(0, 0, 0);
              controlsRef.current.update();
              requestRenderRef.current();
            }
            setZScale(1.25);
          }}
          className="absolute right-2 top-2 h-6 px-2 rounded bg-[#0A0F1C] hover:bg-slate-800 border border-slate-700 text-slate-300 text-[10px] font-medium flex items-center gap-1 cursor-pointer"
        >
          <RotateCcw className="w-3 h-3" />
          <span>{lang === 'zh' ? '重置' : 'Reset'}</span>
        </button>

        <div className="pointer-events-none absolute left-2 bottom-1.5 text-[9.5px] font-mono text-slate-400">
          {lang === 'zh' ? '灰度起伏（非高度）' : 'Grey-level relief (not height)'} · Sa {fieldStats.rough.sa.toFixed(2)} · Sq {fieldStats.rough.sq.toFixed(2)} · Sz{' '}
          {fieldStats.rough.sz.toFixed(2)}
        </div>
      </div>

      {/* Controls & Legend Strip */}
      <div className="px-2.5 py-2 space-y-2 border-t border-slate-200 dark:border-slate-800 text-[10.5px]">
        <div className="flex items-center justify-between gap-2">
          <div className="flex items-center gap-1">
            {(
              [
                { id: 'sem' as ShadingStyle, zh: 'SEM', en: 'SEM' },
                { id: 'colormap' as ShadingStyle, zh: '高度色谱', en: 'Height' },
                { id: 'wireframe' as ShadingStyle, zh: '网格', en: 'Mesh' },
              ] as const
            ).map((s) => (
              <button
                key={s.id}
                type="button"
                onClick={() => setShading(s.id)}
                className={`h-5 px-1.5 rounded border text-[10px] cursor-pointer transition-colors ${
                  shading === s.id
                    ? 'border-blue-600/60 dark:border-cyan-500/70 bg-blue-600/12 dark:bg-cyan-500/15 text-blue-700 dark:text-cyan-300 font-medium'
                    : 'border-slate-300 dark:border-slate-700 text-slate-500 dark:text-slate-400 hover:text-slate-800 dark:hover:text-slate-200'
                }`}
              >
                {lang === 'zh' ? s.zh : s.en}
              </button>
            ))}
          </div>

          <label className="flex items-center gap-1.5 font-mono text-[10px] text-slate-500 dark:text-slate-400">
            <span>Z {zScale.toFixed(1)}×</span>
            <input
              type="range"
              min={0.4}
              max={2.4}
              step={0.1}
              value={zScale}
              onChange={(e) => setZScale(Number(e.target.value))}
              className="w-16 accent-blue-600 dark:accent-cyan-400 cursor-pointer"
            />
          </label>
        </div>

        <div className="flex items-center justify-between gap-2 text-[10px] text-slate-500 dark:text-slate-400 font-mono">
          <div className="flex items-center gap-2.5">
            <span className="inline-flex items-center gap-1">
              <span className="w-2 h-2 rounded-full bg-emerald-500" />
              <span>
                {lang === 'zh' ? `完整目标 ${fieldStats.completeCount}` : `Upper ${fieldStats.completeCount}`}
              </span>
            </span>
            {fieldStats.occludedCount > 0 && (
              <span className="inline-flex items-center gap-1 text-purple-600 dark:text-purple-300">
                <span className="w-2 h-2 rounded-full bg-purple-400" />
                <span>
                  {lang === 'zh'
                    ? `遮挡 ${fieldStats.occludedCount}${occludedDisplayMode === 'hide' ? ' (隐藏)' : ''}`
                    : `Occluded ${fieldStats.occludedCount}`}
                </span>
              </span>
            )}
          </div>
          <span className="text-[9.5px] text-slate-400">WebGL2 GPU</span>
        </div>
      </div>
    </div>
  );
}
