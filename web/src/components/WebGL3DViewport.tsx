import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/examples/jsm/environments/RoomEnvironment.js';
import {
  AlertTriangle,
  Box,
  ChevronDown,
  ChevronUp,
  Layers,
  RotateCcw,
  SlidersHorizontal,
} from 'lucide-react';
import type {
  Lang,
  OccludedDisplayMode,
  RoiRect,
  RunResult,
  Specimen,
  VolumeAnalysisResult,
  VolumeStackDataset,
} from '../types/vexel';
import { computeSurfaceRoughness } from '../engine/volume3d';
import { probeHardwareAcceleration } from '../engine/webgpuProbe';

export type Viewport3DMode = 'volume3d' | 'surface25d';
export type VolumeRenderStyle = 'mip' | 'alpha' | 'iso';
export type VolumeColormap = 'cyan_bio' | 'gray' | 'viridis' | 'magma';
export type SurfaceShadingMode = 'sem' | 'elevation' | 'wireframe';

interface WebGL3DViewportProps {
  lang: Lang;
  mode: Viewport3DMode;
  onChangeMode?: (mode: Viewport3DMode) => void;
  volume: VolumeStackDataset | null;
  volumeAnalysis: VolumeAnalysisResult | null;
  selectedVolumeObjectId: number | null;
  onSelectVolumeObject: (id: number | null) => void;
  specimen: Specimen;
  runResult: RunResult | null;
  roi: RoiRect | null;
  selected2DObjectId?: number | null;
  occludedDisplayMode?: OccludedDisplayMode;
}

const VOLUME_VERT = /* glsl */ `
out vec3 vLocalPos;
out vec3 vCamLocal;

void main() {
  vLocalPos = position;
  vCamLocal = (inverse(modelMatrix) * vec4(cameraPosition, 1.0)).xyz;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

const VOLUME_FRAG = /* glsl */ `
precision highp float;
precision highp sampler3D;

uniform sampler3D uVolume;
uniform sampler3D uLabelVolume;
uniform int uShowLabelOverlay;
uniform float uLabelOpacity;
uniform vec3 uBoxSize;
uniform int uRenderStyle;   // 0=MIP, 1=Alpha, 2=Iso
uniform int uColormap;      // 0=cyan_bio, 1=gray, 2=viridis, 3=magma
uniform float uWindowMin;
uniform float uWindowMax;
uniform float uIsoThreshold;
uniform float uOpacityGain;
uniform float uSteps;
uniform float uClipZ;       // 0..1 upper Z clip

in vec3 vLocalPos;
in vec3 vCamLocal;
out vec4 fragColor;

float hash12(vec2 p) {
  vec3 p3 = fract(vec3(p.xyx) * 0.1031);
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}

vec2 intersectBox(vec3 orig, vec3 dir, vec3 bMin, vec3 bMax) {
  vec3 invDir = 1.0 / dir;
  vec3 t0s = (bMin - orig) * invDir;
  vec3 t1s = (bMax - orig) * invDir;
  vec3 tsmall = min(t0s, t1s);
  vec3 tbig = max(t0s, t1s);
  float tNear = max(max(tsmall.x, tsmall.y), tsmall.z);
  float tFar = min(min(tbig.x, tbig.y), tbig.z);
  return vec2(tNear, tFar);
}

vec3 applyColormap(float t, int cmap) {
  float v = clamp(t, 0.0, 1.0);
  if (cmap == 1) {
    return vec3(v);
  }
  if (cmap == 2) {
    // Viridis scientific ramp
    vec3 c0 = vec3(0.06, 0.08, 0.18);
    vec3 c1 = vec3(0.13, 0.36, 0.55);
    vec3 c2 = vec3(0.14, 0.68, 0.51);
    vec3 c3 = vec3(0.95, 0.89, 0.24);
    if (v < 0.35) return mix(c0, c1, v / 0.35);
    if (v < 0.7) return mix(c1, c2, (v - 0.35) / 0.35);
    return mix(c2, c3, (v - 0.7) / 0.3);
  }
  if (cmap == 3) {
    // Magma microscopy ramp
    vec3 c0 = vec3(0.03, 0.03, 0.08);
    vec3 c1 = vec3(0.42, 0.12, 0.48);
    vec3 c2 = vec3(0.92, 0.38, 0.28);
    vec3 c3 = vec3(0.99, 0.94, 0.74);
    if (v < 0.35) return mix(c0, c1, v / 0.35);
    if (v < 0.72) return mix(c1, c2, (v - 0.35) / 0.37);
    return mix(c2, c3, (v - 0.72) / 0.28);
  }
  // Default: Cyan-Bio confocal fluorescence ramp
  vec3 b0 = vec3(0.04, 0.08, 0.16);
  vec3 b1 = vec3(0.05, 0.55, 0.75);
  vec3 b2 = vec3(0.12, 0.88, 0.76);
  vec3 b3 = vec3(0.92, 0.99, 1.0);
  if (v < 0.35) return mix(b0, b1, v / 0.35);
  if (v < 0.75) return mix(b1, b2, (v - 0.35) / 0.4);
  return mix(b2, b3, (v - 0.75) / 0.25);
}

float sampleVol(vec3 uvw) {
  return texture(uVolume, uvw).r;
}

float sampleLabel(vec3 uvw) {
  return texture(uLabelVolume, uvw).r * 255.0;
}

vec3 labelTint(float lbl) {
  if (lbl > 1.5) return vec3(0.96, 0.25, 0.37); // 3D nucleus (rose)
  return vec3(0.08, 0.86, 0.72); // 3D cytoplasm (emerald-cyan)
}

void main() {
  vec3 halfBox = uBoxSize * 0.5;
  vec3 rayOrig = vCamLocal;
  vec3 rayDir = normalize(vLocalPos - vCamLocal);

  vec2 hit = intersectBox(rayOrig, rayDir, -halfBox, halfBox);
  if (hit.x > hit.y || hit.y < 0.0) discard;

  float tStart = max(hit.x, 0.0);
  float tEnd = hit.y;
  float steps = max(24.0, uSteps);
  float dt = (tEnd - tStart) / steps;
  float jitter = hash12(gl_FragCoord.xy) * dt;

  float wRange = max(0.02, uWindowMax - uWindowMin);
  float t = tStart + jitter;

  if (uRenderStyle == 0) {
    // Maximum Intensity Projection (MIP)
    float maxVal = 0.0;
    float maxLbl = 0.0;
    for (int i = 0; i < 160; i++) {
      if (float(i) >= steps || t > tEnd) break;
      vec3 p = rayOrig + rayDir * t;
      vec3 uvw = (p + halfBox) / uBoxSize;
      if (uvw.z <= uClipZ) {
        float raw = sampleVol(uvw);
        if (raw > maxVal) maxVal = raw;
        if (uShowLabelOverlay == 1) {
          float lbl = sampleLabel(uvw);
          if (lbl > maxLbl) maxLbl = lbl;
        }
      }
      t += dt;
    }
    float norm = clamp((maxVal - uWindowMin) / wRange, 0.0, 1.0);
    if (norm < 0.015 && maxLbl < 0.5) discard;
    vec3 col = applyColormap(norm, uColormap);
    if (uShowLabelOverlay == 1 && maxLbl > 0.5) {
      col = mix(col, labelTint(maxLbl), uLabelOpacity * 0.78);
    }
    fragColor = vec4(col, clamp(max(norm * 1.15, maxLbl > 0.5 ? 0.5 : 0.0), 0.0, 0.96));
    return;
  }

  if (uRenderStyle == 2) {
    // Raymarched Isosurface with 3D central-difference normal shading
    vec3 lightDir = normalize(vec3(0.5, 0.7, 0.6));
    for (int i = 0; i < 160; i++) {
      if (float(i) >= steps || t > tEnd) break;
      vec3 p = rayOrig + rayDir * t;
      vec3 uvw = (p + halfBox) / uBoxSize;
      if (uvw.z <= uClipZ) {
        float val = sampleVol(uvw);
        if (val >= uIsoThreshold) {
          vec3 eps = vec3(1.0 / 96.0);
          vec3 grad = vec3(
            sampleVol(uvw + vec3(eps.x, 0.0, 0.0)) - sampleVol(uvw - vec3(eps.x, 0.0, 0.0)),
            sampleVol(uvw + vec3(0.0, eps.y, 0.0)) - sampleVol(uvw - vec3(0.0, eps.y, 0.0)),
            sampleVol(uvw + vec3(0.0, 0.0, eps.z)) - sampleVol(uvw - vec3(0.0, 0.0, eps.z))
          );
          vec3 n = normalize(-grad + vec3(1e-5));
          float diff = max(0.18, abs(dot(n, lightDir)));
          vec3 halfV = normalize(lightDir - rayDir);
          float spec = pow(max(0.0, abs(dot(n, halfV))), 28.0) * 0.45;
          float norm = clamp((val - uWindowMin) / wRange, 0.0, 1.0);
          vec3 baseCol = applyColormap(norm, uColormap);
          if (uShowLabelOverlay == 1) {
            float lbl = sampleLabel(uvw);
            if (lbl > 0.5) {
              baseCol = mix(baseCol, labelTint(lbl), uLabelOpacity);
            }
          }
          fragColor = vec4(baseCol * diff + vec3(spec), 0.94);
          return;
        }
      }
      t += dt;
    }
    discard;
  }

  // Front-to-back Alpha Blending (Transfer Function + optional voxel label tint)
  vec3 accCol = vec3(0.0);
  float accAlpha = 0.0;
  float stepAlphaScale = (dt / max(uBoxSize.x, max(uBoxSize.y, uBoxSize.z))) * 14.0 * uOpacityGain;

  for (int i = 0; i < 160; i++) {
    if (float(i) >= steps || t > tEnd || accAlpha > 0.96) break;
    vec3 p = rayOrig + rayDir * t;
    vec3 uvw = (p + halfBox) / uBoxSize;
    if (uvw.z <= uClipZ) {
      float raw = sampleVol(uvw);
      float norm = clamp((raw - uWindowMin) / wRange, 0.0, 1.0);
      float lbl = uShowLabelOverlay == 1 ? sampleLabel(uvw) : 0.0;
      if (norm > 0.04 || lbl > 0.5) {
        float a = clamp(norm * norm * stepAlphaScale, 0.0, 0.85);
        vec3 col = applyColormap(norm, uColormap);
        if (lbl > 0.5) {
          col = mix(col, labelTint(lbl), uLabelOpacity);
          float minLblA = lbl > 1.5 ? stepAlphaScale * 0.42 : stepAlphaScale * 0.14;
          a = max(a, clamp(minLblA, 0.0, 0.72));
        }
        accCol += (1.0 - accAlpha) * col * a;
        accAlpha += (1.0 - accAlpha) * a;
      }
    }
    t += dt;
  }
  if (accAlpha < 0.01) discard;
  fragColor = vec4(accCol, accAlpha);
}
`;

const SLICE_VERT = /* glsl */ `
out vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

const SLICE_FRAG = /* glsl */ `
precision highp float;
precision highp sampler3D;

uniform sampler3D uVolume;
uniform sampler3D uLabelVolume;
uniform int uShowLabelOverlay;
uniform float uLabelOpacity;
uniform int uAxis;       // 0=XY(z), 1=XZ(y), 2=YZ(x)
uniform float uSlicePos; // 0..1
uniform float uWindowMin;
uniform float uWindowMax;
uniform int uColormap;

in vec2 vUv;
out vec4 fragColor;

vec3 mapCol(float v, int cmap) {
  float t = clamp(v, 0.0, 1.0);
  if (cmap == 1) return vec3(t);
  if (cmap == 2) return mix(vec3(0.08, 0.14, 0.30), vec3(0.92, 0.88, 0.25), t);
  if (cmap == 3) return mix(vec3(0.10, 0.05, 0.18), vec3(0.98, 0.82, 0.58), t);
  return mix(vec3(0.04, 0.09, 0.18), vec3(0.28, 0.92, 0.95), t);
}

void main() {
  vec3 uvw;
  if (uAxis == 0) {
    uvw = vec3(vUv.x, vUv.y, uSlicePos);
  } else if (uAxis == 1) {
    uvw = vec3(vUv.x, uSlicePos, vUv.y);
  } else {
    uvw = vec3(uSlicePos, vUv.y, vUv.x);
  }
  float raw = texture(uVolume, uvw).r;
  float wRange = max(0.02, uWindowMax - uWindowMin);
  float norm = clamp((raw - uWindowMin) / wRange, 0.0, 1.0);
  vec3 col = mapCol(norm, uColormap);
  if (uShowLabelOverlay == 1) {
    float lbl = texture(uLabelVolume, uvw).r * 255.0;
    if (lbl > 1.5) {
      col = mix(col, vec3(0.96, 0.25, 0.37), uLabelOpacity * 0.78);
    } else if (lbl > 0.5) {
      col = mix(col, vec3(0.08, 0.86, 0.72), uLabelOpacity * 0.62);
    }
  }
  // Crisp thin cyan/amber border on slice frame
  float edge = min(min(vUv.x, 1.0 - vUv.x), min(vUv.y, 1.0 - vUv.y));
  if (edge < 0.008) {
    col = uAxis == 0 ? vec3(0.13, 0.83, 0.93) : uAxis == 1 ? vec3(0.16, 0.85, 0.56) : vec3(0.96, 0.62, 0.04);
  }
  fragColor = vec4(col, 0.92);
}
`;

function elevationRgb(t: number): [number, number, number] {
  const v = Math.max(0, Math.min(1, t));
  const stops: Array<[number, number, number, number]> = [
    [0.0, 0.05, 0.09, 0.19],
    [0.25, 0.06, 0.45, 0.57],
    [0.52, 0.08, 0.72, 0.51],
    [0.78, 0.95, 0.62, 0.06],
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

export function WebGL3DViewport({
  lang,
  mode,
  onChangeMode,
  volume,
  volumeAnalysis,
  selectedVolumeObjectId,
  onSelectVolumeObject,
  specimen,
  runResult,
  roi,
  selected2DObjectId,
}: WebGL3DViewportProps) {
  const mountRef = useRef<HTMLDivElement>(null);

  // Volume3D visual controls
  const [renderStyle, setRenderStyle] = useState<VolumeRenderStyle>('alpha');
  const [colormap, setColormap] = useState<VolumeColormap>('cyan_bio');
  const [windowMin, setWindowMin] = useState(0.12);
  const [windowMax, setWindowMax] = useState(0.88);
  const [isoThreshold, setIsoThreshold] = useState(0.48);
  const [opacityGain, setOpacityGain] = useState(1.25);
  const [clipZ, setClipZ] = useState(1.0);
  const [showOrthoSlices, setShowOrthoSlices] = useState(true);
  const [sliceX, setSliceX] = useState(0.5);
  const [sliceY, setSliceY] = useState(0.5);
  const [sliceZ, setSliceZ] = useState(0.5);
  const [showSegMeshes, setShowSegMeshes] = useState(true);
  const [showLabelOverlay, setShowLabelOverlay] = useState(true);

  // Surface 2.5D controls
  const [surfaceShading, setSurfaceShading] = useState<SurfaceShadingMode>('sem');
  const [zScale, setZScale] = useState(1.2);
  const [surfaceClipX, setSurfaceClipX] = useState(1.0);
  const [surfaceClipY, setSurfaceClipY] = useState(1.0);
  const [show2DSegOverlay, setShow2DSegOverlay] = useState(true);
  const [surfaceResolution, setSurfaceResolution] = useState<156 | 256 | 384>(156);

  // Dock & WebGL status
  const [dockOpen, setDockOpen] = useState(true);
  const [contextLost, setContextLost] = useState(false);
  const [isSoftwareGl, setIsSoftwareGl] = useState(false);
  const [webgpuReady, setWebgpuReady] = useState(false);

  useEffect(() => {
    let alive = true;
    void probeHardwareAcceleration().then((p) => {
      if (alive) setWebgpuReady(p.webgpuSupported);
    });
    return () => {
      alive = false;
    };
  }, []);

  // Refs to mutable Three.js objects
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);
  const requestRenderRef = useRef<() => void>(() => {});
  const isInteractingRef = useRef(false);

  const volumeGroupRef = useRef<THREE.Group | null>(null);
  const surfaceGroupRef = useRef<THREE.Group | null>(null);
  const segMeshesGroupRef = useRef<THREE.Group | null>(null);
  const volumeMatRef = useRef<THREE.ShaderMaterial | null>(null);
  const sliceMatsRef = useRef<THREE.ShaderMaterial[]>([]);
  const sliceMeshesRef = useRef<THREE.Mesh[]>([]);
  const volumeTexRef = useRef<THREE.Data3DTexture | null>(null);
  const labelTexRef = useRef<THREE.Data3DTexture | null>(null);
  const dummyLabelTexRef = useRef<THREE.Data3DTexture | null>(null);
  const boxSizeRef = useRef<THREE.Vector3>(new THREE.Vector3(2.2, 2.2, 1.4));
  const surfaceClipPlaneRef = useRef<THREE.Plane>(new THREE.Plane(new THREE.Vector3(-1, 0, 0), 2));
  const surfaceClipPlaneYRef = useRef<THREE.Plane>(new THREE.Plane(new THREE.Vector3(0, 0, -1), 2));
  const surfaceWorldDimsRef = useRef<{ w: number; h: number }>({ w: 2.4, h: 2.4 });

  // Surface roughness calculation for 2.5D mode
  const surfaceRoughness = useMemo(() => {
    const fullW = specimen.width;
    const fullH = specimen.calibration.cropRows ?? specimen.height;
    const rx = roi ? Math.max(0, Math.min(fullW - 1, Math.round(roi.x))) : 0;
    const ry = roi ? Math.max(0, Math.min(fullH - 1, Math.round(roi.y))) : 0;
    const fovW = roi ? Math.max(16, Math.min(fullW - rx, Math.round(roi.width))) : fullW;
    const fovH = roi ? Math.max(16, Math.min(fullH - ry, Math.round(roi.height))) : fullH;
    const srcGray = runResult?.processedGray ?? specimen.gray;
    const gw = 96;
    const gh = Math.max(32, Math.round((gw * fovH) / Math.max(1, fovW)));
    const vals = new Float32Array(gw * gh);
    let minV = Infinity;
    let maxV = -Infinity;
    for (let y = 0; y < gh; y++) {
      const sy = Math.min(fullH - 1, ry + Math.floor(((y + 0.5) / gh) * fovH));
      for (let x = 0; x < gw; x++) {
        const sx = Math.min(fullW - 1, rx + Math.floor(((x + 0.5) / gw) * fovW));
        const v = srcGray[sy * fullW + sx] ?? 0;
        vals[y * gw + x] = v;
        if (v < minV) minV = v;
        if (v > maxV) maxV = v;
      }
    }
    const span = Math.max(1, maxV - minV);
    for (let i = 0; i < vals.length; i++) {
      vals[i] = (vals[i] - minV) / span;
    }
    return computeSurfaceRoughness(vals, zScale * 10);
  }, [specimen, runResult, roi, zScale]);

  const resetCamera = useCallback(() => {
    const cam = cameraRef.current;
    const ctrl = controlsRef.current;
    if (!cam || !ctrl) return;
    if (mode === 'surface25d') {
      cam.position.set(1.9, 1.85, 2.2);
    } else {
      cam.position.set(2.3, 1.7, 2.4);
    }
    ctrl.target.set(0, 0, 0);
    ctrl.update();
    requestRenderRef.current();
  }, [mode]);

  // Initialize Three.js WebGL2 Scene ONCE
  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;

    const renderer = new THREE.WebGLRenderer({
      antialias: true,
      powerPreference: 'high-performance',
      alpha: false,
    });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setSize(mount.clientWidth || 640, mount.clientHeight || 480);
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.05;
    renderer.localClippingEnabled = true;
    renderer.setClearColor(0x060911, 1);

    // Detect software GL (SwiftShader / llvmpipe) to keep rendering fluid
    const gl = renderer.getContext();
    const dbg = gl.getExtension('WEBGL_debug_renderer_info');
    const rendererStr = dbg ? String(gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL) ?? '') : '';
    const softGl = /swiftshader|llvmpipe|software|microsoft basic/i.test(rendererStr);
    setIsSoftwareGl(softGl);

    mount.innerHTML = '';
    mount.appendChild(renderer.domElement);
    renderer.domElement.style.display = 'block';
    renderer.domElement.style.width = '100%';
    renderer.domElement.style.height = '100%';

    const scene = new THREE.Scene();
    const pmrem = new THREE.PMREMGenerator(renderer);
    const envTex = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
    scene.environment = envTex;

    const camera = new THREE.PerspectiveCamera(
      42,
      Math.max(1, mount.clientWidth) / Math.max(1, mount.clientHeight),
      0.05,
      50
    );
    camera.position.set(2.3, 1.7, 2.4);

    // Subtle scientific studio lighting
    const hemi = new THREE.HemisphereLight(0xdbeafe, 0x0f172a, 0.75);
    scene.add(hemi);
    const keyLight = new THREE.DirectionalLight(0xffffff, 1.45);
    keyLight.position.set(3.2, 4.5, 2.8);
    scene.add(keyLight);
    const rimLight = new THREE.DirectionalLight(0x38bdf8, 0.55);
    rimLight.position.set(-3.0, 2.0, -2.5);
    scene.add(rimLight);

    // Reference coordinate floor grid
    const grid = new THREE.GridHelper(3.4, 14, 0x1e3a5f, 0x111c33);
    grid.position.y = -0.92;
    scene.add(grid);

    const volumeGroup = new THREE.Group();
    const surfaceGroup = new THREE.Group();
    const segMeshesGroup = new THREE.Group();
    volumeGroup.add(segMeshesGroup);
    scene.add(volumeGroup);
    scene.add(surfaceGroup);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.minDistance = 0.8;
    controls.maxDistance = 9.0;

    rendererRef.current = renderer;
    sceneRef.current = scene;
    cameraRef.current = camera;
    controlsRef.current = controls;
    volumeGroupRef.current = volumeGroup;
    surfaceGroupRef.current = surfaceGroup;
    segMeshesGroupRef.current = segMeshesGroup;

    // On-demand render loop (zero idle GPU usage)
    let rafId: number | null = null;
    let needsRender = true;
    let dampingFrames = 0;

    const renderFrame = () => {
      rafId = null;
      const changed = controls.update();
      renderer.render(scene, camera);
      if (changed || dampingFrames > 0) {
        if (dampingFrames > 0) dampingFrames--;
        rafId = requestAnimationFrame(renderFrame);
      } else {
        needsRender = false;
      }
    };

    const requestRender = (extraDamp = 0) => {
      dampingFrames = Math.max(dampingFrames, extraDamp);
      if (rafId === null) {
        needsRender = true;
        rafId = requestAnimationFrame(renderFrame);
      }
    };
    requestRenderRef.current = () => requestRender(2);

    const onControlStart = () => {
      isInteractingRef.current = true;
      if (volumeMatRef.current) {
        volumeMatRef.current.uniforms.uSteps.value = softGl ? 40 : 64;
      }
      requestRender(4);
    };
    const onControlChange = () => {
      requestRender(8);
    };
    const onControlEnd = () => {
      isInteractingRef.current = false;
      if (volumeMatRef.current) {
        volumeMatRef.current.uniforms.uSteps.value = softGl ? 64 : 112;
      }
      requestRender(12);
    };

    controls.addEventListener('start', onControlStart);
    controls.addEventListener('change', onControlChange);
    controls.addEventListener('end', onControlEnd);

    // WebGL Context Loss & Restore Handlers
    const onContextLost = (e: Event) => {
      e.preventDefault();
      setContextLost(true);
    };
    const onContextRestored = () => {
      setContextLost(false);
      requestRender(4);
    };
    renderer.domElement.addEventListener('webglcontextlost', onContextLost, false);
    renderer.domElement.addEventListener('webglcontextrestored', onContextRestored, false);

    // Click-to-select 3D segmented cells via Raycaster
    let pointerDownPos = { x: 0, y: 0 };
    const raycaster = new THREE.Raycaster();
    const ndc = new THREE.Vector2();

    const onPointerDown = (e: PointerEvent) => {
      pointerDownPos = { x: e.clientX, y: e.clientY };
    };
    const onPointerUp = (e: PointerEvent) => {
      const dist = Math.hypot(e.clientX - pointerDownPos.x, e.clientY - pointerDownPos.y);
      if (dist > 5) return; // Ignore orbit drags
      if (!segMeshesGroupRef.current || !segMeshesGroupRef.current.visible) return;
      const rect = renderer.domElement.getBoundingClientRect();
      ndc.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
      ndc.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
      raycaster.setFromCamera(ndc, camera);
      const hits = raycaster.intersectObjects(segMeshesGroupRef.current.children, true);
      if (hits.length > 0) {
        const hitObj = hits[0].object;
        const oid = hitObj.userData?.objectId;
        if (typeof oid === 'number') {
          onSelectVolumeObject(oid);
        }
      }
    };
    renderer.domElement.addEventListener('pointerdown', onPointerDown);
    renderer.domElement.addEventListener('pointerup', onPointerUp);

    // Responsive ResizeObserver
    const ro = new ResizeObserver(() => {
      const w = Math.max(160, mount.clientWidth);
      const h = Math.max(160, mount.clientHeight);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
      requestRender(2);
    });
    ro.observe(mount);

    requestRender(4);

    return () => {
      ro.disconnect();
      if (rafId !== null) cancelAnimationFrame(rafId);
      controls.removeEventListener('start', onControlStart);
      controls.removeEventListener('change', onControlChange);
      controls.removeEventListener('end', onControlEnd);
      controls.dispose();
      renderer.domElement.removeEventListener('webglcontextlost', onContextLost);
      renderer.domElement.removeEventListener('webglcontextrestored', onContextRestored);
      renderer.domElement.removeEventListener('pointerdown', onPointerDown);
      renderer.domElement.removeEventListener('pointerup', onPointerUp);
      if (volumeTexRef.current) {
        volumeTexRef.current.dispose();
        volumeTexRef.current = null;
      }
      if (labelTexRef.current) {
        labelTexRef.current.dispose();
        labelTexRef.current = null;
      }
      if (dummyLabelTexRef.current) {
        dummyLabelTexRef.current.dispose();
        dummyLabelTexRef.current = null;
      }
      envTex.dispose();
      pmrem.dispose();
      renderer.dispose();
      rendererRef.current = null;
      sceneRef.current = null;
      if (needsRender) needsRender = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Toggle visibility between Volume3D and Surface25D groups
  useEffect(() => {
    if (volumeGroupRef.current) volumeGroupRef.current.visible = mode === 'volume3d';
    if (surfaceGroupRef.current) surfaceGroupRef.current.visible = mode === 'surface25d';
    requestRenderRef.current();
  }, [mode]);

  // Build / Update Volume3D Data3DTexture + Raymarching Cube + Tri-Planar Ortho-Slices
  useEffect(() => {
    const group = volumeGroupRef.current;
    if (!group || !volume) return;

    // Preserve segMeshesGroup while clearing volume & slice meshes
    const segGroup = segMeshesGroupRef.current;
    for (let i = group.children.length - 1; i >= 0; i--) {
      const child = group.children[i];
      if (child === segGroup) continue;
      group.remove(child);
      if (child instanceof THREE.Mesh || child instanceof THREE.LineSegments) {
        child.geometry.dispose();
        if (Array.isArray(child.material)) child.material.forEach((m) => m.dispose());
        else child.material.dispose();
      }
    }
    if (volumeTexRef.current) {
      volumeTexRef.current.dispose();
      volumeTexRef.current = null;
    }

    const srcW = volume.displayLod ? volume.displayLod.width : volume.width;
    const srcH = volume.displayLod ? volume.displayLod.height : volume.height;
    const srcD = volume.displayLod ? volume.displayLod.depth : volume.depth;
    const srcData = volume.displayLod ? volume.displayLod.data : volume.data;

    const tex = new THREE.Data3DTexture(srcData, srcW, srcH, srcD);
    tex.format = THREE.RedFormat;
    tex.type = THREE.UnsignedByteType;
    tex.unpackAlignment = 1;
    tex.minFilter = THREE.LinearFilter;
    tex.magFilter = THREE.LinearFilter;
    tex.generateMipmaps = false;
    tex.needsUpdate = true;
    volumeTexRef.current = tex;

    // Anisotropic physical proportions normalized so largest side = 2.2 world units
    const physX = volume.width * volume.voxelSizeXY;
    const physY = volume.height * volume.voxelSizeXY;
    const physZ = volume.depth * volume.voxelSizeZ;
    const maxPhys = Math.max(physX, physY, physZ, 1e-3);
    const bx = (physX / maxPhys) * 2.2;
    const by = (physY / maxPhys) * 2.2;
    const bz = Math.max(0.45, (physZ / maxPhys) * 2.2);
    boxSizeRef.current.set(bx, by, bz);

    // Subtle wireframe bounding box
    const boxGeo = new THREE.BoxGeometry(bx, by, bz);
    const edges = new THREE.EdgesGeometry(boxGeo);
    const frameLines = new THREE.LineSegments(
      edges,
      new THREE.LineBasicMaterial({ color: 0x334155, transparent: true, opacity: 0.7 })
    );
    group.add(frameLines);

    if (!dummyLabelTexRef.current) {
      const dummy = new THREE.Data3DTexture(new Uint8Array(1), 1, 1, 1);
      dummy.format = THREE.RedFormat;
      dummy.type = THREE.UnsignedByteType;
      dummy.unpackAlignment = 1;
      dummy.minFilter = THREE.NearestFilter;
      dummy.magFilter = THREE.NearestFilter;
      dummy.generateMipmaps = false;
      dummy.needsUpdate = true;
      dummyLabelTexRef.current = dummy;
    }
    const activeLabelTex = labelTexRef.current ?? dummyLabelTexRef.current;

    // Raymarching shader cube (BackSide so camera can inspect closely without clipping out)
    const volMat = new THREE.ShaderMaterial({
      glslVersion: THREE.GLSL3,
      vertexShader: VOLUME_VERT,
      fragmentShader: VOLUME_FRAG,
      transparent: true,
      depthWrite: false,
      side: THREE.BackSide,
      uniforms: {
        uVolume: { value: tex },
        uLabelVolume: { value: activeLabelTex },
        uShowLabelOverlay: { value: showLabelOverlay && labelTexRef.current ? 1 : 0 },
        uLabelOpacity: { value: 0.62 },
        uBoxSize: { value: new THREE.Vector3(bx, by, bz) },
        uRenderStyle: { value: renderStyle === 'mip' ? 0 : renderStyle === 'iso' ? 2 : 1 },
        uColormap: {
          value: colormap === 'gray' ? 1 : colormap === 'viridis' ? 2 : colormap === 'magma' ? 3 : 0,
        },
        uWindowMin: { value: windowMin },
        uWindowMax: { value: windowMax },
        uIsoThreshold: { value: isoThreshold },
        uOpacityGain: { value: opacityGain },
        uSteps: { value: isSoftwareGl ? 64 : 112 },
        uClipZ: { value: clipZ },
      },
    });
    const volMesh = new THREE.Mesh(boxGeo, volMat);
    volMesh.renderOrder = 2;
    group.add(volMesh);
    volumeMatRef.current = volMat;

    // Tri-Planar Ortho-Slicer (XY, XZ, YZ planes sampling the same Data3DTexture + uLabelVolume)
    const makeSliceMat = (axis: number, initPos: number) =>
      new THREE.ShaderMaterial({
        glslVersion: THREE.GLSL3,
        vertexShader: SLICE_VERT,
        fragmentShader: SLICE_FRAG,
        transparent: true,
        side: THREE.DoubleSide,
        uniforms: {
          uVolume: { value: tex },
          uLabelVolume: { value: activeLabelTex },
          uShowLabelOverlay: { value: showLabelOverlay && labelTexRef.current ? 1 : 0 },
          uLabelOpacity: { value: 0.62 },
          uAxis: { value: axis },
          uSlicePos: { value: initPos },
          uWindowMin: { value: windowMin },
          uWindowMax: { value: windowMax },
          uColormap: {
            value:
              colormap === 'gray' ? 1 : colormap === 'viridis' ? 2 : colormap === 'magma' ? 3 : 0,
          },
        },
      });

    const xyMat = makeSliceMat(0, sliceZ);
    const xyMesh = new THREE.Mesh(new THREE.PlaneGeometry(bx, by), xyMat);
    xyMesh.position.z = (sliceZ - 0.5) * bz;
    xyMesh.visible = showOrthoSlices;

    const xzMat = makeSliceMat(1, sliceY);
    const xzMesh = new THREE.Mesh(new THREE.PlaneGeometry(bx, bz), xzMat);
    xzMesh.rotation.x = Math.PI / 2;
    xzMesh.position.y = (sliceY - 0.5) * by;
    xzMesh.visible = showOrthoSlices;

    const yzMat = makeSliceMat(2, sliceX);
    const yzMesh = new THREE.Mesh(new THREE.PlaneGeometry(bz, by), yzMat);
    yzMesh.rotation.y = Math.PI / 2;
    yzMesh.position.x = (sliceX - 0.5) * bx;
    yzMesh.visible = showOrthoSlices;

    group.add(xyMesh, xzMesh, yzMesh);
    sliceMatsRef.current = [xyMat, xzMat, yzMat];
    sliceMeshesRef.current = [xyMesh, xzMesh, yzMesh];

    requestRenderRef.current();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [volume, isSoftwareGl]);

  // Upload real 3D segmentation labelVolume (0=bg, 1=cytoplasm, 2=nucleus) as WebGL2 Data3DTexture
  useEffect(() => {
    if (labelTexRef.current) {
      labelTexRef.current.dispose();
      labelTexRef.current = null;
    }

    let nextTex: THREE.Data3DTexture | null = null;
    if (
      volume &&
      volumeAnalysis?.labelVolume &&
      volumeAnalysis.labelVolume.length === volume.width * volume.height * volume.depth
    ) {
      const targetW = volume.displayLod ? volume.displayLod.width : volume.width;
      const targetH = volume.displayLod ? volume.displayLod.height : volume.height;
      const targetD = volume.displayLod ? volume.displayLod.depth : volume.depth;
      let texBytes = volumeAnalysis.labelVolume;
      if (targetW !== volume.width || targetH !== volume.height || targetD !== volume.depth) {
        const down = new Uint8Array(targetW * targetH * targetD);
        const srcPlane = volume.width * volume.height;
        const dstPlane = targetW * targetH;
        for (let z = 0; z < targetD; z++) {
          const sz = Math.min(volume.depth - 1, Math.floor((z / targetD) * volume.depth));
          for (let y = 0; y < targetH; y++) {
            const sy = Math.min(volume.height - 1, Math.floor((y / targetH) * volume.height));
            for (let x = 0; x < targetW; x++) {
              const sx = Math.min(volume.width - 1, Math.floor((x / targetW) * volume.width));
              down[z * dstPlane + y * targetW + x] =
                volumeAnalysis.labelVolume[sz * srcPlane + sy * volume.width + sx];
            }
          }
        }
        texBytes = down;
      }

      const lTex = new THREE.Data3DTexture(texBytes, targetW, targetH, targetD);
      lTex.format = THREE.RedFormat;
      lTex.type = THREE.UnsignedByteType;
      lTex.unpackAlignment = 1;
      lTex.minFilter = THREE.NearestFilter;
      lTex.magFilter = THREE.NearestFilter;
      lTex.generateMipmaps = false;
      lTex.needsUpdate = true;
      labelTexRef.current = lTex;
      nextTex = lTex;
    }

    const boundTex = nextTex ?? dummyLabelTexRef.current;
    const enabled = nextTex && showLabelOverlay ? 1 : 0;
    if (volumeMatRef.current && boundTex) {
      volumeMatRef.current.uniforms.uLabelVolume.value = boundTex;
      volumeMatRef.current.uniforms.uShowLabelOverlay.value = enabled;
    }
    for (const sm of sliceMatsRef.current) {
      if (boundTex) sm.uniforms.uLabelVolume.value = boundTex;
      sm.uniforms.uShowLabelOverlay.value = enabled;
    }
    requestRenderRef.current();
  }, [volume, volumeAnalysis, showLabelOverlay]);

  // Fast uniform updates for raymarching & ortho-slicer (no geometry rebuild!)
  useEffect(() => {
    const volMat = volumeMatRef.current;
    const cmapIdx =
      colormap === 'gray' ? 1 : colormap === 'viridis' ? 2 : colormap === 'magma' ? 3 : 0;
    const labelOn = labelTexRef.current && showLabelOverlay ? 1 : 0;
    if (volMat) {
      volMat.uniforms.uRenderStyle.value =
        renderStyle === 'mip' ? 0 : renderStyle === 'iso' ? 2 : 1;
      volMat.uniforms.uColormap.value = cmapIdx;
      volMat.uniforms.uWindowMin.value = windowMin;
      volMat.uniforms.uWindowMax.value = windowMax;
      volMat.uniforms.uIsoThreshold.value = isoThreshold;
      volMat.uniforms.uOpacityGain.value = opacityGain;
      volMat.uniforms.uClipZ.value = clipZ;
      volMat.uniforms.uShowLabelOverlay.value = labelOn;
    }
    const [bx, by, bz] = [boxSizeRef.current.x, boxSizeRef.current.y, boxSizeRef.current.z];
    const [xyMat, xzMat, yzMat] = sliceMatsRef.current;
    const [xyMesh, xzMesh, yzMesh] = sliceMeshesRef.current;
    if (xyMat && xyMesh) {
      xyMat.uniforms.uSlicePos.value = sliceZ;
      xyMat.uniforms.uWindowMin.value = windowMin;
      xyMat.uniforms.uWindowMax.value = windowMax;
      xyMat.uniforms.uColormap.value = cmapIdx;
      xyMat.uniforms.uShowLabelOverlay.value = labelOn;
      xyMesh.position.z = (sliceZ - 0.5) * bz;
      xyMesh.visible = showOrthoSlices;
    }
    if (xzMat && xzMesh) {
      xzMat.uniforms.uSlicePos.value = sliceY;
      xzMat.uniforms.uWindowMin.value = windowMin;
      xzMat.uniforms.uWindowMax.value = windowMax;
      xzMat.uniforms.uColormap.value = cmapIdx;
      xzMat.uniforms.uShowLabelOverlay.value = labelOn;
      xzMesh.position.y = (sliceY - 0.5) * by;
      xzMesh.visible = showOrthoSlices;
    }
    if (yzMat && yzMesh) {
      yzMat.uniforms.uSlicePos.value = sliceX;
      yzMat.uniforms.uWindowMin.value = windowMin;
      yzMat.uniforms.uWindowMax.value = windowMax;
      yzMat.uniforms.uColormap.value = cmapIdx;
      yzMat.uniforms.uShowLabelOverlay.value = labelOn;
      yzMesh.position.x = (sliceX - 0.5) * bx;
      yzMesh.visible = showOrthoSlices;
    }
    requestRenderRef.current();
  }, [
    renderStyle,
    colormap,
    windowMin,
    windowMax,
    isoThreshold,
    opacityGain,
    clipZ,
    showOrthoSlices,
    showLabelOverlay,
    sliceX,
    sliceY,
    sliceZ,
  ]);

  // Build / Highlight 3D Segmented Parent Cell Shells + Nested Nuclei Meshes
  useEffect(() => {
    const segGroup = segMeshesGroupRef.current;
    if (!segGroup) return;
    while (segGroup.children.length > 0) {
      const ch = segGroup.children.pop()!;
      if (ch instanceof THREE.Mesh) {
        ch.geometry.dispose();
        if (Array.isArray(ch.material)) ch.material.forEach((m) => m.dispose());
        else ch.material.dispose();
      }
    }
    segGroup.visible = showSegMeshes;
    if (!volume || !volumeAnalysis || !showSegMeshes) {
      requestRenderRef.current();
      return;
    }

    const bx = boxSizeRef.current.x;
    const by = boxSizeRef.current.y;
    const bz = boxSizeRef.current.z;
    const sphereGeo = new THREE.SphereGeometry(0.5, 22, 16);

    for (const obj of volumeAnalysis.objects) {
      const cx = (obj.centroid[0] / Math.max(1, volume.width) - 0.5) * bx;
      const cy = (obj.centroid[1] / Math.max(1, volume.height) - 0.5) * by;
      const cz = (obj.centroid[2] / Math.max(1, volume.depth) - 0.5) * bz;

      const sx = Math.max(
        0.04,
        ((obj.bboxVoxel.x1 - obj.bboxVoxel.x0 + 1) / Math.max(1, volume.width)) * bx
      );
      const sy = Math.max(
        0.04,
        ((obj.bboxVoxel.y1 - obj.bboxVoxel.y0 + 1) / Math.max(1, volume.height)) * by
      );
      const sz = Math.max(
        0.04,
        ((obj.bboxVoxel.z1 - obj.bboxVoxel.z0 + 1) / Math.max(1, volume.depth)) * bz
      );

      const isSelected =
        selectedVolumeObjectId !== null &&
        (obj.id === selectedVolumeObjectId || obj.parentId === selectedVolumeObjectId);

      const isNucleus = obj.kind === 'nucleus_3d';
      const mat = new THREE.MeshStandardMaterial({
        color: isSelected
          ? 0xf59e0b
          : isNucleus
          ? 0xf43f5e
          : 0x22d3ee,
        emissive: isSelected ? 0xb45309 : isNucleus ? 0x4c0519 : 0x083344,
        emissiveIntensity: isSelected ? 0.55 : 0.25,
        roughness: isNucleus ? 0.35 : 0.25,
        metalness: 0.1,
        transparent: !isNucleus,
        opacity: isNucleus ? 0.92 : isSelected ? 0.42 : 0.24,
        depthWrite: isNucleus,
      });

      const mesh = new THREE.Mesh(sphereGeo.clone(), mat);
      mesh.position.set(cx, cy, cz);
      mesh.scale.set(sx, sy, sz);
      mesh.userData = { objectId: obj.parentId ?? obj.id };
      mesh.renderOrder = isNucleus ? 4 : 3;
      segGroup.add(mesh);
    }
    sphereGeo.dispose();
    requestRenderRef.current();
  }, [volume, volumeAnalysis, selectedVolumeObjectId, showSegMeshes]);

  // Build High-Resolution 2.5D Displaced Surface Mesh from active 2D Specimen + optional 2D Segmentation Overlay
  useEffect(() => {
    const sGroup = surfaceGroupRef.current;
    if (!sGroup) return;
    while (sGroup.children.length > 0) {
      const ch = sGroup.children.pop()!;
      if (ch instanceof THREE.Mesh) {
        ch.geometry.dispose();
        if (Array.isArray(ch.material)) ch.material.forEach((m) => m.dispose());
        else ch.material.dispose();
      }
    }

    const fullW = specimen.width;
    const fullH = specimen.calibration.cropRows ?? specimen.height;
    const rx = roi ? Math.max(0, Math.min(fullW - 1, Math.round(roi.x))) : 0;
    const ry = roi ? Math.max(0, Math.min(fullH - 1, Math.round(roi.y))) : 0;
    const fovW = roi ? Math.max(16, Math.min(fullW - rx, Math.round(roi.width))) : fullW;
    const fovH = roi ? Math.max(16, Math.min(fullH - ry, Math.round(roi.height))) : fullH;

    const segX = isSoftwareGl ? Math.min(156, surfaceResolution) : surfaceResolution;
    const aspect = fovW / Math.max(1, fovH);
    const segY = Math.max(40, Math.min(segX, Math.round(segX / Math.max(0.4, aspect))));
    const worldW = aspect >= 1 ? 2.4 : 2.4 * aspect;
    const worldH = aspect >= 1 ? 2.4 / aspect : 2.4;
    surfaceWorldDimsRef.current = { w: worldW, h: worldH };

    const geo = new THREE.PlaneGeometry(worldW, worldH, segX, segY);
    geo.rotateX(-Math.PI / 2);

    // Optional Canvas2D rasterization of 2D segmentation shapes onto the heightfield vertex grid
    let segOverlayPixels: Uint8ClampedArray | null = null;
    if (show2DSegOverlay && runResult && runResult.shapes.length > 0 && typeof document !== 'undefined') {
      const cw = segX + 1;
      const ch = segY + 1;
      const ovCanvas = document.createElement('canvas');
      ovCanvas.width = cw;
      ovCanvas.height = ch;
      const ctx = ovCanvas.getContext('2d');
      if (ctx) {
        ctx.clearRect(0, 0, cw, ch);
        const sxScale = cw / Math.max(1, fovW);
        const syScale = ch / Math.max(1, fovH);
        for (const shape of runResult.shapes) {
          if (shape.layer !== 'segmentation' && shape.layer !== 'edges') continue;
          const isSel =
            selected2DObjectId !== undefined &&
            selected2DObjectId !== null &&
            shape.objectId === selected2DObjectId;
          const fillStyle = isSel
            ? 'rgba(245, 158, 11, 0.88)'
            : shape.fill || 'rgba(34, 211, 238, 0.42)';
          const strokeStyle = isSel ? '#fde047' : shape.color || '#22d3ee';
          ctx.fillStyle = fillStyle;
          ctx.strokeStyle = strokeStyle;
          ctx.lineWidth = isSel ? 2.2 : 1.1;

          if (shape.kind === 'polygon' && shape.points.length >= 3) {
            ctx.beginPath();
            shape.points.forEach(([px, py], idx) => {
              const cx = (px - rx) * sxScale;
              const cy = (py - ry) * syScale;
              if (idx === 0) ctx.moveTo(cx, cy);
              else ctx.lineTo(cx, cy);
            });
            ctx.closePath();
            if (shape.holes) {
              for (const hole of shape.holes) {
                if (hole.length < 3) continue;
                hole.forEach(([hx, hy], idx) => {
                  const cx = (hx - rx) * sxScale;
                  const cy = (hy - ry) * syScale;
                  if (idx === 0) ctx.moveTo(cx, cy);
                  else ctx.lineTo(cx, cy);
                });
                ctx.closePath();
              }
            }
            ctx.fill('evenodd');
            ctx.stroke();
          } else if (shape.kind === 'circle' && shape.points.length >= 2) {
            const [cx0, cy0] = shape.points[0];
            const [radX, radY] = shape.points[1];
            ctx.beginPath();
            ctx.ellipse(
              (cx0 - rx) * sxScale,
              (cy0 - ry) * syScale,
              Math.max(1, Math.abs(radX) * sxScale),
              Math.max(1, Math.abs(radY) * syScale),
              0,
              0,
              Math.PI * 2
            );
            ctx.fill();
            ctx.stroke();
          } else if (shape.kind === 'box' && shape.points.length >= 2) {
            const [x0, y0] = shape.points[0];
            const [x1, y1] = shape.points[1];
            const bx0 = (Math.min(x0, x1) - rx) * sxScale;
            const by0 = (Math.min(y0, y1) - ry) * syScale;
            const bw = Math.abs(x1 - x0) * sxScale;
            const bh = Math.abs(y1 - y0) * syScale;
            ctx.fillRect(bx0, by0, bw, bh);
            ctx.strokeRect(bx0, by0, bw, bh);
          }
        }
        segOverlayPixels = ctx.getImageData(0, 0, cw, ch).data;
      }
    }

    const srcGray = runResult?.processedGray ?? specimen.gray;
    const pos = geo.attributes.position as THREE.BufferAttribute;
    const cols = new Float32Array(pos.count * 3);
    const rawVals = new Float32Array(pos.count);

    let minV = Infinity;
    let maxV = -Infinity;
    for (let i = 0; i < pos.count; i++) {
      const gx = i % (segX + 1);
      const gy = Math.floor(i / (segX + 1));
      const sx = Math.min(fullW - 1, rx + Math.floor((gx / segX) * (fovW - 1)));
      const sy = Math.min(fullH - 1, ry + Math.floor((gy / segY) * (fovH - 1)));
      const v = srcGray[sy * fullW + sx] ?? 0;
      rawVals[i] = v;
      if (v < minV) minV = v;
      if (v > maxV) maxV = v;
    }
    const span = Math.max(12, maxV - minV);
    const zAmp = 0.55 * zScale;

    for (let i = 0; i < pos.count; i++) {
      const hNorm = Math.max(0, Math.min(1, (rawVals[i] - minV) / span));
      pos.setY(i, (hNorm - 0.35) * zAmp);
      let r: number;
      let g: number;
      let b: number;
      if (surfaceShading === 'elevation') {
        [r, g, b] = elevationRgb(hNorm);
      } else {
        const luma = 0.18 + hNorm * 0.78;
        r = luma * 0.94;
        g = luma * 0.98;
        b = Math.min(1, luma * 1.04);
      }
      if (segOverlayPixels) {
        const oi = i * 4;
        const oa = (segOverlayPixels[oi + 3] ?? 0) / 255;
        if (oa > 0.02) {
          const blend = Math.min(0.82, oa * 0.85);
          r = r * (1 - blend) + (segOverlayPixels[oi] / 255) * blend;
          g = g * (1 - blend) + (segOverlayPixels[oi + 1] / 255) * blend;
          b = b * (1 - blend) + (segOverlayPixels[oi + 2] / 255) * blend;
        }
      }
      cols[i * 3] = r;
      cols[i * 3 + 1] = g;
      cols[i * 3 + 2] = b;
    }
    geo.setAttribute('color', new THREE.BufferAttribute(cols, 3));
    geo.computeVertexNormals();

    surfaceClipPlaneRef.current.constant = (surfaceClipX - 0.5) * worldW * 1.02;
    surfaceClipPlaneYRef.current.constant = (surfaceClipY - 0.5) * worldH * 1.02;

    const mat = new THREE.MeshStandardMaterial({
      vertexColors: true,
      roughness: surfaceShading === 'sem' ? 0.38 : 0.45,
      metalness: surfaceShading === 'sem' ? 0.35 : 0.08,
      wireframe: surfaceShading === 'wireframe',
      side: THREE.DoubleSide,
      clippingPlanes: [surfaceClipPlaneRef.current, surfaceClipPlaneYRef.current],
    });

    const mesh = new THREE.Mesh(geo, mat);
    sGroup.add(mesh);
    requestRenderRef.current();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    specimen,
    runResult,
    roi,
    selected2DObjectId,
    surfaceShading,
    zScale,
    show2DSegOverlay,
    surfaceResolution,
    isSoftwareGl,
  ]);

  // Fast X/Y cross-section clip plane updates without rebuilding 2.5D mesh geometry
  useEffect(() => {
    const { w, h } = surfaceWorldDimsRef.current;
    surfaceClipPlaneRef.current.constant = (surfaceClipX - 0.5) * w * 1.02;
    surfaceClipPlaneYRef.current.constant = (surfaceClipY - 0.5) * h * 1.02;
    requestRenderRef.current();
  }, [surfaceClipX, surfaceClipY]);

  return (
    <div className="relative flex-1 min-w-0 min-h-[380px] bg-[#060911] flex flex-col overflow-hidden select-none">
      {/* Top Floating Mode & Telemetry Strip (Opaque, Zero backdrop-filter) */}
      <div className="h-11 px-3 border-b border-slate-800 bg-[#0A0F1C] flex items-center justify-between gap-3 shrink-0 z-10">
        <div className="flex items-center gap-2">
          <div className="flex items-center rounded border border-slate-700 bg-[#060911] p-0.5 text-[11.5px]">
            <button
              type="button"
              onClick={() => onChangeMode?.('volume3d')}
              className={`h-7 px-2.5 rounded flex items-center gap-1.5 font-medium cursor-pointer transition-colors ${
                mode === 'volume3d'
                  ? 'bg-cyan-500 text-slate-950 font-semibold'
                  : 'text-slate-300 hover:text-white'
              }`}
            >
              <Box className="w-3.5 h-3.5" />
              <span>{lang === 'zh' ? '真实 3D Z-Stack 体渲染' : '3D Z-Stack Volume'}</span>
            </button>
            <button
              type="button"
              onClick={() => onChangeMode?.('surface25d')}
              className={`h-7 px-2.5 rounded flex items-center gap-1.5 font-medium cursor-pointer transition-colors ${
                mode === 'surface25d'
                  ? 'bg-cyan-500 text-slate-950 font-semibold'
                  : 'text-slate-300 hover:text-white'
              }`}
            >
              <Layers className="w-3.5 h-3.5" />
              <span>{lang === 'zh' ? '2D 伪 3D 表面重建' : '2.5D Surface Relief'}</span>
            </button>
          </div>
          {isSoftwareGl && (
            <span className="text-[10.5px] font-mono text-amber-400">
              {lang === 'zh' ? '软件 GL 自适应步长' : 'SoftGL Adaptive'}
            </span>
          )}
          <span
            className={`hidden md:inline-block text-[10px] font-mono px-1.5 py-0.5 rounded border ${
              webgpuReady
                ? 'border-emerald-500/40 bg-emerald-500/10 text-emerald-300'
                : 'border-slate-700 bg-slate-900 text-slate-400'
            }`}
          >
            {webgpuReady ? 'WebGL2 · WebGPU Ready' : 'WebGL2'}
          </span>
        </div>

        <div className="flex items-center gap-2 font-mono text-[11px] text-slate-300">
          {mode === 'volume3d' && volume ? (
            <span>
              {volume.width}×{volume.height}×{volume.depth} · XY {volume.voxelSizeXY} / Z{' '}
              {volume.voxelSizeZ} {volume.spaceUnit}
            </span>
          ) : (
            <span title={lang === 'zh' ? '由灰度换算的起伏，仅作可视化，不是物理高度或粗糙度' : 'Relief from grey levels, for viewing only: not a physical height or roughness'}>
              {lang === 'zh' ? '灰度起伏（非高度）' : 'Grey-level relief (not height)'} · Sa {surfaceRoughness.sa.toFixed(2)} · Sq {surfaceRoughness.sq.toFixed(2)} · Sz{' '}
              {surfaceRoughness.sz.toFixed(2)}
            </span>
          )}
          <button
            type="button"
            onClick={resetCamera}
            className="h-7 px-2 rounded border border-slate-700 bg-slate-900 hover:bg-slate-800 text-slate-200 flex items-center gap-1 cursor-pointer"
          >
            <RotateCcw className="w-3 h-3" />
            <span>{lang === 'zh' ? '重置视角' : 'Reset'}</span>
          </button>
        </div>
      </div>

      {/* Main WebGL2 Canvas Mount */}
      <div className="relative flex-1 min-h-0">
        <div ref={mountRef} className="w-full h-full cursor-grab active:cursor-grabbing" />

        {contextLost && (
          <div className="absolute inset-0 bg-[#060911]/95 flex flex-col items-center justify-center gap-2 text-amber-300 p-4 text-center">
            <AlertTriangle className="w-6 h-6" />
            <div className="text-[13px] font-medium">
              {lang === 'zh'
                ? 'WebGL2 渲染上下文已重置，正在自动恢复…'
                : 'WebGL2 context lost — recovering automatically…'}
            </div>
          </div>
        )}

        {/* Subtle Bottom-Left Interaction Hint */}
        <div className="pointer-events-none absolute left-3 bottom-3 px-2 py-1 rounded bg-[#0A0F1C] border border-slate-800 font-mono text-[10px] text-slate-400">
          {mode === 'volume3d'
            ? lang === 'zh'
              ? '左键旋转 · 滚轮缩放 · 右键平移 · 点击 3D 细胞壳体同步定位表格'
              : 'LMB Orbit · Wheel Zoom · RMB Pan · Click 3D cell to inspect'
            : lang === 'zh'
            ? '左键旋转 · 滚轮缩放 · 拖动下方截面滑块观察 2.5D 剖面起伏'
            : 'LMB Orbit · Wheel Zoom · Drag cross-section slider below'}
        </div>
      </div>

      {/* Collapsible Opaque Bottom Control Dock (Zero backdrop-filter over WebGL) */}
      <div className="border-t border-slate-800 bg-[#0A0F1C] text-slate-200 shrink-0">
        <button
          type="button"
          onClick={() => setDockOpen((o) => !o)}
          className="w-full h-8 px-3 flex items-center justify-between text-[11px] font-mono text-slate-300 hover:text-white cursor-pointer"
        >
          <span className="flex items-center gap-1.5">
            <SlidersHorizontal className="w-3.5 h-3.5 text-cyan-400" />
            <span>
              {mode === 'volume3d'
                ? lang === 'zh'
                  ? 'WebGL2 体渲染参数与三平面正交切片控制'
                  : 'WebGL2 Raymarch & Tri-Planar Ortho-Slicer Controls'
                : lang === 'zh'
                ? '2.5D 表面着色与截面剖切控制'
                : '2.5D Surface Shading & Cross-Section Controls'}
            </span>
          </span>
          {dockOpen ? <ChevronDown className="w-3.5 h-3.5" /> : <ChevronUp className="w-3.5 h-3.5" />}
        </button>

        {dockOpen && mode === 'volume3d' && (
          <div className="px-3 pb-3 pt-1 grid grid-cols-2 xl:grid-cols-4 gap-3 text-[11px] font-mono border-t border-slate-800/80">
            {/* Render Mode & Colormap */}
            <div className="space-y-1.5">
              <div className="text-[10px] text-slate-400">
                {lang === 'zh' ? '体渲染模式' : 'Render Mode'}
              </div>
              <div className="grid grid-cols-3 gap-1">
                {(
                  [
                    { id: 'alpha', label: 'Alpha' },
                    { id: 'mip', label: 'MIP' },
                    { id: 'iso', label: 'Iso' },
                  ] as const
                ).map((opt) => (
                  <button
                    key={opt.id}
                    type="button"
                    onClick={() => setRenderStyle(opt.id)}
                    className={`h-6 rounded border text-[10.5px] cursor-pointer ${
                      renderStyle === opt.id
                        ? 'border-cyan-400 bg-cyan-500/20 text-cyan-300 font-semibold'
                        : 'border-slate-700 bg-slate-900 text-slate-400 hover:text-white'
                    }`}
                  >
                    {opt.label}
                  </button>
                ))}
              </div>
              <select
                value={colormap}
                onChange={(e) => setColormap(e.target.value as VolumeColormap)}
                className="w-full h-6 px-1.5 rounded border border-slate-700 bg-slate-900 text-slate-200 text-[10.5px]"
              >
                <option value="cyan_bio">Cyan-Bio (Confocal)</option>
                <option value="gray">Grayscale</option>
                <option value="viridis">Viridis</option>
                <option value="magma">Magma</option>
              </select>
            </div>

            {/* Transfer Function Window/Level */}
            <div className="space-y-1.5">
              <label className="block">
                <div className="flex justify-between text-[10px] text-slate-400">
                  <span>{lang === 'zh' ? '窗宽下限 (Min)' : 'Window Min'}</span>
                  <span>{Math.round(windowMin * 255)}</span>
                </div>
                <input
                  type="range"
                  min={0}
                  max={0.85}
                  step={0.01}
                  value={windowMin}
                  onChange={(e) => setWindowMin(Math.min(windowMax - 0.05, Number(e.target.value)))}
                  className="w-full accent-cyan-400"
                />
              </label>
              <label className="block">
                <div className="flex justify-between text-[10px] text-slate-400">
                  <span>
                    {renderStyle === 'iso'
                      ? lang === 'zh'
                        ? '等值面阈值 (Iso)'
                        : 'Iso Threshold'
                      : lang === 'zh'
                      ? '不透明度增益'
                      : 'Opacity Gain'}
                  </span>
                  <span>
                    {renderStyle === 'iso'
                      ? Math.round(isoThreshold * 255)
                      : `${opacityGain.toFixed(1)}×`}
                  </span>
                </div>
                {renderStyle === 'iso' ? (
                  <input
                    type="range"
                    min={0.1}
                    max={0.95}
                    step={0.01}
                    value={isoThreshold}
                    onChange={(e) => setIsoThreshold(Number(e.target.value))}
                    className="w-full accent-cyan-400"
                  />
                ) : (
                  <input
                    type="range"
                    min={0.3}
                    max={3.0}
                    step={0.1}
                    value={opacityGain}
                    onChange={(e) => setOpacityGain(Number(e.target.value))}
                    className="w-full accent-cyan-400"
                  />
                )}
              </label>
            </div>

            {/* Tri-Planar Ortho Slicer */}
            <div className="space-y-1">
              <div className="flex items-center justify-between text-[10px] text-slate-400">
                <span>{lang === 'zh' ? '三平面正交切片' : 'Ortho-Slices'}</span>
                <label className="flex items-center gap-1 cursor-pointer text-cyan-300">
                  <input
                    type="checkbox"
                    checked={showOrthoSlices}
                    onChange={(e) => setShowOrthoSlices(e.target.checked)}
                    className="accent-cyan-400"
                  />
                  <span>{showOrthoSlices ? 'ON' : 'OFF'}</span>
                </label>
              </div>
              <div className="flex items-center gap-1.5 text-[10px]">
                <span className="w-4 text-cyan-400">XY</span>
                <input
                  type="range"
                  min={0.02}
                  max={0.98}
                  step={0.01}
                  value={sliceZ}
                  onChange={(e) => setSliceZ(Number(e.target.value))}
                  className="flex-1 accent-cyan-400"
                />
              </div>
              <div className="flex items-center gap-1.5 text-[10px]">
                <span className="w-4 text-emerald-400">XZ</span>
                <input
                  type="range"
                  min={0.02}
                  max={0.98}
                  step={0.01}
                  value={sliceY}
                  onChange={(e) => setSliceY(Number(e.target.value))}
                  className="flex-1 accent-emerald-400"
                />
              </div>
              <div className="flex items-center gap-1.5 text-[10px]">
                <span className="w-4 text-amber-400">YZ</span>
                <input
                  type="range"
                  min={0.02}
                  max={0.98}
                  step={0.01}
                  value={sliceX}
                  onChange={(e) => setSliceX(Number(e.target.value))}
                  className="flex-1 accent-amber-400"
                />
              </div>
            </div>

            {/* Z-Clip & 3D Segmentation Overlay */}
            <div className="space-y-1.5">
              <label className="block">
                <div className="flex justify-between text-[10px] text-slate-400">
                  <span>{lang === 'zh' ? 'Z 轴剖切上限' : 'Z-Clip Ceiling'}</span>
                  <span>{Math.round(clipZ * 100)}%</span>
                </div>
                <input
                  type="range"
                  min={0.1}
                  max={1.0}
                  step={0.01}
                  value={clipZ}
                  onChange={(e) => setClipZ(Number(e.target.value))}
                  className="w-full accent-cyan-400"
                />
              </label>
              <label className="flex items-center justify-between pt-1 text-[10.5px] text-slate-300 cursor-pointer">
                <span>{lang === 'zh' ? '3D 体素标签叠加 (胞质/核)' : '3D Voxel Label Overlay'}</span>
                <input
                  type="checkbox"
                  checked={showLabelOverlay}
                  onChange={(e) => setShowLabelOverlay(e.target.checked)}
                  className="accent-cyan-400"
                />
              </label>
              <label className="flex items-center justify-between text-[10.5px] text-slate-300 cursor-pointer">
                <span>{lang === 'zh' ? '显示 3D 分割细胞与核体' : '3D Segmented Shells'}</span>
                <input
                  type="checkbox"
                  checked={showSegMeshes}
                  onChange={(e) => setShowSegMeshes(e.target.checked)}
                  className="accent-cyan-400"
                />
              </label>
            </div>
          </div>
        )}

        {dockOpen && mode === 'surface25d' && (
          <div className="px-3 pb-3 pt-1.5 grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-3 text-[11px] font-mono border-t border-slate-800/80">
            <div className="space-y-1.5">
              <div className="text-[10px] text-slate-400">
                {lang === 'zh' ? '表面材质着色' : 'Surface Shading'}
              </div>
              <div className="grid grid-cols-3 gap-1">
                {(
                  [
                    { id: 'sem', zh: 'SEM 金属', en: 'SEM' },
                    { id: 'elevation', zh: '高度色谱', en: 'Height' },
                    { id: 'wireframe', zh: '拓扑网格', en: 'Mesh' },
                  ] as const
                ).map((s) => (
                  <button
                    key={s.id}
                    type="button"
                    onClick={() => setSurfaceShading(s.id)}
                    className={`h-6 rounded border text-[10.5px] cursor-pointer ${
                      surfaceShading === s.id
                        ? 'border-cyan-400 bg-cyan-500/20 text-cyan-300 font-semibold'
                        : 'border-slate-700 bg-slate-900 text-slate-400 hover:text-white'
                    }`}
                  >
                    {lang === 'zh' ? s.zh : s.en}
                  </button>
                ))}
              </div>
              <label className="flex items-center justify-between text-[10.5px] text-slate-300 cursor-pointer pt-0.5">
                <span>{lang === 'zh' ? '叠加 2D 分割掩膜投影' : 'Project 2D Seg Overlay'}</span>
                <input
                  type="checkbox"
                  checked={show2DSegOverlay}
                  onChange={(e) => setShow2DSegOverlay(e.target.checked)}
                  className="accent-cyan-400"
                />
              </label>
            </div>

            <div className="space-y-1.5">
              <div className="text-[10px] text-slate-400">
                {lang === 'zh' ? '表面网格精细度' : 'Mesh Resolution'}
              </div>
              <div className="grid grid-cols-3 gap-1">
                {([156, 256, 384] as const).map((res) => (
                  <button
                    key={res}
                    type="button"
                    onClick={() => setSurfaceResolution(res)}
                    className={`h-6 rounded border text-[10.5px] cursor-pointer ${
                      surfaceResolution === res
                        ? 'border-cyan-400 bg-cyan-500/20 text-cyan-300 font-semibold'
                        : 'border-slate-700 bg-slate-900 text-slate-400 hover:text-white'
                    }`}
                  >
                    {res}²
                  </button>
                ))}
              </div>
              <label className="block space-y-0.5">
                <div className="flex justify-between text-[10px] text-slate-400">
                  <span>{lang === 'zh' ? 'Z 轴起伏倍率' : 'Z-Exaggeration'}</span>
                  <span>{zScale.toFixed(1)}×</span>
                </div>
                <input
                  type="range"
                  min={0.3}
                  max={2.6}
                  step={0.1}
                  value={zScale}
                  onChange={(e) => setZScale(Number(e.target.value))}
                  className="w-full accent-cyan-400"
                />
              </label>
            </div>

            <label className="space-y-1">
              <div className="flex justify-between text-[10px] text-slate-400">
                <span>{lang === 'zh' ? 'X 轴实时剖切截面' : 'X Cross-Section Clip'}</span>
                <span>{Math.round(surfaceClipX * 100)}%</span>
              </div>
              <input
                type="range"
                min={0.12}
                max={1.0}
                step={0.01}
                value={surfaceClipX}
                onChange={(e) => setSurfaceClipX(Number(e.target.value))}
                className="w-full accent-cyan-400"
              />
            </label>

            <label className="space-y-1">
              <div className="flex justify-between text-[10px] text-slate-400">
                <span>{lang === 'zh' ? 'Y 轴实时剖切截面' : 'Y Cross-Section Clip'}</span>
                <span>{Math.round(surfaceClipY * 100)}%</span>
              </div>
              <input
                type="range"
                min={0.12}
                max={1.0}
                step={0.01}
                value={surfaceClipY}
                onChange={(e) => setSurfaceClipY(Number(e.target.value))}
                className="w-full accent-cyan-400"
              />
            </label>
          </div>
        )}
      </div>
    </div>
  );
}
