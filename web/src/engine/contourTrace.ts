/**
 * Exact Pixel-Edge (Crack-Code) Contour Tracer & Corner-Preserving Light Smoother
 *
 * Traces the exact unit-grid pixel boundaries ([x, x+1] x [y, y+1]) of any 4-connected
 * binary mask component, preserving all concavities, narrow bridges, and internal holes
 * with 0.00 px geometric error.
 *
 * Also provides an optional corner-preserving light Taubin smoother (< 0.6 px max deviation)
 * that eliminates diagonal staircase jaggies while keeping straight horizontal/vertical edges
 * flat and preserving enclosed area and concavities.
 */

export interface TracedContours {
  /** Outer clockwise boundary loop in image coordinates ([x, x+1] x [y, y+1] grid vertices). */
  outer: Array<[number, number]>;
  /** Zero or more counter-clockwise internal hole boundary loops. */
  holes: Array<Array<[number, number]>>;
}

function signedArea2(pts: Array<[number, number]>): number {
  let sum = 0;
  const n = pts.length;
  for (let i = 0; i < n; i++) {
    const [x0, y0] = pts[i];
    const [x1, y1] = pts[(i + 1) % n];
    sum += x0 * y1 - x1 * y0;
  }
  return sum;
}

/** Merge strictly collinear consecutive vertices on a closed polygon loop. */
function mergeCollinearLoop(pts: Array<[number, number]>): Array<[number, number]> {
  const n = pts.length;
  if (n < 3) return pts;
  const out: Array<[number, number]> = [];
  for (let i = 0; i < n; i++) {
    const prev = pts[(i - 1 + n) % n];
    const curr = pts[i];
    const next = pts[(i + 1) % n];
    const dx1 = curr[0] - prev[0];
    const dy1 = curr[1] - prev[1];
    const dx2 = next[0] - curr[0];
    const dy2 = next[1] - curr[1];
    if (dx1 * dy2 !== dy1 * dx2) {
      out.push(curr);
    }
  }
  return out.length >= 3 ? out : pts;
}

/**
 * Core crack-code contour tracer given a pixel membership predicate over a tight bounding box.
 * Every directed unit edge keeps the foreground pixel strictly on its RIGHT side
 * (clockwise outer boundary in screen coordinates where +x is right and +y is down).
 */
function tracePixelEdgeLoops(
  isForeground: (x: number, y: number) => boolean,
  w: number,
  h: number,
  minX: number,
  maxX: number,
  minY: number,
  maxY: number
): TracedContours {
  const x0Bound = Math.max(0, minX);
  const x1Bound = Math.min(w - 1, maxX);
  const y0Bound = Math.max(0, minY);
  const y1Bound = Math.min(h - 1, maxY);

  const stride = w + 1;
  // Adjacency map: start vertex key -> list of end vertex keys (degree <= 2)
  const outEdges = new Map<number, number[]>();

  const addEdge = (x0: number, y0: number, x1: number, y1: number) => {
    const k0 = y0 * stride + x0;
    const k1 = y1 * stride + x1;
    const list = outEdges.get(k0);
    if (list) list.push(k1);
    else outEdges.set(k0, [k1]);
  };

  for (let y = y0Bound; y <= y1Bound; y++) {
    for (let x = x0Bound; x <= x1Bound; x++) {
      if (!isForeground(x, y)) continue;
      // Top edge: (x, y) -> (x+1, y)
      if (y === 0 || !isForeground(x, y - 1)) addEdge(x, y, x + 1, y);
      // Right edge: (x+1, y) -> (x+1, y+1)
      if (x + 1 >= w || !isForeground(x + 1, y)) addEdge(x + 1, y, x + 1, y + 1);
      // Bottom edge: (x+1, y+1) -> (x, y+1)
      if (y + 1 >= h || !isForeground(x, y + 1)) addEdge(x + 1, y + 1, x, y + 1);
      // Left edge: (x, y+1) -> (x, y)
      if (x === 0 || !isForeground(x - 1, y)) addEdge(x, y + 1, x, y);
    }
  }

  const loops: Array<Array<[number, number]>> = [];

  for (const [startKey, startTargets] of outEdges.entries()) {
    while (startTargets.length > 0) {
      const nextKey = startTargets.pop()!;
      const loop: Array<[number, number]> = [];
      let prevKey = startKey;
      let currKey = nextKey;

      const sx = startKey % stride;
      const sy = (startKey - sx) / stride;
      loop.push([sx, sy]);

      // Safety guard against malformed graphs (bounded by total pixel-edge count)
      let guard = 0;
      const maxSteps = (x1Bound - x0Bound + 3) * (y1Bound - y0Bound + 3) * 4 + 16;

      while (currKey !== startKey && guard < maxSteps) {
        guard++;
        const cx = currKey % stride;
        const cy = (currKey - cx) / stride;
        loop.push([cx, cy]);

        const targets = outEdges.get(currKey);
        if (!targets || targets.length === 0) break;

        let chosenIdx = targets.length - 1;
        if (targets.length > 1) {
          // Saddle vertex (diagonal touch): prefer sharpest RIGHT turn (-dy, dx) in screen coords
          // so 4-connected foreground components stay cleanly separated without crossing.
          const px = prevKey % stride;
          const py = (prevKey - px) / stride;
          const inDx = cx - px;
          const inDy = cy - py;
          const rightDx = -inDy;
          const rightDy = inDx;

          let bestScore = -1;
          for (let i = 0; i < targets.length; i++) {
            const tk = targets[i];
            const tx = tk % stride;
            const ty = (tk - tx) / stride;
            const outDx = tx - cx;
            const outDy = ty - cy;
            // Priority: right turn (3) > straight (2) > left turn (1)
            let score = 1;
            if (outDx === rightDx && outDy === rightDy) score = 3;
            else if (outDx === inDx && outDy === inDy) score = 2;
            if (score > bestScore) {
              bestScore = score;
              chosenIdx = i;
            }
          }
        }

        const chosen = targets[chosenIdx];
        targets.splice(chosenIdx, 1);
        prevKey = currKey;
        currKey = chosen;
      }

      if (loop.length >= 4) {
        const merged = mergeCollinearLoop(loop);
        if (merged.length >= 3) loops.push(merged);
      }
    }
  }

  if (loops.length === 0) {
    return {
      outer: [
        [x0Bound, y0Bound],
        [x1Bound + 1, y0Bound],
        [x1Bound + 1, y1Bound + 1],
        [x0Bound, y1Bound + 1],
      ],
      holes: [],
    };
  }

  // Outer boundary has the largest positive signed area (clockwise in +x right / +y down)
  let outerIdx = 0;
  let maxArea = -Infinity;
  for (let i = 0; i < loops.length; i++) {
    const sa = signedArea2(loops[i]);
    if (sa > maxArea) {
      maxArea = sa;
      outerIdx = i;
    }
  }

  const outer = loops[outerIdx];
  const holes: Array<Array<[number, number]>> = [];
  for (let i = 0; i < loops.length; i++) {
    if (i !== outerIdx) holes.push(loops[i]);
  }

  return { outer, holes };
}

/** Trace exact pixel-edge outer and hole contours for a labeled connected component. */
export function traceComponentPixelContours(
  labels: Int32Array,
  labelId: number,
  w: number,
  h: number,
  minX: number,
  maxX: number,
  minY: number,
  maxY: number
): TracedContours {
  return tracePixelEdgeLoops(
    (x, y) => labels[y * w + x] === labelId,
    w,
    h,
    minX,
    maxX,
    minY,
    maxY
  );
}

/** Trace exact pixel-edge outer and hole contours for a binary Uint8Array mask. */
export function traceMaskPixelContours(
  mask: Uint8Array,
  w: number,
  h: number,
  minX: number,
  maxX: number,
  minY: number,
  maxY: number
): TracedContours {
  return tracePixelEdgeLoops(
    (x, y) => mask[y * w + x] !== 0,
    w,
    h,
    minX,
    maxX,
    minY,
    maxY
  );
}

/**
 * Light corner-preserving smoothing of an exact axis-aligned pixel-edge contour.
 *
 * - Re-densifies unit steps only within 3 px of corners so long straight edges stay
 *   strictly flat (0.00 px drift) and fast.
 * - Applies 2 gentle Taubin shrink/inflate passes (lambda=+0.5 / mu=-0.45, then +0.34 / -0.31),
 *   converting 1-px diagonal staircases into crisp smooth curves (< 0.6 px max deviation)
 *   while preserving concavities and enclosed area.
 * - Drops nearly collinear intermediate points (< 0.08 px perpendicular deviation).
 */
export function smoothPixelContour(exactPoly: Array<[number, number]>): Array<[number, number]> {
  const n = exactPoly.length;
  if (n < 3) return exactPoly;

  // Step 1: Densify within 3 px of corners along each edge
  const dense: Array<[number, number]> = [];
  for (let i = 0; i < n; i++) {
    const [x0, y0] = exactPoly[i];
    const [x1, y1] = exactPoly[(i + 1) % n];
    const dx = x1 - x0;
    const dy = y1 - y0;
    const len = Math.hypot(dx, dy);
    if (len < 1e-6) continue;
    const ux = dx / len;
    const uy = dy / len;

    if (len <= 6.5) {
      const steps = Math.max(1, Math.round(len));
      for (let s = 0; s < steps; s++) {
        dense.push([x0 + (dx * s) / steps, y0 + (dy * s) / steps]);
      }
    } else {
      dense.push([x0, y0]);
      dense.push([x0 + ux, y0 + uy]);
      dense.push([x0 + ux * 2, y0 + uy * 2]);
      dense.push([x0 + ux * 3, y0 + uy * 3]);
      dense.push([x1 - ux * 3, y1 - uy * 3]);
      dense.push([x1 - ux * 2, y1 - uy * 2]);
      dense.push([x1 - ux, y1 - uy]);
    }
  }

  const m = dense.length;
  if (m < 4) return exactPoly;

  // Step 2: Light area-preserving Taubin smoothing (2 pairs of shrink/inflate)
  let xs = new Float64Array(m);
  let ys = new Float64Array(m);
  for (let i = 0; i < m; i++) {
    xs[i] = dense[i][0];
    ys[i] = dense[i][1];
  }

  const applyStep = (factor: number) => {
    const nx = new Float64Array(m);
    const ny = new Float64Array(m);
    for (let i = 0; i < m; i++) {
      const prev = (i - 1 + m) % m;
      const next = (i + 1) % m;
      const mx = 0.5 * (xs[prev] + xs[next]);
      const my = 0.5 * (ys[prev] + ys[next]);
      nx[i] = xs[i] + factor * (mx - xs[i]);
      ny[i] = ys[i] + factor * (my - ys[i]);
    }
    xs = nx;
    ys = ny;
  };

  applyStep(0.5);
  applyStep(-0.45);
  applyStep(0.34);
  applyStep(-0.31);

  // Step 3: Lightly prune redundant nearly-collinear vertices (< 0.08 px deviation)
  const out: Array<[number, number]> = [];
  for (let i = 0; i < m; i++) {
    const prev = (i - 1 + m) % m;
    const next = (i + 1) % m;
    const ax = xs[prev], ay = ys[prev];
    const bx = xs[i], by = ys[i];
    const cx = xs[next], cy = ys[next];
    const abx = cx - ax;
    const aby = cy - ay;
    const abLen = Math.hypot(abx, aby);
    if (abLen < 1e-6) continue;
    const perpDist = Math.abs((bx - ax) * aby - (by - ay) * abx) / abLen;
    if (perpDist >= 0.08) {
      out.push([bx, by]);
    }
  }

  return out.length >= 3 ? out : exactPoly;
}
