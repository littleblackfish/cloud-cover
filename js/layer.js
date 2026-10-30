// A horizontal cloud layer seen from one point on the ground.
//
// Two kinds of layer are modelled. A cumuliform layer is made of separate
// cloud elements (below). A stratiform layer is a continuous sheet with
// holes: a smooth random field over the plane, cloudy where it exceeds a
// threshold chosen to give the requested map-view cover. Its holes have
// vertical walls, so a thick sheet hides more sky near the horizon too.
//
// Clouds are vertical cylinders (flat base, flat top, given thickness) placed
// on an endless square lattice. Because the lattice is generated from a hash
// of the cell index, the layer is infinite and costs no memory.
//
// For every sample direction on the dome we cast a ray from the observer and
// ask: does it hit any cloud? Earth curvature is included: a ray leaving at
// elevation e has height  z(d) = d tan(e) + d^2 / (2R)  after horizontal
// distance d, which is why even a ray aimed exactly at the horizon eventually
// reaches the layer (about sqrt(2Rh) away).

import { grid, W, H, N, makeNoise3 } from './sky.js';

const EARTH_R = 6371000;

function hash3(ix, iy, seed) {
  let h = Math.imul(ix, 0x27d4eb2d) ^ Math.imul(iy, 0x165667b1) ^ Math.imul(seed, 0x9e3779b1);
  h = Math.imul(h ^ (h >>> 15), 0x85ebca6b);
  h = Math.imul(h ^ (h >>> 13), 0xc2b2ae35);
  h ^= h >>> 16;
  const a = (h >>> 0) / 4294967296;
  let g = Math.imul(h ^ 0x5bd1e995, 0x01000193);
  g ^= g >>> 15;
  const b = (g >>> 0) / 4294967296;
  let q = Math.imul(g ^ 0x2545f491, 0x9e3779b1);
  q ^= q >>> 13;
  const c = (q >>> 0) / 4294967296;
  return [a, b, c];
}

// Cloud elements sit on a lattice of spacing `size`: each cell holds a cloud
// with probability P_OCC, centred anywhere in the cell, so neighbours may
// overlap. Coverage is set by the clouds' radius, so asking for more cover
// makes the same clouds bigger rather than adding new ones. Because of the
// overlaps, cover is a non-linear function of radius; it depends only on
// radius / spacing, so it is measured once per random pattern and tabulated.
const P_OCC = 0.85;
const TABLE_N = 48;
const tableCache = new Map();

function coverTable(seed) {
  if (tableCache.has(seed)) return tableCache.get(seed);
  const t = new Float64Array(TABLE_N + 1);
  for (let i = 1; i <= TABLE_N; i++) t[i] = measurePlanCover({ r: i / TABLE_N, s: 1, p: P_OCC, seed }, 110);
  for (let i = 1; i <= TABLE_N; i++) t[i] = Math.max(t[i], t[i - 1]); // keep monotone
  if (tableCache.size > 20) tableCache.clear();
  tableCache.set(seed, t);
  return t;
}

// Largest map-view cover separate elements can reach (radius = spacing).
export const maxElementCover = (seed) => coverTable(seed)[TABLE_N];

export function layerGeometry({ cover, size, seed }) {
  const t = coverTable(seed);
  const c = Math.min(cover, t[TABLE_N]);
  let i = 1;
  while (i < TABLE_N && t[i] < c) i++;
  const f = t[i] > t[i - 1] ? (c - t[i - 1]) / (t[i] - t[i - 1]) : 1;
  const k = (i - 1 + f) / TABLE_N;
  return { r: k * size, s: size, p: P_OCC, seed, planCover: c };
}

function coveredAt(px, py, geo) {
  const ix = Math.floor(px / geo.s), iy = Math.floor(py / geo.s);
  const r2 = geo.r * geo.r;
  for (let ox = -1; ox <= 1; ox++) {
    for (let oy = -1; oy <= 1; oy++) {
      const c = cloudAt(ix + ox, iy + oy, geo);
      if (c && (c.x - px) ** 2 + (c.y - py) ** 2 <= r2) return true;
    }
  }
  return false;
}

// Map-view cover measured on an n x n sample grid spanning 40 cells.
function measurePlanCover(geo, n) {
  const span = 40 * geo.s;
  let hit = 0;
  for (let a = 0; a < n; a++) {
    for (let b = 0; b < n; b++) {
      if (coveredAt(((a + 0.5) / n - 0.5) * span, ((b + 0.5) / n - 0.5) * span, geo)) hit++;
    }
  }
  return hit / (n * n);
}

// Horizontal distance at which a ray of elevation e reaches height h.
export function distanceToHeight(e, h, curved) {
  const t = Math.tan(e);
  if (!curved) return t <= 0 ? Infinity : h / t;
  // root of d^2/(2R) + d t - h = 0, written in a cancellation-free form
  return (2 * h) / (t + Math.sqrt(t * t + (2 * h) / EARTH_R));
}

export function cloudAt(ix, iy, geo) {
  const [occ, jx, jy] = hash3(ix, iy, geo.seed);
  if (occ >= geo.p) return null;
  return { x: (ix + jx) * geo.s, y: (iy + jy) * geo.s };
}

// Cast rays over a w x h azimuth/elevation grid (ss x ss rays per cell) and
// return the covered fraction of the dome. If `cover` is given, the per-cell
// fractions are written into it.
function castGrid(params, w, h, ss, cover) {
  if (cover) cover.fill(0);
  if (params.cover <= 0) return { geo: { planCover: 0 }, frac: 0 };
  return params.form === 'stratiform'
    ? castSheet(params, w, h, ss, cover)
    : castElements(params, w, h, ss, cover);
}

// The full-resolution sky grid. `params.quick` casts one ray per cell instead
// of four (used while the layer is being animated).
export function castLayer(params, cover) {
  return castGrid(params, W, H, params.quick ? 1 : 2, cover).geo;
}

// Dome fraction from a coarse grid: cheap enough to search with.
const domeFraction = (params) => castGrid(params, 240, 60, 1, null).frac;

// Map-view cover that makes the layer hide `target` of the dome. Dome cover
// grows with map cover, so bisect; `hint` (the last answer) narrows the start.
export function solvePlanCover(params, target, hint) {
  if (target <= 0) return 0;
  const max = params.form === 'stratiform' ? 1 : maxElementCover(params.seed);
  if (target >= 1) return max;
  let lo = 0, hi = max;
  if (hint > 0 && hint < max) {
    const at = domeFraction({ ...params, cover: hint });
    if (at < target) lo = hint; else hi = hint;
    const iters = 6;
    for (let i = 0; i < iters; i++) {
      // walk outward until the bracket holds the answer
      const probe = lo === hint ? Math.min(max, hint + (hint * 0.25) * 2 ** i) : Math.max(0, hint - (hint * 0.25) * 2 ** i);
      const f = domeFraction({ ...params, cover: probe });
      if (lo === hint) { if (f >= target) { hi = probe; break; } lo = probe; } else { if (f < target) { lo = probe; break; } hi = probe; }
    }
  }
  for (let i = 0; i < 14; i++) {
    const mid = (lo + hi) / 2;
    if (domeFraction({ ...params, cover: mid }) < target) lo = mid; else hi = mid;
  }
  return (lo + hi) / 2;
}

function rowWeights(h) {
  const wts = new Float64Array(h);
  for (let j = 0; j < h; j++) wts[j] = Math.sin(((j + 1) / h) * Math.PI / 2) - Math.sin((j / h) * Math.PI / 2);
  return wts;
}

function castElements(params, w, h, SS, cover) {
  const { base, thickness, extent, curved } = params;
  const geo = layerGeometry(params);
  const { r, s } = geo;
  const r2 = r * r;
  const top = base + thickness;
  const maxD = extent * 1000;

  // does the ray (dx, dy) hit a cloud of cell (cx, cy) between ds and de?
  const hits = (cx, cy, dx, dy, ds, de) => {
    const c = cloudAt(cx, cy, geo);
    if (!c) return false;
    const t0 = c.x * dx + c.y * dy;
    const perp2 = c.x * c.x + c.y * c.y - t0 * t0;
    if (perp2 > r2) return false;
    const half = Math.sqrt(r2 - perp2);
    return t0 + half >= ds && t0 - half <= de;
  };

  const hitRay = (a, ds, de) => {
    const dx = Math.sin(a), dy = Math.cos(a);
    // 2D DDA through lattice cells from distance ds to de along (dx, dy).
    // A cloud can overhang its cell by up to one cell, so each visited
    // cell also checks its eight neighbours.
    const px = ds * dx, py = ds * dy;
    let ix = Math.floor(px / s), iy = Math.floor(py / s);
    const stepX = dx > 0 ? 1 : -1, stepY = dy > 0 ? 1 : -1;
    const tdx = Math.abs(dx) < 1e-12 ? Infinity : s / Math.abs(dx);
    const tdy = Math.abs(dy) < 1e-12 ? Infinity : s / Math.abs(dy);
    let tmx = Math.abs(dx) < 1e-12 ? Infinity : ds + ((dx > 0 ? (ix + 1) * s : ix * s) - px) / dx;
    let tmy = Math.abs(dy) < 1e-12 ? Infinity : ds + ((dy > 0 ? (iy + 1) * s : iy * s) - py) / dy;
    for (let guard = 0; guard < 100000; guard++) {
      for (let ox = -1; ox <= 1; ox++) {
        for (let oy = -1; oy <= 1; oy++) {
          if (hits(ix + ox, iy + oy, dx, dy, ds, de)) return true;
        }
      }
      if (tmx < tmy) {
        if (tmx > de) return false;
        tmx += tdx;
        ix += stepX;
      } else {
        if (tmy > de) return false;
        tmy += tdy;
        iy += stepY;
      }
    }
    return false;
  };

  const wts = rowWeights(h);
  let frac = 0;
  for (let j = 0; j < h; j++) {
    let row = 0;
    for (let sj = 0; sj < SS; sj++) {
      const e = ((j + (sj + 0.5) / SS) / h) * (Math.PI / 2);
      const ds = distanceToHeight(e, base, curved);
      const de = Math.min(distanceToHeight(e, top, curved), maxD);
      if (!(ds < de)) continue;
      for (let i = 0; i < w; i++) {
        for (let si = 0; si < SS; si++) {
          const a = ((i + (si + 0.5) / SS) / w) * 2 * Math.PI;
          if (hitRay(a, ds, de)) {
            row++;
            if (cover) cover[j * w + i] += 1 / (SS * SS);
          }
        }
      }
    }
    frac += (row / (w * SS * SS)) * wts[j];
  }
  return { geo, frac };
}

// ---------------------------------------------------------------- sheets

const fieldCache = new Map();

// Smooth random field over the ground plane (metres) with features about
// `size` across, plus a sorted sample of its values for finding thresholds.
export function sheetField({ size, seed }) {
  const key = `${size}|${seed}`;
  if (fieldCache.has(key)) return fieldCache.get(key);
  const noise = makeNoise3(seed);
  const f = 1 / size;
  const field = (x, y) => noise(x * f, y * f, 0.37) + 0.45 * noise(x * f * 2.3 + 17, y * f * 2.3, 1.7);
  const n = 220, span = 60 * size, vals = new Float32Array(n * n);
  for (let a = 0; a < n; a++) {
    for (let b = 0; b < n; b++) vals[a * n + b] = field((a / n - 0.5) * span, (b / n - 0.5) * span);
  }
  vals.sort();
  const out = { field, vals, perm: noise.perm };
  if (fieldCache.size > 30) fieldCache.clear();
  fieldCache.set(key, out);
  return out;
}

// The sheet is cloudy where the field exceeds T; lowering T grows the patches
// and shrinks the gaps.
export function sheetGeometry(params) {
  const { field, vals } = sheetField(params);
  const { cover } = params;
  const T = cover >= 0.999 ? -Infinity : vals[Math.min(vals.length - 1, Math.floor((1 - cover) * vals.length))];
  return { field, T, planCover: Math.min(cover, 1) };
}

function castSheet(params, w, h, SS, cover) {
  const { base, thickness, extent, curved } = params;
  const geo = sheetGeometry(params);
  const { field, T } = geo;
  const top = base + thickness;
  const maxD = extent * 1000;
  const step = params.size / 3;
  const wts = rowWeights(h);
  let frac = 0;
  for (let j = 0; j < h; j++) {
    let row = 0;
    for (let sj = 0; sj < SS; sj++) {
      const e = ((j + (sj + 0.5) / SS) / h) * (Math.PI / 2);
      const ds = distanceToHeight(e, base, curved);
      if (!(ds < maxD)) continue;
      const de = Math.min(distanceToHeight(e, top, curved), maxD);
      // sample the ray's path through the sheet: a hole shows sky only if it
      // is open all the way through
      const n = Math.min(6, 1 + Math.ceil((de - ds) / step));
      for (let i = 0; i < w; i++) {
        for (let si = 0; si < SS; si++) {
          const a = ((i + (si + 0.5) / SS) / w) * 2 * Math.PI;
          const dx = Math.sin(a), dy = Math.cos(a);
          for (let q = 0; q < n; q++) {
            const d = n === 1 ? ds : ds + ((de - ds) * q) / (n - 1);
            if (field(d * dx, d * dy) > T) {
              row++;
              if (cover) cover[j * w + i] += 1 / (SS * SS);
              break;
            }
          }
        }
      }
    }
    frac += (row / (w * SS * SS)) * wts[j];
  }
  return { geo, frac };
}

// ---------------------------------------------------------------- drawing

// Cloud elements within `reach` metres (horizontally) of the observer, nearest
// first, in scene units (`k` scene units per metre).
export function nearbyClouds(params, k, reach, limit) {
  if (params.cover <= 0) return [];
  const geo = layerGeometry(params);
  if (geo.r <= 0) return [];
  const base = params.base * k;
  const top = (params.base + params.thickness) * k;
  // only the nearest `limit` are wanted: a square of n cells each way holds
  // a circle with about that many clouds
  const n = Math.min(Math.ceil(reach / geo.s) + 1, Math.ceil(Math.sqrt(limit / (geo.p * Math.PI))) + 2);
  const out = [];
  for (let ix = -n; ix <= n; ix++) {
    for (let iy = -n; iy <= n; iy++) {
      const c = cloudAt(ix, iy, geo);
      if (!c) continue;
      const d = Math.hypot(c.x, c.y);
      if (d > reach) continue;
      out.push({ x: c.x * k, y: c.y * k, r: geo.r * k, base, top, d });
    }
  }
  out.sort((p, q) => p.d - q.d);
  return out.slice(0, limit);
}

export { EARTH_R, N };
