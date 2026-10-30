// A horizontal cloud layer seen from one point on the ground.
//
// Clouds are vertical cylinders (flat base, flat top, given thickness) placed
// on an endless square lattice: each cell holds at most one cloud, centred at
// a random spot inside the cell (it may overhang into neighbours). Because the lattice is generated from a hash of the
// cell index, the layer is infinite and costs no memory.
//
// For every sample direction on the dome we cast a ray from the observer and
// ask: does it hit any cloud? Earth curvature is included: a ray leaving at
// elevation e has height  z(d) = d tan(e) + d^2 / (2R)  after horizontal
// distance d, which is why even a ray aimed exactly at the horizon eventually
// reaches the layer (about sqrt(2Rh) away).

import { grid, W, H, N } from './sky.js';

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

// Turn the user-facing description into lattice parameters. Each cell holds a
// cloud with probability P_OCC, centred anywhere in the cell, so neighbouring
// clouds may overlap. Overlaps make the covered fraction a non-linear function
// of spacing, so the spacing is tuned by measuring the map-view cover.
const P_OCC = 0.85;
const geoCache = new Map();

export function layerGeometry({ cover, diameter, seed }) {
  const key = `${cover}|${diameter}|${seed}`;
  if (geoCache.has(key)) return geoCache.get(key);
  const r = diameter / 2;
  const target = Math.min(cover, 0.95);
  // Boolean-model first guess: cover = 1 - exp(-density * pi r^2)
  let s = r * Math.sqrt((P_OCC * Math.PI) / -Math.log(1 - target));
  let geo;
  for (let it = 0; it < 4; it++) {
    s = Math.max(s, r); // a cloud may reach at most one cell beyond its own
    geo = { r, s, p: P_OCC, seed };
    geo.planCover = measurePlanCover(geo, it < 3 ? 120 : 240);
    if (Math.abs(geo.planCover - target) < 0.002) break;
    s *= Math.sqrt(Math.log(1 - geo.planCover) / Math.log(1 - target));
  }
  if (geoCache.size > 200) geoCache.clear();
  geoCache.set(key, geo);
  return geo;
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

export function castLayer(params, cover) {
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

  // 2 x 2 rays per grid cell, so cover holds fractions 0, .25, ... 1
  const SS = 2;
  cover.fill(0);
  for (let j = 0; j < H; j++) {
    for (let sj = 0; sj < SS; sj++) {
      const e = ((j + (sj + 0.5) / SS) / H) * (Math.PI / 2);
      const ds = distanceToHeight(e, base, curved);
      const de = Math.min(distanceToHeight(e, top, curved), maxD);
      if (!(ds < de)) continue;
      for (let i = 0; i < W; i++) {
        for (let si = 0; si < SS; si++) {
          const a = ((i + (si + 0.5) / SS) / W) * 2 * Math.PI;
          if (hitRay(a, ds, de)) cover[j * W + i] += 1 / (SS * SS);
        }
      }
    }
  }
  return geo;
}

// Clouds close enough to draw inside a unit dome where the cloud base sits at
// height `sceneBase`. Returns positions/sizes in scene units.
export function nearbyClouds(params, sceneBase, limit = 4000) {
  const geo = layerGeometry(params);
  const k = sceneBase / params.base; // scene units per metre
  const top = (params.base + params.thickness) * k;
  const rr = geo.r * k;
  const out = [];
  const reach = Math.sqrt(Math.max(0, 1 - sceneBase * sceneBase)) / k; // metres
  const n = Math.ceil(reach / geo.s) + 1;
  for (let ix = -n; ix <= n; ix++) {
    for (let iy = -n; iy <= n; iy++) {
      const c = cloudAt(ix, iy, geo);
      if (!c) continue;
      const cx = c.x * k, cy = c.y * k;
      const far = Math.hypot(cx, cy) + rr;
      if (far * far + top * top > 0.995) continue;
      out.push({ x: cx, y: cy, r: rr, base: sceneBase, top });
      if (out.length >= limit) return out;
    }
  }
  return out;
}

export { EARTH_R, N };
