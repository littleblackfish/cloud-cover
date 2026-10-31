// A flat cloud layer seen from one point on the ground.
//
// The layer holds one round patch, placed in the sky by azimuth and
// elevation, optionally drawn as one of the cloud types an observer reports
// (stratocumulus, altocumulus, cirrus ...). Each type is a pattern that lives
// on the layer itself, so perspective squeezes it toward the horizon.
//
// Cloudiness for a sky direction = the patch's field for that direction (as on
// the Celestial dome tab) + the type's pattern where that line of sight meets
// the layer. The cloud is where cloudiness exceeds a threshold chosen so the
// dome is covered by exactly the requested amount.
//
// Lines of sight follow the Earth's curve: a ray leaving at elevation e has
// height z(d) = d tan(e) + d^2 / (2R) after horizontal distance d, so even a
// ray aimed exactly at the horizon reaches the layer (about sqrt(2Rh) away).
// The GPU copy of all this lives in main.js (LAYER_GLSL) and must match; the
// patterns themselves are shared, baked once below.

import { grid, W, H, shapeField } from './sky.js';

export const EARTH_R = 6371000;

// Horizontal distance at which a ray of elevation e reaches height h.
export function distanceToHeight(e, h) {
  const t = Math.tan(e);
  // root of d^2/(2R) + d t - h = 0, written in a cancellation-free form
  return (2 * h) / (t + Math.sqrt(t * t + (2 * h) / EARTH_R));
}

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

// ---------------------------------------------------------------- patterns
//
// Each pattern is baked once into a square tile that repeats across the
// layer. Inside the tile, coordinates u, v run 0..1, and every ingredient
// repeats a whole number of times across it, so the tile joins up seamlessly.
// Two channels: cover (-ENC..ENC; added to the patch's field, so it decides
// where the cloud is) and shade (0..1; thicker parts, drawn darker).
//
// The GPU samples the same bytes, with mipmaps, so a pixel looking far along
// the layer sees the pattern averaged instead of flickering; the CPU reads
// them back with the same bilinear filter for the octa numbers. A second,
// shared tile of large, slow blotches is laid over every type at another
// scale and angle, so the repeat of the first does not show.

const ENC = 1.5; // cover byte b stands for (b / 255) * 2 ENC - ENC

// Gradient noise on an nx x ny lattice over the tile (wraps at the edges).
const latticeCache = new Map();
function lattice(nx, ny, seed) {
  const key = (nx * 4096 + ny) * 65536 + seed;
  let g = latticeCache.get(key);
  if (!g) {
    g = new Float32Array(nx * ny * 2);
    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < nx; i++) {
        const t = hash3(i, j, seed)[0] * 2 * Math.PI;
        g[(j * nx + i) * 2] = Math.cos(t);
        g[(j * nx + i) * 2 + 1] = Math.sin(t);
      }
    }
    latticeCache.set(key, g);
  }
  return g;
}
const fade = (t) => t * t * t * (t * (t * 6 - 15) + 10);
function gnoise(u, v, nx, ny, seed) {
  const g = lattice(nx, ny, seed);
  const x = u * nx, y = v * ny;
  const ix = Math.floor(x), iy = Math.floor(y);
  const fx = x - ix, fy = y - iy;
  let i0 = ix % nx; if (i0 < 0) i0 += nx;
  let j0 = iy % ny; if (j0 < 0) j0 += ny;
  const i1 = i0 + 1 === nx ? 0 : i0 + 1, j1 = j0 + 1 === ny ? 0 : j0 + 1;
  const k00 = (j0 * nx + i0) * 2, k10 = (j0 * nx + i1) * 2;
  const k01 = (j1 * nx + i0) * 2, k11 = (j1 * nx + i1) * 2;
  const d00 = g[k00] * fx + g[k00 + 1] * fy;
  const d10 = g[k10] * (fx - 1) + g[k10 + 1] * fy;
  const d01 = g[k01] * fx + g[k01 + 1] * (fy - 1);
  const d11 = g[k11] * (fx - 1) + g[k11 + 1] * (fy - 1);
  const a = fade(fx), b = fade(fy);
  const n0 = d00 + (d10 - d00) * a, n1 = d01 + (d11 - d01) * a;
  return 1.6 * (n0 + (n1 - n0) * b); // about -1..1
}
// Several octaves; nx, ny set the coarsest. nx < ny stretches it along u.
function fbm(u, v, nx, ny, octaves, seed) {
  let s = 0, amp = 1, norm = 0;
  for (let o = 0; o < octaves; o++) {
    s += amp * gnoise(u, v, nx << o, ny << o, seed + 101 * o);
    norm += amp;
    amp *= 0.5;
  }
  return (s / norm) * 1.4;
}
// Bends u, v by a smooth random offset, for organic outlines.
function warp(u, v, n, amount, seed) {
  return [u + (amount / n) * gnoise(u, v, n, n, seed + 7), v + (amount / n) * gnoise(u, v, n, n, seed + 13)];
}
// Rounded elements on an nx x ny lattice, at most one per cell at a random
// spot in it (`occupied`: the share of cells holding one). 1 at an element's
// centre, 0 at its rim (radius in cells), falling to -1 between elements.
// nx > ny makes them longer along v: rolls.
const cellCache = new Map();
function cells(u, v, nx, ny, seed, occupied, radius) {
  const key = (nx * 4096 + ny) * 65536 + seed;
  let c = cellCache.get(key);
  if (!c) {
    c = new Float32Array(nx * ny * 3);
    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < nx; i++) {
        const [a, b, d] = hash3(i, j, seed + 999);
        c.set([a, 0.15 + 0.7 * b, 0.15 + 0.7 * d], (j * nx + i) * 3);
      }
    }
    cellCache.set(key, c);
  }
  const x = u * nx, y = v * ny;
  const ix = Math.floor(x), iy = Math.floor(y);
  let best = -1;
  for (let oy = -1; oy <= 1; oy++) {
    for (let ox = -1; ox <= 1; ox++) {
      const cx = ix + ox, cy = iy + oy;
      let wx = cx % nx; if (wx < 0) wx += nx;
      let wy = cy % ny; if (wy < 0) wy += ny;
      const k = (wy * nx + wx) * 3;
      if (c[k] >= occupied) continue;
      // element size varies a little from cell to cell
      const r = radius * (0.8 + 0.4 * c[k] / occupied);
      const e = 1 - Math.hypot(cx + c[k + 1] - x, cy + c[k + 2] - y) / r;
      if (e > best) best = e;
    }
  }
  return Math.max(-1, best);
}
const clamp01 = (x) => Math.min(1, Math.max(0, x));

// ---------------------------------------------------------------- cloud types
//
// The types reported in Canadian METARs (MANOBS). Sizes are real sizes on the
// layer, chosen so that at the type's usual base the elements look the size
// the cloud atlas gives, seen more than 30 degrees up: cirrocumulus grains
// under 1 degree across, altocumulus 1 to 5, stratocumulus over 5.
//
//   tileM    the tile's size on the layer (m)
//   featureM the pattern's typical feature (m): where pixels outgrow it, the
//            pattern gives way to its average
//   amount   how strongly the pattern breaks up the patch; large = scattered
//   blotch   how strongly the shared large blotches do
//   soft     extra blur of the cloud's edge (field units, radians)
//   opacity  how much of the sky behind shows through: thin veils are low
//   grey     the cloud's colour (sRGB) and shade, how much darker its thick
//            parts are
//   wisp     how much its thin parts let the sky through (fibrous veils)
//   angle    the pattern's direction on the layer (deg): streaks, rolls
//   flat     false: needs height to draw properly (towering types), not yet
export const TYPES = {
  CI: {
    name: 'Cirrus', etage: 'high', etages: ['high'], baseFt: 28000,
    tileM: 36000, featureM: 700, amount: 1.1, blotch: 0.45, soft: 0.1,
    opacity: 0.85, wisp: 0.9, grey: [248, 249, 252], shade: 0.04, angle: 35, res: 512,
    help: 'Cirrus (CI): detached white filaments, fibrous or silky, often hooked or in streaks. Thin: the sky shows through.',
    gen(u, v, s) {
      // detached tufts, stretched along the wind and gently bent; inside
      // them, fine parallel fibres decide how much sky shows through
      const [a, b] = warp(u, v, 3, 0.3, s);
      const tuft = fbm(a, b, 3, 8, 4, s + 5);
      const fibre = (1 - Math.abs(fbm(a, b, 4, 64, 3, s + 9) * 0.8)) ** 3;
      return [1.2 * tuft + 0.3 * fibre, clamp01(0.1 + 0.9 * fibre)];
    },
  },
  CC: {
    name: 'Cirrocumulus', etage: 'high', etages: ['high'], baseFt: 25000,
    tileM: 7000, featureM: 95, amount: 1.0, blotch: 0.6, soft: 0.01,
    opacity: 0.85, grey: [244, 246, 249], shade: 0.12, angle: 15, res: 512,
    help: 'Cirrocumulus (CC): a thin white sheet of very small grains or ripples, without shading, each under 1° across (smaller than your little finger at arm’s length).',
    gen(u, v, s) {
      const [a, b] = warp(u, v, 9, 0.25, s);
      const e = cells(a, b, 72, 56, s, 0.86, 0.48);
      return [e + 0.12 * fbm(u, v, 24, 24, 2, s + 3), clamp01(e)];
    },
  },
  CS: {
    name: 'Cirrostratus', etage: 'high', etages: ['high'], baseFt: 26000,
    tileM: 40000, featureM: 6000, amount: 0.08, blotch: 0.1, soft: 0.15,
    opacity: 0.4, wisp: 0.5, grey: [246, 248, 251], shade: 0.05, angle: 30, res: 512,
    help: 'Cirrostratus (CS): a transparent whitish veil, fibrous or smooth, often over most of the sky. The sun shines through it and it may make a halo.',
    gen(u, v, s) {
      return [0.5 * fbm(u, v, 2, 2, 3, s), clamp01(0.5 + 0.6 * fbm(u, v, 3, 24, 4, s + 7))];
    },
  },
  AC: {
    name: 'Altocumulus', etage: 'mid', etages: ['mid'], baseFt: 12000,
    tileM: 9000, featureM: 220, amount: 0.9, blotch: 0.5, soft: 0.015,
    opacity: 0.93, grey: [218, 222, 229], shade: 0.35, angle: 20, res: 512,
    help: 'Altocumulus (AC): white or grey rounded masses or rolls with shading, in sheets or patches, each 1° to 5° across (about one to three fingers at arm’s length).',
    gen(u, v, s) {
      const [a, b] = warp(u, v, 10, 0.45, s);
      const e = cells(a, b, 40, 28, s, 0.92, 0.55);
      return [e + 0.2 * fbm(u, v, 64, 64, 2, s + 3), clamp01(e)];
    },
  },
  ACC: {
    name: 'Altocumulus castellanus', etage: 'mid', etages: ['mid'], baseFt: 12000, flat: false,
    help: 'Altocumulus castellanus (ACC): altocumulus with turrets rising from a common base. Needs height to draw: coming next.',
  },
  AS: {
    name: 'Altostratus', etage: 'mid', etages: ['mid'], baseFt: 10000,
    tileM: 40000, featureM: 6000, amount: 0.1, blotch: 0.2, soft: 0.08,
    opacity: 0.9, wisp: 0.15, grey: [178, 184, 194], shade: 0.15, angle: 30, res: 512,
    help: 'Altostratus (AS): a grey or bluish sheet, streaky or uniform, over much of the sky. The sun shows dimly, as through ground glass, and there is no halo.',
    gen(u, v, s) {
      return [0.5 * fbm(u, v, 2, 2, 3, s), clamp01(0.5 + 0.6 * fbm(u, v, 2, 16, 4, s + 5))];
    },
  },
  NS: {
    name: 'Nimbostratus', etage: 'low', etages: ['low', 'mid'], baseFt: 2500,
    tileM: 30000, featureM: 5000, amount: 0.12, blotch: 0.15, soft: 0.12,
    opacity: 0.98, grey: [140, 145, 152], shade: 0.2, angle: 0, res: 512,
    help: 'Nimbostratus (NS): a thick, dark grey layer, made diffuse by falling rain or snow, that hides the sun completely. A mid-level cloud in the atlas, but its base is usually low.',
    gen(u, v, s) {
      return [0.5 * fbm(u, v, 2, 2, 4, s), clamp01(0.5 + 0.6 * fbm(u, v, 6, 6, 4, s + 3))];
    },
  },
  SC: {
    name: 'Stratocumulus', etage: 'low', etages: ['low'], baseFt: 3500,
    tileM: 10000, featureM: 700, amount: 0.55, blotch: 0.4, soft: 0.03,
    opacity: 0.95, grey: [205, 209, 215], shade: 0.35, angle: 20, res: 512,
    help: 'Stratocumulus (SC): grey or whitish patches, sheets or rolls with dark parts, usually joined. Each element over 5° across (more than three fingers at arm’s length).',
    gen(u, v, s) {
      const [a, b] = warp(u, v, 6, 0.25, s);
      const e = cells(a, b, 14, 9, s, 0.97, 0.78);
      return [e + 0.25 * fbm(u, v, 28, 28, 3, s + 3), clamp01(e)];
    },
  },
  ST: {
    name: 'Stratus', etage: 'low', etages: ['low'], baseFt: 1000,
    tileM: 8000, featureM: 700, amount: 0.12, blotch: 0.15, soft: 0.05,
    opacity: 0.92, grey: [196, 200, 206], shade: 0.15, angle: 0, res: 512,
    help: 'Stratus (ST): a generally grey layer with a fairly uniform base, often low. The sun’s outline is clear through thin stratus.',
    gen(u, v, s) {
      return [0.7 * fbm(u, v, 3, 3, 4, s), clamp01(0.5 + 0.6 * fbm(u, v, 12, 12, 3, s + 9))];
    },
  },
  SF: {
    name: 'Stratus fractus', etage: 'low', etages: ['low'], baseFt: 800,
    tileM: 2500, featureM: 90, amount: 2.0, blotch: 0.3, soft: 0.02,
    opacity: 0.85, grey: [184, 188, 195], shade: 0.25, angle: 25, res: 512,
    help: 'Stratus fractus (SF): ragged grey shreds of stratus, often low under rain clouds, changing shape quickly.',
    gen(u, v, s) {
      const [a, b] = warp(u, v, 8, 0.9, s);
      return [1.2 * fbm(a, b, 10, 5, 4, s), clamp01(0.5 + 0.6 * fbm(a, b, 20, 20, 3, s + 4))];
    },
  },
  CU: {
    name: 'Cumulus', etage: 'low', etages: ['low'], baseFt: 4000, flat: false,
    help: 'Cumulus (CU): detached, dense heaps with flat bases and sharp, bulging tops. Needs height to draw: coming next.',
  },
  CF: {
    name: 'Cumulus fractus', etage: 'low', etages: ['low'], baseFt: 2000,
    tileM: 4000, featureM: 300, amount: 1.2, blotch: 0.3, soft: 0.02,
    opacity: 0.9, grey: [222, 226, 231], shade: 0.3, angle: 0, res: 512,
    help: 'Cumulus fractus (CF): small ragged cumulus fragments with torn edges, changing quickly; often the first or last stage of cumulus.',
    gen(u, v, s) {
      const [a, b] = warp(u, v, 8, 0.45, s);
      const e = cells(a, b, 10, 10, s, 0.55, 0.5);
      return [e + 0.35 * fbm(u, v, 20, 20, 3, s + 3), clamp01(e)];
    },
  },
  TCU: {
    name: 'Towering cumulus', etage: 'low', etages: ['low'], baseFt: 4000, flat: false,
    help: 'Towering cumulus (TCU): cumulus grown much taller than wide, with cauliflower tops. Needs height to draw: coming next.',
  },
  CB: {
    name: 'Cumulonimbus', etage: 'low', etages: ['low'], baseFt: 3000, flat: false,
    help: 'Cumulonimbus (CB): a heavy, dense thunder cloud of great height, its top often spread into an anvil. Needs height to draw: coming next.',
  },
};
export const TYPE_ROWS = [
  ['high', ['CI', 'CC', 'CS']],
  ['mid', ['AC', 'ACC', 'AS']],
  ['low', ['NS', 'SC', 'ST', 'SF', 'CU', 'CF', 'TCU', 'CB']],
];

const bakeCache = new Map();
// The type's tile: Uint8 pairs (cover, shade), res x res. Baked a few rows
// at a time (see prebake), or all at once when needed now.
function* bakeRows(code, seed) {
  const t = TYPES[code];
  const n = t.res;
  const data = new Uint8Array(n * n * 2);
  for (let j = 0; j < n; j++) {
    for (let i = 0; i < n; i++) {
      const [c, sh] = t.gen((i + 0.5) / n, (j + 0.5) / n, seed);
      const k = (j * n + i) * 2;
      data[k] = Math.round(((Math.max(-ENC, Math.min(ENC, c)) + ENC) / (2 * ENC)) * 255);
      data[k + 1] = Math.round(clamp01(sh) * 255);
    }
    if ((j & 15) === 15) yield;
  }
  bakeCache.set(`${code}:${seed}`, { data, size: n, channels: 2 });
}
const pending = new Map();
export function bakeType(code, seed) {
  const key = `${code}:${seed}`;
  if (!bakeCache.has(key)) {
    const it = pending.get(key) ?? bakeRows(code, seed);
    while (!it.next().done);
    pending.delete(key);
  }
  return bakeCache.get(key);
}
// Bakes every type in the background, in slices of a few milliseconds.
export function prebake(seed, onIdle) {
  const queue = Object.keys(TYPES).filter((c) => TYPES[c].flat !== false);
  const step = (deadline) => {
    const until = performance.now() + (deadline?.timeRemaining?.() ?? 8);
    while (queue.length && performance.now() < until) {
      const key = `${queue[0]}:${seed}`;
      if (bakeCache.has(key)) { queue.shift(); continue; }
      if (!pending.has(key)) pending.set(key, bakeRows(queue[0], seed));
      if (pending.get(key).next().done) { pending.delete(key); queue.shift(); }
    }
    if (queue.length) onIdle(step);
  };
  onIdle(step);
}
// The shared tile of large, slow blotches: one cover channel.
export const BLOTCH_SCALE = 6.3; // its size, in the type's tiles
export const BLOTCH_ANGLE = 37; // and its turn against them (deg)
export function bakeBlotches(seed) {
  const key = `blotch:${seed}`;
  if (bakeCache.has(key)) return bakeCache.get(key);
  const n = 256;
  const data = new Uint8Array(n * n);
  for (let j = 0; j < n; j++) {
    for (let i = 0; i < n; i++) {
      const c = fbm((i + 0.5) / n, (j + 0.5) / n, 3, 3, 4, seed + 77);
      data[j * n + i] = Math.round(((Math.max(-ENC, Math.min(ENC, c)) + ENC) / (2 * ENC)) * 255);
    }
  }
  const out = { data, size: n, channels: 1 };
  bakeCache.set(key, out);
  return out;
}
// GLSL decode of a sampled cover byte (0..1) back to its value
export const DECODE_GLSL = (t) => `(${t} * ${(2 * ENC).toFixed(3)} - ${ENC.toFixed(3)})`;

// Bilinear, wrapping read of a tile's first channel at tile coordinates
// (u, v), as the GPU's linear filter does.
function sampleCover(tile, u, v) {
  const { data, size: n, channels: ch } = tile;
  const x = u * n - 0.5, y = v * n - 0.5;
  const i0 = Math.floor(x), j0 = Math.floor(y);
  const fx = x - i0, fy = y - j0;
  const M = n - 1;
  const i = (i0 & M) * ch, i1 = ((i0 + 1) & M) * ch, j = (j0 & M) * n * ch, j1 = ((j0 + 1) & M) * n * ch;
  const a = data[j + i] + (data[j + i1] - data[j + i]) * fx;
  const b = data[j1 + i] + (data[j1 + i1] - data[j1 + i]) * fx;
  return ((a + (b - a) * fy) / 255) * 2 * ENC - ENC;
}

// Cloudiness for every sky-grid direction. params: the patch (as on the
// Celestial dome tab), type (a TYPES code, or 'plain'), seed, and height (m
// above the eye).
export function layerField(params, out) {
  shapeField(params, out);
  const t = TYPES[params.type];
  if (!t || t.flat === false) return out;
  const tile = bakeType(params.type, params.seed);
  const blot = bakeBlotches(params.seed);
  const ca = Math.cos(t.angle * Math.PI / 180), sa = Math.sin(t.angle * Math.PI / 180);
  const cb = Math.cos(BLOTCH_ANGLE * Math.PI / 180), sb = Math.sin(BLOTCH_ANGLE * Math.PI / 180);
  const blotchM = t.tileM * BLOTCH_SCALE;
  for (let j = 0; j < H; j++) {
    const d = distanceToHeight(grid.el[j * W], params.height);
    for (let i = 0; i < W; i++) {
      const k = j * W + i;
      const a = grid.az[k];
      const gx = d * Math.sin(a), gy = d * Math.cos(a);
      // the pattern's own frame on the layer
      const qx = ca * gx - sa * gy, qy = sa * gx + ca * gy;
      const bx = cb * qx - sb * qy, by = sb * qx + cb * qy;
      out[k] += t.amount * sampleCover(tile, qx / t.tileM, qy / t.tileM) +
        t.blotch * sampleCover(blot, bx / blotchM, by / blotchM);
    }
  }
  return out;
}

// Area (m^2) of the layer that the covered sky cells look at. A grid row
// between elevations e0 < e1 sees the ring of the layer between horizontal
// distances d(e1) and d(e0), so each cell's share is (d0^2 - d1^2)/2 x its
// width in azimuth. cover holds each cell's covered fraction.
export function layerArea(cover, height) {
  const dphi = (2 * Math.PI) / W;
  let area = 0;
  for (let j = 0; j < H; j++) {
    const dNear = distanceToHeight(((j + 1) / H) * (Math.PI / 2), height);
    const dFar = distanceToHeight((j / H) * (Math.PI / 2), height);
    const ring = ((dFar * dFar - dNear * dNear) / 2) * dphi;
    let row = 0;
    for (let i = 0; i < W; i++) row += cover[j * W + i];
    area += row * ring;
  }
  return area;
}
