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

import { grid, W, H, shapeField, patchCentre } from './sky.js';
import { CLIMATE } from './climate.js';

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
const lerp = (a, b, t) => a + (b - a) * clamp01(t);
const mixRGB = (a, b, t) => a.map((v, i) => Math.round(lerp(v, b[i], t)));
//
// The types reported in Canadian METARs. Bases and descriptions follow MANOBS
// (8th edition) table 7-1. Sizes are real sizes on the layer, chosen so that
// at the type's usual base the elements look the width MANOBS gives at arm's
// length: cirrocumulus under one finger, altocumulus one to three, and
// stratocumulus more than three.
//
//   baseFt, baseRangeFt  usual base and MANOBS range (ft)
//   vary     how the look changes with the base (ft), as MANOBS describes
//
//   tileM    the tile's size on the layer (m)
//   featureM the pattern's typical feature (m): where pixels outgrow it, the
//            pattern gives way to its average
//   amount   how strongly the pattern breaks up the patch; large = scattered
//   blotch   how strongly the shared large blotches do
//   soft     extra blur of the cloud's edge (field units, radians)
//   opacity  how much of the sky behind shows through: thin veils are low
//   help     brief description, after MANOBS table 7-1
//   grey     the cloud's colour (sRGB) and shade, how much darker its thick
//            parts are
//   wisp     how much its thin parts let the sky through (fibrous veils)
//   angle    the pattern's direction on the layer (deg): streaks, rolls
//   topM     clouds with height: how far the tallest elements rise above the
//            base (m). Lines of sight are traced up through that height, so
//            near the horizon you see their sides and the gaps close up.
//   profile  the power in the profile: 2 domes, higher steeper-sided towers
//   anvil    how far the top spreads out
//   solid    drawn as solid shapes instead: cumulus.js
//   meshNear drawn as solid shapes near the observer, the pattern beyond:
//            grid spacing across and along the pattern, thickness (m), R0
//            (lump size), turrets, ragged (broken into small blobs), sheet
//            (lit as a sheet: only the tops in the sun)
export const TYPES = {
  CI: {
    name: 'Cirrus', etage: 'high', baseFt: 29000, baseRangeFt: [20000, 39000],
    tileM: 36000, featureM: 700, amount: 1.1, blotch: 0.45, soft: 0.1,
    opacity: 0.85, wisp: 0.9, grey: [248, 249, 252], shade: 0.04, angle: 35, res: 512,
    help: 'Detached white filaments, patches or narrow bands; fibrous (hair-like) or silky. Base 20,000–39,000 ft.',
    gen(u, v, s) {
      // detached tufts, stretched along the wind and gently bent; inside
      // them, fine parallel fibres decide how much sky shows through
      const [a, b] = warp(u, v, 2, 0.12, s);
      const tuft = fbm(a, b, 2, 12, 4, s + 5);
      const fibre = (1 - Math.abs(fbm(a, b, 3, 96, 3, s + 9) * 0.8)) ** 5;
      return [1.2 * tuft + 0.3 * fibre, clamp01(0.1 + 0.9 * fibre)];
    },
  },
  CC: {
    name: 'Cirrocumulus', etage: 'high', baseFt: 29000, baseRangeFt: [20000, 39000],
    tileM: 7000, featureM: 150, amount: 1.1, blotch: 0.6, soft: 0.02,
    opacity: 0.8, wisp: 0.45, grey: [244, 246, 249], shade: 0.06, angle: 15, res: 512,
    help: 'Thin white sheet of very small grains or ripples, without shading; elements under one finger wide. Base 20,000–39,000 ft.',
    gen(u, v, s) {
      // fine ripples (crests across the wind, about 150 m apart), wavering
      // and broken into grains along each crest, merged into thin sheets
      const [a, b] = warp(u, v, 6, 0.2, s);
      const ripple = Math.sin(2 * Math.PI * 47 * a + 1.2 * gnoise(a, b, 12, 12, s + 1));
      const grains = gnoise(a, b, 47, 24, s + 2);
      const sheet = fbm(u, v, 3, 8, 3, s + 3); // streaky patches along the wind
      const c = 0.6 * ripple + 0.3 * grains + 0.9 * sheet;
      return [c, clamp01(0.5 + 0.35 * ripple + 0.3 * grains)];
    },
  },
  CS: {
    name: 'Cirrostratus', etage: 'high', baseFt: 26000, baseRangeFt: [20000, 39000],
    tileM: 40000, featureM: 6000, amount: 0.08, blotch: 0.1, soft: 0.15,
    opacity: 0.4, wisp: 0.5, grey: [246, 248, 251], shade: 0.05, angle: 30, res: 512,
    help: 'Transparent whitish veil, fibrous or smooth; often makes a halo. Base 20,000–39,000 ft.',
    gen(u, v, s) {
      return [0.5 * fbm(u, v, 2, 2, 3, s), clamp01(0.5 + 0.6 * fbm(u, v, 3, 24, 4, s + 7))];
    },
  },
  AC: {
    name: 'Altocumulus', etage: 'mid', baseFt: 16000, baseRangeFt: [6500, 20000],
    tileM: 9000, featureM: 220, amount: 0.9, blotch: 0.5, soft: 0.015,
    opacity: 0.93, grey: [218, 222, 229], shade: 0.35, angle: 20, res: 512,
    meshNear: { across: 200, along: 290, thick: 110, R0: 250, sheet: true },
    // small elements with little shading near 16,000 ft, darker lower
    vary: (ft) => {
      const t = (ft - 7000) / 9000;
      return { grey: mixRGB([204, 208, 215], [228, 231, 236], t), shade: lerp(0.45, 0.12, t) };
    },
    help: 'White or grey patches or sheets of rounded masses or rolls, with shading; elements one to three fingers wide. Base 6,500–20,000 ft.',
    gen(u, v, s) {
      const [a, b] = warp(u, v, 10, 0.45, s);
      const e = cells(a, b, 40, 28, s, 0.92, 0.55);
      return [e + 0.2 * fbm(u, v, 64, 64, 2, s + 3), clamp01(e)];
    },
  },
  ACC: {
    name: 'Altocumulus castellanus', etage: 'mid', baseFt: 16000, baseRangeFt: [6500, 20000],
    tileM: 9000, featureM: 220, amount: 0.9, blotch: 0.5, soft: 0.015,
    opacity: 0.95, grey: [196, 201, 210], shade: 0.2, angle: 20, res: 512,
    // near the observer: altocumulus elements with turrets on the thicker
    // ones, some taller than wide
    meshNear: { across: 200, along: 400, thick: 110, R0: 250, turrets: true, sheet: true },
    vary: (ft) => ({ grey: mixRGB([190, 195, 204], [214, 218, 226], (ft - 7000) / 9000) }),
    help: 'Altocumulus with turrets, some taller than wide, on a common base, in lines. Base 6,500–20,000 ft.',
    gen(u, v, s) {
      // altocumulus elements, in lines, with knobbly turrets
      const [a, b] = warp(u, v, 10, 0.35, s);
      const e = cells(a, b, 40, 20, s, 0.8, 0.5);
      return [e + 0.25 * fbm(u, v, 96, 96, 2, s + 3), clamp01(e)];
    },
  },

  AS: {
    name: 'Altostratus', etage: 'mid', baseFt: 8000, baseRangeFt: [6500, 20000],
    tileM: 40000, featureM: 6000, amount: 0.1, blotch: 0.2, soft: 0.08,
    opacity: 0.9, wisp: 0.15, grey: [178, 184, 194], shade: 0.15, angle: 30, res: 512,
    // thick near 8,000 ft, thinner (sun shows) higher
    vary: (ft) => {
      const t = (ft - 8000) / 12000;
      return { opacity: lerp(0.95, 0.72, t), wisp: lerp(0.1, 0.35, t), grey: mixRGB([172, 178, 188], [196, 202, 212], t) };
    },
    help: 'Greyish or bluish sheet, striated or uniform; the sun shows vaguely, as through ground glass; no halo. Base 6,500–20,000 ft.',
    gen(u, v, s) {
      return [0.5 * fbm(u, v, 2, 2, 3, s), clamp01(0.5 + 0.6 * fbm(u, v, 2, 16, 4, s + 5))];
    },
  },
  NS: {
    name: 'Nimbostratus', etage: 'low', baseFt: 2500, baseRangeFt: [100, 6500],
    tileM: 30000, featureM: 5000, amount: 0.12, blotch: 0.15, soft: 0.12,
    opacity: 0.98, grey: [140, 145, 152], shade: 0.2, angle: 0, res: 512,
    // the darker, the lower
    vary: (ft) => ({ grey: mixRGB([116, 121, 128], [160, 165, 172], (ft - 1000) / 5500) }),
    help: 'Grey, often dark layer made diffuse by steady rain or snow; blots out the sun. Base near the surface to 6,500 ft.',
    gen(u, v, s) {
      return [0.5 * fbm(u, v, 2, 2, 4, s), clamp01(0.5 + 0.6 * fbm(u, v, 6, 6, 4, s + 3))];
    },
  },
  SC: {
    name: 'Stratocumulus', etage: 'low', baseFt: 3500, baseRangeFt: [500, 6500],
    tileM: 10000, featureM: 300, amount: 0.7, blotch: 0.35, soft: 0.04,
    opacity: 0.96, wisp: 0.08, grey: [214, 218, 224], shade: 0.5, angle: 20, res: 512,
    // near the observer, drawn as solid rolls (main.js sheetSpheres):
    // grid spacing across and along the rolls, and thickness (m)
    meshNear: { across: 170, along: 420, thick: 140, R0: 400, sheet: true },
    help: 'Grey or whitish patches or sheets of rounded masses or rolls, with dark parts; elements over three fingers wide. Base 500–6,500 ft.',
    gen(u, v, s) {
      // rolls in rows across the wind (about 300 m apart, a km or two long),
      // mostly merged, thick and dark in the middle, thin at the seams
      const [a, b] = warp(u, v, 4, 0.12, s);
      const e = cells(a, b, 32, 7, s, 0.98, 0.8);
      return [e + 0.2 * fbm(u, v, 32, 32, 3, s + 3), clamp01(0.05 + 1.1 * e)];
    },
  },
  ST: {
    name: 'Stratus', etage: 'low', baseFt: 800, baseRangeFt: [100, 1500],
    tileM: 8000, featureM: 700, amount: 0.12, blotch: 0.15, soft: 0.05,
    opacity: 0.92, grey: [196, 200, 206], shade: 0.15, angle: 0, res: 512,
    help: 'Grey layer with a fairly uniform base; may give drizzle; the sun’s outline is clear through it. Base near the surface to 1,500 ft.',
    gen(u, v, s) {
      return [0.7 * fbm(u, v, 3, 3, 4, s), clamp01(0.5 + 0.6 * fbm(u, v, 12, 12, 3, s + 9))];
    },
  },
  SF: {
    name: 'Stratus fractus', etage: 'low', baseFt: 800, baseRangeFt: [100, 1500],
    tileM: 2500, featureM: 90, amount: 2.0, blotch: 0.3, soft: 0.02,
    opacity: 0.85, grey: [184, 188, 195], shade: 0.25, angle: 25, res: 512,
    // near the observer: ragged clusters of small, thin blobs, drawn out
    // along the wind
    meshNear: { across: 70, along: 150, thick: 45, R0: 90, ragged: true, sheet: true },
    help: 'Ragged stratus. Base near the surface to 1,500 ft.',
    gen(u, v, s) {
      const [a, b] = warp(u, v, 8, 0.9, s);
      return [1.2 * fbm(a, b, 10, 5, 4, s), clamp01(0.5 + 0.6 * fbm(a, b, 20, 20, 3, s + 4))];
    },
  },
  CU: {
    name: 'Cumulus', etage: 'low', baseFt: 4000, baseRangeFt: [1500, 10000],
    tileM: 8000, featureM: 500, amount: 1.1, blotch: 0.4, soft: 0.01,
    opacity: 1, grey: [168, 173, 182], shade: 0.15, angle: 0, res: 512,
    // drawn as solid shapes (cumulus.js)
    solid: true,
    help: 'Detached, dense heaps with sharp outlines, bulging like a cauliflower; dark, nearly flat bases. Base usually 1,500–6,500 ft.',
    gen(u, v, s) {
      // detached heaps about 1 km across, lumpy (cauliflower) all over
      const [a, b] = warp(u, v, 8, 0.25, s);
      const e = cells(a, b, 8, 8, s, 0.6, 0.45);
      return [e + 0.22 * fbm(u, v, 48, 48, 3, s + 3), clamp01(e)];
    },
  },

  CF: {
    name: 'Cumulus fractus', etage: 'low', baseFt: 2500, baseRangeFt: [1500, 10000],
    tileM: 4000, featureM: 300, amount: 1.2, blotch: 0.3, soft: 0.02,
    opacity: 0.9, grey: [222, 226, 231], shade: 0.3, angle: 0, res: 512,
    // near the observer: ragged clusters of small puffs, sunlit like cumulus
    meshNear: { across: 110, along: 110, thick: 110, R0: 160, ragged: true, sheet: false },
    help: 'Ragged cumulus. Base usually 1,500–6,500 ft.',
    gen(u, v, s) {
      const [a, b] = warp(u, v, 8, 0.45, s);
      const e = cells(a, b, 10, 10, s, 0.55, 0.5);
      return [e + 0.35 * fbm(u, v, 20, 20, 3, s + 3), clamp01(e)];
    },
  },
  TCU: {
    name: 'Towering cumulus', etage: 'low', baseFt: 4000, baseRangeFt: [1500, 10000],
    tileM: 12000, featureM: 1200, amount: 1.2, blotch: 0.4, soft: 0.01,
    opacity: 1, grey: [150, 156, 166], shade: 0.15, angle: 0, res: 512,
    // drawn as solid shapes (cumulus.js)
    solid: true,
    help: 'Strongly sprouting cumulus of great vertical extent, cauliflower top, dark flat base. Base usually 1,500–6,500 ft.',
    gen(u, v, s) {
      const [a, b] = warp(u, v, 6, 0.2, s);
      const e = cells(a, b, 5, 5, s, 0.5, 0.42);
      return [e + 0.2 * fbm(u, v, 40, 40, 3, s + 3), clamp01(e)];
    },
  },

  CB: {
    name: 'Cumulonimbus', etage: 'low', baseFt: 3500, baseRangeFt: [1500, 10000],
    tileM: 40000, featureM: 4000, amount: 1.4, blotch: 0.3, soft: 0.01,
    opacity: 1, grey: [108, 114, 124], shade: 0.15, angle: 0, res: 512,
    // drawn as solid shapes (cumulus.js)
    solid: true,
    help: 'Heavy, dense cloud like a mountain or huge tower, its top often spread into an anvil; very dark base. Base usually 1,500–6,500 ft.',
    gen(u, v, s) {
      const [a, b] = warp(u, v, 4, 0.2, s);
      const e = cells(a, b, 2, 2, s, 0.7, 0.4);
      return [e + 0.15 * fbm(u, v, 24, 24, 3, s + 3), clamp01(e)];
    },
  },

};
// The usual base: what Canadian observers report most (the median, see
// climate.js), within the MANOBS range.
for (const [code, t] of Object.entries(TYPES)) {
  const med = CLIMATE[code]?.base[2];
  if (med) t.baseFt = Math.min(t.baseRangeFt[1], Math.max(t.baseRangeFt[0], Math.round(med / 100) * 100));
}

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
  if (t.topM) return towerField(params, t, out);
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

// Cloudiness at a point of the layer (x east, y north, metres from the
// observer), as for a direction in layerField: the patch's field toward
// that point plus the type's pattern there. For clouds drawn as meshes near
// the observer (stratocumulus): they sit where this beats the threshold.
export function cloudinessSampler(params) {
  const t = TYPES[params.type];
  const tile = bakeType(params.type, params.seed);
  const blot = bakeBlotches(params.seed);
  const ca = Math.cos(t.angle * Math.PI / 180), sa = Math.sin(t.angle * Math.PI / 180);
  const cb = Math.cos(BLOTCH_ANGLE * Math.PI / 180), sb = Math.sin(BLOTCH_ANGLE * Math.PI / 180);
  const blotchM = t.tileM * BLOTCH_SCALE;
  const [a0, e0] = patchCentre(params.azimuth, params.elevation);
  const px = Math.sin(a0) * Math.cos(e0), py = Math.cos(a0) * Math.cos(e0), pz = Math.sin(e0);
  const h0 = params.height;
  return (x, y) => {
    const d2 = x * x + y * y;
    const up = h0 - d2 / (2 * EARTH_R);
    const dot = (x * px + y * py + up * pz) / Math.sqrt(d2 + up * up);
    const r = -Math.acos(Math.max(-1, Math.min(1, dot)));
    const qx = ca * x - sa * y, qy = sa * x + ca * y;
    const bx = cb * qx - sb * qy, by = sb * qx + cb * qy;
    return r + t.amount * sampleCover(tile, qx / t.tileM, qy / t.tileM) +
      t.blotch * sampleCover(blot, bx / blotchM, by / blotchM);
  };
}

// Clouds with height. A cloud element rises above the base where its
// cloudiness c (as for a flat layer) beats the threshold T: its top is at
// height topM * sqrt((c - T) / amount), a dome over each element. A line of
// sight is in cloud if, at some height z on its way up through the clouds,
// c - profile(z) > T, where profile(z) = amount (z / topM)^2 inverts that dome
// (a higher power than 2 gives steeper sides and a rounder top: towers)
// (an anvil eases it near the top, so the top spreads). So a direction's
// cloudiness is the best of c - profile(z) along its line of sight: still one
// number per direction, cut at one threshold, so the octas stay exact. The
// GPU copy is towerMarch in main.js.
export const TOWER_STEPS = 32;
// Samples needed along a line of sight at elevation e: about one per half
// feature it crosses on its way up through the clouds (steep lines cross
// little), at least 2, at most TOWER_STEPS. main.js towerMarch matches it.
export const towerSteps = (e, t) => Math.max(2, Math.min(TOWER_STEPS,
  Math.ceil(t.topM / Math.max(1e-6, Math.tan(e)) / (0.5 * t.featureM)) + 1));
// acos to about 1e-4 rad (Abramowitz and Stegun 4.4.46, refined)
function fastAcos(x) {
  const ax = Math.min(1, Math.abs(x));
  const r = Math.sqrt(1 - ax) * (1.5707288 + ax * (-0.2121144 + ax * (0.0742610 - 0.0187293 * ax)));
  return x < 0 ? Math.PI - r : r;
}
export const towerProfile = (zz, t) => {
  const s = Math.min(1, Math.max(0, (zz - 0.7) / 0.25));
  return t.amount * zz ** (t.profile ?? 2) * (1 - (t.anvil ?? 0) * s * s * (3 - 2 * s));
};
function towerField(params, t, out) {
  const tile = bakeType(params.type, params.seed);
  const blot = bakeBlotches(params.seed);
  const ca = Math.cos(t.angle * Math.PI / 180), sa = Math.sin(t.angle * Math.PI / 180);
  const cb = Math.cos(BLOTCH_ANGLE * Math.PI / 180), sb = Math.sin(BLOTCH_ANGLE * Math.PI / 180);
  const blotchM = t.tileM * BLOTCH_SCALE;
  // the patch centre, for the patch field at each point along the way
  const [a0, e0] = patchCentre(params.azimuth, params.elevation);
  const px = Math.sin(a0) * Math.cos(e0), py = Math.cos(a0) * Math.cos(e0), pz = Math.sin(e0);
  const sinA = new Float32Array(W), cosA = new Float32Array(W);
  for (let i = 0; i < W; i++) { sinA[i] = Math.sin(grid.az[i]); cosA[i] = Math.cos(grid.az[i]); }
  const h0 = params.height;
  // Traced on every other row and column (1 degree apart; the clouds are
  // drawn per pixel on the GPU), and filled in between.
  for (let j = 0; j < H; j += 2) {
    const e = grid.el[j * W];
    for (let i = 0; i < W; i += 2) out[j * W + i] = -Infinity;
    const n = towerSteps(e, t);
    for (let k = 0; k < n; k++) {
      const zz = k / (n - 1);
      const d = distanceToHeight(e, h0 + zz * t.topM);
      const up = h0 - (d * d) / (2 * EARTH_R); // the base under that point
      const len = Math.sqrt(d * d + up * up);
      const pen = towerProfile(zz, t);
      for (let i = 0; i < W; i += 2) {
        const gx = d * sinA[i], gy = d * cosA[i];
        const r = -fastAcos((gx * px + gy * py + up * pz) / len);
        const qx = ca * gx - sa * gy, qy = sa * gx + ca * gy;
        const bx = cb * qx - sb * qy, by = sb * qx + cb * qy;
        const v = r + t.amount * sampleCover(tile, qx / t.tileM, qy / t.tileM) +
          t.blotch * sampleCover(blot, bx / blotchM, by / blotchM) - pen;
        const kk = j * W + i;
        if (v > out[kk]) out[kk] = v;
      }
    }
  }
  for (let j = 0; j < H; j += 2) {
    const row = j * W;
    for (let i = 1; i < W; i += 2) out[row + i] = 0.5 * (out[row + i - 1] + out[row + ((i + 1) % W)]);
  }
  for (let j = 1; j < H; j += 2) {
    const row = j * W, below = row - W, above = j + 1 < H ? row + W : below;
    for (let i = 0; i < W; i++) out[row + i] = 0.5 * (out[below + i] + out[above + i]);
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
