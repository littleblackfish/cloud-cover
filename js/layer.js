// A flat cloud layer seen from one point on the ground.
//
// The layer holds one of the dome shapes (a horizontal slice, a vertical
// slice or a round patch), optionally broken up by a texture that lives on the
// layer itself: cumuli (separate rounded elements) or strati (a sheet with
// gaps). Each shape is a region of the layer: a horizontal slice is a wedge
// spreading out from overhead, a vertical slice a straight strip, a round
// patch a disc.
//
// Cloudiness for a sky direction = the shape's field for that direction (as on
// the Octas tab) + amount x the texture where that line of sight meets the
// layer. The cloud is where cloudiness exceeds a threshold chosen so the dome
// is covered by exactly the requested amount. Because the texture sits on the
// layer, perspective squeezes it toward the horizon.
//
// Lines of sight follow the Earth's curve: a ray leaving at elevation e has
// height z(d) = d tan(e) + d^2 / (2R) after horizontal distance d, so even a
// ray aimed exactly at the horizon reaches the layer (about sqrt(2Rh) away).
// The GPU copy of all this lives in main.js (LAYER_GLSL) and must match; the
// textures themselves are shared, baked once below.

import { grid, W, H, shapeField } from './sky.js';

export const EARTH_R = 6371000;
export const P_OCC = 0.85; // fraction of lattice cells holding a cumulus element
export const ELEMENT_R = 0.55; // element radius, in lattice spacings

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

// The textures are baked once into a tile that repeats across the layer:
// TILE x TILE texels holding CELLS x CELLS features. The GPU samples the
// same bytes (with mipmaps, so a pixel looking far along the layer sees the
// texture averaged instead of flickering), and the CPU reads them back with
// the same bilinear filter for the octa numbers.
export const TILE = 1024;
export const CELLS = 16;
const ENC = 1.5; // byte b stands for (b / 255) * 2 ENC - ENC

// Cumuli: rounded elements on a lattice, one per occupied cell at a random
// spot in it. 1 at an element's centre, 0 at its edge, falling to -1 between
// elements. x, y in lattice spacings.
function cumuliAt(x, y, seed) {
  const ix = Math.floor(x), iy = Math.floor(y);
  let best = -1;
  for (let ox = -1; ox <= 1; ox++) {
    for (let oy = -1; oy <= 1; oy++) {
      const cx0 = ix + ox, cy0 = iy + oy;
      const [occ, jx, jy] = hash3(((cx0 % CELLS) + CELLS) % CELLS, ((cy0 % CELLS) + CELLS) % CELLS, seed);
      if (occ >= P_OCC) continue;
      const v = 1 - Math.hypot(cx0 + jx - x, cy0 + jy - y) / ELEMENT_R;
      if (v > best) best = v;
    }
  }
  return Math.max(-1, best);
}

// Strati: smooth gradient noise that repeats every `period` lattice units.
function tileNoise(x, y, period, seed) {
  const ix = Math.floor(x), iy = Math.floor(y);
  const fx = x - ix, fy = y - iy;
  const g = (cx, cy) => {
    const [a] = hash3(((cx % period) + period) % period, ((cy % period) + period) % period, seed);
    const t = a * 2 * Math.PI;
    return (cx0, cy0) => Math.cos(t) * cx0 + Math.sin(t) * cy0;
  };
  const fade = (t) => t * t * t * (t * (t * 6 - 15) + 10);
  const u = fade(fx), v = fade(fy);
  const n00 = g(ix, iy)(fx, fy), n10 = g(ix + 1, iy)(fx - 1, fy);
  const n01 = g(ix, iy + 1)(fx, fy - 1), n11 = g(ix + 1, iy + 1)(fx - 1, fy - 1);
  const a = n00 + (n10 - n00) * u, b = n01 + (n11 - n01) * u;
  return 1.4 * (a + (b - a) * v);
}
function stratiAt(x, y, seed) {
  return tileNoise(x, y, CELLS, seed) + 0.45 * tileNoise(2 * x + 3, 2 * y, 2 * CELLS, seed + 7);
}

const bakeCache = new Map();
export function bakeTexture(kind, seed) {
  const key = `${kind}:${seed}`;
  if (bakeCache.has(key)) return bakeCache.get(key);
  const f = kind === 'cumuli' ? cumuliAt : stratiAt;
  const data = new Uint8Array(TILE * TILE);
  const per = CELLS / TILE;
  for (let j = 0; j < TILE; j++) {
    for (let i = 0; i < TILE; i++) {
      const v = f((i + 0.5) * per, (j + 0.5) * per, seed);
      data[j * TILE + i] = Math.round(((Math.max(-ENC, Math.min(ENC, v)) + ENC) / (2 * ENC)) * 255);
    }
  }
  bakeCache.set(key, data);
  return data;
}
// GLSL decode of a sampled byte (0..1) back to texture value
export const DECODE_GLSL = `(t * ${(2 * ENC).toFixed(3)} - ${ENC.toFixed(3)})`;

// Bilinear, wrapping read of a baked tile, as the GPU's linear filter does.
// x, y in metres; size: the lattice spacing in metres.
export function sampleTexture(data, x, y, size) {
  const s = TILE / CELLS / size;
  const u = x * s - 0.5, v = y * s - 0.5;
  const i0 = Math.floor(u), j0 = Math.floor(v);
  const fu = u - i0, fv = v - j0;
  const M = TILE - 1;
  const i = i0 & M, i1 = (i0 + 1) & M, j = (j0 & M) * TILE, j1 = ((j0 + 1) & M) * TILE;
  const a = data[j + i] + (data[j + i1] - data[j + i]) * fu;
  const b = data[j1 + i] + (data[j1 + i1] - data[j1 + i]) * fu;
  return ((a + (b - a) * fv) / 255) * 2 * ENC - ENC;
}

// Cloudiness for every sky-grid direction. params: shape fields as on the
// Octas tab, plus modifier, amount (how strongly the texture breaks up the
// shape), size (m), seed, and height (m above the eye).
export function layerField(params, out) {
  shapeField(params, out);
  const { modifier, amount, size, seed, height } = params;
  if (modifier === 'none' || !(amount > 0)) return out;
  const data = bakeTexture(modifier, seed);
  for (let j = 0; j < H; j++) {
    const d = distanceToHeight(grid.el[j * W], height);
    for (let i = 0; i < W; i++) {
      const k = j * W + i;
      const a = grid.az[k];
      out[k] += amount * sampleTexture(data, d * Math.sin(a), d * Math.cos(a), size);
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
