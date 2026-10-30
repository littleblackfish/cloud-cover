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
// The GPU copy of all this lives in main.js (LAYER_GLSL) and must match.

import { grid, W, H, shapeField, makeNoise3 } from './sky.js';

export const EARTH_R = 6371000;
export const P_OCC = 0.85; // fraction of lattice cells holding a cumulus element
export const ELEMENT_R = 0.55; // element radius, in lattice spacings
export const LAYER_AMOUNT = 2.0; // how strongly the texture breaks up the shape

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

// Cumuli: rounded elements on a lattice of the given spacing, one per
// occupied cell at a random spot in it. 1 at an element's centre, 0 at its
// edge, falling to -1 between elements.
export function cumuliTexture(x, y, spacing, seed) {
  const ix = Math.floor(x / spacing), iy = Math.floor(y / spacing);
  const r = ELEMENT_R * spacing;
  let best = -1;
  for (let ox = -1; ox <= 1; ox++) {
    for (let oy = -1; oy <= 1; oy++) {
      const [occ, jx, jy] = hash3(ix + ox, iy + oy, seed);
      if (occ >= P_OCC) continue;
      const cx = (ix + ox + jx) * spacing, cy = (iy + oy + jy) * spacing;
      const v = 1 - Math.hypot(cx - x, cy - y) / r;
      if (v > best) best = v;
    }
  }
  return best;
}

// Strati: smooth random sheet with features about `size` across, about -1..1.
const noiseCache = new Map();
export function noiseFor(seed) {
  if (!noiseCache.has(seed)) noiseCache.set(seed, makeNoise3(seed));
  return noiseCache.get(seed);
}
export function stratiTexture(x, y, size, seed) {
  const noise = noiseFor(seed);
  const f = 1 / size;
  return noise(x * f, y * f, 0.37) + 0.45 * noise(x * f * 2.3 + 17, y * f * 2.3, 1.7);
}

// Cloudiness for every sky-grid direction. params: shape fields as on the
// Octas tab, plus modifier, size (m), seed, and height (m above the eye).
export function layerField(params, out) {
  shapeField(params, out);
  const { modifier, size, seed, height } = params;
  if (modifier === 'none') return out;
  const tex = modifier === 'cumuli' ? cumuliTexture : stratiTexture;
  for (let j = 0; j < H; j++) {
    const d = distanceToHeight(grid.el[j * W], height);
    for (let i = 0; i < W; i++) {
      const k = j * W + i;
      const a = grid.az[k];
      out[k] += LAYER_AMOUNT * tex(d * Math.sin(a), d * Math.cos(a), size, seed);
    }
  }
  return out;
}
