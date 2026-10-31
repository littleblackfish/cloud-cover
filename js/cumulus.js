// Cumuliform clouds as solid shapes: each cloud is a cluster of spheres
// (slightly flattened), rising from a flat base. main.js draws them as
// meshes; here they are laid out, and the amount of sky they cover is worked
// out exactly.
//
// Clouds come and go one by one: each has its own full size (sizes vary a
// lot), and they join in order outward from the patch (azimuth,
// elevation), with a ragged edge. The amount is set by one number, the
// reach: clouds whose turn the reach has passed are full size, the next few
// are growing from their anchor (the middle of their base), the rest are
// not there. For each sky direction there is a smallest reach at which a
// cloud covers it (a quadratic for each sphere: see touchScale). So the sky
// is one number per direction, and the reach that covers exactly the chosen
// octas is a threshold on it, as for the flat clouds.
//
// Coordinates: metres from the observer's eye, east, north, up. Farther
// clouds sit lower by d^2 / 2R (the Earth's curve).

import { grid, W, H, patchCentre } from './sky.js';

const EARTH_R = 6371000;
const DEG = Math.PI / 180;

function rng(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Cloud shapes, in units of the cloud's size R0, from the anchor: spheres
// [east, north, up, radius, squash] (squash < 1 flattens it vertically).
// Listed most important first (the body before its lumps), so a far cloud can
// be drawn with only the first few: see LOD.
const SHAPES = {
  // a heap a little wider than tall: a core, a ring of lobes round the
  // base, smaller ones above, a few cauliflower knobs on top
  CU(r) {
    const s = [[0, 0, 0.55, 0.62, 0.85]];
    const ring = 5 + Math.floor(r() * 3);
    for (let i = 0; i < ring; i++) {
      const a = (i / ring + 0.15 * r()) * 2 * Math.PI, o = 0.5 + 0.25 * r();
      s.push([o * Math.cos(a), o * Math.sin(a), 0.32, 0.38 + 0.12 * r(), 0.78]);
    }
    const up = 3 + Math.floor(r() * 3);
    for (let i = 0; i < up; i++) {
      const a = r() * 2 * Math.PI, o = 0.2 + 0.25 * r();
      s.push([o * Math.cos(a), o * Math.sin(a), 0.8 + 0.15 * r(), 0.3 + 0.12 * r(), 0.95]);
    }
    const top = 1 + Math.floor(r() * 3);
    for (let i = 0; i < top; i++) {
      const a = r() * 2 * Math.PI, o = 0.2 * r();
      s.push([o * Math.cos(a), o * Math.sin(a), 1.1 + 0.15 * r(), 0.22 + 0.1 * r(), 1]);
    }
    return s;
  },
  // a column several times taller than wide, lumpy all the way up, with a
  // bulging cauliflower top
  TCU(r) {
    const cores = [], s = [];
    const levels = 5 + Math.floor(r() * 2);
    let x = 0, y = 0;
    for (let l = 0; l < levels; l++) {
      const z = 0.45 + l * 0.6;
      x += 0.1 * (r() - 0.5); y += 0.1 * (r() - 0.5);
      cores.push([x, y, z, 0.6 - 0.02 * l, 0.95]);
      const lumps = 4 + Math.floor(r() * 3);
      for (let i = 0; i < lumps; i++) {
        const a = (i / lumps + 0.2 * r()) * 2 * Math.PI, o = 0.32 + 0.15 * r();
        s.push([x + o * Math.cos(a), y + o * Math.sin(a), z + 0.3 * (r() - 0.5), 0.3 + 0.14 * r(), 0.95]);
      }
    }
    const zt = 0.45 + levels * 0.6;
    for (let i = 0; i < 5; i++) {
      const a = r() * 2 * Math.PI, o = 0.3 * r();
      s.push([x + o * Math.cos(a), y + o * Math.sin(a), zt - 0.2 + 0.25 * r(), 0.32 + 0.12 * r(), 1]);
    }
    return cores.concat(s);
  },
  // a mountain of cloud whose top spreads downwind into a flat anvil
  CB(r) {
    const cores = [], anvil = [], s = [];
    for (let l = 0; l < 5; l++) {
      const z = 0.5 + l * 0.65;
      cores.push([0.05 * l, 0, z, 0.85 - 0.04 * l, 0.9]);
      for (let i = 0; i < 4; i++) {
        const a = r() * 2 * Math.PI, o = 0.55 + 0.2 * r();
        s.push([0.05 * l + o * Math.cos(a), o * Math.sin(a), z + 0.2 * (r() - 0.5), 0.4 + 0.15 * r(), 0.9]);
      }
    }
    // the anvil: overlapping flat lobes from the top of the column out
    // downwind, thickest where they join it
    for (let i = 0; i < 9; i++) {
      const o = 2.4 * (i / 8);
      const side = 0.5 * (r() - 0.5) * (1 + o);
      anvil.push([0.25 + o, side, 3.35 + 0.1 * r() - 0.05 * o, 0.9 + 0.35 * r() + 0.15 * o, 0.38 - 0.06 * o]);
    }
    return cores.concat(anvil, s);
  },
};

// R0: the cloud's size (m); spacing between clouds (m); topR: height of the
// top in R0. LOD: how many of a cloud's spheres to draw when it is middling
// far (a degree or two across) and far (under a degree).
export const SOLID = {
  CU: { R0: 600, spacing: 2800, topR: 1.35, lod: [4, 1] },
  TCU: { R0: 900, spacing: 6000, topR: 4.8, lod: [10, 5] },
  CB: { R0: 3200, spacing: 30000, topR: 3.6, lod: [14, 9] },
};
// Clouds join one after another as the amount grows: each takes BAND
// (radians of "reach", see order) to grow from nothing to full size. Once
// all have joined (REACH_ALL), all of them grow further, up to GROW_MAX
// times their size, for the larger amounts (Canadian observers report CU at
// 6 or 7 octas in about one report in six).
const BAND = 0.12;
const REACH_ALL = Math.PI * 1.6 + BAND;
const GROW_MAX = 3;
const GROW_PER = 1; // size added per unit of reach past REACH_ALL
export const REACH_MAX = REACH_ALL + (GROW_MAX - 1) / GROW_PER;
// a cloud's scale (of its R0) at a reach, given its order and full size
const scaleAt = (reach, ord, size) => (reach <= REACH_ALL
  ? Math.min(1, Math.max(0, (reach - ord) / BAND)) * size
  : size * (1 + (reach - REACH_ALL) * GROW_PER));
export const UNREACHED = -100; // field value where no cloud ever reaches

// The clouds over the observer: anchors on a jittered grid out to as far as
// a cloud top can be seen, each with its size and shape. Fixed for a type,
// seed and base height.
const layoutCache = new Map();
function layout(type, seed, height) {
  const key = `${type}:${seed}:${Math.round(height)}`;
  if (layoutCache.has(key)) return layoutCache.get(key);
  const sp = SOLID[type];
  const top = height + sp.topR * sp.R0 * 1.7;
  const reach = Math.min(220000, Math.sqrt(2 * EARTH_R * top) + 3000);
  const n = Math.ceil(reach / sp.spacing);
  const clouds = [];
  const r = rng(seed * 7919 + type.length * 104729);
  for (let gy = -n; gy <= n; gy++) {
    for (let gx = -n; gx <= n; gx++) {
      const e = (gx + 0.9 * (r() - 0.5)) * sp.spacing;
      const no = (gy + 0.9 * (r() - 0.5)) * sp.spacing;
      // sizes vary a lot: many small clouds, a few big ones
      const size = 0.6 + 1.4 * r() ** 2.2;
      const shapeSeed = Math.floor(r() * 1e9);
      const jitter = r();
      const d = Math.hypot(e, no);
      if (d > reach) continue;
      const up = height - (d * d) / (2 * EARTH_R);
      // a far cloud is only a few degrees across, or less: draw it with
      // its main spheres only
      const ang = (sp.R0 * size * 2) / Math.max(1, Math.hypot(d, up));
      const all = SHAPES[type](rng(shapeSeed));
      const keep = Math.min(all.length, ang > 3 * DEG ? all.length : ang > 0.8 * DEG ? sp.lod[0] : sp.lod[1]);
      const spheres = all.slice(0, keep);
      // a lone sphere stands for the whole heap
      if (keep === 1) spheres[0] = [0, 0, spheres[0][2], spheres[0][3] * 1.25, spheres[0][4]];
      clouds.push({ e, n: no, up, d, size, jitter, spheres, detail: keep === all.length });
    }
  }
  const out = { clouds, sp };
  layoutCache.set(key, out);
  if (layoutCache.size > 12) layoutCache.delete(layoutCache.keys().next().value);
  return out;
}

// The order in which clouds join: by their angle from the patch centre,
// stretched or shrunk at random so the group's edge is ragged. A cloud
// starts to grow when the reach passes its order, and is full size BAND
// later; the reach is what the octas set.
function order(L, params) {
  const [a0, e0] = patchCentre(params.azimuth, params.elevation);
  const px = Math.sin(a0) * Math.cos(e0), py = Math.cos(a0) * Math.cos(e0), pz = Math.sin(e0);
  return L.clouds.map((c) => {
    const len = Math.hypot(c.d, c.up);
    const ang = Math.acos(Math.max(-1, Math.min(1, (c.e * px + c.n * py + c.up * pz) / len)));
    return ang * (0.6 + 0.8 * c.jitter);
  });
}

// Smallest scale sigma at which a sphere, centred A + sigma v with radius
// sigma rho and squashed vertically by k, touches the line of sight through
// the eye along u: the distance from the line to the centre equals the
// radius. In coordinates stretched vertically by 1/k the sphere is round,
// and that distance is a quadratic in sigma; the root wanted is
// (-b - sqrt(b^2 - 4ac)) / 2a whether the quadratic opens up or down.
// Returns Infinity if it never touches.
function touchScale(Ax, Ay, Az, vx, vy, vz, rho, ux, uy, uz) {
  const un = Math.sqrt(ux * ux + uy * uy + uz * uz);
  const hx = ux / un, hy = uy / un, hz = uz / un;
  const Av = Ax * hx + Ay * hy + Az * hz;
  const vv = vx * hx + vy * hy + vz * hz;
  const a = vx * vx + vy * vy + vz * vz - vv * vv - rho * rho;
  const b = 2 * (Ax * vx + Ay * vy + Az * vz - Av * vv);
  const c = Ax * Ax + Ay * Ay + Az * Az - Av * Av;
  if (Math.abs(a) < 1e-12) return b < 0 ? -c / b : Infinity;
  const D = b * b - 4 * a * c;
  if (D < 0) return Infinity;
  const s = (-b - Math.sqrt(D)) / (2 * a);
  return s > 0 ? s : Infinity;
}

// Fills out[k] with minus the reach at which each sky direction is first
// covered (minus, so that the cloud is where out > threshold, as for the
// flat types): a cloud at full size `size` covers a direction once it has
// grown to scale sigma (touchScale), which it reaches at reach
// order + BAND sigma / size. Directions no cloud ever covers get UNREACHED.
// Returns what the meshes need.
export function solidField(params, out) {
  const L = layout(params.type, params.seed, params.height);
  const ord = order(L, params);
  out.fill(Infinity);
  const { x: gx, y: gy, z: gz } = grid;
  const R0 = L.sp.R0;
  for (let ci = 0; ci < L.clouds.length; ci++) {
    const c = L.clouds[ci];
    const full = c.size, oc = ord[ci];
    for (const [sx, sy, sz, sr, sk] of c.spheres) {
      // in coordinates stretched vertically by 1/k
      const k = sk;
      const Ax = c.e, Ay = c.n, Az = c.up / k;
      const vx = sx * R0, vy = sy * R0, vz = (sz * R0) / k, rho = sr * R0;
      // the directions it can reach at its largest
      const big = full * GROW_MAX;
      const cx = Ax + big * vx, cy = Ay + big * vy, cz = (Az + big * vz) * k;
      const dist = Math.sqrt(cx * cx + cy * cy + cz * cz);
      const rad = big * rho; // the largest semi-axis
      if (rad >= dist) continue; // around the eye: not possible above the base
      const beta = Math.asin(rad / dist);
      const elc = Math.asin(cz / dist);
      const el0 = elc - beta, el1 = elc + beta;
      if (el1 <= 0) continue;
      const j0 = Math.max(0, Math.floor((el0 / (Math.PI / 2)) * H));
      const j1 = Math.min(H - 1, Math.floor((el1 / (Math.PI / 2)) * H));
      const azc = Math.atan2(cx, cy);
      const maxEl = Math.max(Math.abs(el0), Math.abs(el1));
      const half = maxEl >= Math.PI / 2 - 1e-3 ? Math.PI
        : Math.min(Math.PI, Math.asin(Math.min(1, Math.sin(beta) / Math.cos(maxEl))) + 1e-3);
      const span = half >= Math.PI ? W : Math.min(W, Math.ceil((2 * half) / (2 * Math.PI) * W) + 2);
      const i0 = Math.floor(((azc - half) / (2 * Math.PI)) * W);
      // every other row and column (1 degree apart); filled in below
      for (let j = j0 & ~1; j <= j1; j += 2) {
        for (let ii = 0; ii < span; ii++) {
          const i = ((i0 + ii) % W + W) % W;
          if (i & 1) continue;
          const kk = j * W + i;
          const sig = touchScale(Ax, Ay, Az, vx, vy, vz, rho, gx[kk], gy[kk], gz[kk] / k);
          if (sig > big) continue;
          // the reach at which the cloud has grown to sig (inverse of scaleAt)
          const reach = sig <= full ? oc + (BAND * sig) / full : REACH_ALL + (sig / full - 1) / GROW_PER;
          if (reach < out[kk]) out[kk] = reach;
        }
      }
    }
  }
  for (let kk = 0; kk < out.length; kk++) out[kk] = out[kk] === Infinity ? UNREACHED : -out[kk];
  for (let j = 0; j < H; j += 2) {
    const row = j * W;
    for (let i = 1; i < W; i += 2) out[row + i] = 0.5 * (out[row + i - 1] + out[row + ((i + 1) % W)]);
  }
  for (let j = 1; j < H; j += 2) {
    const row = j * W, below = row - W, above = j + 1 < H ? row + W : below;
    for (let i = 0; i < W; i++) out[row + i] = 0.5 * (out[below + i] + out[above + i]);
  }
  return { L, ord };
}

// The spheres to draw at a given reach: centre (east, north, up), radii
// (horizontal, vertical), base height and the cloud's top, all in metres;
// detail: whether the cloud is drawn in full (near) or simplified (far).
export function solidSpheres(state, reach) {
  const { L, ord } = state;
  const R0 = L.sp.R0;
  const list = [];
  for (let ci = 0; ci < L.clouds.length; ci++) {
    const c = L.clouds[ci];
    const sig = scaleAt(reach, ord[ci], c.size);
    if (sig * R0 < 5) continue;
    for (const [sx, sy, sz, sr, sk] of c.spheres) {
      list.push({
        e: c.e + sig * sx * R0, n: c.n + sig * sy * R0, up: c.up + sig * sz * R0,
        rh: sig * sr * R0, rv: sig * sr * R0 * sk,
        base: c.up, top: c.up + sig * L.sp.topR * R0, detail: c.detail,
      });
    }
  }
  return list;
}
