// Sky grid, area weights, cloud-shape fields and coverage maths.
//
// The celestial dome is sampled on an azimuth x elevation grid. Grid cells are
// NOT equal in area: a cell near the zenith is a thin sliver, a cell near the
// horizon is wide. Every area calculation therefore uses the true solid-angle
// weight of each cell, which is proportional to sin(top) - sin(bottom).

export const W = 720; // azimuth samples (0.5 deg each)
export const H = 180; // elevation samples (0.5 deg each), row 0 = horizon
export const N = W * H;

const DEG = Math.PI / 180;

export const grid = (() => {
  const az = new Float32Array(N);
  const el = new Float32Array(N);
  const x = new Float32Array(N); // east
  const y = new Float32Array(N); // north
  const z = new Float32Array(N); // up
  const weight = new Float64Array(N); // fraction of the dome, sums to 1
  const rowWeight = new Float64Array(H);
  for (let j = 0; j < H; j++) {
    const e0 = (j / H) * 90 * DEG;
    const e1 = ((j + 1) / H) * 90 * DEG;
    const e = ((j + 0.5) / H) * 90 * DEG;
    rowWeight[j] = Math.sin(e1) - Math.sin(e0);
    for (let i = 0; i < W; i++) {
      const a = ((i + 0.5) / W) * 360 * DEG;
      const k = j * W + i;
      az[k] = a;
      el[k] = e;
      x[k] = Math.sin(a) * Math.cos(e);
      y[k] = Math.cos(a) * Math.cos(e);
      z[k] = Math.sin(e);
      weight[k] = rowWeight[j] / W;
    }
  }
  return { az, el, x, y, z, weight, rowWeight };
})();

// Elevation (deg) below which a fraction `f` of the dome lies: sin(e) = f.
export const elevationForFraction = (f) => Math.asin(Math.min(1, Math.max(0, f))) / DEG;

// Elevations that split the dome into eight equal-area horizontal bands.
export const OCTA_BAND_ELEVATIONS = Array.from({ length: 9 }, (_, k) => elevationForFraction(k / 8));

// WMO / MANOBS style rounding of a true covered fraction to reported octas:
// any cloud at all is at least 1, any gap at all is at most 7.
export function reportedOctas(frac) {
  if (frac <= 1e-6) return 0;
  if (frac >= 1 - 1e-6) return 8;
  return Math.min(7, Math.max(1, Math.round(frac * 8)));
}

// ---------------------------------------------------------------- noise

export function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Classic 3D gradient (Perlin) noise with a seeded permutation table.
export function makeNoise3(seed) {
  const rnd = mulberry32(seed);
  const p = new Uint8Array(256);
  for (let i = 0; i < 256; i++) p[i] = i;
  for (let i = 255; i > 0; i--) {
    const j = Math.floor(rnd() * (i + 1));
    [p[i], p[j]] = [p[j], p[i]];
  }
  const perm = new Uint8Array(512);
  for (let i = 0; i < 512; i++) perm[i] = p[i & 255];
  const fade = (t) => t * t * t * (t * (t * 6 - 15) + 10);
  const lerp = (a, b, t) => a + t * (b - a);
  const grad = (h, x, y, z) => {
    const u = (h & 15) < 8 ? x : y;
    const v = (h & 15) < 4 ? y : (h & 15) === 12 || (h & 15) === 14 ? x : z;
    return ((h & 1) === 0 ? u : -u) + ((h & 2) === 0 ? v : -v);
  };
  return (x, y, z) => {
    const X = Math.floor(x) & 255, Y = Math.floor(y) & 255, Z = Math.floor(z) & 255;
    x -= Math.floor(x); y -= Math.floor(y); z -= Math.floor(z);
    const u = fade(x), v = fade(y), w = fade(z);
    const A = perm[X] + Y, AA = perm[A] + Z, AB = perm[A + 1] + Z;
    const B = perm[X + 1] + Y, BA = perm[B] + Z, BB = perm[B + 1] + Z;
    return lerp(
      lerp(lerp(grad(perm[AA], x, y, z), grad(perm[BA], x - 1, y, z), u),
           lerp(grad(perm[AB], x, y - 1, z), grad(perm[BB], x - 1, y - 1, z), u), v),
      lerp(lerp(grad(perm[AA + 1], x, y, z - 1), grad(perm[BA + 1], x - 1, y, z - 1), u),
           lerp(grad(perm[AB + 1], x, y - 1, z - 1), grad(perm[BB + 1], x - 1, y - 1, z - 1), u), v),
      w);
  };
}

// ---------------------------------------------------------------- shapes
//
// Each shape is a scalar "cloudiness" field over the dome. The cloud is the
// region where the field exceeds a threshold, and the threshold is chosen so
// the covered solid angle is exactly the requested fraction. Adding smooth
// noise to the field deforms the outline but, because the threshold is
// re-solved every time, never changes the amount of sky covered.

const wrapPi = (a) => {
  a = (a + Math.PI) % (2 * Math.PI);
  if (a < 0) a += 2 * Math.PI;
  return a - Math.PI;
};

export function computeField(params, out) {
  const { shape, azimuth, centerElevation, clumps, deform, scale, time, seed } = params;
  const { az, el, x, y, z } = grid;
  const a0 = azimuth * DEG;

  if (shape === 'slice') {
    for (let k = 0; k < N; k++) out[k] = -Math.abs(wrapPi(az[k] - a0));
  } else if (shape === 'lying') {
    // An orange lying on its side, its axis along the horizon toward azimuth
    // a0. Every slice is a lune bounded by two half great circles running from
    // one end of the axis, over the sky, to the other (180 degree arcs). The
    // dome holds 180 degrees of lune, so each 22.5 degrees is one octa: four
    // slices on each side of the zenith. Cloud fills from the horizon at a0+90.
    const px = Math.cos(a0), py = -Math.sin(a0);
    for (let k = 0; k < N; k++) out[k] = -Math.atan2(z[k], x[k] * px + y[k] * py);
  } else if (shape === 'cap') {
    const e0 = centerElevation * DEG;
    const cx = Math.sin(a0) * Math.cos(e0), cy = Math.cos(a0) * Math.cos(e0), cz = Math.sin(e0);
    for (let k = 0; k < N; k++) {
      const d = Math.max(-1, Math.min(1, x[k] * cx + y[k] * cy + z[k] * cz));
      out[k] = -Math.acos(d);
    }
  } else if (shape === 'band') {
    for (let k = 0; k < N; k++) out[k] = -el[k];
  } else if (shape === 'clumps') {
    const rnd = mulberry32(seed * 7919 + 17);
    const centers = [];
    for (let c = 0; c < clumps; c++) {
      // sin(elevation) uniform => centres spread evenly over the dome's area
      const e = Math.asin(rnd());
      const a = rnd() * 2 * Math.PI;
      centers.push([Math.sin(a) * Math.cos(e), Math.cos(a) * Math.cos(e), Math.sin(e), 0.6 + 0.8 * rnd()]);
    }
    for (let k = 0; k < N; k++) {
      let best = -Infinity;
      for (const [cx, cy, cz, s] of centers) {
        const d = Math.max(-1, Math.min(1, x[k] * cx + y[k] * cy + z[k] * cz));
        const v = -Math.acos(d) / s;
        if (v > best) best = v;
      }
      out[k] = best;
    }
  } else {
    // 'patchy': noise only
    out.fill(0);
  }

  const amp = shape === 'patchy' ? 1 : deform;
  if (amp > 0) {
    const noise = params.noise;
    const f = scale;
    const t = time;
    for (let k = 0; k < N; k++) {
      const px = x[k] * f + t * 0.31, py = y[k] * f - t * 0.17, pz = z[k] * f + t * 0.23;
      const n = noise(px, py, pz) + 0.5 * noise(px * 2.03 + 11, py * 2.03, pz * 2.03) +
        0.25 * noise(px * 4.1 - 7, py * 4.1, pz * 4.1);
      out[k] += amp * n;
    }
  }
  // Break ties (e.g. every cell in one azimuth column of a slice has the same
  // value) so the threshold can split a column and hit the area exactly.
  for (let k = 0; k < N; k++) out[k] += 1e-6 * tieJitter[k];
  return out;
}

const tieJitter = (() => {
  const rnd = mulberry32(12345);
  return Float32Array.from({ length: N }, rnd);
})();

// Weighted quantile: the threshold T such that the dome fraction where
// field > T equals `frac`. Histogram pass, then exact sort inside one bin.
export function solveThreshold(field, frac) {
  if (frac <= 0) return Infinity;
  if (frac >= 1) return -Infinity;
  const { weight } = grid;
  let lo = Infinity, hi = -Infinity;
  for (let k = 0; k < N; k++) {
    const v = field[k];
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  if (hi - lo < 1e-12) return lo - 1e-9;
  const B = 4096;
  const hist = new Float64Array(B);
  const scale = (B - 1e-6) / (hi - lo);
  for (let k = 0; k < N; k++) hist[Math.floor((field[k] - lo) * scale)] += weight[k];
  let acc = 0, b = B - 1;
  for (; b >= 0; b--) {
    if (acc + hist[b] >= frac) break;
    acc += hist[b];
  }
  if (b < 0) return -Infinity;
  const members = [];
  for (let k = 0; k < N; k++) {
    if (Math.floor((field[k] - lo) * scale) === b) members.push(k);
  }
  members.sort((p, q) => field[q] - field[p]);
  for (const k of members) {
    acc += weight[k];
    if (acc >= frac) return field[k] - 1e-12;
  }
  return lo + b / scale;
}

// Hard (0/1) cover from field and threshold; returns the covered fraction.
export function coverFromField(field, T, cover) {
  const { weight } = grid;
  let sum = 0;
  for (let k = 0; k < N; k++) {
    const c = field[k] > T ? 1 : 0;
    cover[k] = c;
    sum += c * weight[k];
  }
  return sum;
}

export function coveredFraction(cover) {
  const { weight } = grid;
  let sum = 0;
  for (let k = 0; k < N; k++) sum += cover[k] * weight[k];
  return sum;
}

// Cover fraction inside each equal-area octa band (8 values).
export function bandFractions(cover) {
  const out = new Float64Array(8);
  const tot = new Float64Array(8);
  const { rowWeight } = grid;
  for (let j = 0; j < H; j++) {
    const s = Math.sin(((j + 0.5) / H) * Math.PI / 2);
    const b = Math.min(7, Math.floor(s * 8));
    let row = 0;
    for (let i = 0; i < W; i++) row += cover[j * W + i];
    out[b] += (row / W) * rowWeight[j];
    tot[b] += rowWeight[j];
  }
  for (let b = 0; b < 8; b++) out[b] /= tot[b];
  return out;
}
