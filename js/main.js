import * as THREE from 'three';
import { OrbitControls } from '../vendor/OrbitControls.js';
import {
  W, H, N, grid, computeField, solveThreshold, coverFromField,
  reportedOctas, makeNoise3, mulberry32, vertSliceFrame, vertSliceCentre, patchCentre,
} from './sky.js';
import { layerField, EARTH_R, P_OCC, ELEMENT_R } from './layer.js';

const DEG = Math.PI / 180;
const FT = 0.3048; // metres per foot
const $ = (s) => document.querySelector(s);
const $$ = (s) => [...document.querySelectorAll(s)];

// ================================================================ state

// shape: 'lying' (slice) or 'cap' (patch), placed by azimuth and elevation
// (see sky.js shapeField). form: one signed number for how the shape is broken up:
// negative = cumuli, positive = strati, 0 = a plain shape; its size is the
// strength. modifier and amount are derived from it (see setForm).
const shapeParams = {
  octas: 3, shape: 'lying', azimuth: 90, elevation: 25,
  form: 0.3, modifier: 'strati', amount: 0.3, time: 0, seed: 1, animate: true,
};
function setForm(p, v) {
  p.form = v;
  p.modifier = v < 0 ? 'cumuli' : v > 0 ? 'strati' : 'none';
  p.amount = Math.abs(v);
}
// The cloud layer uses the same shapes, broken up by a texture on the layer
// itself; size is the texture's scale in metres (element spacing or patch
// size), typical for the étage.
const layerParams = {
  etage: 'low', shape: 'cap', azimuth: 90, elevation: 25, // round patch only
  form: 0.3, modifier: 'strati', amount: 0.3, octas: 3,
  base: 4000 * FT, size: 2000, seed: 1, showClouds: true,
};
let quizParams = null;
const quiz = { answered: false, right: 0, total: 0, streak: 0 };

let mode = 'octas'; // 'octas' | 'layer' | 'quiz'
let view = 'outside';
let guides = 'lying';
let dirty = true;
let needsRender = true; // draw the next frame (see frame())

// ================================================================ transitions
//
// Nothing should jump: camera moves, view changes, shape changes and the like
// all run as short eased animations. animate(key, ms, step, done) calls
// step(e) every frame with e easing from 0 to 1; a new animation with the
// same key replaces the old one.

const REDUCED_MOTION = matchMedia('(prefers-reduced-motion: reduce)').matches;
const easeInOut = (t) => (t < 0.5 ? 2 * t * t : 1 - (-2 * t + 2) ** 2 / 2);
const tweens = new Map();
function animate(key, ms, step, done) {
  const tw = { start: performance.now(), ms: REDUCED_MOTION ? 0 : ms, step, done };
  tweens.set(key, tw);
  if (tw.ms === 0) runTween(key, tw, tw.start);
}
function runTween(key, tw, now) {
  const t = tw.ms > 0 ? Math.min(1, (now - tw.start) / tw.ms) : 1;
  tw.step(easeInOut(t));
  if (t >= 1) {
    if (tweens.get(key) === tw) tweens.delete(key);
    tw.done?.();
  }
}
function runTweens(now) {
  for (const [key, tw] of [...tweens]) runTween(key, tw, now);
}

const field = new Float32Array(N);
const cover = new Float32Array(N);
let threshold = 0;
let coveredFrac = 0;
const noiseCache = new Map();
const noiseFor = (seed) => {
  if (!noiseCache.has(seed)) noiseCache.set(seed, makeNoise3(seed));
  return noiseCache.get(seed);
};

// ================================================================ three.js scene

const canvas = $('#view');
// The scene spans the observer's 1.7 m to hundreds of km, so depth needs a log buffer.
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, logarithmicDepthBuffer: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0f1622);
const camera = new THREE.PerspectiveCamera(45, 1, 1e-4, 1e4);

scene.add(new THREE.HemisphereLight(0xffffff, 0x556070, 1.1));
const sun = new THREE.DirectionalLight(0xffffff, 1.2);
sun.position.set(2, 3, 1);
scene.add(sun);

// sky (east, north, up) -> three (x, y, z) with north = -z
const toScene = (e, n, u) => new THREE.Vector3(e, u, -n);
const dirFromAzEl = (a, e, r = 1) =>
  toScene(Math.sin(a) * Math.cos(e) * r, Math.cos(a) * Math.cos(e) * r, Math.sin(e) * r);

// Grid layout: one texel per grid cell, row 0 = horizon.
const texImage = new ImageData(W, H);

function domeGeometry(segA = 192, segE = 64) {
  const pos = [], uv = [], idx = [];
  for (let j = 0; j <= segE; j++) {
    const e = (j / segE) * 90 * DEG;
    for (let i = 0; i <= segA; i++) {
      const a = (i / segA) * 2 * Math.PI;
      const v = dirFromAzEl(a, e);
      pos.push(v.x, v.y, v.z);
      uv.push(i / segA, j / segE);
    }
  }
  for (let j = 0; j < segE; j++) {
    for (let i = 0; i < segA; i++) {
      const a = j * (segA + 1) + i, b = a + 1, c = a + segA + 1, d = c + 1;
      idx.push(a, c, b, b, c, d);
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  g.setAttribute('uv', new THREE.Float32BufferAttribute(uv, 2));
  g.setIndex(idx);
  return g;
}
// ---------------------------------------------------------------- layer pattern
//
// The cloud layer is drawn per pixel on the GPU, on the dome and on the 3D
// layer alike, from the same fields the CPU uses for the octa numbers (see
// layer.js; the hash and noise below copy layer.js and sky.js exactly).
// Cloudiness = shape field for the line of sight + amount x the texture where
// that line of sight meets the layer; cloud where it exceeds layerT.
const permTexture = new THREE.DataTexture(new Uint8Array(512), 512, 1, THREE.RedFormat, THREE.UnsignedByteType);
let permSeed = null;
const layerUniforms = {
  shapeKind: { value: 0 }, // 0 horizontal slice, 1 vertical slice, 2 round patch
  modKind: { value: 1 }, // 0 none, 1 cumuli, 2 strati
  seed: { value: 1 },
  shapeAz: { value: 0 },
  sliceCentre: { value: 0 },
  capDir: { value: new THREE.Vector3(0, 0, 1) }, // east, north, up
  texSize: { value: 1500 },
  amount: { value: 0.7 },
  layerT: { value: 0 },
  layerHeight: { value: 1200 }, // cloud base above the eye, m
  layerFade: { value: 1 }, // fades the clouds out and in when the layer is swapped
  perm: { value: permTexture },
};
const LAYER_GLSL = `
  uniform int shapeKind, modKind, seed;
  uniform float shapeAz, sliceCentre, texSize, amount, layerT, layerHeight, layerFade;
  uniform vec3 capDir;
  uniform sampler2D perm;
  const float LPI = 3.141592653589793;

  // sky.js shapeField, for a unit sky direction s = (east, north, up)
  float region(vec3 s) {
    if (shapeKind == 0) {
      float d = mod(atan(s.x, s.y) - shapeAz + LPI, 2.0 * LPI) - LPI;
      return -abs(d);
    }
    if (shapeKind == 1) {
      vec2 p = vec2(cos(shapeAz), -sin(shapeAz));
      return -abs(atan(s.z, dot(s.xy, p)) - sliceCentre);
    }
    return -acos(clamp(dot(s, capDir), -1.0, 1.0));
  }

  // layer.js hash3
  uvec3 hash3(int ix, int iy) {
    uint h = (uint(ix) * 0x27d4eb2du) ^ (uint(iy) * 0x165667b1u) ^ (uint(seed) * 0x9e3779b1u);
    h = (h ^ (h >> 15u)) * 0x85ebca6bu;
    h = (h ^ (h >> 13u)) * 0xc2b2ae35u;
    h ^= h >> 16u;
    uint g = (h ^ 0x5bd1e995u) * 0x01000193u;
    g ^= g >> 15u;
    uint q = (g ^ 0x2545f491u) * 0x9e3779b1u;
    q ^= q >> 13u;
    return uvec3(h, g, q);
  }
  // layer.js cumuliTexture
  float cumuli(vec2 p) {
    int ix = int(floor(p.x / texSize)), iy = int(floor(p.y / texSize));
    float r = ${ELEMENT_R.toFixed(3)} * texSize, best = -1.0;
    for (int ox = -1; ox <= 1; ox++) for (int oy = -1; oy <= 1; oy++) {
      uvec3 h = hash3(ix + ox, iy + oy);
      if (float(h.x) / 4294967296.0 >= ${P_OCC.toFixed(3)}) continue;
      vec2 c = (vec2(float(ix + ox), float(iy + oy)) + vec2(h.yz) / 4294967296.0) * texSize;
      best = max(best, 1.0 - length(c - p) / r);
    }
    return best;
  }

  // sky.js makeNoise3, and layer.js stratiTexture
  int P(int i) { return int(texelFetch(perm, ivec2(i, 0), 0).r * 255.0 + 0.5); }
  float grad(int h, float x, float y, float z) {
    h &= 15;
    float u = h < 8 ? x : y;
    float v = h < 4 ? y : (h == 12 || h == 14 ? x : z);
    return ((h & 1) == 0 ? u : -u) + ((h & 2) == 0 ? v : -v);
  }
  float fade(float t) { return t * t * t * (t * (t * 6.0 - 15.0) + 10.0); }
  float noise(vec3 p) {
    int X = int(floor(p.x)) & 255, Y = int(floor(p.y)) & 255, Z = int(floor(p.z)) & 255;
    vec3 f = p - floor(p);
    float u = fade(f.x), v = fade(f.y), w = fade(f.z);
    int A = P(X) + Y, AA = P(A) + Z, AB = P(A + 1) + Z;
    int B = P(X + 1) + Y, BA = P(B) + Z, BB = P(B + 1) + Z;
    float x = f.x, y = f.y, z = f.z;
    return mix(
      mix(mix(grad(P(AA), x, y, z), grad(P(BA), x - 1.0, y, z), u),
          mix(grad(P(AB), x, y - 1.0, z), grad(P(BB), x - 1.0, y - 1.0, z), u), v),
      mix(mix(grad(P(AA + 1), x, y, z - 1.0), grad(P(BA + 1), x - 1.0, y, z - 1.0), u),
          mix(grad(P(AB + 1), x, y - 1.0, z - 1.0), grad(P(BB + 1), x - 1.0, y - 1.0, z - 1.0), u), v),
      w);
  }
  float strati(vec2 p) {
    vec2 q = p / texSize;
    return noise(vec3(q, 0.37)) + 0.45 * noise(vec3(q * 2.3 + vec2(17.0, 0.0), 1.7));
  }
  float layerTex(vec2 p) { return modKind == 1 ? cumuli(p) : strati(p); }

  // Cloud amount (0..1). r, rw: the shape field here and its change across a
  // pixel. g: where the line of sight meets the layer; ddx, ddy: how far g
  // moves across one pixel. Toward the horizon a pixel covers a long, thin
  // strip of the layer, so the texture is averaged along the strip's long
  // axis, and only where even its short axis outgrows the texture does it
  // give way to the expected cover.
  float layerCloud(float r, float rw, vec2 g, vec2 ddx, vec2 ddy) {
    if (modKind == 0) return smoothstep(layerT - rw, layerT + rw, r) * layerFade;
    float feature = modKind == 1 ? 2.0 * ${ELEMENT_R.toFixed(3)} * texSize : 0.5 * texSize;
    float lx = length(ddx), ly = length(ddy);
    vec2 major = lx > ly ? ddx : ddy;
    float minor = max(min(lx, ly), 1e-3);
    float k = smoothstep(0.3, 1.0, minor / feature);
    float w = rw + 1.5 * amount * minor / feature;
    float c = 0.0;
    if (k < 0.999) {
    #ifdef ONE_SAMPLE
      c = smoothstep(layerT - w, layerT + w, r + amount * layerTex(g));
    #else
      vec4 f = r + amount * vec4(layerTex(g - 0.375 * major), layerTex(g - 0.125 * major),
                                 layerTex(g + 0.125 * major), layerTex(g + 0.375 * major));
      vec4 s = smoothstep(vec4(layerT - w), vec4(layerT + w), f);
      c = 0.25 * (s.x + s.y + s.z + s.w);
    #endif
    }
    float expected = smoothstep(layerT - amount, layerT + amount, r);
    return mix(c, expected, k) * layerFade;
  }
`;

// The dome is shaded from the cloudiness field itself (half-float texture,
// linearly interpolated) so cloud outlines stay crisp however far you zoom.
const fieldData = new Uint16Array(N);
const fieldTexture = new THREE.DataTexture(fieldData, W, H, THREE.RedFormat, THREE.HalfFloatType);
fieldTexture.wrapS = THREE.RepeatWrapping;
fieldTexture.magFilter = THREE.LinearFilter;
fieldTexture.minFilter = THREE.LinearFilter;
const domeUniforms = {
  ...layerUniforms,
  layerOn: { value: 0 },
  field: { value: fieldTexture },
  threshold: { value: 0 },
  skyColor: { value: new THREE.Vector3(120 / 255, 170 / 255, 225 / 255) },
  cloudColor: { value: new THREE.Vector3(244 / 255, 246 / 255, 249 / 255) },
  skyAlpha: { value: 0.2 },
  cloudAlpha: { value: 0.93 },
};
const dome = new THREE.Mesh(
  domeGeometry(),
  new THREE.ShaderMaterial({
    uniforms: domeUniforms,
    transparent: true,
    side: THREE.DoubleSide,
    depthWrite: false,
    vertexShader: `
      #include <common>
      #include <logdepthbuf_pars_vertex>
      varying vec2 vUv;
      varying vec3 vDir;
      void main() {
        vUv = uv;
        vDir = position;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        #include <logdepthbuf_vertex>
      }`,
    fragmentShader: `
      uniform sampler2D field;
      uniform float threshold, skyAlpha, cloudAlpha;
      uniform vec3 skyColor, cloudColor;
      uniform int layerOn;
      varying vec2 vUv;
      varying vec3 vDir;
      #include <logdepthbuf_pars_fragment>
      ${LAYER_GLSL}
      void main() {
        #include <logdepthbuf_fragment>
        float a;
        if (layerOn == 1) {
          // Follow this pixel's line of sight out to the (curved) cloud layer.
          vec3 n = normalize(vDir);
          vec3 s = vec3(n.x, -n.z, n.y); // east, north, up
          float hl = max(length(s.xy), 1e-6);
          float t = max(s.z, 0.0) / hl;
          float dist = 2.0 * layerHeight / (t + sqrt(t * t + 2.0 * layerHeight / 6371000.0));
          vec2 g = s.xy / hl * dist;
          vec2 gx = dFdx(g), gy = dFdy(g);
          float r = region(s), rw = fwidth(r);
          a = s.z < 0.0 ? 0.0 : layerCloud(r, rw, g, gx, gy);
        } else {
          float f = texture2D(field, vUv).r;
          float w = max(fwidth(f), 1e-5);
          a = smoothstep(threshold - w, threshold + w, f);
        }
        gl_FragColor = vec4(mix(skyColor, cloudColor, a), mix(skyAlpha, cloudAlpha, a));
      }`,
  }),
);
dome.renderOrder = 2;
scene.add(dome);

const groundMaterial = new THREE.MeshBasicMaterial({ color: 0x33402f });

function circleLine(e, r, color, opacity) {
  const pts = [];
  for (let i = 0; i <= 256; i++) pts.push(dirFromAzEl((i / 256) * 2 * Math.PI, e, r));
  return new THREE.Line(
    new THREE.BufferGeometry().setFromPoints(pts),
    new THREE.LineBasicMaterial({ color, transparent: true, opacity, depthWrite: false }),
  );
}
function meridianLine(a, r, color, opacity) {
  const pts = [];
  for (let i = 0; i <= 64; i++) pts.push(dirFromAzEl(a, (i / 64) * 90 * DEG, r));
  return new THREE.Line(
    new THREE.BufferGeometry().setFromPoints(pts),
    new THREE.LineBasicMaterial({ color, transparent: true, opacity, depthWrite: false }),
  );
}
scene.add(circleLine(0, 1.001, 0xaec6e0, 0.9));

let guideGroup = new THREE.Group();
scene.add(guideGroup);
let fadingGuides = null;
// Vert-slice boundaries: half great circles from one end of the slicing axis
// to the other, cut at the horizon. Points are [east, north, up].
// Vert-slice boundaries: half great circles from one end of the axis, over
// the sky, to the other, every 22.5 degrees. Points are [east, north, up].
function lyingArcs(a0) {
  const { u, p, q } = vertSliceFrame(a0);
  const arcs = [];
  for (let k = 1; k < 8; k++) {
    const turn = k * 22.5 * DEG;
    const w = [0, 1, 2].map((i) => p[i] * Math.cos(turn) + q[i] * Math.sin(turn));
    const pts = [];
    for (let i = 0; i <= 120; i++) {
      const t = (i / 120) * Math.PI;
      pts.push([0, 1, 2].map((j) => u[j] * Math.cos(t) + w[j] * Math.sin(t)));
    }
    arcs.push(pts);
  }
  return arcs;
}
function guideSpec(g) {
  if (g === 'slices') return { circles: [], meridians: [0, 45, 90, 135, 180, 225, 270, 315], arcs: [] };
  // guides are fixed to the compass: vertical slices always run north-south
  if (g === 'lying') return { circles: [], meridians: [], arcs: lyingArcs(0) };
  return { circles: [], meridians: [], arcs: [] };
}
// fade: crossfade from the old guides (button changes); sliders that move
// the guides continuously rebuild them in place.
function buildGuides(fade = false) {
  if (fadingGuides) scene.remove(fadingGuides);
  fadingGuides = null;
  const old = guideGroup;
  guideGroup = new THREE.Group();
  scene.add(guideGroup);
  const { circles, meridians, arcs } = guideSpec(guides);
  for (const e of circles) guideGroup.add(circleLine(e * DEG, 1.004, 0xffd27a, 0.75));
  for (const a of meridians) guideGroup.add(meridianLine(a * DEG, 1.004, 0xffd27a, 0.75));
  for (const pts of arcs) {
    guideGroup.add(new THREE.Line(
      new THREE.BufferGeometry().setFromPoints(pts.map(([e, n, u]) => toScene(e, n, u).multiplyScalar(1.004))),
      new THREE.LineBasicMaterial({ color: 0xffd27a, transparent: true, opacity: 0.75, depthWrite: false }),
    ));
  }
  if (!fade) {
    scene.remove(old);
    return;
  }
  fadingGuides = old;
  const setOpacity = (g, o) => g.children.forEach((l) => { l.material.opacity = o; });
  setOpacity(guideGroup, 0);
  animate('guides', 500, (e) => {
    setOpacity(guideGroup, 0.75 * e);
    setOpacity(old, 0.75 * (1 - e));
  }, () => {
    scene.remove(old);
    if (fadingGuides === old) fadingGuides = null;
  });
}

function textSprite(text, color = '#e6ebf1', size = 0.12) {
  const c = document.createElement('canvas');
  c.width = c.height = 128;
  const g = c.getContext('2d');
  g.font = '600 80px system-ui, sans-serif';
  g.textAlign = 'center';
  g.textBaseline = 'middle';
  g.fillStyle = color;
  g.fillText(text, 64, 68);
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: t, depthWrite: false, transparent: true }));
  s.scale.set(size, size, size);
  return s;
}
for (const [label, a] of [['N', 0], ['E', 90], ['S', 180], ['W', 270]]) {
  const s = textSprite(label, label === 'N' ? '#ffb3a1' : '#e6ebf1');
  s.position.copy(dirFromAzEl(a * DEG, 0, 1.13)).setY(0.05);
  scene.add(s);
}

// a little person standing at the centre
const observer = new THREE.Group();
{
  const mat = new THREE.MeshLambertMaterial({ color: 0xff9f43 });
  const body = new THREE.Mesh(new THREE.CylinderGeometry(0.012, 0.016, 0.06, 16), mat);
  body.position.y = 0.03;
  const head = new THREE.Mesh(new THREE.SphereGeometry(0.013, 16, 12), mat);
  head.position.y = 0.073;
  observer.add(body, head);
}
scene.add(observer);

// ---------------------------------------------------------------- scale
//
// The scene is to scale around the observer. Their eye, 1.7 m above the
// ground, is the dome's centre, and the dome's radius is the distance to their
// horizon, sqrt(2 R h) = 4.65 km: the dome stands on the horizon, and the
// ground inside it is all the ground they can see. In the cloud layer, clouds
// are drawn out to where the observer's line of sight, grazing that horizon,
// still reaches them. Everything follows the Earth's curve.

const EYE_M = 1.7;
const REACH_M = Math.sqrt(2 * EARTH_R * EYE_M); // distance to the ground horizon
const SCENE_M = REACH_M; // metres per scene unit: the dome's radius
const K = 1 / SCENE_M;
const REACH = REACH_M * K;
const GROUND_Y = -EYE_M * K; // the scene's origin is the observer's eye
// how far the Earth's surface has curved away below the observer's level, at
// horizontal distance d metres (scene units)
const drop = (d) => ((d * d) / (2 * EARTH_R)) * K;
// Farthest point at height h (m above ground) the observer can see: the line
// of sight grazing the ground horizon, then rising to h on the far side.
const sightRange = (h) => REACH_M + Math.sqrt(2 * EARTH_R * Math.max(0, h));

// A horizontal disc of radius r (scene units) that follows the Earth's curve.
function curvedDisc(r, segments, rings = 48) {
  const g = new THREE.RingGeometry(r * 1e-4, r, segments, rings);
  g.rotateX(-Math.PI / 2);
  const pos = g.attributes.position;
  for (let i = 0; i < pos.count; i++) {
    pos.setY(i, -drop(Math.hypot(pos.getX(i), pos.getZ(i)) / K));
  }
  return g;
}

const groundLayer = new THREE.Mesh(curvedDisc(REACH, 160, 8), groundMaterial);
groundLayer.position.y = GROUND_Y;
scene.add(groundLayer);

// The observer is far too small to see at this scale, so a marker of fixed
// on-screen size shows where they stand.
const observerMarker = (() => {
  const c = document.createElement('canvas');
  c.width = c.height = 64;
  const g = c.getContext('2d');
  g.fillStyle = '#ff9f43';
  g.strokeStyle = '#1b1f27';
  g.lineWidth = 6;
  g.beginPath();
  g.arc(32, 32, 22, 0, 2 * Math.PI);
  g.fill();
  g.stroke();
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: t, sizeAttenuation: false, depthTest: false, transparent: true }));
  s.scale.set(0.018, 0.018, 1);
  s.position.y = GROUND_Y;
  s.renderOrder = 6;
  scene.add(s);
  return s;
})();
const setObserverVisible = (v) => { observer.visible = v; observerMarker.visible = v; };

// Étage limits in feet, as used for reporting, and the shade each étage's
// clouds get: low clouds are thick water clouds with shaded bases, so the
// greyest; mid clouds lighter; high clouds thin ice, nearly pure white. The
// shade follows the cloud base, so it blends as the layer moves between
// étages.
const ETAGE_TINT = { low: [208, 212, 218], mid: [230, 233, 237], high: [250, 251, 253] };
function tintFor(baseM) {
  const ft = baseM / FT;
  const blend = (a, b, t) => a.map((v, i) => v + (b[i] - v) * Math.min(1, Math.max(0, t)));
  if (ft < 6550) return blend(ETAGE_TINT.low, ETAGE_TINT.mid, (ft - 5500) / 1050);
  if (ft < 19500) return blend(ETAGE_TINT.mid, ETAGE_TINT.high, (ft - 19000) / 1000);
  return ETAGE_TINT.high;
}
const ETAGES = {
  low: { name: 'Low', base: [300, 6500], band: [0, 6500] },
  mid: { name: 'Mid', base: [6600, 19000], band: [6500, 20000] },
  high: { name: 'High', base: [20000, 40000], band: [20000, 45000] },
};

// The layer in 3D: one curved surface out to the observer's line of sight.
const metresPerUnit = { value: SCENE_M };
const SURFACE_VERT = `
  #include <common>
  #include <logdepthbuf_pars_vertex>
  uniform float metresPerUnit;
  varying vec2 vGround; // east, north in metres
  void main() {
    vGround = vec2(position.x, -position.z) * metresPerUnit;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    #include <logdepthbuf_vertex>
  }`;
const SURFACE_FRAG = `
  #include <logdepthbuf_pars_fragment>
  uniform vec3 color;
  uniform float metresPerUnit; // also the dome's radius in metres
  varying vec2 vGround;
  ${LAYER_GLSL}
  void main() {
    #include <logdepthbuf_fragment>
    // the line of sight from the eye to this point of the (curved) layer
    float d = length(vGround);
    vec3 s = normalize(vec3(vGround, layerHeight - d * d / 12742000.0));
    float r = region(s), rw = fwidth(r);
    float a = s.z < 0.0 ? 0.0 : layerCloud(r, rw, vGround, dFdx(vGround), dFdy(vGround));
    if (a < 0.01) discard;
    // full brightness inside the dome, dimming with distance beyond it
    float far = smoothstep(1.0, 4.0, length(vec3(vGround, layerHeight)) / metresPerUnit);
    gl_FragColor = vec4(color * mix(1.0, 0.6, far), mix(0.95, 0.75, far) * a);
  }`;
const layerSurface = new THREE.Mesh(
  new THREE.BufferGeometry(),
  new THREE.ShaderMaterial({
    uniforms: { ...layerUniforms, metresPerUnit, color: { value: new THREE.Color(0xeef1f5) } },
    vertexShader: SURFACE_VERT,
    fragmentShader: SURFACE_FRAG,
    // seen from outside, one sample per pixel is enough (the dome uses four)
    defines: { ONE_SAMPLE: '' },
    side: THREE.DoubleSide,
    transparent: true,
    depthWrite: true, // so the dome is hidden where the layer lies in front of it
  }),
);
layerSurface.renderOrder = 1; // before the dome
layerSurface.frustumCulled = false;
layerSurface.visible = false;
layerSurface.userData.radius = 0;
scene.add(layerSurface);

// Point the shared layer uniforms at the current layer.
let layerThreshold = 0;
function updatePattern() {
  const p = layerParams;
  const U = layerUniforms;
  U.shapeKind.value = { slice: 0, lying: 1, cap: 2 }[p.shape];
  U.modKind.value = { none: 0, cumuli: 1, strati: 2 }[p.modifier];
  U.seed.value = p.seed;
  U.shapeAz.value = p.azimuth * DEG;
  U.sliceCentre.value = vertSliceCentre(layerOctasShown, p.elevation);
  const [a, e] = patchCentre(p.azimuth, p.elevation);
  U.capDir.value.set(Math.sin(a) * Math.cos(e), Math.cos(a) * Math.cos(e), Math.sin(e));
  U.texSize.value = p.size;
  U.amount.value = p.amount;
  U.layerT.value = Math.max(-1e4, Math.min(1e4, layerThreshold));
  U.layerHeight.value = Math.max(1, p.base - EYE_M);
  if (permSeed !== p.seed) {
    permTexture.image.data.set(noiseFor(p.seed).perm);
    permTexture.needsUpdate = true;
    permSeed = p.seed;
  }
  const tint = tintFor(p.base);
  domeUniforms.cloudColor.value.set(tint[0] / 255, tint[1] / 255, tint[2] / 255);
  layerSurface.material.uniforms.color.value.setRGB(tint[0] / 255, tint[1] / 255, tint[2] / 255, THREE.SRGBColorSpace);
}

function updateCloudMeshes() {
  const on = mode === 'layer';
  domeUniforms.layerOn.value = on ? 1 : 0;
  if (!on) domeUniforms.cloudColor.value.set(CLOUD[0] / 255, CLOUD[1] / 255, CLOUD[2] / 255);
  // The observer's eye is the dome's centre: the figure (to scale, its eye
  // 0.078 units up) stands one eye height below it, as does the ground.
  observer.scale.setScalar((EYE_M * K) / 0.078);
  observer.position.y = GROUND_Y;
  layerSurface.visible = false;
  if (!on) return;
  updatePattern();
  if (!layerParams.showClouds || layerOctasShown <= 0) return;
  const reach = sightRange(layerParams.base);
  // rebuild the curved disc only when its size changes noticeably
  if (Math.abs(layerSurface.userData.radius - reach) > reach * 0.01) {
    layerSurface.geometry.dispose();
    layerSurface.geometry = curvedDisc(reach * K, 192, 64);
    layerSurface.userData.radius = reach;
  }
  layerSurface.position.y = GROUND_Y + layerParams.base * K;
  // From inside, the 3D layer lines up exactly with the clouds on the dome,
  // so drawing it again there would only cost time.
  layerSurface.visible = !settledInside;
}

// ================================================================ cameras

const orbit = new OrbitControls(camera, canvas);
orbit.enableDamping = true;
orbit.minDistance = 1.2;
orbit.maxDistance = 7; // widened in cloud-layer mode
orbit.maxPolarAngle = Math.PI / 2 + 0.35;

const look = { yaw: 0, pitch: 25 * DEG, fov: 80, drag: null };

// True once the camera has arrived inside the dome (not while flying in).
let settledInside = false;

// The camera flies between views. While it flies, neither the orbit
// controls nor the inside look-around drive it.
let camFlying = false;

// Default outside pose per scene: what it looks at, how far out and up the
// camera sits, its field of view, and its default heading.
const OUTSIDE = {
  octas: { target: new THREE.Vector3(0, 0.25, 0), out: 3.19, up: 1.25, fov: 45, heading: Math.atan2(-2.1, 2.4) },
  // low, under the cloud layer, looking up at it past the dome
  layer: { target: new THREE.Vector3(0, 0.12, 0), out: 3.2, up: 0.05, fov: 55, heading: Math.atan2(-2.1, 2.4) },
};
// Outside the cloud layer the view is framed for a base up to about 4,500 ft
// (0.3 dome radii); higher layers scale the whole framing up, so the clouds
// stay in view.
const layerViewScale = () => Math.max(1, (layerParams.base * K) / 0.3);
let viewScaleShown = 1;
// heading: compass azimuth (radians) the camera faces
function cameraHeading() {
  const f = new THREE.Vector3();
  camera.getWorldDirection(f);
  return Math.atan2(f.x, -f.z);
}
function setOrbitLimits() {
  if (mode === 'layer') {
    orbit.maxDistance = 150; // far enough to see the whole layer, out to the line of sight
    orbit.maxPolarAngle = Math.PI / 2 - 0.01; // stay above the ground
  } else {
    orbit.maxDistance = 7;
    orbit.maxPolarAngle = Math.PI / 2 + 0.35;
  }
}
function flyTo(pos, target, fov, ms, done) {
  needsRender = true; // an instant move still has to be drawn
  const p0 = camera.position.clone();
  const f = new THREE.Vector3();
  camera.getWorldDirection(f);
  const t0 = p0.clone().add(f);
  const fov0 = camera.fov;
  const at = new THREE.Vector3();
  camFlying = true;
  animate('camera', ms, (e) => {
    camera.position.lerpVectors(p0, pos, e);
    camera.fov = fov0 + (fov - fov0) * e;
    camera.updateProjectionMatrix();
    camera.lookAt(at.lerpVectors(t0, target, e));
  }, () => {
    camFlying = false;
    done?.();
  });
}

// opts.reset: go to the default pose instead of keeping the current heading.
// opts.instant: no flight (used at start-up and behind a fade).
function setView(v, { yaw, reset = false, instant = false } = {}) {
  const from = view;
  view = v;
  $$('.view-switch [data-view]').forEach((b) => b.classList.toggle('on', b.dataset.view === v));
  orbit.enabled = false;
  const ms = instant ? 0 : 1200;
  // keep facing the same way while moving in or out
  const heading = yaw ?? (from === 'inside' ? look.yaw : cameraHeading());
  if (v === 'outside') {
    setOrbitLimits();
    setObserverVisible(true);
    settledInside = false;
    dirty = true; // bring the 3D layer back for the way out
    const o = OUTSIDE[mode === 'layer' ? 'layer' : 'octas'];
    const h = reset ? o.heading : heading;
    const sc = mode === 'layer' ? layerViewScale() : 1;
    viewScaleShown = sc;
    const target = o.target.clone().multiplyScalar(sc);
    // stand back from the target, opposite the way we face
    const pos = target.clone().add(new THREE.Vector3(-Math.sin(h) * o.out, o.up, Math.cos(h) * o.out).multiplyScalar(sc));
    flyTo(pos, target, o.fov, ms, () => {
      orbit.target.copy(target);
      orbit.enabled = view === 'outside';
      orbit.update();
    });
    $('#view-hint').textContent = 'Drag to orbit · scroll to zoom';
  } else {
    look.yaw = heading;
    look.pitch = 25 * DEG;
    look.fov = 80;
    flyTo(new THREE.Vector3(0, 0, 0), dirFromAzEl(look.yaw, look.pitch), look.fov, ms, () => {
      const inside = view === 'inside';
      setObserverVisible(!inside);
      settledInside = inside;
      dirty = true; // the 3D layer can go now
    });
    $('#view-hint').textContent = 'Drag to look around · scroll to change field of view';
  }
  // the sky is tinted more strongly from inside
  const a0 = domeUniforms.skyAlpha.value, a1 = v === 'inside' ? 0.62 : 0.2;
  animate('sky', ms, (e) => { domeUniforms.skyAlpha.value = a0 + (a1 - a0) * e; });
}

canvas.addEventListener('pointerdown', (ev) => {
  if (view !== 'inside') return;
  look.drag = { x: ev.clientX, y: ev.clientY };
  canvas.setPointerCapture(ev.pointerId);
});
canvas.addEventListener('pointermove', (ev) => {
  if (view !== 'inside' || !look.drag) return;
  const k = (look.fov / 80) * 0.005;
  look.yaw -= (ev.clientX - look.drag.x) * k;
  look.pitch += (ev.clientY - look.drag.y) * k;
  look.pitch = Math.max(-15 * DEG, Math.min(89.9 * DEG, look.pitch));
  look.drag = { x: ev.clientX, y: ev.clientY };
  needsRender = true;
});
canvas.addEventListener('pointerup', () => { look.drag = null; });
canvas.addEventListener('wheel', (ev) => {
  if (view !== 'inside') return;
  ev.preventDefault();
  look.fov = Math.max(30, Math.min(120, look.fov + ev.deltaY * 0.05));
  needsRender = true;
}, { passive: false });

function updateInsideCamera() {
  camera.position.set(0, 0.0, 0);
  camera.fov = look.fov;
  camera.updateProjectionMatrix();
  camera.lookAt(dirFromAzEl(look.yaw, look.pitch));
}

function resize() {
  const r = canvas.parentElement.getBoundingClientRect();
  renderer.setSize(r.width, r.height, false);
  camera.aspect = r.width / Math.max(1, r.height);
  camera.updateProjectionMatrix();
  needsRender = true;
}
new ResizeObserver(resize).observe(canvas.parentElement);

// ================================================================ painting

const SKY = [120, 170, 225];
const CLOUD = [244, 246, 249];

function paintDome(soft) {
  const cc = mode === 'layer' ? tintFor(layerParams.base) : CLOUD;
  const skyAlpha = view === 'inside' ? 0.62 : 0.2; // for the sky map's copy
  const cloudAlpha = 0.93;
  const toHalf = THREE.DataUtils.toHalfFloat;
  const T = soft ? threshold : 0.5;
  domeUniforms.threshold.value = Math.max(-6e4, Math.min(6e4, T));
  // The canvas copy feeds the 2D sky map.
  const d = texImage.data;
  const edge = 0.01;
  for (let k = 0; k < N; k++) {
    let a;
    if (soft) {
      fieldData[k] = toHalf(field[k]);
      a = (field[k] - threshold) / edge + 0.5;
      a = a < 0 ? 0 : a > 1 ? 1 : a;
    } else {
      fieldData[k] = toHalf(cover[k]);
      a = cover[k];
    }
    const o = k * 4;
    d[o] = SKY[0] + (cc[0] - SKY[0]) * a;
    d[o + 1] = SKY[1] + (cc[1] - SKY[1]) * a;
    d[o + 2] = SKY[2] + (cc[2] - SKY[2]) * a;
    d[o + 3] = 255 * (skyAlpha + (cloudAlpha - skyAlpha) * a);
  }
  fieldTexture.needsUpdate = true;
}

// ---------------------------------------------------------------- sky map

const mapCanvas = $('#skymap');
const mapCtx = mapCanvas.getContext('2d');
const MAP = mapCanvas.width;
const MAP_R = MAP / 2 - 26;
const mapImage = mapCtx.createImageData(MAP, MAP);
let mapLookup = null;

// Equal-area projection, so area on the map is proportional to octas:
// zenith angle (rad) at normalised map radius r, and the inverse.
const proj = { theta: (r) => 2 * Math.asin(r / Math.SQRT2), radius: (t) => Math.SQRT2 * Math.sin(t / 2) };

function buildMapLookup() {
  mapLookup = new Int32Array(MAP * MAP).fill(-1);
  for (let py = 0; py < MAP; py++) {
    for (let px = 0; px < MAP; px++) {
      const dx = (px + 0.5 - MAP / 2) / MAP_R;
      const dy = (py + 0.5 - MAP / 2) / MAP_R;
      const r = Math.hypot(dx, dy);
      if (r > 1) continue;
      const e = Math.PI / 2 - proj.theta(r);
      // looking up at the sky: north at top, east on the LEFT
      let a = Math.atan2(-dx, -dy);
      if (a < 0) a += 2 * Math.PI;
      const i = Math.min(W - 1, Math.floor((a / (2 * Math.PI)) * W));
      const j = Math.min(H - 1, Math.max(0, Math.floor((e / (Math.PI / 2)) * H)));
      mapLookup[py * MAP + px] = j * W + i;
    }
  }
}

function drawSkyMap() {
  if (!mapLookup) buildMapLookup();
  const src = texImage.data;
  const d = mapImage.data;
  for (let p = 0; p < MAP * MAP; p++) {
    const k = mapLookup[p];
    const o = p * 4;
    if (k < 0) { d[o + 3] = 0; continue; }
    // composite the dome texel over a dark night-blue base so the map is opaque
    const a = src[k * 4 + 3] / 255;
    const t = Math.min(1, (a - 0.15) / 0.78 + 0.25);
    d[o] = 30 + (src[k * 4] - 30) * t;
    d[o + 1] = 52 + (src[k * 4 + 1] - 52) * t;
    d[o + 2] = 84 + (src[k * 4 + 2] - 84) * t;
    d[o + 3] = 255;
  }
  mapCtx.clearRect(0, 0, MAP, MAP);
  mapCtx.putImageData(mapImage, 0, 0);

  const c = MAP / 2;
  const ringR = (eDeg) => proj.radius((90 - eDeg) * DEG) * MAP_R;
  mapCtx.save();
  mapCtx.strokeStyle = 'rgba(255, 210, 122, 0.9)';
  mapCtx.fillStyle = 'rgba(255, 210, 122, 0.95)';
  mapCtx.lineWidth = 1.5;
  mapCtx.font = '600 15px system-ui, sans-serif';
  const { circles, meridians, arcs } = guideSpec(guides);
  for (const pts of arcs) {
    mapCtx.beginPath();
    pts.forEach(([e, n, u], i) => {
      const r = proj.radius(Math.acos(Math.min(1, Math.max(-1, u)))) * MAP_R;
      const h = Math.hypot(e, n) || 1;
      // north up, east on the left (looking up)
      const x = c - (e / h) * r, y = c - (n / h) * r;
      if (i) mapCtx.lineTo(x, y); else mapCtx.moveTo(x, y);
    });
    mapCtx.stroke();
  }
  for (const e of circles) {
    mapCtx.beginPath();
    mapCtx.arc(c, c, ringR(e), 0, 2 * Math.PI);
    mapCtx.stroke();
  }
  for (const a of meridians) {
    mapCtx.beginPath();
    mapCtx.moveTo(c, c);
    mapCtx.lineTo(c - Math.sin(a * DEG) * MAP_R, c - Math.cos(a * DEG) * MAP_R);
    mapCtx.stroke();
  }
  mapCtx.strokeStyle = 'rgba(174, 198, 224, 0.9)';
  mapCtx.beginPath();
  mapCtx.arc(c, c, MAP_R, 0, 2 * Math.PI);
  mapCtx.stroke();
  mapCtx.font = '700 17px system-ui, sans-serif';
  mapCtx.textAlign = 'center';
  mapCtx.textBaseline = 'middle';
  for (const [label, a] of [['N', 0], ['E', 90], ['S', 180], ['W', 270]]) {
    mapCtx.fillStyle = label === 'N' ? '#ffb3a1' : '#dfe6ee';
    mapCtx.fillText(label, c - Math.sin(a * DEG) * (MAP_R + 13), c - Math.cos(a * DEG) * (MAP_R + 13));
  }
  mapCtx.restore();
}

// ================================================================ readouts

const fmtOctas = (f) => (f * 8).toFixed(2);
const fmtKm = (m) => (m === Infinity ? '∞' : m >= 10000 ? `${Math.round(m / 1000)} km` : m >= 1000 ? `${(m / 1000).toFixed(1)} km` : `${Math.round(m)} m`);

function updateReadout() {
  const el = $('#readout');
  if (mode === 'quiz' && !quiz.answered) {
    el.innerHTML = '<div class="dim">How many octas?</div><div class="big">? / 8</div>';
    return;
  }
  const rep = reportedOctas(coveredFrac);
  {
    el.innerHTML = `<div class="dim">Report</div>
      <div class="big">${rep} / 8</div>
      <div>${fmtOctas(coveredFrac)} octas exactly</div>
      <div class="dim">${(coveredFrac * 100).toFixed(1)}% of the dome</div>`;
  }
}


// ================================================================ update loop

// Octas and quiz: the amount shown eases toward the chosen one, and a new
// shape morphs out of the old. Both fields are turned into area ranks (the
// fraction of the dome less cloudy than each point), blended, and the
// blend is cut at the chosen amount, so the covered area is exact throughout.
let octasShown = shapeParams.octas;
let morph = null; // { from: ranks of the field shown when the morph began, start }
const shownField = new Float32Array(N);
const rankB = new Float32Array(N);
function ranksOf(src, out) {
  const B = 4096;
  let lo = Infinity, hi = -Infinity;
  for (let k = 0; k < N; k++) { const v = src[k]; if (v < lo) lo = v; if (v > hi) hi = v; }
  const scale = hi > lo ? (B - 1e-6) / (hi - lo) : 0;
  const hist = new Float64Array(B + 1);
  for (let k = 0; k < N; k++) hist[Math.floor((src[k] - lo) * scale) + 1] += grid.weight[k];
  for (let b = 1; b <= B; b++) hist[b] += hist[b - 1];
  for (let k = 0; k < N; k++) {
    const x = (src[k] - lo) * scale, b = Math.floor(x);
    out[k] = hist[b] + (hist[Math.min(B, b + 1)] - hist[b]) * (x - b);
  }
  return out;
}
function startMorph() {
  if (REDUCED_MOTION) return;
  morph = { from: ranksOf(shownField, new Float32Array(N)), start: performance.now() };
  dirty = true;
}
function easeOctasTo(target, ms = 700) {
  const from = octasShown;
  animate('octas', ms, (e) => { octasShown = from + (target - from) * e; dirty = true; });
}

// Cloud layer: the octas ease too.
let layerOctasShown = layerParams.octas;

function recompute() {
  if (mode === 'layer') {
    // The GPU draws the layer; the same field here gives the threshold for
    // the chosen octas, the numbers and the sky map. Lines of sight start at
    // the observer's eye, 1.7 m above the ground.
    layerField({
      ...layerParams, octas: layerOctasShown, height: Math.max(1, layerParams.base - EYE_M),
    }, field);
    layerThreshold = solveThreshold(field, layerOctasShown / 8);
    coveredFrac = coverFromField(field, layerThreshold, cover);
    paintDome(false);
    updateCloudMeshes();
  } else {
    const p = mode === 'quiz' ? quizParams : shapeParams;
    computeField({ ...p, noise: noiseFor(p.seed) }, field);
    if (morph) {
      const t = Math.min(1, (performance.now() - morph.start) / 900);
      const e = easeInOut(t);
      ranksOf(field, rankB);
      for (let k = 0; k < N; k++) field[k] = morph.from[k] + (rankB[k] - morph.from[k]) * e;
      if (t >= 1) morph = null;
      else dirty = true; // keep morphing next frame
    }
    shownField.set(field);
    threshold = solveThreshold(field, octasShown / 8);
    coveredFrac = coverFromField(field, threshold, cover);
    paintDome(true);
    updateCloudMeshes();
  }
  drawSkyMap();
  updateReadout();
}

// Smooth move of the cloud base when the étage changes.
let baseTween = null;

let last = performance.now();
function frame(now) {
  const dt = Math.min(0.1, (now - last) / 1000);
  last = now;
  if (mode === 'octas' && shapeParams.animate && shapeParams.modifier !== 'none' && shapeParams.amount > 0) {
    shapeParams.time += dt * 0.6;
    dirty = true;
  }
  runTweens(now);
  if (baseTween) {
    const t = Math.min(1, (now - baseTween.start) / baseTween.duration);
    const ease = t < 0.5 ? 2 * t * t : 1 - (-2 * t + 2) ** 2 / 2;
    for (const key of Object.keys(baseTween.to)) {
      layerParams[key] = baseTween.from[key] + (baseTween.to[key] - baseTween.from[key]) * ease;
    }
    if (t >= 1) baseTween = null;
    layerCtl.base.sync();
    dirty = true;
  }
  if (dirty) {
    dirty = false;
    recompute();
    needsRender = true;
  }
  if (tweens.size || camFlying || baseTween) needsRender = true;
  if (!camFlying && view === 'outside' && mode === 'layer') {
    // follow the cloud base: scale the view about the observer as it rises
    // or falls, keeping the angle the user chose
    const sc = layerViewScale();
    if (Math.abs(sc - viewScaleShown) > 1e-4) {
      const r = sc / viewScaleShown;
      const offset = camera.position.clone().sub(orbit.target).multiplyScalar(r);
      orbit.target.multiplyScalar(r);
      camera.position.copy(orbit.target).add(offset);
      viewScaleShown = sc;
    }
  }
  if (!camFlying) {
    if (view === 'inside') updateInsideCamera();
    else if (orbit.update()) needsRender = true; // true while moving or easing
  }
  // From outside, draw the layer over the dome's tinted shell, so the clouds
  // inside the dome are not darkened by it; from inside, the dome's own cloud
  // pattern goes on top.
  layerSurface.renderOrder = view === 'outside' ? 3 : 1;
  // Only draw when something changed: the cloud layer is costly per pixel,
  // and a still scene needs no redrawing.
  if (needsRender) {
    needsRender = false;
    renderer.render(scene, camera);
  }
  requestAnimationFrame(frame);
}

// ================================================================ UI wiring

function bindRange(id, obj, key, fmt, after) {
  const input = $(`#${id}`);
  const out = $(`#${id}-out`);
  const sync = () => { if (out) out.textContent = fmt(obj[key]); };
  input.value = obj[key];
  input.addEventListener('input', () => {
    obj[key] = parseFloat(input.value);
    sync();
    after?.();
    dirty = true;
  });
  sync();
  return { input, sync, set: (v) => { obj[key] = v; input.value = v; sync(); } };
}

const domeCtl = {
  azimuth: bindRange('azimuth', shapeParams, 'azimuth', (v) => `${v}°`),
  elevation: bindRange('elevation', shapeParams, 'elevation', (v) => `${v}°`),
};
// The Form slider: cumuli to the left, strati to the right, none in the
// middle. Values near the middle snap to exactly zero, so it is easy to find.
const fmtForm = (v) => (v < 0 ? `cumuli ${(-v).toFixed(2)}` : v > 0 ? `strati ${v.toFixed(2)}` : 'none');
function bindForm(id, p, after) {
  const input = $(`#${id}`);
  const out = $(`#${id}-out`);
  const snap = 0.06 * parseFloat(input.max);
  const show = () => { input.value = p.form; out.textContent = fmtForm(p.form); };
  input.addEventListener('input', () => {
    let v = parseFloat(input.value);
    if (Math.abs(v) < snap) {
      v = 0;
      input.value = 0;
    }
    const was = p.modifier;
    setForm(p, v);
    out.textContent = fmtForm(v);
    after?.(was);
    dirty = true;
  });
  show();
  return { input, set: (v) => { setForm(p, v); show(); } };
}
domeCtl.form = bindForm('form', shapeParams);

const chips = $('#octa-chips');
for (let k = 0; k <= 8; k++) {
  const b = document.createElement('button');
  b.textContent = k;
  b.addEventListener('click', () => {
    shapeParams.octas = k;
    $('#octas-out').textContent = k;
    syncChips();
    easeOctasTo(k);
  });
  chips.append(b);
}
function syncChips() {
  [...chips.children].forEach((b, k) => b.classList.toggle('on', shapeParams.octas === k));
}
syncChips();

const shapeHelp = {
  lying: 'An orange lying on its side: each slice runs between two opposite horizon points, over the sky, and every eighth of the turn is one octa. The cloud rises from the horizon at the azimuth; elevation slides it up, over the zenith, and down the far side.',
  cap: 'A round patch, centred at the azimuth and elevation (past 90° it has crossed the zenith). Lower it toward the horizon: the horizon clips it and it stretches to keep its area.',
};
function syncShape() {
  $$('#shape button').forEach((b) => b.classList.toggle('on', b.dataset.shape === shapeParams.shape));
  $('#shape-help').textContent = shapeHelp[shapeParams.shape];
}
function setGuides(g) {
  guides = g;
  $$('#guides button').forEach((x) => x.classList.toggle('on', x.dataset.guides === g));
  buildGuides(true);
  drawSkyMap();
}
$$('#shape button').forEach((b) => b.addEventListener('click', () => {
  if (b.dataset.shape === shapeParams.shape) return;
  startMorph();
  shapeParams.shape = b.dataset.shape;
  syncShape();
  dirty = true;
}));
syncShape();

// Freeze stops the outline drifting (only the Octas tab drifts).
$('#freeze').addEventListener('click', (ev) => {
  shapeParams.animate = !shapeParams.animate;
  const frozen = !shapeParams.animate;
  ev.currentTarget.textContent = frozen ? 'Unfreeze' : 'Freeze';
  ev.currentTarget.setAttribute('aria-pressed', String(frozen));
  ev.currentTarget.title = frozen ? 'Let the cloud outline drift again' : 'Stop the cloud outline drifting';
});

$$('#guides button').forEach((b) => b.addEventListener('click', () => setGuides(b.dataset.guides)));

// ---------------------------------------------------------------- layer tab

// Cloud base height is shown in feet or metres; everything is
// stored in metres. Horizontal distances stay metric.
let heightUnit = 'ft';
const heightRange = () => {
  const [lo, hi] = ETAGES[layerParams.etage].base;
  return heightUnit === 'ft'
    ? [lo, hi, 100]
    : [Math.ceil((lo * FT) / 10) * 10, Math.floor((hi * FT) / 10) * 10, 10];
};
const fmtHeight = (m) => (heightUnit === 'ft'
  ? `${(Math.round(m / FT / 10) * 10).toLocaleString('en-CA')} ft`
  : `${Math.round(m).toLocaleString('en-CA')} m`);

function bindHeight(id, key) {
  const input = $(`#${id}`);
  const out = $(`#${id}-out`);
  const per = () => (heightUnit === 'ft' ? FT : 1);
  const sync = () => { out.textContent = fmtHeight(layerParams[key]); };
  const place = () => {
    const [min, max, step] = heightRange();
    Object.assign(input, { min, max, step });
    input.value = Math.round(layerParams[key] / per() / step) * step;
    sync();
  };
  input.addEventListener('input', () => {
    layerParams[key] = parseFloat(input.value) * per();
    sync();
    dirty = true;
  });
  place();
  return { input, sync: place, set: (v) => { layerParams[key] = v; place(); } };
}

const layerCtl = {
  base: bindHeight('lbase', 'base'),
  azimuth: bindRange('laz', layerParams, 'azimuth', (v) => `${v}°`),
  centre: bindRange('lcentre', layerParams, 'elevation', (v) => `${v}°`),
};
// Form: switching between cumuli and strati happens at zero strength, so the
// texture's size can change there unseen.
layerCtl.form = bindForm('lform', layerParams, (was) => {
  if (layerParams.modifier !== was && !baseTween) {
    layerParams.size = TEX_SIZE[layerParams.etage][layerParams.modifier];
  }
  syncLayerControls();
});

// Coverage seen on the dome, in whole octas.
const lchips = $('#lcover-chips');
for (let k = 0; k <= 8; k++) {
  const b = document.createElement('button');
  b.textContent = k;
  b.addEventListener('click', () => {
    layerParams.octas = k;
    syncLayerChips();
    const from = layerOctasShown;
    animate('layerOctas', 700, (e) => { layerOctasShown = from + (k - from) * e; dirty = true; });
  });
  lchips.append(b);
}
// Swap the layer for a different one (form, random pattern): the clouds fade
// out, the change is made, and they fade back in.
function swapLayer(change) {
  const f0 = layerUniforms.layerFade.value;
  animate('layerFade', 250 * f0, (e) => { layerUniforms.layerFade.value = f0 * (1 - e); }, () => {
    change();
    dirty = true;
    animate('layerFade', 350, (e) => { layerUniforms.layerFade.value = e; });
  });
}
function syncLayerChips() {
  [...lchips.children].forEach((b, k) => b.classList.toggle('on', layerParams.octas === k));
  $('#lcover-out').textContent = `${layerParams.octas}/8`;
}
$$('#units button').forEach((b) => b.addEventListener('click', () => {
  heightUnit = b.dataset.units;
  $$('#units button').forEach((x) => x.classList.toggle('on', x === b));
  layerCtl.base.sync();
}));
$('#lshow').addEventListener('change', (e) => { layerParams.showClouds = e.target.checked; dirty = true; });
$('#lreseed').addEventListener('click', () => swapLayer(() => { layerParams.seed = 1 + Math.floor(Math.random() * 1e6); }));

// Typical base for each étage (ft), and texture size (m): element spacing for
// cumuli, patch size for strati. No cumuli in the high étage.
const ETAGE_BASE_FT = { low: 4000, mid: 12000, high: 28000 };
const TEX_SIZE = {
  low: { none: 1000, cumuli: 1000, strati: 2000 },
  mid: { none: 600, cumuli: 600, strati: 8000 },
  high: { none: 10000, cumuli: 10000, strati: 10000 },
};
const GENUS = {
  low: {
    none: 'A plain shape, to see how the layer maps onto the dome.',
    cumuli: 'Cumulus (Cu). Towering cumulus (TCU) and cumulonimbus (Cb) have low bases too, even when their tops reach the high étage.',
    strati: 'Stratus (St) or a stratocumulus (Sc) sheet. Nimbostratus (Ns) bases are often low too.',
  },
  mid: {
    none: 'A plain shape, to see how the layer maps onto the dome.',
    cumuli: 'Altocumulus (Ac): small rounded elements, often in rows or patches.',
    strati: 'Altostratus (As): a grey or bluish sheet; the sun may show as through ground glass.',
  },
  high: {
    none: 'A plain shape, to see how the layer maps onto the dome.',
    strati: 'Cirrostratus (Cs) or cirrus (Ci): a thin veil or wisps. A thin veil can cover much of the sky while hiding little.',
  },
};

function syncLayerControls() {
  const { etage, modifier } = layerParams;
  $$('#etage button').forEach((b) => b.classList.toggle('on', b.dataset.etage === etage));
  // no cumuli in the high étage: the slider stops at the middle
  layerCtl.form.input.min = etage === 'high' ? 0 : -3;
  $('#genus').textContent = `Typical: ${GENUS[etage][modifier]}`;
  layerCtl.base.sync();
  syncLayerChips();
  dirty = true;
}
$$('#etage button').forEach((b) => b.addEventListener('click', () => {
  if (b.dataset.etage === layerParams.etage) return;
  layerParams.etage = b.dataset.etage;
  const to = { base: ETAGE_BASE_FT[layerParams.etage] * FT };
  if (layerParams.etage === 'high' && layerParams.form < 0) {
    // no cumuli up there: slide the form over to strati while the layer
    // rises (it passes through zero, so the texture changes unseen)
    const f0 = layerParams.form;
    animate('lform', 1500, (e) => {
      const was = layerParams.modifier;
      layerCtl.form.set(f0 + (2 - f0) * e);
      if (layerParams.modifier !== was) layerParams.size = TEX_SIZE.high[layerParams.modifier];
      dirty = true;
    }, syncLayerControls);
  } else {
    to.size = TEX_SIZE[layerParams.etage][layerParams.modifier];
  }
  // The layer glides to a typical base for the new étage, with that étage's
  // typical texture size; the octas stay as chosen.
  const from = Object.fromEntries(Object.keys(to).map((k) => [k, layerParams[k]]));
  baseTween = { from, to, start: performance.now(), duration: 1500 };
  syncLayerControls();
}));
syncLayerControls();

// ---------------------------------------------------------------- quiz

const answers = $('#quiz-answers');
for (let k = 0; k <= 8; k++) {
  const b = document.createElement('button');
  b.textContent = k;
  b.disabled = true;
  b.addEventListener('click', () => answerQuiz(k));
  answers.append(b);
}

function newQuiz() {
  const rnd = mulberry32(Math.floor(Math.random() * 1e9));
  const n = Math.floor(rnd() * 9);
  let frac;
  if (n === 0) frac = 0;
  else if (n === 8) frac = 1;
  else if (n === 1) frac = 0.01 + rnd() * (1.5 / 8 - 0.01);
  else if (n === 7) frac = 6.5 / 8 + rnd() * (1.5 / 8 - 0.01);
  else frac = (n - 0.5 + rnd() * 0.999) / 8;
  const shapes = ['lying', 'cap'];
  const mods = ['none', 'cumuli', 'cumuli', 'strati', 'strati'];
  quizParams = {
    octas: frac * 8,
    shape: shapes[Math.floor(rnd() * shapes.length)],
    azimuth: Math.floor(rnd() * 360),
    elevation: rnd() * 180,
    modifier: mods[Math.floor(rnd() * mods.length)],
    amount: 0.3 + rnd() * 0.9,
    form: 0, // unused by the quiz; modifier and amount are set directly
    time: 0,
    seed: 1 + Math.floor(rnd() * 1e6),
  };
  startMorph();
  easeOctasTo(frac * 8, 900);
  quiz.answered = false;
  [...answers.children].forEach((b) => { b.disabled = false; b.className = ''; });
  $('#quiz-result').textContent = '';
  setView('inside', { yaw: rnd() * 2 * Math.PI });
  syncQuizLock();
  dirty = true;
}

function answerQuiz(k) {
  if (quiz.answered || !quizParams) return;
  quiz.answered = true;
  const truth = reportedOctas(coveredFrac);
  const ok = k === truth;
  quiz.total++;
  if (ok) { quiz.right++; quiz.streak++; } else quiz.streak = 0;
  [...answers.children].forEach((b, i) => {
    b.disabled = true;
    if (i === truth) b.className = 'right';
    else if (i === k) b.className = 'wrong';
  });
  const off = k - truth;
  $('#quiz-result').innerHTML = ok
    ? `<span class="ok">Correct.</span> ${fmtOctas(coveredFrac)} octas exactly, reported as ${truth}.`
    : `<span class="no">${off > 0 ? 'Too high' : 'Too low'} by ${Math.abs(off)}.</span> ${fmtOctas(coveredFrac)} octas exactly, reported as ${truth}.`;
  $('#quiz-score').textContent = `Score ${quiz.right} / ${quiz.total} · streak ${quiz.streak}`;
  syncQuizLock();
  updateReadout();
}
$('#quiz-new').addEventListener('click', newQuiz);

// ---------------------------------------------------------------- tabs, view

const canvasFade = (o) => { canvas.style.opacity = o; };
// Until a quiz sky is answered the observer stays inside the dome, and the
// sky map (which would give the answer away) stays hidden.
function syncQuizLock() {
  const locked = mode === 'quiz' && !quiz.answered;
  const out = $('.view-switch [data-view="outside"]');
  out.disabled = locked;
  out.title = locked ? 'Answer first: then you can step outside' : '';
  $('.skymap').classList.toggle('hidden', locked);
}
const syncFreeze = () => { $('#freeze').hidden = mode !== 'octas'; };
$$('.tabs button').forEach((b) => b.addEventListener('click', () => {
  const tab = b.dataset.tab;
  $$('.tabs button').forEach((x) => x.classList.toggle('on', x === b));
  $$('.tab').forEach((s) => { s.hidden = s.id !== `tab-${tab}`; });
  const prev = mode;
  let next = mode;
  if (tab === 'octas' || tab === 'layer' || tab === 'quiz') next = tab;
  if (next === prev) return;
  // Azimuth, elevation and form are shared by the Celestial dome and Cloud
  // layer tabs: carry them over (no cumuli in the high étage).
  if (next === 'layer') {
    layerCtl.azimuth.set(shapeParams.azimuth);
    layerCtl.centre.set(shapeParams.elevation);
    layerCtl.form.set(layerParams.etage === 'high' ? Math.max(0, shapeParams.form) : shapeParams.form);
    layerParams.size = TEX_SIZE[layerParams.etage][layerParams.modifier];
    syncLayerControls();
  } else if (next === 'octas' && prev === 'layer') {
    domeCtl.azimuth.set(layerParams.azimuth);
    domeCtl.elevation.set(layerParams.elevation);
    domeCtl.form.set(layerParams.form);
  }
  if ((prev === 'layer') === (next === 'layer')) {
    // octas <-> quiz: same scene, so the sky morphs from one to the other
    startMorph();
    mode = next;
    if (mode === 'quiz' && !quizParams) newQuiz();
    else easeOctasTo(mode === 'quiz' ? quizParams.octas : shapeParams.octas);
    if (mode === 'quiz' && !quiz.answered && view === 'outside') setView('inside');
    syncQuizLock();
    syncFreeze();
    dirty = true;
    return;
  }
  // to or from the cloud layer: a different scene and scale, so fade through
  canvasFade(0);
  setTimeout(() => {
    mode = next;
    if (mode === 'quiz' && !quizParams) newQuiz();
    octasShown = mode === 'quiz' ? quizParams.octas : shapeParams.octas;
    morph = null;
    tweens.delete('octas');
    if (mode === 'quiz' && !quiz.answered) setView('inside', { instant: true });
    else if (view === 'outside') setView('outside', { reset: true, instant: true });
    recompute();
    needsRender = true;
    syncQuizLock();
    syncFreeze();
    canvasFade(1);
  }, REDUCED_MOTION ? 0 : 250);
}));

$$('.view-switch [data-view]').forEach((b) => b.addEventListener('click', () => {
  if (b.dataset.view !== view) setView(b.dataset.view);
}));
$('#reset-view').addEventListener('click', () => setView(view, { reset: true }));

// Hide or show the side panel; the 3D view takes the whole window.
$('#panel-toggle').addEventListener('click', (ev) => {
  const full = $('.app').classList.toggle('full');
  const b = ev.currentTarget;
  b.textContent = full ? 'Show controls' : 'Hide controls';
  b.title = full ? 'Show the controls' : 'Hide the controls';
  b.setAttribute('aria-expanded', String(!full));
  $('#panel').inert = full;
});

// ================================================================ go

buildGuides();
setView('outside', { reset: true, instant: true });
resize();
requestAnimationFrame(frame);
