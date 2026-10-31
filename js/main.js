import * as THREE from 'three';
import { OrbitControls } from '../vendor/OrbitControls.js';
import {
  W, H, N, grid, computeField, solveThreshold, coverFromField,
  reportedOctas, makeNoise3, mulberry32, vertSliceFrame, vertSliceCentre, patchCentre,
} from './sky.js';
import {
  layerField, cloudinessSampler, bakeType, bakeBlotches, prebake, TYPES, TYPE_ROWS,
  BLOTCH_SCALE, BLOTCH_ANGLE, EARTH_R, DECODE_GLSL, TOWER_STEPS,
} from './layer.js';
import { solidField, solidSpheres, REACH_MAX, SOLID } from './cumulus.js';
import { dayOfYear, solarDay, halfDay, sunPosition, utcMinutes } from './sun.js';
import { CLIMATE } from './climate.js';

const DEG = Math.PI / 180;
const FT = 0.3048; // metres per foot
const $ = (s) => document.querySelector(s);
const $$ = (s) => [...document.querySelectorAll(s)];

// ================================================================ state

// shape: 'lying' (slice) or 'cap' (patch), placed by azimuth and elevation
// (see sky.js shapeField). Plain geometry: real cloud textures live on the
// Cloud layer tab.
const shapeParams = {
  octas: 1, shape: 'cap', azimuth: 270, elevation: 60,
  modifier: 'none', amount: 0, time: 0, seed: 1,
};
// The cloud layer: one round patch on a flat layer, plain or drawn as one of
// the cloud types (layer.js TYPES), whose pattern lives on the layer.
const layerParams = {
  etage: 'low', shape: 'cap', azimuth: 270, elevation: 60, // round patch only
  type: 'plain', octas: 1, base: 4000 * FT, seed: 1,
};
// The quiz draws its own random cloud layer, so the Cloud layer tab keeps
// its settings.
let quizLayer = null;
const quiz = { answered: false, right: 0, total: 0, typeRight: 0, typeTotal: 0, streak: 0 };

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
// layer.js). Cloudiness = the patch's field for the line of sight + the cloud
// type's pattern where that line of sight meets the layer; cloud where it
// exceeds layerT. The patterns are baked once into repeating tiles
// (layer.js bakeType, bakeBlotches) and sampled with mipmaps: two lookups
// per pixel.
function tileTexture(size, format) {
  const ch = format === THREE.RGFormat ? 2 : 1;
  const t = new THREE.DataTexture(new Uint8Array(size * size * ch), size, size, format, THREE.UnsignedByteType);
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  t.magFilter = THREE.LinearFilter;
  t.minFilter = THREE.LinearMipmapLinearFilter;
  t.generateMipmaps = true;
  t.anisotropy = renderer.capabilities.getMaxAnisotropy();
  return t;
}
const cloudTex = tileTexture(512, THREE.RGFormat); // every type's tile is 512 square
let cloudTexKey = null;
const blotchTex = tileTexture(256, THREE.RedFormat);
let blotchSeed = null;
const layerUniforms = {
  shapeKind: { value: 0 }, // 0 horizontal slice, 1 vertical slice, 2 round patch
  typeOn: { value: 0 }, // 0 a plain patch, 1 drawn as a cloud type
  shapeAz: { value: 0 },
  sliceCentre: { value: 0 },
  capDir: { value: new THREE.Vector3(0, 0, 1) }, // east, north, up
  // the type (layer.js TYPES)
  tileM: { value: 10000 },
  featureM: { value: 500 },
  amount: { value: 0 },
  blotch: { value: 0 },
  soft: { value: 0 },
  texAngle: { value: 0 },
  layerT: { value: 0 },
  layerHeight: { value: 1200 }, // cloud base above the eye, m
  layerFade: { value: 1 }, // fades the clouds out and in when the type changes
  opacity: { value: 1 },
  shadeAmt: { value: 0 },
  wisp: { value: 0 },
  // clouds with height (layer.js towerField)
  towerOn: { value: 0 },
  topM: { value: 1000 },
  anvil: { value: 0 },
  profile: { value: 2 },
  cloudTex: { value: cloudTex },
  blotchTex: { value: blotchTex },
};
const LAYER_GLSL = `
  uniform int shapeKind, typeOn, towerOn;
  uniform float topM, anvil, profile;
  uniform float shapeAz, sliceCentre, layerT, layerHeight, layerFade;
  uniform float tileM, featureM, amount, blotch, soft, texAngle, opacity, shadeAmt, wisp;
  uniform vec3 capDir;
  uniform sampler2D cloudTex, blotchTex;
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

  // layer.js layerField: the pattern's own frame on the layer, and its two tiles
  vec2 patternFrame(vec2 g) {
    float c = cos(texAngle), s = sin(texAngle);
    return vec2(c * g.x - s * g.y, s * g.x + c * g.y);
  }
  float blotches(vec2 q) {
    const float c = ${Math.cos(BLOTCH_ANGLE * DEG).toFixed(6)}, s = ${Math.sin(BLOTCH_ANGLE * DEG).toFixed(6)};
    float t = texture2D(blotchTex, vec2(c * q.x - s * q.y, s * q.x + c * q.y) / (tileM * ${BLOTCH_SCALE.toFixed(3)})).r;
    return ${DECODE_GLSL('t')};
  }

  // Cloud amount (0..1). r, rw: the patch's field here and its change across
  // a pixel. g: where the line of sight meets the layer; ddx, ddy: how far g
  // moves across one pixel. The tiles' mipmaps average the pattern over the
  // pixel; where even the pixel's short axis outgrows the pattern's features
  // it gives way to the cover expected there.
  float layerCloud(float r, float rw, vec2 g, vec2 ddx, vec2 ddy) {
    if (typeOn == 0) return smoothstep(layerT - rw, layerT + rw, r) * layerFade;
    vec2 q = patternFrame(g);
    float t = texture2D(cloudTex, q / tileM).r;
    float base = r + blotch * blotches(q);
    float minor = max(min(length(ddx), length(ddy)), 1e-3);
    float k = smoothstep(0.3, 1.0, minor / featureM);
    float w = rw + soft + 1.5 * amount * minor / featureM;
    float c = smoothstep(layerT - w, layerT + w, base + amount * ${DECODE_GLSL('t')});
    float expected = smoothstep(layerT - amount - soft, layerT + amount + soft, base);
    return mix(c, expected, k) * layerFade;
  }
  // How thick the cloud is here, 0..1: thick parts are drawn darker, and
  // thin parts of fibrous veils let the sky through (wisp).
  float layerShade(vec2 g) {
    if (typeOn == 0) return 0.0;
    return texture2D(cloudTex, patternFrame(g) / tileM).g;
  }
  float layerAlpha(float shade) { return opacity * (1.0 - wisp * (1.0 - shade)); }

  // Clouds with height, as layer.js towerField: the best of
  // cloudiness - profile(height) along the line of sight s, traced from the
  // base up to the tops. Returns that, and where (0 base .. 1 top) it was.
  // (sampled at a level of detail set by the pixel's footprint there, fp
  // metres, so the loop can stop early: see towerMarch)
  float cloudinessAt(vec2 p, float fp) {
    float d = length(p);
    vec3 sp = normalize(vec3(p, layerHeight - d * d / 12742000.0));
    vec2 q = patternFrame(p);
    float t = textureLod(cloudTex, q / tileM, max(0.0, log2(fp * 512.0 / tileM))).r;
    const float c = ${Math.cos(BLOTCH_ANGLE * DEG).toFixed(6)}, s = ${Math.sin(BLOTCH_ANGLE * DEG).toFixed(6)};
    float bm = tileM * ${BLOTCH_SCALE.toFixed(3)};
    float b = textureLod(blotchTex, vec2(c * q.x - s * q.y, s * q.x + c * q.y) / bm, max(0.0, log2(fp * 256.0 / bm))).r;
    return region(sp) + blotch * ${DECODE_GLSL('b')} + amount * ${DECODE_GLSL('t')};
  }
  vec2 towerMarch(vec3 s) {
    float hl = max(length(s.xy), 1e-6);
    float tn = max(s.z, 1e-6) / hl;
    vec2 u = s.xy / hl;
    float n = clamp(ceil(topM / tn / (0.5 * featureM)) + 1.0, 2.0, ${TOWER_STEPS.toFixed(1)});
    float pa = length(fwidth(s)); // the pixel's angular size
    float best = -1e9, at = 0.0;
    // each pixel shifts its samples by a fixed fraction of a step, so the
    // steps between samples show as fine grain instead of bands
    float jit = fract(52.9829189 * fract(dot(gl_FragCoord.xy, vec2(0.06711056, 0.00583715)))) - 0.5;
    for (int k = 0; k < ${TOWER_STEPS}; k++) {
      if (float(k) >= n) break;
      float zz = clamp((float(k) + jit) / (n - 1.0), 0.0, 1.0);
      float h = layerHeight + zz * topM;
      float d = 2.0 * h / (tn + sqrt(tn * tn + 2.0 * h / 6371000.0));
      float a = clamp((zz - 0.7) / 0.25, 0.0, 1.0);
      float v = cloudinessAt(u * d, d * pa) - amount * pow(zz, profile) * (1.0 - anvil * a * a * (3.0 - 2.0 * a));
      if (v > best) { best = v; at = zz; }
    }
    return vec2(best, at);
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
  // solid clouds: the dome shows their outline from the sky field, faded
  // like the cloud layer's projection
  solidOn: { value: 0 },
  // how much of the cloud layer's projection the dome shows: all of it from
  // outside, none from inside, where the layer itself lines up with it
  projection: { value: 1 },
  field: { value: fieldTexture },
  threshold: { value: 0 },
  // colours are given in sRGB and stored linear (THREE.Color); the shaders
  // convert to the output's colour space, so they look the same whether
  // drawn to the screen or into the stereographic view's cube map
  skyColor: { value: new THREE.Color().setRGB(120 / 255, 170 / 255, 225 / 255, THREE.SRGBColorSpace) },
  cloudColor: { value: new THREE.Color().setRGB(244 / 255, 246 / 255, 249 / 255, THREE.SRGBColorSpace) },
  topColor: { value: new THREE.Color().setRGB(250 / 255, 251 / 255, 253 / 255, THREE.SRGBColorSpace) },
  skyAlpha: { value: 0.2 },
  sunEnu: { value: new THREE.Vector3(0, -0.5, 0.8).normalize() },
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
      uniform vec3 sunEnu; // the sun's direction: east, north, up
      uniform vec3 skyColor, cloudColor, topColor;
      uniform int layerOn, solidOn;
      uniform float projection;
      varying vec2 vUv;
      varying vec3 vDir;
      #include <logdepthbuf_pars_fragment>
      ${LAYER_GLSL}
      void main() {
        #include <logdepthbuf_fragment>
        float a, shade = 0.0;
        vec3 cc = cloudColor;
        if (layerOn == 1 && towerOn == 1) {
          // clouds with height: traced through their height, drawn on the
          // dome from inside and out. Bases are dark, sunlit sides and tops
          // bright.
          vec3 n = normalize(vDir);
          vec3 s = vec3(n.x, -n.z, n.y); // east, north, up
          vec2 m = towerMarch(s);
          float w = fwidth(m.x) + soft;
          a = s.z < 0.0 ? 0.0 : opacity * smoothstep(layerT - w, layerT + w, m.x) * layerFade;
          cc = mix(cloudColor, topColor, smoothstep(0.0, 0.45, m.y));
        } else if (layerOn == 1) {
          // Follow this pixel's line of sight out to the (curved) cloud layer.
          vec3 n = normalize(vDir);
          vec3 s = vec3(n.x, -n.z, n.y); // east, north, up
          float hl = max(length(s.xy), 1e-6);
          float t = max(s.z, 0.0) / hl;
          float dist = 2.0 * layerHeight / (t + sqrt(t * t + 2.0 * layerHeight / 6371000.0));
          vec2 g = s.xy / hl * dist;
          vec2 gx = dFdx(g), gy = dFdy(g);
          float r = region(s), rw = fwidth(r);
          float th = layerShade(g);
          a = s.z < 0.0 ? 0.0 : projection * layerAlpha(th) * layerCloud(r, rw, g, gx, gy);
          shade = shadeAmt * th;
        } else {
          float f = texture2D(field, vUv).r;
          float w = max(fwidth(f), 1e-5);
          a = smoothstep(threshold - w, threshold + w, f);
          if (solidOn == 1) a *= projection;
        }
        // the sun: a disc (drawn twice its real size, to be seen) with a glow,
        // behind any cloud
        vec3 sd = normalize(vec3(vDir.x, -vDir.z, vDir.y));
        float cs = sd.z < 0.0 ? 0.0 : max(dot(sd, sunEnu), 0.0);
        float disc = smoothstep(${Math.cos(0.6 * DEG).toFixed(8)}, ${Math.cos(0.45 * DEG).toFixed(8)}, cs);
        float glow = 0.5 * pow(cs, 800.0) + 0.12 * pow(cs, 60.0);
        vec3 sky = mix(skyColor + glow * vec3(1.0, 0.95, 0.85), vec3(1.0, 0.98, 0.92), disc);
        float sa = max(skyAlpha, max(disc, glow));
        gl_FragColor = vec4(mix(sky, cc * (1.0 - shade), a), mix(sa, cloudAlpha, a));
        #include <colorspace_fragment>
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

// A common frame for both tabs, beyond the observer's own horizon:
//
// The rest of the planet: the ground carries on past the horizon, curving
// away below it, out to PLANET_M. It looks the same as the ground inside:
// the horizon is the observer's, and the dome already marks it. Clouds are
// drawn out to the farthest the observer can see them.
const PLANET_M = 600000;
function curvedRing(r0, r1, segments, rings) {
  const g = new THREE.RingGeometry(r0, r1, segments, rings);
  g.rotateX(-Math.PI / 2);
  const pos = g.attributes.position;
  for (let i = 0; i < pos.count; i++) {
    pos.setY(i, -drop(Math.hypot(pos.getX(i), pos.getZ(i)) / K));
  }
  return g;
}
const planet = new THREE.Mesh(
  curvedRing(REACH, PLANET_M * K, 192, 96),
  groundMaterial,
);
planet.position.y = GROUND_Y;
scene.add(planet);

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
const ETAGES = { low: { name: 'Low' }, mid: { name: 'Mid' }, high: { name: 'High' } };

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
  uniform float nearM; // nearer than this, the clouds are drawn as meshes
  varying vec2 vGround;
  ${LAYER_GLSL}
  void main() {
    #include <logdepthbuf_fragment>
    // the line of sight from the eye to this point of the (curved) layer
    float d = length(vGround);
    if (d < nearM) discard;
    vec3 s = normalize(vec3(vGround, layerHeight - d * d / 12742000.0));
    float r = region(s), rw = fwidth(r);
    float a = s.z < 0.0 ? 0.0 : layerCloud(r, rw, vGround, dFdx(vGround), dFdy(vGround));
    // Only the cloud patch is drawn: full brightness inside the dome,
    // dimming with distance beyond it.
    if (a < 0.01) discard;
    float dist = length(vec3(vGround, layerHeight));
    float far = smoothstep(1.0, 4.0, dist / metresPerUnit);
    float th = layerShade(vGround);
    vec3 c = color * (1.0 - shadeAmt * th);
    gl_FragColor = vec4(c * mix(1.0, 0.6, far), mix(0.95, 0.75, far) * a * layerAlpha(th));
    #include <colorspace_fragment>
  }`;
const layerSurface = new THREE.Mesh(
  new THREE.BufferGeometry(),
  new THREE.ShaderMaterial({
    uniforms: { ...layerUniforms, metresPerUnit, nearM: { value: 0 }, color: { value: new THREE.Color(0xeef1f5) } },
    vertexShader: SURFACE_VERT,
    fragmentShader: SURFACE_FRAG,
    side: THREE.DoubleSide,
    transparent: true,
    depthWrite: true, // so the dome is hidden where the layer lies in front of it
  }),
);
// drawn after the dome, so neither the dome's tint nor its projection
// covers the layer
layerSurface.renderOrder = 3;
layerSurface.frustumCulled = false;
layerSurface.visible = false;
layerSurface.userData.radius = 0;
scene.add(layerSurface);

// Solid clouds (cumulus.js): spheres, flattened a little, their surfaces
// lumped a little with Perlin noise, cut flat at the base, lit by the sun. Near clouds are drawn with finer
// spheres than far ones. Looking up into a cut sphere shows its inside: that
// is the flat, dark base.
// (the sun's direction in the scene: set by the sun controls)
const SUN = toScene(Math.sin(220 * DEG) * Math.cos(40 * DEG), Math.cos(220 * DEG) * Math.cos(40 * DEG), Math.sin(40 * DEG)).normalize();
const solidUniforms = {
  litCol: { value: new THREE.Color() },
  shadeCol: { value: new THREE.Color() },
  baseCol: { value: new THREE.Color() },
  hazeCol: { value: new THREE.Color().setRGB(196 / 255, 210 / 255, 228 / 255, THREE.SRGBColorSpace) },
  sunDir: { value: SUN },
  metresPerUnit,
  fade: layerUniforms.layerFade,
  bumpM: { value: 130 },
  lumpM: { value: 200 },
  lumpAmp: { value: 0.1 }, // how far the lumps push the surface (of the radius)
  sheet: { value: 0 }, // 1 for a sheet (stratocumulus), see the shader
  wispy: { value: 0 }, // 1 for see-through shreds (stratus fractus)
};
const solidMaterial = new THREE.ShaderMaterial({
  uniforms: solidUniforms,
  side: THREE.DoubleSide,
  transparent: true,
  vertexShader: `
    #include <common>
    #include <logdepthbuf_pars_vertex>
    attribute float aBase;
    attribute float aTop;
    uniform float metresPerUnit, lumpM, lumpAmp;
    varying vec3 vN, vW;
    varying float vBase, vTop;
    // Perlin (gradient) noise, about -1..1
    vec3 grad3(vec3 i) {
      return fract(sin(vec3(dot(i, vec3(127.1, 311.7, 74.7)), dot(i, vec3(269.5, 183.3, 246.1)),
        dot(i, vec3(113.5, 271.9, 124.6)))) * 43758.5453) * 2.0 - 1.0;
    }
    float perlin(vec3 p) {
      vec3 i = floor(p), f = fract(p);
      vec3 u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
      float n000 = dot(grad3(i), f), n100 = dot(grad3(i + vec3(1, 0, 0)), f - vec3(1, 0, 0));
      float n010 = dot(grad3(i + vec3(0, 1, 0)), f - vec3(0, 1, 0)), n110 = dot(grad3(i + vec3(1, 1, 0)), f - vec3(1, 1, 0));
      float n001 = dot(grad3(i + vec3(0, 0, 1)), f - vec3(0, 0, 1)), n101 = dot(grad3(i + vec3(1, 0, 1)), f - vec3(1, 0, 1));
      float n011 = dot(grad3(i + vec3(0, 1, 1)), f - vec3(0, 1, 1)), n111 = dot(grad3(i + vec3(1, 1, 1)), f - vec3(1, 1, 1));
      return 1.4 * mix(mix(mix(n000, n100, u.x), mix(n010, n110, u.x), u.y),
                       mix(mix(n001, n101, u.x), mix(n011, n111, u.x), u.y), u.z);
    }
    void main() {
      // Small lumps on the outline: each sphere's surface pushed in and out
      // by up to about a tenth of its radius, with noise fixed in space so
      // neighbouring spheres agree.
      vec4 w0 = modelMatrix * instanceMatrix * vec4(position, 1.0);
      float lump = perlin(w0.xyz * metresPerUnit / lumpM);
      vec4 w = modelMatrix * instanceMatrix * vec4(position * (1.0 + lumpAmp * lump), 1.0);
      vec3 sc = vec3(length(instanceMatrix[0].xyz), length(instanceMatrix[1].xyz), length(instanceMatrix[2].xyz));
      // the normal of a scaled (and turned) sphere
      vN = normalize(mat3(modelMatrix) * mat3(instanceMatrix) * (normal / (sc * sc)));
      vW = w.xyz;
      vBase = aBase;
      vTop = aTop;
      gl_Position = projectionMatrix * viewMatrix * w;
      #include <logdepthbuf_vertex>
    }`,
  fragmentShader: `
    #include <logdepthbuf_pars_fragment>
    uniform vec3 litCol, shadeCol, baseCol, hazeCol, sunDir;
    uniform float metresPerUnit, fade, bumpM, sheet, wispy;
    varying vec3 vN, vW;
    varying float vBase, vTop;
    // value noise, for the lumpy (cauliflower) surface
    float hash(vec3 p) { return fract(sin(dot(p, vec3(127.1, 311.7, 74.7))) * 43758.5453); }
    float vnoise(vec3 p) {
      vec3 i = floor(p), f = fract(p);
      f = f * f * (3.0 - 2.0 * f);
      return mix(mix(mix(hash(i), hash(i + vec3(1, 0, 0)), f.x), mix(hash(i + vec3(0, 1, 0)), hash(i + vec3(1, 1, 0)), f.x), f.y),
                 mix(mix(hash(i + vec3(0, 0, 1)), hash(i + vec3(1, 0, 1)), f.x), mix(hash(i + vec3(0, 1, 1)), hash(i + vec3(1, 1, 1)), f.x), f.y), f.z);
    }
    vec3 bumps(vec3 p) {
      return vec3(vnoise(p), vnoise(p + 17.3), vnoise(p + 41.9)) - 0.5
        + 0.5 * (vec3(vnoise(p * 2.7 + 5.1), vnoise(p * 2.7 + 23.4), vnoise(p * 2.7 + 61.2)) - 0.5);
    }
    void main() {
      #include <logdepthbuf_fragment>
      if (vW.y < vBase) discard; // the flat base
      float h = clamp((vW.y - vBase) / max(vTop - vBase, 1e-9), 0.0, 1.0);
      vec3 c;
      if (!gl_FrontFacing) {
        c = baseCol; // seen through the cut: the base
      } else {
        vec3 n = normalize(vN);
        // cauliflower lumps, fading out where they get too small on screen
        vec3 bp = vW * metresPerUnit / bumpM;
        float keep = (1.0 - wispy) * (1.0 - smoothstep(0.25, 0.8, length(fwidth(bp))));
        vec3 bmp = bumps(bp);
        n = normalize(n + 1.3 * keep * bmp);
        // wrapped diffuse light, so sunlit sides keep their modelling; the
        // creases between lumps a little darker
        float sunlit = pow(0.5 + 0.5 * dot(n, sunDir), 1.3);
        // in a sheet, the cloud above shades all but its top from the sun
        sunlit *= mix(1.0, smoothstep(0.55, 1.0, h), sheet);
        c = mix(shadeCol, litCol, sunlit);
        c *= 1.0 - 0.12 * keep * (0.5 - bmp.x - 0.3 * bmp.y);
        // darker low down and on the undersides
        c = mix(baseCol, c, smoothstep(0.0, 0.3, h) * (0.55 + 0.45 * smoothstep(-0.8, 0.1, n.y)));
      }
      float dist = length(vW) * metresPerUnit;
      c = mix(c, hazeCol, 0.7 * smoothstep(8000.0, 120000.0, dist));
      // wispy (stratus fractus): thin where seen edge-on, so the sky shows
      // through the edges of each shred
      float facing = abs(dot(normalize(vN), normalize(cameraPosition - vW)));
      float alpha = mix(1.0, 0.08 + 0.8 * smoothstep(0.05, 0.7, facing), wispy);
      gl_FragColor = vec4(c, fade * alpha);
      #include <colorspace_fragment>
    }`,
});
function solidMesh(detail) {
  const g = new THREE.IcosahedronGeometry(1, detail);
  const m = new THREE.InstancedMesh(g, solidMaterial, 1);
  m.count = 0;
  m.frustumCulled = false;
  m.renderOrder = 3;
  scene.add(m);
  return m;
}
const solidNear = solidMesh(2), solidFar = solidMesh(1);
// Grows a mesh's instance buffers when more spheres are needed.
function ensureCapacity(m, n) {
  if (m.instanceMatrix.count >= n) return m;
  const cap = Math.ceil(n * 1.3);
  m.instanceMatrix = new THREE.InstancedBufferAttribute(new Float32Array(cap * 16), 16);
  m.geometry.setAttribute('aBase', new THREE.InstancedBufferAttribute(new Float32Array(cap), 1));
  m.geometry.setAttribute('aTop', new THREE.InstancedBufferAttribute(new Float32Array(cap), 1));
  return m;
}
for (const m of [solidNear, solidFar]) {
  m.geometry.setAttribute('aBase', new THREE.InstancedBufferAttribute(new Float32Array(1), 1));
  m.geometry.setAttribute('aTop', new THREE.InstancedBufferAttribute(new Float32Array(1), 1));
}
let solidState = null; // from cumulus.js solidField
let solidScale = 0;
const tmpM = new THREE.Matrix4(), tmpP = new THREE.Vector3(), tmpS = new THREE.Vector3(), tmpQ = new THREE.Quaternion();
// The solid clouds to draw: cumuliform clouds (cumulus.js), or the near
// part of a stratocumulus layer (sheetSpheres).
function updateSolidMeshes(show) {
  solidNear.visible = solidFar.visible = show;
  const type = activeLayer().type;
  const list = !show ? null
    : TYPES[type]?.meshNear ? sheetSpheres(activeLayer(), layerThreshold)
      : solidState ? solidSpheres(solidState, solidScale) : null;
  if (!list) { solidNear.count = solidFar.count = 0; return; }
  const near = list.filter((q) => q.detail), far = list.filter((q) => !q.detail);
  for (const [m, part] of [[solidNear, near], [solidFar, far]]) {
    ensureCapacity(m, part.length);
    const base = m.geometry.getAttribute('aBase'), top = m.geometry.getAttribute('aTop');
    part.forEach((q, i) => {
      // the icosahedron's faces lie a little inside the unit sphere
      const f = m === solidNear ? 1.01 : 1.04;
      tmpP.set(q.e * K, q.up * K, -q.n * K);
      tmpS.set(q.rh * K * f, q.rv * K * f, (q.rh2 ?? q.rh) * K * f);
      tmpQ.setFromAxisAngle(Y_AXIS, q.rot ?? 0);
      tmpM.compose(tmpP, tmpQ, tmpS);
      m.setMatrixAt(i, tmpM);
      base.array[i] = q.base * K;
      top.array[i] = q.top * K;
    });
    m.count = part.length;
    m.instanceMatrix.needsUpdate = true;
    base.needsUpdate = top.needsUpdate = true;
  }
  solidLook();
  const R0 = (SOLID[type] ?? TYPES[type].meshNear).R0;
  solidUniforms.sheet.value = TYPES[type]?.meshNear?.sheet ? 1 : 0;
  // fractus: torn outlines
  solidUniforms.lumpAmp.value = TYPES[type]?.meshNear?.ragged ? 0.32 : 0.1;
  const wispy = !!TYPES[type]?.meshNear?.shreds;
  solidUniforms.wispy.value = wispy ? 1 : 0;
  // see-through shreds blend over one another instead of hiding each other
  if (solidMaterial.depthWrite === wispy) { solidMaterial.depthWrite = !wispy; solidMaterial.needsUpdate = true; }
  solidUniforms.bumpM.value = R0 * 0.09;
  solidUniforms.lumpM.value = R0 * 0.3;
}

// Stratocumulus and altocumulus near the observer: flattened ellipsoids on
// a grid lined up with the rolls (the type's meshNear: across x along and
// thickness, metres), wherever the layer's pattern beats the threshold
// (layer.js cloudinessSampler), bigger and thicker where it beats it more,
// so they follow the same rolls and gaps as the pattern drawn beyond them.
// Altocumulus castellanus adds turrets on its thicker elements. Out to about
// 3 degrees above the horizon; the flat pattern takes over from there (see
// the layer surface's nearM).
const SHEET = { nearDeg: 3, maxM: 20000 };
const Y_AXIS = new THREE.Vector3(0, 1, 0);
// (also no farther than about 15,000 grid points reach, so small elements
// do not swamp the GPU)
const sheetNearM = (L) => {
  const g = TYPES[L.type].meshNear;
  const cap = Math.sqrt((15000 * g.across * g.along) / Math.PI);
  return Math.min(SHEET.maxM, cap, Math.max(3000, Math.max(1, L.base - EYE_M) / Math.tan(SHEET.nearDeg * DEG)));
};
function sheetSpheres(L, T) {
  const t = TYPES[L.type];
  const g = t.meshNear;
  const h = Math.max(1, L.base - EYE_M);
  const near = sheetNearM(L);
  const at = cloudinessSampler({ ...L, height: h });
  // the pattern's axes on the ground: across the rolls and along them
  const ang = t.angle * DEG;
  const ux = Math.cos(ang), uy = -Math.sin(ang), vx = Math.sin(ang), vy = Math.cos(ang);
  const na = Math.ceil(near / g.across), nb = Math.ceil(near / g.along);
  const full = 0.5 * t.amount; // how far past the threshold is full thickness
  const frac = (v) => v - Math.floor(v);
  const list = [];
  for (let i = -na; i <= na; i++) {
    for (let j = -nb; j <= nb; j++) {
      const h1 = frac(Math.sin(i * 12.9898 + j * 78.233) * 43758.5453);
      const h2 = frac(Math.sin(i * 39.3468 + j * 11.135) * 24634.6345);
      const a = (i + 0.6 * (h1 - 0.5)) * g.across, b = (j + 0.6 * (h2 - 0.5)) * g.along;
      const x = a * ux + b * vx, y = a * uy + b * vy;
      const d = Math.hypot(x, y);
      if (d > near) continue;
      const c = at(x, y) - T;
      if (c <= 0) continue;
      const th = Math.min(1, c / full);
      const up = h - (d * d) / (2 * EARTH_R);
      const rv = g.thick * (0.35 + 0.9 * th);
      const grow = 0.75 + 0.45 * th;
      const top = up + 2 * g.thick + (g.turrets ? 4 * g.thick : 0);
      if (g.ragged) {
        // fractus: a few small blobs scattered round the point, of uneven
        // size, so the edges come out torn
        const n = 1 + Math.floor(h1 * 2.99);
        for (let k = 0; k < n; k++) {
          const r1 = frac(Math.sin((i + 0.37 * k) * 91.17 + j * 47.3) * 13758.21);
          const r2 = frac(Math.sin(i * 23.71 + (j + 0.61 * k) * 63.9) * 29481.37);
          const r3 = frac(Math.sin((i - k) * 7.13 + j * 17.9) * 51327.77);
          const sz = (0.35 + 0.45 * r3) * grow;
          const ox = (r1 - 0.5) * 0.9 * g.across, oy = (r2 - 0.5) * 0.9 * g.along;
          const bv = rv * (0.6 + 0.6 * r3);
          if (g.shreds) {
            // a shred: long and thin, at its own height (up to a quarter of
            // the base height above or below it), not cut flat
            const z = up + (r1 - 0.5) * 0.5 * h + bv;
            list.push({
              e: x + ox * ux + oy * vx, n: y + ox * uy + oy * vy, up: z,
              rh: g.across * sz, rh2: g.along * sz * (0.8 + 0.8 * r2), rv: bv,
              base: z - 2 * bv, top: z + bv, detail: true, rot: -ang + 0.3 * (r3 - 0.5),
            });
            continue;
          }
          list.push({
            e: x + ox * ux + oy * vx, n: y + ox * uy + oy * vy, up: up + 0.5 * bv + 0.4 * rv * r1,
            rh: g.across * sz, rh2: g.along * sz, rv: bv, base: up, top, detail: true, rot: -ang,
          });
        }
        continue;
      }
      list.push({
        e: x, n: y, up: up + 0.45 * rv,
        rh: 0.8 * g.across * grow, rh2: 0.75 * g.along * grow, rv,
        base: up, top, detail: true, rot: -ang,
      });
      // a turret on the thicker elements: narrower, and up to twice as tall
      // as it is wide
      if (g.turrets && th > 0.45) {
        const tw = 0.3 * g.across * (0.7 + 0.6 * h1);
        const tv = tw * (1 + 1.6 * (th - 0.45) / 0.55) * (0.6 + 0.8 * h2);
        list.push({
          e: x + 0.1 * g.across * (h2 - 0.5), n: y, up: up + 0.9 * rv + 0.8 * tv,
          rh: tw, rh2: tw, rv: tv, base: up, top, detail: true, rot: 0,
        });
        list.push({
          e: x, n: y, up: up + 0.9 * rv + 1.7 * tv,
          rh: 0.75 * tw, rh2: 0.75 * tw, rv: 0.6 * tv, base: up, top, detail: true, rot: 0,
        });
      }
    }
  }
  return list;
}
// The solid clouds' colours: the type's, with the sunlit side warmer and
// everything a little dimmer when the sun is low.
function solidLook() {
  const look = SOLID_LOOK[activeLayer().type];
  if (!look) return;
  const low = Math.max(0, 1 - sunState.el / (15 * DEG));
  const warm = [255, 206, 160];
  const lit = look.lit.map((v, i) => v + (warm[i] - v) * 0.55 * low);
  const dim = 1 - 0.25 * low;
  const set = (u, rgb, k = 1) => u.value.setRGB((rgb[0] * k) / 255, (rgb[1] * k) / 255, (rgb[2] * k) / 255, THREE.SRGBColorSpace);
  set(solidUniforms.litCol, lit);
  set(solidUniforms.shadeCol, look.shade, dim);
  set(solidUniforms.baseCol, look.base, dim);
}
// sunlit tops, shaded sides and bases (sRGB)
const SOLID_LOOK = {
  CU: { lit: [252, 252, 253], shade: [198, 204, 214], base: [150, 157, 168] },
  TCU: { lit: [250, 251, 253], shade: [188, 195, 206], base: [132, 139, 151] },
  CB: { lit: [238, 241, 246], shade: [158, 165, 178], base: [84, 90, 101] },
  SC: { lit: [236, 239, 244], shade: [176, 182, 193], base: [140, 147, 158] },
  AC: { lit: [246, 248, 251], shade: [198, 204, 214], base: [168, 175, 186] },
  ACC: { lit: [246, 248, 251], shade: [192, 198, 209], base: [160, 167, 179] },
  CF: { lit: [250, 251, 253], shade: [200, 206, 216], base: [160, 167, 178] },
  SF: { lit: [168, 173, 181], shade: [146, 151, 160], base: [126, 131, 140] },
};

// Point the shared layer uniforms at the current layer.
let layerThreshold = 0;
const flatType = (code) => !!TYPES[code] && TYPES[code].flat !== false;
// The type's look at the current base (some types change with it).
const typeLook = (code, baseM) => ({ ...TYPES[code], ...TYPES[code].vary?.(baseM / FT) });
// The quiz and the Cloud layer tab both show a cloud layer: the quiz's own,
// or the tab's.
const layerScene = () => mode === 'layer' || mode === 'quiz';
const activeLayer = () => (mode === 'quiz' && quizLayer ? quizLayer : layerParams);
const shownLayerOctas = () => (mode === 'quiz' ? octasShown : layerOctasShown);
const layerTint = () => {
  const L = activeLayer();
  return flatType(L.type) ? typeLook(L.type, L.base).grey : tintFor(L.base);
};
function updatePattern() {
  const p = activeLayer();
  const U = layerUniforms;
  U.shapeKind.value = { slice: 0, lying: 1, cap: 2 }[p.shape];
  U.shapeAz.value = p.azimuth * DEG;
  U.sliceCentre.value = vertSliceCentre(shownLayerOctas(), p.elevation);
  const [a, e] = patchCentre(p.azimuth, p.elevation);
  U.capDir.value.set(Math.sin(a) * Math.cos(e), Math.cos(a) * Math.cos(e), Math.sin(e));
  U.layerT.value = Math.max(-1e4, Math.min(1e4, layerThreshold));
  U.layerHeight.value = Math.max(1, p.base - EYE_M);
  const on = flatType(p.type);
  const t = on ? typeLook(p.type, p.base) : null;
  U.typeOn.value = on ? 1 : 0;
  if (on) {
    U.tileM.value = t.tileM;
    U.featureM.value = t.featureM;
    U.amount.value = t.amount;
    U.blotch.value = t.blotch;
    U.soft.value = t.soft;
    U.texAngle.value = t.angle * DEG;
    U.opacity.value = t.opacity;
    U.shadeAmt.value = t.shade;
    U.wisp.value = t.wisp ?? 0;
    U.towerOn.value = t.topM ? 1 : 0;
    U.topM.value = t.topM ?? 1000;
    U.anvil.value = t.anvil ?? 0;
    U.profile.value = t.profile ?? 2;
    const key = `${p.type}:${p.seed}`;
    if (cloudTexKey !== key) {
      cloudTex.image.data.set(bakeType(p.type, p.seed).data);
      cloudTex.needsUpdate = true;
      cloudTexKey = key;
    }
    if (blotchSeed !== p.seed) {
      blotchTex.image.data.set(bakeBlotches(p.seed).data);
      blotchTex.needsUpdate = true;
      blotchSeed = p.seed;
    }
  } else {
    U.opacity.value = 1;
    U.shadeAmt.value = 0;
    U.wisp.value = 0;
    U.towerOn.value = 0;
  }
  const tint = layerTint();
  domeUniforms.cloudColor.value.setRGB(tint[0] / 255, tint[1] / 255, tint[2] / 255, THREE.SRGBColorSpace);
  layerSurface.material.uniforms.color.value.setRGB(tint[0] / 255, tint[1] / 255, tint[2] / 255, THREE.SRGBColorSpace);
}

function updateCloudMeshes() {
  const on = layerScene();
  const solid = on && !!TYPES[activeLayer().type]?.solid;
  const meshNear = on && !!TYPES[activeLayer().type]?.meshNear;
  domeUniforms.layerOn.value = on && !solid ? 1 : 0;
  domeUniforms.solidOn.value = solid ? 1 : 0;
  updateSolidMeshes(solid || meshNear);
  // where the near clouds are meshes, the flat layer starts beyond them
  layerSurface.material.uniforms.nearM.value = meshNear ? sheetNearM(activeLayer()) : 0;
  if (!on) domeUniforms.cloudColor.value.setRGB(CLOUD[0] / 255, CLOUD[1] / 255, CLOUD[2] / 255, THREE.SRGBColorSpace);
  // The observer's eye is the dome's centre: the figure (to scale, its eye
  // 0.078 units up) stands one eye height below it, as does the ground.
  observer.scale.setScalar((EYE_M * K) / 0.078);
  observer.position.y = GROUND_Y;
  layerSurface.visible = false;
  if (!on) return;
  updatePattern();
  const reach = sightRange(activeLayer().base);
  // rebuild the curved disc only when its size changes noticeably
  if (Math.abs(layerSurface.userData.radius - reach) > reach * 0.01) {
    layerSurface.geometry.dispose();
    layerSurface.geometry = curvedDisc(reach * K, 192, 64);
    layerSurface.userData.radius = reach;
  }
  layerSurface.position.y = GROUND_Y + activeLayer().base * K;
  // Clouds with height are drawn on the dome (traced through their height);
  // the flat layer then shows only their bases, from outside.
  layerSurface.visible = !solid && !(layerUniforms.towerOn.value === 1 && view === 'inside');
}

// ================================================================ cameras

const orbit = new OrbitControls(camera, canvas);
orbit.enableDamping = true;
orbit.minDistance = 1.2;
orbit.maxDistance = 7; // widened in cloud-layer mode
// The camera never goes below the ground: it orbits a point above the
// ground, never below that point's level, and cannot be panned.
orbit.maxPolarAngle = Math.PI / 2 - 0.01;
orbit.enablePan = false;
// The controls apply zooms (and some moves) inside their own event handlers,
// so redraw whenever they report a change, not only when the frame loop sees
// one.
orbit.addEventListener('change', () => { needsRender = true; });

const look = { yaw: 0, pitch: 25 * DEG, fov: 80, drag: null };

// The camera flies between views. While it flies, neither the orbit
// controls nor the inside look-around drive it.
let camFlying = false;

// Default outside pose per scene: what it looks at, how far out and up the
// camera sits, its field of view, and its default heading.
const OUTSIDE = {
  octas: { target: new THREE.Vector3(0, 0.25, 0), out: 3.19, up: 1.25, fov: 45, heading: Math.atan2(-2.1, 2.4) },
  // the same view of the dome as the Celestial dome tab (scaled up for
  // higher étages)
  layer: { target: new THREE.Vector3(0, 0.25, 0), out: 3.19, up: 1.25, fov: 45, heading: Math.atan2(-2.1, 2.4) },
};
// Outside the cloud layer the view is framed for a base up to about 4,500 ft
// (0.3 dome radii); a higher étage scales the whole framing up, so its clouds
// stay in view. The framing follows the étage's typical base, not the base
// slider, so moving the base within an étage never moves the camera.
const layerViewScale = () => Math.max(1, (ETAGE_BASE_FT[activeLayer().etage] * FT * K) / 0.3);
let viewScaleShown = 1; // the scale the camera is framed for
let viewScaleWanted = 1; // eases to the new étage's scale
// heading: compass azimuth (radians) the camera faces
function cameraHeading() {
  const f = new THREE.Vector3();
  camera.getWorldDirection(f);
  return Math.atan2(f.x, -f.z);
}
function setOrbitLimits() {
  // far enough to see the whole layer out to the line of sight, or the
  // planet around the dome
  orbit.maxDistance = layerScene() ? 150 : 40;
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
function setView(v, { yaw, pitch = 25 * DEG, reset = false, instant = false } = {}) {
  const from = view;
  view = v;
  dirty = true; // what is drawn can depend on the view (clouds with height)
  // on a phone or tablet the observer view follows the device
  if (v === 'inside' && hasMotionSensors && !motion.on) setMotion(true);
  if (v !== 'inside' && motion.on) setMotion(false);
  $$('.view-switch [data-view]').forEach((b) => b.classList.toggle('on', b.dataset.view === v));
  orbit.enabled = false;
  const ms = instant ? 0 : 1200;
  // keep facing the same way while moving in or out
  const heading = yaw ?? (from === 'inside' ? look.yaw : cameraHeading());
  if (v === 'outside') {
    setOrbitLimits();
    setObserverVisible(true);
    const o = OUTSIDE[layerScene() ? 'layer' : 'octas'];
    const h = reset ? o.heading : heading;
    const sc = layerScene() ? layerViewScale() : 1;
    viewScaleShown = sc;
    viewScaleWanted = sc;
    tweens.delete('viewScale');
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
    look.pitch = pitch;
    look.fov = OBSERVER_FOV; // fixed: wider views distort too much
    flyTo(new THREE.Vector3(0, 0, 0), dirFromAzEl(look.yaw, look.pitch), look.fov, ms, () => {
      const inside = view === 'inside';
      setObserverVisible(!inside);
    });
    $('#view-hint').textContent = motion.on ? 'Move the device to look around' : 'Drag to look around';
  }
  // The sky is tinted more strongly from inside. There the cloud layer itself
  // lines up with its projection on the dome, so the projection fades out.
  const a0 = domeUniforms.skyAlpha.value, a1 = v === 'inside' ? 0.62 : 0.2;
  const p0 = domeUniforms.projection.value, p1 = v === 'inside' ? 0 : 1;
  // ... and the view turns stereographic (see renderStereo)
  const s0 = stereoUniforms.stereo.value, s1 = v === 'inside' ? 1 : 0;
  animate('sky', ms, (e) => {
    domeUniforms.skyAlpha.value = a0 + (a1 - a0) * e;
    domeUniforms.projection.value = p0 + (p1 - p0) * e;
    stereoUniforms.stereo.value = s0 + (s1 - s0) * e;
  });
}

// Switching tabs from outside: keep the direction the camera looks from, and
// reframe it for the new scene's scale.
function keepOutsideAngle() {
  const dir = camera.position.clone().sub(orbit.target).normalize();
  const o = OUTSIDE[layerScene() ? 'layer' : 'octas'];
  const sc = layerScene() ? layerViewScale() : 1;
  viewScaleShown = viewScaleWanted = sc;
  tweens.delete('viewScale');
  setOrbitLimits();
  orbit.target.copy(o.target).multiplyScalar(sc);
  camera.position.copy(orbit.target).addScaledVector(dir, Math.hypot(o.out, o.up) * sc);
  camera.fov = o.fov;
  camera.updateProjectionMatrix();
  orbit.update(); // also applies the new limits (e.g. stay above the ground)
  needsRender = true;
}

canvas.addEventListener('pointerdown', (ev) => {
  if (view !== 'inside') return;
  // a tap may be needed to start the sensors (iOS asks only from a tap)
  if (hasMotionSensors && !motion.on) setMotion(true);
  if (motion.on && motion.has) return;
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
// The observer's field of view is fixed (no zoom): any wider and the
// projection distorts the sky noticeably.
const OBSERVER_FOV = 80;

// ---------------------------------------------------------------- device motion
//
// In the observer view, on a phone or tablet, the view follows the device
// (no button): hold it up and look around the sky. Until the sensors give a
// reading (or if they are refused), dragging still works. The device's orientation
// (alpha, beta, gamma: the deviceorientation event) becomes the camera's,
// as three.js's old DeviceOrientationControls did. With a compass heading
// (Android's deviceorientationabsolute, iOS's webkitCompassHeading) the
// view's north is the real north; without one, north is wherever the
// device pointed when the sensor started.
const motion = { on: false, has: false, pending: false, refused: false, q: new THREE.Quaternion() };
const Z_AXIS = new THREE.Vector3(0, 0, 1);
const motionEuler = new THREE.Euler();
const qScreen = new THREE.Quaternion();
const qUp = new THREE.Quaternion(-Math.SQRT1_2, 0, 0, Math.SQRT1_2); // camera looks out of the back of the device
const motionEvent = 'ondeviceorientationabsolute' in window ? 'deviceorientationabsolute' : 'deviceorientation';
function onOrientation(ev) {
  let alpha = ev.alpha;
  if (typeof ev.webkitCompassHeading === 'number') alpha = 360 - ev.webkitCompassHeading; // iOS
  if (alpha == null || ev.beta == null || ev.gamma == null) return;
  const orient = (screen.orientation?.angle ?? window.orientation ?? 0) * DEG;
  motionEuler.set(ev.beta * DEG, alpha * DEG, -ev.gamma * DEG, 'YXZ');
  motion.q.setFromEuler(motionEuler).multiply(qUp).multiply(qScreen.setFromAxisAngle(Z_AXIS, -orient));
  motion.has = true;
  needsRender = true;
}
const hasMotionSensors = typeof DeviceOrientationEvent !== 'undefined'
  && ('ontouchstart' in window || navigator.maxTouchPoints > 0);
async function setMotion(on) {
  if (on && (motion.pending || motion.refused)) return;
  if (on && typeof DeviceOrientationEvent.requestPermission === 'function') {
    // iOS asks first, and only from a tap (asked otherwise, it throws: the
    // next tap on the view asks again); a refusal is final
    motion.pending = true;
    let answer = 'error';
    try { answer = await DeviceOrientationEvent.requestPermission(); } catch { /* not from a tap */ }
    motion.pending = false;
    if (answer === 'denied') motion.refused = true;
    if (answer !== 'granted') on = false;
    if (on && view !== 'inside') return; // left the observer view meanwhile
  }
  motion.on = on;
  motion.has = false;
  if (on) window.addEventListener(motionEvent, onOrientation);
  else window.removeEventListener(motionEvent, onOrientation);
  if (on) look.drag = null;
  if (view === 'inside') $('#view-hint').textContent = on ? 'Move the device to look around' : 'Drag to look around';
  needsRender = true;
}

// ---------------------------------------------------------------- stereographic view
//
// A flat screen cannot show part of a sphere without distortion. Ordinary
// perspective stretches things toward the edges of the view (by 1/cos of the
// angle off-centre), so a round cloud turns oval as you look away from it.
// Inside the dome the view is stereographic instead: it keeps shapes, so a
// circle on the sky stays a circle wherever it is on screen (things only grow
// a little toward the edges). The scene is drawn into a cube map around the
// camera, then each screen pixel looks up its direction:
//   perspective:    angle off-centre = atan(r tan(fov/2))
//   stereographic:  angle off-centre = 2 atan(r tan(fov/4))
// with r the distance from the screen centre (1 at the top edge). `stereo`
// blends the two, so flying in and out changes the view smoothly.
const cubeTarget = new THREE.WebGLCubeRenderTarget(1024, { type: THREE.HalfFloatType });
const cubeCamera = new THREE.CubeCamera(1e-4, 1e4, cubeTarget);
const stereoUniforms = {
  cube: { value: cubeTarget.texture },
  right: { value: new THREE.Vector3() },
  up: { value: new THREE.Vector3() },
  forward: { value: new THREE.Vector3() },
  aspect: { value: 1 },
  fov: { value: 80 * DEG },
  stereo: { value: 0 },
};
const stereoScene = new THREE.Scene();
const stereoCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
stereoScene.add(new THREE.Mesh(
  new THREE.PlaneGeometry(2, 2),
  new THREE.ShaderMaterial({
    uniforms: stereoUniforms,
    depthTest: false,
    depthWrite: false,
    vertexShader: `
      varying vec2 vXY;
      void main() {
        vXY = position.xy;
        gl_Position = vec4(position.xy, 0.0, 1.0);
      }`,
    fragmentShader: `
      uniform samplerCube cube;
      uniform vec3 right, up, forward;
      uniform float aspect, fov, stereo;
      varying vec2 vXY;
      void main() {
        vec2 p = vec2(vXY.x * aspect, vXY.y);
        float r = length(p);
        float persp = atan(r * tan(0.5 * fov));
        float stereog = 2.0 * atan(r * tan(0.25 * fov));
        float th = mix(persp, stereog, stereo);
        vec2 u = r > 1e-6 ? p / r : vec2(0.0);
        vec3 dir = cos(th) * forward + sin(th) * (u.x * right + u.y * up);
        gl_FragColor = textureCube(cube, dir);
        #include <colorspace_fragment>
      }`,
  }),
));
// the cube map holds linear colour at half-float precision
cubeTarget.texture.colorSpace = THREE.LinearSRGBColorSpace;

function renderStereo() {
  camera.updateMatrixWorld();
  const m = camera.matrixWorld.elements;
  stereoUniforms.right.value.set(m[0], m[1], m[2]).normalize();
  stereoUniforms.up.value.set(m[4], m[5], m[6]).normalize();
  stereoUniforms.forward.value.set(-m[8], -m[9], -m[10]).normalize();
  stereoUniforms.aspect.value = camera.aspect;
  stereoUniforms.fov.value = camera.fov * DEG;
  cubeCamera.position.copy(camera.position);
  cubeCamera.update(renderer, scene);
  renderer.render(stereoScene, stereoCamera);
}

function updateInsideCamera() {
  camera.position.set(0, 0.0, 0);
  camera.fov = look.fov;
  camera.updateProjectionMatrix();
  if (motion.on && motion.has) {
    // follow the device, smoothed a little against sensor jitter
    camera.quaternion.slerp(motion.q, 0.35);
    if (camera.quaternion.angleTo(motion.q) > 1e-3) needsRender = true;
    // keep the drag view in step, for when motion is turned off
    const d = new THREE.Vector3(0, 0, -1).applyQuaternion(camera.quaternion);
    look.pitch = Math.asin(Math.max(-1, Math.min(1, d.y)));
    look.yaw = Math.atan2(d.x, -d.z);
    return;
  }
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
  const cc = layerScene() ? layerTint() : CLOUD;
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

// The report, under the sky map: the octas as reported and the sky
// condition (with the base and CB/TCU on the cloud layer); the exact amount
// on hover. (In the quiz, map and report stay hidden until answered.)
function updateReadout() {
  const el = $('#readout');
  const rep = reportedOctas(coveredFrac);
  let cond = skyTerm(rep);
  if (layerScene() && rep > 0) {
    const L = activeLayer();
    cond = skyGroup(rep, Math.round(L.base / FT)) + (L.type === 'CB' || L.type === 'TCU' ? L.type : '');
  }
  el.innerHTML = `<span class="big">${rep}/8</span><span class="term">${cond}</span>`;
  $('.skymap').title = `${fmtOctas(coveredFrac)} octas exactly (${(coveredFrac * 100).toFixed(1)}% of the sky)`;
}
// MANOBS table 7-2: the amount term for a layer's summation amount
const skyTerm = (octas) => (octas === 0 ? 'SKC' : octas <= 2 ? 'FEW' : octas <= 4 ? 'SCT' : octas < 8 ? 'BKN' : 'OVC');


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
  if (layerScene()) {
    if (mode === 'layer') followEtage();
    const L = activeLayer();
    // The GPU draws the layer; the same field here gives the threshold for
    // the chosen octas, the numbers and the sky map. Lines of sight start at
    // the observer's eye, 1.7 m above the ground.
    const params = { ...L, octas: shownLayerOctas(), height: Math.max(1, L.base - EYE_M) };
    if (TYPES[L.type]?.solid) {
      // solid clouds: the field is minus the reach at which each direction
      // is first covered; the threshold gives the reach. Where even all the
      // clouds cannot cover the amount asked, they cover what they can, and
      // the numbers say so.
      solidState = solidField(params, field);
      let T = solveThreshold(field, shownLayerOctas() / 8);
      if (!(T > -REACH_MAX)) T = -REACH_MAX;
      solidScale = Number.isFinite(T) ? Math.max(0, -T) : 0;
      threshold = Number.isFinite(T) ? T : 1;
      coveredFrac = coverFromField(field, threshold, cover);
      paintDome(true);
    } else {
      solidState = null;
      layerField(params, field);
      layerThreshold = solveThreshold(field, shownLayerOctas() / 8);
      coveredFrac = coverFromField(field, layerThreshold, cover);
      paintDome(false);
    }
    updateCloudMeshes();
  } else {
    const p = shapeParams;
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

function frame(now) {
  // a tween that finishes this frame still needs its last step drawn
  if (tweens.size) needsRender = true;
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
  if (!camFlying && view === 'outside' && layerScene()) {
    // follow an étage change: scale the view about the observer, keeping the
    // angle the user chose
    const sc = viewScaleWanted;
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
  // Only draw when something changed: the cloud layer is costly per pixel,
  // and a still scene needs no redrawing.
  if (needsRender) {
    needsRender = false;
    if (stereoUniforms.stereo.value > 0.001) renderStereo();
    else renderer.render(scene, camera);
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

function syncShape() {
  $$('#shape button').forEach((b) => b.classList.toggle('on', b.dataset.shape === shapeParams.shape));
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


$$('#guides button').forEach((b) => b.addEventListener('click', () => setGuides(b.dataset.guides)));

// ---------------------------------------------------------------- layer tab

// Heights are given in feet, as MANOBS does, with metres in brackets;
// everything is stored in metres.
// The base range (ft): the type's, from MANOBS; any height for a plain patch.
const PLAIN_RANGE_FT = [100, 39000];
const baseRangeFt = (type) => TYPES[type]?.baseRangeFt ?? PLAIN_RANGE_FT;
// The étage follows the base.
const etageOf = (ft) => (ft <= 6500 ? 'low' : ft < 20000 ? 'mid' : 'high');
const heightRange = () => {
  const [lo, hi] = baseRangeFt(layerParams.type);
  return [lo, hi, 100];
};
const fmtHeight = (m) => `${(Math.round(m / FT / 10) * 10).toLocaleString('en-CA')} ft
  (${(Math.round(m / 10) * 10).toLocaleString('en-CA')} m)`;

function bindHeight(id, key) {
  const input = $(`#${id}`);
  const out = $(`#${id}-out`);
  const per = () => FT;
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

// Under the base slider, a little histogram of the bases observers report
// for the type (climate.js), lined up with the slider.
function drawBaseFreq() {
  const cv = $('#lbase-freq');
  const c = CLIMATE[layerParams.type];
  const w = cv.clientWidth, h = 14, dpr = window.devicePixelRatio || 1;
  if (!w) return;
  cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr);
  const g = cv.getContext('2d');
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, w, h);
  cv.style.visibility = c ? 'visible' : 'hidden';
  if (!c) return;
  const [lo, hi] = baseRangeFt(layerParams.type);
  const pad = 8; // the thumb's half width: where the slider's ends are
  const x = (ft) => pad + ((ft - lo) / (hi - lo)) * (w - 2 * pad);
  const top = Math.max(...c.baseHist.map(([, k]) => k));
  g.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--freq-bar').trim() || 'rgba(47,111,235,0.55)';
  for (const [bin, k] of c.baseHist) {
    if (bin + 500 <= lo || bin >= hi) continue;
    const x0 = x(Math.max(lo, bin)), x1 = x(Math.min(hi, bin + 500));
    const bh = Math.max(1, Math.sqrt(k / top) * (h - 2));
    g.fillRect(x0, h - bh, Math.max(1, x1 - x0 - 0.5), bh);
  }
}
window.addEventListener('resize', () => drawBaseFreq());

const layerCtl = {
  base: bindHeight('lbase', 'base'),
  azimuth: bindRange('laz', layerParams, 'azimuth', (v) => `${v}°`),
  centre: bindRange('lcentre', layerParams, 'elevation', (v) => `${v}°`),
};
// Coverage seen on the dome, in whole octas.
const lchips = $('#lcover-chips');
for (let k = 0; k <= 8; k++) {
  const b = document.createElement('button');
  b.textContent = k;
  b.addEventListener('click', () => setLayerOctas(k));
  lchips.append(b);
}
function setLayerOctas(k) {
  layerParams.octas = k;
  syncLayerChips();
  const from = layerOctasShown;
  animate('layerOctas', 700, (e) => { layerOctasShown = from + (k - from) * e; dirty = true; });
}
// For a cloud type, each amount's button is filled from below by how often
// observers report it (climate.js); amounts never reported, or that solid
// clouds cannot reach (8: separate heaps never close every gap), are off.
function syncLayerChips() {
  const code = layerParams.type;
  const c = CLIMATE[code];
  const top = c ? Math.max(...c.octas) : 1;
  [...lchips.children].forEach((b, k) => {
    b.classList.toggle('on', layerParams.octas === k);
    const f = c && k > 0 ? c.octas[k - 1] : null;
    b.classList.toggle('freq', f !== null);
    b.style.setProperty('--f', f === null ? 0 : f / top);
    b.disabled = k > 0 && ((f !== null && f < 0.05) || (k === 8 && !!TYPES[code]?.solid));
    b.title = f === null ? '' : b.disabled ? 'not reported, or not possible for this type'
      : `${f.toFixed(1)}% of ${code} reports`;
  });
  $('#lcover-out').textContent = `${layerParams.octas}/8`;
}

// Typical base for each étage (ft; frames the outside view).
const ETAGE_BASE_FT = { low: 4000, mid: 12000, high: 28000 };
// The amount reported most often for a type (climate.js), 1..8.
const usualOctas = (code) => {
  const c = CLIMATE[code];
  return c ? c.octas.indexOf(Math.max(...c.octas)) + 1 : null;
};
// What Canadian observers report for a type, in a sentence.
function climateLine(code) {
  const c = CLIMATE[code];
  if (!c) return '';
  const ft = (v) => v.toLocaleString('en-CA');
  const most = usualOctas(code);
  return `In Canadian METARs (2023–24, ${ft(c.n)} reports with it as the only layer), its base is usually
    ${ft(c.base[1])}–${ft(c.base[3])} ft and it most often covers ${most} octa${most > 1 ? 's' : ''}
    (${Math.round(c.octas[most - 1])}% of reports).`;
}
function syncLayerControls() {
  const t = TYPES[layerParams.type];
  $$('#ltypes button').forEach((b) => b.classList.toggle('on', b.dataset.type === layerParams.type));
  $('#ltype-out').textContent = t ? `${layerParams.type} · ${t.name}` : 'plain patch';
  $('#type-info').hidden = !t;
  if (t) {
    $('#type-std').textContent = t.help;
    $('#type-emp').textContent = climateLine(layerParams.type);
  }
  layerCtl.base.sync();
  drawBaseFreq();
  syncLayerChips();
  dirty = true;
}
// The layer glides to a new base; the octas stay.
function glideBase(ft, type = layerParams.type) {
  const [lo, hi] = baseRangeFt(type);
  const to = { base: Math.min(hi, Math.max(lo, ft)) * FT };
  baseTween = { from: { base: layerParams.base }, to, start: performance.now(), duration: REDUCED_MOTION ? 1 : 1500 };
}
// The étage follows the base. When it changes, the outside view reframes
// for the new étage (not for every change of base), and the notes follow.
function followEtage() {
  const etage = etageOf(layerParams.base / FT);
  if (etage === layerParams.etage) return;
  layerParams.etage = etage;
  const s0 = viewScaleWanted, s1 = layerViewScale();
  animate('viewScale', 1500, (e) => { viewScaleWanted = s0 + (s1 - s0) * e; });
}

// Cloud types, by étage. Picking one fades the old pattern out and the new
// one in, and moves the layer to the type's usual base (MANOBS); the base
// can then be set anywhere in the type's range.
const typeBox = $('#ltypes');
const typeButton = (code, label, title) => {
  const b = document.createElement('button');
  b.dataset.type = code;
  b.innerHTML = label;
  b.title = title;
  b.addEventListener('click', () => setType(code));
  return b;
};
{
  const row = document.createElement('div');
  row.className = 'type-row';
  row.innerHTML = '<span class="type-etage"></span>';
  const chips = document.createElement('div');
  chips.className = 'type-chips';
  chips.append(typeButton('plain', 'Plain patch', 'A plain round patch'));
  row.append(chips);
  typeBox.append(row);
}
for (const [etage, codes] of TYPE_ROWS) {
  const row = document.createElement('div');
  row.className = 'type-row';
  row.innerHTML = `<span class="type-etage">${ETAGES[etage].name}</span>`;
  const chips = document.createElement('div');
  chips.className = 'type-chips';
  for (const code of codes) {
    const t = TYPES[code];
    chips.append(typeButton(code, code, t.name));
  }
  row.append(chips);
  typeBox.append(row);
}
function setType(code) {
  if (code === layerParams.type) return;
  const t = TYPES[code];
  if (t) {
    // the most likely amount and base for this type
    glideBase(t.baseFt, code);
    const k = usualOctas(code);
    if (k) setLayerOctas(TYPES[code].solid ? Math.min(7, k) : k);
  }
  animate('typeFade', 250, (e) => { layerUniforms.layerFade.value = 1 - e; }, () => {
    layerParams.type = code;
    syncLayerControls();
    dirty = true;
    animate('typeFade', 350, (e) => { layerUniforms.layerFade.value = e; });
  });
}
// Bake every type's pattern in the background, so picking one is instant.
setTimeout(() => {
  const idle = window.requestIdleCallback ?? ((f) => setTimeout(f, 50));
  prebake(layerParams.seed, idle);
  bakeBlotches(layerParams.seed);
}, 1500);
syncLayerControls();

// ---------------------------------------------------------------- sun
//
// The sun lights the solid clouds and shows in the sky. Its place comes from
// a latitude (50 N unless the browser gives a location), a date (a season's
// equinox or solstice, or any date) and the time of day, from sunrise to
// sunset. Without a longitude, times are local solar time.
const SEASONS = { spring: [2, 20], summer: [5, 21], autumn: [8, 22], winter: [11, 21] }; // month (0-based), day
const sunState = { lat: 50, lon: null, date: null, t: 0.4, el: 40 * DEG, az: 220 * DEG, season: 'summer' };
const pad2 = (v) => String(v).padStart(2, '0');
function fmtClock(minutesUtc) {
  const d = sunState.date;
  if (sunState.lon === null) {
    // local solar time: noon when the sun is due south
    const m = ((minutesUtc % 1440) + 1440) % 1440;
    return `${pad2(Math.floor(m / 60))}:${pad2(Math.floor(m % 60))}`;
  }
  const at = new Date(Date.UTC(d.getUTCFullYear(), d.getUTCMonth(), d.getUTCDate(), 0, 0) + minutesUtc * 60000);
  return at.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}
function updateSun() {
  const s = sunState;
  const { decl, eot } = solarDay(dayOfYear(s.date));
  const h0 = halfDay(s.lat, decl);
  const timeInput = $('#sun-time');
  timeInput.disabled = h0 === 0;
  const h = -h0 + s.t * 2 * h0;
  const pos = sunPosition(s.lat, decl, h);
  s.el = pos.el;
  s.az = pos.az;
  // without a longitude, clock = local solar time (noon at h = 0)
  const clock = (hh) => fmtClock(s.lon === null ? 720 + (hh / DEG) * 4 : utcMinutes(hh, s.lon, eot));
  if (h0 === 0) {
    $('#sun-out').textContent = 'below the horizon all day';
    $('#sun-rise').textContent = $('#sun-set').textContent = '';
  } else {
    $('#sun-out').textContent = `${clock(h)} · ${Math.round(pos.el / DEG)}° up, ${Math.round(pos.az / DEG)}°`;
    $('#sun-rise').textContent = h0 >= Math.PI ? 'midnight sun' : `sunrise ${clock(-h0)}`;
    $('#sun-set').textContent = h0 >= Math.PI ? '' : `sunset ${clock(h0)}`;
  }
  // east, north, up
  const e = Math.sin(pos.az) * Math.cos(pos.el), n = Math.cos(pos.az) * Math.cos(pos.el), u = Math.sin(pos.el);
  domeUniforms.sunEnu.value.set(e, n, u).normalize();
  SUN.copy(toScene(e, n, u)).normalize();
  solidLook();
  needsRender = true;
}
function setSunDate(date, season = null) {
  sunState.date = date;
  sunState.season = season;
  $$('#season button').forEach((b) => b.classList.toggle('on', b.dataset.season === season));
  $('#sun-date').value = `${date.getUTCFullYear()}-${pad2(date.getUTCMonth() + 1)}-${pad2(date.getUTCDate())}`;
  updateSun();
}
const seasonDate = (k) => new Date(Date.UTC(new Date().getFullYear(), SEASONS[k][0], SEASONS[k][1], 12));
$$('#season button').forEach((b) => b.addEventListener('click', () => setSunDate(seasonDate(b.dataset.season), b.dataset.season)));
$('#sun-date').addEventListener('change', (ev) => {
  const [y, m, d] = ev.target.value.split('-').map(Number);
  if (y && m && d) setSunDate(new Date(Date.UTC(y, m - 1, d, 12)));
});
$('#sun-time').addEventListener('input', (ev) => {
  sunState.t = parseFloat(ev.target.value) / 1000;
  updateSun();
});
function showPlace() {
  const s = sunState;
  $('#sun-place').textContent = s.lon === null
    ? `${s.lat}° N · local solar time`
    : `${Math.abs(s.lat).toFixed(2)}° ${s.lat >= 0 ? 'N' : 'S'}, ${Math.abs(s.lon).toFixed(2)}° ${s.lon >= 0 ? 'E' : 'W'} · local clock time`;
}
$('#sun-locate').addEventListener('click', () => {
  if (!navigator.geolocation) { $('#sun-place').textContent = 'This browser cannot give a location.'; return; }
  $('#sun-place').textContent = 'Asking for your location…';
  navigator.geolocation.getCurrentPosition((p) => {
    sunState.lat = Math.round(p.coords.latitude * 100) / 100;
    sunState.lon = Math.round(p.coords.longitude * 100) / 100;
    showPlace();
    // today, at this place
    const now = new Date();
    setSunDate(new Date(Date.UTC(now.getFullYear(), now.getMonth(), now.getDate(), 12)));
  }, () => {
    showPlace();
    $('#sun-place').textContent += ' (Location not available.)';
  }, { maximumAge: 3600000, timeout: 15000 });
});
showPlace();
setSunDate(seasonDate('summer'), 'summer');

// ---------------------------------------------------------------- quiz
//
// A random sky: one cloud layer of a random type, at a base drawn around its
// usual one, covering a random amount (not a whole number of octas, so the
// rounding rules matter). You start inside, looking straight up with the
// whole sky in view, as an observer turning round would see it; the guides,
// the outside view and the sky map wait until you have answered.
const QUIZ_TYPES = Object.keys(TYPES).filter(flatType);
const QUIZ_LOOK = { pitch: 45 * DEG };
const answers = $('#quiz-answers');
const quizTypes = $('#quiz-types');
const pick = { amount: null, type: null };
for (let k = 0; k <= 8; k++) {
  const b = document.createElement('button');
  b.textContent = k;
  b.addEventListener('click', () => choose('amount', k));
  answers.append(b);
}
for (const code of QUIZ_TYPES) {
  const b = document.createElement('button');
  b.textContent = code;
  b.dataset.type = code;
  b.title = TYPES[code].name;
  b.addEventListener('click', () => choose('type', code));
  quizTypes.append(b);
}
function choose(what, v) {
  if (quiz.answered) return;
  pick[what] = pick[what] === v ? null : v;
  [...answers.children].forEach((b, k) => b.classList.toggle('on', pick.amount === k));
  [...quizTypes.children].forEach((b) => b.classList.toggle('on', pick.type === b.dataset.type));
  $('#quiz-check').disabled = pick.amount === null;
}

// a random index, with the given weights
function pickWeighted(rnd, weights) {
  let r = rnd() * weights.reduce((a, b) => a + b, 0), i = 0;
  while (i < weights.length - 1 && r > weights[i]) r -= weights[i++];
  return i;
}
function newQuiz(instant = false) {
  const rnd = mulberry32(Math.floor(Math.random() * 1e9));
  // Skies as Canadian observers report them (climate.js): the type by how
  // often it is reported (evened out a little, so rare types still come
  // up), then its amount and base from its own reports. Now and then a
  // clear sky.
  const type = QUIZ_TYPES[pickWeighted(rnd, QUIZ_TYPES.map((c) => Math.sqrt(CLIMATE[c]?.n ?? 100)))];
  const t = TYPES[type];
  const clim = CLIMATE[type];
  const n = rnd() < 0.04 ? 0 : clim ? 1 + pickWeighted(rnd, clim.octas) : 1 + Math.floor(rnd() * 8);
  let frac;
  // a true amount that reports as n (MANOBS rounds up: n - 1 < 8 frac <= n)
  if (n === 0) frac = 0;
  else if (n === 8) frac = 1;
  else frac = (n - 1 + 0.03 + rnd() * 0.94) / 8;
  // base: from the type's reported bases (500 ft bins), within the MANOBS range
  let ft = t.baseFt;
  if (clim) {
    const bin = clim.baseHist[pickWeighted(rnd, clim.baseHist.map(([, k]) => k))][0];
    ft = Math.round((bin + rnd() * 500) / 100) * 100;
  }
  ft = Math.min(t.baseRangeFt[1], Math.max(t.baseRangeFt[0], ft));
  const next = {
    shape: 'cap', type, octas: frac * 8, base: ft * FT, etage: etageOf(ft), seed: 1,
    azimuth: Math.floor(rnd() * 360), elevation: 15 + rnd() * 150,
  };
  quiz.answered = false;
  pick.amount = pick.type = null;
  [...answers.children].forEach((b) => { b.disabled = false; b.className = ''; });
  [...quizTypes.children].forEach((b) => { b.disabled = false; b.className = ''; });
  $('#quiz-check').disabled = true;
  $('#quiz-result').innerHTML = '';
  baseShown = false;
  const show = () => {
    quizLayer = next;
    showQuizBase();
    octasShown = next.octas;
    tweens.delete('octas');
    dirty = true;
  };
  // a new sky fades in over the old one
  if (quizLayer && mode === 'quiz' && !instant && !REDUCED_MOTION) {
    animate('typeFade', 300, (e) => { layerUniforms.layerFade.value = 1 - e; }, () => {
      show();
      animate('typeFade', 400, (e) => { layerUniforms.layerFade.value = e; });
    });
  } else {
    show();
    layerUniforms.layerFade.value = 1;
  }
  setView('inside', { yaw: rnd() * 2 * Math.PI, instant, ...QUIZ_LOOK });
  syncQuizLock();
}

// The base, as a ceilometer would give it, is a hint for the type: hidden
// behind a button until asked for (or answered).
let baseShown = false;
function showQuizBase() {
  const box = $('#quiz-base');
  if (!baseShown && !quiz.answered) {
    box.innerHTML = '<button id="quiz-hint" title="What a ceilometer would read">Hint: cloud base</button>';
    $('#quiz-hint').addEventListener('click', () => { baseShown = true; showQuizBase(); });
    return;
  }
  box.innerHTML = quizLayer.octas > 0
    ? `Cloud base <strong>${fmtHeight(quizLayer.base)}</strong>`
    : 'Cloud base: none detected';
}

// The METAR sky condition group: amount and base in hundreds of feet,
// rounded down (to 100 ft up to 10,000 ft, then to 1,000 ft).
function skyGroup(octas, ft) {
  if (octas === 0) return 'SKC';
  const step = ft <= 10000 ? 100 : 1000;
  return `${skyTerm(octas)}${String(Math.floor(ft / step) * step / 100).padStart(3, '0')}`;
}

function answerQuiz() {
  if (quiz.answered || !quizLayer || pick.amount === null) return;
  quiz.answered = true;
  const truth = reportedOctas(coveredFrac);
  const k = pick.amount;
  const amountOk = k === truth;
  quiz.total++;
  if (amountOk) quiz.right++;
  // the type is scored when you named one and there was cloud to name
  const typeAsked = pick.type !== null && truth > 0;
  const typeOk = typeAsked && pick.type === quizLayer.type;
  if (typeAsked) { quiz.typeTotal++; if (typeOk) quiz.typeRight++; }
  quiz.streak = amountOk && (!typeAsked || typeOk) ? quiz.streak + 1 : 0;
  [...answers.children].forEach((b, i) => {
    b.disabled = true;
    b.className = i === truth ? 'right' : i === k ? 'wrong' : '';
  });
  [...quizTypes.children].forEach((b) => {
    b.disabled = true;
    const code = b.dataset.type;
    b.className = truth > 0 && code === quizLayer.type ? 'right' : code === pick.type ? 'wrong' : '';
  });
  $('#quiz-check').disabled = true;
  showQuizBase();
  const off = k - truth;
  const t = TYPES[quizLayer.type];
  const ft = Math.round(quizLayer.base / FT);
  const amountLine = amountOk
    ? `<span class="ok">Amount correct.</span>`
    : `<span class="no">Amount ${off > 0 ? 'too high' : 'too low'} by ${Math.abs(off)}.</span>`;
  const typeLine = truth === 0 ? ''
    : typeAsked
      ? (typeOk ? `<span class="ok">Type correct:</span> ${quizLayer.type}, ${t.name.toLowerCase()}.`
        : `<span class="no">Type:</span> it was ${quizLayer.type}, ${t.name.toLowerCase()}, not ${pick.type}.`)
      : `Type: ${quizLayer.type}, ${t.name.toLowerCase()}.`;
  $('#quiz-result').innerHTML = `${amountLine} ${fmtOctas(coveredFrac)} octas exactly, reported as
    ${truth} (${skyTerm(truth)}).<br>${typeLine}
    ${truth > 0 ? `Sky condition: <code>${skyGroup(truth, ft)}</code>` : ''}`;
  $('#quiz-score').textContent = `Amount ${quiz.right} / ${quiz.total}`
    + (quiz.typeTotal ? ` · type ${quiz.typeRight} / ${quiz.typeTotal}` : '') + ` · streak ${quiz.streak}`;
  syncQuizLock();
  updateReadout();
}
$('#quiz-check').addEventListener('click', answerQuiz);
$('#quiz-new').addEventListener('click', () => newQuiz());

// ---------------------------------------------------------------- tabs, view

const canvasFade = (o) => { canvas.style.opacity = o; };
// Until a quiz sky is answered the observer stays inside the dome, and the
// sky map (which would give the answer away) stays hidden.
function syncQuizLock() {
  const locked = mode === 'quiz' && !quiz.answered;
  const out = $('.view-switch [data-view="outside"]');
  out.disabled = locked;
  out.title = locked ? 'Answer first: then the sky cam opens' : '';
  $('.skymap').classList.toggle('hidden', locked);
  // no guide lines in the real sky
  guideGroup.visible = !locked;
  needsRender = true;
}
$$('.tabs button').forEach((b) => b.addEventListener('click', () => {
  const tab = b.dataset.tab;
  $$('.tabs button').forEach((x) => x.classList.toggle('on', x === b));
  $$('.tab').forEach((s) => { s.hidden = s.id !== `tab-${tab}`; });
  if (tab === 'layer') requestAnimationFrame(drawBaseFreq); // now it has a width
  const prev = mode;
  let next = mode;
  if (tab === 'octas' || tab === 'layer' || tab === 'quiz') next = tab;
  if (next === prev) return;
  // Azimuth and elevation are shared by the Celestial dome and Cloud layer
  // tabs: carry them over.
  if (next === 'layer') {
    layerCtl.azimuth.set(shapeParams.azimuth);
    layerCtl.centre.set(shapeParams.elevation);
    syncLayerControls();
  } else if (next === 'octas' && prev === 'layer') {
    domeCtl.azimuth.set(layerParams.azimuth);
    domeCtl.elevation.set(layerParams.elevation);
  }
  // a different scene (and scale): fade through
  canvasFade(0);
  setTimeout(() => {
    mode = next;
    if (mode === 'quiz' && !quizLayer) newQuiz(true);
    octasShown = mode === 'quiz' ? quizLayer.octas : shapeParams.octas;
    morph = null;
    tweens.delete('octas');
    if (mode === 'quiz' && !quiz.answered) setView('inside', { instant: true, ...QUIZ_LOOK });
    else if (view === 'outside') keepOutsideAngle();
    recompute();
    needsRender = true;
    syncQuizLock();
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
