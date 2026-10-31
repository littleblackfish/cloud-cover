import * as THREE from 'three';
import { OrbitControls } from '../vendor/OrbitControls.js';
import {
  W, H, N, grid, computeField, solveThreshold, coverFromField,
  reportedOctas, makeNoise3, mulberry32, vertSliceFrame, vertSliceCentre, patchCentre,
} from './sky.js';
import {
  layerField, layerArea, bakeType, bakeBlotches, prebake, TYPES, TYPE_ROWS,
  BLOTCH_SCALE, BLOTCH_ANGLE, EARTH_R, DECODE_GLSL,
} from './layer.js';

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
  cloudTex: { value: cloudTex },
  blotchTex: { value: blotchTex },
};
const LAYER_GLSL = `
  uniform int shapeKind, typeOn;
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
      uniform float projection;
      varying vec2 vUv;
      varying vec3 vDir;
      #include <logdepthbuf_pars_fragment>
      ${LAYER_GLSL}
      void main() {
        #include <logdepthbuf_fragment>
        float a, shade = 0.0;
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
          float th = layerShade(g);
          a = s.z < 0.0 ? 0.0 : projection * layerAlpha(th) * layerCloud(r, rw, g, gx, gy);
          shade = shadeAmt * th;
        } else {
          float f = texture2D(field, vUv).r;
          float w = max(fwidth(f), 1e-5);
          a = smoothstep(threshold - w, threshold + w, f);
        }
        gl_FragColor = vec4(mix(skyColor, cloudColor * (1.0 - shade), a), mix(skyAlpha, cloudAlpha, a));
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
// the horizon is the observer's, and the dome already marks it.
//
// FAR_M is a cutoff, not drawn: cloud-layer clouds farther than this from the
// observer are drawn only as a trace. They sit within about a degree of the
// horizon, too low to judge. 66 km is a working figure rather than a
// standard one: a 4,000 ft cloud base 66 km away is only about 0.8 degrees
// above the horizon.
const FAR_M = 66000;
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
  uniform float farM; // cloud cutoff distance in metres
  varying vec2 vGround;
  ${LAYER_GLSL}
  void main() {
    #include <logdepthbuf_fragment>
    // the line of sight from the eye to this point of the (curved) layer
    float d = length(vGround);
    vec3 s = normalize(vec3(vGround, layerHeight - d * d / 12742000.0));
    float r = region(s), rw = fwidth(r);
    float a = s.z < 0.0 ? 0.0 : layerCloud(r, rw, vGround, dFdx(vGround), dFdy(vGround));
    // Only the cloud patch is drawn. Full brightness inside the dome,
    // dimming with distance beyond it; past the cutoff (farM) only a trace
    // of it shows.
    if (a < 0.01) discard;
    float dist = length(vec3(vGround, layerHeight));
    float far = smoothstep(1.0, 4.0, dist / metresPerUnit);
    float beyond = smoothstep(0.92, 1.08, dist / farM);
    float th = layerShade(vGround);
    vec3 c = color * (1.0 - shadeAmt * th);
    gl_FragColor = vec4(c * mix(1.0, 0.6, far), mix(mix(0.95, 0.75, far), 0.1, beyond) * a * layerAlpha(th));
    #include <colorspace_fragment>
  }`;
const layerSurface = new THREE.Mesh(
  new THREE.BufferGeometry(),
  new THREE.ShaderMaterial({
    uniforms: { ...layerUniforms, metresPerUnit, farM: { value: FAR_M }, color: { value: new THREE.Color(0xeef1f5) } },
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

// Point the shared layer uniforms at the current layer.
let layerThreshold = 0;
const flatType = (code) => !!TYPES[code] && TYPES[code].flat !== false;
// The type's look at the current base (some types change with it).
const typeLook = (code, baseM) => ({ ...TYPES[code], ...TYPES[code].vary?.(baseM / FT) });
const layerTint = () => (flatType(layerParams.type) ? typeLook(layerParams.type, layerParams.base).grey : tintFor(layerParams.base));
function updatePattern() {
  const p = layerParams;
  const U = layerUniforms;
  U.shapeKind.value = { slice: 0, lying: 1, cap: 2 }[p.shape];
  U.shapeAz.value = p.azimuth * DEG;
  U.sliceCentre.value = vertSliceCentre(layerOctasShown, p.elevation);
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
  }
  const tint = layerTint();
  domeUniforms.cloudColor.value.setRGB(tint[0] / 255, tint[1] / 255, tint[2] / 255, THREE.SRGBColorSpace);
  layerSurface.material.uniforms.color.value.setRGB(tint[0] / 255, tint[1] / 255, tint[2] / 255, THREE.SRGBColorSpace);
}

function updateCloudMeshes() {
  const on = mode === 'layer';
  domeUniforms.layerOn.value = on ? 1 : 0;
  if (!on) domeUniforms.cloudColor.value.setRGB(CLOUD[0] / 255, CLOUD[1] / 255, CLOUD[2] / 255, THREE.SRGBColorSpace);
  // The observer's eye is the dome's centre: the figure (to scale, its eye
  // 0.078 units up) stands one eye height below it, as does the ground.
  observer.scale.setScalar((EYE_M * K) / 0.078);
  observer.position.y = GROUND_Y;
  layerSurface.visible = false;
  if (!on) return;
  updatePattern();
  const reach = sightRange(layerParams.base);
  // rebuild the curved disc only when its size changes noticeably
  if (Math.abs(layerSurface.userData.radius - reach) > reach * 0.01) {
    layerSurface.geometry.dispose();
    layerSurface.geometry = curvedDisc(reach * K, 192, 64);
    layerSurface.userData.radius = reach;
  }
  layerSurface.position.y = GROUND_Y + layerParams.base * K;
  layerSurface.visible = true;
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
const layerViewScale = () => Math.max(1, (ETAGE_BASE_FT[layerParams.etage] * FT * K) / 0.3);
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
  orbit.maxDistance = mode === 'layer' ? 150 : 40;
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
    const o = OUTSIDE[mode === 'layer' ? 'layer' : 'octas'];
    const h = reset ? o.heading : heading;
    const sc = mode === 'layer' ? layerViewScale() : 1;
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
    look.pitch = 25 * DEG;
    look.fov = 80;
    flyTo(new THREE.Vector3(0, 0, 0), dirFromAzEl(look.yaw, look.pitch), look.fov, ms, () => {
      const inside = view === 'inside';
      setObserverVisible(!inside);
    });
    $('#view-hint').textContent = 'Drag to look around · scroll to change field of view';
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
  const o = OUTSIDE[mode === 'layer' ? 'layer' : 'octas'];
  const sc = mode === 'layer' ? layerViewScale() : 1;
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
  look.fov = Math.max(30, Math.min(150, look.fov + ev.deltaY * 0.05));
  needsRender = true;
}, { passive: false });

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
  const cc = mode === 'layer' ? layerTint() : CLOUD;
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
  // On the cloud layer, also the cloud's real size: the same octas take a
  // bigger cloud the higher the layer is.
  const size = mode === 'layer' && patchArea > 0
    ? `<div class="dim">cloud on the layer: ${fmtArea(patchArea)}
        (as a circle, ${fmtKm(2 * Math.sqrt(patchArea / Math.PI))} across)</div>`
    : '';
  el.innerHTML = `<div class="dim">Report</div>
    <div class="big">${rep} / 8 <span class="term">${skyTerm(rep)}</span></div>
    <div>${fmtOctas(coveredFrac)} octas exactly</div>
    <div class="dim">${(coveredFrac * 100).toFixed(1)}% of the dome</div>${size}`;
}
let patchArea = 0;
// MANOBS table 7-2: the amount term for a layer's summation amount
const skyTerm = (octas) => (octas === 0 ? 'SKC' : octas <= 2 ? 'FEW' : octas <= 4 ? 'SCT' : octas < 8 ? 'BKN' : 'OVC');
const fmtArea = (m2) => {
  const km2 = m2 / 1e6;
  return `${km2 >= 100 ? Math.round(km2).toLocaleString('en-CA') : km2 >= 10 ? km2.toFixed(0) : km2.toFixed(1)} km²`;
};


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
    followEtage();
    // The GPU draws the layer; the same field here gives the threshold for
    // the chosen octas, the numbers and the sky map. Lines of sight start at
    // the observer's eye, 1.7 m above the ground.
    layerField({
      ...layerParams, octas: layerOctasShown, height: Math.max(1, layerParams.base - EYE_M),
    }, field);
    layerThreshold = solveThreshold(field, layerOctasShown / 8);
    coveredFrac = coverFromField(field, layerThreshold, cover);
    patchArea = layerArea(cover, Math.max(1, layerParams.base - EYE_M));
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

function frame(now) {
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


$$('#guides button').forEach((b) => b.addEventListener('click', () => setGuides(b.dataset.guides)));

// ---------------------------------------------------------------- layer tab

// Cloud base height is shown in feet or metres; everything is
// stored in metres. Horizontal distances stay metric.
let heightUnit = 'ft';
// The base range (ft): the type's, from MANOBS; any height for a plain patch.
const PLAIN_RANGE_FT = [100, 39000];
const baseRangeFt = (type) => TYPES[type]?.baseRangeFt ?? PLAIN_RANGE_FT;
// The étage follows the base.
const etageOf = (ft) => (ft <= 6500 ? 'low' : ft < 20000 ? 'mid' : 'high');
const heightRange = () => {
  const [lo, hi] = baseRangeFt(layerParams.type);
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
function syncLayerChips() {
  [...lchips.children].forEach((b, k) => b.classList.toggle('on', layerParams.octas === k));
  $('#lcover-out').textContent = `${layerParams.octas}/8`;
}
$$('#units button').forEach((b) => b.addEventListener('click', () => {
  heightUnit = b.dataset.units;
  $$('#units button').forEach((x) => x.classList.toggle('on', x === b));
  layerCtl.base.sync();
}));

// Typical base for each étage (ft; frames the outside view), and the clouds
// you would find there.
const ETAGE_BASE_FT = { low: 4000, mid: 12000, high: 28000 };
const GENUS = {
  low: 'Low étage: cumulus (Cu), stratocumulus (Sc), stratus (St); cumulonimbus (Cb) and nimbostratus (Ns) bases are low too.',
  mid: 'Mid étage: altocumulus (Ac) and altostratus (As).',
  high: 'High étage: cirrus (Ci), cirrocumulus (Cc) and cirrostratus (Cs).',
};

function syncLayerControls() {
  $('#genus').textContent = GENUS[layerParams.etage];
  const t = TYPES[layerParams.type];
  $$('#ltypes button').forEach((b) => b.classList.toggle('on', b.dataset.type === layerParams.type));
  $('#ltype-out').textContent = t ? `${layerParams.type} · ${t.name}` : 'plain patch';
  $('#type-help').textContent = t ? t.help
    : 'A plain round patch: pick a cloud type to see how it looks from the ground.';
  layerCtl.base.sync();
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
  $('#genus').textContent = GENUS[etage];
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
    const b = typeButton(code, code, t.name);
    if (t.flat === false) {
      b.disabled = true;
      b.title = `${t.name}: needs height to draw, coming next`;
    }
    chips.append(b);
  }
  row.append(chips);
  typeBox.append(row);
}
function setType(code) {
  if (code === layerParams.type) return;
  const t = TYPES[code];
  if (t) glideBase(t.baseFt, code);
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
  quizParams = {
    octas: frac * 8,
    shape: shapes[Math.floor(rnd() * shapes.length)],
    azimuth: Math.floor(rnd() * 360),
    elevation: rnd() * 180,
    // plain shapes only, for now
    modifier: 'none',
    amount: 0,
    form: 0,
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
$$('.tabs button').forEach((b) => b.addEventListener('click', () => {
  const tab = b.dataset.tab;
  $$('.tabs button').forEach((x) => x.classList.toggle('on', x === b));
  $$('.tab').forEach((s) => { s.hidden = s.id !== `tab-${tab}`; });
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
  if ((prev === 'layer') === (next === 'layer')) {
    // octas <-> quiz: same scene, so the sky morphs from one to the other
    startMorph();
    mode = next;
    if (mode === 'quiz' && !quizParams) newQuiz();
    else easeOctasTo(mode === 'quiz' ? quizParams.octas : shapeParams.octas);
    if (mode === 'quiz' && !quiz.answered && view === 'outside') setView('inside');
    syncQuizLock();
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
