import * as THREE from 'three';
import { OrbitControls } from '../vendor/OrbitControls.js';
import {
  W, H, N, grid, computeField, solveThreshold, coverFromField, bandFractions,
  reportedOctas, makeNoise3, mulberry32, OCTA_BAND_ELEVATIONS, vertSliceFrame, vertSliceBoundaries,
} from './sky.js';
import {
  castLayer, nearbyClouds, sheetField, sheetGeometry, solvePlanCover, distanceToHeight, EARTH_R,
} from './layer.js';

const DEG = Math.PI / 180;
const FT = 0.3048; // metres per foot
const $ = (s) => document.querySelector(s);
const $$ = (s) => [...document.querySelectorAll(s)];

// ================================================================ state

const shapeParams = {
  octas: 3, shape: 'slice', azimuth: 0, centerElevation: 90, axisElevation: 0, clumps: 6,
  deform: 0, scale: 3, time: 0, seed: 1, animate: false,
};
const layerParams = {
  etage: 'low', form: 'cumuliform',
  // octas: chosen amount; coverMode says whether it is the amount seen on the
  // dome or the map-view amount. cover: resulting map-view fraction.
  octas: 3, coverMode: 'dome', cover: 3 / 8,
  base: 4000 * FT, thickness: 3000 * FT, size: 1500, extent: 400,
  curved: true, // Earth curvature: always on
  seed: 1, showClouds: true,
};
let quizParams = null;
const quiz = { answered: false, right: 0, total: 0, streak: 0 };

let mode = 'octas'; // 'octas' | 'layer' | 'quiz'
let view = 'outside';
let guides = 'quadrants';
let mapProjection = 'equal-area';
let dirty = true;

const field = new Float32Array(N);
const cover = new Float32Array(N);
let threshold = 0;
let coveredFrac = 0;
let layerGeo = null;
const noiseCache = new Map();
const noiseFor = (seed) => {
  if (!noiseCache.has(seed)) noiseCache.set(seed, makeNoise3(seed));
  return noiseCache.get(seed);
};

// ================================================================ three.js scene

const canvas = $('#view');
// The cloud-layer scene spans 100 ft to 100 km, so depth needs a log buffer.
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, logarithmicDepthBuffer: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0f1622);
const camera = new THREE.PerspectiveCamera(45, 1, 0.005, 1e5);

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
// The dome is shaded from the cloudiness field itself (half-float texture,
// linearly interpolated) so cloud outlines stay crisp however far you zoom.
const fieldData = new Uint16Array(N);
const fieldTexture = new THREE.DataTexture(fieldData, W, H, THREE.RedFormat, THREE.HalfFloatType);
fieldTexture.wrapS = THREE.RepeatWrapping;
fieldTexture.magFilter = THREE.LinearFilter;
fieldTexture.minFilter = THREE.LinearFilter;
const domeUniforms = {
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
      void main() {
        vUv = uv;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        #include <logdepthbuf_vertex>
      }`,
    fragmentShader: `
      uniform sampler2D field;
      uniform float threshold, skyAlpha, cloudAlpha;
      uniform vec3 skyColor, cloudColor;
      varying vec2 vUv;
      #include <logdepthbuf_pars_fragment>
      void main() {
        #include <logdepthbuf_fragment>
        float f = texture2D(field, vUv).r;
        float w = max(fwidth(f), 1e-5);
        float a = smoothstep(threshold - w, threshold + w, f);
        gl_FragColor = vec4(mix(skyColor, cloudColor, a), mix(skyAlpha, cloudAlpha, a));
      }`,
  }),
);
dome.renderOrder = 2;
scene.add(dome);

const ground = new THREE.Mesh(
  new THREE.CircleGeometry(1.6, 96),
  new THREE.MeshBasicMaterial({ color: 0x33402f }),
);
ground.rotation.x = -Math.PI / 2;
ground.position.y = -0.003;
scene.add(ground);

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

const guideGroup = new THREE.Group();
scene.add(guideGroup);
// Vert-slice boundaries: half great circles from one end of the slicing axis
// to the other, cut at the horizon. Points are [east, north, up].
function lyingArcs(a0, e0) {
  const { u, p, q } = vertSliceFrame(a0, e0);
  const arcs = [];
  for (const psi of vertSliceBoundaries(a0, e0)) {
    // half-plane direction at angle psi, counted from -q
    const phi = psi - Math.PI / 2;
    const w = [0, 1, 2].map((i) => p[i] * Math.cos(phi) + q[i] * Math.sin(phi));
    let pts = [];
    for (let i = 0; i <= 120; i++) {
      const t = (i / 120) * Math.PI;
      const d = [0, 1, 2].map((j) => u[j] * Math.cos(t) + w[j] * Math.sin(t));
      if (d[2] >= -1e-6) pts.push([d[0], d[1], Math.max(0, d[2])]);
      else if (pts.length) { if (pts.length > 1) arcs.push(pts); pts = []; }
    }
    if (pts.length > 1) arcs.push(pts);
  }
  return arcs;
}
function guideSpec(g) {
  if (g === 'quadrants') return { circles: [30, 60], meridians: [0, 90, 180, 270], arcs: [] };
  if (g === 'slices') return { circles: [], meridians: [0, 45, 90, 135, 180, 225, 270, 315], arcs: [] };
  if (g === 'lying') return { circles: [], meridians: [], arcs: lyingArcs(shapeParams.azimuth, shapeParams.axisElevation) };
  if (g === 'bands') return { circles: OCTA_BAND_ELEVATIONS.slice(1, 8), meridians: [], arcs: [] };
  return { circles: [], meridians: [], arcs: [] };
}
function buildGuides() {
  guideGroup.clear();
  const { circles, meridians, arcs } = guideSpec(guides);
  for (const e of circles) guideGroup.add(circleLine(e * DEG, 1.004, 0xffd27a, 0.75));
  for (const a of meridians) guideGroup.add(meridianLine(a * DEG, 1.004, 0xffd27a, 0.75));
  for (const pts of arcs) {
    guideGroup.add(new THREE.Line(
      new THREE.BufferGeometry().setFromPoints(pts.map(([e, n, u]) => toScene(e, n, u).multiplyScalar(1.004))),
      new THREE.LineBasicMaterial({ color: 0xffd27a, transparent: true, opacity: 0.75, depthWrite: false }),
    ));
  }
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

// ---------------------------------------------------------------- cloud layer
//
// In cloud-layer mode the scene is to scale around the observer: the dome's
// radius is 100 ft, so it sits between the observer (about 6 ft tall) and
// the lowest possible cloud. The observer's eye is 1.7 m above the ground and
// the scene is drawn only out to their horizon, sqrt(2 R h) = 4.65 km away,
// with the ground and the clouds following the Earth's curve. The projection
// on the dome includes the whole layer, out to the horizon.

const SCENE_M = 100 * FT; // metres per scene unit: dome radius = 100 ft
const K = 1 / SCENE_M;
const EYE_M = 1.7;
const REACH_M = Math.sqrt(2 * EARTH_R * EYE_M); // distance to the horizon
const REACH = REACH_M * K;
const GROUND_Y = -EYE_M * K; // the scene's origin is the observer's eye
// how far the Earth's surface has curved away below the observer's level, at
// horizontal distance d metres (scene units)
const drop = (d) => ((d * d) / (2 * EARTH_R)) * K;

// A horizontal disc of radius r (scene units) that follows the Earth's curve.
function curvedDisc(r, segments) {
  const g = new THREE.CircleGeometry(r, segments, 0, 2 * Math.PI);
  g.rotateX(-Math.PI / 2);
  const pos = g.attributes.position;
  for (let i = 0; i < pos.count; i++) {
    pos.setY(i, -drop(Math.hypot(pos.getX(i), pos.getZ(i)) / K));
  }
  return g;
}

const groundLayer = new THREE.Mesh(curvedDisc(REACH, 160), ground.material);
groundLayer.position.y = GROUND_Y;
groundLayer.visible = false;
scene.add(groundLayer);

// Étage limits in feet, as used for reporting.
const ETAGES = {
  low: { name: 'Low', base: [300, 6500], band: [0, 6500] },
  mid: { name: 'Mid', base: [6600, 19000], band: [6500, 20000] },
  high: { name: 'High', base: [20000, 40000], band: [20000, 45000] },
};

const layerDecor = new THREE.Group();
scene.add(layerDecor);
const etageLabels = {};

function labelSprite(text, size) {
  const c = document.createElement('canvas');
  c.width = 512;
  c.height = 128;
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  // fixed on-screen size, so labels stay readable at any zoom
  const s = new THREE.Sprite(new THREE.SpriteMaterial({
    map: t, depthWrite: false, depthTest: false, transparent: true, sizeAttenuation: false,
  }));
  s.center.set(0, 0.5);
  s.scale.set(size * 4, size, 1);
  s.renderOrder = 5;
  s.userData.draw = (active) => {
    const g = c.getContext('2d');
    g.clearRect(0, 0, 512, 128);
    g.font = `${active ? 700 : 500} 56px system-ui, sans-serif`;
    g.textBaseline = 'middle';
    g.fillStyle = active ? '#ffd27a' : '#c4cfdb';
    g.fillText(text, 8, 64);
    t.needsUpdate = true;
  };
  s.userData.draw(false);
  return s;
}

// A height mast beside the observer with the étage boundaries marked.
const MAST = dirFromAzEl(320 * DEG, 0, 50).setY(GROUND_Y); // 5,000 ft away, behind the dome as first seen
const etageMarker = new THREE.Mesh(
  new THREE.CylinderGeometry(0.25, 0.25, 1, 12),
  new THREE.MeshBasicMaterial({ color: 0xffd27a, transparent: true, opacity: 0.6 }),
);
{
  const top = 45000 * FT * K;
  layerDecor.add(new THREE.Line(
    new THREE.BufferGeometry().setFromPoints([MAST.clone(), MAST.clone().setY(GROUND_Y + top)]),
    new THREE.LineBasicMaterial({ color: 0xaec6e0, transparent: true, opacity: 0.8 }),
  ));
  layerDecor.add(etageMarker);
  for (const ft of [6500, 20000]) {
    const y = ft * FT * K;
    const disc = new THREE.Mesh(
      curvedDisc(REACH, 96),
      new THREE.MeshBasicMaterial({ color: 0xaec6e0, transparent: true, opacity: 0.035, side: THREE.DoubleSide, depthWrite: false }),
    );
    disc.position.y = GROUND_Y + y;
    const tick = labelSprite(`${ft.toLocaleString('en-CA')} ft`, 0.035);
    tick.position.copy(MAST).setY(GROUND_Y + y);
    layerDecor.add(disc, tick);
  }
  for (const [key, e] of Object.entries(ETAGES)) {
    const label = labelSprite(`${e.name} étage`, 0.045);
    label.position.copy(MAST).setY(GROUND_Y + ((e.band[0] + Math.min(e.band[1], 40000)) / 2) * FT * K);
    etageLabels[key] = label;
    layerDecor.add(label);
  }
}

function updateEtageDecor() {
  const [lo, hi] = ETAGES[layerParams.etage].band;
  const y0 = lo * FT * K, y1 = Math.min(hi, 45000) * FT * K;
  etageMarker.scale.set(1, y1 - y0, 1);
  etageMarker.position.copy(MAST).setY(GROUND_Y + (y0 + y1) / 2);
  for (const [key, label] of Object.entries(etageLabels)) label.userData.draw(key === layerParams.etage);
}

// Cumuliform elements, drawn as instanced cylinders.
const MAX_CLOUDS = 12000;
const cloudMesh = new THREE.InstancedMesh(
  new THREE.CylinderGeometry(1, 1, 1, 16),
  new THREE.MeshLambertMaterial({ color: 0xf2f4f7 }),
  MAX_CLOUDS,
);
cloudMesh.count = 0;
cloudMesh.frustumCulled = false;
scene.add(cloudMesh);

// Stratiform sheet: its base and top surfaces, cut by the same holes. The
// texture stores the cloudiness field; alphaTest at the threshold cuts the
// holes, so changing the cover only moves the threshold.
const SHEET_PX = 1024;
const sheetCanvas = document.createElement('canvas');
sheetCanvas.width = sheetCanvas.height = SHEET_PX;
const sheetTexture = new THREE.CanvasTexture(sheetCanvas);
const sheetMeshes = [0xd9dee5, 0xf6f7f9].map((color) => {
  const m = new THREE.Mesh(
    curvedDisc(REACH, 160),
    new THREE.MeshBasicMaterial({ color, alphaMap: sheetTexture, alphaTest: 0.5, side: THREE.DoubleSide }),
  );
  m.visible = false;
  scene.add(m);
  return m;
});
const FIELD_LO = -1.6, FIELD_SPAN = 3.2; // field values -> 0..1 in the texture

let sheetKey = '';
function drawSheet() {
  const { base, thickness, size, seed } = layerParams;
  const key = `${size}|${seed}`;
  if (key !== sheetKey) {
    sheetKey = key;
    const { field } = sheetField(layerParams);
    const g = sheetCanvas.getContext('2d');
    const img = g.createImageData(SHEET_PX, SHEET_PX);
    for (let py = 0; py < SHEET_PX; py++) {
      const north = (1 - ((py + 0.5) / SHEET_PX) * 2) * REACH_M;
      for (let px = 0; px < SHEET_PX; px++) {
        const east = (((px + 0.5) / SHEET_PX) * 2 - 1) * REACH_M;
        const v = Math.max(0, Math.min(1, (field(east, north) - FIELD_LO) / FIELD_SPAN));
        const o = (py * SHEET_PX + px) * 4;
        img.data[o] = img.data[o + 1] = img.data[o + 2] = 255 * v;
        img.data[o + 3] = 255;
      }
    }
    g.putImageData(img, 0, 0);
    sheetTexture.needsUpdate = true;
  }
  const { T } = sheetGeometry(layerParams);
  const test = T === -Infinity ? 0 : Math.max(0.001, (T - FIELD_LO) / FIELD_SPAN);
  sheetMeshes.forEach((m, i) => {
    if ((m.material.alphaTest > 0) !== (test > 0)) m.material.needsUpdate = true;
    m.material.alphaTest = test;
    m.position.y = GROUND_Y + (i === 0 ? base : base + thickness) * K;
    m.visible = !(i === 1 && thickness < 20);
  });
}

function updateCloudMeshes() {
  const on = mode === 'layer';
  layerDecor.visible = on;
  // to scale: 1.7 m tall, feet on the ground (the figure is 0.086 units tall)
  observer.scale.setScalar(on ? (EYE_M * K) / 0.086 : 1);
  observer.position.y = on ? GROUND_Y : 0;
  ground.visible = !on;
  groundLayer.visible = on;
  cloudMesh.count = 0;
  sheetMeshes.forEach((m) => { m.visible = false; });
  if (!on) return;
  updateEtageDecor();
  if (!layerParams.showClouds || layerParams.cover <= 0) return;
  if (layerParams.form === 'stratiform') {
    drawSheet();
    return;
  }
  const list = nearbyClouds(layerParams, K, REACH_M, MAX_CLOUDS);
  const m = new THREE.Matrix4();
  list.forEach((c, i) => {
    const h = c.top - c.base;
    m.makeScale(c.r, h, c.r);
    m.setPosition(c.x, GROUND_Y + c.base + h / 2 - drop(c.d), -c.y);
    cloudMesh.setMatrixAt(i, m);
  });
  cloudMesh.count = list.length;
  cloudMesh.instanceMatrix.needsUpdate = true;
}

// ================================================================ cameras

const orbit = new OrbitControls(camera, canvas);
orbit.enableDamping = true;
orbit.minDistance = 1.2;
orbit.maxDistance = 7; // widened in cloud-layer mode
orbit.maxPolarAngle = Math.PI / 2 + 0.35;

const look = { yaw: 0, pitch: 25 * DEG, fov: 80, drag: null };

function setView(v, { yaw } = {}) {
  view = v;
  $$('.view-switch [data-view]').forEach((b) => b.classList.toggle('on', b.dataset.view === v));
  observer.visible = v === 'outside';
  for (const label of Object.values(etageLabels)) label.visible = v === 'outside';
  orbit.enabled = v === 'outside';
  if (v === 'outside') {
    resetOutside();
    $('#view-hint').textContent = 'Drag to orbit · scroll to zoom';
  } else {
    look.yaw = yaw ?? 0;
    look.pitch = 25 * DEG;
    look.fov = 80;
    $('#view-hint').textContent = 'Drag to look around · scroll to change field of view';
  }
  camera.updateProjectionMatrix();
}
function resetOutside() {
  if (mode === 'layer') {
    // low, looking up: the 100 ft dome in front, the cloud layer above
    orbit.maxDistance = 600; // the drawn scene ends at the horizon, 152 units out
    orbit.maxPolarAngle = Math.PI / 2 - 0.01; // stay above the ground
    camera.fov = 60;
    camera.position.set(7, 2.6, 8.5);
    orbit.target.set(0, 2, 0);
  } else {
    orbit.maxDistance = 7;
    orbit.maxPolarAngle = Math.PI / 2 + 0.35;
    camera.fov = 45;
    camera.position.set(2.1, 1.5, 2.4);
    orbit.target.set(0, 0.25, 0);
  }
  camera.updateProjectionMatrix();
  orbit.update();
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
});
canvas.addEventListener('pointerup', () => { look.drag = null; });
canvas.addEventListener('wheel', (ev) => {
  if (view !== 'inside') return;
  ev.preventDefault();
  look.fov = Math.max(30, Math.min(120, look.fov + ev.deltaY * 0.05));
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
}
new ResizeObserver(resize).observe(canvas.parentElement);

// ================================================================ painting

const SKY = [120, 170, 225];
const CLOUD = [244, 246, 249];

function paintDome(soft) {
  const skyAlpha = view === 'inside' ? 0.62 : 0.2;
  const cloudAlpha = 0.93;
  domeUniforms.skyAlpha.value = skyAlpha;
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
    d[o] = SKY[0] + (CLOUD[0] - SKY[0]) * a;
    d[o + 1] = SKY[1] + (CLOUD[1] - SKY[1]) * a;
    d[o + 2] = SKY[2] + (CLOUD[2] - SKY[2]) * a;
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

// zenith angle (rad) at normalised map radius r, and the inverse
const projections = {
  'equal-area': { theta: (r) => 2 * Math.asin(r / Math.SQRT2), radius: (t) => Math.SQRT2 * Math.sin(t / 2) },
  fisheye: { theta: (r) => (r * Math.PI) / 2, radius: (t) => t / (Math.PI / 2) },
};

function buildMapLookup() {
  const proj = projections[mapProjection];
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

  const proj = projections[mapProjection];
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
    if (guides !== 'bands') mapCtx.fillText(`${e}°`, c + 4, c - ringR(e) + 16);
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
  if (mode === 'layer') {
    el.innerHTML = `<div class="dim">Seen on the dome</div>
      <div class="big">${rep} / 8</div>
      <div>${fmtOctas(coveredFrac)} octas exactly</div>
      <div class="dim">map-view cover ${fmtOctas(layerGeo.planCover)} octas</div>`;
  } else {
    el.innerHTML = `<div class="dim">Report</div>
      <div class="big">${rep} / 8</div>
      <div>${fmtOctas(coveredFrac)} octas exactly</div>
      <div class="dim">${(coveredFrac * 100).toFixed(1)}% of the dome</div>`;
  }
}

function updateLayerPanel() {
  const geo = layerGeo;
  const rep = reportedOctas(coveredFrac);
  $('#layer-compare').innerHTML = `
    <div><div class="k">Real (map view)</div><div class="v">${fmtOctas(geo.planCover)}/8</div>
      <div class="s">fraction of the ground shaded from straight above</div></div>
    <div><div class="k">Seen on the dome</div><div class="v">${fmtOctas(coveredFrac)}/8</div>
      <div class="s">you would report <strong>${rep}</strong> octas</div></div>`;
  $('#lsize-out').textContent = layerParams.form === 'cumuliform' && geo.r
    ? `${fmtKm(layerParams.size)} · clouds ${fmtKm(2 * geo.r)} wide`
    : fmtKm(layerParams.size);
  const bands = bandFractions(cover);
  const rows = [];
  for (let b = 7; b >= 0; b--) {
    const e0 = OCTA_BAND_ELEVATIONS[b], e1 = OCTA_BAND_ELEVATIONS[b + 1];
    const dNear = distanceToHeight(e1 * DEG, layerParams.base, layerParams.curved);
    const dFar = distanceToHeight(e0 * DEG, layerParams.base, layerParams.curved);
    const pct = bands[b] * 100;
    rows.push(`<tr><td>${e0.toFixed(0)}°–${e1.toFixed(0)}°</td>
      <td>${b === 7 ? '0' : fmtKm(dNear)} – ${fmtKm(dFar)}</td>
      <td><div class="bar-row"><div class="bar" title="${pct.toFixed(0)}% of this band covered">
        <i style="width:${pct}%"></i><b style="left:calc(${geo.planCover * 100}% - 1px)"></b></div>
        <span>${pct.toFixed(0)}%</span></div></td></tr>`);
  }
  $('#layer-bands tbody').innerHTML = rows.join('');
}

// ================================================================ update loop

// Rays start at the observer's eye, 1.7 m above the ground.
const rayParams = (quick) => ({ ...layerParams, base: layerParams.base - EYE_M, quick });

function recompute() {
  if (mode === 'layer') {
    const quick = !!baseTween;
    layerParams.cover = layerParams.coverMode === 'dome'
      ? solvePlanCover(rayParams(quick), layerParams.octas / 8, layerParams.cover)
      : layerParams.octas / 8;
    layerGeo = castLayer(rayParams(quick), cover);
    let s = 0;
    for (let k = 0; k < N; k++) s += cover[k] * grid.weight[k];
    coveredFrac = s;
    paintDome(false);
    updateCloudMeshes();
    updateLayerPanel();
  } else {
    const p = mode === 'quiz' ? quizParams : shapeParams;
    computeField({ ...p, noise: noiseFor(p.seed) }, field);
    threshold = solveThreshold(field, p.octas / 8);
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
  if (mode === 'octas' && shapeParams.animate) {
    shapeParams.time += dt * 0.6;
    dirty = true;
  }
  if (baseTween) {
    const t = Math.min(1, (now - baseTween.start) / baseTween.duration);
    const ease = t < 0.5 ? 2 * t * t : 1 - (-2 * t + 2) ** 2 / 2;
    for (const key of Object.keys(baseTween.to)) {
      layerParams[key] = baseTween.from[key] + (baseTween.to[key] - baseTween.from[key]) * ease;
    }
    if (t >= 1) {
      // a sheet's pattern depends on its patch size, so that changes once, at the end
      if (baseTween.sizeAtEnd) layerCtl.size.set(baseTween.sizeAtEnd);
      baseTween = null;
    }
    layerCtl.base.sync();
    layerCtl.thickness.sync();
    layerCtl.size.set(layerParams.size);
    dirty = true;
  }
  if (dirty) {
    dirty = false;
    recompute();
  }
  if (view === 'inside') updateInsideCamera();
  else orbit.update();
  renderer.render(scene, camera);
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

const octasCtl = bindRange('octas', shapeParams, 'octas', (v) => (Number.isInteger(v) ? v : v.toFixed(2)), () => syncChips());
bindRange('azimuth', shapeParams, 'azimuth', (v) => `${v}°`, () => {
  if (guides === 'lying') buildGuides();
});
bindRange('centre', shapeParams, 'centerElevation', (v) => `${v}°`);
bindRange('axis', shapeParams, 'axisElevation', (v) => `${v}°`, () => {
  if (guides === 'lying') buildGuides();
});
bindRange('clumps', shapeParams, 'clumps', (v) => v);
bindRange('deform', shapeParams, 'deform', (v) => (v === 0 ? 'none' : v.toFixed(2)));
bindRange('scale', shapeParams, 'scale', (v) => (v < 2.5 ? 'large' : v < 5 ? 'medium' : 'small'));

const chips = $('#octa-chips');
for (let k = 0; k <= 8; k++) {
  const b = document.createElement('button');
  b.textContent = k;
  b.addEventListener('click', () => { octasCtl.set(k); syncChips(); dirty = true; });
  chips.append(b);
}
function syncChips() {
  [...chips.children].forEach((b, k) => b.classList.toggle('on', shapeParams.octas === k));
}
syncChips();

const shapeHelp = {
  slice: 'Wedges from horizon to zenith. Each 45° of azimuth is exactly one octa.',
  lying: 'The orange lies on its side. Each slice runs from the north horizon over the sky to the south horizon, a 180° arc. There are four slices on each side of the zenith, each one octa, filling from the east horizon. Direction turns the orange; axis elevation tips it up until, at 90°, it becomes the ordinary slice.',
  cap: 'A round patch. Move its centre down toward the horizon: the horizon clips it and it stretches to keep its area.',
  band: 'Cloud rising from the horizon all round. 4/8 only reaches 30° elevation.',
  clumps: 'Several round clouds of mixed sizes. Change the count: the total area stays the same.',
  patchy: 'Random patches. The pattern changes, the covered area does not.',
};
function syncShape() {
  $$('#shape button').forEach((b) => b.classList.toggle('on', b.dataset.shape === shapeParams.shape));
  $('#shape-help').textContent = shapeHelp[shapeParams.shape];
  const s = shapeParams.shape;
  $('#f-azimuth').hidden = !(s === 'slice' || s === 'lying' || s === 'cap');
  $('#f-centre').hidden = s !== 'cap';
  $('#f-axis').hidden = s !== 'lying';
  $('#f-clumps').hidden = s !== 'clumps';
  $('#f-deform').hidden = s === 'patchy';
}
function setGuides(g) {
  guides = g;
  $$('#guides button').forEach((x) => x.classList.toggle('on', x.dataset.guides === g));
  buildGuides();
  drawSkyMap();
}
$$('#shape button').forEach((b) => b.addEventListener('click', () => {
  shapeParams.shape = b.dataset.shape;
  syncShape();
  // show the matching slice outlines when switching between the two oranges
  if (shapeParams.shape === 'lying' && guides === 'slices') setGuides('lying');
  if (shapeParams.shape === 'slice' && guides === 'lying') setGuides('slices');
  dirty = true;
}));
syncShape();

$('#animate').addEventListener('change', (e) => { shapeParams.animate = e.target.checked; });
$('#reseed').addEventListener('click', () => {
  shapeParams.seed = 1 + Math.floor(Math.random() * 1e6);
  if (shapeParams.deform === 0 && shapeParams.shape !== 'patchy') $('#deform').value = shapeParams.deform = 0.4;
  $('#deform').dispatchEvent(new Event('input'));
});

$$('#guides button').forEach((b) => b.addEventListener('click', () => setGuides(b.dataset.guides)));

// ---------------------------------------------------------------- layer tab

// Heights (cloud base, thickness) are shown in feet or metres; everything is
// stored in metres. Horizontal distances stay metric.
let heightUnit = 'ft';
const THICKNESS_FT = { cumuliform: [100, 35000], stratiform: [50, 10000] };
const SIZE_M = { cumuliform: [100, 5000, 10], stratiform: [300, 20000, 100] };
const MAX_COVER = { cumuliform: 0.9, stratiform: 1 };
const heightRange = (key) => {
  const [lo, hi] = key === 'base' ? ETAGES[layerParams.etage].base : THICKNESS_FT[layerParams.form];
  return heightUnit === 'ft'
    ? [lo, hi, key === 'base' ? 100 : 50]
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
    const [min, max, step] = heightRange(key);
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
  thickness: bindHeight('lthick', 'thickness'),
  size: bindRange('lsize', layerParams, 'size', fmtKm),
  extent: bindRange('lext', layerParams, 'extent', (v) => `${v} km`),
};

// Map-view coverage, in whole octas.
const lchips = $('#lcover-chips');
for (let k = 0; k <= 8; k++) {
  const b = document.createElement('button');
  b.textContent = k;
  b.addEventListener('click', () => {
    layerParams.octas = k;
    syncLayerChips();
    dirty = true;
  });
  lchips.append(b);
}
$$('#lcover-mode button').forEach((b) => b.addEventListener('click', () => {
  layerParams.coverMode = b.dataset.cmode;
  syncLayerChips();
  dirty = true;
}));
function syncLayerChips() {
  const cumuliform = layerParams.form === 'cumuliform';
  [...lchips.children].forEach((b, k) => {
    b.classList.toggle('on', layerParams.octas === k);
    // separate elements can't close every gap: a full layer is a sheet
    b.disabled = cumuliform && k === 8;
    b.title = b.disabled ? 'A fully covered layer is a sheet: choose Stratiform' : '';
  });
  $('#lcover-out').textContent = `${layerParams.octas}/8`;
  $$('#lcover-mode button').forEach((b) => b.classList.toggle('on', b.dataset.cmode === layerParams.coverMode));
}
$$('#units button').forEach((b) => b.addEventListener('click', () => {
  heightUnit = b.dataset.units;
  $$('#units button').forEach((x) => x.classList.toggle('on', x === b));
  layerCtl.base.sync();
  layerCtl.thickness.sync();
}));
$('#lshow').addEventListener('change', (e) => { layerParams.showClouds = e.target.checked; dirty = true; });
$('#lreseed').addEventListener('click', () => { layerParams.seed = 1 + Math.floor(Math.random() * 1e6); dirty = true; });

// Typical layer for each étage and form (heights in ft, size in m).
const LAYER_DEFAULTS = {
  low: {
    cumuliform: { base: 4000, thickness: 3000, size: 1500 },
    stratiform: { base: 1500, thickness: 800, size: 1500 },
  },
  mid: {
    cumuliform: { base: 12000, thickness: 600, size: 800 },
    stratiform: { base: 10000, thickness: 4000, size: 8000 },
  },
  high: {
    cumuliform: { base: 25000, thickness: 200, size: 400 },
    stratiform: { base: 28000, thickness: 1500, size: 10000 },
  },
};
const GENUS = {
  low: {
    cumuliform: 'Cumulus (Cu). Make it thick for towering cumulus (TCU) or cumulonimbus (Cb): bases are low even when tops reach the high étage.',
    stratiform: 'Stratus (St) or a stratocumulus (Sc) sheet. Nimbostratus (Ns) bases are often low too.',
  },
  mid: {
    cumuliform: 'Altocumulus (Ac): small rounded elements, often in rows or patches.',
    stratiform: 'Altostratus (As): a grey or bluish sheet; the sun may show as through ground glass.',
  },
  high: {
    cumuliform: 'Cirrocumulus (Cc): tiny grains or ripples, no shading.',
    stratiform: 'Cirrostratus (Cs): a thin veil, often with a halo. Note that a thin veil can cover much of the sky while hiding little.',
  },
};

// Étage or form changed. Coverage is kept; a new étage moves the same layer
// smoothly to a typical base there, and a new form loads typical thickness
// and element size.
function syncLayerForm() {
  const { etage, form } = layerParams;
  $$('#etage button').forEach((b) => b.classList.toggle('on', b.dataset.etage === etage));
  $$('#form button').forEach((b) => b.classList.toggle('on', b.dataset.form === form));
  $('#genus').textContent = `Typical: ${GENUS[etage][form]}`;
  $('#lsize-label').textContent = form === 'stratiform' ? 'Size of patches and gaps' : 'Cloud spacing';
  const [smin, smax, sstep] = SIZE_M[form];
  Object.assign(layerCtl.size.input, { min: smin, max: smax, step: sstep });
  if (form === 'cumuliform' && layerParams.octas === 8) layerParams.octas = 7;
  syncLayerChips();
  dirty = true;
}
function applyFormDefaults() {
  const d = LAYER_DEFAULTS[layerParams.etage][layerParams.form];
  layerCtl.thickness.set(d.thickness * FT);
  layerCtl.size.set(d.size);
  syncLayerForm();
}
$$('#etage button').forEach((b) => b.addEventListener('click', () => {
  if (b.dataset.etage === layerParams.etage) return;
  layerParams.etage = b.dataset.etage;
  // Map view: the same layer just moves. Seen on the dome: it also takes the
  // new étage's typical thickness and spacing, and the solver keeps the octas.
  const d = LAYER_DEFAULTS[layerParams.etage][layerParams.form];
  const to = { base: d.base * FT };
  let sizeAtEnd = null;
  if (layerParams.coverMode === 'dome') {
    to.thickness = d.thickness * FT;
    if (layerParams.form === 'cumuliform') to.size = d.size;
    else sizeAtEnd = d.size;
  }
  const from = Object.fromEntries(Object.keys(to).map((k) => [k, layerParams[k]]));
  baseTween = { from, to, sizeAtEnd, start: performance.now(), duration: 1500 };
  syncLayerForm();
}));
$$('#form button').forEach((b) => b.addEventListener('click', () => {
  if (b.dataset.form === layerParams.form) return;
  layerParams.form = b.dataset.form;
  applyFormDefaults();
}));
applyFormDefaults();

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
  const shapes = ['slice', 'cap', 'clumps', 'clumps', 'patchy', 'patchy'];
  quizParams = {
    octas: frac * 8,
    shape: shapes[Math.floor(rnd() * shapes.length)],
    azimuth: Math.floor(rnd() * 360),
    centerElevation: 15 + rnd() * 75,
    axisElevation: 0,
    clumps: 2 + Math.floor(rnd() * 12),
    deform: 0.15 + rnd() * 0.6,
    scale: 2 + rnd() * 4,
    time: 0,
    seed: 1 + Math.floor(rnd() * 1e6),
  };
  quiz.answered = false;
  [...answers.children].forEach((b) => { b.disabled = false; b.className = ''; });
  $('#quiz-result').textContent = '';
  if ($('#quiz-inside').checked) setView('inside', { yaw: rnd() * 2 * Math.PI });
  $('.skymap').classList.toggle('hidden', $('#quiz-hidemap').checked);
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
  $('.skymap').classList.remove('hidden');
  updateReadout();
}
$('#quiz-new').addEventListener('click', newQuiz);

// ---------------------------------------------------------------- tabs, view

$$('.tabs button').forEach((b) => b.addEventListener('click', () => {
  const tab = b.dataset.tab;
  $$('.tabs button').forEach((x) => x.classList.toggle('on', x === b));
  $$('.tab').forEach((s) => { s.hidden = s.id !== `tab-${tab}`; });
  const prev = mode;
  if (tab === 'octas' || tab === 'layer') mode = tab;
  if (tab === 'quiz') {
    mode = 'quiz';
    if (!quizParams) newQuiz();
  }
  if (mode !== 'quiz') $('.skymap').classList.remove('hidden');
  if (mode === 'quiz' && !quiz.answered) $('.skymap').classList.toggle('hidden', $('#quiz-hidemap').checked);
  if (prev !== mode) {
    dirty = true;
    if (view === 'outside' && (prev === 'layer' || mode === 'layer')) resetOutside();
  }
}));

$$('.view-switch [data-view]').forEach((b) => b.addEventListener('click', () => {
  setView(b.dataset.view);
  dirty = true; // sky tint differs inside vs outside
}));
$('#reset-view').addEventListener('click', () => { setView(view); dirty = true; });

$('#skymap-proj').addEventListener('click', () => {
  mapProjection = mapProjection === 'equal-area' ? 'fisheye' : 'equal-area';
  $('#skymap-title').textContent = mapProjection === 'equal-area'
    ? 'All-sky map, equal-area' : 'All-sky map, fisheye (equidistant)';
  $('#skymap-proj').textContent = mapProjection === 'equal-area' ? 'switch to fisheye' : 'switch to equal-area';
  mapLookup = null;
  drawSkyMap();
});

// notes: equal-area band table
$('#band-table').innerHTML = OCTA_BAND_ELEVATIONS.slice(1)
  .map((e, k) => `<tr><td>${k + 1}/8</td><td>${e.toFixed(1)}°</td></tr>`).join('');

// ================================================================ go

buildGuides();
setView('outside');
resize();
requestAnimationFrame(frame);
