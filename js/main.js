import * as THREE from 'three';
import { OrbitControls } from '../vendor/OrbitControls.js';
import {
  W, H, N, grid, computeField, solveThreshold, coverFromField, bandFractions,
  reportedOctas, makeNoise3, mulberry32, OCTA_BAND_ELEVATIONS,
} from './sky.js';
import { castLayer, nearbyClouds, sheetGeometry, distanceToHeight } from './layer.js';

const DEG = Math.PI / 180;
const FT = 0.3048; // metres per foot
const $ = (s) => document.querySelector(s);
const $$ = (s) => [...document.querySelectorAll(s)];

// ================================================================ state

const shapeParams = {
  octas: 3, shape: 'slice', azimuth: 0, centerElevation: 90, clumps: 6,
  deform: 0, scale: 3, time: 0, seed: 1, animate: false,
};
const layerParams = {
  etage: 'low', form: 'cumuliform',
  cover: 0.3, base: 4000 * FT, thickness: 3000 * FT, size: 900, extent: 400,
  curved: true, seed: 1, showClouds: true,
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
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0f1622);
const camera = new THREE.PerspectiveCamera(45, 1, 0.005, 100);

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
      varying vec2 vUv;
      void main() {
        vUv = uv;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }`,
    fragmentShader: `
      uniform sampler2D field;
      uniform float threshold, skyAlpha, cloudAlpha;
      uniform vec3 skyColor, cloudColor;
      varying vec2 vUv;
      void main() {
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
// Half great circles from one end of a horizontal axis (azimuth a0, degrees)
// to the other, tilted every 22.5 deg: the lying orange's slice boundaries.
// Points are [east, north, up].
function lyingArcs(a0) {
  const a = a0 * DEG;
  const ax = [Math.sin(a), Math.cos(a), 0];
  const px = [Math.cos(a), -Math.sin(a), 0];
  const arcs = [];
  for (let k = 1; k < 8; k++) {
    const tilt = k * 22.5 * DEG;
    const pts = [];
    for (let i = 0; i <= 96; i++) {
      const t = (i / 96) * Math.PI;
      const c = Math.cos(t), sn = Math.sin(t);
      pts.push([
        ax[0] * c + px[0] * Math.cos(tilt) * sn,
        ax[1] * c + px[1] * Math.cos(tilt) * sn,
        Math.sin(tilt) * sn,
      ]);
    }
    arcs.push(pts);
  }
  return arcs;
}
function guideSpec(g) {
  if (g === 'quadrants') return { circles: [30, 60], meridians: [0, 90, 180, 270], arcs: [] };
  if (g === 'slices') return { circles: [], meridians: [0, 45, 90, 135, 180, 225, 270, 315], arcs: [] };
  if (g === 'lying') return { circles: [], meridians: [], arcs: lyingArcs(shapeParams.azimuth) };
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
// In cloud-layer mode the scene is to scale: one dome radius = SCENE_M metres
// (about 46,000 ft), so the three étages sit at their true relative heights.

const SCENE_M = 14000;
const K = 1 / SCENE_M; // scene units per metre

// Étage limits in feet, as used for reporting.
const ETAGES = {
  low: { name: 'Low', base: [300, 6500], band: [0, 6500] },
  mid: { name: 'Mid', base: [6600, 19000], band: [6500, 20000] },
  high: { name: 'High', base: [20000, 40000], band: [20000, 45000] },
};

const layerDecor = new THREE.Group();
scene.add(layerDecor);
const etageBand = new THREE.Mesh(
  new THREE.BufferGeometry(),
  new THREE.MeshBasicMaterial({ color: 0xffd27a, transparent: true, opacity: 0.12, side: THREE.DoubleSide, depthWrite: false }),
);
layerDecor.add(etageBand);
const etageLabels = {};

function etageSprite(text) {
  const c = document.createElement('canvas');
  c.width = 512;
  c.height = 128;
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: t, depthWrite: false, transparent: true }));
  s.scale.set(0.44, 0.11, 1);
  s.userData.draw = (active) => {
    const g = c.getContext('2d');
    g.clearRect(0, 0, 512, 128);
    g.font = `${active ? 700 : 500} 52px system-ui, sans-serif`;
    g.textBaseline = 'middle';
    g.fillStyle = active ? '#ffd27a' : '#9fb0c3';
    g.fillText(text, 8, 64);
    t.needsUpdate = true;
  };
  return s;
}

{
  // boundaries between étages: a faint disc and a ring where the plane meets the dome
  for (const ft of [6500, 20000]) {
    const y = ft * FT * K;
    const r = Math.sqrt(1 - y * y);
    const disc = new THREE.Mesh(
      new THREE.CircleGeometry(r, 96),
      new THREE.MeshBasicMaterial({ color: 0xaec6e0, transparent: true, opacity: 0.05, side: THREE.DoubleSide, depthWrite: false }),
    );
    disc.rotation.x = -Math.PI / 2;
    disc.position.y = y;
    layerDecor.add(disc, circleLine(Math.asin(y), 1.003, 0xaec6e0, 0.6));
  }
  for (const [key, e] of Object.entries(ETAGES)) {
    const label = etageSprite(`${e.name} étage`);
    const y = ((e.band[0] + Math.min(e.band[1], 40000)) / 2) * FT * K;
    const r = Math.sqrt(1 - y * y) + 0.3;
    label.position.copy(dirFromAzEl(50 * DEG, 0, r)).setY(y);
    label.material.depthTest = false;
    label.renderOrder = 5;
    etageLabels[key] = label;
    layerDecor.add(label);
  }
}

function updateEtageDecor() {
  const [lo, hi] = ETAGES[layerParams.etage].band;
  const y0 = lo * FT * K, y1 = Math.min(0.999, hi * FT * K);
  const pts = [];
  for (let i = 0; i <= 24; i++) {
    const y = y0 + ((y1 - y0) * i) / 24;
    pts.push(new THREE.Vector2(Math.sqrt(1 - y * y) * 1.006, y));
  }
  etageBand.geometry.dispose();
  etageBand.geometry = new THREE.LatheGeometry(pts, 128);
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
cloudMesh.renderOrder = 1;
scene.add(cloudMesh);

// Stratiform sheet, drawn as its base and top surfaces with the same holes.
const sheetCanvas = document.createElement('canvas');
sheetCanvas.width = sheetCanvas.height = 512;
const sheetTexture = new THREE.CanvasTexture(sheetCanvas);
sheetTexture.colorSpace = THREE.SRGBColorSpace;
const sheetMeshes = [0xd9dee5, 0xf6f7f9].map((color) => {
  const m = new THREE.Mesh(
    new THREE.CircleGeometry(1, 128),
    new THREE.MeshBasicMaterial({ color, map: sheetTexture, transparent: true, side: THREE.DoubleSide, depthWrite: false }),
  );
  m.rotation.x = -Math.PI / 2;
  m.renderOrder = 1;
  m.visible = false;
  scene.add(m);
  return m;
});

function drawSheet() {
  const { base, thickness } = layerParams;
  const yb = base * K, yt = (base + thickness) * K;
  const rr = Math.sqrt(Math.max(0, 1 - yb * yb)) * 0.995;
  const { field, T } = sheetGeometry(layerParams);
  const g = sheetCanvas.getContext('2d');
  const img = g.createImageData(512, 512);
  const edge = 0.04;
  for (let py = 0; py < 512; py++) {
    const north = (1 - ((py + 0.5) / 512) * 2) * rr / K;
    for (let px = 0; px < 512; px++) {
      const east = (((px + 0.5) / 512) * 2 - 1) * rr / K;
      let a = (field(east, north) - T) / edge + 0.5;
      a = a < 0 ? 0 : a > 1 ? 1 : a;
      const o = (py * 512 + px) * 4;
      img.data[o] = img.data[o + 1] = img.data[o + 2] = 255;
      img.data[o + 3] = 235 * a;
    }
  }
  g.putImageData(img, 0, 0);
  sheetTexture.needsUpdate = true;
  // both surfaces share one texture, so they use the base's radius; the top
  // is only drawn while that disc still fits inside the dome
  sheetMeshes.forEach((m, i) => {
    const y = i === 0 ? yb : yt;
    m.scale.set(rr, rr, 1);
    m.position.y = y;
    m.visible = y < 1 && rr * rr + y * y < 1.0001 && !(i === 1 && thickness * K < 0.002);
  });
}

function updateCloudMeshes() {
  const on = mode === 'layer';
  layerDecor.visible = on;
  observer.scale.setScalar(on ? 0.4 : 1);
  cloudMesh.count = 0;
  sheetMeshes.forEach((m) => { m.visible = false; });
  if (!on) return;
  updateEtageDecor();
  if (!layerParams.showClouds) return;
  if (layerParams.form === 'stratiform') {
    drawSheet();
    return;
  }
  const list = nearbyClouds(layerParams, K, MAX_CLOUDS);
  const m = new THREE.Matrix4();
  list.forEach((c, i) => {
    const h = c.top - c.base;
    m.makeScale(c.r, h, c.r);
    m.setPosition(c.x, c.base + h / 2, -c.y);
    cloudMesh.setMatrixAt(i, m);
  });
  cloudMesh.count = list.length;
  cloudMesh.instanceMatrix.needsUpdate = true;
}

// ================================================================ cameras

const orbit = new OrbitControls(camera, canvas);
orbit.enableDamping = true;
orbit.minDistance = 1.2;
orbit.maxDistance = 7;
orbit.maxPolarAngle = Math.PI / 2 + 0.35;

const look = { yaw: 0, pitch: 25 * DEG, fov: 80, drag: null };

function setView(v, { yaw } = {}) {
  view = v;
  $$('.view-switch [data-view]').forEach((b) => b.classList.toggle('on', b.dataset.view === v));
  observer.visible = v === 'outside';
  for (const label of Object.values(etageLabels)) label.visible = v === 'outside';
  orbit.enabled = v === 'outside';
  if (v === 'outside') {
    camera.fov = 45;
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
  camera.position.set(2.1, 1.5, 2.4);
  orbit.target.set(0, 0.25, 0);
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

function recompute() {
  if (mode === 'layer') {
    layerGeo = castLayer(layerParams, cover);
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

let last = performance.now();
function frame(now) {
  const dt = Math.min(0.1, (now - last) / 1000);
  last = now;
  if (mode === 'octas' && shapeParams.animate) {
    shapeParams.time += dt * 0.6;
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
  lying: 'The orange lies on its side. Each slice runs from the north horizon over the sky to the south horizon, a 180° arc. There are four slices on each side of the zenith, each one octa, filling from the east horizon. Direction turns the orange.',
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
const SIZE_M = { cumuliform: [50, 4000, 10], stratiform: [300, 20000, 100] };
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
  cover: bindRange('lcover', layerParams, 'cover', (v) => `${(v * 8).toFixed(1)}/8`),
  base: bindHeight('lbase', 'base'),
  thickness: bindHeight('lthick', 'thickness'),
  size: bindRange('lsize', layerParams, 'size', fmtKm),
  extent: bindRange('lext', layerParams, 'extent', (v) => `${v} km`),
};
$$('#units button').forEach((b) => b.addEventListener('click', () => {
  heightUnit = b.dataset.units;
  $$('#units button').forEach((x) => x.classList.toggle('on', x === b));
  layerCtl.base.sync();
  layerCtl.thickness.sync();
}));
$('#lcurve').addEventListener('change', (e) => { layerParams.curved = e.target.checked; dirty = true; });
$('#lshow').addEventListener('change', (e) => { layerParams.showClouds = e.target.checked; dirty = true; });
$('#lreseed').addEventListener('click', () => { layerParams.seed = 1 + Math.floor(Math.random() * 1e6); dirty = true; });

// Typical starting layer for each étage and form (heights in ft, size in m).
const LAYER_DEFAULTS = {
  low: {
    cumuliform: { cover: 0.3, base: 4000, thickness: 3000, size: 900 },
    stratiform: { cover: 0.6, base: 1500, thickness: 800, size: 1500 },
  },
  mid: {
    cumuliform: { cover: 0.4, base: 12000, thickness: 600, size: 600 },
    stratiform: { cover: 0.8, base: 10000, thickness: 4000, size: 8000 },
  },
  high: {
    cumuliform: { cover: 0.3, base: 25000, thickness: 200, size: 300 },
    stratiform: { cover: 0.7, base: 28000, thickness: 1500, size: 10000 },
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

function applyLayerDefaults() {
  const { etage, form } = layerParams;
  const d = LAYER_DEFAULTS[etage][form];
  $$('#etage button').forEach((b) => b.classList.toggle('on', b.dataset.etage === etage));
  $$('#form button').forEach((b) => b.classList.toggle('on', b.dataset.form === form));
  $('#genus').textContent = `Typical: ${GENUS[etage][form]}`;
  $('#lsize-label').textContent = form === 'stratiform' ? 'Size of patches and gaps' : 'Cloud diameter';
  const [smin, smax, sstep] = SIZE_M[form];
  Object.assign(layerCtl.size.input, { min: smin, max: smax, step: sstep });
  layerCtl.cover.input.max = MAX_COVER[form];
  layerCtl.cover.set(d.cover);
  layerCtl.base.set(d.base * FT);
  layerCtl.thickness.set(d.thickness * FT);
  layerCtl.size.set(d.size);
  dirty = true;
}
$$('#etage button').forEach((b) => b.addEventListener('click', () => {
  layerParams.etage = b.dataset.etage;
  applyLayerDefaults();
}));
$$('#form button').forEach((b) => b.addEventListener('click', () => {
  layerParams.form = b.dataset.form;
  applyLayerDefaults();
}));
applyLayerDefaults();

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
  if (prev !== mode) dirty = true;
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
