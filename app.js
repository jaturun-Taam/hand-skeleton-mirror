// Hand Skeleton Mirror — webcam hand tracking (MediaPipe Hands) driving a real-size
// anatomical hand skeleton (Three.js). Right hand copies the user's right hand.
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// ---------------------------------------------------------------- DOM
const $ = (id) => document.getElementById(id);
const video = $('video'), overlay = $('overlay'), octx = overlay.getContext('2d');
const statusEl = $('status'), msgEl = $('msg'), hudEl = $('hud');
const setStatus = (t, ok = false) => { statusEl.textContent = t; statusEl.className = ok ? 'ok' : ''; };

// ---------------------------------------------------------------- MediaPipe landmark indices
const LM = { WRIST: 0, THUMB: [1, 2, 3, 4], INDEX: [5, 6, 7, 8], MIDDLE: [9, 10, 11, 12], RING: [13, 14, 15, 16], LITTLE: [17, 18, 19, 20] };
const FINGER_LM = { Thumb: LM.THUMB, Index: LM.INDEX, Middle: LM.MIDDLE, Ring: LM.RING, Little: LM.LITTLE };
const CONNECTIONS = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],[9,13],[13,14],[14,15],[15,16],[13,17],[0,17],[17,18],[18,19],[19,20]];

// ---------------------------------------------------------------- Three.js scene
const canvas = $('three');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(35, 1, 1, 5000);
camera.position.set(0, 110, 560);
const controls = new OrbitControls(camera, canvas);
controls.target.set(0, 95, 0);
controls.enableDamping = true;
scene.add(new THREE.HemisphereLight(0xffffff, 0x334155, 1.6));
scene.add(new THREE.AmbientLight(0xffffff, 0.35));
const key = new THREE.DirectionalLight(0xffffff, 1.6); key.position.set(150, 300, 400); scene.add(key);
const rim = new THREE.DirectionalLight(0x9ec5ff, 0.6); rim.position.set(-200, 100, -300); scene.add(rim);
const grid = new THREE.GridHelper(600, 12, 0x2a313b, 0x1d232c); grid.position.y = -120; scene.add(grid);

const boneMat = new THREE.MeshStandardMaterial({ color: 0xe9e2cf, roughness: 0.6, metalness: 0.0, side: THREE.DoubleSide });

// Rig containers
const handRoot = new THREE.Group();    // at wrist; carries palm orientation
scene.add(handRoot);
const forearm = new THREE.Group();     // rigid with the hand
handRoot.add(forearm);
const rig = { joints: {}, meshes: [] };  // joints[finger] = [{obj, rest(THREE.Vector3), q(target)}]
let skel = null;
const V = (a) => new THREE.Vector3(a[0], a[1], a[2]);

async function loadModel() {
  skel = await (await fetch('skeleton.json')).json();
  const gltf = await new GLTFLoader().loadAsync('hand_skeleton.glb');
  const meshes = {};
  gltf.scene.traverse((o) => { if (o.isMesh) { meshes[o.name] = o; o.material = boneMat; o.geometry.deleteAttribute('normal'); o.geometry.computeVertexNormals(); } });
  const wrist = V(skel.wrist);
  const place = (parent, name, pivot) => {   // mesh geometry is in model coords; put it under `parent` whose origin is `pivot`
    const m = meshes[name]; if (!m) { console.warn('missing mesh', name); return; }
    m.position.copy(pivot).negate(); m.quaternion.identity(); m.scale.set(1, 1, 1);
    parent.add(m); rig.meshes.push(m);
  };
  skel.forearm.forEach((n) => place(forearm, n, wrist));
  skel.carpals.forEach((n) => place(handRoot, n, wrist));
  for (const [fname, f] of Object.entries(skel.fingers)) {
    let parent = handRoot, parentPivot = wrist; rig.joints[fname] = [];
    f.joints.forEach((j) => {
      const pivot = V(j.pivot);
      const obj = new THREE.Group(); obj.name = fname + '_' + j.name;
      obj.position.copy(pivot).sub(parentPivot);
      parent.add(obj); place(obj, j.bone, pivot);
      rig.joints[fname].push({ obj, name: j.name, rest: V(j.axis).normalize(), q: new THREE.Quaternion(), angle: 0 });
      parent = obj; parentPivot = pivot;
    });
  }
  setRestPose();
}

// Rest pose: fingers up (+Y world), palm toward the viewer (+Z)
const qRest = new THREE.Quaternion().setFromRotationMatrix(new THREE.Matrix4().makeBasis(
  new THREE.Vector3(1, 0, 0), new THREE.Vector3(0, 0, -1), new THREE.Vector3(0, 1, 0)));
let qHandTarget = qRest.clone();
function setRestPose() {
  qHandTarget.copy(qRest);
  for (const chain of Object.values(rig.joints)) chain.forEach((j) => { j.q.identity(); j.angle = 0; });
  for (const k of Object.keys(angles)) delete angles[k];
}

// ---------------------------------------------------------------- landmarks -> pose
const tmp = { a: new THREE.Vector3(), b: new THREE.Vector3(), c: new THREE.Vector3(), q: new THREE.Quaternion(), m: new THREE.Matrix4() };
let ema = null;           // smoothed world landmarks (THREE.Vector3[21])
const angles = {};        // HUD

// MediaPipe world coords (x right, y down, z away) -> Three (x right, y up, z toward viewer)
function toThree(p, mirrorX) { return new THREE.Vector3(mirrorX ? -p.x : p.x, -p.y, -p.z); }

function applyLandmarks(world, isRightHand) {
  const alpha = 1 - parseFloat($('rngSmooth').value) * 0.6; // landmark EMA
  const pts = world.map((p) => toThree(p, !isRightHand));
  if (!ema) ema = pts; else ema = ema.map((e, i) => e.lerp(pts[i], alpha));
  const P = ema;
  // palm frame: f = fingers, n = palm-out normal, s = thumb side
  const f = tmp.a.subVectors(P[9], P[0]).normalize().clone();
  const n = new THREE.Vector3().crossVectors(tmp.b.subVectors(P[5], P[0]), tmp.c.subVectors(P[17], P[0])).normalize();
  const s = new THREE.Vector3().crossVectors(f, n).normalize();
  n.crossVectors(s, f).normalize();
  // model frame: X = thumb side, Y = -palmOut (right hand), Z = fingers
  tmp.m.makeBasis(s, n.clone().negate(), f);
  qHandTarget.setFromRotationMatrix(tmp.m);
  const qInvHand = qHandTarget.clone().invert();

  for (const [fname, chain] of Object.entries(rig.joints)) {
    const idx = FINGER_LM[fname];
    // observed bone directions (model frame): [MC?, PP, MP, DP] — finger metacarpals stay rigid with the palm
    const dirs = [];
    if (fname === 'Thumb') {
      dirs.push(dirBetween(P[idx[0]], P[idx[1]]), dirBetween(P[idx[1]], P[idx[2]]), dirBetween(P[idx[2]], P[idx[3]]));
    } else {
      dirs.push(null, dirBetween(P[idx[0]], P[idx[1]]), dirBetween(P[idx[1]], P[idx[2]]), dirBetween(P[idx[2]], P[idx[3]]));
    }
    const acc = new THREE.Quaternion();
    chain.forEach((j, k) => {
      const d = dirs[k];
      if (!d) { j.q.identity(); j.angle = 0; return; }
      const dm = d.applyQuaternion(qInvHand);                 // into model (palm) frame
      const local = dm.applyQuaternion(acc.clone().invert()); // into parent-rotated frame
      j.q.setFromUnitVectors(j.rest, local.normalize());
      j.angle = THREE.MathUtils.radToDeg(j.rest.angleTo(local));
      acc.multiply(j.q);
    });
    angles[fname] = chain.map((j) => j.angle);
  }
}
function feed(world, isRight) { lastSeen = performance.now(); demo = false; applyLandmarks(world, isRight); }
function dirBetween(a, b) { return new THREE.Vector3().subVectors(b, a).normalize(); }

// ---------------------------------------------------------------- render loop (with smoothing)
const clock = new THREE.Clock();
let demo = false, lastSeen = 0;
function animate() {
  requestAnimationFrame(animate);
  const t = clock.getElapsedTime();
  const r = 1 - parseFloat($('rngSmooth').value);  // responsiveness
  if (demo) demoPose(t);
  else if (performance.now() - lastSeen > 1500 && ema) { ema = null; setRestPose(); }
  handRoot.quaternion.slerp(qHandTarget, r);
  for (const chain of Object.values(rig.joints)) chain.forEach((j) => j.obj.quaternion.slerp(j.q, r));
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (canvas.width !== Math.floor(w * renderer.getPixelRatio()) || canvas.height !== Math.floor(h * renderer.getPixelRatio())) {
    renderer.setSize(w, h, false); camera.aspect = w / h; camera.updateProjectionMatrix();
  }
  controls.update();
  renderer.render(scene, camera);
  drawHUD();
}

function demoPose(t) {
  const palmOut = new THREE.Vector3(0, -1, 0);
  const fingers = ['Thumb', 'Index', 'Middle', 'Ring', 'Little'];
  fingers.forEach((fname, i) => {
    const chain = rig.joints[fname]; if (!chain) return;
    const phase = i * 0.5;
    const curl = 0.5 - 0.5 * Math.cos(t * 1.4 - phase);            // 0..1 wave
    chain.forEach((j, k) => {
      if (fname !== 'Thumb' && k === 0) return;                    // rigid metacarpals
      const maxDeg = fname === 'Thumb' ? [35, 45, 60][k] : [0, 85, 100, 70][k];
      const axis = new THREE.Vector3().crossVectors(j.rest, palmOut).normalize();
      j.q.setFromAxisAngle(axis, THREE.MathUtils.degToRad(maxDeg * curl));
      j.angle = maxDeg * curl;
    });
    angles[fname] = chain.map((j) => j.angle);
  });
  const yaw = 0.35 * Math.sin(t * 0.5), pitch = 0.2 * Math.sin(t * 0.33);
  qHandTarget.copy(qRest).premultiply(new THREE.Quaternion().setFromEuler(new THREE.Euler(pitch, yaw, 0)));
}

function drawHUD() {
  const names = { Thumb: ['CMC', 'MCP', 'IP'], Index: ['MCP', 'PIP', 'DIP'], Middle: ['MCP', 'PIP', 'DIP'], Ring: ['MCP', 'PIP', 'DIP'], Little: ['MCP', 'PIP', 'DIP'] };
  let html = '<table><tr><th></th><th>J1</th><th>J2</th><th>J3</th></tr>';
  for (const f of ['Thumb', 'Index', 'Middle', 'Ring', 'Little']) {
    const a = angles[f] || [];
    const vals = f === 'Thumb' ? a : a.slice(1);
    html += `<tr><td class="f">${f}</td>` + [0, 1, 2].map((i) => `<td>${vals[i] != null ? vals[i].toFixed(0) + '°' : '–'}</td>`).join('') + '</tr>';
  }
  html += '</table><div style="color:#8b95a5;margin-top:4px">Thumb: CMC/MCP/IP · others: MCP/PIP/DIP</div>';
  if (hudEl.innerHTML !== html) hudEl.innerHTML = html;
}

// ---------------------------------------------------------------- webcam + MediaPipe Hands
let hands = null, running = false;
async function startCamera() {
  if (running) return;
  demo = false;
  setStatus('กำลังเปิดกล้อง…');
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' }, audio: false });
    video.srcObject = stream;
    await video.play();
  } catch (e) {
    setStatus('เปิดกล้องไม่ได้: ' + e.message); msgEl.textContent = 'เปิดกล้องไม่ได้ (' + e.name + ') — ต้องเปิดหน้านี้ผ่าน http://localhost หรือ https และอนุญาตกล้อง'; return;
  }
  overlay.width = video.videoWidth || 1280; overlay.height = video.videoHeight || 720;
  msgEl.textContent = '';
  if (!hands) {
    hands = new Hands({ locateFile: (f) => `${f}` });
    hands.setOptions({ maxNumHands: 2, modelComplexity: 1, minDetectionConfidence: 0.6, minTrackingConfidence: 0.5, selfieMode: false });
    hands.onResults(onResults);
  }
  running = true;
  setStatus('กำลังโหลดโมเดล MediaPipe…');
  const loop = async () => {
    if (!running) return;
    if (video.readyState >= 2) { try { await hands.send({ image: video }); } catch (e) { console.error(e); } }
    requestAnimationFrame(loop);
  };
  loop();
}

function onResults(res) {
  // draw mirrored video + landmarks
  const W = overlay.width, H = overlay.height;
  octx.save(); octx.translate(W, 0); octx.scale(-1, 1);
  octx.drawImage(res.image, 0, 0, W, H);
  let pick = -1, isRight = true;
  if (res.multiHandLandmarks && res.multiHandLandmarks.length) {
    // input frame is NOT mirrored, so MediaPipe's "Left" label is the user's real right hand
    res.multiHandedness.forEach((h, i) => { if (h.label === 'Left' && pick < 0) pick = i; });
    if (pick < 0) { pick = 0; isRight = false; }
    if ($('chkLandmarks').checked) {
      res.multiHandLandmarks.forEach((lm, i) => drawLandmarks(lm, i === pick ? '#f2c94c' : '#5b6675', W, H));
    }
  }
  octx.restore();
  if (pick >= 0) {
    feed(res.multiHandWorldLandmarks[pick], isRight);
    setStatus(isRight ? 'ติดตามมือขวา' : 'ติดตามมือซ้าย (แปลงเป็นมือขวา)', true);
  } else setStatus('ไม่พบมือ — ยกมือขวาให้เห็นทั้งฝ่ามือ');
}

function drawLandmarks(lm, color, W, H) {
  octx.lineWidth = 3; octx.strokeStyle = color; octx.fillStyle = color;
  octx.beginPath();
  for (const [a, b] of CONNECTIONS) { octx.moveTo(lm[a].x * W, lm[a].y * H); octx.lineTo(lm[b].x * W, lm[b].y * H); }
  octx.stroke();
  for (const p of lm) { octx.beginPath(); octx.arc(p.x * W, p.y * H, 5, 0, Math.PI * 2); octx.fill(); }
}

// ---------------------------------------------------------------- UI
$('btnCam').addEventListener('click', startCamera);
$('btnDemo').addEventListener('click', () => { demo = !demo; if (!demo) setRestPose(); setStatus(demo ? 'โหมดสาธิต (ไม่ใช้กล้อง)' : 'พร้อม', demo); $('btnDemo').textContent = demo ? 'หยุดสาธิต' : 'โหมดสาธิต'; });
$('chkForearm').addEventListener('change', (e) => { forearm.visible = e.target.checked; });
$('chkMirrorPalm').addEventListener('change', (e) => { rig.meshes.forEach((m) => { m.scale.y = e.target.checked ? -1 : 1; }); });
window.addEventListener('keydown', (e) => { if (e.key === 'd') $('btnDemo').click(); if (e.key === 'c') startCamera(); });

// ---------------------------------------------------------------- boot
loadModel().then(() => { setStatus('พร้อม — กดเปิดกล้อง หรือ โหมดสาธิต'); animate(); window.__ready = true; })
  .catch((e) => { console.error(e); setStatus('โหลดโมเดลไม่สำเร็จ: ' + e.message); });
window.__debug = { rig, applyLandmarks: feed, angles, get ema() { return ema; } };
