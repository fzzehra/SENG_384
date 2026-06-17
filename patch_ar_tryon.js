const fs = require('fs');
let content = fs.readFileSync('templates/ar_tryon.html', 'utf8');

// 1. HTML UI for Hats
content = content.replace(
`        <div>
          <div class="section-label">Hat</div>
          <div class="acc-grid">
            <button class="acc-btn full active" onclick="selectAcc('hat',null,this)">No Hat</button>
            <button class="acc-btn" onclick="selectAcc('hat','hat1.png',this)">Hat 1</button>
            <button class="acc-btn" onclick="selectAcc('hat','hat2.png',this)">Hat 2</button>
            <button class="acc-btn" onclick="selectAcc('hat','hat3.png',this)">Hat 3</button>
            <button class="acc-btn" onclick="selectAcc('hat','hat4.png',this)">Hat 4</button>
          </div>
        </div>`,
`        <div>
          <div class="section-label">3D Hat</div>
          <div class="acc-grid" id="hatsGrid">
            <button class="acc-btn full active" id="btn-no-hat" onclick="clearModel('hat', this)">No Hat</button>
          </div>
        </div>`
);

// Fix UI for glasses
content = content.replace(
`onclick="clearModel(this)"`,
`onclick="clearModel('glasses', this)"`
);

// 2. Configuration for Hats in JS
content = content.replace(
`const glassesConfig = {
  'glasses-1-.glb': { zOffset: 0.5, scale: 2.0 },
  'glasses-7.glb':  { zOffset: 1.4, scale: 1.85 },
  'glasses-10.glb': { zOffset: 2.0, scale: 1.9 },
  'glasses-12.glb': { zOffset: 2.5, scale: 2.0 }
};`,
`const glassesConfig = {
  'glasses-1-.glb': { zOffset: 0.5, scale: 2.0, yOffset: -0.1 },
  'glasses-7.glb':  { zOffset: 1.4, scale: 1.85, yOffset: -0.1 },
  'glasses-10.glb': { zOffset: 2.0, scale: 1.9, yOffset: -0.1 },
  'glasses-12.glb': { zOffset: 2.5, scale: 2.0, yOffset: -0.1 }
};

const HAT_CATALOG = [
  { name: 'Bucket Hat', file: 'bucket_hat.glb' },
  { name: 'FBX Hat', file: 'fbx.glb' },
  { name: 'Lowpoly Hat', file: 'menard_hat_lowpoly.glb' },
  { name: 'Gen8 Hat', file: 'obj_bucket_hat_gen8_toshitimo.glb' },
  { name: 'Victorian Hat', file: 'victorian_abigail_hat.glb' }
];

const hatConfig = {
  'bucket_hat.glb': { zOffset: 0.0, scale: 2.5, yOffset: 1.2 },
  'fbx.glb': { zOffset: 0.0, scale: 2.5, yOffset: 1.2 },
  'menard_hat_lowpoly.glb': { zOffset: 0.0, scale: 2.5, yOffset: 1.2 },
  'obj_bucket_hat_gen8_toshitimo.glb': { zOffset: 0.0, scale: 2.5, yOffset: 1.2 },
  'victorian_abigail_hat.glb': { zOffset: 0.0, scale: 2.5, yOffset: 1.2 }
};`
);

// 3. Three.js State Variables
content = content.replace(
`const modelGroup = new THREE.Group();
scene3d.add(modelGroup);

const gltfLoader = new THREE.GLTFLoader();
let currentModel = null;
let modelNormScale = 1;
let modelReady = false;`,
`const headGroup = new THREE.Group();
scene3d.add(headGroup);

const gltfLoader = new THREE.GLTFLoader();

let activeModels = {
  glasses: { group: new THREE.Group(), model: null, normScale: 1, cfg: null, ready: false },
  hat: { group: new THREE.Group(), model: null, normScale: 1, cfg: null, ready: false }
};
headGroup.add(activeModels.glasses.group);
headGroup.add(activeModels.hat.group);`
);

// 4. Building Grid for Hats and Glasses
content = content.replace(
`function buildGlassesGrid() {
  const grid = document.getElementById('glassesGrid');
  const noGlassesBtn = document.getElementById('btn-no-glasses');
  grid.innerHTML = '';
  grid.appendChild(noGlassesBtn);

  GLASSES_CATALOG.forEach((item, idx) => {
    const btn = document.createElement('button');
    btn.className = 'acc-btn';
    const canvasId = \`prevCanvas_\${idx}\`;
    
    btn.innerHTML = \`<canvas id="\${canvasId}" width="80" height="40"></canvas><span>\${item.name}</span>\`;
    btn.addEventListener('click', () => loadModel(item.file, btn));
    grid.appendChild(btn);

    setTimeout(() => {
      const cEl = document.getElementById(canvasId);
      if (cEl) generatePreview(item.file, cEl);
    }, 100);
  });
}`,
`function buildModelGrid(catalog, type, gridId, noBtnId, path) {
  const grid = document.getElementById(gridId);
  const noBtn = document.getElementById(noBtnId);
  grid.innerHTML = '';
  grid.appendChild(noBtn);

  catalog.forEach((item, idx) => {
    const btn = document.createElement('button');
    btn.className = 'acc-btn';
    const canvasId = \`prevCanvas_\${type}_\${idx}\`;
    
    btn.innerHTML = \`<canvas id="\${canvasId}" width="80" height="40"></canvas><span>\${item.name}</span>\`;
    btn.addEventListener('click', () => loadModel(item.file, type, btn, path));
    grid.appendChild(btn);

    setTimeout(() => {
      const cEl = document.getElementById(canvasId);
      if (cEl) generatePreview(item.file, cEl, path);
    }, 100);
  });
}

function buildGlassesGrid() {
  buildModelGrid(GLASSES_CATALOG, 'glasses', 'glassesGrid', 'btn-no-glasses', '/static/accessories/glasses/');
  buildModelGrid(HAT_CATALOG, 'hat', 'hatsGrid', 'btn-no-hat', '/static/accessories/hats_3d/');
}`
);

// 5. Modify generatePreview
content = content.replace(
`function generatePreview(file, canvasEl) {
  const pRenderer = new THREE.WebGLRenderer({ canvas: canvasEl, antialias: true, alpha: true });
  pRenderer.setSize(80, 40);
  pRenderer.setClearColor(0x000000, 0);
  
  const pScene = new THREE.Scene();
  pScene.add(new THREE.AmbientLight(0xffffff, 2.5));
  const pLight = new THREE.DirectionalLight(0xffffff, 1.5);
  pLight.position.set(0, 1, 2);
  pScene.add(pLight);

  const pCamera = new THREE.PerspectiveCamera(35, 2, 0.1, 100);

  gltfLoader.load('/static/accessories/glasses/' + file, gltf => {`,
`function generatePreview(file, canvasEl, path) {
  const pRenderer = new THREE.WebGLRenderer({ canvas: canvasEl, antialias: true, alpha: true });
  pRenderer.setSize(80, 40);
  pRenderer.setClearColor(0x000000, 0);
  
  const pScene = new THREE.Scene();
  pScene.add(new THREE.AmbientLight(0xffffff, 2.5));
  const pLight = new THREE.DirectionalLight(0xffffff, 1.5);
  pLight.position.set(0, 1, 2);
  pScene.add(pLight);

  const pCamera = new THREE.PerspectiveCamera(35, 2, 0.1, 100);

  gltfLoader.load(path + file, gltf => {`
);

// 6. Load Model Logic
content = content.replace(
`window.loadModel = function(file, btn) {
  if (btn) {
    document.querySelectorAll('#glassesGrid .acc-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
  }
  window.clearModel(null, false);
  statusBar.textContent = 'Loading 3D Model: ' + file;

  gltfLoader.load('/static/accessories/glasses/' + file, gltf => {
    currentModel = gltf.scene;
    const box = new THREE.Box3().setFromObject(currentModel);
    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3());
    
    currentModel.position.sub(center).add(new THREE.Vector3(0, -0.05, 0));
    const cfg = glassesConfig[file] || { zOffset: 0.5, scale: 1.8 };
    currentModel.position.z -= cfg.zOffset;
    currentModel._scaleMultiplier = cfg.scale;
    
    modelGroup.add(currentModel);
    modelNormScale = 1.0 / size.x;
    modelReady = true;
    statusBar.textContent = 'Face detected ✓';
  }, undefined, err => {
    console.error(err);
    statusBar.textContent = 'Failed to load model!';
  });
};

window.clearModel = function(btn, updateUI = true) {
  if (updateUI && btn) {
    document.querySelectorAll('#glassesGrid .acc-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
  }
  while (modelGroup.children.length > 0) modelGroup.remove(modelGroup.children[0]);
  currentModel = null;
  modelReady = false;
};`,
`window.loadModel = function(file, type, btn, path) {
  if (btn) {
    const gridSelector = type === 'glasses' ? '#glassesGrid' : '#hatsGrid';
    document.querySelectorAll(\`\${gridSelector} .acc-btn\`).forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
  }
  window.clearModel(type, null, false);
  statusBar.textContent = \`Loading 3D \${type}: \` + file;

  gltfLoader.load(path + file, gltf => {
    const m = gltf.scene;
    const box = new THREE.Box3().setFromObject(m);
    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3());
    
    const cfg = type === 'glasses' ? (glassesConfig[file] || { zOffset: 0.5, scale: 1.8, yOffset: -0.1 }) 
                                   : (hatConfig[file] || { zOffset: 0.0, scale: 2.5, yOffset: 1.2 });
    
    m.position.sub(center); // Center the model
    
    activeModels[type].model = m;
    activeModels[type].cfg = cfg;
    activeModels[type].normScale = 1.0 / size.x;
    
    activeModels[type].group.add(m);
    activeModels[type].ready = true;
    statusBar.textContent = 'Face detected ✓';
  }, undefined, err => {
    console.error(err);
    statusBar.textContent = \`Failed to load \${type}!\`;
  });
};

window.clearModel = function(type, btn, updateUI = true) {
  if (updateUI && btn) {
    const gridSelector = type === 'glasses' ? '#glassesGrid' : '#hatsGrid';
    document.querySelectorAll(\`\${gridSelector} .acc-btn\`).forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
  }
  
  const mGroup = activeModels[type].group;
  while (mGroup.children.length > 0) mGroup.remove(mGroup.children[0]);
  activeModels[type].model = null;
  activeModels[type].ready = false;
};`
);

// 8. Update renderLoop 3D processing (part 1)
content = content.replace(
`  if (!result.faceLandmarks?.length) {
    statusBar.textContent = 'No face detected';
    if (currentModel) currentModel.visible = false;
    renderer3d.render(scene3d, camera3d);
    return;
  }`,
`  if (!result.faceLandmarks?.length) {
    statusBar.textContent = 'No face detected';
    headGroup.visible = false;
    renderer3d.render(scene3d, camera3d);
    return;
  }`
);

// 9. Update renderLoop 3D processing (part 2)
content = content.replace(
`  // ─── 1. 3D GLASSES POSITION & CALCULATION ───
  if (result.facialTransformationMatrixes?.length && currentModel && modelReady) {
    currentModel.visible = true;
    const mp = result.facialTransformationMatrixes[0].data;
    const m = new THREE.Matrix4().set(
      mp[0], mp[4], mp[8],  mp[12],
      mp[1], mp[5], mp[9],  mp[13],
      mp[2], mp[6], mp[10], mp[14],
      mp[3], mp[7], mp[11], mp[15]
    );

    const pos = new THREE.Vector3();
    const quat = new THREE.Quaternion();
    const scaleV = new THREE.Vector3();
    m.decompose(pos, quat, scaleV);

    const bridgeNose = lm[168];
    const ndcX = (bridgeNose.x * 2) - 1;
    const ndcY = -(bridgeNose.y * 2) + 1;

    const depth = Math.abs(pos.z);
    const fovRad = THREE.MathUtils.degToRad(30);
    const halfScreenWorldHeight = Math.tan(fovRad) * depth;
    const halfScreenWorldWidth = halfScreenWorldHeight * camera3d.aspect;

    pos.x = -(ndcX * halfScreenWorldWidth);
    pos.y = (ndcY * halfScreenWorldHeight) - 0.05;

    const lEyePx = lm[33].x * W;
    const rEyePx = lm[263].x * W;
    const eyeDistPx = Math.abs(rEyePx - lEyePx);
    const eyeDistWorld = (eyeDistPx / W) * halfScreenWorldWidth * 2;

    pos.z = -depth - (eyeDistWorld * 0.5);
    const qAynali = new THREE.Quaternion(quat.x, -quat.y, -quat.z, quat.w);

    const targetWidth = eyeDistWorld * (currentModel._scaleMultiplier || 1.8);
    const finalScale = targetWidth * modelNormScale;

    modelGroup.position.set(pos.x, pos.y - 0.1, pos.z);
    modelGroup.quaternion.copy(qAynali);
    modelGroup.scale.setScalar(finalScale);

    debugEl.textContent = \`pos:(\${pos.x.toFixed(1)},\${pos.y.toFixed(1)},\${pos.z.toFixed(1)}) scale:\${finalScale.toFixed(3)}\`;

    const euler = new THREE.Euler().setFromQuaternion(qAynali, 'YXZ');
    const yawAbs = Math.abs(euler.y);
    const occlusionOpacity = THREE.MathUtils.clamp(THREE.MathUtils.mapLinear(yawAbs, 0.3, 0.7, 1.0, 0.0), 0.0, 1.0);
    
    currentModel.traverse(child => {
      if (!child.isMesh) return;
      if (!child.material._originalOpacity) {
        child.material = child.material.clone();
        child.material.transparent = true;
        child.material._originalOpacity = child.material.opacity || 1.0;
      }
      child.material.opacity = occlusionOpacity * child.material._originalOpacity;
    });
  }`,
`  // ─── 1. 3D MODELS POSITION & CALCULATION ───
  if (result.facialTransformationMatrixes?.length) {
    headGroup.visible = true;
    const mp = result.facialTransformationMatrixes[0].data;
    const m = new THREE.Matrix4().set(
      mp[0], mp[4], mp[8],  mp[12],
      mp[1], mp[5], mp[9],  mp[13],
      mp[2], mp[6], mp[10], mp[14],
      mp[3], mp[7], mp[11], mp[15]
    );

    const pos = new THREE.Vector3();
    const quat = new THREE.Quaternion();
    const scaleV = new THREE.Vector3();
    m.decompose(pos, quat, scaleV);

    const bridgeNose = lm[168];
    const ndcX = (bridgeNose.x * 2) - 1;
    const ndcY = -(bridgeNose.y * 2) + 1;

    const depth = Math.abs(pos.z);
    const fovRad = THREE.MathUtils.degToRad(30);
    const halfScreenWorldHeight = Math.tan(fovRad) * depth;
    const halfScreenWorldWidth = halfScreenWorldHeight * camera3d.aspect;

    pos.x = -(ndcX * halfScreenWorldWidth);
    pos.y = (ndcY * halfScreenWorldHeight) - 0.05;

    const lEyePx = lm[33].x * W;
    const rEyePx = lm[263].x * W;
    const eyeDistPx = Math.abs(rEyePx - lEyePx);
    const eyeDistWorld = (eyeDistPx / W) * halfScreenWorldWidth * 2;

    pos.z = -depth - (eyeDistWorld * 0.5);
    const qAynali = new THREE.Quaternion(quat.x, -quat.y, -quat.z, quat.w);

    headGroup.position.set(pos.x, pos.y, pos.z);
    headGroup.quaternion.copy(qAynali);

    const euler = new THREE.Euler().setFromQuaternion(qAynali, 'YXZ');
    const yawAbs = Math.abs(euler.y);
    const occlusionOpacity = THREE.MathUtils.clamp(THREE.MathUtils.mapLinear(yawAbs, 0.3, 0.7, 1.0, 0.0), 0.0, 1.0);

    ['glasses', 'hat'].forEach(type => {
      const active = activeModels[type];
      if (active.ready && active.model) {
        const targetWidth = eyeDistWorld * active.cfg.scale;
        const finalScale = targetWidth * active.normScale;
        active.model.scale.setScalar(finalScale);
        
        // Apply individual offsets
        active.model.position.set(0, active.cfg.yOffset * eyeDistWorld, -active.cfg.zOffset * eyeDistWorld);

        active.model.traverse(child => {
          if (!child.isMesh) return;
          if (!child.material._originalOpacity) {
            child.material = child.material.clone();
            child.material.transparent = true;
            child.material._originalOpacity = child.material.opacity || 1.0;
          }
          child.material.opacity = occlusionOpacity * child.material._originalOpacity;
        });
      }
    });

    debugEl.textContent = \`pos:(\${pos.x.toFixed(1)},\${pos.y.toFixed(1)},\${pos.z.toFixed(1)})\`;
  } else {
    headGroup.visible = false;
  }`
);

// Disable PNG hat drawing to avoid overlap
content = content.replace(
`  if (selected.hat && pngImages[selected.hat]) drawHat(pt);`,
`  // if (selected.hat && pngImages[selected.hat]) drawHat(pt); // Disabled PNG hats since we have 3D now`
);

fs.writeFileSync('templates/ar_tryon.html', content);
console.log('Done!');
