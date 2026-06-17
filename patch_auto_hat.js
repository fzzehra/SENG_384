const fs = require('fs');
let content = fs.readFileSync('templates/ar_tryon.html', 'utf8');

// 1. Remove the 3D Hat Adjust from the Adjust tab
content = content.replace(
`        <div>
          <div class="section-label">3D Hat Adjust</div>
          <div class="slider-row">
            <div class="slider-label"><span>Scale</span><span id="hScaleVal">1.00x</span></div>
            <input type="range" id="hatScale" min="0.5" max="3.0" step="0.05" value="1.0" oninput="document.getElementById('hScaleVal').textContent=parseFloat(this.value).toFixed(2)+'x'">
          </div>
          <div class="slider-row" style="margin-top:8px;">
            <div class="slider-label"><span>Vertical offset</span><span id="hOffsetVal">0px</span></div>
            <input type="range" id="hatOffset" min="-100" max="100" step="1" value="0" oninput="document.getElementById('hOffsetVal').textContent=this.value+'px'">
          </div>
        </div>
        <div class="divider"></div>`,
``); // Removed

// 2. Modify the render loop to dynamically use lm[10] for hat placement
content = content.replace(
`    const bridgeNose = lm[168];
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
        let scaleModifier = 1.0;
        let yOffsetModifier = 0.0;
        
        if (type === 'hat') {
           scaleModifier = parseFloat(document.getElementById('hatScale').value);
           // Slider goes from -100 to 100. We divide by 50 so it ranges from -2.0 to 2.0
           yOffsetModifier = parseFloat(document.getElementById('hatOffset').value) / 50.0;
        }

        const targetWidth = eyeDistWorld * active.cfg.scale * scaleModifier;
        const finalScale = targetWidth * active.normScale;
        active.model.scale.setScalar(finalScale);
        
        // Apply individual offsets
        active.model.position.set(0, (active.cfg.yOffset + yOffsetModifier) * eyeDistWorld, -active.cfg.zOffset * eyeDistWorld);`,
`    const bridgeNose = lm[168];
    const foreHead = lm[10]; // Top of forehead landmark

    const ndcX = (bridgeNose.x * 2) - 1;
    const ndcY = -(bridgeNose.y * 2) + 1;
    
    const foreNdcY = -(foreHead.y * 2) + 1;

    const depth = Math.abs(pos.z);
    const fovRad = THREE.MathUtils.degToRad(30);
    const halfScreenWorldHeight = Math.tan(fovRad) * depth;
    const halfScreenWorldWidth = halfScreenWorldHeight * camera3d.aspect;

    pos.x = -(ndcX * halfScreenWorldWidth);
    pos.y = (ndcY * halfScreenWorldHeight) - 0.05;
    
    // Calculate world position of forehead
    const forePosY = (foreNdcY * halfScreenWorldHeight) - 0.05;
    const autoHatYOffset = Math.abs(forePosY - pos.y); // Dynamic distance from nose to forehead

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
        
        // Dynamic positioning: If it's a hat, place it exactly at the forehead distance plus a tiny base offset
        let finalY = active.cfg.yOffset * eyeDistWorld;
        if (type === 'hat') {
            finalY = autoHatYOffset + (0.1 * eyeDistWorld); 
        }

        active.model.position.set(0, finalY, -active.cfg.zOffset * eyeDistWorld);`
);

fs.writeFileSync('templates/ar_tryon.html', content);
console.log('Done auto hat patching!');
