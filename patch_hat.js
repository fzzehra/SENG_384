const fs = require('fs');
let content = fs.readFileSync('templates/ar_tryon.html', 'utf8');

// 1. Add head occluder to scene
content = content.replace(
`const headGroup = new THREE.Group();
scene3d.add(headGroup);`,
`const headGroup = new THREE.Group();
scene3d.add(headGroup);

// Head Occluder (Invisible sphere that hides the back of the hat)
const occluderGeometry = new THREE.SphereGeometry(1, 32, 32);
const occluderMaterial = new THREE.MeshBasicMaterial({ color: 0x00ff00, colorWrite: false });
const headOccluder = new THREE.Mesh(occluderGeometry, occluderMaterial);
// We will scale it in render loop based on head size
headGroup.add(headOccluder);`
);

// 2. Fix hat config rotations and positions
content = content.replace(
`const hatConfig = {
  'bucket_hat.glb': { zOffset: 1.5, scale: 2.2, yOffset: 0.2, rotX: 0.4 },
  'fbx.glb': { zOffset: 1.5, scale: 2.2, yOffset: 0.2, rotX: 0.4 },
  'menard_hat_lowpoly.glb': { zOffset: 1.5, scale: 2.2, yOffset: 0.2, rotX: 0.4 },
  'obj_bucket_hat_gen8_toshitimo.glb': { zOffset: 1.5, scale: 2.2, yOffset: 0.2, rotX: 0.4 },
  'victorian_abigail_hat.glb': { zOffset: 1.5, scale: 2.2, yOffset: 0.2, rotX: 0.4 }
};`,
`const hatConfig = {
  'bucket_hat.glb': { zOffset: 0.3, scale: 2.2, yOffset: 0.4, rotX: 0.1 },
  'fbx.glb': { zOffset: 0.3, scale: 2.2, yOffset: 0.4, rotX: 0.1 },
  'menard_hat_lowpoly.glb': { zOffset: 0.4, scale: 2.4, yOffset: 0.4, rotX: 0.1 },
  'obj_bucket_hat_gen8_toshitimo.glb': { zOffset: 0.3, scale: 2.2, yOffset: 0.4, rotX: 0.1 },
  'victorian_abigail_hat.glb': { zOffset: 0.3, scale: 2.2, yOffset: 0.4, rotX: 0.1 }
};`
);

// 3. Update render loop for occluder and hat opacity
content = content.replace(
`    ['glasses', 'hat'].forEach(type => {
      const active = activeModels[type];
      if (active.ready && active.model) {
        const targetWidth = eyeDistWorld * active.cfg.scale;
        const finalScale = targetWidth * active.normScale;
        active.model.scale.setScalar(finalScale);
        
        // Dynamic positioning: If it's a hat, place it exactly at the forehead distance plus a tiny base offset
        let finalY = active.cfg.yOffset * eyeDistWorld;
        if (type === 'hat') {
            finalY = autoHatYOffset + (active.cfg.yOffset * eyeDistWorld); 
        }

        active.model.position.set(0, finalY, -active.cfg.zOffset * eyeDistWorld);

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
    });`,
`    // Scale and position occluder
    const occluderRadius = eyeDistWorld * 1.0;
    headOccluder.scale.set(occluderRadius, occluderRadius * 1.3, occluderRadius * 1.1);
    headOccluder.position.set(0, eyeDistWorld * 0.2, -eyeDistWorld * 0.5);

    ['glasses', 'hat'].forEach(type => {
      const active = activeModels[type];
      if (active.ready && active.model) {
        const targetWidth = eyeDistWorld * active.cfg.scale;
        const finalScale = targetWidth * active.normScale;
        active.model.scale.setScalar(finalScale);
        
        // Dynamic positioning: If it's a hat, place it exactly at the forehead distance plus a tiny base offset
        let finalY = active.cfg.yOffset * eyeDistWorld;
        if (type === 'hat') {
            finalY = autoHatYOffset + (active.cfg.yOffset * eyeDistWorld); 
        }

        active.model.position.set(0, finalY, -active.cfg.zOffset * eyeDistWorld);

        active.model.traverse(child => {
          if (!child.isMesh) return;
          if (!child.material._originalOpacity) {
            child.material = child.material.clone();
            // child.material.transparent = true;
            child.material._originalOpacity = child.material.opacity || 1.0;
          }
          // Don't fade out hats on rotation, only fade glasses
          if (type === 'glasses') {
            child.material.transparent = true;
            child.material.opacity = occlusionOpacity * child.material._originalOpacity;
          } else {
            child.material.opacity = child.material._originalOpacity;
            // Fix sorting issues for hat overlapping with occluder
            child.renderOrder = 10;
          }
        });
      }
    });`
);

fs.writeFileSync('templates/ar_tryon.html', content);
console.log('Hat patch applied.');
