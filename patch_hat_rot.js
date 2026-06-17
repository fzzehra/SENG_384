const fs = require('fs');
let content = fs.readFileSync('templates/ar_tryon.html', 'utf8');

// Update hatConfig to include rotation and adjusted offsets
content = content.replace(
`const hatConfig = {
  'bucket_hat.glb': { zOffset: 0.0, scale: 2.5, yOffset: 0.6 },
  'fbx.glb': { zOffset: 0.0, scale: 2.5, yOffset: 0.6 },
  'menard_hat_lowpoly.glb': { zOffset: 0.0, scale: 2.5, yOffset: 0.6 },
  'obj_bucket_hat_gen8_toshitimo.glb': { zOffset: 0.0, scale: 2.5, yOffset: 0.6 },
  'victorian_abigail_hat.glb': { zOffset: 0.0, scale: 2.5, yOffset: 0.6 }
};`,
`const hatConfig = {
  'bucket_hat.glb': { zOffset: -0.2, scale: 2.5, yOffset: 0.3, rotX: -Math.PI / 2 },
  'fbx.glb': { zOffset: -0.2, scale: 2.5, yOffset: 0.3, rotX: -Math.PI / 2 },
  'menard_hat_lowpoly.glb': { zOffset: -0.2, scale: 2.5, yOffset: 0.3, rotX: -Math.PI / 2 },
  'obj_bucket_hat_gen8_toshitimo.glb': { zOffset: -0.2, scale: 2.5, yOffset: 0.3, rotX: -Math.PI / 2 },
  'victorian_abigail_hat.glb': { zOffset: -0.2, scale: 2.5, yOffset: 0.3, rotX: -Math.PI / 2 }
};`
);

// Apply rotation when loading model
content = content.replace(
`    const cfg = type === 'glasses' ? (glassesConfig[file] || { zOffset: 0.5, scale: 1.8, yOffset: -0.1 }) 
                                   : (hatConfig[file] || { zOffset: 0.0, scale: 2.5, yOffset: 1.2 });
    
    m.position.sub(center); // Center the model`,
`    const cfg = type === 'glasses' ? (glassesConfig[file] || { zOffset: 0.5, scale: 1.8, yOffset: -0.1, rotX: 0 }) 
                                   : (hatConfig[file] || { zOffset: 0.0, scale: 2.5, yOffset: 0.3, rotX: -Math.PI / 2 });
    
    m.position.sub(center); // Center the model
    
    if (cfg.rotX) {
        // We wrap the model in a pivot so we can rotate the mesh itself 
        // to fix its orientation permanently before adding to the group
        m.rotation.x = cfg.rotX;
    }`
);

// Update renderLoop to lower the hat so it "wears" on the head
// We will subtract a bit from autoHatYOffset to bring it down.
content = content.replace(
`        let finalY = active.cfg.yOffset * eyeDistWorld;
        if (type === 'hat') {
            finalY = autoHatYOffset + (0.1 * eyeDistWorld); 
        }`,
`        let finalY = active.cfg.yOffset * eyeDistWorld;
        if (type === 'hat') {
            // We want the hat to sit LOWER than the top of the forehead so it looks worn.
            // So instead of adding 0.1, we subtract a fraction, or we use the cfg.yOffset to fine-tune it relative to the forehead.
            finalY = autoHatYOffset + (active.cfg.yOffset * eyeDistWorld) - (0.6 * eyeDistWorld); 
        }`
);

fs.writeFileSync('templates/ar_tryon.html', content);
console.log('Done hat rot patch!');
