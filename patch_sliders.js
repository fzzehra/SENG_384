const fs = require('fs');
let content = fs.readFileSync('templates/ar_tryon.html', 'utf8');

// Change 'PNG Hat' to '3D Hat'
content = content.replace(
  '<div class="section-label">PNG Hat</div>',
  '<div class="section-label">3D Hat Adjust</div>'
);

// Reset slider defaults for 3D Hat
content = content.replace(
  '<input type="range" id="hatScale" min="0.8" max="2.5" step="0.05" value="1.35"',
  '<input type="range" id="hatScale" min="0.5" max="3.0" step="0.05" value="1.0"'
);
content = content.replace(
  '<span id="hScaleVal">1.35x</span>',
  '<span id="hScaleVal">1.00x</span>'
);

content = content.replace(
  '<input type="range" id="hatOffset" min="-80" max="80" step="1" value="0"',
  '<input type="range" id="hatOffset" min="-100" max="100" step="1" value="0"'
);

// Update renderLoop calculation
content = content.replace(
`        const targetWidth = eyeDistWorld * active.cfg.scale;
        const finalScale = targetWidth * active.normScale;
        active.model.scale.setScalar(finalScale);
        
        // Apply individual offsets
        active.model.position.set(0, active.cfg.yOffset * eyeDistWorld, -active.cfg.zOffset * eyeDistWorld);`,
`        let scaleModifier = 1.0;
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
        active.model.position.set(0, (active.cfg.yOffset + yOffsetModifier) * eyeDistWorld, -active.cfg.zOffset * eyeDistWorld);`
);

// Update default hatConfig yOffset to be lower since 1.2 was too high.
content = content.replace(
  /yOffset: 1\.2/g,
  'yOffset: 0.6'
);

fs.writeFileSync('templates/ar_tryon.html', content);
console.log('Done patching sliders!');
