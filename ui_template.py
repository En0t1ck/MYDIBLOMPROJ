HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>PiCar-X Smart Control</title>
    <style>
        body { background: #000; margin: 0; display: flex; justify-content: center; align-items: center; height: 100vh; overflow: hidden; font-family: monospace; color: #fff;}
        .container { position: relative; width: 100%; height: 100%; }
        img { width: 100%; height: 100%; object-fit: cover; }
        .hud { position: absolute; padding: 12px; background: rgba(0,0,0,0.75); border-radius: 8px; border: 1px solid #0f0; }
        #battery { top: 15px; right: 15px; font-size: 18px; font-weight: bold; }
        #status { top: 15px; left: 15px; font-size: 16px; color: #0f0; min-width: 250px; }
        #detected { bottom: 15px; left: 15px; font-size: 14px; color: #ff0; max-height: 150px; overflow-y: auto; max-width: 400px; }
        #controls { bottom: 15px; right: 15px; font-size: 14px; color: #0ff; background: rgba(0,0,0,0.9); padding: 10px; border-radius: 5px; border: 1px solid #0ff; max-width: 280px; }
        .critical { color: #f00; font-weight: bold; }
        .warning { color: #ff0; }
        .info { color: #0f0; }
        .turn { color: #00aaff; font-weight: bold; }
    </style>
</head>
<body>
    <div class="container">
        <img src="/video_feed" id="stream">
        <div id="status" class="hud">
            <div class="info">Status: Initializing...</div>
            <div class="info" id="speed">Speed: --</div>
            <div class="info" id="angle">Angle: --°</div>
            <div class="info" id="timer">Timer: --</div>
        </div>
        <div id="battery" class="hud">🔋 Loading...</div>
        <div id="detected" class="hud">
            <div style="color: #0f0; font-weight: bold;">Detected Objects:</div>
            <div id="objects" style="margin-top: 5px;">--</div>
        </div>
        <div id="controls">
            <div style="color: #0ff; font-weight: bold; margin-bottom: 8px;">⌨️ Управління:</div>
            <div>W / ↑ = Вперед</div>
            <div>S / ↓ = Назад</div>
            <div>A / ← = Вліво</div>
            <div>D / → = Вправо</div>
            <div style="color: #f00; font-weight: bold; margin-top: 5px;">SPACE = ГАЛЬМА (Emergency Stop)</div>
        </div>
    </div>
    
    <script>
        const keys = {};
        let lastSpeed = 0;
        let lastAngle = 0;
        
        function processInput() {
            let speed = 0;
            let angle = 0;
            if (keys['w'] || keys['arrowup']) speed = 70;
            if (keys['s'] || keys['arrowdown']) speed = -70;
            if (keys['a'] || keys['arrowleft']) angle = -54;
            if (keys['d'] || keys['arrowright']) angle = 54;
            if (speed !== lastSpeed || angle !== lastAngle) {
                lastSpeed = speed;
                lastAngle = angle;
                fetch(`/control?speed=${speed}&angle=${angle}`).catch(() => {});
            }
        }
        
        window.addEventListener('keydown', (e) => {
            if (e.repeat) return;
            if (e.code === 'Space') {
                for (let k in keys) keys[k] = false;
                processInput();
                return;
            }
            keys[e.key.toLowerCase()] = true;
            processInput();
        });
        
        window.addEventListener('keyup', (e) => {
            if (e.code === 'Space') return;
            keys[e.key.toLowerCase()] = false;
            processInput();
        });
        
        setInterval(() => {
            fetch('/api/status').then(r => r.json()).then(data => {
                document.getElementById('battery').innerText = data.battery;
                
                let statusClass = 'info';
                if (data.hud_status.includes('STOP') || data.hud_status.includes('RED')) statusClass = 'critical';
                else if (data.hud_status.includes('TURN')) statusClass = 'turn';
                else if (!data.hud_status.includes('NORMAL')) statusClass = 'warning';

                document.getElementById('status').innerHTML = 
                    `<div class="${statusClass}">AI Status: ${data.hud_status}</div>
                     <div class="info">Mot Speed: ${data.current_speed}</div>
                     <div class="info">Angle: ${data.current_angle}°</div>
                     <div class="info">Timer: ${data.stop_timer > 0 ? data.stop_timer + 's' : '--'}</div>`;
                
                const objList = data.detected_objects.length > 0 
                    ? data.detected_objects.map(obj => `${obj[0]} (${(obj[1]*100).toFixed(0)}%)`).join('<br>')
                    : '--';
                document.getElementById('objects').innerHTML = objList;
            }).catch(() => {});
        }, 500);
    </script>
</body>
</html>
"""