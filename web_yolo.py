import sys
import glob
import logging
import subprocess
import cv2
import numpy as np
import json
from flask import Flask, Response, render_template_string
from picarx import Picarx

# --- ХАК ДЛЯ БІБЛІОТЕК ---
sys.path.append('/usr/lib/python3/dist-packages')
sys.path.extend(glob.glob('/usr/local/lib/python3.*/dist-packages'))
sys.path.extend(glob.glob('/home/pi/.local/lib/python3.*/site-packages'))

# --- ЛОГУВАННЯ ---
logging.basicConfig(filename='robot_log.txt', level=logging.INFO, format='%(asctime)s | %(message)s', datefmt='%H:%M:%S')

app = Flask(__name__)
px = Picarx()

def init_robot():
    # Ставимо тільки колеса рівно
    px.set_dir_servo_angle(0)
    # КАМЕРУ НЕ ЧІПАЄМО! Моторчики залишаються розслабленими для ручного налаштування.
    logging.info("Робот ініціалізований. Камера в повністю ручному режимі.")

init_robot()

def get_batt():
    try:
        from robot_hat import utils
        v = utils.get_battery_voltage()
    except:
        try:
            from robot_hat import get_battery_voltage
            v = get_battery_voltage()
        except:
            return "N/A"
            
    percent = int((v - 6.6) / (8.4 - 6.6) * 100)
    percent = max(0, min(100, percent))
    icon = "🟩" if percent > 50 else "🟨" if percent > 20 else "🟥"
    return f"{icon} {percent}% ({round(v, 2)}V)"

# --- ШІ МОДЕЛЬ ---
net = cv2.dnn.readNetFromONNX('best.onnx')
TARGET_CLASSES = ['Children', 'Crosswalk', 'S40', 'Stop', 'T_Green', 'T_Red', 'T_Yel', 'Traffic']

# --- ВЕБ-ІНТЕРФЕЙС ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>PiCar-X Control</title>
    <style>
        body { background: #000; margin: 0; display: flex; justify-content: center; align-items: center; height: 100vh; overflow: hidden; font-family: monospace;}
        img { max-height: 100%; max-width: 100%; }
        .hud { position: absolute; padding: 10px; color: #fff; background: rgba(0,0,0,0.6); border-radius: 8px; }
        #battery { top: 15px; right: 15px; font-size: 20px; font-weight: bold;}
        #debug { top: 15px; left: 15px; font-size: 16px; color: #0f0; }
    </style>
</head>
<body>
    <div id="debug" class="hud">Статус: Стоп</div>
    <div id="battery" class="hud">🔋 Читаю батарею...</div>
    <img src="/video_feed">
    
    <script>
        // Трекінг затиснутих клавіш (ТІЛЬКИ ЇЗДА)
        const keys = { KeyW: false, KeyS: false, KeyA: false, KeyD: false };
        let lastCmd = 'stop';

        function send(cmd) { fetch('/action/' + cmd); }

        setInterval(() => {
            fetch('/status').then(r => r.json()).then(data => {
                document.getElementById('battery').innerText = data.battery;
            });
        }, 5000);

        function evaluateMovement() {
            let cmd = 'stop';
            if (keys.KeyW && keys.KeyA) cmd = 'forward_left';
            else if (keys.KeyW && keys.KeyD) cmd = 'forward_right';
            else if (keys.KeyS && keys.KeyA) cmd = 'backward_left';
            else if (keys.KeyS && keys.KeyD) cmd = 'backward_right';
            else if (keys.KeyW) cmd = 'forward';
            else if (keys.KeyS) cmd = 'backward';
            else if (keys.KeyA) cmd = 'left';
            else if (keys.KeyD) cmd = 'right';

            if (cmd !== lastCmd) {
                send(cmd);
                lastCmd = cmd;
                document.getElementById('debug').innerText = "Статус: " + cmd;
            }
        }

        document.addEventListener('keydown', (e) => {
            if (e.repeat) return;
            const k = e.code; 
            
            if (keys.hasOwnProperty(k)) {
                keys[k] = true;
                evaluateMovement();
            }
            
            // Екстрена зупинка
            if (k === 'Space') {
                keys.KeyW = false; keys.KeyS = false; keys.KeyA = false; keys.KeyD = false;
                evaluateMovement();
            }
        });

        document.addEventListener('keyup', (e) => {
            const k = e.code;
            if (keys.hasOwnProperty(k)) {
                keys[k] = false;
                evaluateMovement();
            }
        });
    </script>
</body>
</html>
"""

@app.route('/status')
def status():
    return json.dumps({"battery": get_batt()})

@app.route('/action/<cmd>')
def action(cmd):
    speed = 70
    angle = 54 
    
    if cmd == 'forward': 
        px.set_dir_servo_angle(0)
        px.forward(speed)
    elif cmd == 'backward': 
        px.set_dir_servo_angle(0)
        px.backward(speed)
    elif cmd == 'left': 
        px.set_dir_servo_angle(-angle)
    elif cmd == 'right': 
        px.set_dir_servo_angle(angle)
    elif cmd == 'forward_left': 
        px.set_dir_servo_angle(-angle)
        px.forward(speed)
    elif cmd == 'forward_right': 
        px.set_dir_servo_angle(angle)
        px.forward(speed)
    elif cmd == 'backward_left': 
        px.set_dir_servo_angle(-angle)
        px.backward(speed)
    elif cmd == 'backward_right': 
        px.set_dir_servo_angle(angle)
        px.backward(speed)
    elif cmd == 'stop':
        px.forward(0)
        px.set_dir_servo_angle(0)
    
    logging.info(f"Рух: {cmd}")
    return "ok"

def gen_frames():
    cmd = ['rpicam-vid', '-t', '0', '--width', '640', '--height', '480', '--codec', 'mjpeg', '--nopreview', '--flush', '-o', '-']
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=0)
    data = b''
    frame_counter = 0
    last_boxes = []

    try:
        while True:
            chunk = process.stdout.read(4096)
            if not chunk: break
            data += chunk
            a, b = data.find(b'\xff\xd8'), data.find(b'\xff\xd9')
            if a != -1 and b != -1:
                jpg_data = data[a:b+2]; data = data[b+2:]
                frame = cv2.imdecode(np.frombuffer(jpg_data, np.uint8), cv2.IMREAD_COLOR)
                if frame is not None:
                    frame_counter += 1
                    if frame_counter % 3 == 0:
                        last_boxes = []
                        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (320, 320), swapRB=True, crop=False)
                        net.setInput(blob); preds = np.squeeze(net.forward()[0]).T
                        for pred in preds:
                            if pred[4:].max() > 0.45:
                                cl_id = np.argmax(pred[4:])
                                x, y, w, h = pred[0]*2, pred[1]*1.5, pred[2]*2, pred[3]*1.5
                                last_boxes.append(([int(x-w/2), int(y-h/2), int(w), int(h)], TARGET_CLASSES[cl_id]))
                    for (box, name) in last_boxes:
                        cv2.rectangle(frame, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), (0, 255, 0), 2)
                        cv2.putText(frame, name, (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    ret, buffer = cv2.imencode('.jpg', frame)
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    finally:
        process.terminate()

@app.route('/')
def index(): return render_template_string(HTML_TEMPLATE)

@app.route('/video_feed')
def video_feed(): return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, threaded=True)
