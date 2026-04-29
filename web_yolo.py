import sys
import glob
import logging
import subprocess
import cv2
import numpy as np
import json
import threading
import time
import queue
from datetime import datetime
from flask import Flask, Response, render_template_string
from picarx import Picarx

# --- ХАК ДЛЯ БІБЛІОТЕК ---
sys.path.append('/usr/lib/python3/dist-packages')
sys.path.extend(glob.glob('/usr/local/lib/python3.*/dist-packages'))
sys.path.extend(glob.glob('/home/pi/.local/lib/python3.*/site-packages'))

# --- ЛОГУВАННЯ ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
px = Picarx()

# --- ГЛОБАЛЬНІ СТАНИ ---
class RobotState:
    def __init__(self):
        self.current_speed = 0
        self.current_angle = 0
        self.detected_objects = []  # [(class_name, confidence), ...]
        self.hud_status = "NORMAL"
        self.stop_timer = 0
        self.debounce_timer = 0
        self.red_light_active = False
        self.lock = threading.Lock()
    
    def update_hud_status(self, msg):
        with self.lock:
            self.hud_status = msg
            logger.info(f"HUD Status: {msg}")

state = RobotState()

# --- ГЛОБАЛЬНА КАМЕРА ---
class CameraManager:
    def __init__(self):
        self.process = None
        self.frame_queue = queue.Queue(maxsize=2)
        self.lock = threading.Lock()
        self.thread = None
        self.running = False
    
    def start(self):
        """Запуск одного глобального процесу камери"""
        with self.lock:
            if self.process is not None:
                return
            
            logger.info("🎥 Запускаю глобальний процес камери...")
            cmd = [
                'rpicam-vid', '-t', '0',
                '--width', '640', '--height', '480',
                '--codec', 'mjpeg',
                '--nopreview', '--flush', '-o', '-'
            ]
            
            try:
                self.process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    bufsize=0
                )
                self.running = True
                self.thread = threading.Thread(target=self._capture_loop, daemon=False)
                self.thread.start()
                logger.info("✅ Камера запущена")
            except Exception as e:
                logger.error(f"❌ Помилка запуску камери: {e}")
                self.process = None
                self.running = False
    
    def _capture_loop(self):
        """Основний цикл читання фреймів з процесу"""
        data = b''
        try:
            while self.running and self.process:
                chunk = self.process.stdout.read(4096)
                if not chunk:
                    break
                
                data += chunk
                a = data.find(b'\xff\xd8')
                b = data.find(b'\xff\xd9')
                
                if a != -1 and b != -1 and b > a:
                    jpg_data = data[a:b+2]
                    data = data[b+2:]
                    
                    frame = cv2.imdecode(
                        np.frombuffer(jpg_data, np.uint8),
                        cv2.IMREAD_COLOR
                    )
                    
                    if frame is not None:
                        try:
                            self.frame_queue.put_nowait(frame)
                        except queue.Full:
                            try:
                                self.frame_queue.get_nowait()
                                self.frame_queue.put_nowait(frame)
                            except:
                                pass
        except Exception as e:
            logger.error(f"Помилка в _capture_loop: {e}")
        finally:
            self.running = False
    
    def get_frame(self, timeout=0.1):
        """Отримати останній фрейм з черги (не чекати)"""
        try:
            frame = self.frame_queue.get(timeout=timeout)
            return frame
        except queue.Empty:
            return None
    
    def stop(self):
        """Зупинка камери"""
        with self.lock:
            if self.process is None:
                return
            
            logger.info("🛑 Зупиняю камеру...")
            self.running = False
            
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except:
                try:
                    self.process.kill()
                except:
                    pass
            
            self.process = None
            if self.thread:
                self.thread.join(timeout=2)

# --- ОКРЕМИЙ ПОТІК ДЛЯ YOLO ОБРОБКИ ---
class YOLOProcessor:
    def __init__(self, camera, state, model_path='best.onnx'):
        self.camera = camera
        self.state = state
        self.net = cv2.dnn.readNetFromONNX(model_path)
        self.running = False
        self.thread = None
        
        # Кешовані результати для всіх клієнтів
        self.last_frame = None
        self.last_boxes = []
        self.detected_classes = set()
        self.detected_with_conf = []
        self.lock = threading.Lock()
        self.frame_counter = 0
    
    def start(self):
        """Запуск потоку обробки"""
        if self.running:
            return
        
        logger.info("🧠 Запускаю YOLO потік обробки...")
        self.running = True
        self.thread = threading.Thread(target=self._process_loop, daemon=False)
        self.thread.start()
    
    def _process_loop(self):
        """Основний цикл YOLO обробки (у окремому потоці)"""
        try:
            while self.running:
                frame = self.camera.get_frame(timeout=0.5)
                if frame is None:
                    continue
                
                self.frame_counter += 1
                
                # Обробляємо кожен 3-й фрейм (YOLO інференція)
                if self.frame_counter % 3 == 0:
                    detected_classes = set()
                    detected_with_conf = []
                    last_boxes = []
                    
                    # YOLO інференція (повільна операція)
                    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (320, 320), swapRB=True, crop=False)
                    self.net.setInput(blob)
                    preds = np.squeeze(self.net.forward()[0]).T
                    
                    for pred in preds:
                        conf = pred[4:].max()
                        if conf > CONFIDENCE_THRESHOLD:
                            cl_id = np.argmax(pred[4:])
                            class_name = TARGET_CLASSES[cl_id]
                            x, y, w, h = pred[0]*2, pred[1]*1.5, pred[2]*2, pred[3]*1.5
                            
                            detected_classes.add(class_name)
                            detected_with_conf.append((class_name, conf))
                            last_boxes.append(([int(x-w/2), int(y-h/2), int(w), int(h)], class_name, conf))
                    
                    # Зберігаємо результати в кеш для всіх клієнтів
                    with self.lock:
                        self.last_frame = frame.copy()
                        self.last_boxes = last_boxes
                        self.detected_classes = detected_classes
                        self.detected_with_conf = detected_with_conf
                    
                    # Прийняття рішення
                    speed, angle = decide_speed_and_angle(detected_classes)
                    move_robot(speed, angle)
                else:
                    # Для проміжних фреймів просто оновлюємо кеш фрейму
                    with self.lock:
                        self.last_frame = frame.copy()
        
        except Exception as e:
            logger.error(f"Помилка в _process_loop: {e}")
        finally:
            self.running = False
    
    def get_annotated_frame(self):
        """Отримати фрейм з рисованими boxes для клієнта"""
        with self.lock:
            if self.last_frame is None:
                return None
            
            frame = self.last_frame.copy()
            
            # Рисуємо boxes
            for (box, name, conf) in self.last_boxes:
                color = (0, 0, 255) if name == 'Stop' else (0, 165, 255) if name == 'T_Red' else (0, 255, 0)
                cv2.rectangle(frame, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), color, 2)
                label = f"{name} {conf*100:.0f}%"
                cv2.putText(frame, label, (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            return frame
    
    def stop(self):
        """Зупинка потоку обробки"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        logger.info("🛑 YOLO потік зупинений")

camera = CameraManager()
camera.start()

# --- КОНФІГ ---
TARGET_CLASSES = ['Children', 'Crosswalk', 'S40', 'Stop', 'T_Green', 'T_Red', 'T_Yel', 'Traffic']
NORMAL_SPEED = 70
CONFIDENCE_THRESHOLD = 0.35
STOP_DURATION = 10
DEBOUNCE_DURATION = 5
SPEED_REDUCTION = {
    'Children': 30,
    'Crosswalk': 30,
    'S40': 40
}

def init_robot():
    px.set_dir_servo_angle(0)
    logger.info("Робот ініціалізований")
    state.update_hud_status("INIT: OK")

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

# Запуск YOLO потоку обробки
yolo_processor = YOLOProcessor(camera, state)
yolo_processor.start()

def timer_stop():
    """Таймер для Stop знака"""
    for i in range(STOP_DURATION, -1, -1):
        with state.lock:
            state.stop_timer = i
        state.update_hud_status(f"STOP: {i}s")
        logger.info(f"[STOP] {i}s remaining")
        time.sleep(1)
    
    logger.info("[DEBOUNCE] Активна на 5 сек")
    for i in range(DEBOUNCE_DURATION, -1, -1):
        with state.lock:
            state.debounce_timer = i
        state.update_hud_status(f"DEBOUNCE: {i}s")
        time.sleep(1)
    
    with state.lock:
        state.stop_timer = 0
        state.debounce_timer = 0
    state.update_hud_status("NORMAL")

def move_robot(speed, angle):
    """Безпечно управляє рухом робота"""
    with state.lock:
        state.current_speed = speed
        state.current_angle = angle
    
    try:
        px.set_dir_servo_angle(angle)
        if speed == 0:
            px.forward(0)
        elif speed > 0:
            px.forward(speed)
        else:
            px.backward(abs(speed))
    except Exception as e:
        logger.error(f"Помилка руху: {e}")

def decide_speed_and_angle(detected):
    """Логіка вибору швидкості"""
    
    if state.debounce_timer > 0:
        return state.current_speed, state.current_angle
    
    if 'Stop' in detected and state.stop_timer == 0:
        logger.warning("⛔ STOP ЗНАК!")
        move_robot(0, 0)
        threading.Thread(target=timer_stop, daemon=True).start()
        return 0, 0
    
    if state.stop_timer > 0:
        move_robot(0, 0)
        return 0, 0
    
    if 'T_Red' in detected:
        if not state.red_light_active:
            logger.warning("🔴 ЧЕРВОНИЙ СВІТЛОФОР!")
            state.red_light_active = True
        move_robot(0, 0)
        state.update_hud_status("RED_LIGHT")
        return 0, 0
    else:
        if state.red_light_active:
            logger.info("🟢 Світлофор змінився")
            state.red_light_active = False
        state.update_hud_status("NORMAL")
    
    speed = NORMAL_SPEED
    angle = 0
    
    if 'Children' in detected:
        speed = NORMAL_SPEED - SPEED_REDUCTION['Children']
        logger.info("👶 Children!")
        state.update_hud_status("Children: -30")
    elif 'Crosswalk' in detected:
        speed = NORMAL_SPEED - SPEED_REDUCTION['Crosswalk']
        logger.info("🚶 Crosswalk!")
        state.update_hud_status("Crosswalk: -30")
    elif 'S40' in detected:
        speed = NORMAL_SPEED - SPEED_REDUCTION['S40']
        logger.info("🔢 S40!")
        state.update_hud_status("S40: -40")
    else:
        state.update_hud_status("NORMAL")
    
    return speed, angle

# --- ВЕБ-ІНТЕРФЕЙС ---
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
        .critical { color: #f00; font-weight: bold; }
        .warning { color: #ff0; }
        .info { color: #0f0; }
    </style>
</head>
<body>
    <div class="container">
        <img src="/video_feed" id="stream">
        
        <div id="status" class="hud">
            <div class="info">Status: Initializing...</div>
            <div class="info" id="speed">Speed: --</div>
            <div class="info" id="timer">Timer: --</div>
        </div>
        
        <div id="battery" class="hud">
            🔋 Loading...
        </div>
        
        <div id="detected" class="hud">
            <div style="color: #0f0; font-weight: bold;">Detected Objects:</div>
            <div id="objects" style="margin-top: 5px;">--</div>
        </div>
    </div>
    
    <script>
        setInterval(() => {
            fetch('/api/status').then(r => r.json()).then(data => {
                document.getElementById('battery').innerText = data.battery;
                document.getElementById('status').innerHTML = 
                    `<div class="${data.hud_status.includes('STOP') ? 'critical' : data.hud_status.includes('RED') ? 'critical' : data.hud_status.includes('NORMAL') ? 'info' : 'warning'}">Status: ${data.hud_status}</div>
                     <div class="info">Speed: ${data.current_speed}</div>
                     <div class="info">Timer: ${data.stop_timer > 0 ? data.stop_timer + 's' : '--'}`;
                
                const objList = data.detected_objects.length > 0 
                    ? data.detected_objects.map(obj => `${obj[0]} (${(obj[1]*100).toFixed(0)}%)`).join('<br>')
                    : '--';
                document.getElementById('objects').innerHTML = objList;
            }).catch(err => console.error('Status:', err));
        }, 500);
    </script>
</body>
</html>
"""

@app.route('/api/status')
def api_status():
    with state.lock:
        return json.dumps({
            "battery": get_batt(),
            "hud_status": state.hud_status,
            "current_speed": state.current_speed,
            "current_angle": state.current_angle,
            "stop_timer": state.stop_timer,
            "debounce_timer": state.debounce_timer,
            "detected_objects": state.detected_objects
        })

@app.route('/action/<cmd>')
def action(cmd):
    """Для клавіатури"""
    speed = 70
    
    if cmd == 'forward': 
        px.set_dir_servo_angle(0)
        px.forward(speed)
    elif cmd == 'backward': 
        px.set_dir_servo_angle(0)
        px.backward(speed)
    elif cmd == 'stop':
        px.forward(0)
        px.set_dir_servo_angle(0)
    
    return "ok"

def gen_frames():
    """Генератор фреймів (клієнт просто отримує готові фрейми)"""
    try:
        while True:
            # Отримуємо вже оброблений фрейм з YOLO потоку
            frame = yolo_processor.get_annotated_frame()
            
            if frame is None:
                time.sleep(0.05)
                continue
            
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
            
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    except GeneratorExit:
        pass
    except Exception as e:
        logger.error(f"Помилка в gen_frames: {e}")

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def cleanup():
    """Очистка при завершенні"""
    logger.info("🛑 Завершую роботу...")
    px.forward(0)
    px.set_dir_servo_angle(0)
    yolo_processor.stop()
    camera.stop()

if __name__ == "__main__":
    logger.info("🚀 Запускаю PiCar-X v2.0...")
    try:
        app.run(host='0.0.0.0', port=5000, threaded=True, debug=False)
    except KeyboardInterrupt:
        cleanup()
    except Exception as e:
        logger.error(f"Помилка: {e}")
        cleanup()
