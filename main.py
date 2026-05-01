import sys
import glob
import logging

# --- ХАК ДЛЯ БІБЛІОТЕК (обов'язково зверху) ---
sys.path.append('/usr/lib/python3/dist-packages')
sys.path.extend(glob.glob('/usr/local/lib/python3.*/dist-packages'))
sys.path.extend(glob.glob('/home/pi/.local/lib/python3.*/site-packages'))

import time
import json
import cv2
from flask import Flask, Response, render_template_string, request

# Імпорти наших модулів
from ui_template import HTML_TEMPLATE
from state import state
from logic import init_robot, evaluate_and_move, get_batt, px
from vision import CameraManager, YOLOProcessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Запуск процесів
camera = CameraManager()
camera.start()
yolo_processor = YOLOProcessor(camera)
yolo_processor.start()
init_robot()

@app.route('/control')
def control():
    try:
        speed = int(request.args.get('speed', 0))
        angle = int(request.args.get('angle', 0))
        with state.lock:
            state.user_speed = speed
            state.user_angle = angle
        evaluate_and_move()
        return "ok"
    except Exception as e:
        logger.error(f"Помилка контролю: {e}")
        return "error"

@app.route('/api/status')
def api_status():
    with state.lock:
        return json.dumps({
            "battery": get_batt(),
            "hud_status": state.hud_status,
            "current_speed": state.current_speed,
            "current_angle": state.current_angle,
            "stop_timer": state.stop_timer,
            "detected_objects": state.detected_objects
        })

def gen_frames():
    try:
        while True:
            frame = yolo_processor.get_annotated_frame()
            if frame is None:
                time.sleep(0.05)
                continue
            ret, buffer = cv2.imencode('.jpg', frame)
            if ret: yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    except Exception: pass

@app.route('/')
def index(): 
    return render_template_string(HTML_TEMPLATE)

@app.route('/video_feed')
def video_feed(): 
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def cleanup():
    logger.info("🛑 Завершую роботу...")
    px.forward(0)
    px.set_dir_servo_angle(0)
    yolo_processor.stop()
    camera.stop()

if __name__ == "__main__":
    logger.info("🚀 Запускаю PiCar-X Auto Assistant (Оновлено з поворотами)...")
    try:
        app.run(host='0.0.0.0', port=5000, threaded=True, debug=False)
    except KeyboardInterrupt: 
        cleanup()
    except Exception as e: 
        logger.error(f"Критична помилка: {e}")
        cleanup()