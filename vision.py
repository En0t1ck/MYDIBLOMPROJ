import cv2
import numpy as np
import threading
import queue
import subprocess
import logging
from config import TARGET_CLASSES, CONFIDENCE_THRESHOLD
from logic import update_ai_state
from state import state

logger = logging.getLogger(__name__)

class CameraManager:
    def __init__(self):
        self.process = None
        self.frame_queue = queue.Queue(maxsize=2)
        self.lock = threading.Lock()
        self.thread = None
        self.running = False
    
    def start(self):
        with self.lock:
            if self.process is not None: return
            logger.info("🎥 Запускаю камеру...")
            cmd = ['rpicam-vid', '-t', '0', '--width', '640', '--height', '480', '--codec', 'mjpeg', '--nopreview', '--flush', '-o', '-']
            try:
                self.process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
                self.running = True
                self.thread = threading.Thread(target=self._capture_loop, daemon=False)
                self.thread.start()
            except Exception as e:
                logger.error(f"❌ Помилка запуску камери: {e}")
    
    def _capture_loop(self):
        data = b''
        try:
            while self.running and self.process:
                chunk = self.process.stdout.read(4096)
                if not chunk: break
                data += chunk
                a = data.find(b'\xff\xd8')
                b = data.find(b'\xff\xd9')
                if a != -1 and b != -1 and b > a:
                    jpg_data = data[a:b+2]
                    data = data[b+2:]
                    frame = cv2.imdecode(np.frombuffer(jpg_data, np.uint8), cv2.IMREAD_COLOR)
                    if frame is not None:
                        try: self.frame_queue.put_nowait(frame)
                        except queue.Full:
                            try: self.frame_queue.get_nowait(); self.frame_queue.put_nowait(frame)
                            except: pass
        except Exception: pass
        finally: self.running = False
    
    def get_frame(self, timeout=0.1):
        try: return self.frame_queue.get(timeout=timeout)
        except queue.Empty: return None
    
    def stop(self):
        with self.lock:
            self.running = False
            if self.process: self.process.terminate()
            self.process = None

class YOLOProcessor:
    def __init__(self, camera, model_path='best.onnx'):
        self.camera = camera
        self.net = cv2.dnn.readNetFromONNX(model_path)
        self.running = False
        self.thread = None
        self.last_frame = None
        self.last_boxes = []
        self.lock = threading.Lock()
        self.frame_counter = 0
    
    def start(self):
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._process_loop, daemon=False)
            self.thread.start()
    
    def _process_loop(self):
        try:
            while self.running:
                frame = self.camera.get_frame(timeout=0.5)
                if frame is None: continue
                self.frame_counter += 1
                
                if self.frame_counter % 3 == 0:
                    detected_classes = set()
                    detected_with_conf = []
                    last_boxes = []
                    
                    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (320, 320), swapRB=True, crop=False)
                    self.net.setInput(blob)
                    preds = np.squeeze(self.net.forward()[0]).T
                    
                    valid_preds = []
                    for pred in preds:
                        conf = pred[4:].max()
                        if conf > CONFIDENCE_THRESHOLD:
                            cl_id = np.argmax(pred[4:])
                            x, y, w, h = pred[0]*2, pred[1]*1.5, pred[2]*2, pred[3]*1.5
                            valid_preds.append({'x1': int(x - w/2), 'y1': int(y - h/2), 'x2': int(x + w/2), 'y2': int(y + h/2), 'conf': conf, 'class_name': TARGET_CLASSES[cl_id]})
                    
                    if valid_preds:
                        valid_preds = sorted(valid_preds, key=lambda x: x['conf'], reverse=True)
                        keep = []
                        for i, pred1 in enumerate(valid_preds):
                            skip = False
                            for pred2 in keep:
                                x1_inter = max(pred1['x1'], pred2['x1'])
                                y1_inter = max(pred1['y1'], pred2['y1'])
                                x2_inter = min(pred1['x2'], pred2['x2'])
                                y2_inter = min(pred1['y2'], pred2['y2'])
                                if x2_inter > x1_inter and y2_inter > y1_inter:
                                    inter = (x2_inter - x1_inter) * (y2_inter - y1_inter)
                                    box1_area = (pred1['x2'] - pred1['x1']) * (pred1['y2'] - pred1['y1'])
                                    box2_area = (pred2['x2'] - pred2['x1']) * (pred2['y2'] - pred2['y1'])
                                    if inter / (box1_area + box2_area - inter) > 0.5:
                                        skip = True; break
                            if not skip: keep.append(pred1)
                        
                        for pred in keep:
                            detected_classes.add(pred['class_name'])
                            detected_with_conf.append((pred['class_name'], float(pred['conf'])))
                            last_boxes.append(([pred['x1'], pred['y1'], pred['x2']-pred['x1'], pred['y2']-pred['y1']], pred['class_name'], pred['conf']))
                    
                    with self.lock:
                        self.last_frame = frame.copy()
                        self.last_boxes = last_boxes
                    
                    update_ai_state(detected_classes)
                    with state.lock:
                        state.detected_objects = detected_with_conf
                else:
                    with self.lock:
                        self.last_frame = frame.copy()
        except Exception: pass
        finally: self.running = False
    
    def get_annotated_frame(self):
        with self.lock:
            if self.last_frame is None: return None
            frame = self.last_frame.copy()
            for (box, name, conf) in self.last_boxes:
                color = (0, 0, 255) if name == 'Stop' else (0, 165, 255) if name == 'T_Red' else (255, 100, 0) if name in ['Turn_Left', 'Turn_Right'] else (0, 255, 0)
                cv2.rectangle(frame, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), color, 2)
                cv2.putText(frame, f"{name} {conf*100:.0f}%", (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            return frame
    
    def stop(self):
        self.running = False