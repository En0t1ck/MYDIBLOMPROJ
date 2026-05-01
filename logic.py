import threading
import time
import logging
from picarx import Picarx
from config import SPEED_LIMITS, STOP_DURATION, DEBOUNCE_DURATION, TURN_ANGLE, TURN_DURATION
from state import state

logger = logging.getLogger(__name__)
px = Picarx()

def init_robot():
    px.set_dir_servo_angle(0)
    px.forward(0)
    logger.info("Робот ініціалізований та ЗУПИНЕНИЙ.")

def get_batt():
    try:
        from robot_hat import utils
        v = utils.get_battery_voltage()
    except:
        return "N/A"
    percent = max(0, min(100, int((v - 6.6) / (8.4 - 6.6) * 100)))
    return f"{'🟩' if percent > 50 else '🟨' if percent > 20 else '🟥'} {percent}% ({round(v, 2)}V)"

def move_robot(speed, angle):
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

def evaluate_and_move():
    with state.lock:
        intended_speed = state.user_speed
        intended_angle = state.user_angle
        detected = state.detected_classes
        stop_t = state.stop_timer
        red_active = state.red_light_active
        turn_t = state.turn_timer
        turn_dir = state.turn_direction

    final_speed = intended_speed
    final_angle = intended_angle

    if final_speed > 0:
        if stop_t > 0:
            final_speed = 0
            state.update_hud_status(f"STOP: {stop_t}s", logger)
        elif red_active:
            final_speed = 0
            state.update_hud_status("RED_LIGHT", logger)
        else:
            limit = 100
            if 'Children' in detected or 'Crosswalk' in detected:
                limit = min(limit, SPEED_LIMITS['Children'])
            if 'S40' in detected:
                limit = min(limit, SPEED_LIMITS['S40'])
            final_speed = min(final_speed, limit)
            
            # Логіка повороту
            if turn_t > 0:
                final_angle = turn_dir
                state.update_hud_status(f"TURNING...", logger)
            else:
                if limit < 100:
                    state.update_hud_status(f"LIMIT: {limit}", logger)
                else:
                    state.update_hud_status("NORMAL", logger)
    else:
        state.update_hud_status("NORMAL" if final_speed == 0 else "REVERSE", logger)

    move_robot(final_speed, final_angle)

def update_ai_state(detected):
    with state.lock:
        state.detected_classes = detected

        if 'Stop' in detected and state.stop_timer == 0 and state.debounce_timer == 0:
            logger.warning("⛔ STOP ЗНАК!")
            threading.Thread(target=timer_stop, daemon=True).start()

        if 'T_Red' in detected:
            if not state.red_light_active:
                logger.warning("🔴 ЧЕРВОНИЙ СВІТЛОФОР!")
                state.red_light_active = True
        else:
            if state.red_light_active:
                logger.info("🟢 Світлофор змінився")
                state.red_light_active = False
                
        # Запуск таймера повороту
        if state.turn_timer == 0: 
            if 'Turn_Left' in detected:
                logger.info("⬅️ ЗНАК ПОВОРОТУ ЛІВОРУЧ!")
                threading.Thread(target=timer_turn, args=(-TURN_ANGLE,), daemon=True).start()
            elif 'Turn_Right' in detected:
                logger.info("➡️ ЗНАК ПОВОРОТУ ПРАВОРУЧ!")
                threading.Thread(target=timer_turn, args=(TURN_ANGLE,), daemon=True).start()

    evaluate_and_move()

def timer_turn(angle):
    with state.lock:
        state.turn_direction = angle
        state.turn_timer = 1 
    
    time.sleep(TURN_DURATION)
    
    with state.lock:
        state.turn_timer = 0
        state.turn_direction = 0
    evaluate_and_move()

def timer_stop():
    for i in range(STOP_DURATION, -1, -1):
        with state.lock:
            state.stop_timer = i
        evaluate_and_move()
        time.sleep(1)
    
    logger.info("[DEBOUNCE] Активна на 5 сек")
    for i in range(DEBOUNCE_DURATION, -1, -1):
        with state.lock:
            state.debounce_timer = i
        time.sleep(1)
    
    with state.lock:
        state.stop_timer = 0
        state.debounce_timer = 0
    evaluate_and_move()