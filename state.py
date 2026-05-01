import threading

class RobotState:
    def __init__(self):
        self.user_speed = 0
        self.user_angle = 0
        self.current_speed = 0
        self.current_angle = 0
        
        self.detected_objects = []  
        self.detected_classes = set()
        
        self.hud_status = "NORMAL"
        self.stop_timer = 0
        self.debounce_timer = 0
        self.red_light_active = False
        
        # Для поворотів
        self.turn_timer = 0      
        self.turn_direction = 0  
        
        self.lock = threading.Lock()
    
    def update_hud_status(self, msg, logger):
        if self.hud_status != msg:
            self.hud_status = msg
            logger.info(f"HUD Status: {msg}")

# Єдиний екземпляр стану для всього проєкту
state = RobotState()