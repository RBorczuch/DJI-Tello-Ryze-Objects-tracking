# tracking_data.py
import threading

class TrackingData:
    def __init__(self):
        self.lock = threading.Lock()
        self.status = "Lost"
        self.dx = 0
        self.dy = 0
        self.distance = 0.0
        self.angle = 0.0
        self.score = 0.0
        self.control_mode = "Manual"  # or "Autonomous"

        # Store bounding-box height (for front/back PID)
        self.roi_height = 0

        # NEW: track whether forward/back targeting is enabled in autonomous mode
        self.forward_enabled = False
