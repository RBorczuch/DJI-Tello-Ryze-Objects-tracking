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

        # --- NEW CODE ---
        # Store the tracked bounding box height so the controller can decide forward/back motion
        self.roi_height = 0
