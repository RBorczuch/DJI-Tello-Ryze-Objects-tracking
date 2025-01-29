import threading

class TrackingData:
    """
    Holds shared tracking values with a thread lock.
    """
    def __init__(self):
        self.lock = threading.Lock()
        self.status = "Lost"
        self.dx = 0
        self.dy = 0
        self.distance = 0.0
        self.angle = 0.0
        self.score = 0.0
        self.control_mode = "Manual"  # or "Autonomous"
        self.roi_height = 0
        self.forward_enabled = False
