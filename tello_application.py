import threading
from status_display import StatusDisplay
from tracking_data import TrackingData
from process_video import process_tello_video
from controller import handle_velocity_control
from tello_manager import TelloManager

class TelloApplication:
    """
    Main application that initializes the Tello manager and runs threads
    for video processing, control, and GUI status.
    """
    def __init__(self):
        self.tello_manager = TelloManager()
        self.status_display = None
        self.tracking_data = TrackingData()
        self.threads = []

    def start_threads(self):
        print("[INFO] Starting background threads...")

        video_thread = threading.Thread(
            target=process_tello_video,
            args=(self.tello_manager.tello, self.tracking_data),
            daemon=True
        )
        self.threads.append(video_thread)
        video_thread.start()

        control_thread = threading.Thread(
            target=handle_velocity_control,
            args=(self.tello_manager.tello, self.tracking_data),
            daemon=True
        )
        self.threads.append(control_thread)
        control_thread.start()

        self.status_display = StatusDisplay(self.tello_manager.tello, self.tracking_data)

    def run(self):
        print("[INFO] Starting Tello Application...")
        try:
            self.tello_manager.initialize()
            self.start_threads()
            self.status_display.run()
        except Exception as e:
            print(f"[ERROR] Application error: {e}")
        finally:
            self.tello_manager.cleanup()
