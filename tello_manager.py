import cv2
from djitellopy import Tello

class TelloManager:
    """
    Manages the Tello drone connection and streaming.
    """
    def __init__(self):
        self.tello = Tello()

    def initialize(self):
        print("[INFO] Initializing Tello drone...")
        try:
            self.tello.connect()
            print(f"[INFO] Battery: {self.tello.get_battery()}%")
            self.tello.streamon()
        except Exception as e:
            print(f"[ERROR] Failed to initialize Tello: {e}")
            raise

    def cleanup(self):
        print("[INFO] Cleaning up...")
        try:
            self.tello.land()
        except Exception as e:
            print(f"[WARNING] Failed to land: {e}")
        finally:
            self.tello.streamoff()
            self.tello.end()
            cv2.destroyAllWindows()
