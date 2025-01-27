# main.py
import threading
import cv2
from djitellopy import Tello
from process_video import process_tello_video
from controller import handle_velocity_control
from status_display import StatusDisplay
from tracking_data import TrackingData  # Import the shared data class


class TelloManager:
    def __init__(self):
        self.tello = Tello()

    def initialize(self):
        """
        Initializes the Tello drone and starts video streaming.
        """
        print("[INFO] Initializing Tello drone...")
        try:
            self.tello.connect()
            print(f"[INFO] Battery: {self.tello.get_battery()}%")
            self.tello.streamon()
        except Exception as e:
            print(f"[ERROR] Failed to initialize Tello: {e}")
            raise

    def cleanup(self):
        """
        Cleans up resources before exiting the application.
        """
        print("[INFO] Cleaning up resources...")
        try:
            self.tello.land()
        except Exception as e:
            print(f"[WARNING] Failed to land: {e}")
        finally:
            self.tello.streamoff()
            self.tello.end()
            cv2.destroyAllWindows()


class TelloApplication:
    def __init__(self):
        self.tello_manager = TelloManager()
        self.status_display = None
        self.tracking_data = TrackingData()  # Create an instance of TrackingData
        self.threads = []

    def start_threads(self):
        """
        Starts the necessary threads for video processing, control, and status updates.
        """
        print("[INFO] Starting threads...")

        # Video processing thread
        video_thread = threading.Thread(
            target=process_tello_video,
            args=(self.tello_manager.tello, self.tracking_data)
        )
        video_thread.daemon = True
        self.threads.append(video_thread)
        video_thread.start()

        # Control thread
        control_thread = threading.Thread(target=handle_velocity_control, args=(self.tello_manager.tello,))
        control_thread.daemon = True
        self.threads.append(control_thread)
        control_thread.start()

        # Status display
        self.status_display = StatusDisplay(self.tello_manager.tello, self.tracking_data)

    def run(self):
        """
        Runs the main application loop.
        """
        print("[INFO] Starting Tello Application...")
        try:
            self.tello_manager.initialize()
            self.start_threads()

            # Start GUI
            self.status_display.run()
        except Exception as e:
            print(f"[ERROR] Application encountered an error: {e}")
        finally:
            self.tello_manager.cleanup()


if __name__ == "__main__":
    app = TelloApplication()
    app.run()
