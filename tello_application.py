# tello_application.py

import threading
import queue
from status_display import StatusDisplay
from tracking_data import TrackingData
from process_video import read_frames, track_frames
from controller import handle_velocity_control
from tello_manager import TelloManager

class TelloApplication:
    """
    Orchestrates Tello drone usage:
      - Connects/initializes the drone
      - Spawns background threads for video capture, tracking, control, and UI
      - Cleans up on exit
    """
    def __init__(self):
        self.tello_manager = TelloManager()
        self.status_display = None
        self.tracking_data = TrackingData()
        self.threads = []
        self.stop_event = threading.Event()

        # This queue will hold frames from the producer to the consumer
        self.frame_queue = queue.Queue(maxsize=10)

    def start_threads(self):
        """
        Starts all background threads for video reading, tracking, control, and GUI.
        """
        print("[INFO] Starting background threads...")

        # Producer: continuously read frames from Tello
        producer_thread = threading.Thread(
            target=read_frames,
            args=(self.tello_manager.tello, self.frame_queue, self.stop_event),
            daemon=True
        )
        self.threads.append(producer_thread)
        producer_thread.start()

        # Consumer: track objects and show overlay
        consumer_thread = threading.Thread(
            target=track_frames,
            args=(self.frame_queue, self.tracking_data, self.stop_event),
            daemon=True
        )
        self.threads.append(consumer_thread)
        consumer_thread.start()

        # Control thread: manual/autonomous
        control_thread = threading.Thread(
            target=handle_velocity_control,
            args=(self.tello_manager.tello, self.tracking_data),
            daemon=True
        )
        self.threads.append(control_thread)
        control_thread.start()

        # Status Display (GUI)
        self.status_display = StatusDisplay(self.tello_manager.tello, self.tracking_data)

    def run(self):
        """
        Initializes the Tello drone, starts background threads, and launches the GUI.
        """
        print("[INFO] Starting Tello Application...")
        try:
            self.tello_manager.initialize()
            self.start_threads()
            self.status_display.run()
        except Exception as e:
            print(f"[ERROR] Application error: {e}")
        finally:
            # Signal threads to stop
            self.stop_event.set()
            self.tello_manager.cleanup()
