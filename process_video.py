# process_video.py

import cv2
import numpy as np
import math
import time
from threading import Lock
from tracking_data import TrackingData  # Ensure this module is available

# Constants for easy resolution changes
RESIZED_WIDTH = 480
RESIZED_HEIGHT = 360

class VitTrack:
    """
    A class encapsulating the ViTTrack tracker using OpenCV's implementation.
    """
    def __init__(self, model_path, backend_id=0, target_id=0):
        self.model_path = model_path
        self.backend_id = backend_id
        self.target_id = target_id

        self.params = cv2.TrackerVit_Params()
        self.params.net = self.model_path
        self.params.backend = self.backend_id
        self.params.target = self.target_id

        self.model = cv2.TrackerVit_create(self.params)

    def init(self, image, roi):
        self.model.init(image, roi)

    def infer(self, image):
        is_located, bbox = self.model.update(image)
        score = self.model.getTrackingScore()
        return is_located, bbox, score

class VideoProcessor:
    """
    A class to handle video processing and tracking using OpenCV.
    """
    def __init__(self, tello, tracking_data, model_path='vittrack.onnx'):
        self.tello = tello
        self.tracking_data = tracking_data
        self.model_path = model_path

        # Variables for frame handling
        self.frame = None
        self.frame_lock = Lock()

        # Tracker variables
        self.tracker = None
        self.tracking_enabled = False
        self.tracking_start_time = None

        # Mouse interaction variables
        self.mouse_x, self.mouse_y = 0, 0
        self.new_bbox = None
        self.roi_size = 50
        self.min_roi_size = 25
        self.max_roi_size = None

        # Set up the OpenCV window
        cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Tracking", 960, 720)
        cv2.setMouseCallback(
            "Tracking", 
            lambda event, x, y, flags, param: self.mouse_callback(event, x, y, flags, param)
        )

    def mouse_callback(self, event, x, y, flags, param):
        self.mouse_x, self.mouse_y = x, y

        if event == cv2.EVENT_LBUTTONDOWN:
            x1, y1 = x - self.roi_size // 2, y - self.roi_size // 2
            self.new_bbox = (x1, y1, self.roi_size, self.roi_size)
            self.tracking_enabled = True
            self.tracker = VitTrack(self.model_path)
            with self.frame_lock:
                if self.frame is not None:
                    self.tracker.init(self.frame, self.new_bbox)
                else:
                    print("[ERROR] Frame is None. Cannot initialize tracker.")
                    self.tracking_enabled = False
            self.tracking_start_time = time.time()

        elif event == cv2.EVENT_RBUTTONDOWN:
            self.tracking_enabled = False
            self.tracker = None
            self.tracking_start_time = None

        elif event == cv2.EVENT_MOUSEWHEEL:
            delta_size = 10 if flags > 0 else -10
            self.roi_size = max(
                self.min_roi_size, 
                min(self.roi_size + delta_size, self.max_roi_size)
            )

    def draw_text(self, img, text_lines, start_x, start_y, font_scale=0.5,
                  color=(255, 255, 255), thickness=1, line_spacing=20):
        for i, line in enumerate(text_lines):
            position = (start_x, start_y + i * line_spacing)
            cv2.putText(img, line, position, cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, color, thickness)

    def draw_rectangle(self, img, bbox, color=(55, 55, 0), thickness=1):
        x, y, w, h = [int(coord) for coord in bbox]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)
        return x + w // 2, y + h // 2

    def draw_focused_area(self, img, x, y, size, color=(255, 0, 0), thickness=1):
        half_size = size // 2
        x1 = max(0, x - half_size)
        y1 = max(0, y - half_size)
        x2 = min(img.shape[1], x + half_size)
        y2 = min(img.shape[0], y + half_size)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

    def calculate_distance_and_angle(self, dx, dy):
        distance = math.hypot(dx, dy)
        angle_rad = math.atan2(dy, dx)
        angle_deg = math.degrees(angle_rad)
        if angle_deg < 0:
            angle_deg += 360
        return distance, angle_deg

    def update_tracking_info(self, frame, center_x, center_y):
        """
        Updates tracking information and displays it on the image.
        """
        stabilization_time = 0.5  # Seconds
        is_located, bbox, score = self.tracker.infer(frame)
        if is_located:
            roi_center_x, roi_center_y = self.draw_rectangle(frame, bbox)
            # Draw line between frame center and ROI center
            cv2.line(frame, (center_x, center_y),
                     (roi_center_x, roi_center_y), (0, 255, 255), 1)

            dx = center_x - roi_center_x
            dy = roi_center_y - center_y  # Inverted y-axis
            distance, angle = self.calculate_distance_and_angle(dx, dy)

            # --- NEW CODE ---
            # Extract height from bbox for front/back PID
            # bbox = (x, y, w, h)
            _, _, _, h = bbox

            tracking_info = [
                "Status: Tracking",
                f"Score: {score:.2f}",
                f"dx: {dx}px",
                f"dy: {dy}px",
                f"Distance: {distance:.2f}px",
                f"Angle: {angle:.2f}st",
                f"h (ROI): {h}px",  # Just for debugging
            ]
            self.draw_text(frame, tracking_info, 10, 20)

            # Update shared tracking data
            with self.tracking_data.lock:
                self.tracking_data.status = "Tracking"
                self.tracking_data.dx = dx
                self.tracking_data.dy = dy
                self.tracking_data.distance = distance
                self.tracking_data.angle = angle
                self.tracking_data.score = score
                self.tracking_data.roi_height = h  # <--- store ROI height

            # Check for low score after stabilization time
            elapsed_time = time.time() - self.tracking_start_time
            if elapsed_time > stabilization_time and score < 0.30:
                self.tracking_enabled = False
                self.tracker = None
                self.tracking_start_time = None
                self.draw_text(frame, ["Status: Lost"], 10, 20, color=(255, 0, 255))

                # Update tracking data to reflect loss
                with self.tracking_data.lock:
                    self.tracking_data.status = "Lost"
                    self.tracking_data.dx = 0
                    self.tracking_data.dy = 0
                    self.tracking_data.distance = 0.0
                    self.tracking_data.angle = 0.0
                    self.tracking_data.score = 0.0
                    self.tracking_data.roi_height = 0
        else:
            self.draw_text(frame, ["Status: Lost"], 10, 20, color=(255, 0, 255))
            self.tracking_enabled = False
            self.tracker = None
            self.tracking_start_time = None

            # Update tracking data to reflect loss
            with self.tracking_data.lock:
                self.tracking_data.status = "Lost"
                self.tracking_data.dx = 0
                self.tracking_data.dy = 0
                self.tracking_data.distance = 0.0
                self.tracking_data.angle = 0.0
                self.tracking_data.score = 0.0
                self.tracking_data.roi_height = 0

    def process_video(self):
        print("Video processing started.")
        start_time = time.time()
        num_frames = 0

        try:
            while True:
                frame_read = self.tello.get_frame_read()
                if frame_read.stopped:
                    print("[ERROR] Frame read stopped.")
                    break

                new_frame = frame_read.frame
                if new_frame is None:
                    print("[ERROR] Failed to read frame from Tello.")
                    continue

                resized_frame = cv2.resize(new_frame, (RESIZED_WIDTH, RESIZED_HEIGHT))
                rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

                with self.frame_lock:
                    self.frame = rgb_frame.copy()

                if self.max_roi_size is None:
                    self.max_roi_size = min(self.frame.shape[1], self.frame.shape[0])

                center_x = self.frame.shape[1] // 2
                center_y = self.frame.shape[0] // 2
                cv2.circle(self.frame, (center_x, center_y), 2, (0, 0, 255), -1)

                if self.tracking_enabled and self.tracker is not None:
                    with self.frame_lock:
                        frame_copy = self.frame.copy()
                    self.update_tracking_info(frame_copy, center_x, center_y)
                    self.frame = frame_copy
                else:
                    self.draw_text(self.frame, ["Status: Lost"], 10, 20, color=(200, 200, 200))
                    with self.tracking_data.lock:
                        self.tracking_data.status = "Lost"
                        self.tracking_data.dx = 0
                        self.tracking_data.dy = 0
                        self.tracking_data.distance = 0.0
                        self.tracking_data.angle = 0.0
                        self.tracking_data.score = 0.0
                        self.tracking_data.roi_height = 0

                with self.tracking_data.lock:
                    mode_text = f"Mode: {self.tracking_data.control_mode}"
                self.draw_text(
                    self.frame,
                    [mode_text],
                    start_x=10,
                    start_y=140,
                    font_scale=0.5,
                    color=(255, 255, 255),
                    thickness=1,
                    line_spacing=20
                )

                self.draw_focused_area(self.frame, self.mouse_x, self.mouse_y, self.roi_size)

                num_frames += 1
                fps = num_frames / (time.time() - start_time)
                self.draw_text(self.frame, [f"FPS: {fps:.2f}"],
                               10, self.frame.shape[0] - 10)

                cv2.imshow("Tracking", self.frame)

                if cv2.waitKey(1) & 0xFF == 27:
                    print("Video processing terminated.")
                    break

        except Exception as e:
            print(f"Error in video processing: {e}")
        finally:
            cv2.destroyAllWindows()

def process_tello_video(tello, tracking_data):
    video_processor = VideoProcessor(tello, tracking_data)
    video_processor.process_video()
