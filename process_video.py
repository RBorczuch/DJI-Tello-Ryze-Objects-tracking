# process_video.py

import cv2
import numpy as np
import math
import time
import queue
import threading
from threading import Lock
from tracking_data import TrackingData

# -----------------------------
# Global Constants
# -----------------------------
RESIZED_WIDTH = 480      # Width for resizing frames from the Tello camera
RESIZED_HEIGHT = 360     # Height for resizing frames from the Tello camera

DEFAULT_ROI_SIZE = 50    # Default region-of-interest (ROI) size
MIN_ROI_SIZE = 25        # Minimum allowable ROI size
DEFAULT_MAX_ROI_SIZE = 200  # Maximum allowable ROI size

REID_INTERVAL = 5        # Attempt re-identification every 5 frames when tracking is lost
REID_FAILURE_LIMIT = 240 # After 240 frames of failed re-ID attempts, mark status as Lost

SIFT_UPDATE_SCORE_THRESHOLD = 70  # Only update SIFT template if tracker score > 70
SIFT_MATCH_RATIO = 0.75           # Ratio threshold for Lowe's ratio test
SIFT_MIN_GOOD_MATCHES = 10        # Min number of good matches for successful re-ID

MIN_TRACK_DURATION = 0.5          # Minimum tracking time before checking score
MIN_TRACK_SCORE = 0.30            # Threshold for considering tracking lost

IMG_MARGIN = 10                   # Margin from image border to prevent bounding box from filling the frame

FONT_SCALE = 0.5
FONT_THICKNESS = 1
LINE_SPACING = 20

STATUS_TEXT_POS = (10, 20)
REID_TEXT_POS = (10, 60)
CONTROL_MODE_TEXT_POS = (10, 140)

COLOR_WHITE = (255, 255, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_YELLOW = (0, 255, 255)
COLOR_RED = (0, 0, 255)
COLOR_BLUE = (255, 0, 0)

FPS_TEXT_OFFSET_Y = -10  # Offset to place FPS text above the bottom edge of the frame

class VitTrack:
    """
    Wraps the OpenCV TrackerVit to initialize and update tracking.
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
        """
        Initializes the tracker on the provided frame with the specified ROI.

        Args:
            image (np.ndarray): Frame image.
            roi (tuple): (x, y, width, height) bounding box.
        """
        self.model.init(image, roi)

    def infer(self, image):
        """
        Updates the tracker with a new frame.

        Args:
            image (np.ndarray): The current frame.

        Returns:
            found (bool): Indicates if the object is found in this frame.
            bbox (tuple): The updated bounding box.
            score (float): Confidence score from the tracker.
        """
        found, bbox = self.model.update(image)
        score = self.model.getTrackingScore()
        return found, bbox, score


class VideoProcessor:
    """
    Processes frames for object tracking and re-identification, updating shared tracking data.
    """
    STATUS_TRACKING = "Status: Tracking"
    STATUS_REID = "Status: Re-identification"
    STATUS_LOST = "Status: Lost"

    def __init__(self, tracking_data, model_path='vittrack.onnx'):
        """
        Args:
            tracking_data (TrackingData): Shared data object for tracking status/coordinates.
            model_path (str): Path to the ONNX model for TrackerVit.
        """
        self.tracking_data = tracking_data
        self.model_path = model_path

        self.frame_lock = Lock()
        self.frame = None

        self.tracker = None
        self.tracking_enabled = False
        self.tracking_start_time = None

        self.mouse_x = 0
        self.mouse_y = 0
        self.new_bbox = None
        self.roi_size = DEFAULT_ROI_SIZE
        self.min_roi_size = MIN_ROI_SIZE
        self.max_roi_size = None

        # SIFT template stored as (keypoints, descriptors, bbox)
        self.sift_template = None

        # Re-identification control
        self.reid_interval = REID_INTERVAL
        self.frame_number = 0
        self.reid_fail_count = 0
        self.reid_thread_running = False
        self.reid_lock = Lock()

        cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Tracking", 960, 720)
        cv2.setMouseCallback("Tracking", self._on_mouse)

    def _overlay_status(self, text, pos=STATUS_TEXT_POS):
        """
        Helper to overlay a status message on the current frame with consistent styling.
        """
        cv2.putText(self.frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                    FONT_SCALE, COLOR_WHITE, FONT_THICKNESS)

    def _on_mouse(self, event, x, y, flags, param):
        """
        Mouse callback for user interactions:
          - Left click: Initialize tracking with new ROI and record SIFT template.
          - Right click: Reset tracking and clear SIFT template.
          - Mouse wheel: Adjust ROI size.
        """
        self.mouse_x, self.mouse_y = x, y

        if event == cv2.EVENT_LBUTTONDOWN:
            x1, y1 = x - self.roi_size // 2, y - self.roi_size // 2
            self.new_bbox = (x1, y1, self.roi_size, self.roi_size)
            self.tracking_enabled = True
            self.tracker = VitTrack(self.model_path)
            with self.frame_lock:
                if self.frame is not None:
                    self.tracker.init(self.frame, self.new_bbox)
                    roi = self.frame[y1:y1 + self.roi_size, x1:x1 + self.roi_size]
                    if roi.size != 0:
                        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                        sift = cv2.SIFT_create()
                        kp, des = sift.detectAndCompute(gray_roi, None)
                        self.sift_template = (kp, des, self.new_bbox)
                    else:
                        print("[ERROR] Selected ROI is empty.")
                else:
                    print("[ERROR] No frame available for initialization.")
                    self.tracking_enabled = False
            self.tracking_start_time = time.time()
            self.frame_number = 0
            self.reid_fail_count = 0

        elif event == cv2.EVENT_RBUTTONDOWN:
            self.tracking_enabled = False
            self.tracker = None
            self.tracking_start_time = None
            self.sift_template = None

        elif event == cv2.EVENT_MOUSEWHEEL:
            delta_size = 10 if flags > 0 else -10
            if self.max_roi_size is None:
                self.max_roi_size = DEFAULT_MAX_ROI_SIZE
            new_size = self.roi_size + delta_size
            self.roi_size = max(MIN_ROI_SIZE, min(new_size, self.max_roi_size))

    def draw_text(self, img, lines, start_x, start_y, font_scale=FONT_SCALE,
                  color=COLOR_WHITE, thickness=FONT_THICKNESS, line_spacing=LINE_SPACING):
        """
        Draws multiple lines of text at the specified position on the image.
        """
        for i, line in enumerate(lines):
            pos = (start_x, start_y + i * line_spacing)
            cv2.putText(img, line, pos, cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, color, thickness)

    def draw_rectangle(self, img, bbox, color=COLOR_GREEN, thickness=1):
        """
        Draws a rectangle corresponding to the bounding box on the image.
        Returns the center (cx, cy) of the rectangle.
        """
        x, y, w, h = map(int, bbox)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)
        return x + w // 2, y + h // 2

    def draw_focused_area(self, img, x, y, size, color=COLOR_BLUE, thickness=1):
        """
        Draws a focus box around the specified point (e.g., mouse position).
        """
        half_size = size // 2
        x1 = max(0, x - half_size)
        y1 = max(0, y - half_size)
        x2 = min(img.shape[1], x + half_size)
        y2 = min(img.shape[0], y + half_size)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

    def calculate_distance_angle(self, dx, dy):
        """
        Computes Euclidean distance and angle (in degrees) from the given dx, dy.
        """
        dist = math.hypot(dx, dy)
        angle = math.degrees(math.atan2(dy, dx))
        if angle < 0:
            angle += 360
        return dist, angle

    def _run_reid(self, frame):
        """
        Executes SIFT-based re-identification in a separate thread.
        If successful, reinitializes the tracker with the new bounding box.
        """
        sift = cv2.SIFT_create()
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp_frame, des_frame = sift.detectAndCompute(gray_frame, None)
        success = False
        if des_frame is not None and self.sift_template is not None and self.sift_template[1] is not None:
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(self.sift_template[1], des_frame, k=2)
            good = [m for m, n in matches if m.distance < SIFT_MATCH_RATIO * n.distance]
            if len(good) > SIFT_MIN_GOOD_MATCHES:
                src_pts = np.float32([self.sift_template[0][m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                if M is not None:
                    x, y, w, h = self.sift_template[2]
                    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
                    dst = cv2.perspectiveTransform(corners, M)
                    new_bbox = cv2.boundingRect(dst)
                    with self.reid_lock:
                        self.tracker = VitTrack(self.model_path)
                        self.tracker.init(frame, new_bbox)
                        self.tracking_enabled = True
                        self.tracking_start_time = time.time()
                    success = True
        self.reid_thread_running = False
        if success:
            self.reid_fail_count = 0

    def process_frame(self, frame):
        """
        High-level method to process the current frame:
          1. Update the tracker if tracking is enabled.
          2. Handle re-identification if tracking is lost.
          3. Overlay status and control-mode info.
          4. Update shared tracking data (dx, dy, distance, angle, etc.).
        """
        with self.frame_lock:
            self.frame = frame

        img_h, img_w = self.frame.shape[:2]
        if self.max_roi_size is None:
            self.max_roi_size = min(img_h, img_w)

        center_x = self.frame.shape[1] // 2
        center_y = self.frame.shape[0] // 2
        cv2.circle(self.frame, (center_x, center_y), 2, (0, 0, 255), -1)
        frame_copy = self.frame.copy()

        # --- Tracking branch ---
        if self.tracking_enabled and self.tracker is not None:
            found, bbox, score = self.tracker.infer(frame_copy)
            if found:
                x, y, w, h = map(int, bbox)
                # If bbox nearly fills the image, force re-identification
                if w >= img_w - IMG_MARGIN or h >= img_h - IMG_MARGIN:
                    self._overlay_status(self.STATUS_REID, STATUS_TEXT_POS)
                    self.tracking_enabled = False
                    self.tracker = None
                else:
                    cx, cy = self.draw_rectangle(frame_copy, bbox)
                    cv2.line(frame_copy, (center_x, center_y), (cx, cy), COLOR_YELLOW, 1)
                    dx = center_x - cx
                    dy = cy - center_y
                    dist, angle = self.calculate_distance_angle(dx, dy)

                    info = [
                        self.STATUS_TRACKING,
                        f"Score: {score:.2f}",
                        f"dx: {dx}px",
                        f"dy: {dy}px",
                        f"Distance: {dist:.2f}px",
                        f"Angle: {angle:.2f}°",
                    ]
                    self.draw_text(frame_copy, info, STATUS_TEXT_POS[0], STATUS_TEXT_POS[1])

                    with self.tracking_data.lock:
                        self.tracking_data.status = self.STATUS_TRACKING
                        self.tracking_data.dx = dx
                        self.tracking_data.dy = dy
                        self.tracking_data.distance = dist
                        self.tracking_data.angle = angle
                        self.tracking_data.score = score
                        self.tracking_data.roi_height = h

                    # If confidence too low after MIN_TRACK_DURATION, consider tracking lost
                    if (time.time() - self.tracking_start_time > MIN_TRACK_DURATION) and (score < MIN_TRACK_SCORE):
                        self.tracking_enabled = False
                        self.tracker = None
                        self.tracking_start_time = None
                        self._overlay_status(self.STATUS_LOST, STATUS_TEXT_POS)
                        with self.tracking_data.lock:
                            self.tracking_data.status = self.STATUS_LOST
            else:
                self._overlay_status(self.STATUS_LOST, STATUS_TEXT_POS)
                self.tracking_enabled = False
                self.tracker = None
                self.tracking_start_time = None
                with self.tracking_data.lock:
                    self.tracking_data.status = self.STATUS_LOST

            self.frame = frame_copy

            # Update SIFT template if tracking is confident
            if found and score > SIFT_UPDATE_SCORE_THRESHOLD:
                x, y, w, h = map(int, bbox)
                if w > 0 and h > 0:
                    roi = frame_copy[y:y+h, x:x+w]
                    if roi.size != 0:
                        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                        sift = cv2.SIFT_create()
                        kp, des = sift.detectAndCompute(gray_roi, None)
                        self.sift_template = (kp, des, bbox)

        # --- Re-identification branch ---
        if not self.tracking_enabled and self.sift_template is not None:
            self.frame_number += 1
            if (self.frame_number % self.reid_interval == 0) and (not self.reid_thread_running):
                self.reid_thread_running = True
                threading.Thread(target=self._run_reid, args=(self.frame.copy(),)).start()
                self.reid_fail_count += self.reid_interval
                if self.reid_fail_count >= REID_FAILURE_LIMIT:
                    self._overlay_status(self.STATUS_LOST, REID_TEXT_POS)
                    with self.tracking_data.lock:
                        self.tracking_data.status = self.STATUS_LOST
                    self.sift_template = None
                    self.tracking_enabled = False
                else:
                    self._overlay_status(self.STATUS_REID, REID_TEXT_POS)
            else:
                self._overlay_status(self.STATUS_REID, REID_TEXT_POS)

        # --- Overlay control mode info ---
        with self.tracking_data.lock:
            mode_text = f"Mode: {self.tracking_data.control_mode}"
            if self.tracking_data.control_mode == "Autonomous":
                auto_mode_text = "Targeting" if self.tracking_data.forward_enabled else "Tracking"
            else:
                auto_mode_text = ""
        status_lines = [mode_text]
        if auto_mode_text:
            status_lines.append(auto_mode_text)
        self.draw_text(self.frame, status_lines, CONTROL_MODE_TEXT_POS[0], CONTROL_MODE_TEXT_POS[1])

        # Draw the focus box at the current mouse position
        self.draw_focused_area(self.frame, self.mouse_x, self.mouse_y, self.roi_size)
        return self.frame


def read_frames(tello, frame_queue, stop_event):
    """
    Producer thread: continuously reads frames from Tello camera,
    resizes them, and enqueues them for processing.
    """
    print("[INFO] Frame reader thread started.")
    while not stop_event.is_set():
        frame_read = tello.get_frame_read()
        if frame_read.stopped:
            print("[WARN] Tello frame read was stopped.")
            break

        raw_frame = frame_read.frame
        if raw_frame is None:
            continue

        resized = cv2.resize(raw_frame, (RESIZED_WIDTH, RESIZED_HEIGHT))
        if frame_queue.full():
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                pass
        frame_queue.put(resized)
    print("[INFO] Frame reader thread exiting.")


def track_frames(frame_queue, tracking_data, stop_event):
    """
    Consumer thread: retrieves frames from the queue, processes them for
    tracking/re-identification, and displays the result.
    """
    processor = VideoProcessor(tracking_data)
    print("[INFO] Frame tracker thread started.")
    start_time = time.time()
    frame_count = 0

    try:
        while not stop_event.is_set():
            try:
                frame = frame_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed = processor.process_frame(rgb_frame)
            frame_count += 1
            fps = frame_count / (time.time() - start_time)
            processor.draw_text(processor.frame, [f"FPS: {fps:.2f}"],
                                10, processor.frame.shape[0] + FPS_TEXT_OFFSET_Y)
            cv2.imshow("Tracking", processor.frame)
            if cv2.waitKey(1) & 0xFF == 27:
                print("[INFO] ESC pressed, stopping tracker.")
                break
    except Exception as e:
        print(f"[ERROR] track_frames: {e}")
    finally:
        print("[INFO] Stopping track_frames, cleaning up.")
        cv2.destroyAllWindows()
