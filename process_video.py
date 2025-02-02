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
# Frame resize dimensions.
RESIZED_WIDTH = 480
RESIZED_HEIGHT = 360

# Default ROI sizes.
DEFAULT_ROI_SIZE = 50        # Default region-of-interest (ROI) size.
MIN_ROI_SIZE = 25            # Minimum allowable ROI size.
DEFAULT_MAX_ROI_SIZE = 200   # Maximum allowable ROI size.

# Re-identification control.
REID_INTERVAL = 5            # Attempt re-identification every 5 frames when tracking is lost.
REID_FAILURE_LIMIT = 240     # After 240 frames of failed re-ID attempts, mark status as Lost.

# SIFT matching parameters.
SIFT_UPDATE_SCORE_THRESHOLD = 70    # Only update (record) SIFT template if tracker score exceeds 70.
SIFT_MATCH_RATIO = 0.75               # Ratio threshold for Lowe's ratio test.
SIFT_MIN_GOOD_MATCHES = 10            # Minimum number of good matches required for successful re-ID.

# Tracking loss thresholds.
MIN_TRACK_DURATION = 0.5    # Minimum time (in seconds) before checking tracking score.
MIN_TRACK_SCORE = 0.30      # If tracking score falls below this value, tracking is considered lost.

# Oversized object protection.
IMG_MARGIN = 10             # Margin from image border; if bbox width/height >= (image dimension - IMG_MARGIN), trigger re-ID.

# Overlay text styling.
FONT_SCALE = 0.5
FONT_THICKNESS = 1
LINE_SPACING = 20

# Predefined text positions.
STATUS_TEXT_POS = (10, 20)
REID_TEXT_POS = (10, 60)
CONTROL_MODE_TEXT_POS = (10, 140)

# Color definitions (BGR format).
COLOR_WHITE = (255, 255, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_YELLOW = (0, 255, 255)
COLOR_RED = (0, 0, 255)
COLOR_BLUE = (255, 0, 0)

# FPS text offset from the bottom of the frame.
FPS_TEXT_OFFSET_Y = -10  # Negative value to move the text upward.

# -----------------------------
# VitTrack Class: Wrapper for OpenCV TrackerVit.
# -----------------------------
class VitTrack:
    """
    Wraps the OpenCV TrackerVit to initialize tracking with a given ROI and update tracking on subsequent frames.
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
        """
        self.model.init(image, roi)

    def infer(self, image):
        """
        Updates the tracker with a new frame.

        Returns:
            found (bool): Indicates if the object is found.
            bbox (tuple): Bounding box of the object.
            score (float): Confidence score.
        """
        found, bbox = self.model.update(image)
        score = self.model.getTrackingScore()
        return found, bbox, score

# -----------------------------
# VideoProcessor Class: Processes frames for tracking and re-identification.
# -----------------------------
class VideoProcessor:
    """
    Processes frames for object tracking and re-identification.

    Key functionalities:
      - Maintains the active tracker and updates tracking info.
      - Updates the SIFT template only if the tracker score is high (> SIFT_UPDATE_SCORE_THRESHOLD).
      - If the tracked object's bounding box nearly fills the frame, disables tracking to force re-identification.
      - When tracking is lost, attempts SIFT-based re-identification every REID_INTERVAL frames in a separate thread.
      - If re-identification fails for REID_FAILURE_LIMIT frames, sets status to "Lost".
      - Overlays unified status and control mode information using helper methods.

    Uses the following status strings:
        STATUS_TRACKING: "Status: Tracking"
        STATUS_REID: "Status: Re-identification"
        STATUS_LOST: "Status: Lost"
    """
    # Status string constants.
    STATUS_TRACKING = "Status: Tracking"
    STATUS_REID = "Status: Re-identification"
    STATUS_LOST = "Status: Lost"

    def __init__(self, tracking_data, model_path='vittrack.onnx'):
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

        # Re-identification control variables.
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
        Helper method to overlay a status message on the current frame with unified styling.
        
        Args:
            text (str): The status message.
            pos (tuple): Position to display the text.
        """
        cv2.putText(self.frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                    FONT_SCALE, COLOR_WHITE, FONT_THICKNESS)

    def _on_mouse(self, event, x, y, flags, param):
        """
        Mouse callback for user interactions:
          - Left click: Initializes tracking with a new ROI and records the initial SIFT template.
          - Right click: Resets tracking and clears the SIFT template.
          - Mouse wheel: Adjusts the ROI size.
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
        Draws multiple lines of text on the provided image.
        
        Args:
            img: Image on which to draw.
            lines (list of str): List of text lines.
            start_x (int): Starting x-coordinate.
            start_y (int): Starting y-coordinate.
            font_scale (float): Scale of the text.
            color (tuple): Color of the text.
            thickness (int): Thickness of the text.
            line_spacing (int): Vertical spacing between lines.
        """
        for i, line in enumerate(lines):
            pos = (start_x, start_y + i * line_spacing)
            cv2.putText(img, line, pos, cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, color, thickness)

    def draw_rectangle(self, img, bbox, color=COLOR_GREEN, thickness=1):
        """
        Draws a rectangle on the image for the given bounding box.
        
        Args:
            img: The image on which to draw.
            bbox (tuple): Bounding box (x, y, width, height).
            color (tuple): Rectangle color.
            thickness (int): Line thickness.
            
        Returns:
            tuple: Center (cx, cy) of the rectangle.
        """
        x, y, w, h = map(int, bbox)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)
        return x + w // 2, y + h // 2

    def draw_focused_area(self, img, x, y, size, color=COLOR_BLUE, thickness=1):
        """
        Draws a focus box around the given point (typically the mouse position).
        
        Args:
            img: The image on which to draw.
            x (int): x-coordinate.
            y (int): y-coordinate.
            size (int): Size of the focus box.
            color (tuple): Color of the focus box.
            thickness (int): Line thickness.
        """
        half_size = size // 2
        x1 = max(0, x - half_size)
        y1 = max(0, y - half_size)
        x2 = min(img.shape[1], x + half_size)
        y2 = min(img.shape[0], y + half_size)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

    def calculate_distance_angle(self, dx, dy):
        """
        Calculates the Euclidean distance and angle (in degrees) from differences dx and dy.
        
        Args:
            dx (float): Difference in x-coordinates.
            dy (float): Difference in y-coordinates.
            
        Returns:
            tuple: (distance, angle)
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
        
        Args:
            frame: A copy of the current frame used for re-identification.
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
        Processes the input frame for tracking and re-identification.

        Workflow:
          1. Updates the active tracker and overlays tracking info.
          2. If the tracker is confident (score > SIFT_UPDATE_SCORE_THRESHOLD),
             updates the SIFT template.
          3. If the tracked object's bbox nearly fills the frame (with IMG_MARGIN), forces re-ID.
          4. When tracking is lost, attempts re-identification every REID_INTERVAL frames.
          5. If re-ID fails for REID_FAILURE_LIMIT frames, sets status to "Lost".
          6. Overlays unified status and control mode information.

        Args:
            frame: The current frame.
            
        Returns:
            The processed frame with overlayed tracking and status information.
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
                # If bbox nearly fills the image, trigger re-identification.
                if w >= img_w - IMG_MARGIN or h >= img_h - IMG_MARGIN:
                    self._overlay_status(self.STATUS_REID, STATUS_TEXT_POS)
                    self.tracking_enabled = False
                    self.tracker = None
                    # SIFT template remains for re-ID.
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

                    # If tracking confidence is too low, disable tracking.
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

            # Update SIFT template if tracking is confident.
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

        # --- Overlay control mode information ---
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

        # Draw the focus box at the current mouse position.
        self.draw_focused_area(self.frame, self.mouse_x, self.mouse_y, self.roi_size)
        return self.frame

def read_frames(tello, frame_queue, stop_event):
    """
    Producer thread: Continuously reads frames from the Tello drone, resizes them,
    and enqueues them into frame_queue.
    
    Args:
        tello: Tello drone instance.
        frame_queue (queue.Queue): Queue to store frames.
        stop_event (threading.Event): Event to signal thread stopping.
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
    Consumer thread: Retrieves frames from frame_queue, processes each frame for tracking
    and re-identification, overlays status and control mode information, and displays the frame.
    
    Args:
        frame_queue (queue.Queue): Queue from which frames are retrieved.
        tracking_data (TrackingData): Shared tracking data.
        stop_event (threading.Event): Event to signal thread stopping.
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
