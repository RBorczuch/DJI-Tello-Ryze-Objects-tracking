import cv2
import numpy as np
import math
import time
import queue
import threading
import os
import datetime
from threading import Lock
from tracking_data import TrackingData

# -----------------------------
# Global Constants
# -----------------------------
RESIZED_WIDTH = 480
RESIZED_HEIGHT = 360

DEFAULT_ROI_SIZE = 50
MIN_ROI_SIZE = 25
DEFAULT_MAX_ROI_SIZE = 200

REID_INTERVAL = 5
REID_FAILURE_LIMIT = 240

SIFT_UPDATE_SCORE_THRESHOLD = 70
SIFT_MATCH_RATIO = 0.75
SIFT_MIN_GOOD_MATCHES = 10

MIN_TRACK_DURATION = 0.5
MIN_TRACK_SCORE = 0.30

IMG_MARGIN = 10

FONT_SCALE = 0.5
FONT_THICKNESS = 1
LINE_SPACING = 20

STATUS_TEXT_POS = (10, 20)
REID_TEXT_POS = (10, 60)
CONTROL_MODE_TEXT_POS = (10, 140)
RECORDING_TEXT_POS = (10, 180)

COLOR_WHITE = (255, 255, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_YELLOW = (0, 255, 255)
COLOR_RED = (0, 0, 255)
COLOR_BLUE = (255, 0, 0)

FPS_TEXT_OFFSET_Y = -10


class VitTrack:
    """
    A wrapper for the OpenCV TrackerVit.
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
        found, bbox = self.model.update(image)
        score = self.model.getTrackingScore()
        return found, bbox, score


class VideoProcessor:
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

        # SIFT template: (keypoints, descriptors, bbox)
        self.sift_template = None

        self.reid_interval = REID_INTERVAL
        self.frame_number = 0
        self.reid_fail_count = 0
        self.reid_thread_running = False
        self.reid_lock = Lock()

        cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Tracking", 960, 720)
        cv2.setMouseCallback("Tracking", self._on_mouse)

    def _overlay_status(self, text, pos=STATUS_TEXT_POS):
        cv2.putText(self.frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                    FONT_SCALE, COLOR_WHITE, FONT_THICKNESS)

    def _on_mouse(self, event, x, y, flags, param):
        """
        Mouse callback for selecting and resetting ROIs or adjusting ROI size via mouse wheel.
        """
        self.mouse_x, self.mouse_y = x, y

        if event == cv2.EVENT_LBUTTONDOWN:
            # Left-click: initialize tracking at the selected ROI
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
            # Right-click: disable tracking
            self.tracking_enabled = False
            self.tracker = None
            self.tracking_start_time = None
            self.sift_template = None

        elif event == cv2.EVENT_MOUSEWHEEL:
            # Mouse wheel: adjust ROI size
            delta_size = 10 if flags > 0 else -10
            if self.max_roi_size is None:
                self.max_roi_size = DEFAULT_MAX_ROI_SIZE
            new_size = self.roi_size + delta_size
            self.roi_size = max(MIN_ROI_SIZE, min(new_size, self.max_roi_size))

    def draw_text(self, img, lines, start_x, start_y,
                  font_scale=FONT_SCALE, color=COLOR_WHITE,
                  thickness=FONT_THICKNESS, line_spacing=LINE_SPACING):
        """
        Draws multiple lines of text on the image at a given starting coordinate.
        """
        for i, line in enumerate(lines):
            pos = (start_x, start_y + i * line_spacing)
            cv2.putText(img, line, pos, cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, color, thickness)

    def draw_rectangle(self, img, bbox, color=COLOR_GREEN, thickness=1):
        """
        Draws a rectangle on the image based on the bounding box (x, y, w, h).
        Returns the center coordinates of the rectangle.
        """
        x, y, w, h = map(int, bbox)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)
        return x + w // 2, y + h // 2

    def draw_focused_area(self, img, x, y, size, color=COLOR_BLUE, thickness=1):
        """
        Draws a 'focus box' around the mouse cursor to show the intended ROI size.
        """
        half_size = size // 2
        x1 = max(0, x - half_size)
        y1 = max(0, y - half_size)
        x2 = min(img.shape[1], x + half_size)
        y2 = min(img.shape[0], y + half_size)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

    def calculate_distance_angle(self, dx, dy):
        """
        Calculates the Euclidean distance and angle (in degrees) relative to the center.
        dx, dy are offsets along the x and y axes, respectively.
        """
        dist = math.hypot(dx, dy)
        angle = math.degrees(math.atan2(dy, dx))
        if angle < 0:
            angle += 360
        return dist, angle

    def _run_reid(self, frame):
        """
        Threaded function to run SIFT-based re-identification.
        If it succeeds, we re-initialize the tracker with the new bounding box.
        """
        sift = cv2.SIFT_create()
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp_frame, des_frame = sift.detectAndCompute(gray_frame, None)
        success = False

        if des_frame is not None and self.sift_template and self.sift_template[1] is not None:
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(self.sift_template[1], des_frame, k=2)
            good = [m for m, n in matches if m.distance < SIFT_MATCH_RATIO * n.distance]
            if len(good) > SIFT_MIN_GOOD_MATCHES:
                src_pts = np.float32([self.sift_template[0][m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
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
        Processes the received BGR frame and returns a BGR frame with overlays.
        """
        with self.frame_lock:
            self.frame = frame

        img_h, img_w = self.frame.shape[:2]
        if self.max_roi_size is None:
            self.max_roi_size = min(img_h, img_w)

        center_x = img_w // 2
        center_y = img_h // 2

        # Draw a small dot in the frame center
        cv2.circle(self.frame, (center_x, center_y), 2, (0, 0, 255), -1)

        frame_copy = self.frame.copy()

        # --- Tracking ---
        if self.tracking_enabled and self.tracker is not None:
            found, bbox, score = self.tracker.infer(frame_copy)
            if found:
                x, y, w, h = map(int, bbox)
                # If the bounding box is too large, switch to re-identification
                if w >= img_w - IMG_MARGIN or h >= img_h - IMG_MARGIN:
                    self._overlay_status(self.STATUS_REID, STATUS_TEXT_POS)
                    self.tracking_enabled = False
                    self.tracker = None
                    # [MODIFIED] Force manual control during re-identification
                    with self.tracking_data.lock:
                        self.tracking_data.status = self.STATUS_REID
                        self.tracking_data.control_mode = "Manual"
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
                        f"Angle: {angle:.2f}st",
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

                    # If score is too low after minimum tracking duration -> tracking lost
                    if (time.time() - self.tracking_start_time > MIN_TRACK_DURATION) and (score < MIN_TRACK_SCORE):
                        self.tracking_enabled = False
                        self.tracker = None
                        self.tracking_start_time = None
                        self._overlay_status(self.STATUS_LOST, STATUS_TEXT_POS)
                        with self.tracking_data.lock:
                            self.tracking_data.status = self.STATUS_LOST
                            self.tracking_data.control_mode = "Manual"
            else:
                # Could not find the target
                self._overlay_status(self.STATUS_LOST, STATUS_TEXT_POS)
                self.tracking_enabled = False
                self.tracker = None
                self.tracking_start_time = None
                with self.tracking_data.lock:
                    self.tracking_data.status = self.STATUS_LOST
                    self.tracking_data.control_mode = "Manual"

            self.frame = frame_copy

            # Update the SIFT template if score is sufficiently high
            if found and score > SIFT_UPDATE_SCORE_THRESHOLD:
                x, y, w, h = map(int, bbox)
                if w > 0 and h > 0:
                    roi = frame_copy[y:y + h, x:x + w]
                    if roi.size != 0:
                        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                        sift = cv2.SIFT_create()
                        kp, des = sift.detectAndCompute(gray_roi, None)
                        self.sift_template = (kp, des, bbox)

        # --- Re-identification ---
        if not self.tracking_enabled and self.sift_template is not None:
            self.frame_number += 1
            if (self.frame_number % self.reid_interval == 0) and not self.reid_thread_running:
                self.reid_thread_running = True
                threading.Thread(target=self._run_reid, args=(self.frame.copy(),)).start()
                self.reid_fail_count += self.reid_interval

                # [MODIFIED] Force manual control during ReID
                with self.tracking_data.lock:
                    self.tracking_data.status = self.STATUS_REID
                    self.tracking_data.control_mode = "Manual"

                if self.reid_fail_count >= REID_FAILURE_LIMIT:
                    self._overlay_status(self.STATUS_LOST, REID_TEXT_POS)
                    with self.tracking_data.lock:
                        self.tracking_data.status = self.STATUS_LOST
                        self.tracking_data.control_mode = "Manual"
                    self.sift_template = None
                    self.tracking_enabled = False
                else:
                    self._overlay_status(self.STATUS_REID, REID_TEXT_POS)
            else:
                # Even if we're not triggering a new REID thread, display that we are in re-id mode
                self._overlay_status(self.STATUS_REID, REID_TEXT_POS)
                with self.tracking_data.lock:
                    self.tracking_data.status = self.STATUS_REID
                    self.tracking_data.control_mode = "Manual"

        # --- Control mode display ---
        with self.tracking_data.lock:
            mode_text = f"Mode: {self.tracking_data.control_mode}"
            if self.tracking_data.control_mode == "Autonomous":
                auto_mode_text = "Targeting" if self.tracking_data.forward_enabled else "Tracking"
            else:
                auto_mode_text = ""

        status_lines = [mode_text]
        if auto_mode_text:
            status_lines.append(auto_mode_text)

        self.draw_text(self.frame, status_lines,
                       CONTROL_MODE_TEXT_POS[0],
                       CONTROL_MODE_TEXT_POS[1])

        # Draw the focus box at the mouse cursor
        self.draw_focused_area(self.frame, self.mouse_x, self.mouse_y, self.roi_size)

        return self.frame


def read_frames(tello, frame_queue, stop_event):
    """
    Producer thread: reads frames from Tello and puts them into the queue.
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

        # Convert from RGB to BGR
        raw_frame = cv2.cvtColor(raw_frame, cv2.COLOR_RGB2BGR)

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
    Consumer thread: processes (BGR) frames, shows them in a window, and can record them.
    """
    processor = VideoProcessor(tracking_data)
    print("[INFO] Frame tracker thread started.")
    start_time = time.time()
    frame_count = 0

    # Recording settings
    recording = False
    video_writer = None
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps_record = 30

    # Folder for saved recordings
    recordings_folder = "recordings"
    os.makedirs(recordings_folder, exist_ok=True)

    try:
        while not stop_event.is_set():
            try:
                frame = frame_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            # Process frame (BGR -> BGR)
            processed = processor.process_frame(frame)
            frame_count += 1

            elapsed = time.time() - start_time
            fps_now = frame_count / elapsed if elapsed > 0 else 0.0

            # Draw FPS text
            processor.draw_text(
                processed,
                [f"FPS: {fps_now:.2f}"],
                10,
                processed.shape[0] + FPS_TEXT_OFFSET_Y
            )

            # Draw recording status
            rec_text = f"Recording: {'ON' if recording else 'OFF'}"
            processor.draw_text(processed, [rec_text],
                                RECORDING_TEXT_POS[0],
                                RECORDING_TEXT_POS[1])

            # Keyboard check (only works if this window is in focus)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                print("[INFO] ESC pressed, stopping tracker.")
                break
            elif key == ord('n'):
                # Toggle recording
                recording = not recording
                if recording:
                    import datetime
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"{recordings_folder}/output_{timestamp}.mp4"
                    h, w = processed.shape[:2]
                    video_writer = cv2.VideoWriter(filename, fourcc, fps_record, (w, h))
                    print(f"[INFO] Recording started: {filename}")
                else:
                    if video_writer is not None:
                        video_writer.release()
                        video_writer = None
                    print("[INFO] Recording stopped.")

            if recording and video_writer is not None:
                video_writer.write(processed)

            # Show in window
            cv2.imshow("Tracking", processed)

    except Exception as e:
        print(f"[ERROR] track_frames: {e}")
    finally:
        print("[INFO] Stopping track_frames, cleaning up.")
        if video_writer is not None:
            video_writer.release()
        cv2.destroyAllWindows()
