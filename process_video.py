# process_video.py

import cv2
import numpy as np
import math
import time
import queue
from threading import Lock
from tracking_data import TrackingData

RESIZED_WIDTH = 480
RESIZED_HEIGHT = 360

class VitTrack:
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
    """
    Consumes frames, applies tracking, updates tracking_data,
    and displays results in an OpenCV window.
    """
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
        self.roi_size = 50
        self.min_roi_size = 25
        self.max_roi_size = None

        cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Tracking", 960, 720)
        cv2.setMouseCallback("Tracking", self._on_mouse)

    def _on_mouse(self, event, x, y, flags, param):
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
                    print("[ERROR] No frame to init tracker.")
                    self.tracking_enabled = False
            self.tracking_start_time = time.time()

        elif event == cv2.EVENT_RBUTTONDOWN:
            self.tracking_enabled = False
            self.tracker = None
            self.tracking_start_time = None

        elif event == cv2.EVENT_MOUSEWHEEL:
            delta_size = 10 if flags > 0 else -10
            if self.max_roi_size is None:
                self.max_roi_size = 200
            new_size = self.roi_size + delta_size
            self.roi_size = max(self.min_roi_size, min(new_size, self.max_roi_size))

    def draw_text(self, img, lines, start_x, start_y, font_scale=0.5,
                  color=(255, 255, 255), thickness=1, line_spacing=20):
        for i, line in enumerate(lines):
            pos = (start_x, start_y + i * line_spacing)
            cv2.putText(img, line, pos, cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, color, thickness)

    def draw_rectangle(self, img, bbox, color=(0, 255, 0), thickness=1):
        x, y, w, h = map(int, bbox)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)
        return x + w // 2, y + h // 2

    def draw_focused_area(self, img, x, y, size, color=(255, 0, 0), thickness=1):
        half_size = size // 2
        x1 = max(0, x - half_size)
        y1 = max(0, y - half_size)
        x2 = min(img.shape[1], x + half_size)
        y2 = min(img.shape[0], y + half_size)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

    def calculate_distance_angle(self, dx, dy):
        dist = math.hypot(dx, dy)
        angle = math.degrees(math.atan2(dy, dx))
        if angle < 0:
            angle += 360
        return dist, angle

    def process_frame(self, frame):
        """
        Processes a single frame: apply tracking if enabled, update tracking_data,
        draw overlays, and show it in the 'Tracking' window.
        """
        with self.frame_lock:
            self.frame = frame

        if self.max_roi_size is None:
            self.max_roi_size = min(self.frame.shape[0], self.frame.shape[1])

        center_x = self.frame.shape[1] // 2
        center_y = self.frame.shape[0] // 2
        cv2.circle(self.frame, (center_x, center_y), 2, (0, 0, 255), -1)

        if self.tracking_enabled and self.tracker is not None:
            frame_copy = self.frame.copy()
            found, bbox, score = self.tracker.infer(frame_copy)

            if found:
                cx, cy = self.draw_rectangle(frame_copy, bbox)
                cv2.line(frame_copy, (center_x, center_y), (cx, cy), (0, 255, 255), 1)
                dx = center_x - cx
                dy = cy - center_y
                dist, angle = self.calculate_distance_angle(dx, dy)
                _, _, _, h = bbox

                info = [
                    "Status: Tracking",
                    f"Score: {score:.2f}",
                    f"dx: {dx}px",
                    f"dy: {dy}px",
                    f"Distance: {dist:.2f}px",
                    f"Angle: {angle:.2f}°",
                ]
                self.draw_text(frame_copy, info, 10, 20)

                # Update shared tracking data
                with self.tracking_data.lock:
                    self.tracking_data.status = "Tracking"
                    self.tracking_data.dx = dx
                    self.tracking_data.dy = dy
                    self.tracking_data.distance = dist
                    self.tracking_data.angle = angle
                    self.tracking_data.score = score
                    self.tracking_data.roi_height = h

                # Check if score is too low
                if (time.time() - self.tracking_start_time > 0.5) and (score < 0.30):
                    self.tracking_enabled = False
                    self.tracker = None
                    self.tracking_start_time = None
                    self.draw_text(frame_copy, ["Status: Lost"], 10, 20, color=(255, 0, 255))
                    with self.tracking_data.lock:
                        self.tracking_data.status = "Lost"
                        self.tracking_data.dx = 0
                        self.tracking_data.dy = 0
                        self.tracking_data.distance = 0.0
                        self.tracking_data.angle = 0.0
                        self.tracking_data.score = 0.0
                        self.tracking_data.roi_height = 0
            else:
                self.draw_text(frame_copy, ["Status: Lost"], 10, 20, color=(255, 0, 255))
                self.tracking_enabled = False
                self.tracker = None
                self.tracking_start_time = None
                with self.tracking_data.lock:
                    self.tracking_data.status = "Lost"
                    self.tracking_data.dx = 0
                    self.tracking_data.dy = 0
                    self.tracking_data.distance = 0.0
                    self.tracking_data.angle = 0.0
                    self.tracking_data.score = 0.0
                    self.tracking_data.roi_height = 0

            self.frame = frame_copy
        else:
            # If not tracking
            self.draw_text(self.frame, ["Status: Lost"], 10, 20, color=(200, 200, 200))
            with self.tracking_data.lock:
                self.tracking_data.status = "Lost"
                self.tracking_data.dx = 0
                self.tracking_data.dy = 0
                self.tracking_data.distance = 0.0
                self.tracking_data.angle = 0.0
                self.tracking_data.score = 0.0
                self.tracking_data.roi_height = 0

        # Show current control mode
        with self.tracking_data.lock:
            mode_text = f"Mode: {self.tracking_data.control_mode}"
            if self.tracking_data.control_mode == "Autonomous":
                if self.tracking_data.forward_enabled:
                    auto_mode_text = "Targeting"
                else:
                    auto_mode_text = "Tracking"
            else:
                auto_mode_text = ""

        lines = [mode_text]
        if auto_mode_text:
            lines.append(auto_mode_text)

        self.draw_text(self.frame, lines, 10, 140)

        # Draw the focus box for mouse
        self.draw_focused_area(self.frame, self.mouse_x, self.mouse_y, self.roi_size)
        return self.frame


def read_frames(tello, frame_queue, stop_event):
    """
    Producer: Continuously read frames from Tello, resize to standard,
    and push them into frame_queue.
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
    Consumer: Pop frames from frame_queue, run object tracking/drawing
    with VideoProcessor, then show them in an OpenCV window.
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
            processor.draw_text(processor.frame,
                                [f"FPS: {fps:.2f}"],
                                10,
                                processor.frame.shape[0] - 10)

            cv2.imshow("Tracking", processor.frame)
            if cv2.waitKey(1) & 0xFF == 27:
                print("[INFO] ESC pressed, stopping tracker.")
                break
    except Exception as e:
        print(f"[ERROR] track_frames: {e}")
    finally:
        print("[INFO] Stopping track_frames, cleaning up.")
        cv2.destroyAllWindows()
