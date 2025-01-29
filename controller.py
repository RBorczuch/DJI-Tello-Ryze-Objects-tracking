# controller.py
import keyboard
import time
from PID import PIDController

class DroneController:
    def __init__(
        self,
        tello,
        tracking_data,
        initial_velocity=20,
        min_velocity=10,
        max_velocity=100
    ):
        self.tello = tello
        self.tracking_data = tracking_data
        self.velocity = initial_velocity
        self.min_velocity = min_velocity
        self.max_velocity = max_velocity
        self.last_command = (0, 0, 0, 0)

        # PID controllers for autonomous mode
        self.yaw_pid = PIDController(
            kp=0.4, ki=0.0, kd=0.0,
            setpoint=0.0,
            output_limits=(-100, 100)
        )
        self.vertical_pid = PIDController(
            kp=0.4, ki=0.0, kd=0.0,
            setpoint=0.0,
            output_limits=(-100, 100)
        )

        # --- NEW CODE ---
        # PID for forward/back based on ROI height
        # Desired bounding box height = 1/3 of 360 = 120 px
        self.forward_pid = PIDController(
            kp=0.4, ki=0.0, kd=0.0,
            setpoint=120.0,  # target ROI height
            output_limits=(-100, 100)
        )

        self.error_threshold = 20  # Allowed error in px (for dx, dy)

        # Toggle for forward/back control in autonomous mode
        self.autonomous_forward_enabled = False  # user toggles with 's'

    def control_drone(self):
        try:
            self._display_controls()
            while True:
                self._handle_mode_switch()
                self._handle_forward_switch()  # <--- check if user pressed 's'

                if self.tracking_data.control_mode == "Manual":
                    command = self._get_velocity_command()
                    if command != self.last_command:
                        self._send_velocity_command(command)
                    self._handle_takeoff_and_landing()
                else:
                    self._autonomous_control()

                if self._check_exit():
                    break

                time.sleep(0.05)
        except Exception as e:
            print(f"[ERROR] Control loop error: {e}")

    def _display_controls(self):
        print("Sterowanie aktywne. Użyj klawiszy:")
        print("  w, s, a, d: ruch w poziomie (przód/tył/lewo/prawo)")
        print("  r, f: ruch w osi pionowej (góra/dół)")
        print("  q, e: rotacja (yaw)")
        print("  t: start | l: lądowanie")
        print("  <, >: zmniejszenie/zwiększenie prędkości (domyślnie 20 cm/s)")
        print("  SPACJA: przełączanie między trybem Manual i Autonomous")
        print("  s (tylko w Autonomous): włącz/wyłącz śledzenie przód/tył po rozmiarze ROI")
        print("Naciśnij ESC, aby zakończyć sterowanie.")

    def _adjust_velocity(self):
        if keyboard.is_pressed('<'):
            self.velocity = max(self.min_velocity, self.velocity - 5)
            print(f"[DEBUG] Zmieniono prędkość: {self.velocity} cm/s")
            time.sleep(0.2)
        elif keyboard.is_pressed('>'):
            self.velocity = min(self.max_velocity, self.velocity + 5)
            print(f"[DEBUG] Zmieniono prędkość: {self.velocity} cm/s")
            time.sleep(0.2)

    def _get_velocity_command(self):
        self._adjust_velocity()
        y_velocity = x_velocity = z_velocity = yaw_velocity = 0

        # Manual controls
        if keyboard.is_pressed('w'):  # Forward
            x_velocity = self.velocity
        elif keyboard.is_pressed('s'):  # Backward
            x_velocity = -self.velocity

        if keyboard.is_pressed('a'):  # Left
            y_velocity = -self.velocity
        elif keyboard.is_pressed('d'):  # Right
            y_velocity = self.velocity

        if keyboard.is_pressed('r'):  # Up
            z_velocity = self.velocity
        elif keyboard.is_pressed('f'):  # Down
            z_velocity = -self.velocity

        if keyboard.is_pressed('q'):  # Rotate left
            yaw_velocity = -self.velocity
        elif keyboard.is_pressed('e'):  # Rotate right
            yaw_velocity = self.velocity

        return (y_velocity, x_velocity, z_velocity, yaw_velocity)

    def _send_velocity_command(self, command):
        self.tello.send_rc_control(*command)
        self.last_command = command

    def _handle_takeoff_and_landing(self):
        if keyboard.is_pressed('t') and not self.tello.is_flying:
            print("[DEBUG] Start")
            self.tello.takeoff()
            time.sleep(2)
        elif keyboard.is_pressed('l') and self.tello.is_flying:
            print("[DEBUG] Lądowanie")
            self.tello.land()

    def _check_exit(self):
        if keyboard.is_pressed('esc'):
            print("[DEBUG] Naciśnięto ESC. Kończenie sterowania.")
            return True
        return False

    def _handle_mode_switch(self):
        if keyboard.is_pressed('space'):
            time.sleep(0.3)  # prevent double-trigger
            with self.tracking_data.lock:
                if self.tracking_data.control_mode == "Manual":
                    self.tracking_data.control_mode = "Autonomous"
                    print("[INFO] Switched to AUTONOMOUS mode.")
                else:
                    self.tracking_data.control_mode = "Manual"
                    print("[INFO] Switched to MANUAL mode.")

    # --- NEW CODE ---
    def _handle_forward_switch(self):
        """
        Toggles autonomous forward/back tracking if in Autonomous mode
        when the user presses 's'.
        """
        if self.tracking_data.control_mode == "Autonomous":
            if keyboard.is_pressed('s'):
                time.sleep(0.3)  # prevent double-trigger
                self.autonomous_forward_enabled = not self.autonomous_forward_enabled
                state = "ENABLED" if self.autonomous_forward_enabled else "DISABLED"
                print(f"[INFO] Front/Back tracking {state} in Autonomous mode.")

    def _autonomous_control(self):
        with self.tracking_data.lock:
            if self.tracking_data.status == "Lost":
                # Object is lost; revert to Manual automatically
                self.tracking_data.control_mode = "Manual"
                print("[INFO] Target lost. Reverting to MANUAL mode.")
                return

            dx = self.tracking_data.dx
            dy = self.tracking_data.dy
            roi_height = self.tracking_data.roi_height

        # Yaw control
        if abs(dx) < self.error_threshold:
            yaw_output = 0
        else:
            yaw_output = self.yaw_pid.compute(dx)

        # Vertical control
        if abs(dy) < self.error_threshold:
            vertical_output = 0
        else:
            vertical_output = self.vertical_pid.compute(dy)

        yaw_velocity = int(yaw_output)
        z_velocity = int(vertical_output)

        # --- NEW CODE ---
        # Forward/back control based on ROI height if enabled
        if self.autonomous_forward_enabled:
            forward_output = self.forward_pid.compute(roi_height)
            x_velocity = int(forward_output)  # Positive => forward, Negative => backward
        else:
            x_velocity = 0

        # We don't move left/right in this example
        y_velocity = 0

        # Send the command
        self._send_velocity_command((y_velocity, x_velocity, z_velocity, yaw_velocity))

def handle_velocity_control(tello, tracking_data):
    controller = DroneController(tello, tracking_data)
    controller.control_drone()
