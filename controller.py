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
        """
        tracking_data is the shared TrackingData instance (for dx, dy, status, etc.)
        """
        self.tello = tello
        self.tracking_data = tracking_data
        self.velocity = initial_velocity
        self.min_velocity = min_velocity
        self.max_velocity = max_velocity
        self.last_command = (0, 0, 0, 0)  # (y_velocity, x_velocity, z_velocity, yaw_velocity)

        # PID controllers for autonomous mode
        # Tweak gains (kp, ki, kd) and output_limits as needed
        self.yaw_pid = PIDController(kp=0.4, ki=0.0, kd=0.0,
                                     setpoint=0.0,
                                     output_limits=(-100, 100))
        self.vertical_pid = PIDController(kp=0.4, ki=0.0, kd=0.0,
                                          setpoint=0.0,
                                          output_limits=(-100, 100))

        self.error_threshold = 20  # Allowed error in px (for dx, dy)

    def control_drone(self):
        """
        Main control loop for drone movement and velocity adjustments.
        """
        try:
            self._display_controls()
            while True:
                # Check if user toggles mode (SPACE)
                self._handle_mode_switch()

                # If we are in manual mode, run the manual controls
                if self.tracking_data.control_mode == "Manual":
                    command = self._get_velocity_command()
                    if command != self.last_command:
                        self._send_velocity_command(command)
                    self._handle_takeoff_and_landing()

                # If we are in autonomous mode, run the autonomous logic
                else:
                    self._autonomous_control()

                if self._check_exit():
                    break

                time.sleep(0.05)  # Avoid high CPU usage
        except Exception as e:
            print(f"[ERROR] Control loop error: {e}")

    def _display_controls(self):
        """
        Displays the control instructions.
        """
        print("Sterowanie aktywne. Użyj klawiszy:")
        print("  w, s, a, d: ruch w poziomie (przód/tył/lewo/prawo)")
        print("  r, f: ruch w osi pionowej (góra/dół)")
        print("  q, e: rotacja (yaw)")
        print("  t: start | l: lądowanie")
        print("  <, >: zmniejszenie/zwiększenie prędkości (domyślnie 20 cm/s)")
        print("  SPACJA: przełączanie między trybem Manual i Autonomous")
        print("Naciśnij ESC, aby zakończyć sterowanie.")

    def _adjust_velocity(self):
        """
        Adjusts the velocity based on user input.
        """
        if keyboard.is_pressed('<'):
            self.velocity = max(self.min_velocity, self.velocity - 5)
            print(f"[DEBUG] Zmieniono prędkość: {self.velocity} cm/s")
            time.sleep(0.2)
        elif keyboard.is_pressed('>'):
            self.velocity = min(self.max_velocity, self.velocity + 5)
            print(f"[DEBUG] Zmieniono prędkość: {self.velocity} cm/s")
            time.sleep(0.2)

    def _get_velocity_command(self):
        """
        Returns the current velocity command based on user input (Manual mode).
        """
        self._adjust_velocity()
        y_velocity = x_velocity = z_velocity = yaw_velocity = 0

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
        """
        Sends the velocity command to the drone.
        """
        self.tello.send_rc_control(*command)
        self.last_command = command
        # print(f"[DEBUG] Wysłano komendę: {command}")

    def _handle_takeoff_and_landing(self):
        """
        Handles takeoff and landing commands (Manual mode).
        """
        if keyboard.is_pressed('t') and not self.tello.is_flying:
            print("[DEBUG] Start")
            self.tello.takeoff()
            time.sleep(2)
        elif keyboard.is_pressed('l') and self.tello.is_flying:
            print("[DEBUG] Lądowanie")
            self.tello.land()

    def _check_exit(self):
        """
        Checks if the ESC key is pressed to exit the control loop.
        """
        if keyboard.is_pressed('esc'):
            print("[DEBUG] Naciśnięto ESC. Kończenie sterowania.")
            return True
        return False

    def _handle_mode_switch(self):
        """
        Toggles between Manual and Autonomous modes on SPACE press.
        """
        if keyboard.is_pressed('space'):
            time.sleep(0.3)  # to prevent double-trigger
            with self.tracking_data.lock:
                if self.tracking_data.control_mode == "Manual":
                    self.tracking_data.control_mode = "Autonomous"
                    print("[INFO] Switched to AUTONOMOUS mode.")
                else:
                    self.tracking_data.control_mode = "Manual"
                    print("[INFO] Switched to MANUAL mode.")

    def _autonomous_control(self):
        """
        Uses the PID controllers to keep the tracked object in the center
        of the screen by rotating (yaw) and moving up/down (z_velocity).
        Switches to Manual mode if the object is lost.
        """
        with self.tracking_data.lock:
            if self.tracking_data.status == "Lost":
                # Object is lost; revert to Manual automatically
                self.tracking_data.control_mode = "Manual"
                print("[INFO] Target lost. Reverting to MANUAL mode.")
                return

            dx = self.tracking_data.dx
            dy = self.tracking_data.dy

        # If |dx| < error_threshold, don't rotate
        if abs(dx) < self.error_threshold:
            yaw_output = 0
        else:
            yaw_output = self.yaw_pid.compute(dx)
        # If |dy| < error_threshold, don't move up/down
        if abs(dy) < self.error_threshold:
            vertical_output = 0
        else:
            vertical_output = self.vertical_pid.compute(dy)

        # Convert the PID outputs (which can be negative or positive) to int
        # Tello yaw velocity is the 4th parameter (CW/CCW).
        # Tello vertical velocity is the 3rd parameter (up/down).
        yaw_velocity = int(yaw_output)
        z_velocity = int(vertical_output)

        # For this autonomous mode, we are NOT moving forward/back or left/right:
        y_velocity = 0
        x_velocity = 0

        # Send to drone
        self._send_velocity_command((y_velocity, x_velocity, z_velocity, yaw_velocity))


def handle_velocity_control(tello, tracking_data):
    """
    Entry point to control the drone's velocity (either manual or autonomous).
    """
    controller = DroneController(tello, tracking_data)
    controller.control_drone()
