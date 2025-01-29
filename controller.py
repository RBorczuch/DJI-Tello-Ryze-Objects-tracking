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

        # PID controllers for yaw & vertical movement
        self.yaw_pid = PIDController(kp=0.4, ki=0.0, kd=0.0,
                                     setpoint=0.0, output_limits=(-100, 100))
        self.vertical_pid = PIDController(kp=0.4, ki=0.0, kd=0.0,
                                          setpoint=0.0, output_limits=(-100, 100))

        # PID for forward/back movement (target ROI height is 120 px)
        self.forward_pid = PIDController(kp=0.4, ki=0.0, kd=0.0,
                                         setpoint=120.0,
                                         output_limits=(-100, 100))

        self.error_threshold = 20
        self.autonomous_forward_enabled = False

    def control_drone(self):
        try:
            self._display_controls()
            while True:
                self._handle_mode_switch()
                self._handle_forward_switch()

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
            print(f"[ERROR] Control loop: {e}")

    def _display_controls(self):
        print("Controls:")
        print("  w/s: forward/backward")
        print("  a/d: left/right")
        print("  r/f: up/down")
        print("  q/e: rotate left/right")
        print("  t: takeoff  |  l: land")
        print("  <: decrease speed  |  >: increase speed")
        print("  SPACE: switch between Manual and Autonomous")
        print("  s (in Autonomous): enable/disable front/back targeting")
        print("  ESC: quit")

    def _adjust_velocity(self):
        if keyboard.is_pressed('<'):
            self.velocity = max(self.min_velocity, self.velocity - 5)
            print(f"[INFO] Speed set to: {self.velocity} cm/s")
            time.sleep(0.2)
        elif keyboard.is_pressed('>'):
            self.velocity = min(self.max_velocity, self.velocity + 5)
            print(f"[INFO] Speed set to: {self.velocity} cm/s")
            time.sleep(0.2)

    def _get_velocity_command(self):
        self._adjust_velocity()
        y_velocity = x_velocity = z_velocity = yaw_velocity = 0

        if keyboard.is_pressed('w'):
            x_velocity = self.velocity
        elif keyboard.is_pressed('s'):
            x_velocity = -self.velocity

        if keyboard.is_pressed('a'):
            y_velocity = -self.velocity
        elif keyboard.is_pressed('d'):
            y_velocity = self.velocity

        if keyboard.is_pressed('r'):
            z_velocity = self.velocity
        elif keyboard.is_pressed('f'):
            z_velocity = -self.velocity

        if keyboard.is_pressed('q'):
            yaw_velocity = -self.velocity
        elif keyboard.is_pressed('e'):
            yaw_velocity = self.velocity

        return (y_velocity, x_velocity, z_velocity, yaw_velocity)

    def _send_velocity_command(self, command):
        self.tello.send_rc_control(*command)
        self.last_command = command

    def _handle_takeoff_and_landing(self):
        if keyboard.is_pressed('t') and not self.tello.is_flying:
            print("[INFO] Takeoff")
            self.tello.takeoff()
            time.sleep(2)
        elif keyboard.is_pressed('l') and self.tello.is_flying:
            print("[INFO] Landing")
            self.tello.land()

    def _check_exit(self):
        if keyboard.is_pressed('esc'):
            print("[INFO] Exiting control loop.")
            return True
        return False

    def _handle_mode_switch(self):
        if keyboard.is_pressed('space'):
            time.sleep(0.3)
            with self.tracking_data.lock:
                if self.tracking_data.control_mode == "Manual":
                    self.tracking_data.control_mode = "Autonomous"
                    print("[INFO] Switched to Autonomous mode.")
                else:
                    self.tracking_data.control_mode = "Manual"
                    print("[INFO] Switched to Manual mode.")

    def _handle_forward_switch(self):
        with self.tracking_data.lock:
            if self.tracking_data.control_mode == "Autonomous":
                if keyboard.is_pressed('s'):
                    time.sleep(0.3)
                    self.autonomous_forward_enabled = not self.autonomous_forward_enabled
                    self.tracking_data.forward_enabled = self.autonomous_forward_enabled
                    state_str = "ENABLED" if self.autonomous_forward_enabled else "DISABLED"
                    print(f"[INFO] Front/back targeting {state_str}.")

    def _autonomous_control(self):
        with self.tracking_data.lock:
            if self.tracking_data.status == "Lost":
                self.tracking_data.control_mode = "Manual"
                print("[INFO] Target lost. Switching to Manual mode.")
                return

            dx = self.tracking_data.dx
            dy = self.tracking_data.dy
            roi_height = self.tracking_data.roi_height
            forward_on = self.tracking_data.forward_enabled

        if abs(dx) < self.error_threshold:
            yaw_output = 0
        else:
            yaw_output = self.yaw_pid.compute(dx)

        if abs(dy) < self.error_threshold:
            vertical_output = 0
        else:
            vertical_output = self.vertical_pid.compute(dy)

        if forward_on:
            forward_output = self.forward_pid.compute(roi_height)
            x_velocity = int(forward_output)
        else:
            x_velocity = 0

        yaw_velocity = int(yaw_output)
        z_velocity = int(vertical_output)
        y_velocity = 0

        self._send_velocity_command((y_velocity, x_velocity, z_velocity, yaw_velocity))


def handle_velocity_control(tello, tracking_data):
    controller = DroneController(tello, tracking_data)
    controller.control_drone()
