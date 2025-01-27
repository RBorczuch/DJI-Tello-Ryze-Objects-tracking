import keyboard
import time

class DroneController:
    def __init__(self, tello, initial_velocity=20, min_velocity=10, max_velocity=100):
        self.tello = tello
        self.velocity = initial_velocity
        self.min_velocity = min_velocity
        self.max_velocity = max_velocity
        self.last_command = (0, 0, 0, 0)  # (y_velocity, x_velocity, z_velocity, yaw_velocity)

    def control_drone(self):
        """
        Main control loop for drone movement and velocity adjustments.
        """
        try:
            self._display_controls()
            while True:
                command = self._get_velocity_command()
                if command != self.last_command:
                    self._send_velocity_command(command)
                self._handle_takeoff_and_landing()
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
        Returns the current velocity command based on user input.
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

        return y_velocity, x_velocity, z_velocity, yaw_velocity

    def _send_velocity_command(self, command):
        """
        Sends the velocity command to the drone.
        """
        self.tello.send_rc_control(*command)
        self.last_command = command
        print(f"[DEBUG] Wysłano komendę: {command}")

    def _handle_takeoff_and_landing(self):
        """
        Handles takeoff and landing commands.
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


def handle_velocity_control(tello):
    """
    Entry point to control the drone's velocity.
    """
    controller = DroneController(tello)
    controller.control_drone()
