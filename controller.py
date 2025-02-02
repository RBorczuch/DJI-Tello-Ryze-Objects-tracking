import keyboard
import time
from PID import PIDController

# Timing and control constants
LOOP_SLEEP = 0.05        # 20Hz control loop (50ms)
VELOCITY_SLEEP = 0.2     # Debounce period for velocity adjustments
MODE_SWITCH_SLEEP = 0.3  # Debounce period for mode switches
TAKEOFF_LANDING_WAIT = 1 # Safety delay for takeoff/landing
ERROR_THRESHOLD = 20     # Pixel threshold for PID activation
DEAD_ZONE = 5            # Minimum error to respond to

# PID configuration
PID_CONFIG = {
    'yaw': {'kp': 0.1, 'ki': 0.01, 'kd': 0.0},
    'vertical': {'kp': 0.2, 'ki': 0.01, 'kd': 0.0},
    'forward': {'kp': 0.4, 'ki': 0.0, 'kd': 0.0, 'setpoint': 120}
}

# Velocity configuration
VELOCITY_CONFIG = {
    'initial': 20,
    'min': 10,
    'max': 100,
    'step': 5
}

class DroneController:
    """Optimized drone controller with enhanced PID handling and input management"""
    
    def __init__(self, tello, tracking_data):
        self.tello = tello
        self.tracking_data = tracking_data
        
        # Initialize PID controllers
        self.pids = {
            'yaw': PIDController(**PID_CONFIG['yaw']),
            'vertical': PIDController(**PID_CONFIG['vertical']),
            'forward': PIDController(**PID_CONFIG['forward'])
        }
        
        # State variables
        self.velocity = VELOCITY_CONFIG['initial']
        self.last_command = (0, 0, 0, 0)
        self.last_mode_switch = 0
        self.last_velocity_adjust = 0
        self.loop_time = time.time()

    def control_drone(self):
        """Main control loop with precise timing and error handling"""
        try:
            self._display_controls()
            while True:
                # Maintain consistent loop timing
                elapsed = time.time() - self.loop_time
                sleep_time = max(LOOP_SLEEP - elapsed, 0)
                time.sleep(sleep_time)
                self.loop_time = time.time()

                self._handle_mode_switches()
                self._handle_velocity_adjustment()

                if self.tracking_data.control_mode == "Manual":
                    self._process_manual_control()
                    self._handle_takeoff_and_landing()
                else:
                    self._process_autonomous_control()

                if self._check_exit():
                    break

        except Exception as e:
            print(f"[ERROR] Control loop: {e}")
        finally:
            self._send_command((0, 0, 0, 0))  # Stop all motion

    def _process_manual_control(self):
        """Efficient manual control input processing"""
        command = [0, 0, 0, 0]  # y, x, z, yaw
        
        # Key bindings: (command_index, multiplier)
        key_bindings = {
            'w': (1, 1),    # Forward
            's': (1, -1),   # Backward
            'a': (0, -1),   # Left
            'd': (0, 1),    # Right
            'r': (2, 1),    # Up
            'f': (2, -1),   # Down
            'q': (3, -1),   # Yaw left
            'e': (3, 1)     # Yaw right
        }

        for key, (idx, mult) in key_bindings.items():
            if keyboard.is_pressed(key):
                command[idx] = mult * self.velocity

        self._send_command(tuple(command))

    def _process_autonomous_control(self):
        """Autonomous control with dead zone and PID optimization"""
        with self.tracking_data.lock:
            if self.tracking_data.status == "Lost":
                self._switch_to_manual()
                return

            dx = self.tracking_data.dx
            dy = self.tracking_data.dy
            roi_height = self.tracking_data.roi_height
            forward_enabled = self.tracking_data.forward_enabled

        # Calculate control outputs
        x_vel = self._calculate_pid_output('forward', roi_height) if forward_enabled else 0
        yaw_vel = self._calculate_pid_output('yaw', dx, ERROR_THRESHOLD)
        z_vel = self._calculate_pid_output('vertical', dy, ERROR_THRESHOLD)

        self._send_command((0, x_vel, z_vel, yaw_vel))

    def _calculate_pid_output(self, pid_key, error, threshold=0):
        """Calculate PID output with dead zone and conditional reset"""
        if abs(error) < max(threshold, DEAD_ZONE):
            self.pids[pid_key].reset()
            return 0
        return int(self.pids[pid_key].compute(error))

    def _send_command(self, command):
        """Send command only if changed from previous"""
        if command != self.last_command:
            self.tello.send_rc_control(*command)
            self.last_command = command

    def _handle_velocity_adjustment(self):
        """Debounced velocity adjustment handler"""
        if time.time() - self.last_velocity_adjust < VELOCITY_SLEEP:
            return

        if keyboard.is_pressed('<'):
            self.velocity = max(VELOCITY_CONFIG['min'], 
                              self.velocity - VELOCITY_CONFIG['step'])
            print(f"Speed: {self.velocity} cm/s")
            self.last_velocity_adjust = time.time()
        elif keyboard.is_pressed('>'):
            self.velocity = min(VELOCITY_CONFIG['max'], 
                              self.velocity + VELOCITY_CONFIG['step'])
            print(f"Speed: {self.velocity} cm/s")
            self.last_velocity_adjust = time.time()

    def _handle_mode_switches(self):
        """Debounced mode and feature switching"""
        now = time.time()
        if now - self.last_mode_switch < MODE_SWITCH_SLEEP:
            return

        if keyboard.is_pressed('space'):
            new_mode = "Autonomous" if self.tracking_data.control_mode == "Manual" else "Manual"
            with self.tracking_data.lock:
                self.tracking_data.control_mode = new_mode
            print(f"Mode: {new_mode}")
            self.last_mode_switch = now

        if keyboard.is_pressed('s') and self.tracking_data.control_mode == "Autonomous":
            with self.tracking_data.lock:
                new_state = not self.tracking_data.forward_enabled
                self.tracking_data.forward_enabled = new_state
            print(f"Forward targeting: {'ON' if new_state else 'OFF'}")
            self.last_mode_switch = now

    def _switch_to_manual(self):
        """Handle automatic fallback to manual mode"""
        with self.tracking_data.lock:
            self.tracking_data.control_mode = "Manual"
        print("Target lost - Switching to Manual")
        self._send_command((0, 0, 0, 0))

    def _handle_takeoff_and_landing(self):
        """Handle takeoff/landing commands with safety checks"""
        if keyboard.is_pressed('t') and not self.tello.is_flying:
            print("Takeoff initiated")
            self.tello.takeoff()
            time.sleep(TAKEOFF_LANDING_WAIT)
        elif keyboard.is_pressed('l') and self.tello.is_flying:
            print("Landing initiated")
            self.tello.land()

    def _check_exit(self):
        """Check for exit condition"""
        return keyboard.is_pressed('esc')

    def _display_controls(self):
        """Display control scheme"""
        controls = [
            "Controls:",
            "  W/S: Forward/Backward",
            "  A/D: Left/Right",
            "  R/F: Up/Down",
            "  Q/E: Rotate Left/Right",
            "  T/L: Takeoff/Land",
            "  </>: Decrease/Increase Speed",
            "  SPACE: Toggle Manual/Autonomous",
            "  S: Toggle Forward Targeting (Auto)",
            "  ESC: Quit"
        ]
        print("\n".join(controls))


def handle_velocity_control(tello, tracking_data):
    """Entry point for velocity control system"""
    controller = DroneController(tello, tracking_data)
    controller.control_drone()