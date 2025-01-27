# PID.py
import time

class PIDController:
    """
    A basic PID controller. Call compute() with the current measurement,
    and it returns the control output based on the error to the setpoint.
    """
    def __init__(self, kp=0.5, ki=0.0, kd=0.0, setpoint=0.0, sample_time=0.05,
                 output_limits=(-100, 100)):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.sample_time = sample_time
        self.output_limits = output_limits

        # Internal variables
        self._last_time = None
        self._integral = 0.0
        self._last_error = 0.0

    def compute(self, current_value):
        """
        Computes the PID output given the current value (e.g., dx or dy).
        Returns the control signal to correct the error.
        """
        now = time.time()
        if self._last_time is None:
            # First call
            self._last_time = now
            return 0

        dt = now - self._last_time
        if dt < self.sample_time:
            # Not enough time has passed
            return 0

        error = self.setpoint - current_value
        self._integral += error * dt
        derivative = (error - self._last_error) / dt if dt > 0 else 0.0

        # PID formula
        output = (self.kp * error) + (self.ki * self._integral) + (self.kd * derivative)

        # Limit output
        lower, upper = self.output_limits
        output = max(lower, min(output, upper))

        # Save state
        self._last_error = error
        self._last_time = now

        return output
