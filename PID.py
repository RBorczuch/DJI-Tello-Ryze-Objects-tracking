import time

class PIDController:
    """
    A basic PID controller. Call compute() with the current measurement.
    It returns the control output based on the difference from the setpoint.
    """
    def __init__(
        self,
        kp=0.5,
        ki=0.0,
        kd=0.0,
        setpoint=0.0,
        sample_time=0.05,
        output_limits=(-100, 100)
    ):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.sample_time = sample_time
        self.output_limits = output_limits

        self._last_time = None
        self._integral = 0.0
        self._last_error = 0.0

    def compute(self, current_value):
        now = time.time()
        if self._last_time is None:
            self._last_time = now
            return 0

        dt = now - self._last_time
        if dt < self.sample_time:
            return 0

        error = self.setpoint - current_value
        self._integral += error * dt
        derivative = (error - self._last_error) / dt if dt > 0 else 0.0

        output = (self.kp * error) + (self.ki * self._integral) + (self.kd * derivative)

        lower, upper = self.output_limits
        output = max(lower, min(output, upper))

        self._last_error = error
        self._last_time = now

        return output
