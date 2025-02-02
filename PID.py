import time

class PIDController:
    """
    A basic PID controller implementing proportional, integral, and derivative terms.

    Usage:
        - Create an instance with the desired PID gains, setpoint, sample_time, and output limits.
        - Call compute(current_value) repeatedly. It returns the control output each time.
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
        """
        Initialize the PID controller with coefficients and parameters.

        Args:
            kp (float): Proportional gain.
            ki (float): Integral gain.
            kd (float): Derivative gain.
            setpoint (float): The target value the PID tries to achieve.
            sample_time (float): Minimum time interval between compute() calls.
            output_limits (tuple): Min and max limits for the PID output.
        """
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
        """
        Calculate the PID control output for the given current_value.

        Args:
            current_value (float): The measured value to compare against the setpoint.

        Returns:
            float: The control output within the specified output limits.
        """
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
