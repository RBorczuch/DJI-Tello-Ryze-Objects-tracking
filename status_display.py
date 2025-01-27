# status_display.py

import tkinter as tk
from threading import Thread, Lock
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.animation as animation
import numpy as np
from tracking_data import TrackingData


class StatusDisplay:
    def __init__(self, tello, tracking_data):
        self.tello = tello
        self.tracking_data = tracking_data
        self.state_data_lock = Lock()
        self.state_data = {}
        self.root = tk.Tk()
        self.root.title("Drone Status")
        self.root.geometry("2000x900")  # Adjusted height for additional plots

        # Configure root window to adjust with resizing
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=2)
        self.root.grid_columnconfigure(2, weight=2)
        self.root.grid_rowconfigure(0, weight=1)

        # Panels
        self.left_frame = self._initialize_left_panel()
        self.right_frame = self._initialize_right_panel()
        self.right_3d_frame = self._initialize_3d_panel()

        # State management
        self.sections, self.units, self.state_labels = self._define_sections_and_units()
        self.buffer = self._initialize_buffers()
        self.plot_data = self._initialize_buffers()
        self.position_data = {"x": [0], "y": [0], "z": [0]}  # Initial position at (0, 0, 0)

    def _initialize_left_panel(self):
        left_frame = tk.Frame(self.root)
        left_frame.grid(row=0, column=0, sticky="nsew")

        # Configure left_frame to adjust with resizing
        left_frame.columnconfigure(0, weight=1)
        left_frame.columnconfigure(1, weight=1)
        left_frame.rowconfigure(0, weight=1)

        # Create two columns within the left_frame
        self.left_col_frame = tk.Frame(left_frame)
        self.left_col_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.left_col_frame.columnconfigure(0, weight=1)

        self.right_col_frame = tk.Frame(left_frame)
        self.right_col_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        self.right_col_frame.columnconfigure(0, weight=1)

        return left_frame

    def _initialize_right_panel(self):
        right_frame = tk.Frame(self.root)
        right_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)

        # Configure right_frame to adjust with resizing
        right_frame.columnconfigure(0, weight=1)
        right_frame.rowconfigure(0, weight=1)

        # Initialize Matplotlib figure and axes
        self.fig, ((self.ax1, self.ax2), (self.ax3, self.ax4), (self.ax5, self.ax6)) = plt.subplots(
            3, 2, figsize=(10, 8)
        )
        self.fig.tight_layout(pad=3)

        # Create Matplotlib canvas for Tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        return right_frame

    def _initialize_3d_panel(self):
        right_3d_frame = tk.Frame(self.root)
        right_3d_frame.grid(row=0, column=2, sticky="nsew", padx=5, pady=5)

        # Configure right_3d_frame to adjust with resizing
        right_3d_frame.columnconfigure(0, weight=1)
        right_3d_frame.rowconfigure(0, weight=1)

        # Initialize Matplotlib figure and 3D axes
        self.fig_3d = plt.figure(figsize=(8, 6))
        self.ax_3d = self.fig_3d.add_subplot(111, projection="3d")

        # Configure 3D axes
        self.ax_3d.set_title("Odometry")
        self.ax_3d.set_xlabel("X (cm)")
        self.ax_3d.set_ylabel("Y (cm)")
        self.ax_3d.set_zlabel("Z (cm)")
        self.ax_3d.invert_zaxis()  # Invert Z-axis

        # Create Matplotlib canvas for Tkinter
        self.canvas_3d = FigureCanvasTkAgg(self.fig_3d, master=right_3d_frame)
        self.canvas_3d.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        return right_3d_frame

    def _define_sections_and_units(self):
        sections = {
            "Battery": ["bat"],
            "Temperature": ["templ", "temph"],
            "WiFi": ["wifi"],
            "Time": ["time"],
            "Pressure": ["baro"],
            "Tracking": ["status", "score"],
            "Object Distance": ["dx", "dy", "distance", "angle"],
            "Orientation": ["pitch", "roll", "yaw"],
            "Velocity": ["vgx", "vgy", "vgz"],
            "Height and Distance": ["tof", "h"],
            "Accelerometer": ["agx", "agy", "agz"],
        }
        units = {
            "bat": "%", "templ": "°C", "temph": "°C", "wifi": "dBm",
            "pitch": "°", "roll": "°", "yaw": "°", "vgx": "cm/s",
            "vgy": "cm/s", "vgz": "cm/s", "time": "s", "tof": "cm",
            "h": "cm", "baro": "hPa", "agx": "m/s²", "agy": "m/s²",
            "agz": "m/s²", "dx": "px", "dy": "px", "distance": "px",
            "angle": "°", "score": "",
        }
        state_labels = {}

        # Distribute sections between the two columns
        left_sections = ["Battery", "Temperature", "WiFi", "Time", "Pressure", "Tracking"]
        right_sections = ["Orientation", "Velocity", "Height and Distance", "Accelerometer", "Object Distance"]

        for section in left_sections:
            frame = tk.LabelFrame(self.left_col_frame, text=section, font=("Arial", 12))
            frame.pack(fill="both", expand=True, padx=5, pady=5)
            frame.columnconfigure(0, weight=1)

            for key in sections[section]:
                label = tk.Label(frame, text=f"{key}: -- {units.get(key, '')}", font=("Arial", 12))
                label.pack(anchor="w", padx=10, pady=2)
                state_labels[key] = label

        for section in right_sections:
            frame = tk.LabelFrame(self.right_col_frame, text=section, font=("Arial", 12))
            frame.pack(fill="both", expand=True, padx=5, pady=5)
            frame.columnconfigure(0, weight=1)

            for key in sections[section]:
                label = tk.Label(frame, text=f"{key}: -- {units.get(key, '')}", font=("Arial", 12))
                label.pack(anchor="w", padx=10, pady=2)
                state_labels[key] = label

        return sections, units, state_labels

    def _initialize_buffers(self):
        return {
            "local_time": [],
            "vgx": [], "vgy": [], "vgz": [],
            "pitch": [], "roll": [], "yaw": [], "h": [],
            "agx": [], "agy": [], "agz": [],
            "dx": [], "dy": [], "distance": [], "angle": [],
        }

    def update_wifi(self):
        while True:
            try:
                wifi = self.tello.query_wifi_signal_noise_ratio()
                with self.state_data_lock:
                    self.state_data["wifi"] = wifi
            except Exception as e:
                print(f"[ERROR] WiFi Update Failed: {e}")
            time.sleep(60)

    def update_state(self):
        while True:
            try:
                # Read all state values at once
                state = self.tello.get_current_state()

                # Synchronize acceleration and rotation data
                with self.state_data_lock:
                    # Get tracking data
                    with self.tracking_data.lock:
                        state["status"] = self.tracking_data.status
                        state["dx"] = self.tracking_data.dx
                        state["dy"] = self.tracking_data.dy
                        state["distance"] = self.tracking_data.distance
                        state["angle"] = self.tracking_data.angle
                        state["score"] = self.tracking_data.score

                    # Compensate gravity
                    self._compensate_gravity(state)

                    # Update state data
                    self.state_data.update(state)

                # Buffer data and update position
                self._buffer_data(state)
                self._update_position(state)

            except Exception as e:
                print(f"[ERROR] State Update Failed: {e}")
            time.sleep(0.1)

    def _compensate_gravity(self, state):
        """
        Compensates accelerometer readings for gravity based on orientation.
        """
        # Correct accelerations for gravity based on orientation
        # Get pitch, roll, yaw angles
        pitch = state.get("pitch", 0)
        roll = state.get("roll", 0)
        yaw = state.get("yaw", 0)
        agx = state.get("agx", 0)
        agy = state.get("agy", 0)
        agz = state.get("agz", 0)

        # Convert to radians
        pitch_rad = np.radians(pitch)
        roll_rad = np.radians(roll)
        yaw_rad = np.radians(yaw)

        # Convert accelerations to m/s²
        agx_mps2 = agx * 0.001 * 9.81
        agy_mps2 = agy * 0.001 * 9.81
        agz_mps2 = agz * 0.001 * 9.81

        # Compute rotation matrices
        R_roll = np.array([
            [1, 0, 0],
            [0, np.cos(roll_rad), -np.sin(roll_rad)],
            [0, np.sin(roll_rad), np.cos(roll_rad)],
        ])

        R_pitch = np.array([
            [np.cos(pitch_rad), 0, np.sin(pitch_rad)],
            [0, 1, 0],
            [-np.sin(pitch_rad), 0, np.cos(pitch_rad)],
        ])

        R_yaw = np.array([
            [np.cos(yaw_rad), -np.sin(yaw_rad), 0],
            [np.sin(yaw_rad), np.cos(yaw_rad), 0],
            [0, 0, 1],
        ])

        # Cumulative rotation matrix
        R = R_yaw @ R_pitch @ R_roll

        # Gravity vector in world frame
        g = 9.81  # m/s²
        gravity_world = np.array([0, 0, -g])

        # Gravity vector in drone's body frame
        gravity_body = R.T @ gravity_world

        # Adjusted accelerations
        adjusted_agx = agx_mps2 - gravity_body[0]
        adjusted_agy = agy_mps2 - gravity_body[1]
        adjusted_agz = agz_mps2 - gravity_body[2]

        # Update state with adjusted accelerations
        state["agx"] = adjusted_agx
        state["agy"] = adjusted_agy
        state["agz"] = adjusted_agz

    def update_labels(self):
        # Update the GUI labels
        with self.state_data_lock:
            state = self.state_data.copy()

        for key, value in state.items():
            if key in self.state_labels:
                unit = self.units.get(key, "")
                if isinstance(value, float):
                    self.state_labels[key].config(text=f"{key}: {value:.2f} {unit}")
                else:
                    self.state_labels[key].config(text=f"{key}: {value} {unit}")

        # Schedule next update
        self.root.after(100, self.update_labels)

    def _buffer_data(self, state):
        # Use local time for all data
        current_time = time.time() - self.start_time  # Time since program start
        self.buffer["local_time"].append(current_time)

        # List of all data keys
        data_keys = [
            "vgx", "vgy", "vgz", "pitch", "roll", "yaw", "h",
            "agx", "agy", "agz", "dx", "dy", "distance", "angle"
        ]

        # Add data to buffer
        for key in data_keys:
            if key in state:
                self.buffer[key].append(state[key])
            else:
                self.buffer[key].append(0)  # Add 0 if data is missing

    def _update_position(self, state):
        if all(k in state for k in ["vgx", "vgy", "vgz", "pitch", "roll", "yaw"]):
            # Velocities in drone's body frame (convert from cm/s to m/s)
            v_body = np.array([state["vgx"], state["vgy"], state["vgz"]]) * 0.01  # Convert to m/s

            # Get rotation angles in radians
            roll_rad = np.radians(state["roll"])
            pitch_rad = np.radians(state["pitch"])
            yaw_rad = np.radians(state["yaw"])

            # Compute rotation matrices
            R_roll = np.array([
                [1, 0, 0],
                [0, np.cos(roll_rad), -np.sin(roll_rad)],
                [0, np.sin(roll_rad), np.cos(roll_rad)],
            ])

            R_pitch = np.array([
                [np.cos(pitch_rad), 0, np.sin(pitch_rad)],
                [0, 1, 0],
                [-np.sin(pitch_rad), 0, np.cos(pitch_rad)],
            ])

            R_yaw = np.array([
                [np.cos(yaw_rad), -np.sin(yaw_rad), 0],
                [np.sin(yaw_rad), np.cos(yaw_rad), 0],
                [0, 0, 1],
            ])

            # Cumulative rotation matrix
            R = R_yaw @ R_pitch @ R_roll

            # Transform velocities to world frame
            v_world = R @ v_body  # in m/s

            # Time step
            dt = 0.05  # Time between state updates in seconds

            # Update position (positions in cm)
            x_new = self.position_data["x"][-1] + v_world[0] * dt * 100  # Convert from m to cm
            y_new = self.position_data["y"][-1] + v_world[1] * dt * 100
            z_new = self.position_data["z"][-1] + v_world[2] * dt * 100

            self.position_data["x"].append(x_new)
            self.position_data["y"].append(y_new)
            self.position_data["z"].append(z_new)

    def update_plots(self, _):
        # Update 2D plots
        self._update_2d_plots()

        # Update 3D plot
        self._update_3d_plot()

    def _update_2d_plots(self):
        if not self.buffer["local_time"]:
            return

        # Update plot data
        self.plot_data["local_time"].extend(self.buffer["local_time"])
        data_keys = [
            "vgx", "vgy", "vgz", "pitch", "roll", "yaw", "h",
            "agx", "agy", "agz", "dx", "dy", "distance", "angle"
        ]
        for key in data_keys:
            self.plot_data[key].extend(self.buffer[key])

        # Clear buffers
        for key in self.buffer:
            self.buffer[key] = []

        # Keep only the last 10 seconds of data
        latest_time = self.plot_data["local_time"][-1]
        time_window = latest_time - 10
        indices = [i for i, t in enumerate(self.plot_data["local_time"]) if t >= time_window]

        # Trim data to the last 10 seconds
        self.plot_data["local_time"] = [self.plot_data["local_time"][i] for i in indices]
        for key in data_keys:
            self.plot_data[key] = [self.plot_data[key][i] for i in indices]

        # Update individual 2D plots

        # Velocity Over Time
        self.ax1.clear()
        self.ax1.plot(self.plot_data["local_time"], self.plot_data["vgx"], label="vgx")
        self.ax1.plot(self.plot_data["local_time"], self.plot_data["vgy"], label="vgy")
        self.ax1.plot(self.plot_data["local_time"], self.plot_data["vgz"], label="vgz")
        self.ax1.set_title("Velocity Over Time")
        self.ax1.legend()

        # Orientation Over Time
        self.ax2.clear()
        self.ax2.plot(self.plot_data["local_time"], self.plot_data["pitch"], label="pitch")
        self.ax2.plot(self.plot_data["local_time"], self.plot_data["roll"], label="roll")
        self.ax2.plot(self.plot_data["local_time"], self.plot_data["yaw"], label="yaw")
        self.ax2.set_title("Orientation Over Time")
        self.ax2.legend()

        # Height Over Time
        self.ax3.clear()
        self.ax3.plot(self.plot_data["local_time"], self.plot_data["h"], label="h")
        self.ax3.set_title("Height Over Time")
        self.ax3.legend()

        # Acceleration Over Time
        self.ax4.clear()
        self.ax4.plot(self.plot_data["local_time"], self.plot_data["agx"], label="agx")
        self.ax4.plot(self.plot_data["local_time"], self.plot_data["agy"], label="agy")
        self.ax4.plot(self.plot_data["local_time"], self.plot_data["agz"], label="agz")
        self.ax4.set_title("Acceleration Over Time")
        self.ax4.legend()

        # Object Distance Over Time (dx, dy, distance)
        self.ax5.clear()
        self.ax5.plot(self.plot_data["local_time"], self.plot_data["dx"], label="dx")
        self.ax5.plot(self.plot_data["local_time"], self.plot_data["dy"], label="dy")
        self.ax5.plot(self.plot_data["local_time"], self.plot_data["distance"], label="distance")
        self.ax5.set_title("Object Distance Over Time")
        self.ax5.legend()

        # Angle Over Time
        self.ax6.clear()
        self.ax6.plot(self.plot_data["local_time"], self.plot_data["angle"], label="angle")
        self.ax6.set_title("Angle Over Time")
        self.ax6.legend()

        self.canvas.draw()

    def _update_3d_plot(self):
        # Clear 3D plot
        self.ax_3d.clear()

        # Plot trajectory
        self.ax_3d.plot(
            self.position_data["x"], self.position_data["y"], self.position_data["z"], label="Trajectory"
        )

        # Get current position
        x, y, z = self.position_data["x"][-1], self.position_data["y"][-1], self.position_data["z"][-1]

        # Add orientation vectors
        if len(self.plot_data["pitch"]) > 0 and len(self.plot_data["roll"]) > 0 and len(self.plot_data["yaw"]) > 0:
            pitch, roll, yaw = (
                np.radians(self.plot_data["pitch"][-1]),
                np.radians(self.plot_data["roll"][-1]),
                np.radians(self.plot_data["yaw"][-1]),
            )
            # Create rotation matrix
            R = self._get_rotation_matrix(roll, pitch, yaw)
            # Define unit vectors in body frame
            body_x = np.array([1, 0, 0])
            body_y = np.array([0, 1, 0])
            body_z = np.array([0, 0, 1])
            # Rotate vectors to world frame
            world_x = R @ body_x
            world_y = R @ body_y
            world_z = R @ body_z
            # Plot orientation vectors with reduced length
            self.ax_3d.quiver(
                x, y, z, world_x[0], world_x[1], world_x[2], color="r", length=1, normalize=True, label="X-Axis"
            )
            self.ax_3d.quiver(
                x, y, z, world_y[0], world_y[1], world_y[2], color="g", length=1, normalize=True, label="Y-Axis"
            )
            self.ax_3d.quiver(
                x, y, z, world_z[0], world_z[1], world_z[2], color="b", length=1, normalize=True, label="Z-Axis"
            )

        # Configure 3D plot
        self.ax_3d.set_title("Odometry")
        self.ax_3d.set_xlabel("X (cm)")
        self.ax_3d.set_ylabel("Y (cm)")
        self.ax_3d.set_zlabel("Z (cm)")
        self.ax_3d.invert_zaxis()  # Invert Z-axis
        self.ax_3d.legend()
        self.canvas_3d.draw()

    def _get_rotation_matrix(self, roll, pitch, yaw):
        R_roll = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)],
        ])

        R_pitch = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)],
        ])

        R_yaw = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1],
        ])

        return R_yaw @ R_pitch @ R_roll

    def run(self):
        # Initialize start time
        self.start_time = time.time()

        # Start threads
        Thread(target=self.update_state, daemon=True).start()
        Thread(target=self.update_wifi, daemon=True).start()

        # Schedule GUI label updates
        self.update_labels()

        # Start the Matplotlib animation in the main thread
        ani = animation.FuncAnimation(self.fig, self.update_plots, interval=200, cache_frame_data=False)

        self.root.mainloop()
