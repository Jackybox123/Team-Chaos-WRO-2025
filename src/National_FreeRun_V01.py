#!/usr/bin/env python3

import sys
import time
import signal
from buildhat import ColorSensor, Motor, MotorPair
from sense_hat import SenseHat

# Replace this with your actual Gyro class or mock
class GyroYaw:
    def __init__(self):
        self.angle = 0
    def reset(self):
        self.angle = 0
    def get_angle(self):
        return self.angle

# ---------------------------------------------------------------------------
# ColorLineCounter - counts colored strips (non-white only)
# ---------------------------------------------------------------------------
class ColorLineCounter:
    def __init__(self):
        self.sensor = ColorSensor("D")
        self.prev_on_line = False
        self.last_count_time = 0
        self.total_count = 0
        self.debounce_seconds = 0.5

    def run(self):
        r, g, b, i = self.sensor.get_color_rgbi()

        # Define white color threshold
        is_white = (r > 230 and g > 230 and b > 230 and i > 180)
        on_line = not is_white

        current_time = time.monotonic()
        if on_line and not self.prev_on_line:
            if current_time - self.last_count_time > self.debounce_seconds:
                self.total_count += 1
                self.last_count_time = current_time
                print(f"✅ COUNTED {self.total_count} | RGBI: R={r}, G={g}, B={b}, I={i}")

        self.prev_on_line = on_line
        return self.total_count

# ---------------------------------------------------------------------------
# Dummy alignment and vehicle functions
# ---------------------------------------------------------------------------
def alignment_sequence(mode, gyro):
    print("Running alignment sequence... (stub)")
    gyro.reset()

def build_vehicle(model_path, gyro):
    class DummyVehicle:
        def __init__(self):
            self.running = True
            self.counter = ColorLineCounter()
        def start(self, rate_hz=20):
            print("Vehicle starting...")
            while self.running:
                count = self.counter.run()
                if count >= 24:
                    print("🎉 Goal reached: 24 strips counted. Stopping.")
                    self.running = False
                time.sleep(1.0 / rate_hz)
        def stop(self):
            print("Vehicle stopped.")
        def shutdown(self):
            self.stop()
    return DummyVehicle()

# ---------------------------------------------------------------------------
# Global setup
# ---------------------------------------------------------------------------
sense = SenseHat()
gyro_global = GyroYaw()

DRIVE_MODE = "user"
MODEL_PATH_DEFAULT = "model"
DRIVE_LOOP_HZ = 20

alignment_sequence(DRIVE_MODE, gyro_global)
vehicle = build_vehicle(MODEL_PATH_DEFAULT, gyro_global)

# ---------------------------------------------------------------------------
# Graceful shutdown
# ---------------------------------------------------------------------------
def shutdown(_sig=None, _frm=None):
    print("Shutting down...")
    if hasattr(vehicle, "shutdown"):
        vehicle.shutdown()
    else:
        vehicle.stop()
    sense.clear()
    sys.exit(0)

signal.signal(signal.SIGTERM, shutdown)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
try:
    vehicle.start(rate_hz=DRIVE_LOOP_HZ)
except KeyboardInterrupt:
    print("KeyboardInterrupt received.")
finally:
    shutdown()
