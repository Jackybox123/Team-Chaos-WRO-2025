#!/usr/bin/env python3
# drive_with_gyro_guard.py
# -------------------------------------------------------------
# 1.  Sense HAT joystick lets you pick one of four model paths
#     (FCW / FCCW / OCW / OCCW).
# 2.  Donkey Car vehicle starts in *driving* (autonomous) mode
#     using that model.
# 3.  A gyroscope part integrates cumulative yaw.
#     If abs(yaw) exceeds 3 turns (1080 deg) the throttle stops
#     and the program exits cleanly.

import os
import sys
import time
import signal
from pathlib import Path
from typing import Tuple

# ------------------ Sense HAT selection -----------------------------------
from sense_hat import SenseHat
from time import sleep

# ------------------ Donkey Car imports ------------------------------------
from buildhat import Motor, ColorSensor
import donkeycar as dk
import pygame
from donkeycar.parts.transform import Lambda
from donkeycar.parts.keras import KerasInterpreter, KerasLinear

FREERUN_MODEL_PATH_CW    = "~/projectbuildhat/freeruncwmodels09/mypilot.h5"
FREERUN_MODEL_PATH_CCW   = "~/projectbuildhat/freerunccwmodels09/mypilot.h5"
OBSTACLE_MODEL_PATH_CW   = "~/projectbuildhat/obstacleruncwmodels14/mypilot.h5"
OBSTACLE_MODEL_PATH_CCW  = "~/projectbuildhat/obstacleruncwmodels09/mypilot.h5"


sense = SenseHat()
sense.set_rotation(180)
sense.low_light = True


sense.clear()

def flash(msg, seconds=0.7):
    sense.show_message(msg, scroll_speed=0.10, text_colour=[255, 255, 255])
    sleep(seconds)

sense.clear()

DRIVE_MODE = "FCW"
flash(DRIVE_MODE)
print("Move joystick, press middle to confirm.")
while True:
    for ev in sense.stick.get_events():
        if ev.action != "pressed":
            continue
        if ev.direction == "left":
            DRIVE_MODE = "FCW";  flash("FCW")
        elif ev.direction == "right":
            DRIVE_MODE = "FCCW"; flash("FCCW")
        elif ev.direction == "up":
            DRIVE_MODE = "OCW";  flash("OCW")
        elif ev.direction == "down":
            DRIVE_MODE = "OCCW"; flash("OCCW")
        elif ev.direction == "middle":
            flash(DRIVE_MODE)
            sense.clear()
            print("Selection finished.")
            break
    else:
        continue
    break

if DRIVE_MODE == "FCW":
    MODEL_PATH_DEFAULT = os.path.expanduser(FREERUN_MODEL_PATH_CW)
elif DRIVE_MODE == "FCCW":
    MODEL_PATH_DEFAULT = os.path.expanduser(FREERUN_MODEL_PATH_CCW)
elif DRIVE_MODE == "OCW":
    MODEL_PATH_DEFAULT = os.path.expanduser(OBSTACLE_MODEL_PATH_CW)
elif DRIVE_MODE == "OCCW":
    MODEL_PATH_DEFAULT = os.path.expanduser(OBSTACLE_MODEL_PATH_CCW)
else:
    sys.exit("Invalid DRIVE_MODE")

print("Model:", MODEL_PATH_DEFAULT)


# ------------------ Config ------------------------------------------------
CAP_W, CAP_H = 192, 144
CROP_W, CROP_H = 160, 120
DRIVE_LOOP_HZ = 20
MAX_SPEED_PERCENT = 35
ANGLE_OFFSET = 1.2
YAW_LIMIT_DEG = 1080.0      # 3 turns
GYRO_CAL_SEC  = 3.0
GYRO2DEG      = 57.2957795  # rad/s -> deg/s

# ------------------ Helpers ----------------------------------------------
def center_crop(img, tw=CROP_W, th=CROP_H):
    h, w = img.shape[:2]
    x0 = (w - tw) // 2
    y0 = (h - th) // 2 + 8
    return img[y0:y0 + th, x0:x0 + tw]

# ------------------ Gyro parts -------------------------------------------
from sense_hat import SenseHat as SH

from sense_hat import SenseHat

class GyroYaw:
    def __init__(self):
        self.sh = SenseHat()
        self.sh.set_imu_config(True, False, False)
        self.bias = self._calibrate_bias()
        self.yaw = 0.0
        self.offset = 0.0
        self.last_time = time.time()

    def _calibrate_bias(self):
        print("GyroYaw: calibrating bias for {} s...".format(GYRO_CAL_SEC))
        s = 0.0
        n = 0
        t0 = time.time()
        while time.time() - t0 < GYRO_CAL_SEC:
            s += self.sh.get_gyroscope_raw()["z"]
            n += 1
            time.sleep(0.01)
        bias = s / n if n else 0.0
        print("GyroYaw: bias =", bias, "rad/s")
        return bias

    def set_offset(self, val):
        self.offset = val

    def run(self):
        now = time.time()
        dt = now - self.last_time
        self.last_time = now
        z_rad_s = self.sh.get_gyroscope_raw()["z"] - self.bias
        self.yaw += z_rad_s * GYRO2DEG * dt
        return self.yaw - self.offset

class YawGuard:
    def __init__(self, limit_deg=YAW_LIMIT_DEG):
        self.limit = limit_deg
    def run(self, throttle, yaw):
        if abs(yaw) > self.limit:
            print("\nYaw limit reached ({:.1f} deg) - stopping.".format(yaw))
            raise KeyboardInterrupt
        return throttle

# ------------------ Car hardware parts -----------------------------------
class LegoSteering:
    def __init__(self, port="A", left=-50, right=50):
        self.motor = Motor(port)
        self.left, self.right = left, right
        self.prev = None
        self.motor.run_to_position(0)
        self.motor.stop()
    def run(self, angle):
        angle *= ANGLE_OFFSET
        angle = max(min(angle, 1.0), -1.0)
        raw = self.left + (angle + 1) * (self.right - self.left) / 2
        pos = int(round(raw / 10.0) * 10)
        if pos == self.prev:
            return
        self.motor.stop()
        try:
            self.motor.run_to_position(pos, speed=100, wait=False)
        except TypeError:
            self.motor.run_to_position(pos, speed=100)
        self.prev = pos
    def shutdown(self):
        self.motor.run_to_position(0)
        self.motor.stop()

class LegoThrottle:
    def __init__(self, port_l="B", port_r="C", max_speed=MAX_SPEED_PERCENT):
        self.ml, self.mr = Motor(port_l), Motor(port_r)
        self.max_speed = max_speed
        self.last = None
    def _stop(self):
        self.ml.stop()
        self.mr.stop()
    def run(self, throttle):
        speed = int(round(max(min(throttle, 1.0), -1.0) * self.max_speed / 10.0) * 10)
        if speed == self.last:
            return
        if speed == 0:
            self._stop()
        else:
            self.ml.start(speed=speed)
            self.mr.start(speed=speed)
        self.last = speed
    def shutdown(self):
        self._stop()


# ------------------ ColorLineCounter counts any non-white strip as line ---
class ColorLineCounter:
    def __init__(self):
        self.sensor = ColorSensor("D")
        self.prev_on_line = False
        self.total_count = 0

    def run(self):
        r, g, b, i = self.sensor.get_color_rgbi()

        # Define white as high RGB and Intensity values
        is_white = (r > 200 and g > 200 and b > 200 and i > 180)

        on_line = not is_white

        print(f"R={r} G={g} B={b} I={i} - on_line={on_line}")

        # Count only on rising edge (off line -> on line)
        if on_line and not self.prev_on_line:
            self.total_count += 1
            print(f"Count incremented: {self.total_count}")

        self.prev_on_line = on_line

        # Return total_count as orange count, zero for blue (compatibility)
        return self.total_count, 0


# ------------------ StopGuard stops car after total count of 24 -----------
class StopGuard:
    def __init__(self, yaw_lim=YAW_LIMIT_DEG,
                 total_needed=24, delay_sec=3.0):
        self.yaw_lim = yaw_lim
        self.total_needed = total_needed
        self.delay = delay_sec
        self.line_met_time = None

    def run(self, throttle, yaw, orange_cnt, blue_cnt):
        total_count = orange_cnt + blue_cnt
        now = time.monotonic()

        # Check if total count reached the target
        if total_count >= self.total_needed:
            if self.line_met_time is None:
                self.line_met_time = now  # start delay timer
        else:
            self.line_met_time = None  # reset if not met

        # Stop throttle after delay if conditions met
        if (abs(yaw) > self.yaw_lim or
            (self.line_met_time is not None and now - self.line_met_time >= self.delay)):
            print("\nStopGuard: Conditions met, stopping car.")
            raise KeyboardInterrupt

        return throttle

# ------------------ Console telemetry --------------------------------------
class ConsoleTelemetry:
    def __init__(self, period=0.5):
        self.period = period
        self.next_time = time.monotonic()

    def run(self, angle, throttle, yaw_deg, orange_cnt, blue_cnt):
        now = time.monotonic()
        if now >= self.next_time:
            msg = (
                "Angle {:+6.2f}  Thr {:+6.2f}  "
                "Yaw {:+8.2f} deg  "
                "Count {:2d}"
            ).format(angle, throttle, yaw_deg, orange_cnt + blue_cnt)
            print(msg)
            self.next_time = now + self.period


# ------------------ Camera helper ----------------------------------------
def add_camera(car):
    from donkeycar.parts.camera import PiCamera
    cam = PiCamera(image_w=CAP_W, image_h=CAP_H, image_d=3,
                   vflip=False, hflip=False)
    car.add(cam, outputs=["cam/raw"], threaded=True)
    car.add(Lambda(center_crop), inputs=["cam/raw"], outputs=["cam/image_array"])

# ------------------ Vehicle builder --------------------------------------
def build_vehicle(model_path, gyro):
    car = dk.vehicle.Vehicle()
    add_camera(car)

    # AI pilot
    interp = KerasInterpreter()
    pilot  = KerasLinear(interpreter=interp, input_shape=(CROP_H, CROP_W, 3))
    pilot.load(model_path)
    car.add(pilot, inputs=["cam/image_array"], outputs=["pilot/angle","pilot/throttle"])

    car.add(gyro, outputs=["yaw"])

    line_counter = ColorLineCounter()
    car.add(line_counter, outputs=["cnt/orange", "cnt/blue"])

    guard = StopGuard(total_needed=24)
    car.add(
        guard,
        inputs=["pilot/throttle", "yaw", "cnt/orange", "cnt/blue"],
        outputs=["safe/throttle"]
    )

    car.add(
        ConsoleTelemetry(),
        inputs=[
            "pilot/angle",
            "safe/throttle",
            "yaw",
            "cnt/orange",
            "cnt/blue"
        ],
        outputs=[]
    )

    car.add(LegoSteering(), inputs=["pilot/angle"])
    car.add(LegoThrottle(), inputs=["safe/throttle"])
    return car

# ---------------------------------------------------------------
gyro_global = GyroYaw()

# alignment_sequence can be skipped or implemented as needed
# alignment_sequence(DRIVE_MODE, gyro_global)

vehicle = build_vehicle(MODEL_PATH_DEFAULT, gyro_global)

def shutdown(_sig=None, _frm=None):
    print("Shutting down...")
    if hasattr(vehicle, "shutdown"):
        vehicle.shutdown()
    else:
        vehicle.stop()
    sense.clear()
    sys.exit(0)

signal.signal(signal.SIGTERM, shutdown)

try:
    vehicle.start(rate_hz=DRIVE_LOOP_HZ)
except KeyboardInterrupt:
    print("KeyboardInterrupt received.")
finally:
    shutdown()
