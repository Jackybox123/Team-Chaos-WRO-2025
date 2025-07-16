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
import colorsys

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

# ---------------------------------------------------------------------------
# GyroYaw - uses Z-axis gyro integration, provides offset reset
# ---------------------------------------------------------------------------
class GyroYaw:
    """
    * On startup averages bias for GYRO_CAL_SEC seconds
    * Cumulatively integrates all yaw increments
    * set_offset(val) resets current yaw to zero point
    * run() returns yaw - offset
    """
    def __init__(self):
        self.sh = SenseHat()
        # Enable gyro only, disable accel/mag to reduce noise
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
        """Set current accumulated yaw as new zero point"""
        self.offset = val

    def run(self):
        now = time.time()
        dt = now - self.last_time
        self.last_time = now

        # Integrate angular velocity (minus bias) -> cumulative yaw in degrees
        z_rad_s = self.sh.get_gyroscope_raw()["z"] - self.bias
        self.yaw += z_rad_s * GYRO2DEG * dt

        # Return adjusted yaw (offset removed)
        return self.yaw - self.offset

class YawGuard:
    """Stops throttle and raises KeyboardInterrupt if |yaw| > limit."""
    def __init__(self, limit_deg=YAW_LIMIT_DEG):
        self.limit = limit_deg
    def run(self, throttle, yaw):
        if abs(yaw) > self.limit:
            print("\nYaw limit reached ({:.1f} deg) - stopping.".format(yaw))
            raise KeyboardInterrupt
        return throttle  # safe throttle passes through

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


# ---------------------------------------------------------------------------
# ColorLineCounter - improved BuildHat RGBI color detection with HSV logic
# ---------------------------------------------------------------------------
class ColorLineCounter:
    """
    Uses BuildHat ColorSensor RGBA to detect orange and blue strips.
    Converts RGB to HSV internally for better classification.
    Debounces to avoid multiple counts on one strip.
    """

    def __init__(self):
        self.sensor = ColorSensor("D")
        self.prev_on_line = False
        self.orange_count = 0
        self.blue_count = 0
        self.last_detect_time = 0
        self.debounce_time = 0.5  # seconds between counts to avoid double counting

    def rgb_to_hsv(self, r, g, b):
        # Normalize RGB 0-255 to 0-1
        r_n, g_n, b_n = r / 255.0, g / 255.0, b / 255.0
        return colorsys.rgb_to_hsv(r_n, g_n, b_n)  # returns (h,s,v) with h in [0,1]

    def _classify(self, r, g, b, i):
        """
        Classify color by HSV ranges:
        - orange hue approx 20-40 deg (normalized 0.055 - 0.11)
        - blue hue approx 190-240 deg (normalized 0.53 - 0.67)
        Use saturation and value thresholds to filter out low saturation (gray/white).
        """

        h, s, v = self.rgb_to_hsv(r, g, b)

        # Require some minimum saturation and brightness
        if s < 0.4 or v < 0.3:
            return None

        # Orange hue range (approx 20° to 40°)
        if 0.055 <= h <= 0.11:
            return "orange"

        # Blue hue range (approx 190° to 240°)
        if 0.53 <= h <= 0.67:
            return "blue"

        return None

    def run(self):
        r, g, b, i = self.sensor.get_color_rgbi()

        kind = self._classify(r, g, b, i)

        now = time.monotonic()
        counted = False

        if kind and not self.prev_on_line:
            # Debounce: only count if sufficient time elapsed since last count
            if now - self.last_detect_time > self.debounce_time:
                if kind == "orange":
                    self.orange_count += 1
                elif kind == "blue":
                    self.blue_count += 1
                self.last_detect_time = now
                counted = True

        self.prev_on_line = kind is not None

        # Debug print
        print(f"Detected: {kind if kind else 'none'}  (R={r} G={g} B={b} I={i})"
              + (f" -- Counted {kind}" if counted else ""))

        return self.orange_count, self.blue_count


# ---------------------------------------------------------------------------
# StopGuard - stops throttle after yaw and color counts conditions met + delay
# ---------------------------------------------------------------------------
class StopGuard:
    def __init__(self, yaw_lim=YAW_LIMIT_DEG,
                 need_orange=12, need_blue=12, delay_sec=3.0):
        self.yaw_lim = yaw_lim
        self.need_o  = need_orange
        self.need_b  = need_blue
        self.delay   = delay_sec
        self.line_met_time = None

    def run(self, throttle, yaw, orange_cnt, blue_cnt):
        now = time.monotonic()

        # Already met line count conditions
        if orange_cnt >= self.need_o and blue_cnt >= self.need_b:
            if self.line_met_time is None:
                self.line_met_time = now          # First time met, start timer
        else:
            self.line_met_time = None             # Reset timer if conditions lost

        # Both conditions met and delay elapsed
        if (abs(yaw) > self.yaw_lim and
            self.line_met_time is not None and
            now - self.line_met_time >= self.delay):
            print("\nStopGuard: Conditions met, stopping.")
            raise KeyboardInterrupt

        return throttle  # normal output


# ---------------------------------------------------------------------------
# ConsoleTelemetry - prints status line every period seconds
# ---------------------------------------------------------------------------
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
                "Orange {:2d}  Blue {:2d}"
            ).format(angle, throttle, yaw_deg, orange_cnt, blue_cnt)
            print(msg)
            self.next_time = now + self.period
        # no output


def alignment_sequence(drive_mode, gyro):
    """
    OCW / OCCW alignment maneuver.
    Prints real-time yaw angle at the terminal.
    """
    if drive_mode not in ("OCW", "OCCW"):
        return

    steer = LegoSteering()
    throttle = LegoThrottle(max_speed=40)

    # Simple print function: overwrite the same line
    def print_yaw():
        sys.stdout.write("\rYaw:{:+8.2f} deg".format(gyro.yaw))
        sys.stdout.flush()

    # Wait until condition true, while updating yaw printout
    def wait_until(cond):
        while True:
            yaw_val = gyro.run()   # update yaw
            print_yaw()
            if cond(yaw_val):
                break
            time.sleep(0.01)
        print()  # newline

    def settle(duration, step=0.02):
        end_time = time.monotonic() + duration
        while time.monotonic() < end_time:
            gyro.run()
            print_yaw()
            time.sleep(step)

    # Turn / motion lambdas (keep original logic)
    turn_r = lambda: steer.run(+1.0)
    turn_l = lambda: steer.run(-1.0)
    stop_m = lambda: throttle.run(0.0)
    fwd    = lambda: throttle.run(+0.5)
    back   = lambda: throttle.run(-0.5)

    try:
        if drive_mode == "OCW":
            turn_r();  settle(0.4)
            fwd();  wait_until(lambda y: y >  20);  stop_m()

            turn_l();  settle(0.4)
            back(); wait_until(lambda y: y >  35);  stop_m()

            turn_r();  settle(0.4)
            fwd();  wait_until(lambda y: y >  50);  stop_m()

            turn_l();  settle(0.4)
            fwd();  wait_until(lambda y: y <  10);  stop_m()

        else:  # OCCW
            turn_l();  settle(0.4)
            fwd();  wait_until(lambda y: y < -20);  stop_m()

            turn_r();  settle(0.4)
            back(); wait_until(lambda y: y < -35);  stop_m()

            turn_l();  settle(0.4)
            fwd();  wait_until(lambda y: y < -50);  stop_m()

            turn_r();  settle(0.4)
            fwd();  wait_until(lambda y: y > -10);  stop_m()

        print("Car will stop and gyro settle 3 s ...")
        stop_m()
        steer.shutdown()
        throttle.shutdown()
        t0 = time.time()
        while time.time() - t0 < 3.0:
            yaw_now = gyro_global.run()
            sys.stdout.write("\rIdle yaw:{:+.2f} ".format(yaw_now))
            sys.stdout.flush()
            time.sleep(0.02)
        print()

    except KeyboardInterrupt:
        pass
    finally:
        stop_m()
        steer.shutdown()
        throttle.shutdown()


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

    # Gyro yaw part
    car.add(gyro, outputs=["yaw"])

    # Color sensor line counter
    line_counter = ColorLineCounter()
    car.add(line_counter, outputs=["cnt/orange", "cnt/blue"])

    # Composite guard replaces YawGuard: stops if yaw and color counts meet conditions
    guard = StopGuard()
    car.add(
        guard,
        inputs=["pilot/throttle", "yaw", "cnt/orange", "cnt/blue"],
        outputs=["safe/throttle"]
    )

    # Console telemetry printout
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

    # Actuators
    car.add(LegoSteering(), inputs=["pilot/angle"])
    car.add(LegoThrottle(), inputs=["safe/throttle"])

    return car


# ---------------------------------------------------------------
# one global gyro instance: zero here, then reused everywhere
# ---------------------------------------------------------------
gyro_global = GyroYaw()

alignment_sequence(DRIVE_MODE, gyro_global)

# ------------------ Main --------------------------------------------------

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
