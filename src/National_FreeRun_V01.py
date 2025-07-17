#!/usr/bin/env python3
import os
import sys
import time
import signal
from sense_hat import SenseHat
from buildhat import Motor, ColorSensor
import donkeycar as dk
from donkeycar.parts.keras import KerasInterpreter, KerasLinear
from donkeycar.parts.transform import Lambda
from donkeycar.parts.camera import PiCamera

# ----- CONFIG -----
MODEL_PATH = os.path.expanduser("~/projectbuildhat/obstacleruncwmodels14/mypilot.h5")
CAP_W, CAP_H = 192, 144
CROP_W, CROP_H = 160, 120
MAX_SPEED = 35
DRIVE_LOOP_HZ = 20
ANGLE_OFFSET = 1.2
YAW_LIMIT = 1080
LINE_TARGET = 24
STOP_DELAY = 3.0
GYRO_CAL_SEC = 3.0
GYRO2DEG = 57.2958

# ----- CAMERA -----
def center_crop(img, tw=CROP_W, th=CROP_H):
    h, w = img.shape[:2]
    x0 = (w - tw) // 2
    y0 = (h - th) // 2 + 8
    return img[y0:y0+th, x0:x0+tw]

def add_camera(car):
    cam = PiCamera(image_w=CAP_W, image_h=CAP_H, image_d=3, vflip=False, hflip=False)
    car.add(cam, outputs=["cam/raw"], threaded=True)
    car.add(Lambda(center_crop), inputs=["cam/raw"], outputs=["cam/image_array"])

# ----- GYRO -----
class GyroYaw:
    def __init__(self):
        self.sh = SenseHat()
        self.sh.set_imu_config(True, False, False)
        self.bias = self._calibrate()
        self.yaw = 0.0
        self.last = time.time()

    def _calibrate(self):
        print("GyroYaw: calibrating...")
        total, count = 0, 0
        t0 = time.time()
        while time.time() - t0 < GYRO_CAL_SEC:
            total += self.sh.get_gyroscope_raw()['z']
            count += 1
            time.sleep(0.01)
        return total / count if count else 0

    def run(self):
        now = time.time()
        dt = now - self.last
        self.last = now
        z = self.sh.get_gyroscope_raw()['z'] - self.bias
        self.yaw += z * GYRO2DEG * dt
        return self.yaw

# ----- PARTS -----
class LegoSteering:
    def __init__(self, port="A", left=-50, right=50):
        self.motor = Motor(port)
        self.left, self.right = left, right
        self.prev = None
        self.motor.run_to_position(0)

    def run(self, angle):
        angle *= ANGLE_OFFSET
        angle = max(min(angle, 1), -1)
        pos = int(round((self.left + (angle + 1) * (self.right - self.left) / 2) / 10) * 10)
        if pos != self.prev:
            self.motor.run_to_position(pos, speed=100, wait=False)
            self.prev = pos

    def shutdown(self):
        self.motor.run_to_position(0)
        self.motor.stop()

class LegoThrottle:
    def __init__(self, port_l="B", port_r="C", max_speed=MAX_SPEED):
        self.ml, self.mr = Motor(port_l), Motor(port_r)
        self.max_speed = max_speed
        self.last = 0

    def run(self, throttle):
        speed = int(round(throttle * self.max_speed / 10) * 10)
        if speed != self.last:
            if speed == 0:
                self.ml.stop(); self.mr.stop()
            else:
                self.ml.start(speed=speed)
                self.mr.start(speed=speed)
            self.last = speed

    def shutdown(self):
        self.ml.stop(); self.mr.stop()

class ColorLineCounter:
    def __init__(self):
        self.sensor = ColorSensor("D")
        self.prev = False
        self.count = 0

    def run(self):
        r, g, b, i = self.sensor.get_color_rgbi()
        is_white = (r > 200 and g > 200 and b > 200 and i > 180)
        on_line = not is_white
        if on_line and not self.prev:
            self.count += 1
            print(f"Line detected — Count: {self.count}")
        self.prev = on_line
        return self.count, 0  # return as (orange, blue)

class StopGuard:
    def __init__(self, yaw_limit=YAW_LIMIT, total_needed=LINE_TARGET, delay_sec=STOP_DELAY):
        self.limit = yaw_limit
        self.total = total_needed
        self.delay = delay_sec
        self.time_met = None

    def run(self, throttle, yaw, orange, blue):
        total = orange + blue
        now = time.monotonic()
        if total >= self.total:
            if self.time_met is None:
                self.time_met = now
        else:
            self.time_met = None

        if abs(yaw) > self.limit or (self.time_met and now - self.time_met > self.delay):
            print("Stop condition met.")
            raise KeyboardInterrupt
        return throttle

class ConsoleTelemetry:
    def __init__(self, interval=0.5):
        self.interval = interval
        self.next = time.monotonic()

    def run(self, angle, throttle, yaw, orange, blue):
        now = time.monotonic()
        if now >= self.next:
            print(f"Angle: {angle:+.2f} Throttle: {throttle:+.2f} "
                  f"Yaw: {yaw:+.2f}°  Count: {orange + blue}")
            self.next = now + self.interval

# ----- VEHICLE -----
def build_vehicle():
    car = dk.vehicle.Vehicle()
    add_camera(car)

    # Pilot
    interp = KerasInterpreter()
    pilot = KerasLinear(interpreter=interp, input_shape=(CROP_H, CROP_W, 3))
    pilot.load(MODEL_PATH)
    car.add(pilot, inputs=["cam/image_array"], outputs=["angle", "throttle"])

    # Gyro
    gyro = GyroYaw()
    car.add(gyro, outputs=["yaw"])

    # Line counter
    counter = ColorLineCounter()
    car.add(counter, outputs=["orange", "blue"])

    # Guard
    guard = StopGuard()
    car.add(guard, inputs=["throttle", "yaw", "orange", "blue"], outputs=["safe_throttle"])

    # Telemetry
    car.add(ConsoleTelemetry(), inputs=["angle", "safe_throttle", "yaw", "orange", "blue"])

    # Actuators
    car.add(LegoSteering(), inputs=["angle"])
    car.add(LegoThrottle(), inputs=["safe_throttle"])

    return car

# ----- MAIN -----
def shutdown(_sig=None, _frame=None):
    print("Shutting down...")
    try:
        vehicle.shutdown()
    except:
        pass
    sys.exit(0)

signal.signal(signal.SIGTERM, shutdown)
signal.signal(signal.SIGINT, shutdown)

vehicle = build_vehicle()

try:
    vehicle.start(rate_hz=DRIVE_LOOP_HZ)
except KeyboardInterrupt:
    print("KeyboardInterrupt received.")
finally:
    shutdown()
