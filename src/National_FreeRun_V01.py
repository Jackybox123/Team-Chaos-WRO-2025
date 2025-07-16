#!/usr/bin/env python3
# drive_with_gyro_guard.py

# ... [all your imports and existing code above remain unchanged] ...

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

        # Determine if the color is considered "white"
        is_white = (r > 230 and g > 230 and b > 230 and i > 180)
        on_line = not is_white

        current_time = time.monotonic()
        if on_line and not self.prev_on_line:
            if current_time - self.last_count_time > self.debounce_seconds:
                self.total_count += 1
                self.last_count_time = current_time
                print(f"✅ COUNTED {self.total_count} | RGBI: R={r}, G={g}, B={b}, I={i}")

        self.prev_on_line = on_line
        return self.total_count, 0  # output orange count, blue count unused

# ... rest of your StopGuard, ConsoleTelemetry, and Donkey Car parts unchanged ...

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
