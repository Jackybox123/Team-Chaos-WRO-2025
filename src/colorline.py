from buildhat import ColorSensor
import time

class ColorLineCounter:
    def __init__(self, max_count=24):
        self.sensor = ColorSensor("D")
        self.last_count_time = 0
        self.total_count = 0
        self.debounce_seconds = 0.5
        self.max_count = max_count
        self.prev_was_white = True  # Start assuming it's on white

    def is_white(self, r, g, b, i):
        return (r > 230 and g > 230 and b > 230 and i > 180)

    def run(self):
        while self.total_count < self.max_count:
            # Wait until color changes
            r, g, b, i = self.sensor.wait_for_new_color()
            current_time = time.monotonic()

            is_white = self.is_white(r, g, b, i)

            # Detect transition: white → non-white
            if self.prev_was_white and not is_white:
                if current_time - self.last_count_time > self.debounce_seconds:
                    self.total_count += 1
                    self.last_count_time = current_time
                    print(f"✅ COUNTED {self.total_count} | RGBI: R={r}, G={g}, B={b}, I={i}")

            self.prev_was_white = is_white

        return self.total_count, 0
