from buildhat import ColorSensor
import time

class ColorLineCounter:
    def __init__(self, max_count=24):
        self.sensor = ColorSensor("D")
        self.prev_on_line = False
        self.last_count_time = 0
        self.total_count = 0
        self.debounce_seconds = 0.5  # debounce time between counts
        self.max_count = max_count

    def run(self):
        r, g, b, i = self.sensor.get_color_rgbi()

        # Define white as very bright RGB + intensity
        is_white = (r > 230 and g > 230 and b > 230 and i > 180)
        on_line = not is_white

        current_time = time.monotonic()

        # Only count if we are on a non-white strip, previously off line,
        # debounce passed, and total count is less than max
        if (on_line and not self.prev_on_line and
            current_time - self.last_count_time > self.debounce_seconds and
            self.total_count < self.max_count):

            self.total_count += 1
            self.last_count_time = current_time
            print(f"✅ COUNTED {self.total_count} | RGBI: R={r}, G={g}, B={b}, I={i}")

        self.prev_on_line = on_line

        # Return total_count as orange count, zero for blue (compatibility)
        return self.total_count, 0
