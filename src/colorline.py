from buildhat import ColorSensor
import time

class ColorLineCounter:
    def __init__(self):
        self.sensor = ColorSensor("D")
        self.prev_on_line = False
        self.last_count_time = 0
        self.total_count = 0
        self.debounce_seconds = 0.5

    def run(self):
        r, g, b, i = self.sensor.get_color_rgbi()

        # Consider white if RGB and intensity are very high
        is_white = (r > 230 and g > 230 and b > 230 and i > 180)
        on_line = not is_white

        current_time = time.monotonic()
        if on_line and not self.prev_on_line:
            if current_time - self.last_count_time > self.debounce_seconds:
                self.total_count += 1
                self.last_count_time = current_time
                print(f"✅ COUNTED {self.total_count} | RGBI: R={r}, G={g}, B={b}, I={i}")

        self.prev_on_line = on_line
        return self.total_count, 0  # orange, blue not used
