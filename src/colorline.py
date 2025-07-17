from buildhat import ColorSensor
import time

class ColorLineCounter:
    def __init__(self, max_total_count=24):
        self.sensor = ColorSensor("D")
        self.total_count = 0
        self.max_total_count = max_total_count
        self.last_count_time = 0
        self.debounce_seconds = 0.1  # Prevents double-counts

    def run(self):
        print("🚗 Starting line counting...")

        while self.total_count < self.max_total_count:
            color = self.sensor.wait_for_new_color()
            current_time = time.monotonic()

            # Only count if debounce time has passed
            if current_time - self.last_count_time < self.debounce_seconds:
                continue

            if color in ["orange", "blue"]:
                self.total_count += 1
                self.last_count_time = current_time
                print(f"✅ Line {self.total_count}: Detected {color}")

        print("🎉 Done counting 24 lines!")
        return self.total_count
