import threading
from buildhat import ColorSensor
import time

class ThreadedLineCounter(threading.Thread):
    def __init__(self, max_count=24):
        super().__init__()
        self.sensor = ColorSensor("D")
        self.max_count = max_count
        self.orange_count = 0
        self.blue_count = 0
        self.running = True
        self.lock = threading.Lock()

    def run(self):
        # Initial sync: wait for white to start
        self.sensor.wait_until_color('white')

        while self.running:
            # Wait for next line color
            color = self.sensor.wait_for_new_color()

            with self.lock:
                if color == 'orange':
                    self.orange_count += 1
                    print(f"Orange line counted: {self.orange_count}")
                elif color == 'blue':
                    self.blue_count += 1
                    print(f"Blue line counted: {self.blue_count}")

                total = self.orange_count + self.blue_count
                if total >= self.max_count:
                    print("Max line count reached, stopping counting thread.")
                    self.running = False

            # Wait for white again before next line
            self.sensor.wait_until_color('white')

    def get_counts(self):
        with self.lock:
            return self.orange_count, self.blue_count

    def stop(self):
        self.running = False

# Usage in main driving code:

line_counter = ThreadedLineCounter(max_count=24)
line_counter.start()

try:
    while True:
        # Your driving code here
        # Example: read counts safely to decide when to stop
        orange, blue = line_counter.get_counts()
        total = orange + blue
        print(f"Current counts - Orange: {orange}, Blue: {blue}")

        if total >= 24:
            print("Reached line count goal, stopping car.")
            break

        time.sleep(0.1)  # Run driving loop at ~10Hz

finally:
    line_counter.stop()
    line_counter.join()
