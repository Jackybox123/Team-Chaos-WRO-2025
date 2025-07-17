from buildhat import ColorSensor
import time

class WaitColorLineCounter:
    def __init__(self, max_count=24):
        self.sensor = ColorSensor("D")
        self.max_count = max_count
        self.orange_count = 0
        self.blue_count = 0

    def run(self):
        # Wait for white space (to reset)
        self.sensor.wait_until_color('white')
        
        # Wait for next color - orange or blue
        while self.orange_count + self.blue_count < self.max_count:
            # wait for orange or blue - we can’t pass multiple colors in one call
            # so wait for *any* color other than white by polling wait_for_new_color()

            color = self.sensor.wait_for_new_color()

            if color == 'orange':
                self.orange_count += 1
                print(f"Orange line counted: {self.orange_count}")
            elif color == 'blue':
                self.blue_count += 1
                print(f"Blue line counted: {self.blue_count}")
            else:
                # Ignore other colors, but if white, wait for color again
                if color == 'white':
                    continue
                print(f"Ignored color: {color}")

            # After detecting a color, wait for white again before next count
            self.sensor.wait_until_color('white')

        return self.orange_count, self.blue_count

# Usage example:
counter = WaitColorLineCounter(max_count=24)
orange_total, blue_total = counter.run()
print(f"Counting finished! Orange: {orange_total}, Blue: {blue_total}")
