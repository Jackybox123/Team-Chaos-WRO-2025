class ColorLineCounter:
    def __init__(self):
        self.sensor = ColorSensor("D")
        self.prev_color = "white"  # track previous detected color
        self.orange_count = 0
        self.blue_count = 0

    def run(self):
        r, g, b, i = self.sensor.get_color_rgbi()

        # Define white as high RGB + intensity
        is_white = (r > 200 and g > 200 and b > 200 and i > 180)

        # Detect orange roughly (adjust thresholds as needed)
        is_orange = (r > 180 and g > 50 and g < 140 and b < 50 and i > 100)

        # Detect blue roughly
        is_blue = (r < 50 and g < 100 and b > 150 and i > 100)

        current_color = "white"
        if is_orange:
            current_color = "orange"
        elif is_blue:
            current_color = "blue"
        elif not is_white:
            current_color = "other"

        # Count on transition from white → orange or white → blue
        if self.prev_color == "white":
            if current_color == "orange":
                self.orange_count += 1
                print(f"Orange line detected! Count: {self.orange_count}")
            elif current_color == "blue":
                self.blue_count += 1
                print(f"Blue line detected! Count: {self.blue_count}")

        self.prev_color = current_color

        return self.orange_count, self.blue_count
