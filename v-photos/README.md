Vehicle's photos
====
Front View
The front view showcases the mounted sensors, including the camera and gyroscope.

Rear View
The rear view highlights the motor housing and cable management.

Side View
The side view provides a clear perspective of the car's LEGO frame and wheel assembly.

Top View
The top view reveals the layout of the Raspberry Pi, wiring, and additional components.


class ImageFilter:
    def __init__(self):
        pass

    def run(self, img):
        import cv2
        import numpy as np
        # Step 1: Gaussian Blur
        blurred = cv2.GaussianBlur(img, (5, 5), 0)

        # Step 2: Convert to HSV for brightness boost
        hsv = cv2.cvtColor(blurred, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        v = np.where(v > 128, 255, 0).astype(np.uint8)
        hsv = cv2.merge((h, s, v))

        # Step 3: Back to RGB
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        # Step 4: Simplify to primary colors
        simplified = self.map_to_primary_colors(rgb)
        return simplified

    def map_to_primary_colors(self, img):
        import numpy as np
        pixels = img.reshape((-1, 3))

        color_map = {
            "red":     ([150, 0, 0],    [255, 100, 100]),
            "green":   ([0, 150, 0],    [100, 255, 100]),
            "blue":    ([0, 0, 150],    [100, 100, 255]),
            "yellow":  ([150, 150, 0],  [255, 255, 100]),
            "cyan":    ([0, 150, 150],  [100, 255, 255]),
            "magenta": ([150, 0, 150],  [255, 100, 255]),
            "white":   ([200, 200, 200],[255, 255, 255]),
            "black":   ([0, 0, 0],      [50, 50, 50]),
        }

        rep_colors = {
            "red":     [255, 0, 0],
            "green":   [0, 255, 0],
            "blue":    [0, 0, 255],
            "yellow":  [255, 255, 0],
            "cyan":    [0, 255, 255],
            "magenta": [255, 0, 255],
            "white":   [255, 255, 255],
            "black":   [0, 0, 0],
        }

        new_pixels = []
        for pixel in pixels:
            replaced = False
            for name, (lower, upper) in color_map.items():
                if all(lower[i] <= pixel[i] <= upper[i] for i in range(3)):
                    new_pixels.append(rep_colors[name])
                    replaced = True
                    break
            if not replaced:
                new_pixels.append([0, 0, 0])
        return np.array(new_pixels, dtype=np.uint8).reshape(img.shape)


class FilterPreviewWindow:
    def __init__(self, window_name="Filtered View"):
        import cv2
        self.window_name = window_name
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 320, 240)

    def run(self, img):
        import cv2
        cv2.imshow(self.window_name, img)
        cv2.waitKey(1)
        return

    def shutdown(self):
        import cv2
        cv2.destroyAllWindows()


def add_camera(car):
    from donkeycar.parts.camera import PiCamera
    from donkeycar.parts.transform import Lambda

    CAPTURE_W, CAPTURE_H = 192, 144
    CROP_W, CROP_H = 160, 120

    def center_crop(img, tw=CROP_W, th=CROP_H):
        h, w = img.shape[:2]
        x0 = (w - tw) // 2
        y0 = (h - th) // 2 + 8
        return img[y0:y0 + th, x0:x0 + tw]

    cam = PiCamera(image_w=CAPTURE_W, image_h=CAPTURE_H, image_d=3,
                   vflip=False, hflip=False)
    car.add(cam, outputs=["cam/raw"], threaded=True)

    car.add(ImageFilter(), inputs=["cam/raw"], outputs=["cam/filtered"])
    car.add(FilterPreviewWindow(), inputs=["cam/filtered"], outputs=[])
    car.add(Lambda(center_crop), inputs=["cam/filtered"], outputs=["cam/image_array"])
