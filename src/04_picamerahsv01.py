import cv2
import numpy as np
import time
from picamera2 import Picamera2
from libcamera import Transform

# Initialize PiCamera
config_dict = {"size": (160, 120), "format": "BGR888"}
transform = Transform(False, False)
camera = Picamera2()
config = camera.create_preview_configuration(config_dict, transform=transform)
camera.align_configuration(config)
camera.configure(config)
        # try min / max frame rate as 0.1 / 1 ms (it will be slower though)
camera.set_controls({"FrameDurationLimits": (100, 1000)})
camera.start()

        # initialize the frame and the variable used to indicate
        # if the thread should be stopped
frame = None
on = True
image_d = 3

        # get the first frame or timeout

# Allow the camera to warm up
time.sleep(0.5)

# Define color ranges for red and green in HSV color space
lower_red1 = np.array([0, 120, 70])
upper_red2 = np.array([10, 255, 255])
lower_green = np.array([36, 100, 100])
upper_green = np.array([86, 255, 255])
lower_red2 = np.array([170, 50, 50])
upper_red2 = np.array([180, 255, 255])


counter =0
start_time =time.time()

# Capture frames continuously from the PiCamera
while True:
    frame = camera.capture_array("main")
    # Convert the image to HSV color space
    frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
    image = frame
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Create masks for red and green colors
    mask_red1 = cv2.inRange(hsv_image, lower_red1, upper_red2)
    mask_green = cv2.inRange(hsv_image, lower_green, upper_green)
    mask_red2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)

    # Count non-zero pixels in each mask to detect the presence of color
    red_count = cv2.countNonZero(mask_red)
    green_count = cv2.countNonZero(mask_green)

    if red_count > green_count and red_count > 1000:  # Threshold to avoid small detections
        print("Red detected")
    elif green_count > red_count and green_count > 1000:
        print("Green detected")
    else:
        print("No red or green detected")

    # Display the frame (optional, can be removed if you don't need the display)
    cv2.imshow("Frame", image)

    # Clear the stream for the next frame
#     raw_capture.truncate(0)

#     counter +=1
#     if (time.time() - start_time) > 1:
#         print(counter)
#         counter = 0
#         start_time = time.time()
  
#     previous_time = current_time
   # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the resources
cv2.destroyAllWindows()
camera.close()




