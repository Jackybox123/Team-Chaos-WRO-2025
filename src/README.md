Control software
====
Python: The primary programming language used to develop the control software. Python's simplicity and extensive libraries make it ideal for interfacing with hardware components, processing data, and implementing machine learning algorithms.

TensorFlow: An open-source machine learning framework used to develop and train AI models. TensorFlow allows for the implementation of neural networks that enable the AI RC Car to recognize objects, make decisions, and navigate autonomously.

PuTTY: A terminal emulator used to connect to the Raspberry Pi remotely. With PuTTY, developers can access the Raspberry Pi's command line interface over SSH, allowing for configuration, debugging, and deployment of software updates without needing a direct physical connection.

Pi Camera: The camera module connected to the Raspberry Pi, used for capturing real-time video and images. The Pi Camera is integral to the AI RC Car's vision system, allowing it to process visual data for object detection, line tracking, and environmental awareness. The camera feeds data into the TensorFlow models for real-time processing.

Calibrate: A function or process within the software used to adjust and fine-tune the sensors and motors to ensure accurate readings and movements. Calibration is essential for the car to operate correctly, as it ensures that sensors provide reliable data and motors respond accurately to control signals.

Config: A configuration module or file that stores settings and parameters for the AI RC Car. This includes sensor thresholds, motor speed limits, and neural network hyperparameters. The config module allows for easy adjustments and fine-tuning of the system.

Manage: A management module that oversees the overall operation of the AI RC Car. This includes starting and stopping the car, monitoring system health, and handling communication between different software components. The management module ensures that all parts of the system work together seamlessly.

myconfig: A custom configuration file or module where user-specific settings and preferences are stored. This allows for personalization and customization of the AI RC Car's behavior to suit different environments or user requirements.

Train: A training module that uses TensorFlow to develop and refine the AI models. The train module processes training data, adjusts model parameters, and evaluates performance to improve the car's ability to navigate and make decisions. Training is an iterative process that enhances the car's autonomy and intelligence.

Workflow
Configuration: Using the config and myconfig modules, developers set up the initial parameters and settings for the car, tailoring it to the specific environment and requirements.

Calibration: Sensors and motors are calibrated to ensure accurate data collection and response. This step is crucial for reliable operation.

Training: AI models are trained using TensorFlow. The train module processes data and updates the models to enhance the car's performance.

Deployment: Using PuTTY, the control software is deployed to the Raspberry Pi. Developers can remotely manage and update the software as needed.

Operation: The management module coordinates the car's activities, ensuring that the AI models, sensors, and motors work together to achieve autonomous navigation.


Turkey_FreeRun_V01.py:


The Python code is designed to control an RC car autonomously using machine learning, camera input, and sensor data. It utilizes a Raspberry Pi camera to capture images, which are then processed by a pre-trained AI model to predict the car’s steering and throttle control. This allows the car to navigate without human intervention.

The car’s rotation is monitored using a gyroscope, which measures its yaw angle. The system tracks the car’s rotation through the Z-axis data from the gyroscope, and the car stops after completing three full rotations (1080 degrees). This ensures the car follows a precise path and halts at the correct time.

The movement of the car is controlled by adjusting PWM signals, which are sent to a motor driver to control the motor’s speed and direction based on the AI model's predictions. The car only starts moving once it receives a signal from the GPIO input, ensuring everything is set up before it begins operation.

Real-time feedback is displayed on an LCD screen, showing useful information such as the yaw angle and the AI model’s status. This allows the car's performance to be monitored continuously. The car also includes a basic color detection feature, which can identify specific colors, though it is not fully integrated in the current version of the code.

In the main loop, sensor data is gathered, images are captured, predictions are made by the AI model, and the car’s movement is controlled. Once the three rotations are completed, the car stops as planned. Finally, the code includes a shutdown process to safely power down the system, ensuring that everything, including the camera, is properly turned off after the task is complete.

This Python code is used in our RC car project to monitor an activation button that starts the car. The button, connected to GPIO 25, serves as an input device that triggers the car to start running. The code continuously checks whether the button has been pressed, and when it detects the button’s input, it will activate certain functions to begin the car's movement.


01_detect_GPIO.py:

We use the GPIO.BCM mode to reference the pins by their Broadcom numbers. GPIO 25 is set up as an input pin with an internal pull-up resistor, which means it will be "high" (on) unless the button is pressed, which would then bring the signal "low" (off). This setup helps the code to distinguish when the button is pressed.

The program runs in a loop, checking the button state every 0.1 seconds. If the button is pressed (GPIO 25 goes low), the car can then start running, making this code crucial for triggering the car's activation. The continuous checking ensures the car responds immediately to the button press.


04_picamerahsv01.py:


In this code, we use the PiCamera with the Picamera2 library to continuously capture frames from the camera. The frames are processed to detect specific colors, namely red and green, in the image. We use OpenCV’s cv2 library to convert the captured frames to the HSV (Hue, Saturation, Value) color space, which helps in better detecting color ranges.

The code defines two color ranges for red and one for green. It then creates masks for each color by filtering the HSV image. The number of non-zero pixels in each mask is counted to determine the presence of red or green. If there are more red pixels than green pixels and the red count exceeds a threshold (1000 pixels), it prints "Red detected". Similarly, if green has a higher count, it prints "Green detected". If neither color is detected, it prints "No red or green detected". This code helps the car detect obstacles using the camera. It looks for red and green colors to understand which dirction to go on the third round because at the end of the second round it will be able to use this code to know which obstacle its looking at to determine if it will continue towards the third round or it will reverse directions and continue to the third round going in the same direction.


06_read_GYRO_with_dts-integral.py:


This code helps our car track how much it’s turning by reading data from a WT901 sensor. The sensor measures the car’s rotation around the Z-axis (yaw), which tells us how the car is facing. It works like this:

First, the car uses I2C to talk to the sensor and get data about its Z-axis rotation. The sensor gives us a raw number that represents how fast the car is turning. We convert that number into degrees per second, so we know how much it’s rotating.

Then, the code keeps track of how long it’s been since the last time it checked the sensor. It uses that time difference to calculate how much the car’s yaw angle has changed. This way, we can figure out the car’s exact direction by adding up the changes over time.

So, when the car is running, the code updates the yaw angle and prints out the car’s direction in real time. This helps the car know where it’s facing this is important as it lets us known when to change between ai models for example we trained models for parking and many other stuff once the car reaches a certain degree or location this lets us use a different ai model to accomplish a different task that is required at that point in time of the run.



