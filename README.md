<center><h1>Team Chaos — WRO 2025 Future Engineers</h1></center>

<!-- ====== Banner Section ====== -->
<p align="center">
  <!-- Paste your banner image here -->
  <img src="./images/banner.png" width="100%" alt="Team Chaos Banner">
</p>

[![GitHub](https://img.shields.io/badge/GitHub-Jackybox123/Team--Chaos--WRO--2025-blue?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Jackybox123/Team-Chaos-WRO-2025)
[![YouTube](https://img.shields.io/badge/YouTube-Team%20Chaos-red?style=for-the-badge&logo=youtube&logoColor=white)](https://www.youtube.com/@teamchaos)

---

## Table of Contents
* [The Team](#the-team)
* [Project Summary](#project-summary)
* [Executive Summary](#executive-summary)
* [Communication Design](#communication-design)
* [Technical Solution Design](#technical-solution-design)
* [Robot Photos](#robot-photos)
* [Mobility Management](#mobility-management)
* [Power & Sense Management](#power-and-sense-management)
* [Obstacle Management](#obstacle-management)
* [Software and Coding Explanation](#software-and-coding-explanation)
* [Photos of the Data We Captured](#photos-of-the-data-we-captured)
* [Schematic Diagram of the CNN](#schematic-diagram-of-the-cnn)
* [Code Explanations & Code Blocks](#code-explanations--code-blocks)
* [Car Components and Explanation](#car-components-and-explanation)
* [Component Cost Table](#component-cost-table)
* [Assembly Instructions](#assembly-instructions)
* [Challenges and Solutions](#challenges-and-solutions)
* [Conclusion](#conclusion)

---

## The Team <a name="the-team"></a>

### Team Members

**Kyle Ho:**  
An 9th grader high school student from California, United States. He had participated in WRO in previous seasons. He loves robots and is a very innovative person.  

**Jiapeng Xu:**  
Hello, I am a 11th grader from California United States. In my free time I like to code, swim, and snowboard.  I love to come up with solutions for problems and experiment with robotics and AI. This is my second year competing in WRO.  

**Coach: Fei Guo (Robert Guo):**  
Fei Guo (Robert Guo) provides the technical guidance and leadership required to keep Team Chaos on track. With extensive experience in robotics, he helps us navigate complex challenges, refine our designs, and develop solutions that work in competitive and real-world settings.

---

## Project Summary <a name="project-summary"></a>

Using the method of machine learning, a CNN neural network is built to train and predict the pictures of the competition venue.  
This is our AI Lego Car.

Our project combines semi-artificial intelligence and semi-robotics to create a fully functional AI-powered self-driving robot.

---

## Executive Summary <a name="executive-summary"></a>

The Future Engineers competition is an autonomous driving challenge where the robot must navigate and avoid obstacles on the course. We designed our system using a Raspberry Pi as the core processing unit for AI operations. Running on the Raspberry Pi’s Linux OS, the Python environment integrates seamlessly with TensorFlow, Google’s machine learning framework.

We began by attaching a camera module (lens) to the Raspberry Pi to record video footage of a manually operated vehicle. This footage, along with the associated remote-control input data (steering and throttle), was used to train a Convolutional Neural Network (CNN). The CNN learns patterns between image frames and control inputs. Once trained, the model is used to analyze live camera input and predict throttle and steering values in real time.

These AI-generated control signals are then transmitted to a LEGO-based vehicle, enabling it to drive autonomously without human intervention.

---


## Communication Design <a name="communication-design"></a>

Our design facilitates communication between the remote-control interface and the Raspberry Pi, which in turn communicates seamlessly with the LEGO car.

<p align="center">
  <!-- Paste your image of the communication setup here -->
  <img src="./images/communication_setup.jpg" width="80%" alt="Communication Design">
</p>

Control Logic Design: The control logic involves recording the playing field's images and data through a remote-controlled robot. We designed and trained a neural network using a model of the field. During runs, the neural network accesses the trained behavior model to execute actions reminiscent of human operations. These action commands are sent to the LEGO car, which then performs the designated tasks, such as determining throttle or speed based on the image feedback from the Pi Camera.

---

## Technical Solution Design <a name="technical-solution-design"></a>

### Overall Technical Solution Framework

<p align="center">
  <!-- Paste your image of the communication setup here -->
  <img src="./images/technicalsolution_setup.jpg" width="80%" alt="Technical Design">
</p>

Machine learning (ML) encompasses a spectrum of methodologies that empower computers to autonomously discover and refine algorithms. A convolutional neural network (CNN) is a specialized feed-forward neural network that inherently learns and perfects feature engineering through filter optimization, making it particularly effective in AI image recognition and autonomous driving applications.

TensorFlow is a renowned, open-source software library dedicated to machine learning and artificial intelligence, offering a versatile toolkit for developers.

---
## Robot Photos <a name="robot-photos"></a>

Here are multiple views of our AI Lego Car.  

| <img src="./images/front.jpg" width="90%" /> | <img src="./images/back.jpg" width="85%" /> | 
| :--: | :--: | 
| *Front* | *Back* |
| <img src="./images/left.jpg" width="90%" /> | <img src="./images/right.jpg" width="85%" /> | 
| *Left* | *Right* |
| <img src="./images/top.jpg" width="90%" /> | <img src="./images/bottom.jpg" width="85%" /> | 
| *Top* | *Bottom* |

---

## Mobility Management <a name="mobility-management"></a>

![Mobility Model](./images/mobility_model.jpg)

Our vehicle utilizes a custom-built LEGO car platform, which incorporates a DC motor to control throttle and a servo for steering. The DC motor was chosen for its ability to provide smooth acceleration and consistent speed, while the servo offers precise steering control. Both components are managed by the Build HAT, which interfaces directly with the Raspberry Pi to allow for dynamic control.

**AI-Controlled Mobility:**  
To enable autonomous operation, we integrated a Pi Camera to feed real-time visual data to an AI model. The model predicts the appropriate throttle and steering adjustments based on obstacle detection and navigation requirements. The AI model was trained using data collected from manual control, where the car was driven, and the corresponding throttle and steering values were logged alongside images captured by the Pi Camera.

**Challenges in Mobility Management:**  
One of the challenges faced during development was ensuring the AI model could accurately distinguish between obstacles with similar shapes. We overcame this by incorporating color recognition, which allowed the AI to make more precise decisions. Another challenge was syncing the throttle and steering inputs in real-time during training, especially when switching between AI models for different control tasks.

---

## Power & Sense Management <a name="power-and-sense-management"></a>

![Wiring Diagram](./images/wiring_diagram.png)

Our technical solution incorporated a dual system approach for managing the power and sensing capabilities of the autonomous vehicle.

**Power Management:**  
The LEGO car is powered by a 7.4V lithium battery. The Raspberry Pi, camera, and additional sensor modules are powered by a 5V 2.5A power bank. The Pi Camera provides visual input for the AI model, while sensors connected via the Sense HAT and Build HAT support obstacle detection and decision-making. These sensors help the AI model predict steering and throttle adjustments for accurate navigation. Power consumption is carefully monitored, and the system’s wiring ensures efficient energy distribution with minimal loss.

---

## Obstacle Management <a name="obstacle-management"></a>

![Obstacle Detection](./images/obstacle_detection.png)

In our autonomous Lego Car vehicle project, the approach to obstacle management is distinctively different from traditional automated programming. Our technical philosophy has been consistent from the beginning, focusing on a strategy that does not rely on pre-defined rules or flowcharts for navigating around obstacles.

Instead of traditional programming techniques such as using flowcharts, pseudocode, or annotated code, we employ a data-driven approach with our AI. The AI learns to drive the vehicle through actual operation by a human operator. The machine records how the operator drives, noting steering and throttle data, and then this data is used to train a neural network.

During the training phase, initially, the vehicle could not distinguish colours such as red and green but was able to identify and navigate around obstacles. With the accumulation of more data - around 70,000 data points - the vehicle began to discern obstacles. At 90,000 data points, it could differentiate and react to colors, associating red with turning right and green with turning left.

Our primary focus has been on accurately recording the operator's maneuvers and feeding this extensive dataset to the model, allowing it to learn driving habits and patterns - a process known as 'fitting.' There is no traditional strategy or flowchart involved; the vehicle's ability to navigate through the course and manage obstacles is entirely based on continuous training from the data gathered during these driving sessions.

**Key Points:**  
The LEGO car uses a combination of AI-driven image processing and sensor input from the Pi Camera and Sense HAT.  
Real-time visual data is analyzed to determine throttle and steering commands.  
Color-based recognition was added to distinguish between obstacle types when shape recognition alone was insufficient.  
The AI system was refined through trial and error, with detailed documentation available in the GitHub source files.  
A major challenge was real-time performance; we addressed this by optimizing the recognition algorithm and streamlining throttle/steering prediction.  
The system's ability to adapt to dynamic environments stems from its data-centric learning method, mimicking human decision-making through iterative training.

---

## Software and Coding Explanation <a name="software-and-coding-explanation"></a>


In our project, the hardware components primarily consist of a display, a single-board computer—specifically the Raspberry Pi 4—and a camera. We use the Raspberry Pi 4 in conjunction with the camera to capture real-time images from the play area. Additionally, during the learning phase, we employ a joystick to collect control data. This setup enables us to synchronize live footage with control inputs, such as steering and throttle actions, thereby creating a detailed dataset for the AI to refine its driving algorithms.

At the outset, we gather raw data, such as images, using the Raspberry Pi. This data is subsequently transferred to a conventional computer for advanced processing.

On this computer, we run a series of specialized programs that use the collected data to train an AI model capable of autonomous driving. The result of this training is a file that encapsulates the trained AI model, which we then load back onto the Raspberry Pi.

Armed with this AI model, the Raspberry Pi analyzes live images and generates precise control instructions, including throttle and steering commands. These commands are sent wirelessly to the LEGO car, where they are executed via the Build HAT, which controls the LEGO motors accordingly.


## Photos of the Data We Captured <a name="photos-of-the-data-we-captured"></a>

<p align="center">
  <!-- Paste your data photos here -->
  <img src="./images/data_collected_1.jpg" width="80%" alt="Data Collected 1">
  <img src="./images/data_collected_2.jpg" width="80%" alt="Data Collected 2">
</p>


## Schematic Diagram of the CNN <a name="schematic-diagram-of-the-cnn"></a>

<p align="center">
  <!-- Paste your CNN schematic image here -->
  <img src="./images/cnn_diagram.jpg" width="80%" alt="CNN Schematic">
</p>

---

## Code Explanations & Code Blocks <a name="code-explanations--code-blocks"></a>

### GPIO Button Code

```python
# GPIO Pin setup and car start logic
# Your full code here
```
We use the GPIO.BCM mode to reference the pins by their Broadcom numbers. GPIO 25 is set up as an input pin with an internal pull-up resistor, which means it will be "high" (on) unless the button is pressed, which would then bring the signal "low" (off). This setup helps the code to distinguish when the button is pressed. The program runs in a loop, checking the button state every 0.1 seconds. If the button is pressed (GPIO 25 goes low), the car can then start running, making this code crucial for triggering the car's activation. The continuous checking ensures the car responds immediately to the button press.

---

### PiCamera Color Detection

```python
# PiCamera color detection code for red/green
# Your full code here
```
In this code, we use the PiCamera with the Picamera2 library to continuously capture frames from the camera. The frames are processed to detect specific colors, namely red and green, in the image. We use OpenCV’s cv2 library to convert the captured frames to the HSV (Hue, Saturation, Value) color space, which helps in better detecting color ranges.

The code defines two color ranges for red and one for green. It then creates masks for each color by filtering the HSV image. The number of non-zero pixels in each mask is counted to determine the presence of red or green. If there are more red pixels than green pixels and the red count exceeds a threshold (1000 pixels), it prints "Red detected". Similarly, if green has a higher count, it prints "Green detected". If neither color is detected, it prints "No red or green detected". This code helps the car detect obstacles using the camera. It looks for red and green colors to understand which direction to go on the third round because at the end of the second round it will be able to use this code to know which obstacle its looking at to determine if it will continue towards the third round or it will reverse directions and continue to the third round going in the same direction.

---

### WT901 Yaw Sensor

```python
# WT901 yaw sensor code
# Your full code here
```
This code helps our car track how much it’s turning by reading data from a WT901 sensor. The sensor measures the car’s rotation around the Z-axis (yaw), which tells us how the car is facing. It works like this: First, the car uses I2C to talk to the sensor and get data about its Z-axis rotation. The sensor gives us a raw number that represents how fast the car is turning. We convert that number into degrees per second, so we know how much it’s rotating. Then, the code keeps track of how long it’s been since the last time it checked the sensor. It uses that time difference to calculate how much the car’s yaw angle has changed. This way, we can figure out the car’s exact direction by adding up the changes over time. So, when the car is running, the code updates the yaw angle and prints out the car’s direction in real time. This helps the car know where it’s facing this is important as it lets us know when to change between ai models for example we trained models for parking and many other stuff once the car reaches a certain degree or location this lets us use a different ai model to accomplish a different task that is required at that point in time of the run.

---

### Keyboard & Joystick Control

```python
# Keyboard/Joystick manual control code
# Your full code here
```
This code is used to control the car’s speed and direction using a keyboard. It begins by setting up the signals for controlling the car’s motors. The car starts with a default speed and steering values, both set to 350, which represent the car’s starting movement. You can control how the car moves by pressing certain keys on the keyboard. Pressing the i key increases the car’s speed, while pressing the k key decreases the speed. If you press j, the car will steer left, and pressing L will steer it right. Each time you press one of these keys, the car’s speed or direction will change accordingly.

To stop controlling the car, you can press the q key. This will stop the car’s movement and end the program. After stopping, the car will automatically reset to its default speed and steering position, moving in a neutral state for a short time before the program ends. In summary, this code allows you to control the car’s speed and steering by using the i, k, j, and l keys, and you can stop the car anytime by pressing the q key and this important for us as that lets up to adjust the car by using manual controls from keyboard to drive it around and observe its movements for further adjustments.

This script is designed for us to control a car's movement using a joystick, with the help of the Adafruit PCA9685 module to adjust the car's throttle and steering. First, we set the PWM frequency for the servos at 60Hz, which is standard for servo motors, and initialize the joystick with the Pygame library. After the joystick is connected, the script waits for input. Once data is received from the joystick, it moves forward with setting default values for the throttle and steering signals, which help control the car’s movement. In the main loop, we read the joystick's axis values to control the throttle and steering. The vertical axis (axis4) controls the car’s throttle, and the signal is adjusted between 300 and 400. The horizontal axis (axis0) controls the steering, with the signal adjusted between 220 and 520. These PWM signals are then sent to the PCA9685 module, which controls the servos, allowing us to steer and adjust the throttle in real-time. The script continuously updates the servo positions based on joystick input, giving us full control over the car’s movement this is like 07_Control_RCcar_with_KB.py where the difference is we are controlling the car this time with controller which makes it easier to drive the car and observe the car in at the same time.

---

### Autonomous Car Main Loop

```python
# Main autonomous car loop with CNN prediction
# Your full code here
```
The Python code is designed to control an Lego car autonomously using machine learning, camera input, and sensor data. It utilizes a Raspberry Pi camera to capture images, which are then processed by a pre-trained AI model to predict the car’s steering and throttle control. This allows the car to navigate without human intervention. The car’s rotation is monitored using a gyroscope, which measures its yaw angle. The system tracks the car’s rotation through the Z-axis data from the gyroscope, and the car stops after completing three full rotations (1080 degrees). This ensures the car follows a precise path and halts at the correct time. The movement of the car is controlled by adjusting PWM signals, which are sent to a motor driver to control the motor’s speed and direction based on the AI model's predictions. The car only starts moving once it receives a signal from the GPIO input, ensuring everything is set up before it begins operation. Real-time feedback is displayed on an LCD screen, showing useful information such as the yaw angle and the AI model’s status. This allows the car's performance to be monitored continuously. The car also includes a basic color detection feature, which can identify specific colors, though it is not fully integrated in the current version of the code. In the main loop, sensor data is gathered, images are captured, predictions are made by the AI model, and the car’s movement is controlled. Once the three rotations are completed, the car stops as planned. Finally, the code includes a shutdown process to safely power down the system, ensuring that everything, including the camera, is properly turned off after the task is complete.

This code enables our robot to navigate the obstacle run, detect signals, and park autonomously. It uses a camera and a machine-learning model to process what the robot sees, helping it decides how to steer and move forward. A gyroscope tracks the robot's orientation, ensuring it completes three full laps before stopping. The code also allows the robot to detect red signals, which trigger a U-turn, enabling it to adapt dynamically to the course. The robot's motors and steering are controlled by PWM signals for smooth and precise movement, with a GPIO pin used to start or stop its actions. Its status, like running or waiting, is displayed on an LCD screen. At the end of the obstacle course, the code switches the robot to parking mode, where a specific AI model guides it into a parking space. This makes the robot fully autonomous and ready for robotics competitions.


---

## Car Components and Explanation <a name="car-components-and-explanation"></a>

### LEGO Motor

<p align="center">
  <!-- Replace with your actual LEGO motor image -->
  <img src="./images/lego_motor.jpg" width="60%" alt="LEGO Motor">
</p>

The LEGO motor is a critical part of our robot, responsible for controlling the car’s movement and driving. In our design, the LEGO motor was chosen as an improvement over motors used in previous seasons. We found that the LEGO motor provides smoother acceleration, consistent speed, and precise control, making it easy to integrate with our Build HAT and Raspberry Pi system. The simplicity and reliability of the LEGO motor allowed us to focus on developing the AI and navigation systems, confident that the mechanical aspect of movement would be handled effectively. Its compatibility with LEGO Technic elements also meant we could easily adjust or expand our drivetrain as needed during testing and competition.

---

### VL53L0X ToF (Time-of-Flight) Sensor

<p align="center">
  <!-- Replace with your actual VL53L0X sensor image -->
  <img src="./images/vl53l0x.jpg" width="60%" alt="VL53L0X ToF Sensor">
</p>

The VL53L0X ToF sensor is an essential upgrade in our car’s sensing suite, especially for parking and wall detection. As a laser-based distance sensor, it provides millimeter-accurate measurements of the distance between the car and nearby objects. This sensor plays a crucial role in our parking control system, enabling the robot to detect walls and avoid contact with barriers. By continuously measuring the distance to obstacles in real-time, the VL53L0X allows our car to stop at the perfect position when parking, and to navigate tight spaces with confidence. Its fast response and precise readings have made it the ideal solution for obstacle avoidance and parking maneuvers, boosting our robot’s performance and reliability compared to previous approaches.


### Raspberry Pi Description

![Raspberry Pi](./images/rpi.jpg)

The Raspberry Pi is a compact, affordable computer designed for learning, prototyping, and experimentation. Despite being about the size of a credit card, it functions as a full-fledged computer capable of running complex programs, handling input/output operations, and interfacing with a wide variety of devices. Due to its versatility, portability, and low cost, it is widely used in electronics projects, programming, and robotics.

**Raspberry Pi in Our LEGO Car:**  
In our LEGO car system, the Raspberry Pi plays a central role by handling all core operations required for autonomous driving. It processes the video feed from the Pi Camera, enabling the car to interpret its environment, and runs the AI model responsible for making real-time decisions regarding steering and speed.

Its small form factor and low power consumption allow it to integrate seamlessly into the LEGO structure without compromising space or performance. Additionally, the Raspberry Pi’s compatibility with modules like the Sense HAT and Build HAT makes it ideal for connecting to sensors and motors in a LEGO-based system. Without the Raspberry Pi, the vehicle would not be able to process environmental data or respond to dynamic conditions, making it a vital component for the success of our autonomous LEGO car.

### Pi Camera Description

![Pi Camera](./images/picamera.jpg)

The Raspberry Pi Camera is a compact, high-definition camera module designed specifically for use with Raspberry Pi boards. It connects via the CSI (Camera Serial Interface) port and is capable of capturing high-resolution still images and video. The Pi Camera is widely used in DIY electronics, robotics, and computer vision applications due to its small size, flexibility, and support for various image processing modes. It integrates effectively with machine learning models for tasks such as object detection, tracking, and real-time video analysis.

**Pi Camera in Our LEGO Car:**  
In our LEGO car system, the Pi Camera serves as the "eyes" of the self-driving robot, delivering essential visual input to the AI model. During initial training sessions, the camera captures images of the environment, track layout, and surrounding obstacles. These images are then used to train a machine learning model using TensorFlow, enabling the system to identify patterns and make informed driving decisions.

Once the AI model is trained, the Pi Camera continues to play a vital role during autonomous operation by providing real-time visual data. This live feed is processed by the AI to determine appropriate throttle and steering commands. As such, the Pi Camera is crucial to both the training and real-time navigation phases, making it one of the key components of our LEGO-based autonomous driving platform.

### Sense HAT Description

![Sense HAT](./images/sense_hat.jpg)

The Sense HAT is an add-on board for the Raspberry Pi that provides a suite of sensors and a programmable LED matrix. It includes sensors for temperature, humidity, barometric pressure, and orientation (via gyroscope, accelerometer, and magnetometer). Originally developed for educational and space-related projects, it allows real-time environmental data collection and interactive feedback.

**Sense HAT in Our LEGO Car:**  
In our LEGO car, the Sense HAT is used primarily for motion sensing and directional feedback. By reading values from its onboard gyroscope and accelerometer, we are able to monitor the car’s orientation and rotational changes during movement. This data helps the AI determine when the car has completed certain maneuvers, such as turns or laps, and allows us to switch AI models or driving modes at the right moments. The compact design of the Sense HAT, combined with its direct compatibility with the Raspberry Pi, makes it a convenient and powerful tool for integrating real-time feedback into our autonomous vehicle.

### Build HAT Description

![Build HAT](./images/build_hat.jpg)

The Build HAT (Hardware Attached on Top) is a LEGO-compatible motor and sensor controller developed by Raspberry Pi. It allows up to four LEGO Technic motors or sensors to be controlled directly from a Raspberry Pi using the standard LPF2 connectors. The Build HAT simplifies control of LEGO elements by handling the communication and power management needed to operate motors with precision.

**Build HAT in Our LEGO Car:**  
In our LEGO car, the Build HAT is used to control the car’s movement by managing the LEGO motors responsible for throttle and steering. It interfaces seamlessly with the Raspberry Pi and enables precise motor control through Python code. The Build HAT eliminates the need for third-party servo controllers or custom circuits, providing a clean and reliable solution for motor management. Its native support for LEGO components makes it an essential bridge between our AI control system and the physical LEGO hardware, ensuring smooth and responsive driving behavior during autonomous runs.

---

## Component Cost Table <a name="component-cost-table"></a>

Below is a chart of estimated costs for each major component, with purchase links (Amazon or manufacturer):

| Component                  | Quantity | Estimated Price (USD) | Purchase Link |
|----------------------------|----------|-----------------------|--------------|
| Raspberry Pi 4             | 1        | $50                   | [Amazon](https://www.amazon.com/dp/B07TD42S27) |
| Pi Camera v2               | 1        | $25                   | [Amazon](https://www.amazon.com/dp/B01ER2SKFS) |
| Sense HAT                  | 1        | $40                   | [Amazon](https://www.amazon.com/dp/B01EGB0H7O) |
| Build HAT                  | 1        | $35                   | [Pi Shop](https://www.raspberrypi.com/products/build-hat/) |
| LEGO DC Motor              | 1        | $15                   | [LEGO Shop](https://www.lego.com/en-us/product/m-motor-8883) |
| LEGO Servo                 | 1        | $20                   | [LEGO Shop](https://www.lego.com/en-us/product/servo-motor-88004) |
| 7.4V LiPo Battery          | 1        | $18                   | [Amazon](https://www.amazon.com/dp/B07KZ8YQJH) |
| Power Bank (5V/2.5A)       | 1        | $20                   | [Amazon](https://www.amazon.com/dp/B07QK2SPP7) |
| LEGO Chassis Parts         | -        | $50                   | [LEGO Shop](https://www.lego.com/en-us/themes/technic) |
| Miscellaneous Wires        | -        | $10                   | [Amazon](https://www.amazon.com/dp/B07GD2BWPY) |

**Total Estimated Cost:** ~$283



---

## Assembly Instructions <a name="assembly-instructions"></a>

![LEGO Chassis](./images/chassis.jpg)
![3D Model](./images/3d_model.png)

To facilitate the replication and further development of our Autonomous Robotic Vehicle (ARV), we have provided detailed assembly instructions that guide builders through every step of the construction process. In addition, we have produced a video that gives an overview of our ARV's components, which is available on YouTube.

We hope to share these resources for the use of educational institutions and enthusiasts alike, with the aim of fostering a collaborative and innovative environment in the field of robotics technology.

---

## Challenges and Solutions <a name="challenges-and-solutions"></a>

Throughout the development of this project, we encountered a series of challenges. First, there was the issue of image resolution. When using larger image sizes, the processing demands increased significantly, causing training sessions to last over 8 hours. To resolve this, we downscaled the input data and selected a resolution of 160 by 120 pixels to optimize training efficiency.

Next, we faced difficulties with overly lengthy road models. To make training more effective, we segmented portions of the data model and trained them in smaller batches. Lastly, we ran into software compatibility issues. For example, models trained using TensorFlow 2.9 were not compatible with older versions like TensorFlow 2.4. To address this, we standardized all systems to run TensorFlow 2.9 for consistency and improved integration.

---

## Conclusion <a name="conclusion"></a>

Our team has employed a unique approach in the design of our intelligent vehicle, which is characterized using a purely Artificial Intelligence (AI)-driven model. At the heart of this system is a Convolutional Neural Network (CNN) running in a Linux environment, built using Google’s TensorFlow framework. The CNN is designed to learn and replicate the driving habits of a human operator by analyzing images captured by the camera and movement data from the remote control. This allows the AI model to self-train and adapt based on real-world input.

In practice, the neural network processes visual data from the course and makes predictions about optimal movements. These decisions are then communicated to the vehicle’s control system — in our case, the LEGO car.

Our design philosophy has been consistent from the start: to integrate the strengths of traditional programming with modern AI. By feeding the AI a large volume of real driving data, it can make intelligent predictions while also learning the operator’s behavior, resulting in a more human-like and adaptable driving system. The vehicle is not just mechanically responsive; it is guided by an intelligent system that blends precision control with learned behavior for a more sophisticated autonomous experience.

---
