FVJ WRO 2025 - Future Engineers

FVJ Team Members

Coach Fei Guo

Coach Fei Guo (Robert Guo)  provides the technical guidance and leadership required to keep Team FVJ on track. With extensive experience in robotics, he helps us navigate complex challenges, refine our designs, and develop solutions that work in competitive and real-world settings.

Jiapeng Xu  

Jack is a sophomore at Rancho Cucamonga High School with previous experience in the Robo Mission category.. He contributes to the AI model development, focusing on optimizing the system to improve the car’s decision making abilities.

Kyle


Project Overview: AI Self Driving Car

At the heart of our project is a self-driving car that leverages artificial intelligence to navigate autonomously. The key to its functionality lies in the integration of machine learning and deep learning techniques through the TensorFlow framework, running on a Raspberry Pi. The car's AI model is trained to interpret image data and make real time decisions for navigation, all based on patterns learned from its training data.

 Core Technology Components

1. TensorFlow: The primary framework used to build the AI model. TensorFlow allows us to develop a Convolutional Neural Network (CNN), which forms the foundation for the car’s ability to recognize patterns in images and predict appropriate driving actions.
   
2. Convolutional Neural Networks: The AI system relies on a neural network architecture, particularly a CNN, to process image inputs and predict the best course of action. Neural networks simulate the way the human brain processes visual and sensory data, which is critical for enabling the car to "learn" from training data.

4. Python: The coding language used for the entire project. Python’s extensive libraries and compatibility with TensorFlow make it ideal for AI development. All model training, data processing, and decision making algorithms are written in Python.

5. Raspberry Pi: The AI model runs on a Raspberry Pi system. This small, yet powerful, computer provides the necessary computing power to execute TensorFlow models while maintaining the flexibility and compatibility required for robotics projects.

6. Image Training: The AI model is trained using images collected from the car’s environment. These images are analyzed during the training phase, with the model learning to recognize important features such as road boundaries, obstacles, and turns. The more diverse the image dataset, the better the model becomes at predicting the necessary actions in different scenarios.


Overview of PWM
Pulse Width Modulation (PWM) is a technique used to control the speed and direction of the motors that drive the wheels of the car. In our self-driving car, PWM is essential for achieving precise control over acceleration and steering, enabling smooth and efficient movement. PWM operates by rapidly switching the power supplied to the motors on and off, with the ratio of "on" time to "off" time determining the effective power delivered.

Frequency: The number of times per second the PWM signal repeats. A higher frequency provides finer control, ensuring smoother transitions in motor speed.

How PWM is Used in Our Project
For our self-driving car, PWM is used to control both the DC motors that drive the wheels and the servo motors responsible for steering. The Raspberry Pi sends PWM signals to the motor controller, which then adjusts the voltage to the motors based on the desired speed or steering angle.

Steering and Throttle Control: The car’s steering mechanism is controlled by a servo motor, which adjusts the front wheels' angle. The PWM signal sent to the servo determines how much the wheels turn left or right, allowing the car to navigate curves and make sharp turns. It can move left, right, backwards, and forwards. The smooth control provided by PWM is crucial for ensuring that the car responds accurately to the predictions made by the AI model, translating neural network outputs into physical actions in real-time.

Role of the Gyroscope
The gyroscope plays a key role in tracking the orientation and movement of the self-driving car. Specifically, the WT901C gyroscope is used in our project to measure angular velocity and detect changes in the car’s orientation across multiple axes (pitch, roll, and yaw). This information is critical for real-time adjustments to the car’s path, ensuring it stays balanced and accurately follows its intended course. The gyroscope helps locate the car on the filed, and determine the position when turnings, going back, and parking.

Feedback Loop: The gyroscope works in tandem with the AI model, providing feedback that complements the camera's visual data. While the camera focuses on image-based navigation, the gyroscope ensures the car maintains proper balance and orientation, preventing it from veering off course due to physical forces like momentum or incline.

Obstacle Navigation: During obstacle avoidance, the gyroscope helps the AI determine how the car's orientation shifts as it navigates around objects. By analyzing these shifts, the neural network can make more informed predictions about how the car should adjust its speed and steering to navigate complex environments.

The physical base structure of our car is an RC car, but we added on to this base by designing our own lego structure. The team designed the chassis to house the key components, such as the motor, Raspberry Pi, gyroscope, and wiring, ensuring proper balance and weight distribution. The front wheels are connected to a servo motor to enable precise control of steering angles, while the rear wheels are powered by DC motors for propulsion.

The car uses a combination of DC motors for movement and a servo motor for steering:

These are connected to the rear wheels and controlled using Pulse Width Modulation (PWM) signals. The power delivered to the motors varies depending on the speed the car needs to achieve. For example, lower PWM duty cycles reduce speed, while higher duty cycles deliver more power for acceleration.
  
The servo motor controls the angle of the front wheels, allowing for precise turns. It operates based on a range of motion determined by the PWM signal sent from the Raspberry Pi, which is determined by the AI model's predictions.

At the heart of the car’s autonomous driving capabilities is a Convolutional Neural Network (CNN) developed using TensorFlow. This deep learning framework is essential for training the AI to recognize driving patterns, road boundaries, and obstacles based on visual data collected by the car’s camera.

The CNN processes input images captured by the camera mounted on the car. Each layer of the CNN extracts features from the images, starting with basic edges and progressing to more complex features such as shapes and objects. These features are then passed through multiple layers, allowing the network to learn important aspects of the driving environment (such as turns, obstacles, or road boundaries).

The CNN is trained using a dataset of images captured during the manual driving phase (discussed in detail later). During training, the network adjusts its internal weights through backpropagation which minimizes prediction errors by iteratively improving the network’s ability to associate input images with driving decisions.

Once trained, the CNN processes live input from the car’s camera and predicts appropriate actions such as accelerating, turning, or stopping. These predictions are then used to generate PWM signals that control the motors.

The core AI model operates in real-time, continuously processing visual data from the camera:

Before being fed into the neural network, the captured images are resized and normalized. This step is important because it ensures that the data fed into the neural network is consistent, reducing the computational load and improving the accuracy of predictions.

The AI model is trained to map each image frame to a specific driving action. For example, if the network detects a left curve in the image, it will predict a left turn, which is then executed by adjusting the servo motor accordingly.


The Raspberry Pi acts as the brain of the car, running the AI model and handling all communication between the sensors, motors, and other hardware components.

The car uses a Raspberry Pi 4 with the Bullseye upgrade. The Raspberry Pi has enough computational power to run the TensorFlow model in real-time while processing data from sensors and controlling the motors via PWM signals.
  
The Raspberry Pi’s General-Purpose Input/Output (GPIO) pins are used to interface with the motor controller, gyroscope, and servo motor. The GPIO pins allow for direct control of the PWM signals that regulate motor speed and steering.

During the training phase, the Raspberry Pi stores image data on an SD card. This data is later used to train the neural network offline.


The software running on the Raspberry Pi is written in Python. Python is chosen for its extensive library support, including TensorFlow, and its ease of integration with hardware components.

TensorFlow is used to create and run the neural network models. The Python code loads the trained model onto the Raspberry Pi, processes input images, and uses the model to generate predictions.
  
Additional Python libraries are used to handle PWM signal generation and motor control. For example, the RPi.GPIO library is used to interact with the GPIO pins and control motor speed and direction.

The gyroscope works in conjunction with the AI model to form a feedback loop:

If the gyroscope detects that the car is tilting or rotating unexpectedly, the AI model can use this information to adjust motor power or steering angles, bringing the car back to a stable position.

The gyroscope’s data helps ensure smooth driving, especially when the car encounters obstacles or makes sudden turns. For example, if the car begins to tilt while turning, the gyroscope detects this and sends the data to the AI model, which can then adjust the steering angle or motor power to stabilize the vehicle.


The first step in training the AI model is collecting data through manual driving. This involves driving the car manually around a demo map that includes both clear roads and obstacles.

As the car is driven manually, the camera continuously captures images of the environment. These images are saved to the Raspberry Pi’s storage, creating a dataset that will later be used to train the AI model.

Each image is associated with a specific action, such as turning left, turning right, or accelerating. This labeled data forms the basis of the training dataset.

Once the dataset has been collected, we train the Convolutional Neural Network (CNN) to recognize patterns in the images and predict the appropriate driving actions:

The collected images are split into a training set and a validation set. The training set is used to adjust the model’s internal parameters, while the validation set ensures that the model generalizes well to new data.

The CNN is trained using backpropagation, a technique that allows the network to learn from its mistakes. During each iteration, the network adjusts its internal weights to reduce the difference between its predictions and the correct driving actions.


After training, the AI model is tested on unseen data to evaluate its accuracy and performance:

The model is evaluated using the validation set, ensuring that it performs well on data it has not seen during training. This step is critical for preventing overfitting, where the model performs well on the training data but poorly on new data.

Once the model performs well on the validation set, it is tested in real-world conditions. The car is driven autonomously on the demo map, and the AI’s predictions are compared to the actual driving actions.

The car uses its camera and AI model to detect obstacles in the environment. During training, the model is taught to recognize objects in the car’s path and take appropriate actions to avoid them.

The CNN processes images captured by the camera and identifies obstacles. The AI then decides whether to turn, slow down, or stop based on the location and size of the obstacle.
  
The AI model predicts the safest course of action by analyzing the obstacle’s distance and direction relative to the car. It then adjusts the steering and motor power accordingly, allowing the car to navigate around the object without colliding.

The car is tested on a predefined demo map that includes straight paths, curves, and obstacles. This provides a controlled environment for evaluating the car’s ability to navigate different terrains and avoid obstacles.

In more complex environments, the car must deal with moving obstacles, uneven surfaces, and unpredictable conditions. These challenges require the AI model to make quick decisions based on real-time data, and the integration of PWM and gyroscope feedback ensures the car remains stable during these maneuvers.

 AI Training and Model Development

The training process for our self-driving car's AI model involves a multistep approach designed to simulate real world driving conditions and improve the car's ability to make autonomous decisions. Below is a breakdown of our process:

 1. Preparing the Raspberry Pi  
First, we power up the Raspberry Pi by connecting it to an external power source. We then insert the training SD card, which contains the necessary programs and data collection tools. The training SD card is critical for gathering the image data used to train the neural network.

 2. Establishing a Remote Connection  
We remotely control the Raspberry Pi by opening VNC Viewer on a laptop, allowing us to manage the system and monitor the training process without physically accessing the Raspberry Pi. VNC Viewer lets us execute commands and run the training programs directly on the Raspberry Pi’s operating system.

 3. Data Collection via Remote Control  
Using a remote controller, we manually drive the car on a demo map that simulates a variety of driving conditions, including straight roads, turns, and obstacles. During this process, the car captures images from its surroundings, which are stored for later use in training the neural network.

 4. Image Collection and Data Labeling  
The images collected from the car’s manual driving sessions are saved to the Raspberry Pi’s storage. Along with the images, we record the corresponding actions (e.g., turning left, turning right, accelerating, stopping), creating a dataset that maps visual inputs to driving decisions. This dataset forms the foundation for training the AI model.

 5. Training the AI Model  
We then use TensorFlow to develop a Convolutional Neural Network (CNN) that processes the collected images. The goal of the training process is for the neural network to identify the key features in the images (such as road edges or obstacles) and predict how the car should respond. This involves repeated training cycles where the network adjusts its internal parameters to reduce prediction errors.

 6. Model Optimization  
Once the initial training phase is complete, we finetune the model by running additional training sessions with varied data, ensuring that the neural network becomes more accurate and responsive. The optimization process also includes adjusting hyperparameters such as learning rates, batch sizes, and layer structures to improve the model’s overall performance.

 7. Deployment of the Trained Model  
After the AI model has been sufficiently trained, we swap out the training SD card for the driving SD card. This card contains the finalized model and the programs required to run the car in autonomous mode. When the program is executed, the AI model processes real time images from the car's environment, using its training to predict the best driving actions.

 8. Running the Autonomous Driving Program  
With the trained model running on the Raspberry Pi, the car now operates autonomously. The AI system continuously processes images from the car’s camera, making real time predictions about the car’s next move. The car navigates its environment based entirely on the patterns learned during the training phase, without any human intervention.


AI Model and Neural Network Components

1. Convolutional Neural Network (CNN): The CNN is specifically designed to process image data and identify patterns in visual information. Each layer of the CNN extracts increasingly complex features, from basic edges to higher level concepts like objects and road boundaries, allowing the model to predict driving actions based on the input images.

2. Back propagation and Training: During the training process, the AI model compares its predictions to the correct driving actions and calculates the error. Backpropagation enables the network to learn by minimizing these errors through repeated iterations, improving its accuracy.

3. Real Time Decision Making: Once deployed, the neural network runs in realtime, processing new images as they are captured by the car's camera. It uses the trained model to predict whether it should move left or right based on the color of the obstacle.
4. Image Augmentation: To enhance the robustness of the model, we also use image augmentation techniques during training. This involves creating variations of the original images (such as adjusting brightness, contrast, or orientation) to simulate a wider range of driving conditions. This allows the neural network to perform well even in unfamiliar or challenging environments.

Other Key Components
1. Mobility Management
Vehicle Design and Motor Control:
Our vehicle utilizes a custom-built RC car platform, which incorporates a DC motor to control throttle and a servo for steering. The DC motor was chosen for its ability to provide smooth acceleration and consistent speed, while the servo offers precise steering control. Both components are managed by a custom-designed ESC (Electronic Speed Controller), which interfaces directly with the Raspberry Pi to allow for dynamic control.
AI-Controlled Mobility:
To enable autonomous operation, we integrated a Pi Camera to feed real-time visual data to an AI model. The model predicts the appropriate throttle and steering adjustments based on obstacle detection and navigation requirements. The AI model was trained using data collected from manual control, where the car was driven and the corresponding throttle and steering values were logged alongside images captured by the Pi Camera.
Challenges in Mobility Management:
One of the challenges faced during development was ensuring the AI model could accurately distinguish between obstacles with similar shapes. We overcame this by incorporating color recognition, which allowed the AI to make more precise decisions. Another challenge was syncing the throttle and steering inputs in real-time during training, especially when switching between AI models for different control tasks.


2. Power and Sense Management
Power Source and Sensors:
The vehicle’s power is supplied by a dual-battery system. A battery bank powers the Raspberry Pi, while a separate battery provides power to the RC car platform. The Pi Camera provides visual input for the AI model, and various sensors (e.g., encoders, proximity sensors) support the vehicle's obstacle detection and decision-making. These sensors help the AI predict the necessary steering and throttle adjustments for efficient navigation.
Power Consumption Considerations:
The power consumption of the Raspberry Pi is monitored, and the battery bank ensures consistent operation of both the Pi and the sensors. A well-organized wiring system connects the power sources to the components, and an efficient power distribution setup minimizes energy loss during operation.


3. Obstacle Management
Obstacle Detection and Strategy:
Our vehicle uses a combination of AI driven image processing and sensor input from our Pi Camera to detect and navigate obstacles. The Pi Camera captures live images, which are processed by the AI model to predict the appropriate throttle and steering commands. This real-time processing allows the car to adapt to different obstacles by adjusting speed and direction.
Obstacle Handling Strategy:
The obstacle handling strategy was developed through a process of trial and error. Initially, the AI model struggled to differentiate between obstacles based on shape alone, so color differentiation was integrated into the training process. Additionally, detailed comments were created on each code that was created to handle these strategies; they can be viewed in the src files on Github.
Challenges and Solutions:
A significant challenge was ensuring the AI could process the image data fast enough to make timely adjustments. We addressed this by optimizing the image recognition algorithm and fine-tuning the throttle/steering prediction model. Real-time data handling and efficient algorithm implementation were key to improving obstacle navigation.

Obstacles 
During the development of our AI self-driving car, we hit a major problem when we accidentally burned out the Integrated Circuit (IC) that controlled the motors. The IC is a crucial part because it helps send signals from the Raspberry Pi to the motors, allowing us to control how fast the car moves and in which direction. At first, everything was running smoothly, but after testing the car with more demanding tasks, the IC overheated and stopped working.

We realized something was wrong when the car suddenly stopped moving, even though the Raspberry Pi was still sending commands. After checking, we found that the IC had burned out due to overheating. This happened because the motors were drawing too much power for too long, and the IC wasn’t equipped to handle it. We also hadn’t added any cooling to prevent it from getting too hot, which made the problem worse.

To fix this, we did some research and decided to replace the burned-out IC with a stronger one that could handle more power. We also added heat sinks to help keep the IC cool and installed better airflow around it. On top of that, we added a fuse to the circuit. This fuse would blow if the motors tried to pull too much power, which would stop the IC from overheating again.

After making these changes, we tested the car again under the same conditions that caused the problem, and everything worked perfectly. The new IC and cooling system kept things running smoothly, even when the car was driving for a long time. This experience taught us how important it is to test the car in different scenarios and to make sure all the components are protected against potential problems like overheating.

In the end, this issue turned out to be a valuable learning experience for us. Even though it was frustrating, it helped us understand the importance of making sure all the parts of the car are reliable and safe from damage.

Conclusion

Team FVJ’s AI powered self-driving car represents a sophisticated application of machine learning, neural networks, and real time decision making. Our project combines the power of TensorFlow, Raspberry Pi, a custom built CNN and lego structure to create a self driving system capable of navigating autonomously.

Our team has employed a unique approach in the design of our intelligent vehicle, which is characterized by the use of a pure Artificial Intelligence (AI) model. At the heart of this system is a Convolutional Neural Network (CNN) that operates under a Linux environment, utilizing Google's TensorFlow.


 The purpose of the CNN is to learn and emulate the driving habits of the operator. It does this by processing images captured by the camera and the movements of the remote control, allowing the network to train itself to respond to these inputs. In practice, the neural network analyzes images from the race, making predictions about the best course of action in a given situation. In terms of ground perception modules, our AI captures images with its camera and processes them using the CNN, which then compares them with a well-trained model to make accurate predictions.


Our design philosophy has been clear and consistent from the beginning: to merge the benefits of traditional programming with AI. By feeding the AI high quality data, it processes information and analyzes data like a brain, making predictions while also learning the driver's habits, thus humanizing the vehicle's driving behavior. Our intelligent vehicle is not merely mechanically operated; it is driven by an intelligent system that combines precise programming with an AI model for an enhanced driving experience.



