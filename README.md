FJV WRO 2024 - Future Engineers

FJV Team Members

Coach Fei Guo
Coach Fei Guo (Robert Guo)  provides the technical guidance and leadership required to keep Team FJV on track. With extensive experience in robotics, he helps us navigate complex challenges, refine our designs, and develop solutions that work in competitive and real-world settings.

FengMo Guo  

FengMo, a senior at Los Osos High School, is an experienced WRO participant with a keen interest in robotics hardware. His innovative contributions to the project ensure that the car’s structure supports the advanced capabilities of its AI system.

Jiapeng (Jack) Xu  

Jack is a sophomore at Rancho Cucamonga High School with previous experience in the Robo Mission category.. He contributes to the AI model development, focusing on optimizing the system to improve the car’s decision making abilities.

Vinaya Ayinampudi 
 
Vinaya is a junior at eSTEM Academy and is in her second year participating in the WRO Future Engineers Category internationally. She has a strong passion for AI and coding, and leads the team’s work on developing and refining the AI model for the self-driving car. 





Project Overview: AI Self Driving Car

At the heart of our project is a self-driving car that leverages artificial intelligence to navigate autonomously. The key to its functionality lies in the integration of machine learning and deep learning techniques through the TensorFlow framework, running on a Raspberry Pi. The car's AI model is trained to interpret image data and make real time decisions for navigation, all based on patterns learned from its training data.

 Core Technology Components

1. TensorFlow: The primary framework used to build the AI model. TensorFlow allows us to develop a Convolutional Neural Network (CNN), which forms the foundation for the car’s ability to recognize patterns in images and predict appropriate driving actions.
   
2. Neural Networks: The AI system relies on a neural network architecture, particularly a CNN, to process image inputs and predict the best course of action. Neural networks simulate the way the human brain processes visual and sensory data, which is critical for enabling the car to "learn" from training data.

3. Python: The coding language used for the entire project. Python’s extensive libraries and compatibility with TensorFlow make it ideal for AI development. All model training, data processing, and decision making algorithms are written in Python.

4. Raspberry Pi: The AI model runs on a Raspberry Pi system. This small, yet powerful, computer provides the necessary computing power to execute TensorFlow models while maintaining the flexibility and compatibility required for robotics projects.

5. Image Training: The AI model is trained using images collected from the car’s environment. These images are analyzed during the training phase, with the model learning to recognize important features such as road boundaries, obstacles, and turns. The more diverse the image dataset, the better the model becomes at predicting the necessary actions in different scenarios.





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


 Conclusion

Team FJV’s AI powered self-driving car represents a sophisticated application of machine learning, neural networks, and real time decision making. Our project combines the power of TensorFlow, Raspberry Pi, a custom built CNN and lego structure to create a self driving system capable of navigating autonomously.

Our team has employed a unique approach in the design of our intelligent vehicle, which is characterized by the use of a pure Artificial Intelligence (AI) model. At the heart of this system is a Convolutional Neural Network (CNN) that operates under a Linux environment, utilizing Google's TensorFlow.


 The purpose of the CNN is to learn and emulate the driving habits of the operator. It does this by processing images captured by the camera and the movements of the remote control, allowing the network to train itself to respond to these inputs. In practice, the neural network analyzes images from the race, making predictions about the best course of action in a given situation. In terms of ground perception modules, our AI captures images with its camera and processes them using the CNN, which then compares them with a well-trained model to make accurate predictions.


Our design philosophy has been clear and consistent from the beginning: to merge the benefits of traditional programming with AI. By feeding the AI high quality data, it processes information and analyzes data like a brain, making predictions while also learning the driver's habits, thus humanizing the vehicle's driving behavior. Our intelligent vehicle is not merely mechanically operated; it is driven by an intelligent system that combines precise programming with an AI model for an enhanced driving experience.

