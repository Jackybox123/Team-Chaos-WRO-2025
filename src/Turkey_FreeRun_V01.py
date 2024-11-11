'''
This is a code for the Free Run of the WRO future Engineering competition.
Functions include:
1. Detect camera images and make predictions
2. Output the predicted data to pwm9685 to control the RCcar to move forward and turn.
3. Detect gyroscope data and stop after 3 circles.
4. Detect the status of the GPIO pin to start the vehicle moving.
5. Output the status of the program running to the LCD
'''
import os
import time
import sys
import smbus2
from PCF8574 import PCF8574_GPIO
from Adafruit_LCD1602 import Adafruit_CharLCD
from donkeycar.parts.camera import PiCamera
import cv2
from donkeycar.parts.interpreter import KerasInterpreter
from donkeycar.parts.keras import KerasLinear
import Adafruit_PCA9685
import pygame
import numpy as np
import RPi.GPIO as GPIO


'''
Very important variables
'''
# useing the free run ai model!!!
current_path = '/home/pi/airc_drive'
sys.path.append(current_path)
model_name = "/models23occw/mypilot.h5"

#3 circles stop at Gyro sensor Yaw degree
stop_degree = 1080
# PWM RC car control set data: setup PMW "stop" and "central" data
default_servo_signal = 350
default_servo_signal_offset = 27
servo_signal_scale = 150
default_servo_steering_signal = 350
default_servo_steering_signal_offset = 140
servo_steering_signal_scale = -250
#=======================================================================================

'''
init GYRO
'''
# I2C address of the WT901 sensor (default is 0x50)
DEVICE_ADDRESS = 0x50
# Gyro Z register
GYRO_Z_REGISTER = 0x39  # Example register for Z-axis gyroscope
# Initialize the I2C bus
bus = smbus2.SMBus(1)
# Variable to store the yaw angle
yaw_angle = 0
# Time tracking for integration
previous_time = time.time()
#gyro offset
gyro_offset = 1.008
#=========================================================================================

'''
init GPIO pin
'''
# Set the GPIO mode
GPIO.setmode(GPIO.BCM)  # Use BCM GPIO numbering
gpio_Num = 25 
GPIO.setup(gpio_Num, GPIO.IN,GPIO.PUD_UP)


"""
init pwm9685
"""
# Alternatively specify a different address and/or bus:
pwm = Adafruit_PCA9685.PCA9685(address=0x40, busnum=1)
# Configure min and max servo pulse lengths
# Helper function to make setting a servo pulse width simpler.
def set_servo_pulse(channel, pulse):
    pulse_length = 1000000    # 1,000,000 us per second
    pulse_length //= 60       # 60 Hz
    print('{0}us per period'.format(pulse_length))
    pulse_length //= 4096     # 12 bits of resolution
    print('{0}us per bit'.format(pulse_length))
    pulse *= 1000
    pulse //= pulse_length
    pwm.set_pwm(channel, 0, pulse)
# Set frequency to 60hz, good for servos.
pwm.set_pwm_freq(60)
# ====================================================================================


"""
init camera
from dondeycar/parts/camera
"""
IMAGE_W = 160
IMAGE_H = 120
IMAGE_DEPTH = 3
cam = PiCamera(image_w=IMAGE_W, image_h=IMAGE_H, image_d=IMAGE_DEPTH,
                 vflip=False, hflip=False)
#======================================================================================

"""
init tf and models
"""
model_path= current_path + model_name
input_shape = (120, 160, 3)
model_type = 'linear'
interpreter = KerasInterpreter()
kl = KerasLinear(interpreter=interpreter, input_shape=input_shape)
kl.load(model_path)
#======================================================================================

"""
init color detect 
"""
# Define color ranges for red and green in HSV color space
lower_red1 = np.array([0, 120, 70])
upper_red1 = np.array([10, 255, 255])
lower_green = np.array([36, 100, 100])
upper_green = np.array([86, 255, 255])
lower_red2 = np.array([170, 50, 50])
upper_red2 = np.array([180, 255, 255])
red_flag = 0
green_flag = 0
#==========================================================================================

'''
init LCD1602
'''
PCF8574_address = 0x27  # I2C address of the PCF8574 chip.
mcp = PCF8574_GPIO(PCF8574_address)
# Create LCD, passing in MCP GPIO adapter.
lcd = Adafruit_CharLCD(pin_rs=0, pin_e=2, pins_db=[4,5,6,7], GPIO=mcp)
mcp.output(3,1)     # turn on LCD backlight
lcd.begin(16,2)     # set number of LCD lines and columns
#===========================================================================================

'''
Define Function Program
'''
#Color detection program via saturation mask
def color_predict(frame):
    image = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
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
        red_flag = 1
        print("Red detected")
    elif green_count > red_count and green_count > 1000:
        red_flag = 2
        print("Green detected")
    else:
        red_flag = 0
#         print("No red or green detected")
    
    return red_flag,
#=========================================================================================

# read Gyro sensor z degree per second data
def read_gyro_z():
    # Read two bytes from the gyro Z register
    data = bus.read_i2c_block_data(DEVICE_ADDRESS, GYRO_Z_REGISTER, 2)
    # Combine the two bytes into a 16-bit value
    gyro_z_raw = data[1] << 8 | data[0]
    
    # Convert the raw gyroscope value to angular velocity in degrees per second
    # (You might need to adjust the scale depending on your gyro's sensitivity)
    if gyro_z_raw > 32767:
        gyro_z_raw -= 65536
    gyro_z = gyro_z_raw / 32768.0 * 2000  # Assuming +/- 2000 dps range, adjust if needed
    return gyro_z
#========================================================================================


'''
run Main CODE
'''

if __name__ == '__main__':
    Run_main = True
    pygame.init()
    screen = pygame.display.set_mode((1,1),pygame.NOFRAME)
    #setup steering and Throttle 
    servo_steering_signal= 0
    servo_signal= 0
    # setup car status
    # car_status forward:"1" ; backward:"-1" ; stop:"0"
    car_status_now = int(0)
    car_status_predict = int(0)
    #====================================================================================
    #display ai model name
    lcd.clear()
    for i in range(0,6):                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
        lcd.setCursor(0,0)  # set cursor position
        lcd.message( 'ai model name'+'\n' )# display
        lcd.setCursor(0,1)  # set cursor position
        lcd.message( model_name+'\n' )# display
        time.sleep(1)
        lcd.clear()
    
    #Make sure the Start switch is off
    lcd.clear()
    while GPIO.input(gpio_Num) == 0:
        lcd.setCursor(0,0)  # set cursor position
        lcd.message( 'Turn off PGIO'+'\n' )# display
        lcd.setCursor(0,1)  # set cursor position
        lcd.message( 'Turn on Car'+'\n' )# display
        time.sleep(0.1)
    time.sleep(3)
    
    #setup lcd
    lcd.clear()
    lcd.setCursor(0,0)  # set cursor position
    lcd.message( 'pwm 350'+'\n' )# display 
    lcd.setCursor(0,1)  # set cursor position
    lcd.message( 'wait for start'+'\n' )# display 

    while GPIO.input(gpio_Num) ==1:
        servo_signal = default_servo_signal
        servo_steering_signal = default_servo_steering_signal
        print("run_turn_to_forward throttle PWM = servo_signal", servo_signal)
        pwm.set_pwm(0, 0, servo_signal)
        pwm.set_pwm(1, 0, servo_steering_signal)

        time.sleep(0.1)
    #====================================================================
    # clear lcd
    lcd.clear()
    # Variable to store the yaw angle
    yaw_angle = 0
    # Time tracking for integration
    previous_time = time.time()
    #====================================================================
    
    while Run_main:
        
        # Read the current gyroscope value (Z-axis)
        gyro_z = read_gyro_z()
        if gyro_z > 0:
            gyro_z = gyro_z * gyro_offset
        # Get the current time
        current_time = time.time()
        # Calculate time difference (delta t)
        delta_t = current_time - previous_time
        # Update the yaw angle using gyro data (angular velocity * time)
        yaw_angle += (0-gyro_z) * delta_t
        # Update previous time for the next iteration
        previous_time = current_time
        # Print the current yaw angle
        print(round(yaw_angle,2))
#        print("Yaw Angle: {:.2f} degrees".format(yaw_angle))  
        lcd.setCursor(0,0)  # set cursor position
        lcd.message( 'Gyro degree:'+'\n' )# display
        lcd.setCursor(0,1)  # set cursor position
        lcd.message( str(round(yaw_angle,2))+'\n' )# display
        if abs(yaw_angle) > stop_degree or GPIO.input(gpio_Num) ==1:
            Run_main = False

        frame = PiCamera.run(cam)
        
        #print(color_predict(frame))
#         frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
        outputs = KerasLinear.run(kl,img_arr = frame)
        servo_steering_signal = round(outputs[0],2)
        servo_signal = round(outputs[1],2)
        
        print("steering=",servo_steering_signal,"throttle=",servo_signal)


        #Prepare the PMW data
        servo_signal = int(servo_signal*servo_signal_scale+ default_servo_signal)
        if servo_signal > default_servo_signal+ default_servo_signal_offset:
            servo_signal = int(default_servo_signal+ default_servo_signal_offset)
        if servo_signal < default_servo_signal- default_servo_signal_offset:
            servo_signal = int(default_servo_signal- default_servo_signal_offset)
            

        servo_steering_signal = int(servo_steering_signal*servo_steering_signal_scale+ default_servo_steering_signal)
        if servo_steering_signal > default_servo_steering_signal + default_servo_steering_signal_offset:
            servo_steering_signal = int(default_servo_steering_signal + default_servo_steering_signal_offset)
        if servo_steering_signal < default_servo_steering_signal - default_servo_steering_signal_offset:
            servo_steering_signal = int(default_servo_steering_signal - default_servo_steering_signal_offset)

        #output to 9685    
        pwm.set_pwm(1, 0, servo_steering_signal)
        pwm.set_pwm(0, 0, servo_signal)
        print("streeing PWM = ",servo_steering_signal,"          throttle PWM = ",servo_signal)

        car_status_now = car_status_predict # renewal car status

        time.sleep(0.1)


        #cv2.imshow('Camera Feed', frame)
         # Break the loop on 'q' key press
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    Run_main = False
    
    print("break!")
    PiCamera.shutdown(cam)
    pygame.quit()

    lcd.setCursor(0,0)  # set cursor position
    lcd.message( 'stop:'+'\n' )# display
    lcd.setCursor(0,1)  # set cursor position
    lcd.message( 'pwm 350'+'\n' )# display

    for i in range(0,20):                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
        servo_signal = default_servo_signal
        print("run_turn_to_forward throttle PWM = servo_signal", servo_signal)
        pwm.set_pwm(0, 0, servo_signal)
    #             print("i=",i)
        time.sleep(0.05)
    
    time.sleep(5)
    lcd.clear()
