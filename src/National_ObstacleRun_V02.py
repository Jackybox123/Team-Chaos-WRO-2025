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
#run direction 'ccw' or cw
run_direction_ccw = True

#run the red detect, yes is True, no False
run_red_detect = True #False
##run parking, yes is True, no False
run_parking = True
# useing the free run ai model!!!

current_path = '/home/pi/airc_drive'
sys.path.append(current_path)
obstacle_ccw_model = '/models30occw/mypilot.h5'
obstacle_cw_model = '/models30occw/mypilot.h5'
parking_ccw_model = '/models30occw/mypilot.h5'
parking_cw_model = '/models30occw/mypilot.h5'

if run_direction_ccw :
    obstacle_model_name = obstacle_ccw_model
    obstacle_Reverse_model_name = obstacle_cw_model
    parking_model_name = parking_ccw_model

else:
    obstacle_model_name = obstacle_cw_model
    obstacle_Reverse_model_name = obstacle_ccw_model
    parking_model_name = parking_cw_model

if run_direction_ccw :
    run_direction = 'OCCW'
else:
    run_direction = 'OCW'

#3 circles stop at Gyro sensor Yaw degree
stop_degree = 1080
#gyro offset
gyro_offset = 3.95
#set color detect degree
detect_color_degree_lower = stop_degree-360-160
detect_color_degree_uper = stop_degree-360-80
# PWM RC car control set data: setup PMW "stop" and "central" data
default_servo_signal = 350
default_servo_signal_offset = 22.5
servo_signal_scale = 150
default_servo_steering_signal = 350
default_servo_steering_signal_offset = 140
servo_steering_signal_scale = -250
# set stop delate timer
stop_delate_time = 1.5
stop_timer = 0
#=======================================================================================

'''
init GYRO
'''
# I2C address of the WT901 sensor (default is 0x50)
DEVICE_ADDRESS = 0x50
# Register for yaw data
YAW_REGISTER = 0x3F
# Initialize the I2C bus
bus = smbus2.SMBus(1)
# Variable to store the previous yaw value
previous_yaw = None
# Variable to store the cumulative yaw value
cumulative_yaw = 0

final_gyro_degree = 0
final_gyro_degree_1 = 0
last_cumulative_yaw = 0
gyro_degree_list = [0,0,0,0,0,0,0,0,0,0]
list_n=0

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
model_path= current_path + obstacle_model_name
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
lower_red = np.array([165, 150, 100])  # Lower range for red (Hue: 170-179, Saturation: 120-255, Value: 100-255)
upper_red = np.array([179, 255, 255])  # Upper range for red (Hue: 170-179, Saturation: 120-255, Value: 100-255)
# Green Range: Hue 30-80 (adjust based on your preference)
lower_green = np.array([35, 50, 50])  # Lower hue for green (from 30 to 40) and lower saturation/brightness
upper_green = np.array([85, 255, 255])  # Upper hue for green (from 40 to 80) and full saturation/brightness
red_flag = 0
green_flag = 0
red_flag_T = 0
green_flag_T = 0
red_flag_timer = 0
green_flag_timer = 0
red_counter = 0
green_counter = 0
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
    mask_red = cv2.inRange(hsv_image, lower_red, upper_red)
    mask_green = cv2.inRange(hsv_image, lower_green, upper_green)

    # Count non-zero pixels in each mask to detect the presence of color
    red_count = cv2.countNonZero(mask_red)
    green_count = cv2.countNonZero(mask_green)
#     print('red_count=',red_count)
#     print('green_count=',green_count)

    if red_count > 100:  # Threshold to avoid small detections
        red_flag = 1
    else:
        red_flag = 0

    if green_count > 100:
        green_flag = 1
    else:
        green_flag = 0

    
    return red_flag, green_flag
#=========================================================================================

def read_yaw():
    # Read two bytes from the yaw register
    data = bus.read_i2c_block_data(DEVICE_ADDRESS, YAW_REGISTER, 2)
    # Combine the two bytes into a 16-bit value
    yaw_raw = data[1] << 8 | data[0]
    
    # Convert the raw yaw value to degrees
    if yaw_raw > 32767:
        yaw_raw -= 65536
    yaw = yaw_raw / 32768.0 * 180  # Convert to degrees
    return yaw

def read_cumulative_yaw():
    global previous_yaw
    global cumulative_yaw
    global last_cumulative_yaw
    global list_n
    
    current_yaw = read_yaw()
    if previous_yaw is not None:
        # Calculate the change in yaw
        delta_yaw = current_yaw - previous_yaw
        # Handle the wrapping from +180 to -180 and vice versa
        if delta_yaw > 180:
            delta_yaw -= 360
        elif delta_yaw < -180:
            delta_yaw += 360

        # Add the change to the cumulative yaw
        cumulative_yaw = (cumulative_yaw + delta_yaw)*0.25 + last_cumulative_yaw*0.75
        last_cumulative_yaw = cumulative_yaw
    # Update the previous yaw value for the next iteration
    previous_yaw = current_yaw
#     gyro_degree_list[list_n] = round(0-cumulative_yaw*gyro_offset,0)
#     list_n +=1
#     if list_n ==10:
#         list_n =0
    
#    return round(0-cumulative_yaw*gyro_offset,0)
    return round(0-cumulative_yaw*gyro_offset,0)#,np.mean(gyro_degree_list)
#==================================================================================
def display_lcd(line1,line2):
    lcd.clear()
    lcd.setCursor(0,0)  # set cursor position
    lcd.message( line1 +'\n' )# display
    lcd.setCursor(0,1)  # set cursor position
    lcd.message( line2 +'\n' )# display
    print('lcd show: '+'lcd show:'+ line1 +'\n'+ line2)    


def output_PWM(steering_data,servo_data):
    #Prepare the PMW data
    servo_steering_signal = int(steering_data*servo_steering_signal_scale+ default_servo_steering_signal)
    if servo_steering_signal > default_servo_steering_signal + default_servo_steering_signal_offset:
        servo_steering_signal = int(default_servo_steering_signal + default_servo_steering_signal_offset)
    if servo_steering_signal < default_servo_steering_signal - default_servo_steering_signal_offset:
        servo_steering_signal = int(default_servo_steering_signal - default_servo_steering_signal_offset)

    servo_signal = int(servo_data*servo_signal_scale+ default_servo_signal)
    if servo_signal > default_servo_signal+ default_servo_signal_offset:
        servo_signal = int(default_servo_signal+ default_servo_signal_offset)
    if servo_signal < default_servo_signal- default_servo_signal_offset:
        servo_signal = int(default_servo_signal- default_servo_signal_offset)

    #output to 9685    
    pwm.set_pwm(1, 0, servo_steering_signal)
    pwm.set_pwm(0, 0, servo_signal)
    #print("streeing PWM = ",servo_steering_signal,"          throttle PWM = ",servo_signal)

def output_stop():
    print('putput_stop,pwm 350, stop 0.8s')
    for i in range(0,8):                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
        output_PWM(0,0)
        time.sleep(0.1)    


def run_u_turn():
    car_status = 0
    print('start u turn')
    u_turn_start_degree = read_cumulative_yaw()
    final_gyro_degree = read_cumulative_yaw()
    while abs(abs(final_gyro_degree)- abs(u_turn_start_degree)) < 180 :
        final_gyro_degree = read_cumulative_yaw()

        if car_status == 0:
            servo_steering_signal = -1
            servo_signal = 0
        if car_status == 1:
            servo_steering_signal = -1
            servo_signal = 1
        if car_status == 2:
            servo_steering_signal = 1
            servo_signal = 0
        if car_status == 3:
            servo_steering_signal = 1
            servo_signal = -1
        if car_status == 4:
            servo_steering_signal = 1
            servo_signal = 0
        if car_status == 5:
            servo_steering_signal = 1
            servo_signal = -1

        #Prepare the PMW data
        output_PWM(servo_steering_signal,servo_signal)
        

        if car_status == 1 :
            time.sleep(0.5)
        elif car_status == 5:
            time.sleep(0.8)
        else:
            time.sleep(0.2)
        
        car_status = car_status + 1    
        if car_status ==6:
            car_status = 0
    print('finish a u turn')




#=================================================================
#=================================================================
#=================================================================
'''
run Main CODE
'''

if __name__ == '__main__':
    
    pygame.init()
    screen = pygame.display.set_mode((1,1),pygame.NOFRAME)
    #setup steering and Throttle 
    servo_steering_signal= 0
    servo_signal= 0
    #====================================================================================
    #display ai model name
    lcd.clear()
    for i in range(0,2):                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
        display_lcd(run_direction,obstacle_model_name)
        print('model_path=',model_path)
        time.sleep(0.5)
        lcd.clear()

    while GPIO.input(gpio_Num) == 0:
        display_lcd('Turn off PGIO','Turn on Car')
        time.sleep(0.1)
    time.sleep(3)
    
    display_lcd('pwm out 350','wait for start')
    while GPIO.input(gpio_Num) ==1:
        servo_signal = default_servo_signal
        servo_steering_signal = default_servo_steering_signal
        #print("run_turn_to_forward throttle PWM = servo_signal", servo_signal)
        pwm.set_pwm(0, 0, servo_signal)
        pwm.set_pwm(1, 0, servo_steering_signal)
        time.sleep(0.1)
    #====================================================================

    # Variable to store the previous yaw value
    previous_yaw = None
    # Variable to store the cumulative yaw value
    cumulative_yaw = 0
    
    Run_main = True
    while Run_main:
        
        final_gyro_degree = read_cumulative_yaw()
        print('Gyro_degree:',final_gyro_degree)

        
        if abs(final_gyro_degree) > stop_degree-360+10 or GPIO.input(gpio_Num) ==1:
            Run_main = False
            print('stop at time_end')
        if abs(final_gyro_degree) < stop_degree-360-10:
            stop_timer = time.time()
        
        if time.time()-stop_timer > stop_delate_time:
            Run_main = False
            print('stop at delate time')

        # get image from camera
        frame = PiCamera.run(cam)
        
        if abs(final_gyro_degree) > detect_color_degree_lower and abs(final_gyro_degree) < detect_color_degree_uper:
            red_flag_T, green_glag_T = color_predict(frame)
            red_counter = red_counter + red_flag_T
            green_counter = green_counter + green_glag_T
            if red_counter == 3:
                red_flag_timer = time.time()
                print(red_counter,red_flag_timer)
            if green_counter ==3:
                green_flag_timer = time.time()
                print(green_counter,green_flag_timer)

        
        outputs = KerasLinear.run(kl,img_arr = frame)
        servo_steering_signal = round(outputs[0],2)
        servo_signal = round(outputs[1],2)
        
        
        output_PWM(servo_steering_signal,servo_signal)

        time.sleep(0.01)
        #cv2.imshow('Camera Feed', frame)
         # Break the loop on 'q' key press
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    Run_main = False
    
    output_stop()
    display_lcd('STOP','pwm out 350')
    print('=============================================')
#======================================================================
#=====================================================================
#======================================================================
#detect red and make a uturn    
    if run_red_detect and red_counter > 2 and red_flag_timer > green_flag_timer:
        print('detect a red signal')

        run_u_turn()
        output_stop()
        stop_degree = 360
        # Variable to store the previous yaw value
        previous_yaw = None
        # Variable to store the cumulative yaw value
        cumulative_yaw = 0
        
        """
        init tf and models
        """
        #change parking model name
        if run_direction_ccw :
            parking_model_name = parking_cw_model
        else:
            parking_model_name = parking_ccw_model


        model_path= current_path + obstacle_Reverse_model_name
        input_shape = (120, 160, 3)
        model_type = 'linear'
        interpreter = KerasInterpreter()
        kl = KerasLinear(interpreter=interpreter, input_shape=input_shape)
        kl.load(model_path)
        #======================================================================================

#contiue ren the Third Circle
    Run_main = True
    print('model=',model_path)
    while Run_main:
        
        final_gyro_degree = read_cumulative_yaw()
        print('Gyro_degree:',final_gyro_degree)

        
        if abs(final_gyro_degree) > stop_degree+10 or GPIO.input(gpio_Num) ==1:
            Run_main = False
            print('stop at time_end')
        if abs(final_gyro_degree) < stop_degree-10:
            stop_timer = time.time()
        
        if time.time()-stop_timer > stop_delate_time:
            Run_main = False
            print('stop at delate time')

        # get image from camera
        frame = PiCamera.run(cam)
        outputs = KerasLinear.run(kl,img_arr = frame)
        servo_steering_signal = round(outputs[0],2)
        servo_signal = round(outputs[1],2)

        output_PWM(servo_steering_signal,servo_signal)

        time.sleep(0.01)
        #cv2.imshow('Camera Feed', frame)
         # Break the loop on 'q' key press
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    Run_main = False
    
    output_stop()
    display_lcd('STOP','pwm out 350')
    print('=============================================')
    


#=========================================================================
#==========================================================================
#========================================================================
#run parking
    if run_parking :
        display_lcd('run_parking','')
        print('run_parking,ai model name=','')
        
        """
        init tf and models
        """
        model_path= current_path + parking_model_name
        input_shape = (120, 160, 3)
        model_type = 'linear'
        interpreter = KerasInterpreter()
        kl = KerasLinear(interpreter=interpreter, input_shape=input_shape)
        kl.load(model_path)
        
        Run_main = True
        print('model_path=',model_path)
        while Run_main:
            
            final_gyro_degree = read_cumulative_yaw()
            print('Gyro_degree:',final_gyro_degree)

            
            if GPIO.input(gpio_Num) ==1:
                Run_main = False


            # get image from camera
            frame = PiCamera.run(cam)
            outputs = KerasLinear.run(kl,img_arr = frame)
            servo_steering_signal = round(outputs[0],2)
            servo_signal = round(outputs[1],2)

            output_PWM(servo_steering_signal,servo_signal)

            time.sleep(0.01)
            #cv2.imshow('Camera Feed', frame)
             # Break the loop on 'q' key press
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        Run_main = False
        
        output_stop()
        display_lcd('STOP','pwm out 350')
        print('=============================================')

