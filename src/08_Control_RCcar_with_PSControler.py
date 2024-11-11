#!/usr/bin/env python3
'''
'''

import os
import time
import sys
import serial
import time
import Adafruit_PCA9685
import pygame


#======================================================
#pwm = Adafruit_PCA9685.PCA9685()
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

# ======================================================
"""
init joystick
"""
joystick_data={"axis0":0,"axis1":0,"axis2":0,"axis3":0,"axis4":0,"axis5":0,"button0":0,"button1":0,"button2":0,"button3":0,}
pygame.init()
pygame.joystick.init()
joystick_count = pygame.joystick.get_count()
try:
    joystick = pygame.joystick.Joystick(joystick_count-1)
    joystick.init()
    time.sleep(0.1)
    timer01=time.time()
    while time.time()-timer01<3:
        event=pygame.event.get()
        if event != []:
            print("joystick  No"+str(joystick_count)+"  is connected")
            
            break
        else:
            print("joystick has not data send,please try again.")
            time.sleep(0.01)
except :
    print("no joystick connected")
#=======================================================

'''
run Main CODE
'''
# setup PMW "stop" and "central" data
default_servo_signal = 350
default_servo_steering_signal = 350
servo_steering_signal = 0
servo_signal = 0
# setup car status
time_last = time.time()
# car_status forward:"1" ; backward:"-1" ; stop:"0"




if __name__ == '__main__':

    
    while True:
        


# wireless controller data      
        event=pygame.event.get()
        if event != []:
            joystick_data["axis0"] = round(joystick.get_axis( 0 ),2)
            joystick_data["axis4"] = 0-round(joystick.get_axis( 4 ),2)
            print("a0",joystick_data["axis0"],"                             a4",joystick_data["axis4"])

#Prepare the PMW data
        servo_signal = int((joystick_data["axis4"])*(50) + default_servo_signal)
        if servo_signal > 400:
            servo_signal = int(400)
        if servo_signal < 300:
            servo_signal = int(300)
            
#         servo_steering_signal = int((joystick_data["axis0"])*150+385)
        servo_steering_signal = int((joystick_data["axis0"])*(0-150)+ default_servo_steering_signal)
        if servo_steering_signal > 520:
            servo_steering_signal = int(520)
        if servo_steering_signal < 220:
            servo_steering_signal = int(220)
    
#output to 9685    
        print("streeing PWM = ",servo_steering_signal,"          throttle PWM = ",servo_signal)
        
        pwm.set_pwm(1, 0, servo_steering_signal)
        pwm.set_pwm(0, 0, servo_signal)
           


        time.sleep(0.05)