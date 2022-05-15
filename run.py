

#The object detection part of the program was taken from Edge Electronics Guide
# Link -https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/blob/master/Raspberry_Pi_Guide.md
# Import packages
import os
import argparse
import cv2
import numpy as np
import sys
import time
from time import sleep
from threading import Thread
import importlib.util

import RPi.GPIO as GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)


# Controlling water flow through the pump
pump_en = 13
pump_in1 = 26
pump_in2 = 19
    
GPIO.setup(pump_en,GPIO.OUT)
GPIO.setup(pump_in1,GPIO.OUT)
GPIO.setup(pump_in2,GPIO.OUT)
    
pwm_p = GPIO.PWM(pump_en,1000)
pwm_p.start(0)
# function to water the plants    
def pump(t=0.5):
    GPIO.output(pump_in1,GPIO.HIGH)
    GPIO.output(pump_in2,GPIO.LOW)
    pwm_p.ChangeDutyCycle(100)
    sleep(t)
    
    GPIO.output(pump_in1,GPIO.LOW)
    GPIO.output(pump_in2,GPIO.LOW)
    pwm_p.ChangeDutyCycle(0)

    
    
#Generic Motor Class
class Motor():
    def __init__(self,EnaA,In1A,In2A,EnaB,In1B,In2B):
        self.EnaA = EnaA
        self.In1A = In1A
        self.In2A = In2A
        self.EnaB = EnaB
        self.In1B = In1B
        self.In2B = In2B
        GPIO.setup(self.EnaA,GPIO.OUT)
        GPIO.setup(self.In1A,GPIO.OUT)
        GPIO.setup(self.In2A,GPIO.OUT)
        GPIO.setup(self.EnaB,GPIO.OUT)
        GPIO.setup(self.In1B,GPIO.OUT)
        GPIO.setup(self.In2B,GPIO.OUT)
        self.pwmA = GPIO.PWM(self.EnaA, 100);
        self.pwmA.start(0);
        self.pwmB = GPIO.PWM(self.EnaB, 100);
        self.pwmB.start(0);
        
    # move function
    def move(self,speed=1,turn=0,t=0):
        speed *=100
        turn *=100
        leftSpeed = speed - turn
        rightSpeed = speed + turn
        if leftSpeed>100: leftSpeed=100
        elif leftSpeed<-100: leftSpeed= -100
        if rightSpeed>100: rightSpeed=100
        elif rightSpeed<-100: rightSpeed= -100
 
        self.pwmA.ChangeDutyCycle(abs(leftSpeed))
        self.pwmB.ChangeDutyCycle(abs(rightSpeed))
 
        if leftSpeed>0:
            GPIO.output(self.In1A,GPIO.HIGH)
            GPIO.output(self.In2A,GPIO.LOW)
        else:
            GPIO.output(self.In1A,GPIO.LOW)
            GPIO.output(self.In2A,GPIO.HIGH)
 
        if rightSpeed>0:
            GPIO.output(self.In1B,GPIO.HIGH)
            GPIO.output(self.In2B,GPIO.LOW)
        else:
            GPIO.output(self.In1B,GPIO.LOW)
            GPIO.output(self.In2B,GPIO.HIGH)
 
        sleep(t)
      
    #stop function
    def stop(self,t=0):
        self.pwmA.ChangeDutyCycle(0);
        self.pwmB.ChangeDutyCycle(0);
        sleep(t)
        
    def moveLeft(self,t):
        GPIO.output(self.In1A,GPIO.HIGH)
        GPIO.output(self.In2A,GPIO.LOW)
        GPIO.output(self.In1B,GPIO.LOW)
        GPIO.output(self.In2B,GPIO.HIGH)
        self.pwmA.ChangeDutyCycle(100)
        self.pwmB.ChangeDutyCycle(100)
        
        sleep(t)
        
    def moveRight(self,t):
        GPIO.output(self.In1A,GPIO.LOW)
        GPIO.output(self.In2A,GPIO.HIGH)
        GPIO.output(self.In1B,GPIO.HIGH)
        GPIO.output(self.In2B,GPIO.LOW)
        self.pwmA.ChangeDutyCycle(100)
        self.pwmB.ChangeDutyCycle(100)
        
        sleep(t)

# VideoStream class to handle streaming of video from webcam in separate processing thread
# Source - Adrian Rosebrock, PyImageSearch: https://www.pyimagesearch.com/2015/12/28/increasing-raspberry-pi-fps-with-python-and-opencv/
class VideoStream:
    """Camera object that controls video streaming from the Picamera"""
    def __init__(self,resolution=(640,480),framerate=30):
        # Initialize the PiCamera and the camera image stream
        self.stream = cv2.VideoCapture(0)
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3,resolution[0])
        ret = self.stream.set(4,resolution[1])
            
        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()

	# Variable to control when the camera is stopped
        self.stopped = False

    def start(self):
	# Start the thread that reads frames from the video stream
        Thread(target=self.update,args=()).start()
        return self

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
	# Return the most recent frame
        return self.frame

    def stop(self):
	# Indicate that the camera and thread should be stopped
        self.stopped = True


# Google's sample tflite model has been used for object detection
# coco_ssd_mobilenet_v1_1.0_quant_2018_06_29
#loading the model

MODEL_NAME = 'Object_Detection_Model'
GRAPH_NAME = 'detect.tflite'
LABELMAP_NAME = 'labelmap.txt'
# alter the threshold here
min_conf_threshold = 0.2

resW, resH = (1280,720)
imW, imH = int(resW), int(resH)
#print(imW,imH)

# Import TensorFlow libraries

pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
    

# Get path to current working directory
CWD_PATH = os.getcwd()

# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,"labelmap.txt")

# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Have to do a weird fix for label map if using the COCO "starter model" from
# https://www.tensorflow.org/lite/models/object_detection/overview
# First label is '???', which has to be removed.
if labels[0] == '???':
    del(labels[0])

# Load the Tensorflow Lite model.


interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]


floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()

#Initialise the drive motors
motor= Motor(10,9,11,17,22,27)

# Initialize video stream
videostream = VideoStream(resolution=(imW,imH),framerate=30).start()
time.sleep(1)


iteration =0
plant_found = False
Area =0
xc=0
yc =0
area_max =750000


while True:
    
    # Start timer (for calculating frame rate)
    t1 = cv2.getTickCount()

    # Grab frame from video stream
    frame1 = videostream.read()

    # Acquire frame and resize to expected shape [1xHxWx3]
    frame = frame1.copy()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    #frame_resized = cv2.resize(frame_rgb, (1280, 780))
    input_data = np.expand_dims(frame_resized, axis=0)

    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    # Perform the actual detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]['index'],input_data)
    interpreter.invoke()

    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[1]['index'])[0] # Class index of detected objects
    scores = interpreter.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects
    #num = interpreter.get_tensor(output_details[3]['index'])[0]  # Total number of detected objects (inaccurate and not needed)
    motor.stop()
        
    
    if plant_found==False:
        if iteration!=0:
            iteration+=1
        if iteration%2==1:
            motor.move(-0.1,0,0.5)
        else :
            motor.moveRight(0.03)
   
     #potted plant 63
    # Loop over all detections and draw detection box if confidence is above minimum threshold
    for i in range(len(scores)):
        if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)) and int(classes[i]) == 63:
            
            plant_found =True
            iteration+=1
            
            
            motor.stop()

            
            
            #print("detected with confidence", scores[i])
            
            
            ymax = int(max(1,(boxes[i][0] * imH)))
            xmax = int(max(1,(boxes[i][1] * imW)))
            ymin = int(min(imH,(boxes[i][2] * imH)))
            xmin = int(min(imW,(boxes[i][3] * imW)))
            
            
            Area = (ymax-ymin)*(xmax-xmin)
            #print("Area is " ,Area)
            
            xc = (xmin+xmax)/2
            yc = (ymax+ymin)/2
            #print("centre at ", xc,yc)
            
             
            #If Area of bounding is sufficiently large enough, bot is suposed to stop.
            # In order to water the plant and the since the pipe is fixed, we require
            # the centre of the bounding box to be within a certain range
            
            if xc< 630:
                motor.moveLeft(0.1)
                        
            elif xc> 650:
                motor.moveRight(0.1)
                
                
            if Area < 500000:
                motor.move(1,0,0.1)
                break
                
            if Area < area_max and Area >500000:
                motor.move(0.6,0,0.1)
                break
            if Area >= area_max:
                motor.stop()
                break
                
        
        else :
            #print("No detetion")
            plant_found=False   
                
    
    #print("Area is " ,Area)
    #print("centre at ", xc,yc)
    
    if Area >area_max and xc>630 and xc<650:
        print("exit")
        motor.stop()
        pump(3)
        break
        
    print("fps",frame_rate_calc)
    #uncomment below line if live feed is not neede
    cv2.imshow('Object detector', frame)

    # Calculate framerate
    t2 = cv2.getTickCount()
    time1 = (t2-t1)/freq
    frame_rate_calc= 1/time1

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Clean up
cv2.destroyAllWindows()
videostream.stop()
