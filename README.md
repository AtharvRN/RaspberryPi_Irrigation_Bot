# Plant Watering Bot
**Developing this Read Me File Currently**</br>
This project was done under the **Robotix**, **IITH** <br/>
## Function 
The bot can detect potted plants in its surroundings and move towards it. Once the plant is within a certain range of the plant ,the bot waters it<br/>
## Hardware Used
- Raspberry Pi 4B (Along with fan and heat sink< It is recommended to overclock the Pi) 
- DC Motors X2
- Lipo Battery -1000 Mah
- L298N Motor Drivers X2
- Pump Motor
- Water Container
- Robot Chassis (4 wheeled but controlled using 2 back wheels) 
- USB Camera/Pi Camera
- Wires,tapes etc. </br>
# Software Required
- Python was used to control the bot using Raspberry PI
- Install OpenCV and numpy on RPi (Make sure to install OpenCV from source)
- Google's Sample Tflite model  - coco_ssd_mobilenet_v1_1.0_quant_2018_06_29 has been used for object detection</br>

# FLOWCHART
<img src="https://user-images.githubusercontent.com/82694160/174993769-ad719422-f1e2-4de5-b3ff-1c3fb9696a3d.png" width ="800" height ="540">

# Set Up the Hardware
 Setting up the hardware is rather straight forward. The Motor Drivers are connected to the Raspberry Pi, making use of 9 pins in the pi<br/>
 One Motor Driver is connected to 2 of the DC Motors (used for controlling the wheels). Another one is connected to the pump Motor (used to water the plant)
Make Sure to connect the Raspberry Pi to the USB Camera/Pi Camera (The Code will work for both. Although the code could be more optimised for the pi camera )
I have used a Lipo Battery to power the Motor Drivers and a Power Bank for the pi. You could use the same battery for both of them but make sure to convert voltages to appropriate values using a buck-booster converter<br/>
 This link might be useful : https://www.electronicshub.org/raspberry-pi-l298n-interface-tutorial-control-dc-motor-l298n-raspberry-pi/ </br>
# Water Your Plants
Load the .py file along with the model folder in the **same directory**. Make sure to install OpenCV from source. <br/>
You can now run the file. Videos of Test Runs are there in the videos file
# Future improvements
- Training a custom model for this particular use case would greatly improve the accuracy and speed
- Implementing some form of PID Control can help in better navigation
