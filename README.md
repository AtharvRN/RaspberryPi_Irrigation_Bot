# Plant Watering Bot
**Developing this Read Me File Currently**</br>
This project was done under the **Robotix**, **IITH** <br/>
## Function 
The bot can detect potted plants in its surroundings and move towards it. Once the plant is within a certain distance ,the bot waters it<br/>
## Hardware Used
- Raspberry Pi 4B (Along with fan and heat sink< It is recommended to overclock the Pi) 
- DC Motors X2
- Lipo Battery -1000 Mah
- L298N Motor Drivers X2
- Pump Motor
- Water Container
- Robot Chassis (4 wheeled but controlled using 2 back wheels) 
- USB Camera
- Wires,tapes etc. </br>
# Software Required
- Python was used to control the bot using Raspberry PI
- Install OpenCV and numpy on RPi (Make sure to install OpenCV from source)
- Google's Sample Tflite model  - coco_ssd_mobilenet_v1_1.0_quant_2018_06_29 has been used for object detection</br>

# Setting up the Hardware
/home/atharv/Pictures/Images/PROJECT.png



This code was used to build an autonomous plant watering bot <br />
A webcam was used to take the live feed which was sent to a raspberry pi <br />
The program helps in detecting potted plants and makes the bot go near the plant and water it 
