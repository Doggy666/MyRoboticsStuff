# FinalDrivingCodeForMontreal
This is the final edit of the code that the King's Legacy team will be using during the RoboCup Rapidly Manufactured Robotics League, Montreal, 2018 
----------------------------------------------------------------------------------------------------------------------------
##### ROBOT: <br />
 Wireless code is put on the robot that is being controlled. It does not matter which robot it will be uploaded to, **BUT** you will need to edit which mode each servo is at the beginning of the RoboCupServer-Current.py file. On our robots, this file is located on the desktop.

##### LAPTOP: <br />
 The LaptopCodeMontreal2018 (LCM) is used by the laptop. You will need to access the file within LCM that describes the robot you require to drive. On our laptops, this file is located on the desktop.
 
## DRIVING CODE: <br />
 ##### INSTRUCTIONS: <br />
   1. Connect the robot to a power source and make sure it cannot damage itself or anything around in the rare occasion that the connection breaks and servos continue to move.
   2. Sign into the Laptop. Make sure you are connected to the correct robot's access point, this will take a few moments as the robot boots up and runs some preliminary files to establish an access point. It is important you access the right robot.
   3. Open one(1) terminal window. This is referred to as TW1.
   4. Within TW1 you will need to ssh into the robot. In this window, you will need to type "ssh pi@192.168.100.1". You will be prompted to enter a password. If you have not changed it on the raspberry pi within the robot, it will be "raspberry". **NOTE:** If an error occurs, in the middle of the error statement is a line of code. Run this code into the terminal command line and repeat this step.
   5. Then enter "cd Desktop/Wless_code_RC18". This may be different depending on where you saved the file Wless_code_RC18.
   6. After changing to the file Wless_code_RC18, you are required to run "python RoboCupServer-Current.py" to start the server code on the robot. **NOTE:** At the beggining of this python file, if running on the EmuBot, it is required to have 4 wheels with IDs 1,2,3 and 4, and 3 joints with IDs 5,6,7 respectively. If running on the Flipper robot, it is required to have 4 wheels with IDs 1,2,3 and 4, and 7 joints with IDs 5,6,7,8,9,10,11 respectively. There will be an update with an image of the configuration of each servo.
   7. On the laptop's desktop, you are required to open the LCM file. 
   8. Within this file you will find two python scripts named EmuBotClient.py and FlipperBotClient.py for each robot respectively. Open this file and run the code. NOTE: You are required to have a controller connected to the laptop otherwise an error will be raised. There will be an update with an image of the controls on the controller.
   9. See below for debugging statements.
  
###### TROUBLESHOOTING: <br />
  - When running robot code: <br />
    - Incomplete packet: <br />
      - Make sure servos are recieving power and connected to the USB2AX <br />
    - Wrong Header: <br />
      - Source is unknown. Run the code multiple times until the error stops working. If repeated more than 10 times, reboot the robot, make sure all cables are plugged in and all hardware is recieving necesssary power requirements. <br />
    - Broken Joint or Broken Wheel: <br />
      - A number of servos would have broken. The ID will be displayed on the screen. The most common reason of this is if too much torque experienced by the servo. This is fixed by turning the servos on and off. If this error continues with no pressure on the servo, this servo will be required to be replaced. <br />
  - When running laptop code: <br />
    - Broken Pipe error: <br />
      - Make sure the robot's code is still running and does not contain print statments describing which joint or wheel is broken. See above "Broken Joint or Broken Wheel" after quitting the code and before repeating steps 6 and 8. <br />
 

## CAMERA CODE: <br />
 ##### INSTRUCTIONS: <br />
   1. Connect the robot to a power source and make sure it cannot damage itself or anything around in the rare occasion that the connection breaks and servos continue to move.
   2. Sign into the Laptop. Make sure you are connected to the correct robot's access point, this will take a few moments as the robot boots up and runs some preliminary files to establish an access point. It is important you access the right robot.
   3. Open two(2) terminal window. This is referred to as TW1 and TW2 respectively.
   4. Within TW1 you will need to navigate to the file RCDC. This can be done by entering "cd Desktop/LaptopCodeMontreal2018"
   5. For a clear view of what the robot currently sees, enter "bash Start_MacStream.sh". If you require to find a QRCode or Motion detection, enter "cd Sensors" before entering "bash Start_MacStream.sh | python *enter script of python file here*"
   6. Within TW2 you will need to ssh into the robot. In this window, you will need to type "ssh pi@192.168.100.1". You will be prompted to enter a password. If you have not changed it on the raspberry pi within the robot, it will be "raspberry". **NOTE:** If an error occurs, in the middle of the error statement is a line of code. Run this code into the terminal command line and repeat this step
   7. Then enter "cd Desktop/Wless_code_RC18". This may be different depending on where you saved the file Wless_code_RC18
   8. After navigating to Wless_code_RC18 you will be required to run the script "bash Start_PiStream.sh". After running this, the MacStream cache will fill up. If you chose to pipe the stream into a python file, you will not see the cache fill, but after a few seconds, a window will pop up with the chosen vision.
   
 ###### TROUBLESHOOTING: <br />
   - Large amount of lag: <br />
     - If this happens, within the code on the robot, decrease the frame rate and the size of each frame being sent. The lowest advised setting for frame rate is 10 fps and the lowest advised resolution for each frame is 720 x 480. <br />
   
**NOTE:** Microphone, Temperature and CO2 sensors soon to be included in the next few days
