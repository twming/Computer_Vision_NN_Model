# Computer Vision for Beginner

## Topic 2: Basic Computer Vision Techniques

### Activity : Raspberry Pi Installation
1. Go to https://www.raspberrypi.com/software/, download Raspberry Pi Imager.

<img src="https://github.com/twming/Computer_Vision_NN_Model/blob/Pi5/img/rasp-imager-download.png" alt="ImagerDownload" width="600">

2. Install Ubuntu Desktop 24.04.2 LTS (64-Bit), it will take about 20 mins to clone the SDCard
<img src="https://github.com/twming/Computer_Vision_NN_Model/blob/Pi5/img/rasp-imager.png" alt="Imager" width="300">
<img src="https://github.com/twming/Computer_Vision_NN_Model/blob/Pi5/img/rasp-ubuntu.png" alt="Ubuntu" width="300">
3. System Configuration: Insert the cloned SDCard to RaspberryPi4/5, boot up the system, you need to setup below:

- Language: English
- Keyboard Layout: English(US)
- Wireless: SSID/Password
- Country/Zone: Singapore
- username/password: pi/pi (Require my password to log in)

### Activity: Configure Ubuntu and Environment
1. Login to RaspberryPi4/5, open terminal
<img src="https://github.com/twming/Computer_Vision_NN_Model/blob/Pi5/img/terminal.png" alt="Terminal" width="300">
2. install python 3.9
```
sudo apt update
sudo apt install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.9 python3.9-venv python3.12-venv python3.9-dev
```
2. Install XRDP remote access
```
sudo apt install -y ubuntu-gnome-desktop
sudo apt install -y xrdp
sudo adduser xrdp ssl-cert
sudo ufw enable
sudo ufw allow 3389/tcp
sudo ufw reload
```
3. Check the service is running and the port are allowed
```
sudo ufw status
sudo systemctl xrdp
```
4. Reboot your Raspberry Pi

### Activity: Setup Tensorflow Computer Vision Environment

1. Go to Virtual box website and download the application, install in your laptop.
```
https://www.virtualbox.org/wiki/Downloads
```
<img src="https://github.com/twming/ros2_master_tutorial/blob/main/img/virtualbox.png" alt="Virtual Box" width="600">

### Activity 2.2: Setup environment
```
cd ~
mkdir yolo
cd yolo
python -m venv  yoloenv
source ~/yolo/yoloenv/bin/activate
```
- Install the requirements file
```
git clone https://github.com/ultralytics/yolov5  
cd yolov5
pip install -r requirements.txt
```

### Activity 2.3: Using OpenCV and Camera
- Open Thonny, save below code to "capture_video.py" 
- Run the code "python capture_video.py" to capture video from camera
```
import cv2 as cv

cap = cv.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    cv.imshow('frame', gray)
    if cv.waitKey(1) == ord('q'):
        break
cap.release()
cv.destroyAllWindows()
```
- Open Thonny, save below code to "save_video.py" 
- Run the code "python save_video.py" to save video from camera (output.avi)
```
import cv2 as cv

cap = cv.VideoCapture(0)
fourcc = cv.VideoWriter_fourcc(*'XVID')
out = cv.VideoWriter('output.avi', fourcc, 20.0, (640,  480))

while cap.isOpened():
    ret, frame = cap.read()
    frame = cv.flip(frame, 0)

    out.write(frame)
    cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
out.release()
cv.destroyAllWindows()
```
- Picamera2 reference
```
import cv2
from picamera2 import Picamera2

picam2 = Picamera2()
picam2.start()
while True:
    image = picam2.capture_array()
    cv2.imshow("Frame", image)
    if(cv2.waitKey(1) == ord("q")):
        cv2.imwrite("test_frame.png", image)
        break

cv2.destroyAllWindows()
```
## Topic 3: Image Classification

### Activity 3.1 Teachable Machine
- Go to below link, train 2 different images, let the machine recognize them.
- https://teachablemachine.withgoogle.com/train/image

### Activity 3.2: Raspberry Pi Installation
- Install Raspberry Pi 4 Bullseye 32 Bits using Imager with Python 3.9

### Activity 3.3: Setup environment
- Setup the computer vision environment with opencv2
```
cd ~
mkdir cvision
cd cvision
python -m venv  cvisionenv
source ~/cvision/cvisionenv/bin/activate
```

### Activity 3.4 Image Classification
- Download from tensorflow repository
```
sudo apt update
git clone https://github.com/tensorflow/examples.git
```
- Install packages dependencies
```
pip install argparse
pip install opencv-python==4.5.3.56
pip install protobuf==3.20.3
pip install tflite-runtime==2.13.0
pip install tflite-support==0.4.3
pip install numpy==1.20.0
```
- Setup and install the image classification model
``` 
cd ~/cvision/examples/lite/examples/image_classification/raspberry_pi/
rm requirements.txt
touch requirements.txt
sh setup.sh
````
- Check the correct package lists installed
```
pip list -l
```
```
Package        Version
-------------- --------------
absl-py        2.1.0
cffi           1.17.1
flatbuffers    20181003210633
numpy          1.20.0
opencv-python  4.5.3.56
pkg_resources  0.0.0
protobuf       3.20.3
pybind11       2.13.6
pycparser      2.22
setuptools     44.1.1
sounddevice    0.5.1
tflite-runtime 2.13.0
tflite-support 0.4.3
```

### Activity 3.5 Run Image Classification using Camera
- Make sure are in raspberry pi directory, run below classify.py
```
cd ~/cvision/examples/lite/examples/image_classification/raspberry_pi/
sudo apt install libatlas-base-dev
python classify.py
```

### Activity 3.6 MobileNet TFLite Model
- Download MobileNet TFLite Model
```
https://tfhub.dev/tensorflow/lite-model/mobilenet_v2_1.0_224/1/metadata/1
```
- Copy the file to running folder first, run image classification using MobileNet model
```
python classify --model mobilenet_v2_1.0_224_1_metadata_1.tflite
```

## Topic 4: Object Detection

### Activity 4.1 Object Detection
- Setup object detection
```
cd ~/cvision/examples/lite/examples/object_detection/raspberry_pi/
rm requirements.txt
touch requirements.txt
sh setup.sh
```
- run object detection
```
python detect.py
```
### Activity 4.2 YOLO Object Detection
- Use back the Raspberry Pi 4, Bookworm in Activity 2.1
```
cd ~/yolo
source yoloenv/bin/activate
git clone https://github.com/ultralytics/yolov5
cd ~/yolo/yolov5
pip install -r requirements.txt
```
- Run YOLO object detection
```
python detect.py --source 0
```
### Activity 4.3 YOLO v5 TFLite Model 
- Download YOLO v5 TFLite Model
```
https://tfhub.dev/neso613/lite-model/yolo-v5-tflite/tflite_model/1 
```
- Install dependency
```
pip install tensorflow
```
- Run YOLO v5
```
python detect.py --source 0 --weights lite-model_yolo-v5-tflite_tflite_model_1.tflite
```
### Activity 4.4: Image Segmentation Setup
```
cd ~/cvision/examples/lite/examples/image_segmentation/raspberry_pi/
rm requirements.txt
touch requirements.txt
sh setup.sh
python segment.py
```

### Activity 4.5: Image Segmentation with MobileNetV2
- Download MobileNetV2 Segmentation TFLite Model
```
https://tfhub.dev/sayakpaul/lite-model/mobilenetv2-dm05-coco/int8/1
```
- Run MobileNetV2 Segmentation
```
python segment.py --model lite-model_mobilenetv2-dm05-coco_int8_1.tflite
```
### Revision:
- Classify Image:
```
classify_path=~/cvision/examples/lite/examples/image_classification/raspberry_pi/
python $classify_path/classify.py --model mobilenet_v2_1.0_224_1_metadata_1.tflite 
```
- Object Detection:
```
detect_path=~/cvision/examples/lite/examples/object_detection/raspberry_pi/
python $detect_path/detect.py --model $detect_path/efficientdet_lite0.tflite
```
- Object Segmentation
```
segment_path=~/cvision/examples/lite/examples/image_segmentation/raspberry_pi/
python $segment_path/segment.py --model lite-model_mobilenetv2-dm05-coco_int8_1.tflite
```
