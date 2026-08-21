## Topic 2: Basic Computer Vision Techniques

### Activity : Raspberry Pi Installation and Setup
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
1. Open Terminal (Ctrl+Alt+T)
<img src="https://github.com/twming/Computer_Vision_NN_Model/blob/Pi5/img/terminal.png" alt="Terminal" width="500">

2. Install python 3.9
```
sudo apt update
sudo apt install -y software-properties-common git curl gedit python3-pip
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.9 python3.9-venv python3.12-venv python3.9-dev
```
3. Install XRDP remote access
```
sudo apt install -y ubuntu-gnome-desktop
sudo apt install -y xrdp
sudo adduser xrdp ssl-cert
sudo ufw enable
sudo ufw allow 3389/tcp
sudo ufw reload
```
4. Check the service is running and the port are allowed
```
sudo ufw status
sudo systemctl status xrdp
```
5. Check your Raspberry Pi IP address
```
ip addr
```
6. Reboot your Raspberry Pi
```
sudo poweroff
```
7. Login to RaspberryPi4/5 (ip:xxx.xxx.xxx.xxx) using username/password (pi/pi)

> [!IMPORTANT] 
> - There is ONLY ONE session to allow login, if you have multiple session, please Log Out others.

<img src="https://github.com/twming/Computer_Vision_NN_Model/blob/Pi5/img/remote_desktop.png" alt="RemoteDesktop" width="500">


> [!IMPORTANT] 
> - Please take note of the IP Address, as you need it to remote login after reboot

### Activity: Python Env and Install OpenCV
1. Open a terminal (Ctrl+Alt+T), run below to create python 3.12 environment, called py312. Then install opencv
```
cd ~
python3 -m venv py312
source ~/py312/bin/activate
pip install opencv-python
```
2. deactivate py312
```
deactivate
```
3. Open another terminal (Ctrl+Alt+T), run below to create python 3.9 environment, called py39. Then install opencv
```
cd ~
python3.9 -m venv py39
source ~/py39/bin/activate
pip install opencv-python
```
4. deactivate py312
```
deactivate
```

### Activity : Capture Video from Camera
1. Open TextEdit file "capture_video.py"
```
cd ~/Downloads
gedit capture_video.py
```

2. Save below code to "capture_video.py"
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

3. Give the terminal access to your camera
```
sudo chmod 777 /dev/video0
```

4. Run the capture_video.py
```
python capture_video.py
```

### Activity : Saving Video from Camera
1. Open TextEdit file "save_video.py"
```
cd ~/Downloads
gedit save_video.py
```

2. Save below code to "save_video.py"
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

3. Give the terminal access to your camera (only need to do once per terminal)
```
sudo chmod 777 /dev/video0
```

4. Run the save_video.py
```
python save_video.py
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
### Activity : Play Video
1. Modify from capture_video.py, then play the video "output.avi"

## Topic 3: Image Classification

### Activity: Teachable Machine
- Go to below link, train 3 different classes (scissor, paper, stone), let the machine recognize them.
- https://teachablemachine.withgoogle.com/train/image

### Activity: Setup Tensorflow Computer Vision Environment

1. Open the terminal (Ctrl+Alt+T)
2. Activate py39 environment
```
source ~/py39/bin/activate
```
3. Clone the Tensorflow Example repository
```
cd ~/Downloads
git clone https://github.com/tensorflow/examples.git
```
4. Install packages dependencies
```
pip install argparse
pip install opencv-python==4.5.3.56
pip install protobuf==3.20.3
pip install tflite-runtime==2.13.0
pip install tflite-support==0.4.3
pip install numpy==1.20.0
```
5. Check the correct package lists installed
```
pip list -l
```
```
Package        Version
-------------- --------------
numpy          1.20.0
opencv-python  4.5.3.56
protobuf       3.20.3
tflite-runtime 2.13.0
tflite-support 0.4.3
```

> [!IMPORTANT] 
> - Make sure above packages version are matched, below proceed on.

### Activity: Tensorflow Image Classification with EfficientNet
1. Activate py39 environment
```
source ~/py39/bin/activate
```
2. Setup and install the image classification EfficientNet model
``` 
cd ~/Downloads/examples/lite/examples/image_classification/raspberry_pi/
sh setup.sh
````
3. Connect your USB Camera and give the access permission to /dev/video0
```
sudo chmod 777 /dev/video0
```

> [!IMPORTANT] 
> - Give permission 777 to /dev/video0 your USB camera, before running any computer vision python code

4. Run Image Classification using Camera, q to quit
```
python classify.py
```
5. Deactivate python environment
```
deactivate
```

### Activity: Image Classification Run Option
1. Activate py39 environment
```
source ~/py39/bin/activate
```
2. Run classify image with EfficientNet. Return only 1 result. The threshold score should be more than 0.7
```
cd ~/Downloads/examples/lite/examples/image_classification/raspberry_pi/
python classify.py --model efficientnet_lite0.tflite --maxResults 1 --scoreThreshold 0.7
```
3. Deactivate python environment
```
deactivate
```

### Activity: Tensorflow Image Classification with MobileNet
1. Activate py39 environment
```
source ~/py39/bin/activate
```
2. Go to below link and download MobileNet model
```
curl -L https://github.com/twming/Computer_Vision_NN_Model/raw/Pi5/mobilenet_v2_1.0_224_1_metadata_1.tflite -o ~/Downloads/examples/lite/examples/image_classification/raspberry_pi/mobilenet_v2_1.0_224_1_metadata_1.tflite
```
3. Run image classification using MobileNet model
```
cd ~/Downloads/examples/lite/examples/image_classification/raspberry_pi/
python classify.py --model mobilenet_v2_1.0_224_1_metadata_1.tflite
```
4. Deactivate python environment
```
deactivate
```

## Topic 4: Object Detection

### Activity: Tensorflow Object Detection with EfficientNet

1. Activate py39 environment
```
source ~/py39/bin/activate
```
2. Setup and install the Object Detection EfficientNet model
``` 
cd ~/Downloads/examples/lite/examples/object_detection/raspberry_pi/
sh setup.sh
````
3. Connect your USB Camera and give the access permission to /dev/video0
```
sudo chmod 777 /dev/video0
```
4. Run Object detection using Camera
```
cd ~/Downloads/examples/lite/examples/object_detection/raspberry_pi/
python detect.py
```
5. Deactivate python environment
```
deactivate
```

### Activity: Tensorflow Object Detection with SSD_MobileNet
1. Activate py39 environment
```
source ~/py39/bin/activate
```
2. Go to below link and download SSD_MobileNet model
```
curl -L https://github.com/twming/Computer_Vision_NN_Model/raw/Pi5/ssd_mobilenet_v1_metadata.tflite -o ~/Downloads/examples/lite/examples/object_detection/raspberry_pi/ssd_mobilenet_v1_metadata.tflite
```
3. Run object detection using SSD_MobileNet model
```
cd ~/Downloads/examples/lite/examples/object_detection/raspberry_pi/
python detect.py --model ssd_mobilenet_v1_metadata.tflite
```
4. Deactivate python environment
```
deactivate
```

### Activity: Setup YoLo Computer Vision Environment
1. Open the terminal
2. Activate py312 environment
```
source ~/py312/bin/activate
```
3. Clone the YoLo5 repository and install dependencies from requirements file
```
cd ~/Downloads
git clone https://github.com/ultralytics/yolov5  
cd yolov5
pip install -r requirements.txt
```
4. Deactivate python environment
```
deactivate
```

### Activity: YoLo Object Detection
1. Activate py312 environment
```
source ~/py312/bin/activate
```
2. Connect your USB Camera and give the access permission to /dev/video0
```
sudo chmod 777 /dev/video0
```
> [!IMPORTANT] 
> - Give permission 777 to /dev/video0 your USB camera, before running any computer vision python code

3. Go to yolov5 folder and run detect.py
```
cd ~/Downloads/yolov5
python detect.py --source 0
```
4. Deactivate python environment
```
deactivate
```

### Activity: YoLo v5 TFLite Model
1. Activate py312 environment
```
source ~/py312/bin/activate
```
2. Go to below link and download YOLO v5 TFLite model
```
curl -L https://github.com/twming/Computer_Vision_NN_Model/raw/Pi5/lite-model_yolo-v5-tflite_tflite_model_1.tflite -o ~/Downloads/yolov5/lite-model_yolo-v5-tflite_tflite_model_1.tflite
```
3. Install tensorflow package to read tflite model.
```
pip install tensorflow
```
4. Run YOLO v5 model
```
cd ~/Downloads/yolov5
python detect.py --source 0 --weights lite-model_yolo-v5-tflite_tflite_model_1.tflite --imgsz 320
```
5. Deactivate python environment
```
deactivate
```

### Activity: Tensorflow Image Segmentation with EfficientNet
1. Activate py39 environment
```
source ~/py39/bin/activate
```
2. Setup and install the Image Segmentation EfficientNet model
``` 
cd ~/Downloads/examples/lite/examples/image_segmentation/raspberry_pi/
sh setup.sh
````
3. Run Image Segmentation using Camera
```
python segment.py
```
4. Deactivate python environment
```
deactivate
```

### Activity: Tensorflow Image Segmentation with MobileNetV2
1. Activate py39 environment
```
source ~/py39/bin/activate
```
2. Go to below link and download MobileNetV2 model
```
curl -L https://github.com/twming/Computer_Vision_NN_Model/raw/Pi5/lite-model_mobilenetv2-dm05-coco_int8_1.tflite -o ~/Downloads/examples/lite/examples/image_segmentation/raspberry_pi/lite-model_mobilenetv2-dm05-coco_int8_1.tflite
```
3. Run image segmentation using MobileNetV2 model
```
cd ~/Downloads/examples/lite/examples/image_segmentation/raspberry_pi/
python segment.py --model lite-model_mobilenetv2-dm05-coco_int8_1.tflite
```
4. Deactivate python environment
```
deactivate
```

### Activity: YoLo Image Segmentation 
1. Activate py312 environment
```
source ~/py312/bin/activate
```
2. Go to below link and download YOLO v5 TFLite model
```
curl -L https://github.com/twming/Computer_Vision_NN_Model/raw/Pi5/yolov5s-seg.pt -o ~/Downloads/yolov5/segment/yolov5s-seg.pt
```
3. Run YOLO v5 model
```
cd ~/Downloads/yolov5/segment
python predict.py --source 0 --weights yolov5s-seg.pt
```
4. Deactivate python environment
```
deactivate
```

### Revision:
- Classify Image:
```
classify_path=~/Downloads/examples/lite/examples/image_classification/raspberry_pi/
python $classify_path/classify.py --model $classify_path/mobilenet_v2_1.0_224_1_metadata_1.tflite 
```
- Object Detection:
```
detect_path=~/Downloads/examples/lite/examples/object_detection/raspberry_pi/
python $detect_path/detect.py --model $detect_path/efficientdet_lite0.tflite
```
- Object Segmentation
```
segment_path=~/Downloads/examples/lite/examples/image_segmentation/raspberry_pi/
python $segment_path/segment.py --model $segment_path/lite-model_mobilenetv2-dm05-coco_int8_1.tflite
```
