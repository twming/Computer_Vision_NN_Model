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
<img src="https://github.com/twming/Computer_Vision_NN_Model/blob/Pi5/img/terminal.png" alt="Terminal" width="500">

2. Install python 3.9
```
sudo apt update
sudo apt install -y software-properties-common git curl
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
sudo systemctl xrdp
```
5. Check your Raspberry Pi IP address
```
ip addr
```
6. Reboot your Raspberry Pi
```
sudo poweroff
```

### Activity: Setup Tensorflow Computer Vision Environment
1. Login to RaspberryPi4/5 (ip:xxx.xxx.xxx.xxx) using username/password (pi/pi)

<img src="https://github.com/twming/Computer_Vision_NN_Model/blob/Pi5/img/remote_desktop.png" alt="RemoteDesktop" width="500">

2. Open the terminal
3. Create a py39 environment and activate it
```
cd ~
python3.9 -m venv  py39
source ~/py39/bin/activate
```
4. Install packages dependencies
```
pip3 install argparse
pip3 install opencv-python==4.5.3.56
pip3 install protobuf==3.20.3
pip3 install tflite-runtime==2.13.0
pip3 install tflite-support==0.4.3
pip3 install numpy==1.20.0
```
5. Check the correct package lists installed
```
pip3 list -l
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
6. Clone the Tensorflow Example repository
```
git clone https://github.com/tensorflow/examples.git
```

### Activity: Tensorflow Image Classification with EfficientNet

1. Setup and install the image classification EfficientNet model
``` 
cd ~/examples/lite/examples/image_classification/raspberry_pi/
sh setup.sh
````

2. Connect your USB Camera and give the access permission to /dev/video0
```
sudo chmod 777 /dev/video0
```

3. Run Image Classification using Camera
```
python classify.py
```

### Activity: Tensorflow Image Classification with MobileNet

1. Go to below link and download MobileNet model
```
https://tfhub.dev/tensorflow/lite-model/mobilenet_v2_1.0_224/1/metadata/1
```
2. Move the file (~/Dowload) to image_classification/raspberry_pi folder
```
mv ~/Downloads/1.tflite ~/examples/lite/examples/image_classification/raspberry_pi/mobilenet_v2_1.0_224_1_metadata_1.tflite
```
3. Run image classification using MobileNet model
```
python classify.py --model mobilenet_v2_1.0_224_1_metadata_1.tflite
```

## Topic 4: Object Detection

### Activity: Tensorflow Object Detection with EfficientNet

1. Setup and install the Object Detection EfficientNet model
``` 
cd ~/examples/lite/examples/object_detection/raspberry_pi/
sh setup.sh
````

2. Run Object detection using Camera
```
python detect.py
```

### Activity: Tensorflow Image Segmentation with EfficientNet

1. Setup and install the Object Detection EfficientNet model
``` 
cd ~/examples/lite/examples/image_segmentation/raspberry_pi/
sh setup.sh
````

2. Run Image Segmentation using Camera
```
python segment.py
```


### Activity: Tensorflow Image Segmentation with MobileNetV2

1. Go to below link and download MobileNetV2 model
```
https://tfhub.dev/sayakpaul/lite-model/mobilenetv2-dm05-coco/int8/1
```
2. Move the file (~/Dowload) to image_segmentation/raspberry_pi folder
```
mv ~/Downloads/1.tflite ~/examples/lite/examples/image_segmentation/raspberry_pi/lite-model_mobilenetv2-dm05-coco_int8_1.tflite
```
3. Run image segmentation using MobileNetV2 model
```
python segment.py --model lite-model_mobilenetv2-dm05-coco_int8_1.tflite
```


### Activity: Setup Yolo Computer Vision Environment

```
cd ~
python3 -m venv py312
source ~/py312/bin/activate
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


### Activity 3.1 Teachable Machine
- Go to below link, train 2 different images, let the machine recognize them.
- https://teachablemachine.withgoogle.com/train/image




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
