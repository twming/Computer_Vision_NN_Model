# Computer_Vision_NN_Model

## Topic 2: Basic Computer Vision Techniques

### Activity 2.1: Raspberry Pi Installation
- Install Raspberry Pi 4 Bookworm 64 Bits using Imager with Python 3.11
- 
### Activity 2.2: Setup environment
- Setup the computer vision environment with opencv2
```
cd ~
mkdir yolo
cd yolo
python -m venv  yoloenv
source yoloenv/bin/activate
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
import numpy as np
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
import numpy as np
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
source cvisionenv/bin/activate
```

### Activity 3.4 Image Classification
- Download from tensorflow repository
```
sudo apt update
git clone https://github.com/tensorflow/examples.git
```
- Install packages dependencies
```
pip install numpy==1.20.0
pip install opencv-python==4.5.3.56
pip install protobuf==3.20.3
pip install tflite-runtime==2.13.0
pip install tflite-support==0.4.3
```
- Setup and install the image classification model
``` 
cd examples/lite/examples/image_classification/raspberry_pi/
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
picamera       1.13
pip            24.3.1
pkg_resources  0.0.0
protobuf       3.20.3
pybind11       2.13.6
pycparser      2.22
setuptools     44.1.1
sounddevice    0.5.1
tflite-runtime 2.13.0
tflite-support 0.4.3
wheel          0.40.0
```

### Activity 3.5 Run Image Classification using Camera
- Make sure are in raspberry pi directory, run below classify.py
```
cd examples/lite/examples/image_classification/raspberry_pi/
sudo apt install libatlas-base-dev
python classify.py
```

- Link: https://github.com/ultralytics/yolov5/tree/master
