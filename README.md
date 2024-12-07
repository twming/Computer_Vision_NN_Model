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
pip install argparse
pip install numpy==1.20.0
pip install opencv-python==4.5.3.56
pip install protobuf==3.20.3
pip install tflite-runtime==2.13.0
pip install tflite-support==0.4.3
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
- Run YOLO v5
```
python detect.py --source 0 --weights lite-model_yolo-v5-tflite_tflite_model_1.tflite
```
### Activity 4.4: Image Segmentation Setup
```
cd ~/csvision/examples/lite/examples/image_segmentation/raspberry_pi/
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
