# Computer_Vision_NN_Model


- The Raspberry Pi install on Bullseye 32 Bits, come with Python 3.9
- Capture Video from Camera
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
- Saving Video
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



- Link: https://github.com/tensorflow/examples/tree/master
- Create environment for image_classification 


```
cd ~
mkdir imageclass
cd imageclass
python -m venv  imageclassenv
source imageclassenv/bin/activate
```
- Install dependancies
```
pip install numpy==1.20.0
pip install opencv-python==4.5.3.56
pip install protobuf==3.20.3
pip install tflite-runtime==2.13.0
pip install tflite-support==0.4.3
```
pip list -l
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

- Link: https://github.com/ultralytics/yolov5/tree/master
