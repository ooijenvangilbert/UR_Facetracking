import cv2
from ultralytics import YOLO
import time
from threading import Thread

print(cv2.__version__)

import supervision as sv

class vStream:
    def __init__(self, src, width, height):
        self.width = width
        self.height = height
        self.frame = None
        self.capture = cv2.VideoCapture(src)
        self.thread = Thread(target=self.update,args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        while True:
            _,self.frame2 = self.capture.read()
            self.frame = cv2.resize(self.frame2, (self.width, self.height))

    def getFrame(self):
        return self.frame
    

model = YOLO('yolov8n.pt')
#model = YOLO("yolov8x.pt")
CLASS_NAMES_DICT = model.model.names
box_annotator =  sv.BoundingBoxAnnotator()
#label_annotator = sv.LabelAnnotator(classes=model.names)

print('start recognition')

font = cv2.FONT_HERSHEY_SIMPLEX

displayWidth = 640
displayHeight = 480
flip = 0
fpsReport = 0
scaleFactor = .25
cam = vStream(0, displayWidth, displayHeight)
time.sleep(1)
timeStamp = time.time()

while True:
    try:
        frame = cam.getFrame()

        results = model(frame, verbose=False, conf=0.3, iou=0.7)[0]
        detections = sv.Detections.from_ultralytics(results)
        frame = box_annotator.annotate(scene=frame.copy(), detections=detections)
        #frame = label_annotator.annotate(scene=frame, detections=detections )

        dt = time.time()-timeStamp
        fps = 1/dt
        fpsReport = .90*fpsReport + .1*fps
        print('fps : ', round(fpsReport,2))
        timeStamp = time.time()
        cv2.rectangle(frame, (0, 0), (100, 40), (0, 0, 255), -1)
        cv2.putText(frame, str(round(fpsReport, 1))+' fps', (0, 25), font, .75, (0, 255, 255), 2)
        cv2.imshow('picam', frame)  # show the frame
        cv2.moveWindow('picam', 0, 0)
    except:
        print('frame not available')
    
    if cv2.waitKey(1) == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()

