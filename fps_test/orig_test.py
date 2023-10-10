import cv2
import time
from threading import Thread
import imutils
import numpy as np

print(cv2.__version__)

Encodings = []
Names = []


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
    
pretrained_model = cv2.dnn.readNetFromCaffe("MODELS/deploy.prototxt.txt", "MODELS/res10_300x300_ssd_iter_140000.caffemodel")

print('start recognition')

font = cv2.FONT_HERSHEY_SIMPLEX

video_resolution = (700, 400) 

displayWidth = 640
displayHeight = 480
flip = 0
fpsReport = 0
scaleFactor = .25
cam = vStream(0, displayWidth, displayHeight)
time.sleep(1)
#cam = cv2.VideoCapture(0)
# cam=cv2.VideoCapture('/dev/video1')
#cam.set(cv2.CAP_PROP_FRAME_WIDTH, displayWidth)
#cam.set(cv2.CAP_PROP_FRAME_HEIGHT, displayHeight)

timeStamp = time.time()
while True:
    try:
        frame = cam.getFrame()

        frame = imutils.resize(frame, width= video_resolution[0])

        # grab the frame dimensions and convert it to a blob
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                    (300, 300), (104.0, 177.0, 123.0))

        # pass the blob through the network and obtain the detections and predictions
        pretrained_model.setInput(blob)

        # the following line handles the actual face detection
        # it is the most computationally intensive part of the entire program
        # TODO: find a quicker face detection model
        detections = pretrained_model.forward()

        # loop over the detections
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence < 0.4:
                continue

            # compute the (x, y)-coordinates of the bounding box for the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")


            # draw the bounding box of the face along with the associated probability
            text = "{:.2f}%".format(confidence * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10

            face_center = (int(startX + (endX - startX) / 2), int(startY + (endY - startY) / 2))



            cv2.rectangle(frame, (startX, startY), (endX, endY),
                        (0, 0, 255), 2)
            cv2.putText(frame, text, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

        
        dt = time.time()-timeStamp
        fps = 1/dt
        fpsReport = .90*fpsReport + .1*fps
        # print('fps : ', round(fpsReport,2))
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

