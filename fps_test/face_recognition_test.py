import face_recognition
import cv2
import pickle
import time
from threading import Thread

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
    

print('loading training data')

# read stuff
with open('train.pkl', 'rb') as f:
    Names = pickle.load(f)
    Encodings = pickle.load(f)

print('start recognition')

font = cv2.FONT_HERSHEY_SIMPLEX

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
        framesmall = cv2.resize(frame, (0, 0), fx=scaleFactor, fy=scaleFactor)
        frameRGB = cv2.cvtColor(framesmall, cv2.COLOR_BGR2RGB)
        facePositions = face_recognition.face_locations(frameRGB)
        allEncodings = face_recognition.face_encodings(frameRGB, facePositions)
        for (top, right, bottom, left), face_encoding in zip(facePositions, allEncodings):
            name = 'Unknown person'
            matches = face_recognition.compare_faces(Encodings, face_encoding)
            if True in matches:
                first_match_index = matches.index(True)
                name = Names[first_match_index]
            top = int(top/scaleFactor)
            right = int(right/scaleFactor)
            bottom = int(bottom/scaleFactor)
            left = int(left/scaleFactor)
            cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
            cv2.putText(frame, name, (left, top-6), font, .75, (255, 0, 0), 2)
        
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

