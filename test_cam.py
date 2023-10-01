import cv2
print(cv2.__version__)

displayWidth = 1280
displayHeight = 720

# usb option on jetson nano
#cam=cv2.VideoCapture('/dev/video2')
cam=cv2.VideoCapture(0)

# macbook internal camera
# cam = cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()
    cv2.imshow('picam', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()