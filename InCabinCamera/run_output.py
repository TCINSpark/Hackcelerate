import cv2 as cv

camera = cv.VideoCapture("/Users/dasaradibudhi/OneDrive - TOYOTA Connected India Pvt. Ltd/Project/TCDS/Hackathon/Hackcelerate/Code/output21.avi")
while True:
    _, frame = camera.read()
    cv.imshow('frame', frame)
    key = cv.waitKey(1)
    if key==ord('q') or key ==ord('Q'):
        break