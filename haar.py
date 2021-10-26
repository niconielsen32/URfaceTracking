import numpy as np
import cv2
import time

def haar(cap, face_cascade, write = False, show = False, filename = "image.png"):
    error = False
    start = time.time()

    success, img = cap.read()

    height, width, channels = img.shape
    center_point = (int(width / 2), int(height / 2))
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)        
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) != 0:
        for (x,y,w,h) in faces:
            img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            center_bbox = (int(x + w/2), int(y + h/2))
            x_displacement = center_bbox[0] - center_point[0]
            y_displacement = center_point[1] - center_bbox[1]
            bbox_area = w * h
        end = time.time()
        totalTime = end - start
        fps = 1 / totalTime
        cv2.circle(img, center_bbox, radius=3, color=(0, 0, 255), thickness=2)
        cv2.putText(img, f'x_dis: {int(x_displacement)}', (330,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
        cv2.putText(img, f'y_dis: {int(y_displacement)}', (330,110), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
        cv2.putText(img, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
        cv2.line(img, (center_point[0], 0), (center_point[0], height), (0, 255, 0), thickness=2)
        cv2.line(img, (0, center_point[1]), (width, center_point[1]), (0, 255, 0), thickness=2)
        if(write):
            cv2.imwrite(filename, img)
        if(show):
            cv2.imshow(filename, img)
    else:
        end = start
        totalTime = 0
        fps = 0
        error = True
        bbox_area = 0
        x_displacement = 0
        y_displacement = 0
    

    return x_displacement, y_displacement, totalTime, bbox_area, fps, totalTime, error


if __name__ == "__main__":

    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_alt2.xml')
    x_displacement = 0
    y_displacement = 0
    totalTime = 0
    bbox_area = 0
    fps = 0
    totalTime = 0
    while (True):
        x_displacement, y_displacement, totalTime, bbox_area, fps, totalTime, error = haar(cap, face_cascade, write=True)
