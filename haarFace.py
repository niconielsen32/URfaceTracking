import numpy as np
import cv2
import time

face_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_alt2.xml')
#face_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')


cap = cv2.VideoCapture(0)


while cap.isOpened():

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
            cv2.imshow('Face Detection', img)

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()