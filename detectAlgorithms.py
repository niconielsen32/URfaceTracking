from facenet_pytorch import MTCNN
import torch
import numpy as np
import cv2
import time
import mediapipe as mp


def facenet():

    # Check available hardware
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #Create the model
    mtcnn = MTCNN(keep_all=True, device='cpu')
    
    #Load the video and go from frame to frame
    cap = cv2.VideoCapture(0)

    while cap.isOpened():

        ret, frame = cap.read()
        
        start = time.time()

        frame = cv2.resize(frame, (600, 400))

        height, width, channels = frame.shape
        center_point = (int(width / 2), int(height / 2))

        #Here we are going to use the facenet detector
        boxes, conf = mtcnn.detect(frame)


        if conf[0] != None:
            for (x, y, w, h) in boxes:
                text = f"{conf[0]*100:.2f}%"
                x, y, w, h = int(x), int(y), int(w), int(h)

                center_bbox = (int((x+w)/2), int((y+h)/2))
                x_displacement = center_bbox[0] - center_point[0]
                y_displacement = center_point[1] - center_bbox[1]

                bbox_area = (w-x) * (h-y)
            
            end = time.time()
            totalTime = end - start

            fps = 1 / totalTime
            
            cv2.putText(frame, text, (x, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1.5,(0, 255, 0), 2)
            cv2.rectangle(frame,(x,y),(w,h),(0,255,0),2)
            cv2.circle(frame, center_bbox, radius=3, color=(0, 0, 255), thickness=2)
            cv2.putText(frame, f'x_dis: {int(x_displacement)}', (310,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
            cv2.putText(frame, f'y_dis: {int(y_displacement)}', (310,110), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)

            cv2.putText(frame, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
            cv2.line(frame, (center_point[0], 0), (center_point[0], height), (0, 255, 0), thickness=2)
            cv2.line(frame, (0, center_point[1]), (width, center_point[1]), (0, 255, 0), thickness=2)
            cv2.imshow("Frame", frame)


        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    

    cap.release()
    cv2.destroyAllWindows()

def haar():
    
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

def mediaPipe():

    mp_facedetector = mp.solutions.face_detection
    mp_draw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)


    with mp_facedetector.FaceDetection(min_detection_confidence=0.7) as face_detection:

        while cap.isOpened():

            success, image = cap.read()

            start = time.time()

            # Convert the BGR image to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Process the image and find faces
            results = face_detection.process(image)
            
            # Convert the image color back so it can be displayed
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            h, w, c = image.shape
            center_point = (int(w / 2), int(h / 2))

            if results.detections:
                for id, detection in enumerate(results.detections):
                    mp_draw.draw_detection(image, detection)

                    bBox = detection.location_data.relative_bounding_box

                    boundBox = int(bBox.xmin * w), int(bBox.ymin * h), int(bBox.width * w), int(bBox.height * h)

                    center_bbox = (int(boundBox[0] + boundBox[2] / 2), int(boundBox[1] + boundBox[3] / 2))
                    x_displacement = center_bbox[0] - center_point[0]
                    y_displacement = center_point[1] - center_bbox[1]

                    bbox_area = boundBox[2] * boundBox [3]


                end = time.time()
                totalTime = end - start

                fps = 1 / totalTime


                cv2.circle(image, center_bbox, radius=3, color=(0, 0, 255), thickness=2)
                cv2.putText(image, f'x_dis: {int(x_displacement)}', (330,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
                cv2.putText(image, f'y_dis: {int(y_displacement)}', (330,110), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
                cv2.putText(image, f'{int(detection.score[0]*100)}%', (boundBox[0], boundBox[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 2)

                cv2.putText(image, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
                cv2.line(image, (center_point[0], 0), (center_point[0], h), (0, 255, 0), thickness=2)
                cv2.line(image, (0, center_point[1]), (w, center_point[1]), (0, 255, 0), thickness=2)
                cv2.imshow('Face Detection', image)

            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()