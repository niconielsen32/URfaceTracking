from facenet_pytorch import MTCNN
import torch
import numpy as np
import cv2
import time
import mediapipe as mp

import rtde_control
import rtde_receive


import csv


rtde_c = rtde_control.RTDEControlInterface("127.0.0.1")
rtde_r = rtde_receive.RTDEReceiveInterface("127.0.0.1")

#Load the video and go from frame to frame
cap = cv2.VideoCapture(0)


# Facenet
mtcnn = MTCNN(keep_all=True, device='cpu')

# MediaPipe
mp_facedetector = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils

# Haar cascade
face_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_alt2.xml')
#face_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')

# Data
x_displacement_ur = 0
y_displacement_ur = 0
area_of_bounding_box = []

fps = []
position_error = []
elapsed_time = 0
speed_of_face_on_ur = 0
frames_with_no_detection = 0


def facenet(frame):
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

            area_of_bounding_box.append(bbox_area)
    else:
        return False, False

    return x_displacement, y_displacement


def mediaPipe_face(image):
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

            area_of_bounding_box.append(bbox_area)

    else: 
        return False, False



# Defined start position
start_pos_ur = [1,1,1,1,1,1]
tcp_speed = 0.25
tcp_acc = 0.5
#rtde_c.moveL(start_pos_ur, tcp_speed, tcp_acc)

x_coor_ur = start_pos_ur[0]
y_coor_ur = start_pos_ur[1]


target = rtde_r.getActualTCPPose()

target[0] = x_coor_ur
target[1] = y_coor_ur




with mp_facedetector.FaceDetection(min_detection_confidence=0.7) as face_detection:
    while cap.isOpened():

        success, image = cap.read()

        start = time.time()


        # Facenet
        x_displacement_ur, y_displacement_ur = facenet(image)

        # Haar
        #x_displacement_ur, y_displacement_ur = haar(image)

        # Mediapipe
        #x_displacement_ur, y_displacement_ur = mediaPipe_face(image)


        end = time.time()
        totalTime = end - start

        fps = 1 / totalTime


        # Find init pos, find out if + or - displacement
        if x_displacement_ur is not False and y_displacement_ur is not False:
            x_coor_ur += x_displacement_ur
            y_coor_ur += y_displacement_ur

            target[0] = x_coor_ur
            target[1] = y_coor_ur

            rtde_c.moveL(target, tcp_speed, tcp_acc, True)

            # Maybe a sleep?
            #time.sleep(1)

        else:
            frames_with_no_detection += 1



        cv2.putText(image, f'x_dis: {int(x_displacement_ur)}', (330,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
        cv2.putText(image, f'y_dis: {int(y_displacement_ur)}', (330,110), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)

        cv2.putText(image, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
        #cv2.line(image, (center_point[0], 0), (center_point[0], h), (0, 255, 0), thickness=2)
        #cv2.line(image, (0, center_point[1]), (w, center_point[1]), (0, 255, 0), thickness=2)
        cv2.imshow('Face Detection', image)

        if cv2.waitKey(5) & 0xFF == 27:
            break


avg_fps = np.mean(np.asarray(fps))
avg_bb_area = np.mean(np.asarray(area_of_bounding_box))


# Only once
with open('dataUrTracking.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["fps", "frames_with_no_detection", "area_bb", "position error", "speed_of_face", "elapsed_tiime"])


# Values just for test - calculate and correct them
with open('dataUrTracking.csv', 'a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([avg_fps, frames_with_no_detection, avg_bb_area, position_error, speed_of_face_on_ur,elapsed_time])


cap.release()
cv2.destroyAllWindows()
# Stop the RTDE control script
rtde_c.stopScript()