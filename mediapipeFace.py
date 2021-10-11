import cv2
import mediapipe as mp
import time

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