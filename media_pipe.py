import cv2
import mediapipe as mp
import time

def mediapipe(cap, mp_facedetector, mp_draw, face_detection, write = "False", filename="image.png"):

    success, image = cap.read()
    error = False
    start = time.time()

    end = start
    totalTime = 0
    fps = 0
    error = False
    bbox_area = 0
    x_displacement = 0
    y_displacement = 0

    # Convert the BGR image to RGB
    if (not image is None):
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

            if(write):
                cv2.imwrite(filename, image)
        
        else:
            error = True
    else:
        error = True
        
    return x_displacement, y_displacement, totalTime, bbox_area, fps, totalTime, error


if __name__ == "__main__":

    cap = cv2.VideoCapture(0)
    
    x_displacement = 0
    y_displacement = 0
    totalTime = 0
    bbox_area = 0
    fps = 0
    totalTime = 0

    mp_facedetector = mp.solutions.face_detection
    mp_draw = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(0)

    with mp_facedetector.FaceDetection(min_detection_confidence=0.7) as face_detection:
        while (True):
            x_displacement, y_displacement, totalTime, bbox_area, fps, totalTime, error = mediapipe(cap, mp_facedetector=mp_facedetector, mp_draw=mp_draw, face_detection=face_detection, write = True)

    cap.release()
    cv2.destroyAllWindows()