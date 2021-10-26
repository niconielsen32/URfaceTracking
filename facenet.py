#You can install facenet using PIP by typing "pip install facenet-pytorch"
#Import the required modules
from facenet_pytorch import MTCNN
import torch
import cv2
import time

 
def facenet(cap, write=False, filename="image.png", show=False):
    ret, frame = cap.read()
    start = time.time()
    frame = cv2.resize(frame, (600, 400))
    height, width, channels = frame.shape
    center_point = (int(width / 2), int(height / 2))
    end = start
    totalTime = 0
    fps = 0
    error = False
    bbox_area = 0
    x_displacement = 0
    y_displacement = 0

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
        if show:
            cv2.imshow("Frame", frame)
        if write:
            cv2.imwrite(filename, frame)
    else:
        error=True
    
    return x_displacement, y_displacement, totalTime, bbox_area, fps, totalTime, error
 
if __name__ == "__main__":
    #Create the model
    mtcnn = MTCNN(keep_all=True, device='cpu')
    
    #Load the video and go from frame to frame
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        x_displacement, y_displacement, totalTime, bbox_area, fps, totalTime, error = facenet(cap, write=True)

    cap.release()
    cv2.destroyAllWindows()