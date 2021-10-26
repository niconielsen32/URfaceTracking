
from os import path
import numpy as np
import cv2
import time

#mport rtde_control
#import rtde_receive

#rtde_c = rtde_control.RTDEControlInterface("127.0.0.1")
#rtde_r = rtde_receive.RTDEReceiveInterface("127.0.0.1")

# Defined start position
start_pos_ur = [1,1,1,1,1,1]
tcp_speed = 0.25
tcp_acc = 0.5
#rtde_c.moveL(start_pos_ur, tcp_speed, tcp_acc)

x_coor_ur = start_pos_ur[0]
y_coor_ur = start_pos_ur[1]



##################################################################
# SQUARE
##################################################################

def runSquarePattern():
    #target = rtde_r.getActualTCPPose()
    target = [1,1]

    target[0] = x_coor_ur
    target[1] = y_coor_ur

    path_ur = []

    displacement = 0.2

    path_pose_1 = target
    path_pose_1[0] += displacement
    path_ur.append(np.asarray(path_pose_1))

    path_pose_2 = path_pose_1
    path_pose_2[1] -= displacement
    print(path_pose_2)
    path_ur.append(np.asarray(path_pose_2))

    path_pose_3 = path_pose_2
    path_pose_3[0] -= displacement
    print(path_pose_3)
    path_ur.append(np.asarray(path_pose_3))

    path_pose_4 = path_pose_3
    path_pose_4[1] += displacement
    path_ur.append(np.asarray(path_pose_4))

    print(path_ur)

    # Send a linear path with blending in between - (currently uses separate script)
    #rtde_c.moveL(path_ur, tcp_speed, tcp_acc)
    #rtde_c.stopScript()


##################################################################
# CIRCLE
##################################################################

def runCirclePattern():
    #target = rtde_r.getActualTCPPose()
    target = [1,1]

    radius = 0.5

    path_ur_circle = []
    pose = target

    for theta in range(360):
        x = radius * np.cos(np.deg2rad(theta))
        
        y = radius * np.sin(np.deg2rad(theta))
        x_coor = pose[0] + x
        y_coor = pose[1] + y
        #print(pose)
        path_ur_circle.append([x_coor, y_coor])
        #path_ur_circle.append([x,y,1,1,1,1])

    #print(path_ur_circle)

    for i in range(360):
        print(path_ur_circle[i])
    # Send a linear path with blending in between - (currently uses separate script)
    #rtde_c.moveL(path_ur_circle, tcp_speed, tcp_acc)
    #rtde_c.stopScript()


runSquarePattern()
#runCirclePattern()



