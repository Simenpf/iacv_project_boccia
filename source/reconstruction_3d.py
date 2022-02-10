import imp
import imutils
import cv2 as cv
import numpy as np
from configuration import g


def generate_3d_trajectory(P, traj_2d, t):    
    D = get_some_D(P,traj_2d,t)
    F = give_some_F(P,traj_2d,t)
    E = init_states(D,F)
    return real_3d_coordinate(E,traj_2d,t)


def get_some_D(P, traj_2d, t):
    n = 2*len(traj_2d)
    
    d_0 = [0]*n
    d_1 = [0]*n
    d_2 = [0]*n
    d_3 = [0]*n
    d_4 = [0]*n
    d_5 = [0]*n
    for j in range(0, len(traj_2d)):
        for i in range(0, 2):
            d_0[2*j+i] = P[i, 0]      - traj_2d[j][i]*P[2,0]
            d_1[2*j+i] = P[i, 0]*t[j] - traj_2d[j][i]*P[2,0]*t[j]
            d_2[2*j+i] = P[i, 1]      - traj_2d[j][i]*P[2,1]
            d_3[2*j+i] = P[i, 1]*t[j] - traj_2d[j][i]*P[2,1]*t[j]
            d_4[2*j+i] = P[i, 2]      - traj_2d[j][i]*P[2,2]
            d_5[2*j+i] = P[i, 2]*t[j] - traj_2d[j][i]*P[2,2]*t[j]

    D = np.array([d_0, d_1, d_2, d_3, d_4, d_5],dtype=np.float32)
    D = D.T
    return D

def give_some_F(P, traj_2d, t):
    n = 2*len(traj_2d)
    F = np.array([0]*n,dtype=np.float32)
    for j in range(0, len(traj_2d)):
        for i in range(0, 2):
            F[2*j+i] = traj_2d[j][i]*(0.5*P[2, 2]*g*pow(t[j],2)+1)-(0.5*P[i,2]*g*pow(t[j],2)+P[i,3])
    return F

def init_states(D,F):
    E = np.array(np.dot(np.linalg.pinv(D),F))
    return E

# returns real world coordinates as list of X coordinates, list of Y coordinates, list of Z coordinates
def real_3d_coordinate(E, traj_2d, t):
    n = len(traj_2d)
    X = [0]*n
    Y = [0]*n
    Z = [0]*n
    traj_3d = [0]*n
    for i in range(0, n):
        X[i] = E[0] + E[1]*t[i]
        Y[i] = E[2] + E[3]*t[i]
        Z[i] = E[4] + E[5]*t[i] + .5*g*t[i]**2
        traj_3d[i] = [X[i], Y[i], Z[i]]
    return np.array(traj_3d)

# For manual selection by user of the balls bounces at the ground
def select_bounces(all_tracked_frames, all_traj_2d, win_width):
    keypoints = [[] for i in range(9)]
    for ball in range(0,len(keypoints)):
        traj_2d = np.array(all_traj_2d[ball][1:-1])
        tracked_frames = all_tracked_frames[ball]
        i = 0
        while True and len(tracked_frames)>20:
            frame_copy = tracked_frames[i].copy()
            cv.polylines(frame_copy,[traj_2d],False,(200,170,80),4)
            cv.circle(frame_copy,traj_2d[i],10,(0,0,255),-1)
            for k in keypoints[ball]:
                cv.circle(frame_copy,traj_2d[k-1],10,(0,255,0),-1)

            frame_copy = imutils.resize(frame_copy,width=win_width)
            cv.imshow("",frame_copy)


            key = cv.waitKey(25)
            # Go (b)ack in trajectory
            if key == ord('b'):
                i = max(0,i-1)
            # Go (n)ext in trajectory
            if key == ord('n'):
                i = min(len(traj_2d)-1,i+1)
            # (C)apture keyframe
            if key == ord('c'):
                keypoints[ball].append(i+1)
            # Exit selection
            if key == 13:
                break

    cv.destroyAllWindows()
    return keypoints
            
