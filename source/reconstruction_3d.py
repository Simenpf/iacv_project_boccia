import imutils
import cv2 as cv
import numpy as np
from configuration import g, delay_time, escape_key


def generate_3d_trajectory(P, traj_2d, t):    
    A = calculate_A(P,traj_2d,t)
    a = calculate_a(P,traj_2d,t)
    q = init_states(A,a)
    return real_3d_coordinate(q,traj_2d,t)


def calculate_A(P, traj_2d, t):
    n = 2*len(traj_2d)

    A_0 = [0]*n
    A_1 = [0]*n
    A_2 = [0]*n
    A_3 = [0]*n
    A_4 = [0]*n
    A_5 = [0]*n
    for j in range(0, len(traj_2d)):
        for i in range(0, 2):
            A_0[2*j+i] = P[i, 0]      - traj_2d[j][i]*P[2,0]
            A_1[2*j+i] = P[i, 0]*t[j] - traj_2d[j][i]*P[2,0]*t[j]
            A_2[2*j+i] = P[i, 1]      - traj_2d[j][i]*P[2,1]
            A_3[2*j+i] = P[i, 1]*t[j] - traj_2d[j][i]*P[2,1]*t[j]
            A_4[2*j+i] = P[i, 2]      - traj_2d[j][i]*P[2,2]
            A_5[2*j+i] = P[i, 2]*t[j] - traj_2d[j][i]*P[2,2]*t[j]

    D = np.array([A_0, A_1, A_2, A_3, A_4, A_5],dtype=np.float32)
    D = D.T
    return D

def calculate_a(P, traj_2d, t):
    n = 2*len(traj_2d)
    a = np.array([0]*n,dtype=np.float32)
    for j in range(0, len(traj_2d)):
        for i in range(0, 2):
            a[2*j+i] = traj_2d[j][i]*(0.5*P[2, 2]*g*pow(t[j],2)+1)-(0.5*P[i,2]*g*pow(t[j],2)+P[i,3])
    return a

def init_states(A,a):
    q = np.array(np.dot(np.linalg.pinv(A),a))
    return q

# returns real world coordinates as list of X coordinates, list of Y coordinates, list of Z coordinates
def real_3d_coordinate(q, traj_2d, t):
    n = len(traj_2d)
    X = [0]*n
    Y = [0]*n
    Z = [0]*n
    traj_3d = [0]*n
    for i in range(0, n):
        X[i] = q[0] + q[1]*t[i]
        Y[i] = q[2] + q[3]*t[i]
        Z[i] = q[4] + q[5]*t[i] + .5*g*t[i]**2
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
            cv.imshow("tap 'C' to capture a keypoint, tap 'N'(ext) and 'B'(ack) to go back and fourth in the trajectory",frame_copy)


            key = cv.waitKey(delay_time)
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
            if key == escape_key:
                break

    cv.destroyAllWindows()
    return keypoints
            
def get_all_3d_segements(ball_positions,ball_times, all_keypoints,P):
    traj_2d = [[] for i in range(9)]
    traj_3d = [[] for i in range(9)]
    t = [[] for i in range(9)]
    for ball in range(0,9):
        keypoints = all_keypoints[ball]
        for i in range(1,len(keypoints)):
            traj_2d[ball].append(ball_positions[ball][keypoints[i-1]:keypoints[i]+1])
            t[ball].append(ball_times[ball][keypoints[i-1]:keypoints[i]+1])
            t[ball][i-1]=[t_k - t[ball][i-1][0] for t_k in t[ball][i-1]]

        if len(keypoints)>0:
            traj_3d[ball] = np.array(generate_3d_trajectory(P, traj_2d[ball][0], t[ball][0])) # Should also return times
            for i in range(1,len(traj_2d[ball])):
                traj_3d[ball] = np.concatenate((traj_3d[ball],generate_3d_trajectory(P, traj_2d[ball][i], t[ball][i])),axis=0)
    return traj_3d
