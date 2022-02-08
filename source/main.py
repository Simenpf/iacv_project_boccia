from turtle import pos
import imutils
import cv2 as cv
import numpy as np
from math import dist
from blob_detection import detect_balls
from visualization import plot_trajectory
from rectify_court import get_court_homography
from reconstruction_3d import generate_3d_trajectory, get_E
from camera_calibration import getCameraIntrinsics, getCameraProjectionMatrix

# To separate two balls of same colors we choose the positions that minimize the total distance 
# moved by the two balls
def estimate_ball_positions(pos,new_pos):
    for i in range(0,8,2):
        move1 = dist(pos[i],new_pos[i])+dist(pos[i+1],new_pos[i+1])
        move2 = dist(pos[i],new_pos[i+1])+dist(pos[i+1],new_pos[i])
        if move1 > move2:
            new_pos[i], new_pos[i+1] = new_pos[i+1], new_pos[i]
    return new_pos

# For manual selection by user of the balls bounces at the ground
def select_bounces(frame, start_keypoint, traj_2d, win_width):
    i = 0
    cv.polylines(frame,[traj_2d],False,(200,170,80),4)
    keypoints = [start_keypoint]
    while True:
        frame_copy = frame.copy()
        cv.circle(frame_copy,traj_2d[i],10,(0,0,255),-1)
        for k in keypoints:
            cv.circle(frame_copy,traj_2d[k-start_keypoint],10,(0,255,0),-1)

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
            keypoints.append(i+start_keypoint)
        # Exit selection
        if key == 13:
            break

    cv.destroyAllWindows()
    return keypoints
            

# Load videofeed and reference frame
video = cv.VideoCapture('../media/blue_ball_trimmed.mp4')
fps = video.get(cv.CAP_PROP_FPS)
dt = 1/fps

#video = cv.VideoCapture(0) # For testing with webcam
#_, frame = video.read()
frame = cv.imread("../media/bocce_game_camera2_reference_frame.jpg")

# Frame resolution
frame_width  = frame.shape[1]
frame_height = frame.shape[0]

# Detection frame scaling (0 to 1, resolution-ratio of frame sent to ball detection algorithm)
detection_scaling = 0.2

# For faster tracking some frames can be skipped (set to zero for tracking all frames)
skip_frames = 0

# Screen resolution
screen_width  = 1920
screen_height = 1080

# Window resolution
win_width  = round(screen_width*0.6)
win_height = round(screen_height*0.6)

# Court size
court_length = 202
court_width = 390
court_ratio = court_width/court_length

# Get court rectifying homography
padding = 0.1 # The amount of image used from outside court (in fraction of court length)
H, court_mask, corners_selected = get_court_homography(frame, court_ratio, win_width, padding)

# Camera Calibration
board_size = (9,7)
square_size = 2
calibration_video = cv.VideoCapture('../media/calibration_video_camera2.mp4')
corners_selected = corners_selected[0]
corners_actual = np.array([[0, 0, 0],[0, 202, 0],[court_width, court_length, 0],[court_width, 0, 0]],dtype=np.float32)
#K, dist = getCameraIntrinsics(calibration_video,board_size,square_size)
#P = getCameraProjectionMatrix(K,dist,corners_actual,corners_selected)

P = np.array([[-2.64369884e+00, -1.23577043e+00, -4.71130751e-01,  1.43258252e+03],
 [-4.78026920e-03, -1.95656940e-02, -2.74553103e+00,  5.99706109e+02],
 [-8.11727165e-05, -1.41629246e-03, -4.86481859e-04,  1.00000000e+00]])


# Perform game tracking
pos_out_of_court = [-1,-1]
ball_positions = [[pos_out_of_court] for i in range(9)]
ball_t = [[0] for i in range(9)]
ball_colors = [(200,170,80),(200,170,80),(50,160,240),(50,160,240),(60,220,60),(60,220,60),(0,0,255),(0,0,255),(0,230,255)]
frame_index = [0]*9
start_keypoint = None
t = dt
while True:
    # Retrieve next frame of video-feed
    success, frame = video.read()
    for i in range(0,skip_frames):
        success, frame = video.read()


    # While there are frames in video-feed, run game-tracking
    if success:
        # Mask out court (+ padding), so detection is only done on court
        #frame = cv.bitwise_and(frame, frame ,mask = court_mask)

        # Track balls
        masks, new_ball_positions = detect_balls(frame, detection_scaling)
        current_ball_positions = [row[-1] for row in ball_positions]
        new_ball_positions = estimate_ball_positions(current_ball_positions,new_ball_positions)
        for i in range(0, len(new_ball_positions)):
            if new_ball_positions[i] != pos_out_of_court:
                frame_index[i] += 1
                ball_positions[i].append(new_ball_positions[i])
                ball_t[i].append(t)
            if len(ball_positions[i])>1:
                #cv.polylines(frame,[np.array(ball_positions[i][max(1,len(ball_positions[i])-20):-1])],False,ball_colors[i],4)
                cv.polylines(frame,[np.array(ball_positions[i][1:-1])],False,ball_colors[i],4)

        # Rectify court
        rectified_frame = cv.warpPerspective(frame, H, (frame_width, round(frame_width*court_ratio)))        
        rectified_masks = cv.warpPerspective(masks, H, (frame_width, round(frame_width*court_ratio)))

        # Display results
        #overview = np.concatenate((rectified_frame, rectified_masks), axis=1)
        overview = np.concatenate((frame, masks), axis=1)
        overview = imutils.resize(overview, width=win_width)
        frame_small = imutils.resize(frame, width=win_width)
        cv.imshow("Rectified Court (press enter to exit...)", frame_small)

        # Quit if user presses enter
        key = cv.waitKey(25)
        if key == 13:
            break
        if key == ord('c'):
            start_keypoint = frame_index[1]
    else:
        break
    t+=dt

# Clean workspace
video.release()
cv.destroyAllWindows()

# Select keypoints from bouncing manually
frame = cv.imread("../media/bocce_game_camera2_reference_frame.jpg")
keypoints = select_bounces(frame,start_keypoint, np.array(ball_positions[1][start_keypoint:-1]),win_width)


# Trajectory estimation
traj_2d = []
t = []

for i in range(1,len(keypoints)):
    traj_2d.append(ball_positions[1][keypoints[i-1]:keypoints[i]+1])
    t.append(ball_t[1][keypoints[i-1]:keypoints[i]+1])
    t[i-1]=[t_k - t[i-1][0] for t_k in t[i-1]]

traj_3d = np.array(generate_3d_trajectory(P, traj_2d[0], t[0]))
for i in range(1,len(traj_2d)):
    traj_3d = np.concatenate((traj_3d,generate_3d_trajectory(P, traj_2d[i], t[i])),axis=0)

# Plot results
plot_trajectory(traj_2d,traj_3d,corners_actual,court_width,court_length)

