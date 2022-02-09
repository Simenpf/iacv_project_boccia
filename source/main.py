import cv2 as cv
import numpy as np
from configuration import *
from visualization import plot_trajectory
from rectify_court import get_court_homography
from ball_detection import get_image_trajectories
from reconstruction_3d import generate_3d_trajectory, select_bounces
from camera_calibration import getCameraIntrinsics, getCameraProjectionMatrix

# Window resolution
win_width  = round(screen_width*0.6)
win_height = round(screen_height*0.6)

# Load reference frame
court_reference_frame = cv.imread(court_reference_frame_path)
frame_width  = court_reference_frame.shape[1]
frame_height = court_reference_frame.shape[0]

# Prepare video with parameters
game_video = cv.VideoCapture(game_video_path)
fps = game_video.get(cv.CAP_PROP_FPS)
dt = 1/fps

# Get court rectifying homography
court_ratio = court_width/court_length
H, court_mask, corners_selected = get_court_homography(court_reference_frame, court_ratio, win_width, padding)
#_, court_mask, _ = get_court_homography(court_reference_frame, court_ratio, win_width, padding)

# Camera Calibration
calibration_video = cv.VideoCapture(calibration_video_path)
corners_selected = corners_selected[0]
corners_actual = np.array([[0, 0, 0],[0, court_length, 0],[court_width, court_length, 0],[court_width, 0, 0]],dtype=np.float32)
K, dist = getCameraIntrinsics(calibration_video,board_size,square_size)
P = getCameraProjectionMatrix(K,dist,corners_actual,corners_selected)

# Track balls in image
ball_positions, ball_times, start_keypoint = get_image_trajectories(game_video, H, court_ratio, frame_width, win_width, dt)

# Clean workspace
game_video.release()

# Select keypoints from bouncing manually
keypoints = select_bounces(court_reference_frame,start_keypoint, np.array(ball_positions[1][start_keypoint:-1]),win_width)

# 3D Trajectory estimation
traj_2d = []
t = []

for i in range(1,len(keypoints)):
    traj_2d.append(ball_positions[1][keypoints[i-1]:keypoints[i]+1])
    t.append(ball_times[1][keypoints[i-1]:keypoints[i]+1])
    t[i-1]=[t_k - t[i-1][0] for t_k in t[i-1]]

traj_3d = np.array(generate_3d_trajectory(P, traj_2d[0], t[0])) # Should also return times
for i in range(1,len(traj_2d)):
    traj_3d = np.concatenate((traj_3d,generate_3d_trajectory(P, traj_2d[i], t[i])),axis=0)

# Plot results
plot_trajectory(traj_3d,corners_actual,court_width,court_length)

