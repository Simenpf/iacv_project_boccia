import cv2 as cv
import numpy as np
import imutils
from configuration import *
#from visualization import animate_score_board
#from game_scores import calculate_score
from visualization import plot_trajectory
from rectify_court import get_court_homography
from ball_detection import get_image_trajectories, get_court_mask
from reconstruction_3d import select_bounces, get_all_3d_segements
from camera_calibration import getCameraIntrinsics, getCameraProjectionMatrix
from projective_funcs import transform_point, create_rectified_position_vector

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
H, corners_selected = get_court_homography(court_reference_frame, win_width)
court_mask = get_court_mask(court_reference_frame,win_width)

# Camera Calibration
calibration_video = cv.VideoCapture(calibration_video_path)
corners_selected = corners_selected[0]
corners_actual = np.array([[0, 0, 0],[0, court_length, 0],[court_width, court_length, 0],[court_width, 0, 0]],dtype=np.float32)
#K, dist = getCameraIntrinsics(calibration_video,board_size,square_size)
#P = getCameraProjectionMatrix(K,dist,corners_actual,corners_selected)
P = P_auto

# Track balls in video
ball_positions_im, ball_times, tracked_frames = get_image_trajectories(game_video, H, court_ratio, frame_width, win_width, dt,court_mask)

# Clean workspace
game_video.release()

# Manually select start and end points of parabolas
all_keypoints = select_bounces(tracked_frames, ball_positions_im,win_width)


# 3D Trajectory estimation
traj_3d = get_all_3d_segements(ball_positions_im, ball_times, all_keypoints, P)


# Plot results
plot_trajectory(traj_3d,corners_actual)

# Calculate scores
#ball_positions = create_rectified_position_vector(ball_positions_im)
#animate_score_board(ball_positions)

