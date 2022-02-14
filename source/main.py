import cv2 as cv
import numpy as np
from configuration import *
from rectify_court import get_court_homography
from visualization import plot_trajectory, plot_game
from ball_detection import get_image_trajectories, get_court_mask
from reconstruction_3d import select_bounces, get_all_3d_segements
from camera_calibration import getCameraIntrinsics, getCameraProjectionMatrix
from projective_funcs import create_rectified_position_vector, get_radius_range


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

# Get court rectifying homography from user clicks
H, corners_selected = get_court_homography(court_reference_frame, win_width)

# Get a mask of the area to search for the balls from user clicks
court_mask = get_court_mask(court_reference_frame,win_width)

# Camera Calibration
calibration_video = cv.VideoCapture(calibration_video_path)
corners_actual = np.array([[0, 0, 0],[0, court_length, 0],[court_width, court_length, 0],[court_width, 0, 0]],dtype=np.float32)
K, distortion_parameters = getCameraIntrinsics(calibration_video,board_size,square_size)
P = getCameraProjectionMatrix(K,distortion_parameters,corners_actual,corners_selected)

# Get ball radius range
court_furthest_point = -2*court_length
court_closest_point = court_length
r_min, r_max = get_radius_range(ball_radius,P,court_furthest_point,court_closest_point)
radius_padding = 0.4
r_min = round(r_min*(1-radius_padding))
r_max = round(r_max*(1+radius_padding))

# Track balls in video
court_ratio = court_width/court_length
ball_positions_detected_im, ball_times, tracked_frames, ball_positions_im  = get_image_trajectories(game_video, H, court_ratio, frame_width, win_width, dt,court_mask,r_min,r_max)
game_video.release()

# Transform ball positions from image space to 2D-space
ball_positions = create_rectified_position_vector(ball_positions_im, H)

# Display the 2D game within the goal region
plot_game(ball_positions)

# Manually select start and end points of parabolas
all_keypoints = select_bounces(tracked_frames, ball_positions_detected_im,win_width)

# Estimate 3D trajectories of the balls in the moments selected by user
traj_3d = get_all_3d_segements(ball_positions_detected_im, ball_times, all_keypoints, P)

# Plot the results of the 3D trajectory estimation
plot_trajectory(traj_3d,corners_actual)