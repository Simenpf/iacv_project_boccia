import numpy as np

# Camera Matrix for camera 2
P = np.array([[-2.64369884e+00, -1.23577043e+00, -4.71130751e-01,  1.43258252e+03],
 [-4.78026920e-03, -1.95656940e-02, -2.74553103e+00,  5.99706109e+02],
 [-8.11727165e-05, -1.41629246e-03, -4.86481859e-04,  1.00000000e+00]])

# Corners of goal region for camera 1
corners_selected_1 = [[1453, 608],[1700, 863],[163, 890],[393, 627]]

# Corners of goal region for camera 2
corners_selected_2 = [[1430, 598],[1657, 837],[210, 875],[420, 620]]

# Media
court_reference_frame_path = "../media/bocce_game_camera2_reference_frame.jpg"
game_video_path = '../../trimmed_trajs.mp4'

# Calibration configs
board_size = (9,7)
square_size = 2
calibration_video_path = '../media/calibration_video_camera2.mp4'
num_calibration_imgs = 30

# Detection
detection_scaling = 1 # Detection frame scaling (0 to 1, resolution-ratio of frame sent to ball detection algorithm)
skip_frames = 0 # !!! Will ruin 3d estimation if !=0 !!! For faster tracking some frames can be skipped (set to zero for tracking all frames)

# Screen resolution
screen_width  = 1920
screen_height = 1080

# Court size (in cm)
court_length = 202
court_width = 390

padding = 0 # The amount of image used from outside court (in fraction of court length)

# Gravity constant (in cm/ms^2)
g = -981

# Game play 
team_1_ball_indexes = [0,1,2,3]
team_2_ball_indexes = [4,5,6,7]
