import numpy as np

# Media
court_reference_frame_path = "../media/bocce_game_camera2_reference_frame.jpg"
calibration_video_path     = '../media/calibration_video_camera2.mp4'
game_video_path            = '../../trimmed_trajs.mp4'
#game_video_path = '../media/blue_ball_trimmed.mp4'

# Pre-calculated values
corners_auto = [[1430, 598],[1657, 837],[210, 875],[420, 620]]
P_auto = np.array([[-2.64369884e+00, -1.23577043e+00, -4.71130751e-01,  1.43258252e+03],
                   [-4.78026920e-03, -1.95656940e-02, -2.74553103e+00,  5.99706109e+02],
                   [-8.11727165e-05, -1.41629246e-03, -4.86481859e-04,  1.00000000e+00]])

# Calibration configs
board_size = (9,7)
square_size = 2
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
number_of_balls = 9

team_1_ball_indexes = [0,1,2,3]
team_2_ball_indexes = [4,5,6,7]
ball_score = 10 # points given for each ball closer than opposite team

# Hue of the balls (Found experimentally)
blue_hue   = 95
orange_hue = 17
green_hue  = 47
red_hue    = 176 
yellow_hue = 30

# Colors of the balls in (B,G,R)
ball_colors_bgr = [(200,170,80),(200,170,80),(50,160,240),(50,160,240),(60,220,60),(60,220,60),(0,0,255),(0,0,255),(0,230,255)]

escape_key = 13
delay_time = 25

