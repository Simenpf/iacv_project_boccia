import imutils
import cv2 as cv
import numpy as np
from rectify_court import get_court_homography
from blob_detection import detect_balls
from math import dist

# To separate two balls of same colors we choose the positions that minimize the total distance 
# moved by the two balls
def estimate_ball_positions(pos,new_pos):
    for i in range(0,8,2):
        move1 = dist(pos[i],new_pos[i])+dist(pos[i+1],new_pos[i+1])
        move2 = dist(pos[i],new_pos[i+1])+dist(pos[i+1],new_pos[i])
        if move1 > move2:
            new_pos[i], new_pos[i+1] = new_pos[i+1], new_pos[i]
    return new_pos
        

# Load video feed and first frame
video = cv.VideoCapture('../media/trimmed_game3_no_ornaments.mp4')
#video = cv.VideoCapture(0) # For testing with webcam
_, frame = video.read()

# Frame resolution
frame_width  = frame.shape[1]
frame_height = frame.shape[0]

# Detection frame scaling (0 to 1, resolution-ratio of frame sent to ball detection algorithm)
detection_scaling = 0.5

# For faster tracking some frames can be skipped (set to zero for tracking all frames)
skip_frames = 0

# Screen resolution
#screen_width  = int(input("Screen width (pixels): "))
#screen_height = int(input("Screen length (pixels): "))
screen_width  = 1920
screen_height = 1080

# Window resolution
win_width  = round(screen_width*0.5)
win_height = round(screen_height*0.5)

# Court size
#court_width  = int(input("Court width (mm): "))
#court_length = int(input("Court length (mm): "))
court_length = 2210
court_width = 1500
court_ratio = court_width/court_length


# Get court rectifying homography
padding = 0.1 # The amount of image used from outside court (in fraction of court length)
H, court_mask = get_court_homography(frame, court_ratio, win_width, padding)

# Perform game tracking
pos_out_of_court = (-1,-1)
ball_positions = [[(-1,-1)],[(-1,-1)],[(-1,-1)],[(-1,-1)],[(-1,-1)],[(-1,-1)],[(-1,-1)],[(-1,-1)],[(-1,-1)]]
ball_colors = [(200,170,80),(200,170,80),(50,160,240),(50,160,240),(60,220,60),(60,220,60),(0,0,255),(0,0,255),(0,230,255)]
while True:
    # Retrieve next frame of video-feed
    success, frame = video.read()
    for i in range(0,skip_frames):
        success, frame = video.read()


    # While there are frames in video-feed, run game-tracking
    if success:
        # Mask out court (+ padding), so detection is only done on court
        frame = cv.bitwise_and(frame, frame ,mask = court_mask)

        # Track balls
        masks, new_ball_positions = detect_balls(frame, detection_scaling)
        current_ball_positions = [row[-1] for row in ball_positions]
        new_ball_positions = estimate_ball_positions(current_ball_positions,new_ball_positions)
        for i in range(0, len(new_ball_positions)):
            if new_ball_positions[i] != pos_out_of_court:
                ball_positions[i].append(new_ball_positions[i])
            if len(ball_positions[i])>1:
                cv.polylines(frame,[np.array(ball_positions[i][max(1,len(ball_positions[i])-20):-1])],False,ball_colors[i],4)

        # Rectify court
        rectified_frame = cv.warpPerspective(frame, H, (frame_width, round(frame_width*court_ratio)))        
        rectified_masks = cv.warpPerspective(masks, H, (frame_width, round(frame_width*court_ratio)))

        # Display results
        overview = np.concatenate((rectified_frame, rectified_masks), axis=1)
        #overview = np.concatenate((frame, masks), axis=1)
        overview = imutils.resize(overview, width=win_width)
        cv.imshow("Rectified Court (press enter to exit...)", overview)

        # Quit if user presses enter
        if cv.waitKey(25) == 13:
            break
    else:
        break

# Clean workspace
video.release()
cv.destroyAllWindows()

