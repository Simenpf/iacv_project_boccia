import math
import imutils
import cv2 as cv
import numpy as np
from configuration import * 
from projective_funcs import transform_point


def dilate_mask(mask):
    kernel = np.ones((5,5), np.uint8)
    return cv.dilate(mask, kernel, iterations=4)

def mask_balls(frame):
    hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # Range of included hues in both directions
    hue_range = 4

    # Blue ball HSV range
    low_blue  = np.array([blue_hue-hue_range,100,100])
    high_blue = np.array([blue_hue+hue_range,255,255])

    # Orange ball HSV range
    low_orange  = np.array([orange_hue-hue_range,100,100])
    high_orange = np.array([orange_hue+hue_range,255,255])

    # Green ball HSV range
    low_green  = np.array([green_hue-hue_range,100,100])
    high_green = np.array([green_hue+hue_range,255,255])

    # Red ball HSV range
    low_red  = np.array([red_hue-hue_range,100,100])
    high_red = np.array([red_hue+hue_range,255,255])

    # Yellow ball HSV range
    low_yellow  = np.array([yellow_hue-hue_range,100,100])
    high_yellow = np.array([yellow_hue+hue_range,255,255])

    # Create color masks
    mask_blue   = cv.inRange(hsv_frame, low_blue,   high_blue)
    mask_orange = cv.inRange(hsv_frame, low_orange, high_orange)
    mask_green  = cv.inRange(hsv_frame, low_green,  high_green)
    mask_red    = cv.inRange(hsv_frame, low_red,    high_red)
    mask_yellow = cv.inRange(hsv_frame, low_yellow, high_yellow)

    # Dilate color masks
    mask_blue   = dilate_mask(mask_blue)
    mask_orange = dilate_mask(mask_orange)
    mask_green  = dilate_mask(mask_green)
    mask_red    = dilate_mask(mask_red)
    mask_yellow = dilate_mask(mask_yellow)

    return [mask_blue, mask_orange, mask_green, mask_red, mask_yellow]


def detect_balls(frame, detection_scaling):
    # Prepare frame for detection
    frame_width = frame.shape[1]
    frame = cv.blur(frame, (15,15))
    frame = imutils.resize(frame, width=round(frame_width*detection_scaling))
  
    # Homography between true frame and downscaled frame
    H_detection_scaling = np.array([[1/detection_scaling, 0, 0], [0, 1/detection_scaling, 0], [0, 0, 1]])

    # Compute masks for frame
    frame_masks = mask_balls(frame)
    all_masks = cv.bitwise_or(frame_masks[0],frame_masks[1])
    all_masks = cv.bitwise_or(all_masks,frame_masks[2])
    all_masks = cv.bitwise_or(all_masks,frame_masks[3])
    all_masks = cv.bitwise_or(all_masks,frame_masks[4])


    balls_pos = [[-1,-1],[-1,-1],[-1,-1],[-1,-1],[-1,-1],[-1,-1],[-1,-1],[-1,-1],[-1,-1]]

    # For each color...
    for i in range(0,5):
        # Find contours in all the ball masks
        contours = cv.findContours(frame_masks[i], cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[-2]

        # Sort the contours from large to small areas
        contours = sorted(contours, key=cv.contourArea, reverse=True)

        # If color is not yellow
        if i != 4:
            # For one or two balls
            for j in range(0, min(2,len(contours))):
                    # Find the smallest circle covering the whole contour
                    (x,y), r = cv.minEnclosingCircle(contours[j])

                    if r > 5:
                        pos = np.array([x,y,1])
                        pos = transform_point(pos,H_detection_scaling)
                        balls_pos[i*2+j]= [round(pos[0]),round(pos[1])]
        else:
            if len(contours)>0:
                # Find the smallest circle covering the whole contour
                (x,y), r = cv.minEnclosingCircle(contours[0])

                if r > 5:
                    pos = np.array([x,y,1])
                    pos = transform_point(pos,H_detection_scaling)
                    balls_pos[8] = [round(pos[0]),round(pos[1])]


    # Prepare masks for visualization
    all_masks = cv.cvtColor(all_masks, cv.COLOR_GRAY2BGR)
    all_masks = imutils.resize(all_masks, width=frame_width)

    return all_masks, balls_pos

# To separate two balls of same colors we choose the positions that minimize the total distance 
# moved by the two balls
def estimate_ball_positions(pos,new_pos):
    for i in range(0,8,2):
        # Find the total ball movements for the two possible classifications
        move1 = math.dist(pos[i],new_pos[i])+math.dist(pos[i+1],new_pos[i+1])
        move2 = math.dist(pos[i],new_pos[i+1])+math.dist(pos[i+1],new_pos[i])

        # Switch the ball classifications if this reduces the total movement
        if move1 > move2:
            new_pos[i], new_pos[i+1] = new_pos[i+1], new_pos[i]
    return new_pos

def get_image_trajectories(game_video, H, court_ratio, frame_width, win_width, dt):
    # Perform game tracking
    pos_out_of_court = [-1,-1]
    ball_positions   = [[pos_out_of_court] for i in range(9)]
    ball_times       = [[0] for i in range(9)]
    tracked_frames   = [[] for i in range(9)]
    frame_index      = [0]*9
    t = dt

    while True:
        # Retrieve next frame of video-feed
        success, frame = game_video.read()
        for i in range(0,skip_frames):
            success, frame = game_video.read()


        # While there are frames in video-feed, run game-tracking
        if success:
            frame_copy = frame.copy()

            # Mask out court (+ padding), so detection is only done on court
            #frame = cv.bitwise_and(frame, frame ,mask = court_mask)

            # Track balls
            masks, new_ball_positions = detect_balls(frame, detection_scaling)

            # Extract current position (last column of 2d-array)
            current_ball_positions = [row[-1] for row in ball_positions]

            # Select order of same colored balls
            new_ball_positions = estimate_ball_positions(current_ball_positions,new_ball_positions)

            # Add trajectory information to outputs
            for i in range(0, len(new_ball_positions)):
                if new_ball_positions[i] != pos_out_of_court:
                    ball_positions[i].append(new_ball_positions[i])
                    tracked_frames[i].append(frame)
                    ball_times[i].append(t)
                    frame_index[i] += 1
                if len(ball_positions[i])>1:
                    # Draw a trail behind the ball
                    cv.polylines(frame_copy,[np.array(ball_positions[i][max(1,len(ball_positions[i])-20):-1])],False,ball_colors_bgr[i],4)

            # Rectify court
            rectified_frame = cv.warpPerspective(frame_copy, H, (frame_width, round(frame_width*court_ratio)))        
            rectified_masks = cv.warpPerspective(masks, H, (frame_width, round(frame_width*court_ratio)))

            # Display results
            #overview = np.concatenate((rectified_frame, rectified_masks), axis=1)
            overview = np.concatenate((frame_copy, masks), axis=1)
            overview = imutils.resize(overview, width=win_width)
            frame_small = imutils.resize(frame_copy, width=win_width)
            cv.imshow("press 'C' to capture the balls(press enter to exit...)", frame_small)

            # Quit if user presses enter
            if cv.waitKey(delay_time) == escape_key:
                break
        else:
            break
        t+=dt
    cv.destroyAllWindows()
    return ball_positions, ball_times, tracked_frames
