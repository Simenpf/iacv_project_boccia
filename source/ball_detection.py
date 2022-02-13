import math
import imutils
import cv2 as cv
import numpy as np
from configuration import * 
from projective_funcs import transform_point


# Returns masks for each ball color from the given image
def mask_balls(frame):
    # Convert frame from BGR format to HSV format
    hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

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
    low_yellow  = np.array([yellow_hue-hue_range*2,50,100])
    high_yellow = np.array([yellow_hue+hue_range*2,255,255])

    # Create color masks
    mask_blue   = cv.inRange(hsv_frame, low_blue,   high_blue)
    mask_orange = cv.inRange(hsv_frame, low_orange, high_orange)
    mask_green  = cv.inRange(hsv_frame, low_green,  high_green)
    mask_red    = cv.inRange(hsv_frame, low_red,    high_red)
    mask_yellow = cv.inRange(hsv_frame, low_yellow, high_yellow)

    return [mask_blue, mask_orange, mask_green, mask_red, mask_yellow]


# Returns a value from 0 to 1 indicating the circularity of the given contour
# The calculation is based on the known realtion between perimeter and area of
# a circle
def circularity(con):
    perimeter = cv.arcLength(con, True)
    area = cv.contourArea(con)
    if perimeter == 0:
        return 0
    else:
        return 4*math.pi*(area/(perimeter**2))  

def detect_balls(frame, detection_scaling, r_min, r_max):
    # Downscale radius ranges
    r_min = detection_scaling*r_min
    r_max = detection_scaling*r_max

    # Prepare frame for detection
    frame_width = frame.shape[1]
    frame = imutils.resize(frame, width=round(frame_width*detection_scaling))
  
    # Homography between true frame and downscaled frame
    H_detection_scaling = np.array([[1/detection_scaling, 0, 0], [0, 1/detection_scaling, 0], [0, 0, 1]])

    # Compute masks for frame
    frame_masks = mask_balls(frame)
    all_masks = cv.bitwise_or(frame_masks[0],frame_masks[1])
    all_masks = cv.bitwise_or(all_masks,frame_masks[2])
    all_masks = cv.bitwise_or(all_masks,frame_masks[3])
    all_masks = cv.bitwise_or(all_masks,frame_masks[4])


    # Initialize ball positions
    balls_pos = [[-1,-1],[-1,-1],[-1,-1],[-1,-1],[-1,-1],[-1,-1],[-1,-1],[-1,-1],[-1,-1]]

    # For each color...
    for i in range(0,5):
        # Find contours in all the ball masks
        contours = cv.findContours(frame_masks[i], cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[-2]


        # If color is not yellow
        if i != 4:
            # Remove from list if not in radius range
            contours_in_range = []
            for con in contours:
                (x,y), r = cv.minEnclosingCircle(con)
                if r_min < r < r_max:
                    contours_in_range.append(con)


            # Sort contours on circularity
            contours = sorted(contours_in_range, key=circularity, reverse=True)

            # For one or two balls
            for j in range(0, min(2,len(contours))):

                    # Discard if circularity is below threshold
                    if circularity(contours[j]) > circularity_threshold:
                        (x,y), r = cv.minEnclosingCircle(contours[j])
                        pos = np.array([x,y,1])
                        pos = transform_point(pos,H_detection_scaling)
                        balls_pos[i*2+j]= [round(pos[0]),round(pos[1])]

        # If color is yellow
        else:
            # Remove from list if not in radius range
            contours_in_range = []
            for con in contours:
                (x,y), r = cv.minEnclosingCircle(con)
                if r_min*0.5 < r < r_max*0.5:
                    contours_in_range.append(con)
                
    
            # Sort contours on circularity
            contours = sorted(contours_in_range, key=circularity, reverse=True)
            if len(contours)>0:
                if circularity(contours[0]) > circularity_threshold:
                    (x,y), r = cv.minEnclosingCircle(contours[0])
                    pos = np.array([x,y,1])
                    pos = transform_point(pos,H_detection_scaling)
                    balls_pos[-1] = [round(pos[0]),round(pos[1])]


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

# Performs the complete ball detection, returning trajectories for all balls in image space
def get_image_trajectories(game_video, H, court_ratio, frame_width, win_width, dt, court_mask, r_min, r_max):
    # Initialization
    pos_out_of_court = [-1,-1]
    ball_positions_detected = [[pos_out_of_court] for i in range(9)]
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

            # Mask out the court, so detection is not performed on unwanted objects
            frame = cv.bitwise_and(frame, frame ,mask = court_mask)
            frame_copy = frame.copy()

            # Track balls
            masks, new_ball_positions = detect_balls(frame, detection_scaling, r_min, r_max)

            # Extract current position (last column of 2d-array)
            current_ball_positions = [row[-1] for row in ball_positions_detected]

            # Select order of same colored balls
            new_ball_positions = estimate_ball_positions(current_ball_positions,new_ball_positions)

            # Add trajectory information to outputs
            for i in range(0, len(new_ball_positions)):
                if new_ball_positions[i] != pos_out_of_court:
                    ball_positions[i].append(new_ball_positions[i])
                    ball_positions_detected[i].append(new_ball_positions[i])
                    tracked_frames[i].append(frame)
                    ball_times[i].append(t)
                    frame_index[i] += 1
                else:
                    ball_positions[i].append(ball_positions[i][-1])
                if len(ball_positions_detected[i])>1:
                    # Draw a trail behind the ball

                    cv.polylines(frame_copy,[np.array(ball_positions_detected[i][max(1,len(ball_positions_detected[i])-20):-1])],False,ball_colors_bgr[i],4)

            # Rectify court
            rectified_frame = cv.warpPerspective(frame_copy, H, (frame_width, round(frame_width*court_ratio)))        
            rectified_masks = cv.warpPerspective(masks, H, (frame_width, round(frame_width*court_ratio)))

            # Display results
            #overview = np.concatenate((rectified_frame, rectified_masks), axis=1)
            overview = np.concatenate((rectified_frame, rectified_masks), axis=1)
            overview = imutils.resize(overview, width=win_width)
            frame_small = imutils.resize(frame_copy, width=win_width)
            cv.imshow("Performing tracking (press enter to exit...)", frame_small)

            # Quit if user presses enter
            if cv.waitKey(delay_time) == escape_key:
                break
        else:
            break
        t+=dt
    cv.destroyAllWindows()
    return ball_positions_detected, ball_times, tracked_frames, ball_positions

# select the corners with double click
def select_corner(event, x, y,flags, param):
    if event == cv.EVENT_LBUTTONDOWN :
        # Show the user that the click has been registered
        cv.circle(frame_copy, (x, y), 10, (0, 0, 255), 2)
        # Transform click position to corresponding place in full-sized image and add it
        click = transform_point([x, y, 1], H_resize_inv)
        x = round(click[0])
        y = round(click[1])
        corners_selected.append([x, y])

def get_court_mask(frame,win_width):
    global corners_selected
    global frame_copy
    global H_resize_inv
    corners_selected = []
    frame_width = frame.shape[1]
    frame_height = frame.shape[0]
    frame_copy = imutils.resize(frame, width=win_width)
    resize_ratio = win_width / frame_width
    H_resize_inv = np.array([[1 / resize_ratio, 0, 0], [0, 1 / resize_ratio, 0], [0, 0, 1]])
    title = "Select a suitable points for a cort mask, then press enter..."
    cv.imshow(title, frame_copy)
    cv.setMouseCallback(title, select_corner)

    while True:
        cv.imshow(title, frame_copy)
        if cv.waitKey(20) == 13:
            cv.destroyAllWindows()
            break

    corners = np.array(corners_selected, np.int32)
    mask = np.zeros((frame_height,frame_width), np.uint8)
    cv.fillPoly(mask,[corners],(255,255,255))
    return mask
