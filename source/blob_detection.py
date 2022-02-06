
import cv2 as cv
import numpy as np
import imutils
from projective_funcs import transform_point


def dilate_mask(mask):
    kernel = np.ones((5,5), np.uint8)
    return cv.dilate(mask, kernel, iterations=4)

def mask_balls(frame):
    hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # Hue of the balls (Found experimentally)
    blue_hue   = 95
    orange_hue = 17
    green_hue  = 50
    red_hue    = 176 
    yellow_hue = 30

    range = 4

    # Blue ball HSV range
    low_blue  = np.array([blue_hue-range,100,100])
    high_blue = np.array([blue_hue+range,255,255])

    # Orange ball HSV range
    low_orange  = np.array([orange_hue-range,100,100])
    high_orange = np.array([orange_hue+range,255,255])

    # Green ball HSV range
    low_green  = np.array([green_hue-range,100,100])
    high_green = np.array([green_hue+range,255,255])

    # Red ball HSV range
    low_red  = np.array([red_hue-range,60,100])
    high_red = np.array([red_hue+range,255,255])

    # Yellow ball HSV range
    low_yellow  = np.array([yellow_hue-range,100,100])
    high_yellow = np.array([yellow_hue+range,255,255])

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
    H_detection_scaling = np.array([[1/detection_scaling, 0, 0], [0, 1/detection_scaling, 0], [0, 0, 1]])
    frame = cv.blur(frame, (15,15))
    frame = imutils.resize(frame, width=round(frame_width*detection_scaling))

    # Compute masks for frame
    frame_masks = mask_balls(frame)
    all_masks = cv.bitwise_or(frame_masks[0],frame_masks[1])
    all_masks = cv.bitwise_or(all_masks,frame_masks[2])
    all_masks = cv.bitwise_or(all_masks,frame_masks[3])
    all_masks = cv.bitwise_or(all_masks,frame_masks[4])


    balls_pos = [(-1,-1),(-1,-1),(-1,-1),(-1,-1),(-1,-1),(-1,-1),(-1,-1),(-1,-1),(-1,-1)]
    for i in range(0,5):
        contours = cv.findContours(frame_masks[i], cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[-2]
        contours = sorted(contours, key=cv.contourArea, reverse=True)
        if i != 4:
            for j in range(0, min(2,len(contours))):
                    (x,y), r = cv.minEnclosingCircle(contours[j])
                    if r > 5:
                        pos = np.array([x,y,1])
                        pos = transform_point(pos,H_detection_scaling)
                        balls_pos[i*2+j]= (round(pos[0]),round(pos[1]))
        else:
            if len(contours)>0:
                (x,y), r = cv.minEnclosingCircle(contours[0])
                if r > 10:
                    pos = np.array([x,y,1])
                    pos = transform_point(pos,H_detection_scaling)
                    balls_pos[8] = (round(pos[0]),round(pos[1]))


    # Prepare masks for visualization
    all_masks = cv.cvtColor(all_masks, cv.COLOR_GRAY2BGR)
    all_masks = imutils.resize(all_masks, width=frame_width)

    return all_masks, balls_pos