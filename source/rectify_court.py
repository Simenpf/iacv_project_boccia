import imutils
import cv2 as cv
import numpy as np
from configuration import escape_key, delay_time, corners_auto, court_length, court_width
from projective_funcs import transform_point


# Callback function for mouseclick. Adds corner to list and marks it in frame
def select_corner(event,x,y,flags,param):
    if event == cv.EVENT_LBUTTONDOWN:
        # Show the user that the click has been registered
        cv.circle(frame_copy,(x,y),10,(0,0,255),2)

        # Transform click position to corresponding place in full-sized image and add it
        click = transform_point([x,y,1],H_resize_inv)
        x = round(click[0])
        y = round(click[1])
        corners_selected.append([x,y])
        

# Function for retrieving metric-rectification matrix, based on input frame and user clicks
def get_court_homography(frame, win_width):
    # Variables that must be reached by callback function
    global frame_copy
    global corners_selected
    global H_resize_inv

    
    frame_width  = frame.shape[1]

    # The corners selected by user and their actual positions
    corners_selected = []
    corners_actual = np.array([[0, 0],[0, court_length],[court_width, court_length],[court_width, 0]])

    
    # Create a copy of the frame for user to click on
    frame_copy = imutils.resize(frame, width=win_width)

    # Create homography from user click to positions in the full-sized frame
    resize_ratio = win_width/frame_width
    H_resize_inv = np.array([[1/resize_ratio, 0, 0], [0, 1/resize_ratio, 0], [0, 0, 1]])

    # Show frame and add callback function for mouseclicks on the frame
    title = "Select corners of the court, then press enter..."
    cv.imshow(title,frame_copy)
    cv.setMouseCallback(title, select_corner)
    
    # Continue showing frame until user presses enter
    while True:
        cv.imshow(title,frame_copy)
        if cv.waitKey(delay_time) == escape_key:
            cv.destroyAllWindows()
            break

    # Compute homography based on user-selected corners, and their actual positions 
    #corners_selected = corners_auto
    corners_actual   = np.float32(corners_actual)
    corners_selected = np.float32([corners_selected])
    H = cv.getPerspectiveTransform(corners_selected,corners_actual)

    
    return H, corners_selected



