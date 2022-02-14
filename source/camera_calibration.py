import imutils
import cv2 as cv
import numpy as np
from configuration import num_calibration_imgs, delay_time, screen_width
from rectify_court import get_court_homography


# Function for retrieving camera intrinsics based on a chessboard calibration video
def getCameraIntrinsics(video,board_size,square_size):
    # Declare points to be used for calibration
    true_pts = []
    img_pts = []

    # Create a row vector of all zeros
    true_board_corners = np.zeros((1, board_size[0] * board_size[1], 3), np.float32) 

    # Edit the row vector to contain all corners of the true chessboard
    true_board_corners[0,:,:2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
    true_board_corners *= square_size

    # Specify criteria for subpixel optimization
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.01)

    # Load all calibration video and find chessboard corners in the frames
    num_frames = int(video.get(cv.CAP_PROP_FRAME_COUNT))
    while True:
        # Select a suitable amount of frames for calibrations
        success, img = video.read()
        for i in range(0,num_frames//num_calibration_imgs):
            success, img = video.read()

        # While there are still frames
        if  success:
            gray_img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

            # Search for corners in grayscale image
            success, image_board_corners = cv.findChessboardCorners(gray_img, board_size, cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_FAST_CHECK + cv.CALIB_CB_NORMALIZE_IMAGE)
            if success:
                # Add repeat of true corners as a match to the new image corners
                true_pts.append(true_board_corners)

                # Optimize corners found in image
                image_board_corners = cv.cornerSubPix(gray_img, image_board_corners, (11,11), (-1,-1), criteria)

                # Add the new image corners 
                img_pts.append(image_board_corners)

                # Draw the found corners for visualization
                img = cv.drawChessboardCorners(img, board_size, image_board_corners, success)

            # Show the image for manual supervision
            window_img = imutils.resize(img, width=screen_width*0.5)
            cv.imshow("Calibration", window_img)
            cv.waitKey(delay_time)
        else:
            break
    cv.destroyAllWindows()

    # Get camera intrinsics
    _, K, dist, _, _ = cv.calibrateCamera(true_pts, img_pts, gray_img.shape[::-1], None, None)
    return K, dist

# Function for retrieving the projection matrix based on camera intrinsics and a set of known points
def getCameraProjectionMatrix(camera_intrinsics, distortion_coefficients, true_pts, img_pts):
    # Retrieve camera extrinsics
    _, rvec, tvec = cv.solvePnP(true_pts, img_pts, camera_intrinsics, distortion_coefficients)
    
    # Linear algebra to obtain projection matrix
    R, _ = cv.Rodrigues(rvec)
    RT = np.concatenate((R,tvec),axis=1)
    P = camera_intrinsics.dot(RT)
    P /= P[2,3]
    return P

# Testing the functions
#######################
# # Define chessboard
# board_size = (9,7)
# square_size = 2

# # Load media
# video = cv.VideoCapture('../media/calibration_video_camera2.mp4')
# frame = cv.imread("../media/bocce_game_camera2_reference_frame.jpg")

# # Define known points in image and world frame
# court_width = 390
# court_length = 202
# court_ratio = court_width/court_length
# _,_,corners_selected = get_court_homography(frame,court_ratio,1000,0)
# corners_selected = corners_selected[0]
# corners_actual = np.array([[0, 0, 0],[0, 202, 0],[court_width, court_length, 0],[court_width, 0, 0]],dtype=np.float32)

# # Retrieve intrinsics and extrinsics
# K, dist = getCameraIntrinsics(video,board_size,square_size)
# P = getCameraProjectionMatrix(K,dist,corners_actual,corners_selected)


# # Draw a known point in the three-dimensional world frame on it projected position
# X = [390+37,-202,110,1]
# x = P.dot(X)
# x /= x[2]

# frame = cv.imread("../media/bocce_game_camera2_reference_frame.jpg")
# frame = cv.circle(frame,(round(x[0]),round(x[1])),10,(0,0,255),-1)
# frame_small = imutils.resize(frame,width=800)
# cv.imshow("",frame_small)
# cv.waitKey(0)





