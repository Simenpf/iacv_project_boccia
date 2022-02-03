import cv2 as cv
import numpy as np
import imutils
from math import sqrt

def distance_point_circle(x1,y1,x2,y2,r):
    a = pow((x2-x1),2)
    b = pow((y2-y1),2)
    return abs(sqrt(a+b)-r)

def find_max_distance(contour, x, y, r):
    max1 = 0
    max1_i = 0
    max2 = 0
    max2_i = 0
    for i, e in enumerate(contour):
        e = e[0]
        dist = distance_point_circle(e[0],e[1],x,y,r)
        if dist > max1:
            max1 = dist
            max1_i = i
    for i, e in enumerate(contour):
        if abs(i-max1_i)>len(contour)//10: # This is a very assumptious condition
            e = e[0]
            dist = distance_point_circle(e[0],e[1],x,y,r)
            if dist > max2:
                max2 = dist
                max2_i = i
    return contour[max1_i][0], contour[max2_i][0]

def split_contour(contour,p1,p2):
    i_1 = 0
    i_2 = 0
    for i,e in enumerate(contour):
        if e[0][0] == p1[0] and e[0][1] == p1[1]:
            i_1 = i
        if e[0][0] == p2[0] and e[0][1] == p2[1]:
            i_2 = i

    if i_1 > i_2:
        i_1, i_2 = i_2, i_1
    contour1 = contour[i_1:i_2+1]
    contour2 = np.concatenate((contour[i_2:],contour[0:i_1+1]),axis=0)
    return contour1, contour2


# Read image
img = cv.imread('../media/close_balls.jpg', cv.IMREAD_COLOR)
video = cv.VideoCapture(0) # For testing with webcam


# Create color mask
hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
orange_hue = 17
blue_hue = 95
hue_range = 6
low_orange  = np.array([orange_hue-hue_range,100,100])
high_orange = np.array([orange_hue+hue_range,255,255])
low_blue  = np.array([blue_hue-hue_range,100,100])
high_blue = np.array([blue_hue+hue_range,255,255])
orange_mask = cv.inRange(hsv_img,low_orange,high_orange)
blue_mask   = cv.inRange(hsv_img,low_blue,high_blue)
mask = cv.bitwise_or(blue_mask,orange_mask)

# Show image
cv.imshow("",img)
cv.waitKey(0)

# Find largest contour
contours = cv.findContours(orange_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)[-2]
c = max(contours, key = cv.contourArea)


def detect_overlapping_circles(contour):
    (x,y), r = cv.minEnclosingCircle(contour)
    p1,p2 = find_max_distance(c,x,y,r)
    contour1, contour2 = split_contour(c,p1,p2)
    (x1,y1), r1 = cv.minEnclosingCircle(contour1)
    (x2,y2), r2 = cv.minEnclosingCircle(contour2)
    return round(x1),round(y1),round(x2),round(y2),round(r1),round(r2)

x1,y1,x2,y2,r1,r2 = detect_overlapping_circles(c)
cv.circle(img,(x1,y1),r1,(0,255,0),3)
cv.circle(img,(x2,y2),r2,(255,0,0),3)
cv.circle(img,(x1,y1),5,(0,255,0),-1)
cv.circle(img,(x2,y2),5,(255,0,0),-1)
cv.imshow("",img)
cv.waitKey(0)

