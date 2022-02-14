import math
import numpy as np
from configuration import number_of_balls

def transform_point(x,H):
    x=H.dot(x)
    return x/x[-1]

# Get rectified 2d positions from 2d image positions
def create_rectified_position_vector(image_points, H):
    for ball in range(0,len(image_points)):
        for t in range(0,len(image_points[0])):
            image_points[ball][t]=transform_point(image_points[ball][t]+[1],H)
    return image_points

def get_radius_range(ball_radius,P,court_back,court_front):
    p1_back = np.array([0, court_back, 0, 1])
    p2_back = np.array([ball_radius, court_back, 0, 1])
    p1_front = np.array([0, court_front, 0, 1])
    p2_front = np.array([ball_radius, court_front, 0, 1])
    r_min = get_image_distance(p1_back,p2_back,P)
    r_max = get_image_distance(p1_front,p2_front,P)
    return r_min, r_max

def get_image_distance(p1,p2,P):
    p1_img = transform_point(p1,P)
    p2_img = transform_point(p2,P)
    return math.dist([p1_img[0],p1_img[1]],[p2_img[0],p2_img[1]])