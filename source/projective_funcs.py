import numpy as np
from configuration import number_of_balls

def transform_point(x,H):
    x=H.dot(x)
    return x/x[2]

# Get rectified 2d positions from 2d image positions
def create_rectified_position_vector(image_points, H):
    w = 1
    rect_positions = [[] for i in range(0,number_of_balls)] 

    for ball in range(0, number_of_balls):
        for i in range(0, len(image_points[ball])):
            image_points[ball][i].append(w)
            rect_pos = transform_point(image_points[ball][i], H)
            rect_positions[ball].append([rect_pos[0], rect_pos[1]])
            
    return np.array(rect_positions)