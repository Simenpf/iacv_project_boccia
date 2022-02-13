import cv2 as cv
import numpy as np
from cmath import inf
from math import sqrt
from projective_funcs import transform_point, create_rectified_position_vector
from configuration import court_reference_frame_path, H_auto, team_1_ball_slice, team_2_ball_slice, team_1_ball_indexes, team_2_ball_indexes, court_length, court_width, ball_score, number_of_balls

## MODULE INPUTS: 2D ball position vector
## MODULE OUTPUTS: 1x2 Score vector [Score team 1, Score team 2]

# Check if a (x,y) position is outside defined court
def pos_out_of_bounds(pos):
    out_of_bounds = True
    if pos[0]>=0 and pos[0]<=court_length and pos[1]>=0 and pos[1]<=court_width:
        out_of_bounds = False
    return out_of_bounds

# Get current positions from ball position vector
def get_current_ball_positions(ball_positions, time_index):
    current_ball_positions = []
    for ball in range(0, number_of_balls):
        current_ball_positions.append(ball_positions[ball][time_index])
    return current_ball_positions

# Get position of center ball and vector of distances of all team balls from center ball at one time instant, inf if ball out of bounds
def get_distances_from_center_ball(current_ball_positions):
    distances_from_center_ball = [inf]*(number_of_balls-1)
    center_ball_position = current_ball_positions[-1]
    for ball in range(number_of_balls-1):
        if not pos_out_of_bounds(current_ball_positions[ball]):
            distances_from_center_ball[ball] = sqrt((current_ball_positions[ball][0]-center_ball_position[0])**2+(current_ball_positions[ball][1]-center_ball_position[1])**2)
    return distances_from_center_ball, center_ball_position

# Find the closest ball at this moment and its distance from the center ball
def find_closest_ball(ball_distances):
    closest_ball_distance = min(ball_distances)
    closest_ball = ball_distances.index(closest_ball_distance)
    return closest_ball, closest_ball_distance

# Calculate score from the current ball positions
def calculate_score(current_ball_positions):
    score = [0, 0]
    ball_distances, center_ball_position = get_distances_from_center_ball(current_ball_positions)
    if pos_out_of_bounds(center_ball_position):
        return [-1,-1]
    else:
        closest_ball, _ = find_closest_ball(ball_distances)

        score_count = 0
        # Team 1 has the closest ball
        if closest_ball in team_1_ball_indexes:
            ball_distances_team_1 = ball_distances[team_1_ball_slice]
            ball_distances_team_1.sort()
            _, closest_ball_opposite_team_distance = find_closest_ball(ball_distances[team_2_ball_slice])
            for ball in range(0, int((number_of_balls-1)/2)):
                if closest_ball_opposite_team_distance>ball_distances_team_1[ball]:
                    score_count+=1
            score[0] = score_count*ball_score
        # Team 2 has the closest ball
        elif closest_ball in team_2_ball_indexes:
            for ball in team_2_ball_indexes:
                ball_distances_team_2 = ball_distances[team_2_ball_slice]
                ball_distances_team_2.sort()
                _, closest_ball_opposite_team_distance = find_closest_ball(ball_distances[team_1_ball_slice])
            for ball in range(0, int((number_of_balls-1)/2)):
                if closest_ball_opposite_team_distance>ball_distances_team_2[ball]:
                    score_count+=1
            score[1] = score_count*ball_score
        else:
            print('Only center ball on court')
    return score


# TESTING

#ball_positions = create_rectified_position_vector(image_points, H_auto)

#ball_positions_1 = [[[0,0],[200,200]], [[200,390],[100,100]], [[20,70],[50,100]], [[0,0],[200,200]], [[0,3],[5,15]], [[3,0],[15,150]], [[-1,-1],[-1,-1]],[[-1,-1],[5,15]] ,[[100,190],[100,190]]]
#ball_positions_2 = [[[0,190],[200,200]], [[200,390],[100,100]], [[100,80],[50,100]], [[0,0],[200,200]], [[0,3],[5,15]], [[3,0],[15,150]], [[100,190],[-1,-1]],[[-1,-1],[5,15]] ,[[100,190],[100,190]]]
#current_ball_positions = get_current_ball_positions(ball_positions_2,0)
#print(calculate_score(current_ball_positions))

# Testing point transformation
#H_inv = np.linalg.inv(H_auto)
#point_zero = transform_point([0,202,1],H_inv)
#point_zero = np.array((round(point_zero[0]), round(point_zero[1])))
#test_img = court_reference_frame.copy()
#cv.circle(test_img,point_zero,10,(0,0,255),-1)
#test_img = imutils.resize(test_img, width=800)
#cv.imshow('',test_img)
#cv.waitKey(0)
