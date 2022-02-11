import imutils
import cv2 as cv
import numpy as np
from cmath import inf
from math import sqrt
from projective_funcs import transform_point, create_rectified_position_vector
from configuration import court_reference_frame_path, H_auto, team_1_ball_indexes, team_2_ball_indexes, court_length, court_width, ball_score, number_of_balls

## MODULE INPUTS: 2D ball position vector
## MODULE OUTPUTS: 1x2 Score vector [Score team 1, Score team 2]

#corners = [1430, 598],[1657, 837],[210, 875],[420, 620]
court_reference_frame = cv.imread(court_reference_frame_path)
image_points = [[[1430,598], [300,40]],[[1657,837],[30,30]],[[210,875],[30,30]],[[420,620],[80,80],[450,60],[300,30]],[[1,190],[380,300]],
                [[1,1],[30,80]],[[1,200],[30,30]],[[400,400],[100,360],[900,200], [80,240]],[[0,0]]]

# Testing point transformation
H_inv = np.linalg.inv(H_auto)
point_zero = transform_point([0,202,1],H_inv)
point_zero = np.array((round(point_zero[0]), round(point_zero[1])))
test_img = court_reference_frame.copy()
cv.circle(test_img,point_zero,10,(0,0,255),-1)
test_img = imutils.resize(test_img, width=800)
cv.imshow('',test_img)
cv.waitKey(0)

ball_pos = create_rectified_position_vector(image_points, H_auto)

test_img2 = court_reference_frame.copy()
for ball in range(0, number_of_balls):
    for i in range(0, len(ball_pos[ball])):
        cv.circle(test_img2,ball_pos[ball][i],10,(0,0,255),-1)
test_img2 = imutils.resize(test_img2, width=800)
cv.imshow('', test_img2)
cv.waitKey(0)


# Check if a position is outside defined court
def pos_out_of_bounds(pos):
    out_of_bounds = True
    if pos[0]>=0 and pos[0]<=court_length and pos[1]>=0 and pos[1]<=court_width:
        out_of_bounds = False
    return out_of_bounds

# Get vector of distances of all team balls from center ball, inf if ball out of bounds
def get_distances_from_center_ball(ball_positions):
    distances_from_center_ball = [inf]*(len(ball_positions)-1)
    center_ball_positions = ball_positions[-1]
    for ball in range(len(ball_positions)-1):
        if not pos_out_of_bounds(ball_positions[ball]):
            # Calculate distance from center ball to ball
            distances_from_center_ball[ball] = sqrt((ball_positions[ball][0]-center_ball_positions[0])**2+(ball_positions[ball][1]-center_ball_positions[1])**2)
    return distances_from_center_ball

# Find the closest ball and its distance from the center ball
def find_closest_ball(ball_distances):
    closest_ball_distance = min(ball_distances)
    closest_ball = closest_ball_distance.index(closest_ball_distance)
    return closest_ball, closest_ball_distance

# Calculate score from the ball positions
ball_positions = create_rectified_position_vector(image_points, H_auto)

def calculate_score(ball_positions):
    score = [0, 0]
    center_ball_position = ball_positions[-1]

    while not pos_out_of_bounds(center_ball_position):
        ball_distances = get_distances_from_center_ball(ball_positions)
        closest_ball, closest_ball_distance = find_closest_ball(ball_distances)
        
        score_count = 0
        # Team 1 has the closest ball
        if closest_ball in team_1_ball_indexes:
            ball_positions_team_1 = ball_positions[team_1_ball_indexes].sort()
            _, closest_ball_opposite_team_distance = find_closest_ball(ball_distances[team_2_ball_indexes])
            for ball in range(0, len(ball_positions_team_1)):
                if closest_ball_opposite_team_distance<ball_positions_team_1[ball]:
                    score_count+=1
                return
            score[0] = score_count*ball_score

        # Team 2 has the closest ball
        elif closest_ball in team_2_ball_indexes:
            for ball in team_2_ball_indexes:
                ball_positions_team_2 = ball_positions[team_2_ball_indexes].sort()
                _, closest_ball_opposite_team_distance = find_closest_ball(ball_distances[team_1_ball_indexes])
            for ball in range(0, len(ball_positions_team_2)):
                if closest_ball_opposite_team_distance<ball_positions_team_2[ball]:
                    score_count+=1
                return
            score[1] = score_count*ball_score
        else:
            print('Only center ball on court')
    return score

# OUTDATED: Find the ball closest to the center ball position
def find_closest_ball(ball_positions):
    center_ball_position = ball_positions[-1]
    closest_ball = None
    closest_ball_dist = inf
    for ball in range(len(ball_positions)-1): # go through all balls in court
        if not pos_out_of_bounds(ball_positions[ball]):
            dist_from_center_ball = sqrt((ball_positions[ball][0]-center_ball_position[0])**2+(ball_positions[ball][1]-center_ball_position[1])**2)
            if dist_from_center_ball < closest_ball_dist:
                closest_ball = ball
                closest_ball_dist = dist_from_center_ball
    return closest_ball, closest_ball_dist

#game_score = calculate_score(ball_positions, H)

#if game_score[0] > game_score[1]:
#    print('Team 1 in the lead with score: ' + game_score[0])
#elif game_score[1] > game_score[0]:
#    print('Team 2 in the lead with score: ' + game_score[1])