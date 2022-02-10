from cmath import inf
from cv2 import sqrt
from configuration import team_1_ball_indexes, team_2_ball_indexes, court_length, court_width

out_of_bounds = [-1,-1]

def pos_out_of_bounds(pos):
    out_of_bounds = True
    if pos[0]>=0 and pos[0]<=court_length and pos[1]>=0 and pos[1]<=court_width:
        out_of_bounds = False
    return out_of_bounds


def find_closest_ball(ball_pos):
    center_ball_pos = ball_pos[-1]
    closest_ball = None
    closest_ball_dist = inf
    for ball in range(len(ball_pos)-1): # go through all balls in court
        if ball_pos[ball] != out_of_bounds:
            dist_from_center_ball = sqrt((ball_pos[ball][0]-center_ball_pos[0])**2+(ball_pos[ball][1]-center_ball_pos[1])**2)
            if dist_from_center_ball < closest_ball_dist:
                closest_ball = ball
                closest_ball_dist = dist_from_center_ball
    return closest_ball, closest_ball_dist

def get_distances_from_center_ball(ball_pos):
    ball_distances = []
    center_ball_pos = ball_pos[-1]
    for ball in range(len(ball_pos)-1):
        ball_distances[ball] = sqrt((ball_pos[ball][0]-center_ball_pos[0])**2+(ball_pos[ball][1]-center_ball_pos[1])**2)
    return ball_distances

def find_closest_ball(ball_distances):
    ball_distances = enumerate(ball_distances)
    ball_distances.sorted()
    return

def calculate_score(ball_pos):
    score = [['Team 1', 0],['Team 2', 0]]

    center_ball_pos = ball_pos[-1]

    while not pos_out_of_bounds(center_ball_pos):
        ball_distances = get_distances_from_center_ball(ball_pos)
        closest_ball = find_closest_ball(ball_distances)
        if closest_ball in team_1_ball_indexes:
            closest_ball_opposite_team = find_closest_ball(ball_distances[team_2_ball_indexes])
            for i in team_1_ball_indexes:
                
                return
        elif closest_ball in team_2_ball_indexes:
            for i in team_2_ball_indexes:
                return
        else:
            print('No team balls on court')

    return score