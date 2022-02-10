from cmath import inf
from cv2 import sqrt
from configuration import team_1_ball_indexes, team_2_ball_indexes, court_length, court_width, ball_score

# Check if a position is outside defined court
def pos_out_of_bounds(pos):
    out_of_bounds = True
    if pos[0]>=0 and pos[0]<=court_length and pos[1]>=0 and pos[1]<=court_width:
        out_of_bounds = False
    return out_of_bounds

# Find the ball closest to the center ball position
def find_closest_ball(ball_pos):
    center_ball_pos = ball_pos[-1]
    closest_ball = None
    closest_ball_dist = inf
    for ball in range(len(ball_pos)-1): # go through all balls in court
        if not pos_out_of_bounds(ball_pos[ball]):
            dist_from_center_ball = sqrt((ball_pos[ball][0]-center_ball_pos[0])**2+(ball_pos[ball][1]-center_ball_pos[1])**2)
            if dist_from_center_ball < closest_ball_dist:
                closest_ball = ball
                closest_ball_dist = dist_from_center_ball
    return closest_ball, closest_ball_dist

# Get vector of distances of all team balls from center ball, inf if ball out of bounds
def get_distances_from_center_ball(ball_pos):
    distances_from_center_ball = [inf]*(len(ball_pos)-1)
    center_ball_pos = ball_pos[-1]
    for ball in range(len(ball_pos)-1):
        if not pos_out_of_bounds(ball_pos[ball]):
            # Calculate distance from center ball to ball
            distances_from_center_ball[ball] = sqrt((ball_pos[ball][0]-center_ball_pos[0])**2+(ball_pos[ball][1]-center_ball_pos[1])**2)
    return distances_from_center_ball

# Find the closest ball and its distance from the center ball
def find_closest_ball(ball_distances):
    closest_ball_distance = min(ball_distances)
    closest_ball = closest_ball_distance.index(closest_ball_distance)
    return closest_ball, closest_ball_distance

# Calculate score from the ball positions
def calculate_score(ball_pos):
    score = [0, 0]

    center_ball_pos = ball_pos[-1]

    while not pos_out_of_bounds(center_ball_pos):
        ball_distances = get_distances_from_center_ball(ball_pos)
        closest_ball, closest_ball_distance = find_closest_ball(ball_distances)
        ball_pos_team_1 = ball_pos[team_1_ball_indexes]
        ball_pos_team_2 = ball_pos[team_2_ball_indexes]
        score_count = 0
        # Team 1 has the closest ball
        if closest_ball in team_1_ball_indexes:
            ball_pos_team_1.sort()
            _, closest_ball_opposite_team_distance = find_closest_ball(ball_distances[team_2_ball_indexes])
            for ball in range(0, len(ball_pos_team_1)):
                if closest_ball_opposite_team_distance<ball_pos_team_1[ball]:
                    score_count+=1
                return
            score[0] += score_count*ball_score
        # Team 2 has the closest ball
        elif closest_ball in team_2_ball_indexes:
            for ball in team_2_ball_indexes:
                ball_pos_team_2.sort()
                _, closest_ball_opposite_team_distance = find_closest_ball(ball_distances[team_1_ball_indexes])
            for ball in range(0, len(ball_pos_team_2)):
                if closest_ball_opposite_team_distance<ball_pos_team_2[ball]:
                    score_count+=1
                return
            score[1] += score_count*ball_score
        else:
            print('Only center ball on court')
    return score