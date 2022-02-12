from matplotlib import pyplot as plt, animation as an
import numpy as np
from game_scores import calculate_score
from matplotlib.offsetbox import AnchoredText


def update_plot(ball_pos, board, number_of_balls, ball_radius, ball_colors):
    for ball in range(number_of_balls):
        circle = plt.Circle((ball_pos[ball][0], ball_pos[ball][1]), ball_radius[ball], fc='%s' %ball_colors[ball], ec='k')
        board.add_patch(circle)

        plt.draw()

    score = calculate_score(ball_pos)
    at = AnchoredText("Team 1: %i \n" %score[0] + "Team 2: %i" %score[1], prop=dict(size=15), frameon=True, loc='upper right')
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    board.add_artist(at)
    
    plt.pause(5)
    #plt.figure().clear()
    

def animate():
    fig = plt.figure(figsize=(5,10))
    board = plt.axes(xlim=(0, 202), ylim=(0, 390))
    board.set_aspect(1)

    ball_pos = [[10,10],[20,20],[30,30],[40,200],[0,20],[80,300],[10,50],[90,20],[101,195]]
    ball_pos_2 = [[x+1,y+5] for x,y in ball_pos]
    ball_pos_3 = [[x+1,y+5] for x,y in ball_pos_2]
    number_of_balls = 9
    ball_radius = [5,5,5,5,5,5,5,5,2.5]
    ball_colors  = ['darkturquoise','darkturquoise','orange','orange','limegreen','limegreen','red','red','yellow']

    update_plot(ball_pos, board, number_of_balls, ball_radius, ball_colors)
    #plt.pause(2)
    #board.pop()
    update_plot(ball_pos_2, board, number_of_balls, ball_radius, ball_colors)
    #plt.pause(2)
    #board.pop()
    update_plot(ball_pos_3, board, number_of_balls, ball_radius, ball_colors)
    #plt.pause(2)
    #board.pop()
    plt.show()
    #plt.close()
animate()