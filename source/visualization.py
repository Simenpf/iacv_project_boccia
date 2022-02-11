import numpy as np
from game_scores import *
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from configuration import court_length, court_width, number_of_balls, ball_colors

# 3D plotting
def set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def plot_trajectory(traj_3d,corners_actual):
    # Set up the 3d figure
    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_ylabel('y')
    ax.set_xlabel('x')
    ax.set_zlabel('z')
    ax.set_title('3D Trajectory')

    # Add the court
    extra_court_poles = 2
    big_corners_actual = np.array([[0, -court_length*extra_court_poles, 0],[0, 0, 0],[court_width, 0, 0],[court_width, -court_length*extra_court_poles, 0]],dtype=np.float32)

    # Goal region
    X, Y = np.meshgrid(corners_actual[:,0], corners_actual[:,1])
    Z = np.zeros((4,4))
    ax.plot_surface(X, Y, Z,alpha=0.3,color='g',edgecolors='g')

    # Whole court
    X, Y = np.meshgrid(big_corners_actual[:,0], big_corners_actual[:,1])
    Z = np.zeros((4,4))
    ax.plot_surface(X, Y, Z,alpha=0.3,color='r',edgecolors='r')

    # Add the trajectory
    legends = ["Blue ball 1","Blue ball 2","Orange ball 1","Orange ball 2","Green ball 1","Green ball 2","Red ball 1","Red ball 2","Yellow ball"]
    
    for i, traj in enumerate(traj_3d):
        if(len(traj)>0):
            ax.plot(traj[:,0],traj[:,1],traj[:,2],linestyle='--', marker='o', color=ball_colors[i], label=legends[i],zorder=100)
    
    # Display the figure
    set_axes_equal(ax)
    plt.legend()
    plt.show()

# 2D plotting
def display_current_score_board(current_ball_positions):
    # Create plot
    plt.ylim(0,court_width)
    plt.xlim(0,court_length)
    plt.title('Score Board')
    fig, current_score_board = plt.subplots(figsize=(5, 2.7))

    # Plot balls
    for ball in range(0, number_of_balls):
        if not pos_out_of_bounds(current_ball_positions[ball]):
            current_score_board.plot(current_ball_positions[ball], '%s' %ball_colors[ball], label='Ball %i' %ball)
    
    current_score_board.legend()
    ball_distances = get_distances_from_center_ball(current_ball_positions)
    closest_ball, closest_ball_distance = find_closest_ball(ball_distances)
    
    # Calculate and display score   
    score = calculate_score(ball_positions)
    at = AnchoredText("Team 1: %i \n Team 2: %i)" %score[0] %score[1], prop=dict(size=15), frameon=True, loc='upper left')
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    current_score_board.plot([current_ball_positions[-1], current_ball_positions[closest_ball]], '-k')
    return

def animate_score_board(ball_positions):
    for i in range(0, len(ball_positions[0])): # maybe use something else on upper range
        display_current_score_board(ball_positions[i]) 
        cv.wait(2)
    return

# OUTDATED
## 2d scoring plot
#def create_score_board_frame(image_points, H):
#    score_board_frame = []
#
#    #Define plot params (maybe move to animate plot function)
#    plt.ylim(0,court_width)
#    plt.xlim(0,court_length)
#    plt.title('Score Board')
#
#    ball_pos_2d = create_rectified_position_vector(image_points, H)
#
#    for ball in range(0, number_of_balls):
#        x = ball_pos_2d[ball][0]
#        y = ball_pos_2d[ball][1]
#        score_board_frame = plt.plot(x,y)
#
#    # Calculate and display score   
#    score = calculate_score(image_points, H)
#    at = AnchoredText("Team 1: %i \n Team 2: %i)" %score[0] %score[1], prop=dict(size=15), frameon=True, loc='upper left')
#    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
#    
#    return score_board_frame
#