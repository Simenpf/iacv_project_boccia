import numpy as np
from game_scores import *
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from matplotlib.animation import FuncAnimation
from configuration import ball_radius, court_length, court_width, number_of_balls, ball_colors


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

def update_2dplot(frame):
    pos = get_current_ball_positions(ball_positions,frame)
    for i in range(0,len(balls)):
        if not pos_out_of_bounds(pos[i]):
            balls[i].center = (pos[i][0], pos[i][1])

    score = calculate_score(pos)
    text.txt.set_text("Team 1: "+str(score[0])+"\n"+"Team 2: "+str(score[1]))
    return balls[0],balls[1],balls[2],balls[3],balls[4],balls[5],balls[6],balls[7],balls[8], text


def init_2dplot():
    for ball in balls:
        ax.add_patch(ball)
    text.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax.add_artist(text)
    return balls[0],balls[1],balls[2],balls[3],balls[4],balls[5],balls[6],balls[7],balls[8], text


def plot_game(ball_positions_in):
    global ax
    global balls
    global text
    global ball_positions

    ball_positions = ball_positions_in
    fig = plt.figure(figsize=(5,10))
    ax = plt.axes(xlim=(0, 390), ylim=(0, 202))
    ax.set_aspect(1)

    ball_radius = [5,5,5,5,5,5,5,5,2.5]

    balls = []
    for i in range(0,number_of_balls):
        balls.append(plt.Circle((-1, -1), ball_radius[i], fc='%s' %ball_colors[i], ec='k')
    )
    text = AnchoredText("", prop=dict(size=15), frameon=True, loc='upper right')

    ani = FuncAnimation(fig, update_2dplot, frames=len(ball_positions_in[0]),
                        init_func=init_2dplot, blit=False,interval=50)
    plt.show()



