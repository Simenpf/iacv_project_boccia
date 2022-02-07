import matplotlib.pyplot as plt
import numpy as np


def plot_trajectory(traj_2d,traj_3d,corners_actual,court_width,court_length):
    # Set up the 3d figure
    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_ylabel('y')
    ax.set_xlabel('x')
    ax.set_zlabel('z')
    ax.set_title('3D Trajectory')

    # Add the court
    extra_court_poles = 2
    corners_actual = np.vstack([corners_actual, corners_actual[0,:]])
    big_corners_actual = np.array([[0, -court_length*extra_court_poles, 0],[0, 202, 0],[court_width, court_length, 0],[court_width, -court_length*extra_court_poles, 0],[0,-court_length*extra_court_poles, 0]],dtype=np.float32)
    ax.plot(big_corners_actual[:,0], big_corners_actual[:,1], big_corners_actual[:,2],color='g', label='Entire Court')
    ax.plot(corners_actual[:,0], corners_actual[:,1], corners_actual[:,2],color='r', label='Goal Region')

    # Add the trajectory
    ax.plot(traj_2d[:,0],traj_2d[:,1],linestyle='--', marker='o', color='m', label='Ball trajectory 2d')
    ax.plot(traj_3d[:,0],traj_3d[:,1],traj_3d[:,2],linestyle='--', marker='o', color='b', label='Ball trajectory 3d')
    # Display the figure
    plt.xlim(-50, court_width+50)
    plt.ylim(-50-court_length*extra_court_poles, court_length+50)
    plt.legend()
    plt.show()
