from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np

def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])







def plot_trajectory(traj_2d,traj_3d,corners_actual,court_width,court_length):
    # Set up the 3d figure
    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111, projection='3d')
    #ax.set_aspect('equal')  
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
    set_axes_equal(ax)
    plt.legend()
    plt.show()
