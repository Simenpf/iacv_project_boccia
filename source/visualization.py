import numpy as np
import matplotlib.pyplot as plt

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

def plot_trajectory(traj_3d,corners_actual,court_width,court_length):
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
    colors  = ['darkturquoise','darkturquoise','orange','orange','limegreen','limegreen','red','red','yellow']
    for i, traj in enumerate(traj_3d):
        if(len(traj)>0):
            ax.plot(traj[:,0],traj[:,1],traj[:,2],linestyle='--', marker='o', color=colors[i], label=legends[i],zorder=100)
    
    # Display the figure
    set_axes_equal(ax)
    plt.legend()
    plt.show()
