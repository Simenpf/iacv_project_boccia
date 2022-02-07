import numpy as np

#Inputs:
traj_2d = [[1,1],[2,2],[3,3],[4,4],[5,5]]
P = np.ones((3,4))
fps = 30

# Constants
g = -9.81
dt = 1/fps
T = (len(traj_2d)-1)*dt

t = np.linspace(0,T, len(traj_2d), dtype = np.float32)

def get_some_D(P, traj_2d, t):
    n = 2*len(traj_2d)
    d_0 = [0]*n
    d_1 = [0]*n
    d_2 = [0]*n
    d_3 = [0]*n
    d_4 = [0]*n
    d_5 = [0]*n
    D = [None]*6*n
    for j in range(0, len(traj_2d)):
        for i in range(0, 2):
            d_0[2*j+i] = P[i, 0] - traj_2d[j][i]*P[2, 0]
            d_1[2*j+i] = P[i, 0]*t[j] - traj_2d[j][i]*P[2,0]*t[j]
            d_2[2*j+i] = P[i, 1] - traj_2d[j][i]*P[2,1]
            d_3[2*j+i] = P[i, 1]*t[j] - traj_2d[j][i]*P[2,1]
            d_4[2*j+i] = P[i, 2] - traj_2d[j][i]*P[2,2]
            d_5[2*j+i] = P[i, 2]*t[j] - traj_2d[j][i]*P[2,2]
    D = [d_0, d_1, d_2, d_3, d_4, d_5]
    D = np.reshape(np.array(D),(n,6))
    return D

def give_some_F(P, traj_2d, t):
    n = 2*len(traj_2d)
    F = [0]*n
    for j in range(0, len(traj_2d)):
        for i in range(0, 2):
            F[2*j+i] = traj_2d[j][i]*(0.5*P[2, 2]*g*t[j]**2+1)-(0.5*P[0,2]*g*t[j]**2+P[i,3])
    F = np.transpose(np.array(F))
    return F

def init_states(D,F):
    E = np.array(np.dot(np.linalg.pinv(D),F))
    E = np.transpose(E)
    return E

# returns real world coordinates as list of X coordinates, list of Y coordinates, list of Z coordinates
def real_3d_coordinate(E, traj_2d, t):
    n = len(traj_2d)
    X = [0]*n
    Y = [0]*n
    Z = [0]*n
    #traj_3d = np.empty((len(traj_2d),2))
    traj_3d = [0]*n
    for i in range(0, n):
        X[i] = E[0] + E[1]*t[i]
        Y[i] = E[2] + E[3]*t[i]
        Z[i] = E[4] + E[5]*t[i] + .5*g*t[i]**2
        traj_3d[i] = [X[i], Y[i], Z[i]]
    
    return traj_3d


def generate_3d_trajectory(P, traj_2d, fps):
    dt = 1/fps
    T = (len(traj_2d)-1)*dt
    t = np.linspace(0,T, len(traj_2d), dtype = np.float32)
    
    D = get_some_D(P,traj_2d,t)
    F = give_some_F(P,traj_2d,t)
    E = init_states(D,F)
    return real_3d_coordinate(E,traj_2d,t)


print(generate_3d_trajectory(P,traj_2d,fps))