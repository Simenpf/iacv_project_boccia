import numpy as np

#Inputs:
pos = [[1,1],[2,2],[3,3],[4,4],[5,5]]
C = np.ones((3,4))
fps = 30

# Constants
g = -9.81
dt = 1/fps
T = (len(pos)-1)*dt

t = np.linspace(0,T, len(pos), dtype = np.float32)

def get_some_D(C, pos, t):
    n = 2*len(pos)
    d_0 = [0]*n
    d_1 = [0]*n
    d_2 = [0]*n
    d_3 = [0]*n
    d_4 = [0]*n
    d_5 = [0]*n
    D = [None]*6*n
    for j in range(0, len(pos)):
        for i in range(0, 2):
            d_0[2*j+i] = C[i, 0] - pos[j][i]*C[2, 0]
            d_1[2*j+i] = C[i, 0]*t[j] - pos[j][i]*C[2,0]*t[j]
            d_2[2*j+i] = C[i, 1] - pos[j][i]*C[2,1]
            d_3[2*j+i] = C[i, 1]*t[j] - pos[j][i]*C[2,1]
            d_4[2*j+i] = C[i, 2] - pos[j][i]*C[2,2]
            d_5[2*j+i] = C[i, 2]*t[j] - pos[j][i]*C[2,2]
    D = [d_0, d_1, d_2, d_3, d_4, d_5]
    D = np.reshape(np.array(D),(n,6))
    return D

def give_some_F(C, pos, t):
    n = 2*len(pos)
    F = [0]*n
    for j in range(0, len(pos)):
        for i in range(0, 2):
            F[2*j+i] = pos[j][i]*(0.5*C[2, 2]*g*t[j]**2+1)-(0.5*C[0,2]*g*t[j]**2+C[i,3])
    F = np.transpose(np.array(F))
    return F

def init_states(D,F):
    E = np.array(np.dot(np.linalg.pinv(D),F))
    E = np.transpose(E)
    return E

# returns real world coordinates as list of X coordinates, list of Y coordinates, list of Z coordinates
def real_3d_coordinate(E, pos, t):
    X = [0]*len(pos)
    Y = [0]*len(pos)
    Z = [0]*len(pos)
    
    for i in range(0, len(pos)):
        X[i] = E[0] + E[1]*t[i]
        Y[i] = E[2] + E[3]*t[i]
        Z[i] = E[4] + E[5]*t[i] + .5*g*t[i]**2

    coordinate = [X,Y,Z]
    return coordinate

D = get_some_D(C,pos,t)
F = give_some_F(C,pos,t)
E = init_states(D,F)
print(E)
print(real_3d_coordinate(E,pos,t))
