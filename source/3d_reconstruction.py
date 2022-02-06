from re import T
import numpy as np

g = -9.81
C = np.ones((3,4))
print(C)

pos = [[1,1],[2,2],[3,3],[4,4],[5,5]]
t = [1,2,3,4,5]

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
    #D_T = D.transpose()
    #D_TD_I = np.linalg.inv(D_T.dot(D))
    #E = np.dot(np.dot(D_TD_I,D_T),F)
    #E = np.dot(np.dot(np.linalg.inv(np.dot(D.transpose(),D)),D.transpose()),F)
    E = np.dot(np.linalg.pinv(D),F)
    return E


D = get_some_D(C,pos,t)
print(D.shape)
print(D)
#print(D.transpose())
#F = give_some_F(C,pos,t)

#print(init_states(D,F))
#print(init_states(D,F).shape)

def real_3d_coordinate(E):
    
    X = E[0] + E[1]*t
    Y = E[2] + E[3]*t
    Z = E[4] + E[5]*t + .5*g*t^2

    coordinate = [X,Y,Z]
    return coordinate


