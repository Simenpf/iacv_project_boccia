import numpy as np

# Camera Matrix for camera 2
P = np.array([[-2.64369884e+00, -1.23577043e+00, -4.71130751e-01,  1.43258252e+03],
 [-4.78026920e-03, -1.95656940e-02, -2.74553103e+00,  5.99706109e+02],
 [-8.11727165e-05, -1.41629246e-03, -4.86481859e-04,  1.00000000e+00]])

# Corners of goal region for camera 1
corners_selected = [[1453, 608],[1700, 863],[163, 890],[393, 627]]

# Corners of goal region for camera 1
corners_selected = [[1430, 598],[1657, 837],[210, 875],[420, 620]]

# Number of calibration images
num_calibration_imgs = 30