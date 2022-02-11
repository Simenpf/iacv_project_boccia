import numpy as np

def transform_point(x,H):
    x=H.dot(x)
    return x/x[-1]