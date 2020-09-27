import numpy as np

def centroid(box):
    return np.array([0.5*(box['x1'] + box['x2']), 0.5*(box['y1'] + box['y2'])])

def same_time(epsilon):
    return lambda i1, i2: abs(i1['t1']-i2['t1']) <= epsilon and abs(i1['t2']-i2['t2']) <= epsilon
