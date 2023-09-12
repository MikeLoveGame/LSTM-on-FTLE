import os
import numpy as np
from scipy import linalg
import math
import random
from matplotlib import pyplot as plt


def cubic_field( x_range : int, y_range : int, factor : int):
    vars = []
    for i in range(6):
        vars.append(random.randrange(-factor, factor))
    vars = np.asarray(vars)
    field = np.zeros((x_range, y_range))
    for i in range(field.shape[0]):
        for j in range(field.shape[1]):
            a = vars[0]
            b = vars[1]
            c = vars[2]
            d = vars[3]
            e = vars[4]
            f = vars[5]
            field[i][j] = a * i ** 3 + b * j ** 3 + c * i ** 2 + d * j ** 2 + e * i + f * j

    rand = random.randint(0, 1)

    if rand == 0:
        return field*-1

    return field

def complex_field1(x_range : int, y_range : int, factor : int):

    vars = []
    for i in range(6):
        vars.append(random.randrange(-factor, factor))
    vars = np.asarray(vars)
    field = np.zeros((x_range, y_range))
    for i in range(field.shape[0]):
        for j in range(field.shape[1]):
            a = vars[0]
            b = vars[1]
            c = vars[2]
            d = vars[3]
            e = vars[4]
            f = vars[5]
            field[i][j] = i**abs(a) + b*math.sin(a*math.pi*i) - b*math.cos(c*math.pi*i) + d*math.sin(e*math.pi*j) - d*math.cos(f*math.pi*j)
    max = field.max()
    min = field.min()
    #shift between 1 to -1 not done yet
    field = (field - min) / (max - min)

    return field

def circle_field(x_range : int, y_range : int, inner_r : int, outter_r : int, center : (int, int)):
    field = np.zeros((x_range, y_range))
    a, b = random.randrange(-1, 1), random.randrange(-1, 1)
    for i in range(field.shape[0]):
        for j in range(field.shape[1]):
            if (i - center[0])**2 + (j - center[1])**2 <= inner_r**2:
                field[i][j] = a
            elif (i - center[0])**2 + (j - center[1])**2 <= outter_r**2:
                field[i][j] = b

    return field


def generate_vector():
    # Generate a random vector of size 1000
    vector = np.random.rand(500, 500)

    m = vector.max()
    n = vector.min()

    range = 2 * max(m, n)
    vector = vector/ range

    return vector


def complex_field2():
    pass

'''
for i in range (100):
    vector = generate_vector()
    file_path = os.path.join(folderpath, f"vector{i}.npy")
    np.save(file_path, vector)
    print(f"vector {i} saved")
'''
def converge(submaps : [], submap_shape : (int, int), map_shape : (int, int)):
    map = np.zeros(shape=map_shape)
    for i in range(map_shape[0]):
        for j in range(map_shape[1]):
            total = (i*map_shape[0]+j)
            num =  total // (submap_shape[0]*submap_shape[1])
            x = (total - num*(submap_shape[0]*submap_shape[1])) // submap_shape[0]
            y = (total - num*(submap_shape[0]*submap_shape[1])) % submap_shape[0]
            map[i][j] = submaps[num][x][y]
    return map

def merge(submaps : [], submap_shape : (int, int), map_shape : (int, int)):
    map = np.zeros(shape=map_shape)
    interval_x = int(map_shape[0]/submap_shape[0])
    interval_y = int(map_shape[1]/submap_shape[1])
    location = [0, 0]
    for num in range(len(submaps)):
           for i in range(submap_shape[0]):
                for j in range(submap_shape[1]):
                    location[0] = (num//interval_x)*(submap_shape[0])
                    location[1] = (num%interval_y)*(submap_shape[1])
                    x = location[0] + i%submap_shape[0]
                    y = location[1] + j%submap_shape[1]
                    map[x][y] = submaps[num][i][j]
    return map
def smooth(map):
    newmap = np.zeros(shape=map.shape)
    for i in range(map.shape[0]-1):
        for j in range(map.shape[1]-1):
            newmap[i][j] = (map[i-1][j-1] + map[i-1][j] + map[i-1][j+1] + map[i][j-1] + map[i][j] + map[i][j+1] + map[i+1][j-1] + map[i+1][j] + map[i+1][j+1])/9
    return newmap

save_path = r"C:\Github-repository\LSTM-on-FTLE\LSTM-on-FTLE\data\Training-Set-2\V"
for i in range(100):
    shape = (500, 500)
    submap_shape = (25, 25)
    interval_x = int(shape[0]/submap_shape[0])
    interval_y = int(shape[1]/submap_shape[1])
    submaps = []
    for x in range(interval_x):
        for y in range(interval_y):
            submap = cubic_field(submap_shape[0], submap_shape[1], factor=1)
            submaps.append(submap)

    map = merge(submaps, submap_shape, shape)
    for j in range(10):
        map = smooth(map)
    np.save(os.path.join(save_path, f"vector{i}.npy"), map)
    print(f"vector {i} saved")