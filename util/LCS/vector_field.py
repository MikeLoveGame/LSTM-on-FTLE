import numpy as np
from scipy.interpolate import CubicSpline
from scipy import interpolate
import os
import torch.nn.functional as f
import torch
import torchvision
import math
#from sympy import *

import matplotlib.pyplot as plt

'''
currently, such field only supports linear relationship for particle tracing in x(0) to x(1) space since it 
serves to single frame FTLE calculation to SAM
'''
class Particle():
    #position stored in (u,v)
    # path will be a set of tuples
    def __init__(self, position : tuple):
        self.position = position
        self.path = []
        self.path.append(position)

    def update(self, vector : tuple):
        new_pos = (self.position[0]+vector[0], self.position[1]+vector[1])
        self.position = new_pos
        self.path.append(self.position)
    def new_position(self, pos : tuple):
        self.position = pos
        self.path.append(self.position)

    def __updatePath__(self):
        x = []
        y = []
        for position in self.path:
            x.append(position[0])
            y.append(position[1])


    def get_position(self, time : int):
        self.__updatePath__()
        return self.path[int(time)]
    def get_current_position(self):
        return self.position


class Field():
    """
    the class is to preprocess each point with interval of 1 in vector_map
    first, vector_map will be interpolated in 3D , to approximate time interval to 1ms
    then it will calculate path of each partical seperate with 1 space
    then with each partical, a trejectory poth will be simulated. Such path will be interpolated into a function using
    CubicSpline

    tracing of the particles must consider time as factor not uniform under the same time.
    """
    def __init__(self, mapU : np.ndarray, mapV: np.ndarray, time : int, num_partical : int, interpolate_factor : int):

        shape=int(math.sqrt(num_partical))

        self.particle_shape = (shape, shape)
        self.interpolate_factor = interpolate_factor
        self.mapU= mapU
        self.mapV= mapV
        self.time=time
        self.particles = []


        #self.__interpolate2D__(factor=interpolate_factor)
        self.create_particles()
        self.trace_particles()
        self.ftle = []
        for i in range(0, time-2):
            print ( f"calculating ftle at {i}" )

            self.ftle.append(self.calc_ftle(delta=1, shape=(shape, shape), start=i, end=i + 1))

            print (f"length stored is {len(self.ftle)}")

        ftle = np.asarray(a=self.ftle, dtype=np.float32)
        self.ftle = ftle

    def __interpolate2D__(self, factor):
        mapU = self.mapU
        mapV = self.mapV

        mapU_cp = []
        mapV_cp = []

        count =0
        for time in mapU:
            print(f"executing time at {count}")

            x=np.linspace(start=0, stop=mapU.shape[1], num=mapU.shape[1])
            y=np.linspace(start=0, stop=mapU.shape[1], num=mapU.shape[1])
            z=time
            new_x= np.linspace(start=0, stop=mapU.shape[1], num=mapU.shape[1] * factor)
            new_y= np.linspace(start=0, stop=mapU.shape[2], num=mapU.shape[2] * factor)
            func = interpolate.interp2d(x, y, z, kind='cubic')
            new_z = func(new_x, new_y)

            mapU_cp.append(new_z)
            count+=1

        count=0
        for time in mapV:
            print(f"executing time at {count}")

            x=np.linspace(start=0, stop=mapV.shape[1], num=mapV.shape[1])
            y=np.linspace(start=0, stop=mapV.shape[1], num=mapV.shape[1])
            z=time
            new_x= np.linspace(start=0, stop=mapV.shape[1], num=mapV.shape[1] * factor)
            new_y= np.linspace(start=0, stop=mapV.shape[2], num=mapV.shape[2] * factor)
            func = interpolate.interp2d(x, y, z, kind='cubic')
            new_z = func(new_x, new_y)

            mapV_cp.append(new_z)
            count+=1

        self.mapU = np.asarray(a=mapU_cp, dtype=np.float32)
        self.mapV = np.asarray(a=mapV_cp, dtype=np.float32)

    def create_particles(self):
        x_axis = self.mapU.shape[1]
        y_axis = self.mapU.shape[2]

        distance_x = x_axis/self.particle_shape[0]
        distance_y = y_axis/self.particle_shape[1]

        i = 0
        while(i < x_axis):
            set = []
            j=0
            while(j < y_axis):
                set.append(Particle((i,j)))
                print(f"particle {i} {j} created")
                j+=distance_y
            self.particles.append(set)
            i+=distance_x
    def asTensor(self):
        u = torch.from_numpy(self.mapU)
        v = torch.from_numpy(self.mapV)

        #return a tensor of the field
        return u, v


    def trace_particles(self):
        # return a list of Cubic Spline as traced lines of the particle
        '''this algorithm is to trace particles in space given time and space,
        first is to interpolate within time to increase accuracy, then use cubic spline for each particle
        to return as a set of functions'''

        shape = self.particle_shape

        time = self.mapV.shape[0]

        snap = 0
        step = 1
        while (snap < time-1 ):
            for i in range(shape[0]):
                for j in range(shape[1]):
                    #for the purpose of testing, u, v are turned into int, but instead it should round to the nearest 0.01 or 0.001
                    #by multiply 100 or 1000
                    particle = self.particles[i][j]
                    pos = particle.position
                    u = int(round(particle.position[0]))
                    v = int(round(particle.position[1]))
                    new_u = self.trace_U(t0=snap, u0=u, v0=v, dt=step)
                    new_v = self.trace_V(t0=snap, u0=u, v0=v, dt=step)
                    particle.new_position((new_u, new_v))

            print(f"tracing snap {snap} completed")
            snap += step

        for particleset in self.particles:
            for particle in particleset:
                particle.__updatePath__()

    def trace_U(self, t0, u0, v0, dt=0.05):

        f1 = self.mapU[int(t0)][int(u0)][int(v0)]
        f2 = self.mapU[int(t0 + dt * f1 / 2.0)] [int(u0 + dt * f1 / 2.0)][int(v0)]
        f3 = self.mapU[int(t0 + dt / 2.0)] [int(u0 + dt * f2 / 2.0)] [int(v0)]
        f4 = self.mapU[int(t0 + dt)][int(u0 + dt * f3)] [int(v0)]

        u1 = u0 + dt * (f1 + 2.0 * f2 + 2.0 * f3 + f4) / 6.0

        return u1
    def trace_V(self, t0, u0, v0, dt=1):
        #modify this to fit with interpolation
        f1 = self.mapV[int(t0)][int(u0)][int(v0)]
        f2 = self.mapV[int(t0 + dt * f1 / 2.0)] [int(u0)][int(v0 + dt * f1 / 2.0)]
        f3 = self.mapV[int(t0 + dt / 2.0)] [int(u0)] [int(v0 + dt * f2 / 2.0)]
        f4 = self.mapV[int(t0 + dt)][int(u0)] [int(v0 + dt * f3)]

        v1 = v0 + dt * (f1 + 2.0 * f2 + 2.0 * f3 + f4) / 6.0

        return v1
    def calc_ftle(self, delta: int, shape: tuple, start: int, end: int):  # its all about tracing the end points and evaluate it by linearize the whole thing which is stupid but fast
        ftle = np.zeros(shape=shape)
        end = end
        start = start

        time = end - start

        for i in range(shape[0]-1):
            for j in range(shape[1]-1):
                p1 = self.particles[i-1][j]
                p2 = self.particles[i+1][j]

                p3 = self.particles[i][j-1]
                p4 = self.particles[i][j+1]

                jacob = np.zeros((2, 2))

                o1 = p1.get_position(start)
                t1 = p1.get_position(end)
                o2 = p2.get_position(start)
                t2 = p2.get_position(end)

                o3 = p3.get_position(start)
                t3 = p3.get_position(end)
                o4 = p4.get_position(start)
                t4 = p4.get_position(end)




                jacob[0][0] = (t2[0] - t1[0]) / (o2[0] - o1[0] + 3.647e-5)
                jacob[1][0] = (t2[1] - t1[1]) / (o2[0] - o1[0] + 3.647e-5)
                jacob[0][1] = (t4[0] - t3[0]) / (o4[1] - o3[1] + 3.647e-5)
                jacob[1][1] = (t4[1] - t3[1]) / (o4[1] - o3[1] + 3.647e-5)

                cauchy = np.matmul(jacob.transpose(), jacob)

                val, vec = np.linalg.eig(cauchy)

                lambdas = val
                lambdas = max(lambdas)
                if (lambdas == "nan" or lambdas == 0.0 ):
                    ftle[i][j] = float(1e-3)
                else:
                    ftle[i][j] = np.log(np.sqrt(lambdas))

        showFTLE(ftle)

        return ftle


def readData2D( folderPath : str)-> np.ndarray:
    folder = os.listdir(folderPath)
    data = []
    for file in folder:
        path = os.path.join(folderPath, file)
        temp = np.load(file=path, allow_pickle=True)
        print(f"file {file} loaded")
        data.append(temp)

    data = np.asarray(a=data, dtype=np.float32)
    return data


def showFTLE(ftle : np.ndarray):
    plt.imshow(ftle, cmap='gray')
    plt.show()

def test():
    dirPath = r"C:\AI\Data\Field-data\U-vector-numpy-1D"
    dataU = readData2D(dirPath)
    dirPath = r"C:\AI\Data\Field-data\V-vector-numpy-1D"
    dataV = readData2D(dirPath)
    field = Field(mapU=dataU, mapV=dataV, time=60, num_partical=2500, interpolate_factor=10)







