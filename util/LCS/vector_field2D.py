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
    this class input U,V vector in 2 dimension and set up for 2D particle tracing
    use rk4 to trace the path of each particle
    computeFTLE to compute the FTLE of the field
    accuracy of the map is determined by interpolate factor such if the interpolate factor is 100, the accuracy of the map is 0.01
    """
    def __init__(self, mapU : np.ndarray, mapV: np.ndarray, trace_time : int, num_partical : int, interpolate_factor : int):

        shape=int(math.sqrt(num_partical))
        dim = mapU.ndim

        if (dim != 2):
            raise Exception("3D vector field is not supported")

        self.particle_shape = (shape, shape)
        self.map_shape = (mapU.shape[0] * interpolate_factor, mapU.shape[1] * interpolate_factor)
        self.interpolate_factor = interpolate_factor # interpolate factor decide \
        # the accuracy of the rk4 calculation 10 for 0.1, 100 for 0.01, etc.
        self.mapU= mapU
        self.mapV= mapV
        self.trace_time=trace_time
        self.particles = []


        self.__interpolate2D__(factor=interpolate_factor)
        self.create_particles()
        self.trace_particles(time=trace_time)


    def __interpolate2D__(self, factor):
        mapU = self.mapU
        mapV = self.mapV

        mapU_cp = []
        mapV_cp = []

        x=np.linspace(start=1, stop=mapU.shape[0], num=mapU.shape[0])
        y=np.linspace(start=1, stop=mapU.shape[0], num=mapU.shape[0])
        z=mapU
        new_x= np.linspace(start=1, stop=mapU.shape[0], num=mapU.shape[0] * factor)
        new_y= np.linspace(start=1, stop=mapU.shape[1], num=mapU.shape[1] * factor)
        func = interpolate.interp2d(x, y, z, kind='cubic')
        new_z = func(new_x, new_y)

        mapU_cp=new_z


        x=np.linspace(start=0, stop=mapV.shape[0], num=mapV.shape[0])
        y=np.linspace(start=0, stop=mapV.shape[0], num=mapV.shape[0])
        z=mapV
        new_x= np.linspace(start=0, stop=mapV.shape[0], num=mapV.shape[0] * factor)
        new_y= np.linspace(start=0, stop=mapV.shape[1], num=mapV.shape[1] * factor)
        func = interpolate.interp2d(x, y, z, kind='cubic')
        new_z = func(new_x, new_y)

        mapV_cp=new_z

        self.mapU = np.asarray(a=mapU_cp, dtype=np.float32)
        self.mapV = np.asarray(a=mapV_cp, dtype=np.float32)

        for i in range(mapU.shape[0]):
            for j in range(mapU.shape[1]):
                val = mapU[i][j]
                if (abs(val) < 1e-4):
                    mapU_cp[i][j] = 0

        for i in range(mapV.shape[0]):
            for j in range(mapV.shape[1]):
                val = mapV[i][j]
                if (abs(val) < 1e-4):
                    mapV_cp[i][j] = 0

        self.mapV = self.mapV * factor
        self.mapU = self.mapU * factor

    def create_particles(self):
        x_axis = self.mapU.shape[0]
        y_axis = self.mapU.shape[1]

        distance_x = x_axis/self.particle_shape[0]
        distance_y = y_axis/self.particle_shape[1]

        i = 0
        while(i < x_axis):
            set = []
            j=0
            while(j < y_axis):
                set.append(Particle((i,j)))
                j+=distance_y
            self.particles.append(set)
            i+=distance_x
    def asTensor(self):
        u = torch.from_numpy(self.mapU)
        v = torch.from_numpy(self.mapV)

        #return a tensor of the field
        return u, v

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
        print(f"computing ftle at time {start} to {end}")
        return ftle

    def trace_particles(self, time : int, step : float = 1):
        '''this algorithm is to trace particles in space given time and space,
        first is to interpolate within time to increase accuracy, then use cubic spline for each particle
        to return as a set of functions'''

        shape = self.particle_shape
        if (step < 1):
            print("step is too small, please increase it to 1 or above")
            return

        current = 0
        while (current < time-1 ):
            for i in range(shape[0]):
                for j in range(shape[1]):
                    #for the purpose of testing, u, v are turned into int, but instead it should round to the nearest 0.01 or 0.001
                    #by multiply 100 or 1000
                    particle = self.particles[i][j]
                    pos = particle.position
                    u = int(round(particle.position[0]))
                    v = int(round(particle.position[1]))
                    new_u = self.trace_U( u0=u, v0=v, dt=step)
                    new_v = self.trace_V(u0=u, v0=v, dt=step)
                    particle.new_position((new_u, new_v))

            current += step

        for particleset in self.particles:
            for particle in particleset:
                particle.__updatePath__()

    def trace_U(self, u0, v0, dt=1):

        x_axis = self.mapU.shape[0]
        y_axis = self.mapU.shape[1]
        if (abs(u0) >= x_axis):
            if (u0 >= 0):
                u0 = x_axis - 1
            else:
                u0 = 0
        if (abs(v0) >= y_axis):
            if (v0 >= 0):
                v0 = y_axis - 1
            else:
                v0 = 0


        f1 = self.mapU[int(u0)][int(v0)]

        u2 = u0 + dt * f1 / 2.0
        v2 = v0 + dt * f1 / 2.0

        if (abs(u2) >= self.mapU.shape[0]):
            if (u2 >= 0):
                u2 = x_axis - 1
            else:
                u2 = 0
        if (abs(v2) >= self.mapU.shape[1]):
            if (v2 >= 0):
                v2 = y_axis - 1
            else:
                v2 = 0

        f2 = self.mapU[int(u2)][int(v2)]


        u3 = u0 + dt / 2.0
        v3 = v0 + dt * f2 / 2.0

        if (abs(u3) >= self.mapU.shape[0]):
            if(u3 >= 0):
                u3 = x_axis - 1
            else:
                u3 = 0
        if (abs(v3) >= self.mapU.shape[1]):
            if(v3 >= 0):
                v3 = y_axis - 1
            else:
                v3 = 0

        f3 = self.mapU[int(u3)] [int(v3)]


        u4 = u0 + dt * f3
        v4 = v0 + dt

        if (abs(u4) >= self.mapU.shape[0]):
            if(u4 >= 0):
                u4 = x_axis - 1
            else:
                u4 = 0
        if (abs(v4) >= self.mapU.shape[1]):
            if(v4 >= 0):
                v4 = y_axis - 1
            else:
                v4 = 0

        f4 = self.mapU[int(u4)] [int(v4)]

        u1 = u0 + dt * (f1 + 2.0 * f2 + 2.0 * f3 + f4) / 6.0

        return u1
    def trace_V(self, u0, v0, dt=1):
        #modify this to fit with interpolation

        x_axis = self.mapU.shape[0]
        y_axis = self.mapU.shape[1]

        if (abs(u0) >= x_axis):
            if (u0 >= 0):
                u0 = x_axis - 1
            else:
                u0 = 0
        if (abs(v0) >= y_axis):
            if (v0 >= 0):
                v0 = y_axis - 1
            else:
                v0 = 0

        f1 = self.mapV[int(u0)][int(v0)]

        u2 = u0 + dt * f1 / 2.0
        v2 = v0 + dt * f1 / 2.0

        if (abs(u2) >= self.mapV.shape[0]):
            if (u2 >= 0):
                u2 = x_axis - 1
            else:
                u2 = 0
        if (abs(v2) >= self.mapV.shape[1]):
            if (v2 >= 0):
                v2 = y_axis - 1
            else:
                v2 = 0


        f2 = self.mapV[int(u2)][int(v2)]

        u3 = u0 + dt / 2.0
        v3 = v0 + dt * f2 / 2.0

        if (abs(u3) >= self.mapV.shape[0]):
            if(u3 >= 0):
                u3 = x_axis - 1
            else:
                u3 = 0

        if (abs(v3) >= self.mapV.shape[1]):
            if (v3 >= 0):
                v3 = y_axis - 1
            else:
                v3 = 0

        f3 = self.mapV[int(u3)] [int(v3)]

        u4 = u0 + dt
        v4 = v0 + dt * f3

        if (abs(u4) >= self.mapV.shape[0]):
            if(u4 >= 0):
                u4 = x_axis - 1
            else:
                u4 = 0
        if (abs(v4) >= self.mapV.shape[1]):
            if(v4 >= 0):
                v4 = y_axis - 1
            else:
                v4 = 0
        f4 = self.mapV[int(u4)] [int(v4)]

        v1 = v0 + dt * (f1 + 2.0 * f2 + 2.0 * f3 + f4) / 6.0

        return v1

    def computeFTLE(self, time : int):
        ftle = []
        for i in range(1, time):
            ftle.append(self.calc_ftle(delta=1, shape=self.particle_shape, start=0, end=i))
        ftle = np.asarray(a=ftle, dtype=np.float32)
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
    field = Field(mapU=dataU[0], mapV=dataV[0], trace_time=62, num_partical=250000, interpolate_factor=10)
    ftles = field.computeFTLE(time=60)

    for i in range(ftles.shape[0]-1):
        ftle1 = ftles[i]
        ftle2 = ftles[i+1]
        if ((ftle1==ftle2).all()):
            print(f"ftle {i} and {i+1} are the same")










