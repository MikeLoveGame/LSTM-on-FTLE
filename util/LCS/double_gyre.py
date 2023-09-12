import numpy as np
import matplotlib.pyplot as plt
import os

# Define the double gyre velocity field
def double_gyre_velocity(x, y, t, A=0.15, a=0.15, b=0.15, omega=2*np.pi/10):
    f = a * np.sin(omega * t) * x + b * np.sin(2 * np.pi * x)
    u = -np.pi * A * np.sin(np.pi * f) * np.cos(np.pi * y)
    v = np.pi * A * np.cos(np.pi * f) * np.sin(np.pi * y) * (a * np.sin(omega * t) + b * np.cos(2 * np.pi * x))
    return u, v

# Create a grid of points
x = np.linspace(0, 1, 500)
y = np.linspace(0, 1, 500)
X, Y = np.meshgrid(x, y)

# Get the velocity field for a given time
t = 0.5
U, V = double_gyre_velocity(X, Y, t)


savePath_U = r"C:\Github-repository\LSTM-on-FTLE\LSTM-on-FTLE\data\double-gyre\U"
savePath_V = r"C:\Github-repository\LSTM-on-FTLE\LSTM-on-FTLE\data\double-gyre\V"

UPath = os.path.join(savePath_U, "U.npy")
VPath = os.path.join(savePath_V, "V.npy")
np.save(file=UPath, arr=U)
np.save(file=VPath, arr=V)

'''fig, ax = plt.subplots(figsize=(10, 5))
ax.quiver(X, Y, U, V, scale=15, color='blue')
ax.set_title("Double Gyre Vector Field at t = {}".format(t))
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()
'''