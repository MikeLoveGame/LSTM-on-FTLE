import os
import numpy as np

import vtk
def numpy_to_vtk(numpy_array):
    # Extract dimensions of the array
    dims = numpy_array.particle_shape

    # Create a vtkImageData object
    imageData = vtk.vtkImageData()
    imageData.SetDimensions(dims[2], dims[1], dims[0])  # Set dimensions in reverse order (Z, Y, X)
    imageData.AllocateScalars(vtk.VTK_FLOAT, 1)  # Allocate memory for scalar values

    # Copy NumPy array data to VTK image
    vtk_array = imageData.GetPointData().GetScalars()
    flat_array = numpy_array.flatten(order='F')  # Flatten the NumPy array in column-major order
    vtk_array.SetVoidArray(flat_array, len(flat_array), 1)

    return imageData

def save_vtk(filename, imageData):
    writer = vtk.vtkWriter()

    writer.SetFileName(filename)
    writer.SetInputData(imageData)
    status = writer.Write()
    print(status)

source_dir=r"D:\AI-Data\Field-Data\0001\result-200\0"
files = os.listdir(source_dir)

for file in files:
    if (".npy" in file):

        file_path = os.path.join(source_dir, file)
        x = np.load(file_path)
        data = numpy_to_vtk(x)
        save_vtk(filename="output.vti", imageData=data)
