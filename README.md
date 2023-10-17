# CuPy
This tutorial demonstrates how we can use CuPy to implement NVDIA GPU targeted codes in Python.

Learning outcomes of the tutorial are:
1. Learn the basics of CuPy.

Prerequisite:
1. Experience with Numpy. 

***

## Current Device
CuPy has a concept of a current device, which is the default GPU device on which on which all operation of related to CuPy takes place. Unless specifically mentioned, all operation taskes place in this default device.

```
cp.cuda.runtime.getDevice()
```
***

## cupy.ndarray
CuPy is an open-source array library designed for harnessing GPU acceleration in Python-based computing. Leveraging CUDA Toolkit libraries, such as cuBLAS, CuPy  exploits the capabilities of the NVIDIA GPU architecture. Moreover, CuPy boasts a highly compatible interface with NumPy and SciPy, making it a seamless drop-in replacement in the majority of use cases. Transitioning to CuPy usually entails substituting 'numpy' and 'scipy' with 'cupy.'

``cupy.ndarray `` is akin to the ``numpy.ndarray ``. It is an array object that represents a multidimensional, homogeneous array of fixed-size items. This is the core of CuPy. A call to the ``numpy.array()`` allocates the data in the main memory, while a call to the ``cupy.array()`` allocates the data in the GPU memory. If no device is specified the memory gets allocated in the ``current`` device.

```
x_cpu = np.array([1, 2, 3]) # allocate an ndarray in the main memory
x_gpu = cp.array([1, 2, 3]) # allocate an ndarray in the GPU memory
```
***

## Multi-GPU Operations

Unless specified, CuPy always assumes that the operations are performed on the currently active device (which is usually the device with the device index 0). To make use of multiple GPUs, we can use the  device context manage to switch between the devices.

```
with cp.cuda.Device(1):  # alloacte an ndarray in the gpu memory of the device with device index 1
    x_on_gpu1 = cp.array([1, 2, 3, 4, 5])
```

