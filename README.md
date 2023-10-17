# CuPy
This tutorial demonstrates how we can use CuPy to implement NVDIA GPU targeted codes in Python.

Learning outcomes of the tutorial are:
1. Learn the basics of CuPy.

Prerequisite:
1. Experience with Numpy. 

***

CuPy is an open-source array library designed for harnessing GPU acceleration in Python-based computing. Leveraging CUDA Toolkit libraries, such as cuBLAS, CuPy  exploits the capabilities of the NVIDIA GPU architecture. Moreover, CuPy boasts a highly compatible interface with NumPy and SciPy, making it a seamless drop-in replacement in the majority of use cases. Transitioning to CuPy usually entails substituting 'numpy' and 'scipy' with 'cupy.'

``cupy.ndarray `` is akin to the ``numpy.ndarray ``. It is an array object that represents a multidimensional, homogeneous array of fixed-size items. This is the core of CuPy. A call to the ``numpy.array()`` allocates the data in the main memory, while a call to the ``cupy.array()`` allocates the data in the GPU memory. If no device is specified the memory gets allocated in the ``current`` device.
