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
x_cpu = np.array([1, 2, 3]) # allocate an ndarray in the Host memory
x_gpu = cp.array([1, 2, 3]) # allocate an ndarray in the GPU memory
```
***

## Multi-GPU Operations

Unless specified, CuPy always assumes that the operations are performed on the currently active device (which is usually the device with the device index 0). To make use of multiple GPUs, we can use the  device context manage to switch between the devices.

```
with cp.cuda.Device(1):  # alloacte an ndarray in the gpu memory of the device with device index 1
    x_on_gpu1 = cp.array([1, 2, 3, 4, 5])
```

## Data Transfers

In a normal CUDA workflow we have to allocate the memory on the GPU and then move the data to the GPU memory. In CuPy this is not required, the memory allocation and data movement can be done in a single operation.

```
x_cpu = np.array([1, 2, 3])
x_gpu_0 = cp.asarray(x_cpu)  # move the ndarray from Host memory to GPU0 memeory.
```

We can also transfer data between GPUs. 
```
with cp.cuda.Device(1):
    x_gpu_1 = cp.asarray(x_gpu_0)  # move the ndarray to GPU0 to GPU1.
```
In the past any communication between two GPUs had to go throgh the PCIe card. But now NVIDIA offeres a technology called NVLink. NVLink is a direct GPU-to-GPU interconnect that scales multi-GPU input/output (IO) within a node. This makes GPU-to-GPU transfer (D2D tranfer) much faster than GPU-to-Host (D2H transfer) or Host-to-GPU transfer (H2D transfer). 

There are two ways to transfer data from the GPU memory to Host memory- ``cupy.ndarray.get()`` or ``cupy.asnumpy``. 

```
with cp.cuda.Device(0):
    x_cpu = cp.asnumpy(x_gpu_0)  # move the array from GPU 0 back to the Host memory.

with cp.cuda.Device(1):
    x_cpu = x_gpu_1.get()  # move the array from GPU 1 back to the Host memory.
```

***

## Device Agnostic Codes

As cupy mimicks numpy we can build device agnostics codes. That is, we can make function calls to a data, without the knowledge of where they reside. The ``cupy.get_array_module()`` function in CuPy returns a reference to cupy if any of its arguments resides on a GPU and numpy otherwise.

```
def log_array(x):
    xp = cp.get_array_module(x)  # cupy ndarray array reference is returned if x is in GPU memory
                                 # numpy ndarray array reference is returned if x is in Host memory
    xp.log1p(xp.exp(-abs(x))) 
```

## User Defined Kernels

Kernels are functions that will be run on the GPU. CuPy provides easy ways to define three types of CUDA kernels: *elementwise kernels, reduction kernels* and *raw kernels*.

### Elementwise Kernel
A definition of an elementwise kernel consists of four parts: 
1. An input argument list
2. An output argument list
3. A loop body code
4. Kernel name.
  
A kernel that computes an elementwise  difference can be defined as follows:

```
element_diff = cp.ElementwiseKernel('float32 x, float32 y', 
                                    'float32 z', 
                                    'z = (x - y)', 
                                    'element_diff')
```

### Reduction Kernel
Reduction operation is a computation where we reduce the elements of an array into a single result. In CuPy a reduction kernels can help with reduction operation. A reduction kernel un CuPy has four parts:

1. Identity value: This value is used for the initial value of reduction.
2. Mapping expression: It is used for the pre-processing of each element to be reduced.
3. Reduction expression: It is an operator to reduce the multiple mapped values. The special variables a and b are used for its operands.
4. Post mapping expression: It is used to transform the resulting reduced values. The special variable a is used as its input. Output should be written to the output parameter.

```
reduction_kernel = cp.ReductionKernel(
    'T x', # input param
    'T y',  # output param
    'x',  # map
    'a + b',  # reduce
    'y = a',  # post-reduction map
    '0',  # identity value
    'reduction_kernel'  # kernel name
)
```

### Raw Kernel

Raw Kernels are used to define kernels from raw CUDA source.

```
add_kernel = cp.RawKernel(r'''
    extern "C" __global__
    void my_add(const float* x1, const float* x2, float* y) 
    {
        int tid = blockDim.x * blockIdx.x + threadIdx.x;
        y[tid] = x1[tid] + x2[tid];
    }''', 'my_add')
```

***

## Cuda Events and Cuda Streams

CuPy also provide fucntionalities for creating streams and events. Data copies and kernel launches are enqueued onto the Current Stream, which can be queried via get_current_stream() and changed either by setting up a context manager.

```
e1 = cp.cuda.Event() # create an event
e1.record() # Records an event on the stream
a_cp = b_cp * a_cp + 8
e2 = cp.cuda.get_current_stream().record() # create and record the event
```

Just like a *current device* CuPy also has a concept of *current streams*. CuPu by default launches all operation in the current stream. At the same time, CUDA device can have multiple streams and we can use a different stream from the default stream.  

```
s = cp.cuda.Stream()
with s:
   a_cp = cp.asarray(a_np)  # H2D transfer on stream s
   b_cp = cp.sum(a_cp)      # kernel launched on stream s 
```

***

# References
1. https://docs.cupy.dev/en/stable/user_guide/basic.html
2. https://docs.cupy.dev/en/stable/user_guide/kernel.html
3. https://docs.cupy.dev/en/stable/user_guide/cuda_api.html
4. 


