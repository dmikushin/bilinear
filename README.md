# A simple image filter example for those who study OpenCL programming

[See also the CUDA/HIP version](https://github.com/dmikushin/bilinear/tree/cuda)

OpenCL implementation of bilinear image filter (increases image resolution by factor of two). Runs on NVIDIA, AMD and ARM (Mali) devices.

<img width="400px" src="screenshot.png"/>

## Preparation

Get the code and a sample image:

```bash
git clone https://github.com/dmikushin/bilinear.git
cd bilinear
wget https://photojournal.jpl.nasa.gov/jpeg/PIA00004.jpg
convert PIA00004.jpg PIA00004.bmp
```

Note we use very large input image e.g. from NASA space missions. The larger is an image, the greater is the chance to saturate the **massive parallelism** of many GPU cores, especially if the GPU is big, such as NVIDIA Volta V100 (which is used in the performance measurements below).

The following prerequisites must be fitfuled e.g. on Ubuntu:

```
sudo apt install clinfo opencl-c-headers ocl-icd-libopencl1
```

[CMake](https://cmake.org/download/) is required to build the executables, both Linux and Windows platforms are supported. The CUDA toolkit must present in the system; alternatively, ROCm/HIP could be used to compile the same code for AMD GPUs.

We recommend [Ninja](https://ninja-build.org/) as a CMake generator, because it supports both Linux and Windows, as well as the parallel compilation. The following commands could be executed from the terminal (Visual Studio Command Prompt, in case of Windows), in order to build the executables:

```bash
cd bilinear
mkdir build
cmake -G Ninja -DCMAKE_BUILD_TYPE=Release ..
ninja
```

## Testing

* On [FriendlyARM NanoPi M4](https://wiki.friendlyarm.com/wiki/index.php/NanoPi_M4):

```
$ ./bilinear_opencl ../thefox.bmp 
./bilinear_opencl: /usr/lib/mali/libOpenCL.so.1: no version information available (required by ./bilinear_opencl)
EasyBMP Warning: Extra meta data detected in file ../thefox.bmp
                 Data will be skipped.
arm_release_ver of this libmali is 'r18p0-01rel0', rk_so_ver is '4'.Using OpenCL device "Mali-T860" produced by "ARM" with OpenCL C 1.2 v1.r18p0-01rel0.5cb5681058e8e076ff89747c20c32578
Building kernel for device #0:
GPU kernel time = 0.005610 sec
```

* On AMD Radeon RX Vega gfx900:

```
$ ./bilinear_opencl ../thefox.bmp 
EasyBMP Warning: Extra meta data detected in file ../thefox.bmp
                 Data will be skipped.
Using OpenCL device "Radeon RX Vega" produced by "Advanced Micro Devices, Inc." with OpenCL C 2.0 
Building kernel for device #0:
GPU kernel time = 0.000067 sec
```

