# Bilinear test in OpenCL 

OpenCL implementation of the Bilinear test program

## Prerequisites

```
sudo apt install clinfo opencl-c-headers ocl-icd-libopencl1
```

## Building

On [FriendlyARM NanoPi M4](https://wiki.friendlyarm.com/wiki/index.php/NanoPi_M4):

```
$ make
xxd -i bilinear.cl | gcc  -xc - -c -o bilinear.cl.o
g++ -g -O3 -c EasyBMP/EasyBMP.cpp -o EasyBMP.o
g++ -IEasyBMP -g -O3 bilinear.cpp bilinear.cl.o EasyBMP.o -o bilinear_opencl -L/usr/lib/aarch64-linux-gnu -lOpenCL
```

## Testing

```
$ ./bilinear_opencl ../thefox.bmp 
./bilinear_opencl: /usr/lib/mali/libOpenCL.so.1: no version information available (required by ./bilinear_opencl)
EasyBMP Warning: Extra meta data detected in file ../thefox.bmp
                 Data will be skipped.
arm_release_ver of this libmali is 'r18p0-01rel0', rk_so_ver is '4'.Using OpenCL device "Mali-T860" produced by "ARM" with OpenCL C 1.2 v1.r18p0-01rel0.5cb5681058e8e076ff89747c20c32578
Building kernel for device #0:
GPU kernel time = 0.005610 sec
```

