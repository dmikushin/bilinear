# A simple image filter example for those who study GPU/CUDA programming

Explore the performance advantages of GPU by example of bilinear image filter (increases image resolution by factor of two).

## Preparation

```bash
git clone https://github.com/dmikushin/bilinear.git
cd bilinear
wget https://photojournal.jpl.nasa.gov/jpeg/PIA00004.jpg
convert PIA00004.jpg PIA00004.bmp
```

Note we use very large input image e.g. from NASA space missions. The larger is an image, the greater is the chance to saturate the **massive parallelism** of many GPU cores, especially if the GPU is big, such as NVIDIA Volta V100 (which is used in the performance measurements below).

## The effect of manual RGBA 4-byte coalescing

```bash
cd unoptimized
make
./bilinear_cpu ../PIA00004.bmp
```

```
CPU kernel time = 0.897581 sec
```

```bash
cd unoptimized
make
./bilinear_gpu ../PIA00004.bmp
```

```
GPU kernel time = 0.003332 sec
```

```bash
cd coalescing
make
./bilinear_gpu ../PIA00004.bmp
```

```
GPU kernel time = 0.002590 sec
```

