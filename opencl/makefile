.SUFFIXES: .cl

OPENCL = /usr/lib/aarch64-linux-gnu

all: bilinear_opencl

EasyBMP.o: EasyBMP/EasyBMP.cpp EasyBMP/*.h
	g++ -g -O3 -c $< -o $@

bilinear_opencl: bilinear.cpp bilinear.cl.o EasyBMP.o
	g++ -IEasyBMP -g -O3 $^ -o $@ -L$(OPENCL) -lOpenCL

bilinear.cl.o: bilinear.cl
	xxd -i $< | gcc $(OPT) -xc - -c -o $@

clean:
	rm -rf *.o bilinear_opencl output_gpu.bmp

