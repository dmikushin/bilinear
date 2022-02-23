.SUFFIXES: .cl

CL_LIBS = -L/usr/lib/aarch64-linux-gnu -L/opt/rocm-5.0.1/lib/

all: bilinear_opencl

EasyBMP.o: EasyBMP/EasyBMP.cpp EasyBMP/*.h
	g++ -g -O3 -c $< -o $@

bilinear_opencl: bilinear.cpp bilinear.cl.o EasyBMP.o
	g++ -IEasyBMP -g -O3 $^ -o $@ $(CL_LIBS) -lOpenCL

bilinear.cl.o: bilinear.cl
	xxd -i $< | gcc $(OPT) -xc - -c -o $@

clean:
	rm -rf *.o bilinear_opencl output_gpu.bmp

