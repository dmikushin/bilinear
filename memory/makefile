.SUFFIXES: .cu

all: bilinear_cuda

EasyBMP.o: EasyBMP/EasyBMP.cpp EasyBMP/*.h
	g++ -g -O3 -c $< -o $@

bilinear_cuda: bilinear.cpp bilinear.cu.o EasyBMP.o
	nvcc -IEasyBMP -g -O3 -arch=sm_30 $^ -o $@ -lcuda -lnvrtc

bilinear.cu.o: bilinear.cu
	xxd -i $< | gcc $(OPT) -xc - -c -o $@

clean:
	rm -rf *.o bilinear_cuda output_gpu.bmp

