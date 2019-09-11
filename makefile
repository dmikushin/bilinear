CXX := g++ -g -O3 -fopenmp
NVCC := nvcc -g -O3 -arch=sm_61

all: bilinear_gpu bilinear_cpu

EasyBMP.o: EasyBMP/EasyBMP.cpp EasyBMP/*.h
	$(CXX) -c $< -o $@

bilinear_cpu: bilinear.cpp EasyBMP.o
	$(CXX) -IEasyBMP $^ -o $@

bilinear_gpu: bilinear.cu EasyBMP.o
	$(NVCC) -IEasyBMP $^ -o $@

clean:
	rm -rf *.o bilinear_cpu bilinear_gpu output_cpu.bmp output_gpu.bmp

