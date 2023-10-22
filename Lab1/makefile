CC = gcc

all: local server omp

local: fft.c
	$(CC) fft.c -O3 -fopenmp -lm -Wall -o fft

server: fft.c
	$(CC) fft.c -O3 -fopenmp -lfftw3_omp -lfftw3 -lm -DSERVER -Wall -o fft 

omp: omp_sample.c
	$(CC) omp_sample.c -O3 -fopenmp -o omp 

clean:
	rm -f fft omp