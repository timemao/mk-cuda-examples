#Change the arch to be the type of your GPU
nvcc -O3 -arch=sm_21 --use_fast_math -Xcompiler -fopenmp pcaNM.cu -o pcaNM_GPU32
nvcc -O3 -arch=sm_21 -D USE_DBL --use_fast_math -Xcompiler -fopenmp pcaNM.cu -o pcaNM_GPU64

nvcc -O3 -D USE_HOST -Xcompiler -fopenmp pcaNM.cu -o pcaNM_CPU32
nvcc -O3 -D USE_DBL -D USE_HOST -Xcompiler -fopenmp pcaNM.cu -o pcaNM_CPU64
