#ifndef CELL_H
#define CELL_H
#include <iostream>
#include "math.h"
#include <cuda_runtime.h>
#include <ctime>
#include "device_launch_parameters.h"
//#include "device_atomic_functions.h"
#include "cuda.h"
#include <cublas_v2.h>

using namespace std;

//code acquired from website. (see .cuh comment for details)
//https://gist.github.com/jefflarkin/5390993
#define cudaCheckError() {                                          \
  cudaError_t e=cudaGetLastError();                                 \
  if(e!=cudaSuccess) {                                              \
    printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__, cudaGetErrorString(e));           \
    system("pause");												\
    exit(0); \
  }                                                                 \
}


class Cell{
  unsigned int size;
  unsigned int *numerals; //need to track and update numeral list/size.
  unsigned int prop;

  template<class t>
  __host__ __device__ t *allocateHost(unsigned int);
  template<class t>
  __host__ __device__ t *allocateDevice(unsigned int);
public:
  __host__ __device__ Cell(unsigned int);
  __host__ __device__ unsigned int getSize();
  __host__ __device__ void setNumeral(unsigned int, unsigned int);
  __host__ __device__ unsigned int * getNumeral();
  __host__ __device__ void setProp(unsigned int, unsigned int);
  __host__ __device__ unsigned int getProp();
  __host__ __device__ void free_cell();
};

#endif
