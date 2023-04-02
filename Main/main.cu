#include <iostream>
#include <fstream>
#include "../Initializer/Initializer.cuh"
using namespace std;

int checkDeviceProp(cudaDeviceProp p)
{
	int support = p.concurrentKernels;
	if (support == 0)
		printf("%s does not support concurrent kernels\n",
		p.name);
	else
		printf("%s supports concurrent kernels\n", p.name);
	printf(" compute capability : %d.%d \n",
		p.major, p.minor);
	printf(" number of multiprocessors : %d \n",
		p.multiProcessorCount);
	return support;
}

// Print device properties
void printDevProp(cudaDeviceProp devProp)
{
  printf("Major revision number:         %d\n",  devProp.major);
  printf("Minor revision number:         %d\n",  devProp.minor);
  printf("Name:                          %s\n",  devProp.name);
  printf("Total global memory:           %lu\n",  devProp.totalGlobalMem);
  printf("Total shared memory per block: %lu\n",  devProp.sharedMemPerBlock);
  printf("Total registers per block:     %d\n",  devProp.regsPerBlock);
  printf("Warp size:                     %d\n",  devProp.warpSize);
  printf("Maximum memory pitch:          %lu\n",  devProp.memPitch);
  printf("Maximum threads per block:     %d\n",  devProp.maxThreadsPerBlock);
  for (int i = 0; i < 3; ++i)
      printf("Maximum dimension %d of block:  %d\n", i, devProp.maxThreadsDim[i]);
  for (int i = 0; i < 3; ++i)
      printf("Maximum dimension %d of grid:   %d\n", i, devProp.maxGridSize[i]);
  printf("Clock rate:                    %d\n",  devProp.clockRate);
  printf("Total constant memory:         %lu\n",  devProp.totalConstMem);
  printf("Texture alignment:             %lu\n",  devProp.textureAlignment);
  printf("Concurrent copy and execution: %s\n",  (devProp.deviceOverlap ? "Yes" : "No"));
  printf("Number of multiprocessors:     %d\n",  devProp.multiProcessorCount);
  printf("Kernel execution timeout:      %s\n",  (devProp.kernelExecTimeoutEnabled ?"Yes" : "No"));
  return;
}

void testDevice(){
	// cudaDeviceReset must be called before exiting in order for profiling and
  // tracing tools such as Nsight and Visual Profiler to show complete traces.
  cudaDeviceReset();
  cudaCheckError();

  // Choose which GPU to run on, change this on a multi-GPU system.
  cudaSetDevice(0);
  cudaCheckError();

  //check support for concurrent kernel launching.
  cudaDeviceProp dev;
  cudaGetDeviceProperties(&dev, 0);

  if (checkDeviceProp(dev) != 0){
    cout << "Device support concurrent kernel operations" << endl;
  }
  printDevProp(dev);
}

int main(int argc, char** argv){
	testDevice();

    if(argc != 2) { 
      std::cerr << "Usage: " << argv[0] << " a puzzle text" << std::endl; 
    }
  
    fstream puzFile;  
    puzFile.open (argv[1]);
    if(!puzFile){cerr << "Error opening file" << endl;}

    int size=0;
    puzFile >> size;

	Initializer *aInitializer;
	cudaMallocHost((void**)&aInitializer, sizeof(Initializer));
	aInitializer[0] = Initializer(size);
  aInitializer[0].run(puzFile);
  aInitializer[0].printPuzzle();
  
  puzFile.close();
  return 0;
}
//Hardik start refactoring EPRFS with puzzle class values.
//create project with puzzle class
//work on linking database and program to accept file name to pull info from database done in main
//us that data to construct puzzle class