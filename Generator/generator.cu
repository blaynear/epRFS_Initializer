#include "generator.cuh"

const int baseThreadCount = 64;
/***************************************************************************/
/*   											      Cuda Kernels   														 */
/***************************************************************************/
__global__ void setPuzzle(unsigned int *size, Puzzle *aPuzzle){
  unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

  if(tid == 0){
    aPuzzle[0] = Puzzle(size[0]);
  }
  __syncthreads();
  while(tid < pow((double)size[0], 2)){
    aPuzzle[0].allocatePuzzle(tid);
    tid++;
  }
}
/***************************************************************************/
/*   														Constructors   														 */
/***************************************************************************/

__host__ Generator::Generator(unsigned int size){
  unsigned int *d_size = allocateDevice<unsigned int>(1);
  this->d_puzzle = allocateDevice<Puzzle>(1);
  
  cudaMemcpy(d_size, &size, sizeof(unsigned int), cudaMemcpyHostToDevice);
  cudaCheckError();
  setPuzzle<<<1,this->blockSize>>>(d_size, d_puzzle);

  /***********************************************************/
  /*                    From here, implement:
  /*                       Clue selection
  /*                       Extermination
  /***********************************************************/
  this->blockSize = getThreadCount(size * size);
  this->gridSize = getBlockCount(size * size, this->blockSize);
}

/***************************************************************************/
/*														Getter Functions														 */
/***************************************************************************/

/***************************************************************************/
/*														Setter Functions														 */
/***************************************************************************/

/***************************************************************************/
/*														CUDA Helper Masks														 */
/***************************************************************************/
template<class t>
__host__ t *Generator::allocateHost(unsigned int size){
	t *aValue;
	cudaMallocHost((void **)&aValue, size * sizeof(t));
	cudaCheckError();
	return aValue;
}

template<class t>
__host__ t *Generator::allocateDevice(unsigned int size){
	t *aValue;
	cudaMalloc((void **)&aValue, size * sizeof(t));
	cudaCheckError();
	return aValue;
}

__host__ __device__ unsigned int Generator::getThreadCount(unsigned int offset){
	return min(((offset / 512) + 1)*baseThreadCount, 512);
}

__host__ __device__ unsigned int Generator::getBlockCount(unsigned int offset, unsigned int numberOfThreads){
	return (offset / numberOfThreads) + 1;
}
