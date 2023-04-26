#include "initializer.cuh"

using namespace std;
const int baseThreadCount = 64;


/***************************************************************************/
/*   											      Global Functions													 */
/***************************************************************************/
__host__ __device__ unsigned int getRow(int index, int dim){
	return index % dim;
}

__host__ __device__ unsigned int getCol(int index, int dim){
	return (index / dim) % dim;
}

__host__ __device__ unsigned int getNum(int index, int dim){
	return index / (dim*dim);
}

__host__ __device__ unsigned int get3DIndex(int x, int y, int z, int dim) {
	return x + dim*(y + dim*z);
}

__host__ __device__ unsigned int getThreadCount(unsigned int offset){
	return min(((offset / 512) + 1)*baseThreadCount, 512);
}

__host__ __device__ unsigned int getBlockCount(unsigned int offset, unsigned int numberOfThreads){
	return (offset / numberOfThreads) + 1;
}


/***************************************************************************/
/*   											      Cuda Kernels   														 */
/***************************************************************************/
__global__ void allocatePuzzle(unsigned int *size, Puzzle *aPuzzle, unsigned int *species){
  aPuzzle[0] = Puzzle(size[0], species[0]);
}

__global__ void setPuzzle(unsigned int *size, Puzzle *aPuzzle, unsigned int ***searchSpace){
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int row, col, num, aValue;

  for(int i = tid; i < pow(size[0],3); i += blockDim.x*gridDim.x){
    row = getRow(i, size[0]);
    col = getCol(i, size[0]);
    num = getNum(i, size[0]);
    aValue = searchSpace[row][col][num];
    aPuzzle[0].puzzleSet(row, col, num, aValue);
  }
}

__global__ void exterminateRegions(unsigned int *size, unsigned int ***searchSpace, unsigned int aClue){
  double sqrtSize = size[0];
  sqrtSize = sqrt(sqrtSize);

  int tid = threadIdx.x + blockIdx.x * blockDim.x,
    row = getRow(aClue, size[0]),
	  col = getCol(aClue, size[0]),
	  num = getNum(aClue, size[0]),
	  aClue_region_row = row / sqrtSize,// Region row 0,1, or 2
	  aClue_region_col = col / sqrtSize;

  if(tid < size[0]){
    searchSpace[tid][col][num] = 1;
    searchSpace[row][tid][num] = 1;
    searchSpace[row][col][tid] = 1;
    searchSpace[aClue_region_row * (int)sqrtSize + tid%(int)sqrtSize][aClue_region_col * (int)sqrtSize + tid/(int)sqrtSize][num] = 1;
  }
}

__global__ void exterminate(unsigned int *size, unsigned int *listOfClues, unsigned int ***searchSpace, unsigned int *numClues){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if(tid < numClues[0]){
		cudaStream_t stream;
		cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    int threadCount = getThreadCount(size[0]);
    int blockCount = getBlockCount(size[0], threadCount);
    exterminateRegions<<<blockCount, threadCount, 0, stream>>>(size, searchSpace, listOfClues[tid]);
    cudaDeviceSynchronize();
		cudaStreamDestroy(stream);
  }
}

__global__ void printPuzzleFile(Puzzle *aPuzzle, unsigned int size){
  printf("numSpecies: %d\n", aPuzzle[0].getNumSpecies());
  
  for (int row = 0; row < size; row++){
		for (int col = 0; col < size; col++){
      if (col == size - 1){
        printf("%u\t\n", aPuzzle[0].getCellProp(row, col));
      }else{
        printf("%u, \t", aPuzzle[0].getCellProp(row, col));
      }
		}
	}
}

/***************************************************************************/
/*   														Constructors   														 */
/***************************************************************************/

__host__ Initializer::Initializer(unsigned int size){
  this->d_puzzle = allocateDevice<Puzzle>(1);
  this->threads = getThreadCount(size*size);
  this->blocks = getBlockCount(size*size, this->threads);
  this->size = size;
  cudaCheckError();
}

/***************************************************************************/
/*   														Run           														 */
/***************************************************************************/

__host__ void Initializer::run(fstream &file){
  int row=0, col=0, num=0, clues=0;
  file >> clues;

  unsigned int *d_size = allocateDevice<unsigned int>(1);
  unsigned int *c = allocateDevice<unsigned int>(1);
  unsigned int *d_numClues = allocateDevice<unsigned int>(1);
  unsigned int *d_listClues = allocateDevice<unsigned int>(clues);
  
  cudaMemcpy(d_size, &this->size, sizeof(unsigned int), cudaMemcpyHostToDevice);
  cudaCheckError();
  cudaMemcpy(d_numClues, &clues, sizeof(unsigned int), cudaMemcpyHostToDevice);
  cudaCheckError();

  unsigned int *listOfClues = allocateHost<unsigned int>(clues);

  int index = 0;
  while (file >> row >> col >> num){
    listOfClues[index] = get3DIndex(row - 1, col - 1, num - 1,  this->size);
    index++;
  }

  cudaMemcpy(d_listClues, listOfClues, clues*sizeof(unsigned int), cudaMemcpyHostToDevice);
  cudaCheckError();

  unsigned int ***h_searchSpace = allocateHost<unsigned int**>(this->size);
  for(int i = 0; i < this->size; i++){
    h_searchSpace[i] = allocateHost<unsigned int*>(this->size);
  }

  for(int i = 0; i < this->size; i++){
    for(int j = 0; j < this->size; j++){
      h_searchSpace[i][j] = allocateHost<unsigned int>(this->size);
    }
  }

  unsigned int ***d_searchSpace = allocateDevice<unsigned int**>(this->size);
  cudaMemcpy(d_searchSpace, h_searchSpace, this->size*sizeof(unsigned int**), cudaMemcpyHostToDevice);
  cudaCheckError();

  int threadsC = getThreadCount(clues);
  int blocksC = getBlockCount(clues, threadsC);

  exterminate<<<blocksC, threadsC>>>(d_size, d_listClues, d_searchSpace, d_numClues);
  cudaCheckError();

  cudaMemcpy(h_searchSpace, d_searchSpace, this->size*sizeof(unsigned int**), cudaMemcpyDeviceToHost);
  cudaCheckError();

  unsigned int species = 0;
  for(int i = 0; i < this->size; i++){
    for(int j = 0; j < this->size; j++){
      for(int k = 0; k < this->size; k++){
        species += (h_searchSpace[i][j][k]+1)%2;
      }
    }
  }

  cudaMemcpy(d_numClues, &species, sizeof(unsigned int), cudaMemcpyHostToDevice);

  allocatePuzzle<<<1,1>>>(d_size, d_puzzle, d_numClues);
  cudaCheckError();
  setPuzzle<<<this->blocks,this->threads>>>(d_size, d_puzzle, d_searchSpace);
  cudaCheckError();

  cudaFree(d_size);
  cudaFree(d_numClues);
  cudaFree(d_listClues);
  cudaFree(d_searchSpace);
  cudaCheckError();
}

/***************************************************************************/
/*   														Print          														 */
/***************************************************************************/

__host__ void Initializer::printPuzzle(){
  printf("\nSudoku Puzzle Printout\n");
  this->h_puzzle = allocateHost<Puzzle>(1);

  cudaMemcpy(h_puzzle, this->d_puzzle, sizeof(Puzzle), cudaMemcpyDeviceToHost);
  cudaCheckError();

  printPuzzleFile<<<1,1>>>(h_puzzle, this->size);
  cudaCheckError();

  cudaFreeHost(h_puzzle);
  cudaCheckError();
}

__host__ __device__ Puzzle * Initializer::getPuzzle(){
  Puzzle *temp_puzzle = allocateHost<Puzzle>(1);

  cudaMemcpy(temp_puzzle, this->d_puzzle, sizeof(Puzzle), cudaMemcpyDeviceToHost);
  cudaCheckError();
  return temp_puzzle;
}



/***************************************************************************/
/*														Getter Functions														 */
/***************************************************************************/

__host__ __device__ unsigned int Initializer::get3DIndex(int x, int y, int z, int dim) {
	return x + dim*(y + dim*z);
}

__host__ __device__ unsigned int Initializer::getSize(){
	return this->d_puzzle[0].getSize();
  }

/***************************************************************************/
/*														CUDA Helper Masks														 */
/***************************************************************************/

template<class t>
t *Initializer::allocateDevice(unsigned int size){
	t *aValue;
	cudaMalloc((void **)&aValue, size * sizeof(t));
	cudaCheckError();
	return aValue;
}

template<class t>
t *Initializer::allocateHost(unsigned int size){
	t *aValue;
	cudaMallocHost((void **)&aValue, size * sizeof(t));
	cudaCheckError();
	return aValue;
}