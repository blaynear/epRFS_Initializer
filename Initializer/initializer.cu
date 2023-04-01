#include "initializer.cuh"

using namespace std;
const int baseThreadCount = 64;

__host__ __device__ unsigned int getThreadCount(unsigned int offset){
	return min(((offset / 512) + 1)*baseThreadCount, 512);
}

__host__ __device__ unsigned int getBlockCount(unsigned int offset, unsigned int numberOfThreads){
	return (offset / numberOfThreads) + 1;
}


/***************************************************************************/
/*   											      Cuda Kernels   														 */
/***************************************************************************/
__global__ void setPuzzle(unsigned int *size, Puzzle *aPuzzle){
  aPuzzle[0] = Puzzle(size[0]);
}

__global__ void numeralChanger(unsigned int *size, Puzzle *aPuzzle, unsigned int ***searchSpace){
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int row = tid % size[0];
  int col = (tid / size[0]) % size[0];

  for(int i = 0; i < size[0]; i++){
    aPuzzle[0].allocatePuzzle(row, col, i, searchSpace[row][col][i]);
  }
}

__global__ void exterminateRegions(unsigned int *size, unsigned int ***searchSpace, unsigned int aClue){
  double sqrtSize = size[0];
  sqrtSize = sqrt(sqrtSize);

  int tid = threadIdx.x + blockIdx.x * blockDim.x,
    row = aClue % size[0],
	  col = (aClue / size[0]) % size[0],
	  num = aClue / (size[0]*size[0]),
	  aClue_region_row = row / sqrtSize, // Region row 0,1, or 2
	  aClue_region_col = col / sqrtSize;

  if(tid < size[0]){
    //printf("clue: %d tid: %d row: %d col: %d num: %d\n", aClue, tid, row, col, num);
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

  for (int row = 0; row < size; row++){
		for (int col = 0; col < size; col++){
      if (col == size - 1){
        printf("%u\t\n", aPuzzle[0].getCellNumeral(row, col));
      }else{
        printf("%u, \t", aPuzzle[0].getCellNumeral(row, col));
      }
		}
	}
}

/***************************************************************************/
/*   														Constructors   														 */
/***************************************************************************/

__host__ Initializer::Initializer(unsigned int size){
  this->d_puzzle = allocateDevice<Puzzle>(1);
  this->size = size;
  this->threads = getThreadCount(size * size);
  this->blocks = getBlockCount(size * size, this->threads);
  cudaCheckError();
}

__host__ void Initializer::run(fstream &file){
  int row=0, col=0, num=0, clues=0;
  file >> clues;

  unsigned int *d_size = allocateDevice<unsigned int>(1);
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

  unsigned int ***h_searchSpace = allocateHost<unsigned int**>( this->size);
  
  for(int i = 0; i < this->size; i++){
    h_searchSpace[i] = allocateHost<unsigned int*>( this->size);
  }

  for(int i = 0; i < this->size; i++){
    for(int j = 0; j < this->size; j++){
      h_searchSpace[i][j] = allocateHost<unsigned int>( this->size);
    }
  }

  for(int i = 0; i < this->size; i++){
    for(int j = 0; j < this->size; j++){
      for(int k = 0; k < this->size; k++){
        h_searchSpace[i][j][k] = 0;
      }
    }
  }

  unsigned int ***d_searchSpace = allocateDevice<unsigned int**>(this->size);
  cudaMemcpy(d_searchSpace, h_searchSpace, this->size*sizeof(unsigned int**), cudaMemcpyHostToDevice);
  cudaCheckError();
  
  int threadCount = getThreadCount(clues);
  int blockCount = getBlockCount(clues, threadCount);//launch clueNumber blocks and threads

  exterminate<<<blockCount,threadCount>>>(d_size, d_listClues, d_searchSpace, d_numClues);
  cudaCheckError();

  cudaMemcpy(h_searchSpace, d_searchSpace, this->size*sizeof(unsigned int**), cudaMemcpyDeviceToHost);
  cudaCheckError();
  
  /*
  int numSpecies = 0;
  for(int i = 0; i < this->size; i++){
    for(int j = 0; j < this->size; j++){
      for(int k = 0; k < this->size; k++ ){
        if(h_searchSpace[i][j][k] == 1){numSpecies++;}
      }
    }
  }
  numSpecies = ( this->size * this->size * this->size)-numSpecies;
  */
  
  setPuzzle<<<1,1>>>(d_size, d_puzzle);
  numeralChanger<<<this->blocks,this->threads>>>(d_size, d_puzzle, d_searchSpace);
  cudaCheckError();

}

__host__ void Initializer::printPuzzle(){
  printf("\nSudoku Puzzle Printout\n");
  this->h_puzzle = allocateHost<Puzzle>(1);

  cudaMemcpy(h_puzzle, this->d_puzzle, sizeof(Puzzle), cudaMemcpyDeviceToHost);
  cudaCheckError();

  printPuzzleFile<<<1,1>>>(h_puzzle, this->size);
}



__host__ __device__ unsigned int Initializer::get3DIndex(int x, int y, int z, int dim) {
	return x + dim*(y + dim*z);
}
/***************************************************************************/
/*														Getter Functions														 */
/***************************************************************************/

__host__ __device__ unsigned int Initializer::getSize(){
	return this->d_puzzle[0].getSize();
  }

/***************************************************************************/
/*														Setter Functions														 */
/***************************************************************************/

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



/*template <unsigned int blockSize, int function>
__global__ void getConflictData(Puzzle dPuzzle, int numSpecies){
	extern __shared__ unsigned int sdata[];
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockSize*2) + tid;

	unsigned int offset = numSpecies/2; //get numSpecies

	if(numSpecies % 2 == 1){  
		offset++;
	}

	while(i < offset){
		sdata[tid] += aPuzzle.population[i].conflictSize + aPuzzle.population[i + offset].conflictSize;
		i += blockSize;
	}

	__syncthreads();
	blockReduce<blockSize, function, 512, 256>(sdata, tid);
	blockReduce<blockSize, function, 256, 128>(sdata, tid);
	blockReduce<blockSize, function, 128, 64>(sdata, tid);
	if (tid < 32){
		warpReduce<blockSize, function>(sdata, tid);
	}
	if(tid == 0){
				atomicAdd(&params->totalConflicts, sdata[0]);
				params->avgConflicts = params->totalConflicts / numSpecies + 1;
	}
}

template <unsigned int blockSize, int function, unsigned int blockBoundary, unsigned int tidMax>
__device__ void blockReduce(volatile unsigned int *sdata, unsigned int tid){
	if(blockSize >= blockBoundary){
		if (tid < tidMax){
			conflictOperation<function, tidMax>(sdata, tid);
		}
		__syncthreads();
	}
}

template <unsigned int blockSize, int function>
__device__ void warpReduce(volatile unsigned int *sdata, unsigned int tid){
	if(blockSize >= 64){
		conflictOperation<function, 32>(sdata, tid);
	}
	if (blockSize >= 32){
		conflictOperation<function, 16>(sdata, tid);
	}
	if (blockSize >= 16){
		conflictOperation<function, 8>(sdata, tid);
	}
	if(blockSize >= 8){
		conflictOperation<function, 4>(sdata, tid);
	}
	if (blockSize >= 4){
		conflictOperation<function, 2>(sdata, tid);
	}
	if (blockSize >= 2){
		conflictOperation<function, 1>(sdata, tid);
	}
}

template <int function, unsigned int offset>
__device__ void conflictOperation(volatile unsigned int *sdata, unsigned int tid){
	sdata[tid] += sdata[tid + offset];
}*/
