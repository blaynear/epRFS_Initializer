#include "puzzle.cuh"

const int baseThreadCount = 64;

/***************************************************************************/
/*                           Cuda Kernels   							   */
/***************************************************************************/

/***************************************************************************/
/*   						Constructors                                   */
/***************************************************************************/
__device__ void Puzzle::puzzleSet(int x, int y, unsigned int index, unsigned int set){
	this->grid[x][y].setNumeral(index, set);
	this->grid[x][y].setProp(index, set);
}

__host__ __device__ Puzzle::Puzzle(unsigned int size, unsigned int species){
	this->dim = size;
	this->grid = allocateDevice<Cell *>(size);
	for(int i = 0; i < size; i++){
		this->grid[i] = allocateDevice<Cell>(size);
	}
	for(int i = 0; i < size; i++){
		for(int j = 0; j < size; j++){
			this->grid[i][j] = Cell(size);
		}
	}
	this->numSpecies = species;
}

/***************************************************************************/
/*						Getter Functions   								   */
/***************************************************************************/
__host__ __device__ unsigned int Puzzle::getSize(){
  return this->dim;
}

__host__ __device__ unsigned int Puzzle::getCellSize(unsigned int x, unsigned int y){
	return this->grid[x][y].getSize();
}

__host__ __device__ unsigned int * Puzzle::getCellNumeral(unsigned int x, unsigned int y){
	return this->grid[x][y].getNumeral();
}

__host__ __device__ unsigned int Puzzle::getCellProp(unsigned int x, unsigned int y){
	return this->grid[x][y].getProp();
}

__host__ __device__ void Puzzle::setNumSpecies(int x){
	this->numSpecies = x;
}

__host__ __device__ unsigned int Puzzle::getNumSpecies(){
	return this->numSpecies;
}


__host__ __device__ unsigned int Puzzle::getRow(int index, int dim){
	return index % dim;
}

__host__ __device__ unsigned int Puzzle::getCol(int index, int dim){
	return (index / dim) % dim;
}

__host__ __device__ unsigned int Puzzle::getNum(int index, int dim){
	return index / (dim*dim);
}

__host__ __device__ unsigned int Puzzle::getSquare(int index, int dim){
	return get2DIndex(getRow(index, dim) / sqrtf((double)dim), getCol(index, dim) / sqrtf((double)dim), sqrtf((double)dim));
}

__host__ __device__ unsigned int Puzzle::get3DIndex(int x, int y, int z, int dim) {
	return x + dim*(y + dim*z);
}

__host__ __device__ unsigned int Puzzle::get2DIndex(int x, int y, int dim) {
	return x + dim*y;
}


/***************************************************************************/
/*					 CUDA Helper Masks					                   */
/***************************************************************************/
template<class t>
__host__ __device__ t *Puzzle::allocateHost(unsigned int size){
	t *aValue;
	cudaMallocHost((void **)&aValue, size * sizeof(t));
	cudaCheckError();
	return aValue;
}

template<class t>
__host__ __device__ t *Puzzle::allocateDevice(unsigned int size){
	t *aValue;
	cudaMalloc((void **)&aValue, size * sizeof(t));
	cudaCheckError();
	return aValue;
}
__host__ __device__ void Puzzle::free_puzzle(){
	//grid.free_cell();
	//cudaFree(grid);
}