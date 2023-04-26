#include "cell.cuh"

__host__ __device__ Cell::Cell(unsigned int size){
  this->size = size;
  this->prop = 0;
  this->numerals = allocateDevice<unsigned int>(size); //Species number
}

__host__ __device__ void Cell::setNumeral(unsigned int index, unsigned int set){
	this->numerals[index] = set;
}

__host__ __device__ unsigned int * Cell::getNumeral(){
	return this->numerals;
}

__host__ __device__ void Cell::setProp(unsigned int index, unsigned int set){
	if(set == 0){printf(" ");}
	if(set == 0){
		this->prop +=(set+1)%2;
	}
}

__host__ __device__ unsigned int Cell::getProp(){
	return this->prop;
}

__host__ __device__ unsigned int Cell::getSize(){
  return this->size;
}

/*************************************************************/
/* Helper Functions. Maybe create a misc file to consolidate */
/*************************************************************/

template<class t>
__host__ __device__ t *Cell::allocateHost(unsigned int size){
	t *aValue;
	cudaMallocHost((void **)&aValue, size * sizeof(t));
	cudaCheckError();
	return aValue;
}

template<class t>
__host__ __device__ t *Cell::allocateDevice(unsigned int size){
	t *aValue;
	cudaMalloc((void **)&aValue, size * sizeof(t));
	return aValue;
}

__host__ __device__ void Cell::free_cell(){
	cudaFree(numerals);
}