#include "cell.cuh"

__host__ __device__ Cell::Cell(unsigned int size){
  this->size = size;
  this->numerals = allocateDevice<unsigned int>(size); //Species number
  //this->numeralProp = allocateDevice<unsigned int>(size); //Proportion array for species
  //this->numeralFit = allocateDevice<unsigned int>(size);
}

__host__ __device__ void Cell::setNumeral(unsigned int index, unsigned int set){
	this->numerals[index] = set;
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
	//cudaFree(numeralProp);
	//cudaFree(numeralFit);
}