#include "cell.cuh"

__host__ __device__ Cell::Cell(unsigned int size){
  this->size = size;
  this->numerals = allocateDevice<unsigned int>(size); //Species number
  //this->numeralProp = allocateDevice<unsigned int>(size); //Proportion array for species
  //this->numeralFit = allocateDevice<unsigned int>(size);
  for(int i = 0; i < size; i++){
	this->numerals[i] = i+1;
  }
}

__host__ __device__ void Cell::setNumeral(unsigned int index, unsigned int set){// int index, int set
	
	if(set == 1){
		this->numerals[index] = 0;	
	}
}

__host__ __device__ unsigned int Cell::getNumeral(){
	unsigned int h_prop = 0;
	
	for(int i = 0; i < this->size; i++){
		if(this->numerals[i] != 0){
			h_prop = h_prop + numerals[i];
		}
	}
	return h_prop;
}

__host__ __device__ unsigned int Cell::getSize(){
	int count = 0;
	for(int i = 0; i < this->size; i++){
		if(this->numerals[i] != 0){
			count++;
		}
	}
  return count;
}

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
/*************************************************/
/*Helper Functions. Maybe create a misc file to consolidate */
/*************************************************/

__host__ __device__ void Cell::free_cell(){
	cudaFree(numerals);
	//cudaFree(numeralProp);
	//cudaFree(numeralFit);
}