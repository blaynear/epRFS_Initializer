#ifndef INITIALIZER_H
#define INITIALIZER_H
#include <inttypes.h>
#include <fstream>
#include "../Puzzle/puzzle.cuh"
using namespace std;

class Initializer{

  //Puzzle *h_puzzle; //For testing
  Puzzle *d_puzzle; //aPuzzle allocated
  Puzzle *h_puzzle; //aPuzzle allocated

  unsigned int size;
  unsigned int blocks; //Number of blocks to launch also grid size?
  unsigned int threads; //Number of threads per block
  
  template<class t>
  t *allocateDevice(unsigned int size);
  template<class t>
  t *allocateHost(unsigned int size);

public:
  __host__ Initializer(unsigned int size);
  __host__ void run(fstream &file);
  __host__ void printPuzzle();
  __host__ __device__ unsigned int get3DIndex(int x, int y, int z, int dim);
  __host__ __device__ unsigned int getSize();
  
};

#endif