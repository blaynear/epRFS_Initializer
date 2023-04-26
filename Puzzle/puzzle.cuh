#ifndef PUZZLE_H
#define PUZZLE_H
#include <inttypes.h>
#include "../Cell/cell.cuh"
using namespace std;

class Puzzle{

  unsigned int dim; //Size N of Sudoku puzzle
  unsigned int numSpecies;
  double elapsedTime;
  Cell **grid; //The N x N cell Sudoku cell grid 2d array. MAKE 1D ARRAY

  template<class t>
  __host__ __device__ t *allocateHost(unsigned int);
  template<class t>
  __host__ __device__ t *allocateDevice(unsigned int);
  __host__ __device__ unsigned int getRow(int, int);
  __host__ __device__ unsigned int getCol(int, int);
  __host__ __device__ unsigned int getNum(int, int);
  __host__ __device__ unsigned int getSquare(int, int);
  __host__ __device__ unsigned int get2DIndex(int, int, int);
  __host__ __device__ unsigned int get3DIndex(int, int, int, int);
  //__host__ __device__ void printPopulation(int h_dim, Solution* aSolution); //impliment for printing

public:
  __host__ __device__ unsigned int getSize();
  __host__ __device__ unsigned int getCellSize(unsigned int, unsigned int);
  __host__ __device__ unsigned int getCellProp(unsigned int , unsigned int);
  __host__ __device__ unsigned int *getCellNumeral(unsigned int , unsigned int);
  __host__ __device__ void setCellProp(unsigned int , unsigned int);
  __host__ __device__ void setNumSpecies(int);
  __host__ __device__ unsigned int getNumSpecies();
  __host__ __device__ void free_puzzle();
  __host__ __device__ Puzzle(unsigned int, unsigned int); //Create generator which specifies a unique puzzle of size N
  __device__ void puzzleSet(int, int, unsigned int, unsigned int);
};

#endif
