#ifndef ePRFS_H
#define ePRFS_H
#include <stdio.h>
#include <string>
#include <fstream>
#include "stdlib.h"
#include <algorithm>
#include <limits>
#include "../Puzzle/puzzle.cuh"
using namespace std;

class ePRFS{
/********Cells->Species**********/
  double *prop;                 //proportion for each idiviual numeral array same size as numeral array
  double *fitness;              //value = 1/prop
  double cellProp;              //total prop for cell
  unsigned int *numerals;       //= Cell numerals || change to UN64x
  unsigned int numCount;        //hold number of numerals in cell possible place holder || change to UN64x
  unsigned int offset;          //summation of numerals to cell || change to UN64x
  unsigned int solution;        //value that is belived to be solution to puzzle -- includes clue list
/********Device Params**********/
  double avg_Fitness;           //used in cublas dot product for computing global fitness
  unsigned int dimSqrd;         //optional possibly delete
  unsigned int numSpecies;      //total number of active species
  unsigned int numberOfClues;   //number of clues possibly delete
  unsigned int baseThreadCount; //theard count defined by active species
  unsigned int baseBlockCount;  //block count defined by active species
  unsigned int dim;             //size of puzzle
  unsigned int sqrtDim;         //square root of size
  bool solved;                  //used for solution varification
/********Host Params**********/
  cublasHandle_t cublasHandle;  //used in cublas dot product
  cudaEvent_t start;            //start for elapsedTime for device POSSIBLE DELETE
  cudaEvent_t stop;             //stop for elapsedTime for device POSSIBLE DELETE
  float runTime;                //total execution time for host and device POSSIBLE DELETE
  string puzzleName;            //name of puzzle required for database
  
  template<class t>
  t *allocateDevice(unsigned int);
  template<class t>
  t *allocateHost(unsigned int);

  /*
  //add reduction for puzzle class and cell class
  //puzzle class addReduction reduces prop in cells n elements
  //cell class addReduction reduces prop in numeral array numCount elements
  /*********************************************/
  /************Puzzle Class Functions***********/
  /*********************************************/
  /*
  __device__ addReduction       //performs add reduction on cells with respect to regions
  __device__ warpAddReduction   //performs add reduction on warp level
  __device__ warpMaxReduction   //performs max reduction on warp level
  __device__ blockAddReduction  //performs add reduction on block level
  __device__ blockMaxReduction  //performs max reduction on block level

  __host__ __device__ void printSolution() //prints total population of puzzle debugging function

  /*********************************************/
  /*************Cell Class Functions************/
  /*********************************************/
  /*
  __device__ addReduction       //performs add reduction on cells with respect to regions
  __device__ maxReduction       //performs max reduction on numeral prop array #cell class funcion
  __device__ warpAddReduction   //performs add reduction on warp level
  __device__ warpMaxReduction   //performs max reduction on warp level
  __device__ blockAddReduction  //performs add reduction on block level
  __device__ blockMaxReduction  //performs max reduction on block level
  */

  /*********************************************/
  /*************ePRFS Class Functions***********/
  /*********************************************/

  //__host__ __device__ int getThreadCount(int offset);
  //__host__ __device__ int getBlockCount(int offset, int numberOfThreads);
  //__host__ void savePuzzleFile(host_Params *params, Puzzle aPuzzle, ofstream *ptr, int generation); //pushes puzzle to database
  //__host__ __device__ void printPopulation(int h_dim, Solution* aSolution); //prints puzzle used for debugging
  //__device__ void checkValue(device_Params *params, Puzzle aPuzzle, unsigned int row, unsigned int col, unsigned int cellIndex, unsigned int aNumeral);
  //__device__ void validateSolution(device_Params *params, Puzzle aPuzzle, unsigned int cellIndex, unsigned int current_Answer);

  //bool run_one_generation(host_Params *h_params, device_Params *d_params, Puzzle aPuzzle);
  //void run_Generations(host_Params *h_params, device_Params *d_params, Puzzle aPuzzle);

public:
  __host__ __device__ ePRFS(Puzzle * apuzzle, string name);


  
};
#endif