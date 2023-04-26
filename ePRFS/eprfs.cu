#include "ePRFS.cuh"
cudaError_t cudaStatus;
using namespace std;

const int baseThreadCount = 64;

  /*********************************************/
  /***********Global Kernel Functions**********/
  /*********************************************//*
__global__ void checkSolution(device_Params *params, Puzzle aPuzzle){
	double max_prop = 0.0; // Find the maximum proportion for this square.
	unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x,
		current_Answer = 0;


    if(tid < params->dimSqrd && !aPuzzle.aSolution[tid].fixed){
		for(int i = 0; i < params->numSpecies; i++){
  		if(aPuzzle.population[i].num == tid && aPuzzle.prop[i] > max_prop){
  			current_Answer = getNum(aPuzzle.population[i].index, params->dim) + 1;
  			max_prop = aPuzzle.prop[i];
  		}
  	}
		aPuzzle.aSolution[tid].aValue = current_Answer;

		cudaDeviceSynchronize();
		validateSolution(params, aPuzzle, tid, current_Answer);
    }
}

__global__ void next_Generation(device_Params *params, Puzzle aPuzzle){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < params->numSpecies){
		int beta = 1;
		double prop = aPuzzle.prop[tid],
			shared_Fitness = 1 / aPuzzle.population[tid].niche_count,
			avg_Fitness = params->avg_Fitness;

		//Now calculate new proportions based on just-computed niche counts
		//Update old proportions by replacement with new proportions minProp = 1.0;
		aPuzzle.prop[tid] = max((1.0 - beta) * prop + beta*(prop * shared_Fitness / avg_Fitness), 0.0000000000001);
	}
}

__global__ void getNicheCount(device_Params *params, Puzzle aPuzzle, int index){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	Species aSpecies = aPuzzle.population[index],
	otherSpecies = aPuzzle.population[tid];
  if(tid < params->numSpecies){
		int numConflicts = 0;
		if(otherSpecies.row == aSpecies.row){
			numConflicts++;
		}
		if(otherSpecies.col == aSpecies.col){
			numConflicts++;
		}
		if(otherSpecies.num == aSpecies.num){
			numConflicts++;
		}
		if(otherSpecies.square == aSpecies.square){
			numConflicts++;
		}
		atomicAdd(&aPuzzle.population[index].niche_count, aPuzzle.prop[tid] / 4 * numConflicts);
  }
}

__global__ void cdp_proportionate_Selection(device_Params *params, Puzzle aPuzzle){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if(tid < params->numSpecies){
		getNicheCount<<<params->baseBlockCount, params->baseThreadCount, 0>>>(params, aPuzzle, tid);
		cudaDeviceSynchronize();
		aPuzzle.fitness[tid] = 1 / aPuzzle.population[tid].niche_count * aPuzzle.prop[tid];
		params->avg_Fitness = 0;
	}
}

/*__global__ void mem_proportionate_Selection(device_Params *params, Puzzle aPuzzle){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if(tid < params->numSpecies){
		Species aSpecies = aPuzzle.population[tid],
			otherSpecies;
		int numConflicts;
		for(int i = 0; i < params->numSpecies; i++){
			otherSpecies = aPuzzle.population[i];
			numConflicts = 0;
			if(otherSpecies.row == aSpecies.row){
				numConflicts++;
			}
			if(otherSpecies.col == aSpecies.col){
				numConflicts++;
			}
			if(otherSpecies.num == aSpecies.num){
				numConflicts++;
			}
			if(otherSpecies.square == aSpecies.square){
				numConflicts++;
			}
			aPuzzle.population[tid].niche_count += aPuzzle.prop[i] / 4 * numConflicts;
		}
		aPuzzle.fitness[tid] = 1 / aPuzzle.population[tid].niche_count * aPuzzle.prop[tid];
		params->avg_Fitness = 0;
  }
}*//*

__global__ void mem_proportionate_Selection(device_Params *params, Puzzle aPuzzle){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if(tid < params->numSpecies){
		Species aSpecies = aPuzzle.population[tid],
			otherSpecies;
		int numConflicts;
		for(int i = 0; i < params->numSpecies; i++){
			otherSpecies = aPuzzle.population[i];
			numConflicts = 0;
			if(otherSpecies.row == aSpecies.row){
				numConflicts++;
			}
			if(otherSpecies.col == aSpecies.col){
				numConflicts++;
			}
			if(otherSpecies.num == aSpecies.num){
				numConflicts++;
			}
			if(otherSpecies.square == aSpecies.square){
				numConflicts++;
			}
			aPuzzle.population[tid].niche_count += aPuzzle.prop[i] / 4 * numConflicts;
		}
		aPuzzle.fitness[tid] = 1 / aPuzzle.population[tid].niche_count * aPuzzle.prop[tid];
		params->avg_Fitness = 0;
  }
}

__global__ void am_proportionate_Selection(device_Params *params, Puzzle aPuzzle){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid == 0){
		params->solved = true;
	}
	/*******************************************************/
	/*******Thread count computed w.r.t numSpecies**********/
	/*******Does load balacning on threads**********/
	/*******************************************************/
	/*for(int i = tid; i < params->totalConflicts; i += gridDim.x * blockDim.x){
		Conflict aConflict = aPuzzle.conflictList[i];
		double niche = (aPuzzle.prop[aConflict.cid] / 4.0) * aConflict.weight;
		atomicAdd(&aPuzzle.population[aConflict.sid].niche_count, niche);
	}*/

	/*******************************************************/
	/*******For launching numberOfConlfict threads**********/
	/*******************************************************//*
	if(tid < params->totalConflicts){
		Conflict aConflict = aPuzzle.conflictList[tid];
		double niche = (aPuzzle.prop[aConflict.cid] / 4.0) * aConflict.weight;
		atomicAdd(&aPuzzle.population[aConflict.sid].niche_count, niche);
  }

	cudaDeviceSynchronize();
	if(tid < params->numSpecies){
		aPuzzle.population[tid].niche_count	+= aPuzzle.prop[tid];
		aPuzzle.fitness[tid] = aPuzzle.prop[tid] / aPuzzle.population[tid].niche_count;
		if(tid == 0){
			params->avg_Fitness = 0;
		}
	}
}

  /*********************************************/
  /***********Helper Kernel Functions***********/
  /*********************************************//*

__host__ __device__ int getThreadCount(int offset){
	return min(((offset / 512) + 1)*baseThreadCount, 512);
}

__host__ __device__ int getBlockCount(int offset, int numberOfThreads){
	return (offset / numberOfThreads) + 1;
}

__host__ void ePRFS::savePuzzleFile(Puzzle aPuzzle, ofstream &ptr, int generation){
	int index2D = 0;
	for (int row = 0; row < params->dev_params.dim; row++){ // this->dim
		for (int col = 0; col < params->dev_params.dim; col++){
			index2D = get2DIndex(row, col, params->dev_params.dim);
			if (col == params->dev_params.dim - 1){
				*ptr << aPuzzle.h_aSolution[index2D].aValue << endl;
			}
			else{
				*ptr << aPuzzle.h_aSolution[index2D].aValue << ",";
			}
		}
	}
}

__host__ __device__ void printPopulation(int h_dim, Solution* aSolution){
	int index2D = 0;
	for (int row = 0; row < h_dim; row++){
		for (int column = 0; column < h_dim; column++){
			index2D = get2DIndex(row, column, h_dim);
			if (column == h_dim - 1){
				printf("[%d]\n", aSolution[index2D].aValue);
			}
			else{
				printf("[%d]", aSolution[index2D].aValue);
			}
		}
	}
}

  /*********************************************/
  /***************Main Functions****************/
  /*********************************************/

__host__ __device__ ePRFS::ePRFS(Puzzle *apuzzle, string name){
	Puzzle *d_puzzle = allocateDevice<Puzzle>(1);

	cudaMemcpy(d_puzzle, apuzzle, sizeof(Puzzle), cudaMemcpyHostToDevice);
	cudaCheckError();
  /********Device Params**********/
	this->numSpecies = apuzzle[0].getNumSpecies();      //total number of active species
	this->dim = apuzzle[0].getSize();             //size of puzzle
	this->dimSqrd = pow(this->dim,2);         //optional possibly delete
	this->sqrtDim = sqrt(this->dim);         //square root of size
	this->solved = false;       //used for solution varification
  /********Host Params**********/
	this->puzzleName = name;            //name of puzzle required for database

	this->prop = allocateDevice<double>(this->dimSqrd);
	this->fitness = allocateDevice<double>(this->dimSqrd);
	this->numerals = allocateDevice<unsigned int>(this->numSpecies);
	unsigned int *tempNums = allocateDevice<unsigned int>(this->dim);

	int countOne = 0;
	int countTwo = 0;
	
	while(countOne < this->numSpecies){
		for(int i = 0; i < apuzzle[0].getSize(); i++){
			for(int j = 0; j < apuzzle[0].getSize(); j++){
				if(countTwo < this->dimSqrd){
					this->prop[countTwo] = apuzzle[0].getCellProp(i,j);
					this->fitness[countTwo] = 1/apuzzle[0].getCellProp(i,j);
					countTwo++;
				}
				tempNums = apuzzle[0].getCellNumeral(i, j);
				for(int k = 0; k < this->dim; k++){
					if(tempNums[k] == 1){ 
						this->numerals[countOne] = i + this->dim*(j + this->dim*k);
						countOne++;
					}

				}
			}
		}
	}
	cudaFree(tempNums);

}
/*
void run_Generations(host_Params *h_params, device_Params *d_params, Puzzle aPuzzle){
	int generations = 1000000;

	cudaEventRecord(h_params->start,0);
	cout << "num of generations to do = " << generations << endl;
	for (int gen = 0; gen < generations; gen++){
		if (run_one_generation(h_params, d_params, aPuzzle)){
			cudaEventRecord(h_params->stop,0);
	    cudaEventSynchronize(h_params->stop);
	    cudaEventElapsedTime(&h_params->elapsedTime, h_params->start, h_params->stop);
			h_params->elapsedTime /= 1000;
			gen++;
			cout << "*********************************************************" << endl;
			cout << "***************Sudoku Population Solved!!!***************" << endl;
			cout << "*********************************************************" << endl;
			cout << "Generation: " << gen << endl;
			printf("Completed in %f seconds \n", h_params->elapsedTime);
			cudaDeviceSynchronize();
			cudaCheckError();
			exportResult(h_params, aPuzzle, gen);
			break;
		}
		else if (gen == generations - 1){
			cudaEventRecord(h_params->stop,0);
	    cudaEventSynchronize(h_params->stop);
	    cudaEventElapsedTime(&h_params->elapsedTime, h_params->start, h_params->stop);
			h_params->elapsedTime /= 1000;
			gen++;
			cout << "*********************************************************" << endl;
			cout << "***************Sudoku Population Failed!!!***************" << endl;
			cout << "*********************************************************" << endl;
			cout << "Generation: " << gen << endl;
			printf("Completed in %f seconds \n", h_params->elapsedTime);
			cudaDeviceSynchronize();
			cudaCheckError();
			exportResult(h_params, aPuzzle, gen);
			break;
		}
	}
	cudaFree(d_params);
	cudaCheckError();
	cudaFreeHost(h_params);
	cudaCheckError();
	cudaFreeHost(aPuzzle.h_population); //DELETE
	cudaCheckError();
	cudaFreeHost(aPuzzle.h_aSolution); //dimSqrd
	cudaCheckError();
	cudaFree(aPuzzle.population); //number of Spieces
	cudaCheckError();
  cudaFree(aPuzzle.prop); //number of Spieces
	cudaCheckError();
  cudaFree(aPuzzle.fitness); //number of Spieces
	cudaCheckError();
  cudaFree(aPuzzle.clueList); //number of Clues
	cudaCheckError();
  cudaFree(aPuzzle.aSolution); //dimSqrd
	cudaCheckError();
	//Allocate device memory
}

bool run_one_generation(host_Params *h_params, device_Params *d_params, Puzzle aPuzzle){
    /******************************************************************************/
    /***************************CDP proportionate_Selection************************/
    /******************************************************************************/
      //cdp_proportionate_Selection<<<h_params->dev_params.baseBlockCount, h_params->dev_params.baseThreadCount, h_params->dev_params.baseThreadCount * sizeof(double)>>>(d_params, aPuzzle); //CUDA Dynamic Parallelism optimized
  
      /******************************************************************************/
      /*******************Memory Only proportionate_Selection************************/
      /******************************************************************************/
      //mem_proportionate_Selection<<<h_params->dev_params.baseBlockCount, h_params->dev_params.baseThreadCount>>>(d_params, aPuzzle); //memory optimized
  
      /******************************************************************************/
    /***************************CDP proportionate_Selection************************/
    /******************************************************************************//*
      unsigned int numberOfThreads = getThreadCount(h_params->dev_params.totalConflicts),
          numberOfBlocks = getBlockCount(h_params->dev_params.totalConflicts, numberOfThreads);
  
      am_proportionate_Selection<<<numberOfBlocks, numberOfThreads>>>(d_params, aPuzzle); //CUDA Dynamic Parallelism optimized
      cudaCheckError();
  
      cublasDdot(h_params->cublasHandle, h_params->dev_params.numSpecies, aPuzzle.prop, 1, aPuzzle.fitness, 1, &d_params->avg_Fitness);
      cudaCheckError();
  
      /******************************************************************************/
    /*******************************Next Generation********************************/
    /******************************************************************************//*
      next_Generation<<<h_params->dev_params.baseBlockCount, h_params->dev_params.baseThreadCount>>>(d_params, aPuzzle);
      cudaCheckError();
  
      checkSolution<<<getBlockCount(h_params->dev_params.dimSqrd, h_params->dev_params.baseThreadCount), h_params->dev_params.baseThreadCount>>>(d_params, aPuzzle);
      cudaCheckError();
  
      cudaMemcpy(&h_params->dev_params, d_params, sizeof(device_Params), cudaMemcpyDeviceToHost);
      cudaCheckError();
      return h_params->dev_params.solved;
}

__device__ void checkValue(device_Params *params, Puzzle aPuzzle, unsigned int row, unsigned int col, unsigned int cellIndex, unsigned int aNumeral){
	unsigned int index2D = get2DIndex(row, col, params->dim);
	if(params->solved && index2D != cellIndex && aNumeral == aPuzzle.aSolution[index2D].aValue){
		params->solved = false;
	}
}

__device__ void validateSolution(device_Params *params, Puzzle aPuzzle, unsigned int cellIndex, unsigned int current_Answer){
	unsigned int row = cellIndex % params->dim,
		col = (cellIndex / params->dim) % params->dim,
		my_region_row = row / params->sqrtDim, rowCord,   // Region row 0, 1, or 2
		my_region_col = col / params->sqrtDim, colCord;   // Region col 0, 1, or 2

	if(cellIndex < params->dimSqrd){
		unsigned int region;
		for(int colOffset = 0; colOffset < params->sqrtDim && params->solved; colOffset++){   // Go down the cols
			for(int rowOffset = 0; rowOffset < params->sqrtDim && params->solved; rowOffset++){  // Go down the rows
        rowCord = my_region_row * params->sqrtDim + rowOffset;
        colCord = my_region_col * params->sqrtDim + colOffset;
				region = rowOffset + colOffset * params->sqrtDim;
				checkValue(params, aPuzzle, region, col, cellIndex, current_Answer);
				checkValue(params, aPuzzle, row, region, cellIndex, current_Answer);
	      checkValue(params, aPuzzle, rowCord, colCord, cellIndex, current_Answer);
			}
    }
	}
}

//__global__ void setSpecies(device_Params *params, Puzzle aPuzzle, double *searchSpace){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if(tid < params->numSpecies){
		int region[4];

		Species aSpecies;
    aPuzzle.prop[tid] = 1.0/(double)params->numSpecies;
    aPuzzle.fitness[tid] = 0;
    aSpecies.niche_count = 0;
    aSpecies.index = (int)searchSpace[tid];
		getRegion(region, aSpecies.index, params->dim);
		aSpecies.row = region[0];
    aSpecies.col = region[1];
    aSpecies.num = region[2];
		aSpecies.square = region[3];
		aPuzzle.population[tid] = aSpecies;
	}
	/******************************************************************************/
	/*****************************Fix Clues****************************************/
	/******************************************************************************//*

	if(tid < params->numberOfClues){
		int loc[4],
			anIndex = aPuzzle.clueList[tid];
		getLocation(loc, anIndex, params->dim);
		int index2D = get2DIndex(loc[0], loc[1], params->dim);
		aPuzzle.aSolution[index2D].aValue = loc[2] + 1;
		aPuzzle.aSolution[index2D].fixed = true;
	}
//}

/*Puzzle *setPuzzle(host_Params *h_params, device_Params *d_params, Puzzle *aPuzzle, int *h_listOfClues, int numberOfClues){
	//allocate memory for searchSpace
	double *d_SearchSpace = getDouble(h_params->dev_params.dimCube, 'D');
	double *h_SearchSpace = getDouble(h_params->dev_params.dimCube, 'H');
  aPuzzle->clueList = getListOfClues(h_listOfClues, numberOfClues);

	/******************************************************************************/
  /**************preprocess search space prior to running algorithm**************/
  /******************************************************************************//*

	//refactor so numberOfClue threads launched.
	//searchSpace = 1; itterate with numberOfClue blocks
	exterminate<<<h_params->dev_params.baseBlockCount, h_params->dev_params.baseThreadCount>>>(d_params, aPuzzle->clueList, d_SearchSpace);
	cudaCheckError();
	/******************************************************************************/
  /*************************Count number of remaining species********************/
  /******************************************************************************//*
	//addReduce(h_params, d_params, d_SearchSpace, &d_params->dimCube);
	cublasDdot(h_params->cublasHandle, h_params->dev_params.dimCube, d_SearchSpace, 1, d_SearchSpace, 1, &d_params->avg_Fitness);
	cudaCheckError();
	cudaMemcpy(&h_params->dev_params, d_params, sizeof(device_Params), cudaMemcpyDeviceToHost);
	cudaCheckError();

	h_params->dev_params.numSpecies = h_params->dev_params.dimCube - h_params->dev_params.avg_Fitness;
	h_params->dev_params.avg_Fitness = 0;

  /******************************************************************************/
  /*************************Allocate active species array************************/
  /******************************************************************************//*
  cudaMemcpy(h_SearchSpace, d_SearchSpace, h_params->dev_params.dimCube*sizeof(double), cudaMemcpyDeviceToHost);
	cudaCheckError();
	int newIndex = 0;
	for(int i = 0; i < h_params->dev_params.dimCube; i++){
		if(h_SearchSpace[i] == 0){
			h_SearchSpace[newIndex] = i;
			newIndex++;
		}
	}

	cudaMemcpy(d_SearchSpace, h_SearchSpace, h_params->dev_params.dimCube*sizeof(int), cudaMemcpyHostToDevice);
	cudaCheckError();

	cudaDeviceSetLimit(cudaLimitMallocHeapSize, h_params->dev_params.numSpecies * h_params->dev_params.dim * sizeof(bool));
	size_t limit = 0;
	cudaDeviceGetLimit(&limit, cudaLimitMallocHeapSize);
	printf("cudaLimitMallocHeapSize: %u\n", (unsigned)limit);
	cudaCheckError();
	/******************************************************************************/
  /*********************************Set up Puzzle********************************/
  /******************************************************************************//*
	aPuzzle->population = getSpecies(h_params->dev_params.numSpecies, 'D');
	aPuzzle->h_population = getSpecies(h_params->dev_params.numSpecies, 'H'); //Delete
  aPuzzle->prop = getDouble(h_params->dev_params.numSpecies, 'D');
  aPuzzle->fitness = getDouble(h_params->dev_params.numSpecies, 'D');
  aPuzzle->aSolution = getSolution(h_params->dev_params.dimSqrd, 'D');
  aPuzzle->h_aSolution = getSolution(h_params->dev_params.dimSqrd, 'H');

	/******************************************************************************/
  /*********************Update Thread and Block Counts***************************/
  /******************************************************************************//*
  h_params->dev_params.baseThreadCount = getThreadCount(h_params->dev_params.numSpecies);
	h_params->dev_params.baseBlockCount = getBlockCount(h_params->dev_params.numSpecies, h_params->dev_params.baseThreadCount);
	cudaMemcpy(d_params, &h_params->dev_params, sizeof(device_Params), cudaMemcpyHostToDevice);
	cudaCheckError();

	/******************************************************************************/
	/*********************Set up Temp Conflict List********************************/
	/******************************************************************************//*
	setSpecies<<<h_params->dev_params.baseBlockCount, h_params->dev_params.baseThreadCount>>>(d_params, *aPuzzle, d_SearchSpace);
	cudaCheckError();
	countConflicts<<<h_params->dev_params.baseBlockCount, h_params->dev_params.baseThreadCount>>>(d_params, *aPuzzle, d_SearchSpace);
	cudaCheckError();

	/******************************************************************************/
	/*********************Get data on Conflicts************************************/
	/******************************************************************************//*
	computeConflictData(h_params, d_params, aPuzzle);

	//DELETE: Used for testing
	cudaMemcpy(aPuzzle->h_population, aPuzzle->population, h_params->dev_params.numSpecies * sizeof(Species), cudaMemcpyDeviceToHost);
	cudaCheckError();
	//************************
	cudaMemcpy(&h_params->dev_params, d_params, sizeof(device_Params), cudaMemcpyDeviceToHost);
	cudaCheckError();

	aPuzzle->conflictList = getConflicts(h_params->dev_params.totalConflicts, 'D');
	aPuzzle->h_conflictList = getConflicts(h_params->dev_params.totalConflicts, 'H');

	/*************************Host side test**************************************************************//*
	unsigned int maxi = 0, sum = 0, mini = INT_MAX;
	for(int i = 0; i < h_params->dev_params.numSpecies; i++){
		maxi = max(maxi, aPuzzle->h_population[i].conflictSize);
		mini = min(mini, aPuzzle->h_population[i].conflictSize);
		sum += aPuzzle->h_population[i].conflictSize;
	}
	/*************************Host side test**************************************************************//*
	printf("********************************************************************************************\n");
	printf("#Species %d: remaining %lf | Prop %lf | ", h_params->dev_params.numSpecies, (double)h_params->dev_params.numSpecies/(double)h_params->dev_params.dimCube, 1.0/h_params->dev_params.numSpecies);
	printf("\nSum %d | Max Conf %d | AVG Conf %d | MIN Conf %d\n", h_params->dev_params.totalConflicts, h_params->dev_params.maxConflicts, h_params->dev_params.avgConflicts, h_params->dev_params.minConflicts);
	/*************************Host side test**************************************************************//*
	printf("Sum %d | Max Conf %d | AVG Conf %d | MIN Conf %d\n", sum, maxi, sum/h_params->dev_params.numSpecies + 1, mini);
	/*************************Host side test**************************************************************//*
	printf("********************************************************************************************\n");

	setConflicts<<<h_params->dev_params.baseBlockCount, h_params->dev_params.baseThreadCount, 2 * h_params->dev_params.baseThreadCount * sizeof(int)>>>(d_params, *aPuzzle, d_SearchSpace);
	cudaCheckError();

	cudaFreeHost(h_SearchSpace);
	cudaFree(d_SearchSpace);
	cudaCheckError();
	return aPuzzle;
//}

//void initializeRUN(string aPuzzleName, int dim, int numberOfClues, int* h_listOfClues){
  Puzzle *aPuzzle;
	host_Params *h_params;
	device_Params *d_params;

	cudaMalloc((void **)&d_params, sizeof(device_Params)); //allocate memory for Params
	cudaCheckError();
	h_params = getParameters(dim, d_params, aPuzzleName, numberOfClues); //initialize parameters

  cudaMallocHost((void **)&aPuzzle, sizeof(Puzzle));
	cudaCheckError();
  cudaEventRecord(h_params->start,0); //record start time

	setPuzzle(h_params, d_params, aPuzzle, h_listOfClues, numberOfClues); //remove aPuzzle. Use is for refactoring

  cudaDeviceSynchronize();
	cudaEventRecord(h_params->stop,0); //record stop time
	cudaEventSynchronize(h_params->stop);
	cudaEventElapsedTime(&h_params->setUpTime,h_params->start,h_params->stop);
	printf("Search Space initialization took: %f seconds \n", h_params->setUpTime/1000);
	cudaCheckError();
	run_Generations(h_params, d_params, *aPuzzle);
//}

/***************************************************************************/
/*					CUDA Helper Masks									   */
/***************************************************************************/

template<class t>
t *ePRFS::allocateDevice(unsigned int size){
	t *aValue;
	cudaMalloc((void **)&aValue, size * sizeof(t));
	cudaCheckError();
	return aValue;
}

template<class t>
t *ePRFS::allocateHost(unsigned int size){
	t *aValue;
	cudaMallocHost((void **)&aValue, size * sizeof(t));
	cudaCheckError();
	return aValue;
}