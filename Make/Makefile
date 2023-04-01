# gnu makefile for CUDA compiles
# using as simple pattern rule for .cu files
#NVCC= /opt/nvidia/cuda-6.5/bin/nvcc
#-gencode=arch=compute_75,code=sm_75 -rdc=true required for CDP
NVCC = nvcc -g -m64 -lineinfo -gencode=arch=compute_86,code=sm_86 -rdc=true -Xptxas -O3
LIBS= -lcublas
MAIN = main
CELL = cell
PUZZLE = puzzle
INITIALIZER = initializer
SUDOKU = SudokuGen

# the build target executable:
$(SUDOKU): $(MAIN).obj $(CELL).obj $(PUZZLE).obj $(INITIALIZER).obj
	$(NVCC) $(LIBS) -o $(SUDOKU) $(CELL).obj $(PUZZLE).obj $(MAIN).obj $(INITIALIZER).obj

$(MAIN).obj: ../Main/$(MAIN).cu
	$(NVCC) -c $(FLAGS) ../Main/$(MAIN).cu

$(CELL).obj: ../Cell/$(CELL).cu
	$(NVCC) $(LIBS) -c ../Cell/$(CELL).cu

$(PUZZLE).obj: ../Puzzle/$(PUZZLE).cu
	$(NVCC) $(LIBS) -c ../Puzzle/$(PUZZLE).cu

$(INITIALIZER).obj: ../Initializer/$(INITIALIZER).cu
	$(NVCC) $(LIBS) -c ../Initializer/$(INITIALIZER).cu

clean:
	rm *.obj *.exe *.exp *.lib *.pdb

ARGS = "default"

.PHONY: run
run:
	./$(SUDOKU)
