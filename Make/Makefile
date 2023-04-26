# gnu makefile for CUDA compiles
# using as simple pattern rule for .cu files
#NVCC= /opt/nvidia/cuda-6.5/bin/nvcc
#-gencode=arch=compute_75,code=sm_75 -rdc=true required for CDP
#
NVCC = nvcc -g -m64 -lineinfo -gencode=arch=compute_86,code=sm_86 -rdc=true -Xptxas -O3
LIBS= -lcublas
MAIN = main
CELL = cell
PUZZLE = puzzle
INITIALIZER = initializer
SUDOKU = SudokuGen
ePRFS = ePRFS
SQLITE = -IC:\Users\H-Bro\Desktop\CUDA\SudokuGenerator-main\SudokuGenerator-main\Main\sqlite3 -lsqlite3

# the build target executable:
$(SUDOKU): $(MAIN).obj $(CELL).obj $(PUZZLE).obj $(INITIALIZER).obj ePRFS.obj
	$(NVCC) $(LIBS) -o $(SUDOKU) $(CELL).obj $(PUZZLE).obj $(MAIN).obj $(INITIALIZER).obj ePRFS.obj $(SQLITE)

$(MAIN).obj: ../Main/$(MAIN).cu 
	$(NVCC) -c $(FLAGS) ../Main/$(MAIN).cu $(SQLITE)

$(CELL).obj: ../Cell/$(CELL).cu
	$(NVCC) $(LIBS) -c ../Cell/$(CELL).cu

$(PUZZLE).obj: ../Puzzle/$(PUZZLE).cu
	$(NVCC) $(LIBS) -c ../Puzzle/$(PUZZLE).cu

$(INITIALIZER).obj: ../Initializer/$(INITIALIZER).cu
	$(NVCC) $(LIBS) -c ../Initializer/$(INITIALIZER).cu
	
ePRFS.obj: ../ePRFS/eprfs.cu
	$(NVCC) $(LIBS) -c ../ePRFS/eprfs.cu	

clean:
	rm *.obj *.exe *.exp SudokuGen.lib *.pdb

ARGS = "default"

.PHONY: run
run:
	./$(SUDOKU)
