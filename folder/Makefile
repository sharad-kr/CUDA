# CUDA compiler
# NVCC := /usr/bin/nvcc

# Compiler flags
NVCCFLAGS := -std=c++11

# Target executable
TARGET := LeNet_Assignment2

# CUDA source files
CU_FILES := LeNet_Assignment2.cu

# Object files
OBJ_FILES := $(CU_FILES:.cu=.o)

# Default rule to build the target executable
all: $(TARGET)

# Rule to build the target executable
$(TARGET): $(OBJ_FILES)
        nvcc $(NVCCFLAGS) $^ -o $@

# Rule to compile CUDA source files
%.o: %.cu
        nvcc $(NVCCFLAGS) -c $< -o $@

# Clean rule
clean:
        rm -f $(OBJ_FILES) $(TARGET)
