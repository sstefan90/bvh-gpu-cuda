# bvhproject — LBVH build & test
# Prerequisites: ml course/cme213/nvhpc/24.1 (sets PATH to nvcc + MPI)
# GPU target: Quadro RTX 6000 (sm_75 / Turing)

NVCC      := nvcc
SM        := 75
FLAGS     := -O2 -arch=sm_$(SM) -std=c++17
INC       := -I cuda/include -I cuda/src

MPI_CFLAGS := $(shell mpicxx --showme:compile 2>/dev/null)
MPI_LFLAGS := $(shell mpicxx --showme:link   2>/dev/null)

SRC       := cuda/src
TESTS     := cuda_tests
BUILD     := build

# Shared kernel sources compiled as relocatable device code
KERNEL_SRCS := $(SRC)/morton.cu $(SRC)/karras.cu $(SRC)/refit.cu
KERNEL_OBJS := $(patsubst $(SRC)/%.cu,$(BUILD)/%.o,$(KERNEL_SRCS))

.PHONY: all tests run_tests mpi mpi_tests clean

all: lbvh_build

$(BUILD):
	mkdir -p $(BUILD)

# Compile each kernel to a relocatable device-code object (-dc)
$(BUILD)/%.o: $(SRC)/%.cu | $(BUILD)
	$(NVCC) $(FLAGS) $(INC) -dc $< -o $@

# Main pipeline binary: reads triangles.bin → writes lbvh.bin
lbvh_build: $(KERNEL_OBJS) $(SRC)/lbvh_main.cu | $(BUILD)
	$(NVCC) $(FLAGS) $(INC) $(KERNEL_OBJS) $(SRC)/lbvh_main.cu -o $@

# Unit test executables
$(BUILD)/test_morton: $(BUILD)/morton.o $(TESTS)/test_morton.cu | $(BUILD)
	$(NVCC) $(FLAGS) $(INC) $^ -o $@

$(BUILD)/test_karras: $(BUILD)/morton.o $(BUILD)/karras.o $(TESTS)/test_karras.cu | $(BUILD)
	$(NVCC) $(FLAGS) $(INC) $^ -o $@

tests: $(BUILD)/test_morton $(BUILD)/test_karras

run_tests: tests
	@echo "=== test_morton ===" && ./$(BUILD)/test_morton
	@echo "=== test_karras ===" && ./$(BUILD)/test_karras

# Detect CUDA lib path for host-link step
CUDA_LIBDIR  := $(shell find /home/cme213/software/nvidia-hpc-sdk -name "libcudart.so" 2>/dev/null | head -1 | xargs dirname)
CUDA_STUBDIR := $(CUDA_LIBDIR)/stubs

# MPI multi-GPU build
$(BUILD)/samplesort.o: $(SRC)/samplesort.cu | $(BUILD)
	$(NVCC) $(FLAGS) $(INC) -Xcompiler "$(MPI_CFLAGS)" -dc $< -o $@

$(BUILD)/lbvh_mpi_main.o: $(SRC)/lbvh_mpi.cu | $(BUILD)
	$(NVCC) $(FLAGS) $(INC) -Xcompiler "$(MPI_CFLAGS)" -dc $< -o $@

# Step 1: device link (generates the fat binary glue)
$(BUILD)/lbvh_mpi_dlink.o: $(KERNEL_OBJS) $(BUILD)/samplesort.o $(BUILD)/lbvh_mpi_main.o | $(BUILD)
	$(NVCC) $(FLAGS) -dlink $^ -o $@

# Step 2: host link via mpicxx (handles -pthread and -Wl,... correctly)
lbvh_mpi: $(KERNEL_OBJS) $(BUILD)/samplesort.o $(BUILD)/lbvh_mpi_main.o $(BUILD)/lbvh_mpi_dlink.o | $(BUILD)
	mpicxx $^ -L$(CUDA_LIBDIR) -L$(CUDA_STUBDIR) -lcudart -lcuda -o $@

mpi: lbvh_mpi

# MPI unit tests
$(BUILD)/test_samplesort_main.o: $(TESTS)/test_samplesort.cu | $(BUILD)
	$(NVCC) $(FLAGS) $(INC) -Xcompiler "$(MPI_CFLAGS)" -dc $< -o $@

$(BUILD)/test_samplesort_dlink.o: $(BUILD)/samplesort.o $(BUILD)/test_samplesort_main.o | $(BUILD)
	$(NVCC) $(FLAGS) -dlink $^ -o $@

$(BUILD)/test_samplesort: $(BUILD)/samplesort.o $(BUILD)/test_samplesort_main.o $(BUILD)/test_samplesort_dlink.o | $(BUILD)
	mpicxx $^ -L$(CUDA_LIBDIR) -L$(CUDA_STUBDIR) -lcudart -lcuda -o $@

mpi_tests: $(BUILD)/test_samplesort

clean:
	rm -rf $(BUILD) lbvh_build lbvh_mpi
