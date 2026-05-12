# bvhproject — LBVH build & test
# Prerequisites: ml course/cme213/nvhpc/24.1 (sets PATH to nvcc + MPI)
# GPU target: Quadro RTX 6000 (sm_75 / Turing)

NVCC      := nvcc
SM        := 75
FLAGS     := -O2 -arch=sm_$(SM) -std=c++17
INC       := -I cuda/include -I cuda/src

SRC       := cuda/src
TESTS     := cuda_tests
BUILD     := build

# Shared kernel sources compiled as relocatable device code
KERNEL_SRCS := $(SRC)/morton.cu $(SRC)/karras.cu $(SRC)/refit.cu
KERNEL_OBJS := $(patsubst $(SRC)/%.cu,$(BUILD)/%.o,$(KERNEL_SRCS))

.PHONY: all tests run_tests clean

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

clean:
	rm -rf $(BUILD) lbvh_build
