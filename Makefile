CXX  := g++
NVCC := nvcc

TARGET := build/heat2d


# # Intel oneMKL
# ifndef MKLROOT
# $(error MKLROOT not set)
# endif

# ifneq ($(wildcard $(MKLROOT)/lib/libmkl_rt.so*),)
# MKL_LIBDIR := $(MKLROOT)/lib
# else
# MKL_LIBDIR := $(MKLROOT)/lib/intel64
# endif


CPP_SOURCES  := $(wildcard src/*.cpp)
CUDA_SOURCES := $(wildcard src/*.cu)

CPP_OBJECTS := \
	$(patsubst src/%.cpp,build/%.o,$(CPP_SOURCES))

CUDA_OBJECTS := \
	$(patsubst src/%.cu,build/%.o,$(CUDA_SOURCES))

OBJECTS := $(CPP_OBJECTS) $(CUDA_OBJECTS)
DEPENDENCIES := $(OBJECTS:.o=.d)

CPPFLAGS := \
	-Iinclude \
	-Ithird_party \
	# -I$(MKLROOT)/include

CXXFLAGS := \
	-O3 \
	-std=c++17 \
	-Wall \
	-Wextra \
	-MMD \
	-MP

# NVIDIA L40S uses the Ada architecture.
ARCH ?= sm_89

NVCCFLAGS := \
	-O3 \
	-std=c++17 \
	-arch=$(ARCH) \
	-lineinfo \
	-MMD \
	-MP \
	-Xcompiler=-Wall,-Wextra


# LDFLAGS := \
# 	-L$(MKL_LIBDIR)

# LDLIBS := \
# 	-lmkl_rt \
# 	-lpthread \
# 	-lm \
# 	-ldl


.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJECTS)
	@mkdir -p $(@D)
	$(NVCC) $(NVCCFLAGS) $^ $(LDFLAGS) $(LDLIBS) -o $@

build/%.o: src/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

build/%.o: src/%.cu
	@mkdir -p $(@D)
	$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -rf build

-include $(DEPENDENCIES)