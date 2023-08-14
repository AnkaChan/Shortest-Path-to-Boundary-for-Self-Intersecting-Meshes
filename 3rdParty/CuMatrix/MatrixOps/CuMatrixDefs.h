#pragma once
#include "cuda_runtime.h"
#include <iostream>

#define GPU_CPU_INLINE_FUNC  __inline__ __device__ __host__
#define GPU_CPU_FUNC_NO_INLINE  __device__ __host__

// to avoid the triple bracket when calling global function
// nvcc does not seem to like variadic macros, so we have to define
// one for each kernel parameter list:
#ifdef __CUDACC__
#define KERNEL_ARGS2(grid, block) <<< grid, block >>>
#define KERNEL_ARGS3(grid, block, sh_mem) <<< grid, block, sh_mem >>>
#define KERNEL_ARGS4(grid, block, sh_mem, stream) <<< grid, block, sh_mem, stream >>>
#else
#define KERNEL_ARGS2(grid, block)
#define KERNEL_ARGS3(grid, block, sh_mem)
#define KERNEL_ARGS4(grid, block, sh_mem, stream)
#endif

#define CUDA_CHECK_RET(status)                                                                                                  \
    do                                                                                                                 \
    {                                                                                                                  \
        auto ret = (status);                                                                                           \
        if (ret != 0)                                                                                                  \
        {                                                                                                              \
            std::cerr << "Cuda failure: " << ret << " - " << cudaGetErrorString(ret) <<  std::endl;                                                 \
            abort();                                                                                                   \
        }                                                                                                              \
    } while (0)
