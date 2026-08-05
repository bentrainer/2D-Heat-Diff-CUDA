#pragma once

#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace cuda_check_impl {

inline void check(
    cudaError_t err,
    const char* expr,
    const char* file,
    int line
) {
    if (err == cudaSuccess) {
        return;
    }

    std::ostringstream oss;
    oss << "CUDA error at " << file << ":" << line << "\n"
        << "  expression: " << expr << "\n"
        << "  error code: " << static_cast<int>(err) << "\n"
        << "  error name: " << cudaGetErrorName(err) << "\n"
        << "  message: " << cudaGetErrorString(err);

    throw std::runtime_error(oss.str());
}

inline void check_last_kernel_launch(const char* file, int line) {
    check(cudaGetLastError(), "kernel launch", file, line);
}

inline void check_kernel_sync(const char* file, int line) {
    check(cudaGetLastError(), "kernel launch", file, line);
    check(cudaDeviceSynchronize(), "kernel execution", file, line);
}

}  // end of namespace cuda_check_impl

#define CUDA_CHECK(expr) \
    ::cuda_check_impl::check((expr), #expr, __FILE__, __LINE__)

#define CUDA_CHECK_KERNEL_LAUNCH() \
    ::cuda_check_impl::check_last_kernel_launch(__FILE__, __LINE__)

#define CUDA_CHECK_KERNEL_SYNC() \
    ::cuda_check_impl::check_kernel_sync(__FILE__, __LINE__)
