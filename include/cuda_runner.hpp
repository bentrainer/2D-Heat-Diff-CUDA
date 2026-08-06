#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <utility>
#include <vector>

#include "cuda_check.hpp"
#include "perf.hpp"
#include "solver_result.hpp"


template <typename KernelRunner>
SolvResult solve_cuda_common(
    const std::vector<float>& T_init,
    std::size_t steps,
    KernelRunner kernel_func
) {
    SolvResult result;

    const auto t_func_begin = Clock::now();

    std::vector<float> T_result(T_init.size());
    result.temperature = std::move(T_result);

    float* T_curr_d = nullptr;
    float* T_next_d = nullptr;

    const std::size_t bytes = T_init.size() * sizeof(float);

    CUDA_CHECK(cudaMalloc(&T_curr_d, bytes));
    CUDA_CHECK(cudaMalloc(&T_next_d, bytes));

    cudaEvent_t event_begin;
    cudaEvent_t event_end;

    CUDA_CHECK(cudaEventCreate(&event_begin));
    CUDA_CHECK(cudaEventCreate(&event_end));

    const auto t_setup_end = Clock::now();
    const auto t_h2d_begin = Clock::now();

    // copy & init data to GPU device
    CUDA_CHECK(cudaMemcpy(T_curr_d, T_init.data(), bytes, cudaMemcpyHostToDevice));
    // CUDA_CHECK(cudaMemset(T_next_d, 0, bytes));

    const auto t_h2d_end = Clock::now();

    CUDA_CHECK(cudaEventRecord(event_begin));

    // run solver kernel
    for (std::size_t step = 0; step < steps; ++step) {
        kernel_func(T_curr_d, T_next_d);
        CUDA_CHECK(cudaGetLastError());
        std::swap(T_curr_d, T_next_d);
    }

    CUDA_CHECK(cudaEventRecord(event_end));
    CUDA_CHECK(cudaEventSynchronize(event_end));

    float elapsed_ms_comp = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms_comp, event_begin, event_end));


    const auto t_d2h_begin = Clock::now();
    // copy data back to host
    CUDA_CHECK(cudaMemcpy(result.temperature.data(), T_curr_d, bytes, cudaMemcpyDeviceToHost));
    const auto t_d2h_end = Clock::now();


    CUDA_CHECK(cudaEventDestroy(event_begin));
    CUDA_CHECK(cudaEventDestroy(event_end));

    CUDA_CHECK(cudaFree(T_curr_d));
    CUDA_CHECK(cudaFree(T_next_d));

    const auto t_func_end = Clock::now();


    result.elapsed_us_setup = elapsed_us(t_func_begin, t_setup_end);
    result.elapsed_us_h2d = elapsed_us(t_h2d_begin, t_h2d_end);
    result.elapsed_us_comp = static_cast<double>(elapsed_ms_comp) * 1000.0;
    result.elapsed_us_d2h = elapsed_us(t_d2h_begin, t_d2h_end);
    result.elapsed_us_total = elapsed_us(t_func_begin, t_func_end);

    return result;
}
