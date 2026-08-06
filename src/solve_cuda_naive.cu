#include <cstddef>
#include <vector>

#include <cuda_runtime.h>

#include "solver_result.hpp"
#include "perf.hpp"
#include "cuda_runner.hpp"


__global__ void solve_cuda_naive_kernel(
    const float* T_curr, float* T_next,
    std::size_t Nx, std::size_t Ny,
    float rx, float ry, float cs
) {
    const std::size_t i = blockIdx.y * blockDim.y + threadIdx.y;
    const std::size_t j = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t idx = i * Nx + j;

    if ((i < Ny) && (j < Nx)) {
        const float center = T_curr[idx];
        const float left   = (j > 0) ? T_curr[idx - 1] : 0.0f;
        const float right  = (j < Nx - 1) ? T_curr[idx + 1] : 0.0f;
        const float up     = (i > 0) ? T_curr[idx - Nx] : 0.0f;
        const float down   = (i < Ny - 1) ? T_curr[idx + Nx] : 0.0f;

        T_next[idx] = cs * center + rx * (left + right) + ry * (up + down);
    }
}


SolvResult solve_cuda_naive(
    const std::vector<float>& T_init,
    std::size_t Nx,
    std::size_t Ny,
    float alpha,
    float delta_x,
    float delta_y,
    float delta_t,
    std::size_t steps,
    std::size_t block_size
) {

    const float rx = alpha * delta_t / (delta_x*delta_x);
    const float ry = alpha * delta_t / (delta_y*delta_y);
    const float cs = 1.0f - 2.0f * rx - 2.0f * ry;

    const dim3 block(
        static_cast<unsigned int>(block_size),
        static_cast<unsigned int>(block_size)
    );
    const dim3 grid(
        static_cast<unsigned int>((Nx + block.x - 1) / block.x),
        static_cast<unsigned int>((Ny + block.y - 1) / block.y)
    );

    return solve_cuda_common(
        T_init, steps,
        [=](const float* T_curr, float* T_next) {
            solve_cuda_naive_kernel<<<grid, block>>>(
                T_curr, T_next,
                Nx, Ny,
                rx, ry, cs
            );
        }
    );
}
