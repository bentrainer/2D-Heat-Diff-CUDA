#include <cstddef>
#include <vector>

#include <cuda_runtime.h>

#include "solver_result.hpp"
#include "perf.hpp"
#include "cuda_runner.hpp"


__global__ void solve_cuda_tiled_kernel(
    const float* T_curr, float* T_next,
    std::size_t Nx, std::size_t Ny,
    float rx, float ry, float cs
) {
    extern __shared__ float data[];

    const std::size_t i = blockIdx.y * blockDim.y + threadIdx.y;
    const std::size_t j = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t idx = i * Nx + j;

    const auto di = threadIdx.y + 1;
    const auto dj = threadIdx.x + 1;
    const auto didx = di * (blockDim.x + 2) + dj;

    data[didx] = ((i < Ny) && (j < Nx)) ? T_curr[idx] : 0.0f;

    if (di==1) {
        // load upper
        data[didx - (blockDim.x + 2)] = ((i > 0) && (j < Nx)) ? T_curr[idx - Nx] : 0.0f;
    }
    if (di==blockDim.y) {
        // load lower
        data[didx + (blockDim.x + 2)] = ((i+1 < Ny) && (j < Nx)) ? T_curr[idx + Nx] : 0.0f;
    }
    if (dj==1) {
        // load left
        data[didx - 1] = ((i < Ny) && (j > 0)) ? T_curr[idx - 1] : 0.0f;
    }
    if (dj==blockDim.x) {
        // load right
        data[didx + 1] = ((i < Ny) && (j+1 < Nx)) ? T_curr[idx + 1] : 0.0f;
    }

    __syncthreads();

    if ((i < Ny) && (j < Nx)) {
        T_next[idx] = cs * data[didx] +
            rx * (
                data[didx - 1] +
                data[didx + 1]
            ) +
            ry * (
                data[didx - (blockDim.x + 2)] +
                data[didx + (blockDim.x + 2)]
            );
    }
}


SolvResult solve_cuda_tiled(
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

    const auto shared_bytes = (block_size+2)*(block_size+2) * sizeof(float);

    // TODO: check
    return solve_cuda_common(
        T_init, steps,
        [=](const float* T_curr, float* T_next) {
            solve_cuda_tiled_kernel<<<grid, block, shared_bytes>>>(
                T_curr, T_next,
                Nx, Ny,
                rx, ry, cs
            );
        }
    );
}
