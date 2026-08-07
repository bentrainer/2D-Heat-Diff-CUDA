#include <cstddef>
#include <vector>

#include <cuda_runtime.h>

#include "solver_result.hpp"
#include "perf.hpp"
#include "cuda_runner.hpp"


__global__ void solve_cuda_tiled_coarsen_kernel(
    const float* T_curr, float* T_next,
    const std::size_t Nx, const std::size_t Ny,
    const float rx, const float ry, const float cs,
    const std::size_t cfactor
) {

    extern __shared__ float data[];

    const std::size_t data_i = threadIdx.y + 1;
    const std::size_t data_j0 = threadIdx.x + 1;

    const std::size_t block_width = blockDim.x;
    const std::size_t out_width   = block_width * cfactor;
    const std::size_t tile_width  = block_width * cfactor + 2;
    const std::size_t d_pref = data_i * tile_width;

    const std::size_t i = static_cast<std::size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    const std::size_t j = static_cast<std::size_t>(blockIdx.x) * blockDim.x * cfactor + threadIdx.x;

    const std::size_t iNx = i * Nx;


    for (std::size_t k = 0; k < cfactor; k++) {
        const auto jkx = j + k * static_cast<std::size_t>(blockDim.x);
        const auto idx = iNx + jkx;

        const auto data_j = data_j0 + k * block_width;
        const auto didx = d_pref + data_j;

        data[didx] = ((i < Ny) && (jkx < Nx)) ? T_curr[idx] : 0.0f;

        if (data_i==1) {
            // load upper
            data[didx - tile_width] = ((i < Ny) && (i > 0) && (jkx < Nx)) ? T_curr[idx - Nx] : 0.0f;
        }
        if (data_i==blockDim.y) {
            // load lower
            data[didx + tile_width] = ((i+1 < Ny) && (jkx < Nx)) ? T_curr[idx + Nx] : 0.0f;
        }
        if (data_j==1) {
            // load left
            data[didx - 1] = ((i < Ny) && (jkx > 0)) ? T_curr[idx - 1] : 0.0f;
        }
        if (data_j==out_width) {
            // load right
            data[didx + 1] = ((i < Ny) && (jkx+1 < Nx)) ? T_curr[idx + 1] : 0.0f;
        }
    }
    __syncthreads();


    for (std::size_t k = 0; k < cfactor; k++) {
        const auto jkx = j + k * static_cast<std::size_t>(blockDim.x);
        const auto idx = iNx + jkx;

        const auto data_j = data_j0 + k * block_width;
        const auto didx = d_pref + data_j;

        if ((i < Ny) && (jkx < Nx)) {
            T_next[idx] = cs * data[didx] +
                rx * (
                    data[didx - 1] +
                    data[didx + 1]
                ) +
                ry * (
                    data[didx - tile_width] +
                    data[didx + tile_width]
                );
        }

    }

}


SolvResult solve_cuda_tiled_coarsen(
    const std::vector<float>& T_init,
    std::size_t Nx,
    std::size_t Ny,
    float alpha,
    float delta_x,
    float delta_y,
    float delta_t,
    std::size_t steps,
    std::size_t block_size,
    const std::size_t cfactor = 1
) {

    const float rx = alpha * delta_t / (delta_x*delta_x);
    const float ry = alpha * delta_t / (delta_y*delta_y);
    const float cs = 1.0f - 2.0f * rx - 2.0f * ry;

    const dim3 block(
        static_cast<unsigned int>(block_size),
        static_cast<unsigned int>(block_size)
    );

    // const std::size_t cfactor = 1;
    const std::size_t cbw = cfactor * static_cast<std::size_t>(block.x);

    const dim3 grid(
        static_cast<unsigned int>((Nx + cbw - 1) / cbw),
        static_cast<unsigned int>((Ny + block.y - 1) / block.y)
    );

    const std::size_t shared_bytes = (block_size*cfactor+2)*(block_size+2) * sizeof(float);

    return solve_cuda_common(
        T_init, steps,
        [=](const float* T_curr, float* T_next) {
            solve_cuda_tiled_coarsen_kernel<<<grid, block, shared_bytes>>>(
                T_curr, T_next,
                Nx, Ny,
                rx, ry, cs,
                cfactor
            );
        }
    );
}
