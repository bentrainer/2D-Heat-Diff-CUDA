#include <cstddef>
#include <vector>

#include <cuda_runtime.h>

#include "solver_result.hpp"
#include "perf.hpp"
#include "cuda_runner.hpp"


__global__ void solve_cuda_coarsen_kernel(
    const float* T_curr, float* T_next,
    std::size_t Nx, std::size_t Ny,
    float rx, float ry, float cs,
    const std::size_t cfactor
) {

    const std::size_t i = static_cast<std::size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    const std::size_t j = static_cast<std::size_t>(blockIdx.x) * blockDim.x * cfactor + threadIdx.x;

    if (i < Ny){
        const std::size_t iNx = i * Nx;

        for (std::size_t k = 0; k < cfactor; k++) {
            const auto jkx = j + k * static_cast<std::size_t>(blockDim.x);
            const auto idx = iNx + jkx;

            if (jkx < Nx) {
                const float center = T_curr[idx];
                const float left   = (jkx > 0) ? T_curr[idx - 1] : 0.0f;
                const float right  = (jkx < Nx - 1) ? T_curr[idx + 1] : 0.0f;
                const float up     = (i > 0) ? T_curr[idx - Nx] : 0.0f;
                const float down   = (i < Ny - 1) ? T_curr[idx + Nx] : 0.0f;

                T_next[idx] = cs * center + rx * (left + right) + ry * (up + down);
            }
        }
    }

}


SolvResult solve_cuda_coarsen(
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

    // const auto shared_bytes = (block_size+2)*(block_size+2) * sizeof(float);


    return solve_cuda_common(
        T_init, steps,
        [=](const float* T_curr, float* T_next) {
            solve_cuda_coarsen_kernel<<<grid, block>>>(
                T_curr, T_next,
                Nx, Ny,
                rx, ry, cs,
                cfactor
            );
        }
    );
}
