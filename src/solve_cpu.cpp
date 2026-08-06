#include <cstddef>
#include <vector>

#include "solver_result.hpp"
#include "perf.hpp"

SolvResult solve_cpu(
    const std::vector<float>& T_init,
    std::size_t Nx,
    std::size_t Ny,
    float alpha,
    float delta_x,
    float delta_y,
    float delta_t,
    std::size_t steps
) {

    const auto t_func_begin = Clock::now();

    std::vector<float> T_curr = T_init;
    std::vector<float> T_next(T_init.size(), 0.0f);

    const float rx = alpha * delta_t / (delta_x*delta_x);
    const float ry = alpha * delta_t / (delta_y*delta_y);
    const float cs = (1.0f - 2.0f * rx -2.0f * ry);

    const auto j_bound = Nx - 1;
    const auto i_bound = Ny - 1;

    const auto t_setup_end = Clock::now();

    for (std::size_t step = 0; step < steps; step++) {
        for (std::size_t i = 0; i < Ny; i++) {
            for (std::size_t j = 0; j < Nx; j++) {
                const auto idx = i * Nx + j;

                const auto center = T_curr[idx];
                const auto left   = (j > 0) ? T_curr[idx-1] : 0.0f;
                const auto right  = (j < j_bound) ? T_curr[idx+1] : 0.0f;
                const auto up     = (i > 0) ? T_curr[idx-Nx] : 0.0f;
                const auto down   = (i < i_bound) ? T_curr[idx+Nx] : 0.0f;

                T_next[idx] = cs * center + rx * (left + right) + ry * (up + down);
            }
        }

        std::swap(T_curr, T_next);
    }

    const auto t_comp_end = Clock::now();


    SolvResult result;

    result.temperature = std::move(T_curr);

    const auto t_func_end = Clock::now();

    result.elapsed_us_setup = elapsed_us(t_func_begin, t_setup_end);
    result.elapsed_us_comp = elapsed_us(t_setup_end, t_comp_end);
    result.elapsed_us_total = elapsed_us(t_func_begin, t_func_end);

    result.elapsed_us_h2d = 0.0;
    result.elapsed_us_d2h = 0.0;

    return result;
}
