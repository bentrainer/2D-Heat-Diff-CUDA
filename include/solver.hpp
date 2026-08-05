#pragma once

#include <cstddef>
#include <vector>

#include "solver_result.hpp"


SolvResult solve_cpu(
    const std::vector<float>& T_init,
    std::size_t Nx,
    std::size_t Ny,
    float alpha,
    float delta_x,
    float delta_y,
    float delta_t,
    std::size_t steps
);

inline bool all_close(
    const std::vector<float>& A,
    const std::vector<float>& B,
    float rtol = 1.0e-4f,
    float atol = 1.0e-6f
) {
    if (A.size() != B.size()) {
        std::cerr << "size mismatch: " << A.size() << " vs " << B.size() << "\n";
        return false;
    }

    for (std::size_t k = 0; k < A.size(); k++) {
        const auto a = A[k];
        const auto b = B[k];

        if (!std::isfinite(a) || !std::isfinite(b)) {
            std::cerr << "non-infinite value at index " << k << "\n";
            return false;
        }

        const auto error = std::abs(a - b);
        const auto tolerance = atol + rtol * std::abs(b);

        if (error > tolerance) {
            std::cerr << "mismatch at index " << k << "\n"
                      << " > " << a << " != " << b << "\n"
                      << " > abs(err)  = " << error << "\n"
                      << " > tolerance = " << tolerance << "\n";
            return false;
        }
    }

    return true;
}
