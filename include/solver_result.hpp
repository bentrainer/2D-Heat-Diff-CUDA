#pragma once

#include <iomanip>
#include <iostream>
#include <vector>


struct SolvResult {
    std::vector<float> temperature;
    double elapsed_ms_setup = 0.0;
    double elapsed_ms_h2d   = 0.0;
    double elapsed_ms_comp  = 0.0;
    double elapsed_ms_d2h   = 0.0;
    double elapsed_ms_total = 0.0;
};

inline void print_stats(const SolvResult& result)
{
    const double measured_ms =
        result.elapsed_ms_setup
        + result.elapsed_ms_h2d
        + result.elapsed_ms_comp
        + result.elapsed_ms_d2h;

    const double other_ms = std::max(0.0, result.elapsed_ms_total - measured_ms);

    std::cout << std::fixed << std::setprecision(3)
        << "=== Solver Timing ===\n"
        << "setup_ms="   << result.elapsed_ms_setup << "\n"
        << "h2d_ms="     << result.elapsed_ms_h2d   << "\n"
        << "compute_ms=" << result.elapsed_ms_comp  << "\n"
        << "d2h_ms="     << result.elapsed_ms_d2h   << "\n"
        << "other_ms="   << other_ms                << "\n"
        << "total_ms="   << result.elapsed_ms_total << "\n";
}
