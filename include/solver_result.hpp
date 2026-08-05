#pragma once

#include <iomanip>
#include <iostream>
#include <vector>


struct SolvResult {
    std::vector<float> temperature;
    double elapsed_us_setup = 0.0;
    double elapsed_us_h2d   = 0.0;
    double elapsed_us_comp  = 0.0;
    double elapsed_us_d2h   = 0.0;
    double elapsed_us_total = 0.0;
};

inline void print_stats(const SolvResult& result)
{
    const double measured_us =
        result.elapsed_us_setup
        + result.elapsed_us_h2d
        + result.elapsed_us_comp
        + result.elapsed_us_d2h;

    const double other_us = std::max(0.0, result.elapsed_us_total - measured_us);

    std::cout << std::fixed << std::setprecision(3)
        << "=== Solver Timing ===\n"
        << "setup_us="   << result.elapsed_us_setup << "\n"
        << "h2d_us="     << result.elapsed_us_h2d   << "\n"
        << "compute_us=" << result.elapsed_us_comp  << "\n"
        << "d2h_us="     << result.elapsed_us_d2h   << "\n"
        << "other_us="   << other_us                << "\n"
        << "total_us="   << result.elapsed_us_total << "\n";
}
