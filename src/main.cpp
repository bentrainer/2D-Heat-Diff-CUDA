#include <cstddef>
#include <functional>
#include <stdexcept>
#include <vector>

#include "CLI11.hpp"
#include "init_gaussian.hpp"
#include "solver.hpp"


int main(int argc, char** argv) {
    CLI::App app{"2D Heat Diffusion Test"};
    argv = app.ensure_utf8(argv);

    std::string backend = "cpu";
    app.add_option("-b,--backend", backend, "compute backend: cpu, cuda-naive")->check(
        CLI::IsMember({"cpu", "cuda-naive"})
    );

    float width = 1.0f, height = 1.0f;
    std::size_t Nx = 1024, Ny = 1024, steps = 10;
    app.add_option("-W,--width", width, "problem width Lx");
    app.add_option("-H,--height", height, "problem height Ly");
    app.add_option("-x,--Nx", Nx, "problem discretization Nx");
    app.add_option("-y,--Ny", Ny, "problem discretization Ny");
    app.add_option("-s,--steps", steps, "solver steps");

    float alpha = 1.0f, delta_t = 1e-6;
    app.add_option("--alpha", alpha, "alpha");
    app.add_option("--delta-t", delta_t, "delta t");

    std::size_t block_size = 16;
    app.add_option("--block", block_size, "cuda block size");

    bool verify = false;
    app.add_flag("--verify", verify, "compare cuda result vs cpu");

    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError &e) {
        return app.exit(e);
    }

    const float delta_x = width / static_cast<float>(Nx-1);
    const float delta_y = height / static_cast<float>(Ny-1);

    if (alpha * delta_t * (1/(delta_x*delta_x) + 1/(delta_y*delta_y)) - 0.5f > 0) {
        delta_t = 0.5f / (alpha * (1/(delta_x*delta_x) + 1/(delta_y*delta_y))) / 2.0f;
        std::cout << "[WARN] update delta_t to " << delta_t << " for numerical stability\n";
    }

    std::cout << "using backend: " << backend << "\n"
              << "problem size:  " << width  << "x" << height << "\n"
              << "mesh size:     " << Nx << "x" << Ny << "\n"
              << "block size:    " << block_size << "\n"
              << "solve setup:   " << "delta_x=" << delta_x << ", delta_y=" << delta_y << "\n"
              << "               " << "alpha=" << alpha << ", delta_t=" << delta_t << ", steps=" << steps << "\n"
              << "verify?        " << (verify ? "YES" : "NO") << "\n";


    std::function<SolvResult(const std::vector<float>&)> solve_impl;

    if (backend == "cpu") {
        solve_impl = [&](const std::vector<float>& T) {
            return solve_cpu(
                T,
                Nx, Ny, alpha,
                delta_x, delta_y,
                delta_t, steps
            );
        };
    } else if (backend == "cuda-naive") {
        solve_impl = [&](const std::vector<float>& T) {
            return solve_cuda_naive(
                T,
                Nx, Ny, alpha,
                delta_x, delta_y,
                delta_t, steps,
                block_size
            );
        };
    }

    auto T_init = init_gaussian(Nx, Ny, width, height);
    auto result = solve_impl(T_init);
    auto T_solv = result.temperature;

    if (verify && backend != "cpu") {
        auto result = solve_cpu(
            T_init,
            Nx, Ny, delta_x, delta_y,
            delta_t, alpha, steps
        );
        auto T_comp = result.temperature;

        if (!all_close(T_solv, T_comp)) {
            std::cerr << "!!!FAILED TO VERIFY!!!\n";
        }
    }

    print_stats(result);

    return 0;
}