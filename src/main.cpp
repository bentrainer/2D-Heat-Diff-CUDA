#include "CLI11.hpp"

int main(int argc, char** argv) {
    CLI::App app{"2D Heat Diffusion Test"};
    argv = app.ensure_utf8(argv);

    std::string backend = "cpu";
    app.add_option("-b,--backend", backend, "compute backend: cpu, cuda-naive")->check(
        CLI::IsMember({"cpu", "cuda-naive"})
    );

    unsigned width = 1000, height = 1000;
    app.add_option("-w,--width", width, "problem width size");
    app.add_option("-H,--height", height, "problem height size");

    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError &e) {
        return app.exit(e);
    }

    std::cout << "using backend: " << backend << "\n"
              << "problem size:  " << width  << "×" << height << "\n";

    return 0;
}