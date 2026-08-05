#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

std::vector<float> init_gaussian(
    std::size_t Nx,
    std::size_t Ny,
    float width,
    float height
) {
    std::vector<float> temperature(Nx * Ny, 0.0f);

    const float delta_x = width / static_cast<float>(Nx-1);
    const float delta_y = height / static_cast<float>(Ny-1);

    // peak temeperature at the center
    const float peak_temperature = 1.0f;
    const float center_x = 0.5f * width;
    const float center_y = 0.5f * height;

    const float sigma = 0.1f * std::min(width, height);
    const float oot_sigma2 = 1.0f / (2.0f * sigma * sigma);

    for (std::size_t i = 0; i < Ny; i++) {
        const float y = delta_y * static_cast<float>(i);
        const float dy = y - center_y;

        for (std::size_t j = 0; j < Nx; j++) {
            const float x = delta_x * static_cast<float>(j);
            const float dx = x - center_x;

            const std::size_t index = i * Nx + j;

            // gaussian
            temperature[index] = peak_temperature *
                std::exp(-(dx*dx + dy*dy) * oot_sigma2);
        }
    }

    return temperature;
}
