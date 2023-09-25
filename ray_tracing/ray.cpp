// SYCL port of CUDA By Example, chapter06/ray.cu
#include <sycl/sycl.hpp>
#include "./common/cpu_bitmap.h"

const auto DIM = 1024;

const auto rnd = [](auto x) { return x * rand() / RAND_MAX; };
const auto INF = 2e10f;

struct Sphere {
    float r, b, g;
    float radius;
    float x, y, z;

    float hit(float ox, float oy, float *n) const {
        float dx = ox - x;
        float dy = oy - y;
        if (dx * dx + dy * dy < radius * radius) {
            float dz = sycl::native::sqrt(radius * radius - dx * dx - dy * dy);
            *n = dz / sycl::native::sqrt(radius * radius);
            *n = dz / radius;
            return dz + z;
        }
        return -INF;
    }
};

constexpr auto SPHERES = 20;

// globals needed by the update routine
struct DataBlock {
    unsigned char *dev_bitmap;
};

int main() {
    DataBlock data;

    CPUBitmap bitmap(DIM, DIM, &data);

    Sphere *temp_s = (Sphere *) malloc(sizeof(Sphere) * SPHERES);
    for (int i = 0; i < SPHERES; i++) {
        temp_s[i].r = rnd(1.0f);
        temp_s[i].g = rnd(1.0f);
        temp_s[i].b = rnd(1.0f);
        temp_s[i].x = rnd(1000.0f) - 500;
        temp_s[i].y = rnd(1000.0f) - 500;
        temp_s[i].z = rnd(1000.0f) - 500;
        temp_s[i].radius = rnd(100.0f) + 20;
    }
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::enable_profiling{}};
    const auto start = std::chrono::steady_clock::now();
    {
        sycl::buffer buf_bitmap{bitmap.get_ptr(), sycl::range{static_cast<size_t>(bitmap.image_size())},
                                {sycl::property::buffer::use_host_ptr{}}
        };
        sycl::buffer buf_s{temp_s, sycl::range{SPHERES}};
        auto event = q.submit(
                [&](sycl::handler &h) {
                    sycl::accessor ptr{buf_bitmap, h, sycl::write_only, sycl::no_init};
                    sycl::accessor s{buf_s, h, sycl::read_only};
                    h.parallel_for(
                            sycl::nd_range<2>{{DIM, DIM},
                                              {16,  16}},
                            [=](sycl::nd_item<2> it) {
                                // map from threadIdx/BlockIdx to pixel position
                                ssize_t x = it.get_global_id(1);
                                ssize_t y = it.get_global_id(0);
                                auto offset = it.get_global_linear_id();
                                float ox = (x - DIM / 2);
                                float oy = (y - DIM / 2);

                                float r = 0, g = 0, b = 0;
                                float maxz = -INF;
                                for (int i = 0; i < SPHERES; i++) {
                                    float n;
                                    float t = s[i].hit(ox, oy, &n);
                                    if (t > maxz) {
                                        float fscale = n;
                                        r = s[i].r * fscale;
                                        g = s[i].g * fscale;
                                        b = s[i].b * fscale;
                                        maxz = t;
                                    }
                                }
                                ptr[offset * 4 + 0] = (int) (r * 255);
                                ptr[offset * 4 + 1] = (int) (g * 255);
                                ptr[offset * 4 + 2] = (int) (b * 255);
                                ptr[offset * 4 + 3] = 255;
                            });
                });
        event.wait_and_throw();
        auto i =
                event.get_profiling_info<::sycl::info::event_profiling::command_end>() -
                event.get_profiling_info<::sycl::info::event_profiling::command_submit>();
        std::cout << i * 1e-6 << "ms\n";
    }
    const auto diff = (std::chrono::steady_clock::now() - start);
    std::cout << std::chrono::duration<double, std::milli>(diff).count() << "ms\n";
    bitmap.display_and_exit();
}