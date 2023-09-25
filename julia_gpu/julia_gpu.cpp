// SYCL port of CUDA By Example, chapter04/julia_gpu.cu
#include <sycl/sycl.hpp>
#include "./common/cpu_bitmap.h"

constexpr ssize_t DIM = 1000;

struct cuComplex {
    float r;
    float i;

    cuComplex(float a, float b) : r(a), i(b) {}

    [[nodiscard]] float magnitude2() const {
        return r * r + i * i;
    }

    cuComplex operator*(const cuComplex &a) const {
        return {r * a.r - i * a.i, i * a.r + r * a.i};
    }

    cuComplex operator+(const cuComplex &a) const {
        return {r + a.r, i + a.i};
    }
};

bool julia(ssize_t x, ssize_t y) {
    const float scale = 1.5;
    float jx = scale * (float) (DIM / 2 - x) / (DIM / 2);
    float jy = scale * (float) (DIM / 2 - y) / (DIM / 2);

    cuComplex c(-0.8f, 0.156f);
    cuComplex a(jx, jy);

    for (ssize_t i = 0; i < 200; i++) {
        a = a * a + c;
        if (a.magnitude2() > 1000)
            return false;
    }

    return true;
}

void kernel(sycl::accessor<unsigned char, 1, sycl::access_mode::write> ptr, sycl::item<2> it) {
    // map from blockIdx to pixel position
    auto x = it.get_id(1);
    auto y = it.get_id(0);
    auto offset = it.get_linear_id();
    // now calculate the value at that position
    ptr[offset * 4 + 0] = julia(x, y) ? 255 : 0;
    ptr[offset * 4 + 1] = 0;
    ptr[offset * 4 + 2] = 0;
    ptr[offset * 4 + 3] = 255;
}

// globals needed by the update routine
struct DataBlock {
    unsigned char *dev_bitmap;
};

int main() {
    DataBlock data;
    CPUBitmap bitmap(DIM, DIM, &data);
    {
        sycl::buffer<unsigned char, 1> dev_bitmap{bitmap.get_ptr(),
                                                  sycl::range{static_cast<size_t>(bitmap.image_size())},
                                                  {::sycl::property::no_init{},
                                                   sycl::property::buffer::use_host_ptr{}}};
        sycl::queue{sycl::gpu_selector_v}.submit([&](sycl::handler &h) {
            sycl::accessor acc_bitmap{dev_bitmap, h, sycl::write_only, sycl::no_init};
            h.parallel_for(
                    sycl::nd_range<2>{sycl::range{DIM, DIM}, {1, 1}},
                    [=](sycl::item<2> idx) {
                        kernel(acc_bitmap, idx);
                    });
        });
    }
    bitmap.display_and_exit();
}

