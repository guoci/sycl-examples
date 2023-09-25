// SYCL port of CUDA By Example, chapter05/ripple.cu
#include <sycl/sycl.hpp>
#include "./common/cpu_anim.h"

constexpr ssize_t DIM = 1024;


void kernel(sycl::accessor<unsigned char, 1, sycl::access_mode::write> ptr, int ticks, sycl::item<2> it) {
    // map from threadIdx/BlockIdx to pixel position
    ssize_t x = it.get_id(1);
    ssize_t y = it.get_id(0);
    auto offset = it.get_linear_id();
    // now calculate the value at that position
    float fx = static_cast<float>(x - DIM / 2);
    float fy = static_cast<float>(y - DIM / 2);
    float d = sqrtf(fx * fx + fy * fy);
    unsigned char grey = (unsigned char) (128.0f + 127.0f *
                                                   sycl::native::cos(d / 10.0f - ticks / 7.0f) /
                                                   (d / 10.0f + 1.0f));
    ptr[offset * 4 + 0] = grey;
    ptr[offset * 4 + 1] = grey;
    ptr[offset * 4 + 2] = grey;
    ptr[offset * 4 + 3] = 255;
}

struct DataBlock {
    sycl::buffer<unsigned char, 1> *dev_bitmap;
    sycl::queue q{sycl::gpu_selector_v};
    CPUAnimBitmap *bitmap;
};

void generate_frame(DataBlock *d, int ticks) {
    d->q.submit([&](sycl::handler &h) {
        sycl::accessor acc_bitmap{*d->dev_bitmap, h, sycl::write_only, sycl::no_init};
        h.parallel_for(
                sycl::nd_range<2>{sycl::range{DIM, DIM}, {16, 16}},
                [=](sycl::item<2> idx) {
                    kernel(acc_bitmap, ticks, idx);
                });
    });
    d->q.update_host(sycl::accessor{*d->dev_bitmap, sycl::read_only}).wait_and_throw();
}

// clean up memory allocated on the GPU
void cleanup(DataBlock *) {}

int main() {
    DataBlock data;
    CPUAnimBitmap bitmap(DIM, DIM, &data);
    data.bitmap = &bitmap;
    {
        sycl::buffer<unsigned char, 1> dev_bitmap{bitmap.get_ptr(),
                                                  sycl::range{static_cast<size_t>(bitmap.image_size())},
                                                  ::sycl::property::no_init{}};
        data.dev_bitmap = &dev_bitmap;
        bitmap.anim_and_exit((void (*)(void *, int)) generate_frame,
                             (void (*)(void *)) cleanup);
    }
    data.q.wait_and_throw();
}

