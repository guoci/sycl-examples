# SYCL examples

Some simple SYCL code examples.

## N-body Maxwell-Boltzmann distribution

Uses N-body simulation to
approximate [Maxwell-Boltzmann distribution](https://en.wikipedia.org/wiki/Maxwell%E2%80%93Boltzmann_distribution).

https://github.com/guoci/sycl-examples/assets/1260178/161a5c62-2759-4e99-92a2-502f1f6e88da
<details>
<summary>Notes</summary>

Another implementation written in PyTorch can be
found [here](https://github.com/lukepolson/youtube_channel/blob/main/Python%20GPU/multibody_boltzmann.ipynb).
I found an [issue](https://github.com/lukepolson/youtube_channel/issues/12) with that code. The total energy do not
remain constant over time due to multiple collisions per particle at an iteration.

For my SYCL implementation, I resolved that with allowing only one collision per particle at an iteration.
</details>


## Ripple

SYCL rewrite
of [CUDA code](https://github.com/CodedK/CUDA-by-Example-source-code-for-the-book-s-examples-/blob/master/chapter05/ripple.cu)
in the "CUDA By Example" book.

https://github.com/guoci/sycl-examples/assets/1260178/e31a7f02-8a2e-40f5-a549-9171d0b4b5c9
<details>
<summary>Notes</summary>
The <code>update_host</code> member function can be very useful when we have a loop where a buffer is used repeatedly and we need the host data to update at each iteration.
Since we cannot destroy a buffer in the loop, the only way of forcing the update is to use <code>update_host</code>.
</details>

## Julia set

SYCL rewrite
of [CUDA code](https://github.com/CodedK/CUDA-by-Example-source-code-for-the-book-s-examples-/blob/master/chapter04/julia_gpu.cu)
in the "CUDA By Example" book.
![Julia set](julia_gpu/julia_gpu.png)

## Ray tracing

SYCL rewrite
of [CUDA code](https://github.com/CodedK/CUDA-by-Example-source-code-for-the-book-s-examples-/blob/master/chapter06/ray.cu)
in the "CUDA By Example" book.
![ray tracing](ray_tracing/ray.png)

## build and run
A SYCL compiler is required to compiler SYCL code. A normal C++ compiler would not work. See https://sycl.tech/ for more information.

This is built with the SYCL [compiler](https://software.intel.com/oneapi) from Intel.

`cd` into the directory of the example, then run the following in shell

```shell
cmake -DCMAKE_BUILD_TYPE=Release -S . -B ./cmake-build-release
cmake --build ./cmake-build-release --target <target_name> -j
cmake-build-release/<target_name>
```