cmake_minimum_required(VERSION 3.26)
set(CMAKE_C_COMPILER "clang")
set(CMAKE_CXX_COMPILER "clang++")
set(CMAKE_VERBOSE_MAKEFILE ON CACHE BOOL "ON")
project(ray_tracing)


set(CMAKE_CXX_STANDARD 20)

set(SYCL_FLAGS
        "-fsycl"
        #        "-fsycl-targets=spir64_x86_64,nvptx64-nvidia-cuda"
        #        "-fsycl-targets=spir64_x86_64"
        "-fsycl-targets=nvptx64-nvidia-cuda"
        "-ferror-limit=1" "-Wall" "-Wpedantic"
        "-Wextra" "-Wshadow" "-Wconversion"
)


add_executable(ray_tracing ray.cpp)
target_compile_features(ray_tracing PUBLIC cxx_std_20)
target_compile_options(ray_tracing PUBLIC ${SYCL_FLAGS})
target_link_libraries(ray_tracing PUBLIC ${SYCL_FLAGS})
target_link_libraries(ray_tracing PUBLIC sycl)

#find_package(fmt)
#target_link_libraries(ray_tracing PUBLIC fmt::fmt)
target_link_libraries(ray_tracing PUBLIC "glut" "GL")
