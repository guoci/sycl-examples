cmake_minimum_required(VERSION 3.26)
set(CMAKE_C_COMPILER "clang")
set(CMAKE_CXX_COMPILER "clang++")
set(CMAKE_VERBOSE_MAKEFILE ON CACHE BOOL "ON")
project(ripple)


set(CMAKE_CXX_STANDARD 20)

set(SYCL_FLAGS
        "-fsycl"
        #        "-fsycl-targets=spir64_x86_64,nvptx64-nvidia-cuda"
        #        "-fsycl-targets=spir64_x86_64"
        "-fsycl-targets=nvptx64-nvidia-cuda"
        "-ferror-limit=1" "-Wall" "-Wpedantic"
        "-Wextra" "-Wshadow" "-Wconversion"
)


add_executable(ripple ripple.cpp)
target_compile_features(ripple PUBLIC cxx_std_20)
target_compile_options(ripple PUBLIC ${SYCL_FLAGS})
target_link_libraries(ripple PUBLIC ${SYCL_FLAGS})
target_link_libraries(ripple PUBLIC sycl)

#find_package(fmt)
#target_link_libraries(ripple PUBLIC fmt::fmt)
target_link_libraries(ripple PUBLIC "glut" "GL")
