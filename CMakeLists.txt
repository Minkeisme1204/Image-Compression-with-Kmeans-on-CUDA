cmake_minimum_required(VERSION 3.0)

project(IMAGE_KMEANS VERSION 1.0)
project(IMAGE_KMEANS CXX CUDA)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
# Set the CUDA standard
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED True)

# Declare the include folder for custom headers
include_directories(${CMAKE_SOURCE_DIR}/include)

find_package(CUDA REQUIRED)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_library(img_preprocess STATIC lib/img_preprocess.cpp)
target_link_libraries(img_preprocess ${OpenCV_LIBS})

add_library(kmeans STATIC lib/kmeans.cpp)
target_link_libraries(kmeans img_preprocess)

add_library(cuda_tools STATIC lib/cuda_tools.cu)
set_target_properties(cuda_tools PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON  # For separating device code
    CUDA_ARCHITECTURES "all"       # For compatibility with multiple GPU architectures
)

target_compile_options(cuda_tools PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
    $<$<COMPILE_LANGUAGE:CXX>:-Wall> 
)

add_library(kmeans_gpu STATIC lib/kmeans_gpu.cu)
target_link_libraries(kmeans_gpu cuda_tools img_preprocess)
set_target_properties(kmeans_gpu PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON  # For separating device code
    CUDA_ARCHITECTURES "all"       # For compatibility with multiple GPU architectures
)

target_compile_options(kmeans_gpu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
    $<$<COMPILE_LANGUAGE:CXX>:-Wall> 
)

# Testing Programs Declarations
enable_testing()
add_executable(test_read_img test/test_read_img.cpp)
target_link_libraries(test_read_img img_preprocess)

add_executable(test_write_img test/test_write_img.cpp)
target_link_libraries(test_write_img img_preprocess)

add_executable(test_kmeans test/test_kmeans.cpp)
target_link_libraries(test_kmeans kmeans)

add_executable(get_device_info test/get_device_info.cu)
set_target_properties(get_device_info PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON  # For separating device code
    CUDA_ARCHITECTURES "all"       # For compatibility with multiple GPU architectures
)

target_compile_options(get_device_info PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
    $<$<COMPILE_LANGUAGE:CXX>:-Wall>
)

add_executable(test_kmeans_cuda test/test_kmeans_cuda.cu)
target_link_libraries(test_kmeans_cuda ${OpenCV_LIBS})
set_target_properties(test_kmeans_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON  # For separating device code
    CUDA_ARCHITECTURES "all"       # For compatibility with multiple GPU architectures
)

target_compile_options(test_kmeans_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
    $<$<COMPILE_LANGUAGE:CXX>:-Wall>
)

add_executable(test_function_cuda test/test_function_cuda.cu)
target_link_libraries(test_function_cuda kmeans_gpu)
set_target_properties(test_function_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON  # For separating device code
    CUDA_ARCHITECTURES "all"       # For compatibility with multiple GPU architectures
)

target_compile_options(test_function_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
    $<$<COMPILE_LANGUAGE:CXX>:-Wall>
)