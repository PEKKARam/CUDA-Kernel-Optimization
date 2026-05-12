cmake_minimum_required(VERSION 3.18)
project(cuda_kernel_optimization LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(vector_add vector_add/main.cu)
add_executable(reduction reduction/main.cu)
add_executable(matmul matmul/main.cu)

foreach(tgt vector_add reduction matmul)
  set_target_properties(${tgt} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
  )
endforeach()
