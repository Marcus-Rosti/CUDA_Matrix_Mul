cmake_minimum_required(VERSION 2.8)
project(CUDA_Matrix_Mul)

set(CMAKE_GENERATOR_PLATFORM "Visual Studio 14")

SET (CMAKE_C_COMPILER "/usr/bin/clang")
SET (CMAKE_CXX_COMPILER "/usr/bin/clang++")

SET (CMAKE_AR      "/usr/bin/llvm-ar")
SET (CMAKE_LINKER  "/usr/bin/llvm-ld")
SET (CMAKE_NM      "/usr/bin/llvm-nm")
SET (CMAKE_OBJDUMP "/usr/bin/llvm-objdump")
SET (CMAKE_RANLIB  "/usr/bin/llvm-ranlib")
find_package(CUDA QUIET REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}" -O3)

set(SOURCE_FILES src/mmm_template.cu)

cuda_add_executable(CUDA_Matrix_Mul ${SOURCE_FILES})