# IVF-Flat demo — 使用本地编译的 cuvs
# 必须指向含 cuvs-config.cmake 的目录（cuvs 的 cpp/build，不是 conda prefix）：
#   cmake -Dcuvs_DIR=/home/diy/lzx/cuvs/cpp/build -B build && cmake --build build

cmake_minimum_required(VERSION 3.18)
project(ivfflat_demo LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

# 显式指定 cuvs 的 build 目录（内含 cuvs-config.cmake）
if(NOT cuvs_DIR AND DEFINED ENV{CUVS_BUILD_DIR})
  set(cuvs_DIR "$ENV{CUVS_BUILD_DIR}" CACHE PATH "cuvs build dir")
endif()
if(cuvs_DIR)
  list(APPEND CMAKE_PREFIX_PATH "${cuvs_DIR}")
endif()
# 让 FindHDF5 能在 conda 环境中找到（先装: conda install -c conda-forge hdf5）
if(DEFINED ENV{CONDA_PREFIX})
  list(APPEND CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}")
endif()
if(HDF5_ROOT)
  list(APPEND CMAKE_PREFIX_PATH "${HDF5_ROOT}")
endif()

find_package(cuvs REQUIRED COMPONENTS cuvs_shared cuvs_cpp_headers)
find_package(HDF5)
if(NOT HDF5_FOUND)
  message(FATAL_ERROR "未找到 HDF5（需要 HDF5_LIBRARIES、HDF5_INCLUDE_DIRS）。\n请先安装 HDF5，例如:\n  conda install -c conda-forge hdf5\n  sudo apt install libhdf5-dev\n然后重新配置；若安装到非标准路径，可指定: -DHDF5_ROOT=/path/to/hdf5 或 -DCMAKE_PREFIX_PATH=\$CONDA_PREFIX")
endif()

add_executable(ivfflat_demo ivfflat_demo.cu)
target_include_directories(ivfflat_demo PRIVATE ${HDF5_INCLUDE_DIRS})
target_link_libraries(ivfflat_demo PRIVATE cuvs::cuvs ${HDF5_LIBRARIES})
set_target_properties(ivfflat_demo PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  POSITION_INDEPENDENT_CODE ON
)
