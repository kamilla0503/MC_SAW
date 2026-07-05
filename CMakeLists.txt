
cmake_minimum_required(VERSION 3.20)

# must be set before project(); changing it requires a fresh build dir
set(CMAKE_CXX_COMPILER "/home/knfayzullina/kokkos/bin/nvcc_wrapper" CACHE FILEPATH "")

project(MySimulation LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(Kokkos_DIR "/home/knfayzullina/kokkos/install/lib64/cmake/Kokkos")
find_package(Kokkos REQUIRED)
kokkos_check(DEVICES CUDA)          # fail configure if the install lacks CUDA

add_executable(mc_modeling
  main.cpp
  Model.cpp
  lattice.cpp
  # list every remaining .cpp here explicitly (ls *.cpp)
)

target_link_libraries(mc_modeling PRIVATE Kokkos::kokkos)

set(Q_CLOCK 3 CACHE STRING "Number of clock states")
target_compile_definitions(mc_modeling PRIVATE
  Q_CLOCK=${Q_CLOCK}
  REGIME_3D
  STARTHALF
)

target_compile_options(mc_modeling PRIVATE
  -lineinfo            # device line info for sanitizer / cuda-gdb, ~free
  # -use_fast_math     # re-enable only after one validated run without it
)
# # ------------------------------------------------------------------------------
# # 1) Basic C++/CUDA Settings
# # ------------------------------------------------------------------------------
# set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)

# # If you want to control CUDA standard as well
# set(CMAKE_CUDA_STANDARD 17)
# set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# set(CMAKE_CXX_COMPILER "/home/knfayzullina/kokkos/bin/nvcc_wrapper")

# cmake_minimum_required(VERSION 3.10)
# project(MySimulation LANGUAGES CXX CUDA)


# # ------------------------------------------------------------------------------
# # 2) Kokkos Configuration Options
# #    (You can turn these ON/OFF as needed, or set them via command line)
# # ------------------------------------------------------------------------------
# option(Kokkos_ENABLE_CUDA "Enable CUDA in Kokkos" ON)
# option(Kokkos_ENABLE_OPENMP "Enable OpenMP in Kokkos" OFF)
# option(Kokkos_ENABLE_SERIAL "Enable Serial in Kokkos" ON)
# option(Kokkos_ENABLE_TESTS "Enable Kokkos tests" OFF)
# option(Kokkos_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE ON)

# # GPU architecture flags:
# option(Kokkos_ARCH_VOLTA70 "Build Kokkos for Volta70 (sm_70)" OFF)
# option(Kokkos_ARCH_AMPERE80 "Build Kokkos for Ampere80 (sm_80)" OFF)
# # etc. for other arches if needed:
# # option(Kokkos_ARCH_PASCAL60 "..." OFF)
# # option(Kokkos_ARCH_MAXWELL52 "..." OFF)

# # ------------------------------------------------------------------------------
# # 3) Add Kokkos Subdirectory
# #    This will configure and build Kokkos as part of your project
# # ------------------------------------------------------------------------------
# #add_subdirectory(${HOME}/kokkos)
# #add_subdirectory(
# #        "/home/knfayzullina/kokkos"
# #        "/home/knfayzullina/kokkos/install"
# #)

# set(Kokkos_DIR "/home/knfayzullina/kokkos/install/lib64/cmake/Kokkos")
# find_package(Kokkos REQUIRED)

# # ------------------------------------------------------------------------------
# # 4) Gather Your Simulation Source Files
# # ------------------------------------------------------------------------------
# file(GLOB SIM_SRC
#         "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
#         # add other .cu/.cpp if needed
# )

# # ------------------------------------------------------------------------------
# # 5) Create the Executable for Your Simulation
# # ------------------------------------------------------------------------------
# add_executable(mc_modeling ${SIM_SRC})

# # ------------------------------------------------------------------------------
# # 6) Link Against Kokkos
# #    Kokkos exposes a CMake target named Kokkos::kokkos
# # ------------------------------------------------------------------------------
# target_link_libraries(mc_modeling PRIVATE Kokkos::kokkos)

# # ------------------------------------------------------------------------------
# # 7) Include Directories for Your Headers
# # ------------------------------------------------------------------------------
# target_include_directories(mc_modeling PRIVATE
#         "${CMAKE_CURRENT_SOURCE_DIR}/include"
# )

# set(Q_CLOCK 3 CACHE STRING "Number of clock states")
# target_compile_definitions(mc_modeling PRIVATE Q_CLOCK=${Q_CLOCK})

# # ------------------------------------------------------------------------------
# # 8) Optional: Additional Compile Flags (Equivalent to your CXXFLAGS)
# # ------------------------------------------------------------------------------
# set_property(TARGET mc_modeling PROPERTY CXX_STANDARD 14)
# set_property(TARGET mc_modeling PROPERTY CUDA_STANDARD 14)


# # Example: replicating your -O3 -DREGIME_3D -DSTARTHALF -use_fast_math
# target_compile_options(mc_modeling PRIVATE
#        $<$<COMPILE_LANGUAGE:CXX>:-O3 -DREGIME_3D -DSTARTHALF -use_fast_math> #-maxrregcount=64>
#        $<$<COMPILE_LANGUAGE:CUDA>:-O3 -DREGIME_3D -DSTARTHALF -use_fast_math>
# )

