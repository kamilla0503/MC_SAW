 cmake ..   -DKokkos_ENABLE_CUDA=ON  -DKokkos_ARCH_VOLTA70=ON  -DKokkos_ENABLE_LAMBDA=ON   -DKokkos_ENABLE_FORCE_UVM=ON   -DCMAKE_INSTALL_PREFIX=${HOME}/kokkos/install
cmake .. -DKokkos_ENABLE_CUDA=ON  -DKokkos_ARCH="Volta70" -DKokkos_ENABLE_LAMBDA=ON  -DCMAKE_INSTALL_PREFIX=${HOME}/kokkos/install -DKokkos_ENABLE_DEBUG_BOUNDS_CHECK=ON


rm -rf build/; mkdir build && cd build 
cmake .. -DKokkos_ENABLE_CUDA=ON   -DKokkos_ARCH_VOLTA70=ON -DKokkos_ENABLE_LAMBDA=ON  -DCMAKE_INSTALL_PREFIX=${HOME}/kokkos/install -DKokkos_ENABLE_DEBUG_BOUNDS_CHECK=ON
