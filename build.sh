module load CUDA
module load cmake/3.21.3
rm -rf build; mkdir build; cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_VOLTA70=ON -DKokkos_ENABLE_SERIAL=ON -DKokkos_FORCE_DEVICE=ON  #-DKokkos_ENABLE_HWLOC=OFF
make
