#include<iostream>
#include"lattice.h"
#include"Model.h"
#include"MonteCarlo.h"


#ifndef  MC_STEPS
#define MC_STEPS 10000000000000 //99000000 //10000000000
#endif

int main(int argc, char *argv[]) {

    Kokkos::initialize(Kokkos::InitializationSettings() );
    std::cout << "DefaultExecutionSpace = " 
          << Kokkos::DefaultExecutionSpace::name() << "\n";
    int L = std::atoi(argv[1]);
    float Jmin = std::stod(argv[2]);
    float Jmax = std::stod(argv[3]);


    XY_LI<3>* xy_li = new XY_LI<3>(L, Jmin, Jmax);
    xy_li->HostDataInit();
    xy_li->DeviceDataInit();


    long long iters  = 10 * L;
    long long n_steps_to_equlibrium = 200 * iters;

    long long iters_out = 10 * 10 * L * iters; 

    for (long long i = 0; i < MC_STEPS + 20; i += iters) {
        xy_li->runMCMCOnDevice(iters, (i/iters)+1);
      
        if (i % iters_out == 0) {
          if (i >= n_steps_to_equlibrium) {
            xy_li->append_angle_file("angles.dat", i);
            xy_li->append_dir_file("dirs.dat", i);
          }
        }
      
        xy_li->parallel_tempering_swap();
      }
    // xy_li->runMCMCOnDevice(iters);
    // MonteCarlo_SAW* mc_saw = new MonteCarlo_SAW(xy_li);
    return 0;
}