#include<iostream>
#include <chrono>
#include <iomanip>
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
    std::string angles_file, dirs_file; 

 

    XY_LI<3>* xy_li = new XY_LI<3>(L, Jmin, Jmax);
   // xy_li->HostDataInit();
    
    printf("HostInit finished\n");

    auto now = std::chrono::system_clock::now();
    auto UTC = std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch()).count();


    std::string LogFile = argv[4];
    std::string filename_dirs = LogFile + "/dirs_" + std::to_string(L) +
    "_" + std::to_string(UTC) + ".out";    
    std::string filename_angles = LogFile + "/angles_" + std::to_string(L) +
    "_" + std::to_string(UTC)  + ".out";

    if (argc>5) {
      printf("Start file restore\n");
      angles_file = argv[5];
      dirs_file = argv[6];
      xy_li->restart_from_files(angles_file, dirs_file);
      printf("Finish file restore\n");
    }
    printf("DeviceInit finished\n");
    xy_li->DeviceDataInit();
    if (!xy_li->check_before_output(/*check_nextprev=*/true)) {
      // optionally abort or skip output
      printf("Check did not pass\n");
      return -1;
    }

    long long iters  = 10 * L;
    long long n_steps_to_equlibrium = 700 * L * L;

    long long iters_out = 2 * 10 * L * iters; 
    printf("Start MC\n");
    for (long long i = 0; i < MC_STEPS + 20; i += iters) {
        xy_li->runMCMCOnDevice(iters, (i/iters)+1);
        xy_li->start_kernel_energy_init();
        if (i % iters_out == 0) {
          if (i >= n_steps_to_equlibrium) {

            xy_li->append_angle_file(filename_angles, i);
            xy_li->append_dir_file(filename_dirs, i);

          }
        }
        
        xy_li->parallel_tempering_swap();

        //std::cout << i << " finished" << std::endl;


      }
      
    // xy_li->runMCMCOnDevice(iters);
    // MonteCarlo_SAW* mc_saw = new MonteCarlo_SAW(xy_li);
    return 0;
}