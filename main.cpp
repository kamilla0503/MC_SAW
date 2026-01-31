#include<iostream>
#include"lattice.h"
#include"Model.h"
#include"MonteCarlo.h"


int main(int argc, char *argv[]) {

    int L = std::atoi(argv[1]);
    float Jmin = std::stod(argv[2]);
    float Jmax = std::stod(argv[3]);


    XY_LI* xy_li = new XY_LI(L);
    xy_li->HostDataInit();
    xy_li->DeviceDataInit();
    MonteCarlo_SAW* mc_saw = new MonteCarlo_SAW(xy_li);

 
    return 0;
}