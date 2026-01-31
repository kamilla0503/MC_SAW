#ifndef INTERACTION_SAW_MODELS_MODEL_H
#define INTERACTION_SAW_MODELS_MODEL_H


#include"lattice.h"


struct DeviceData {
    Kokkos::View<int*, Kokkos::CudaSpace> map_of_contacts_int;
    Kokkos::View<int*, Kokkos::CudaSpace> inverse_steps;
};


class Model {
public:

    void DeciveDataInit ();


public:

};

class XY_LI : public Model {


};


class XY_SI : public Model {


};


class Ising_SI : public Model {


};


class Ising_LI : public Model {


};


#endif