#ifndef INTERACTION_SAW_MODELS_MODEL_H
#define INTERACTION_SAW_MODELS_MODEL_H

#include"common.h"
#include"lattice.h"

// struct HostData {
//     Kokkos::View<int*, Kokkos::HostSpace> map_of_contacts_int;
//     Kokkos::View<int*, Kokkos::HostSpace> inverse_steps;

//   };

template<class ExecSpace>
struct DeviceData {
    Kokkos::View<int*, ExecSpace> map_of_contacts_int;
    Kokkos::View<int*, ExecSpace> inverse_steps;
};


class Model {
public:

    Model(int L);

    void HostDataInit();
    void DeviceDataInit ();


public:
    int L;

    Lattice *lattice = nullptr;

    //HostData hostdata;
    DeviceData<Kokkos::CudaSpace> devicedata;
    DeviceData<Kokkos::HostSpace> hostdata;
};


class SAW_model : public Model {
public:

    SAW_model(int L) : Model(L) {};

};


class XY_LI : public SAW_model {
public:
    XY_LI (int L) : SAW_model(L) {};
};


class XY_SI : public SAW_model {
public:

};


class Ising_SI : public Model {
public:

};


class Ising_LI : public Model {
public:

};

// Single source of truth list (add fields once, reuse everywhere)
#define DATA_FIELDS(X) \
  X(map_of_contacts_int) \
  X(inverse_steps)
  
  
template<class ExecSpace>
void upload_all(const DeviceData<Kokkos::HostSpace>& h, DeviceData<ExecSpace>& d) {
#define X(name) realloc_like_and_copy(d.name, h.name, #name);
  DATA_FIELDS(X)
#undef X
}

template<class ExecSpace>
void download_all(const DeviceData<ExecSpace>& d, DeviceData<Kokkos::HostSpace>& h) {
#define X(name) realloc_like_and_copy(h.name, d.name, #name);
  DATA_FIELDS(X)
#undef X
}

#endif