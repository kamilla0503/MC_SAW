#ifndef INTERACTION_SAW_MODELS_MODEL_H
#define INTERACTION_SAW_MODELS_MODEL_H

#include<iostream>
#include <random>
#include <fstream>
#include <chrono>


#include"common.h"
#include"lattice.h"

#ifndef N_CHAINS
#define N_CHAINS 16
#endif

template<class ExecSpace, class T>
struct DeviceData {
  //LATTICE: 
    Kokkos::View<int*, ExecSpace> map_of_contacts_int;
    Kokkos::View<int*, ExecSpace> inverse_steps;


  //MODEL SCALARS:
    Kokkos::View<int, ExecSpace> L;
    Kokkos::View<int, ExecSpace> lattice_side_device;
    

  //MODEl ARRAYS 
  Kokkos::View<T **, ExecSpace> sequence_on_lattice;
  Kokkos::View<int **, ExecSpace> next_monomers;
  Kokkos::View<int **, ExecSpace> previous_monomers;
  Kokkos::View<int **, ExecSpace> directions;
  Kokkos::View<int **, ExecSpace> lattice_nodes_positions;

  Kokkos::View<int*, ExecSpace> start_conformation;
  Kokkos::View<int*, ExecSpace> end_conformation;

  //Energy state 
  Kokkos::View<float*, ExecSpace> E;
  Kokkos::View<float*, ExecSpace> newE;

  //MC usable variables 
  Kokkos::View<T*, ExecSpace> oldspin;
  Kokkos::View<int*, ExecSpace> oldIndex; // not index --- it is really coord 
  Kokkos::View<int*, ExecSpace> newIndex; // not index --- it is really coord 
  Kokkos::View<int*, ExecSpace> save_start_conformation;
  Kokkos::View<int*, ExecSpace> save_end_conformation;
  Kokkos::View<int*, ExecSpace> start_index_in_nodes_position;
  Kokkos::View<int*, ExecSpace> direction;
  Kokkos::View<T*, ExecSpace> spinValue;

  Kokkos::View<int*, ExecSpace> accept_move; 
  Kokkos::View<float*, ExecSpace> flipMoveType;
  Kokkos::View<float*, ExecSpace> d_E_1; 
  Kokkos::View<float*, ExecSpace> J_chain;

};


class Model {
public:

    Model(int L);

    virtual void HostDataInit() = 0;
    virtual void DeviceDataInit () = 0;

public:
    int L;
    Kokkos::View<int, Kokkos::HostSpace> L_host;

    Lattice *lattice = nullptr;

};

template<class T>
class SAW_model : public Model {
public:
    SAW_model<T>(int L);

    // SAW_model(int L) : Model(L) {};
    virtual void spin_init_random() = 0;

    void geometry_initialization_arrays();

    void geometry_initialization_stick();

    void scalars_MC_preparation();


    void HostDataInit();
    void DeviceDataInit ();

    //HostData hostdata;
    DeviceData<Kokkos::CudaSpace, T> devicedata;
    DeviceData<Kokkos::HostSpace, T> hostdata;
};


class XY_LI : public SAW_model<float> {
public:
    XY_LI (int L);

    void spin_init_random();
};


class XY_SI : public SAW_model<float> {
public:

};


class Ising_SI : public SAW_model<int> {
public:

};


class Ising_LI : public SAW_model<int> {
public:

};

// Single source of truth list (add fields once, reuse everywhere)
#define DATA_FIELDS(X) \
  X(map_of_contacts_int) \
  X(inverse_steps) \
  X(L) \
  X(lattice_side_device) \
  X(sequence_on_lattice) \
  X(next_monomers) \
  X(previous_monomers) \
  X(directions) \
  X(lattice_nodes_positions) \
  X(start_conformation) \
  X(end_conformation) \
  /* Energy state */ \
  X(E) \
  X(newE) \
  \
  /* MC usable variables */ \
  X(oldspin) \
  X(oldIndex) \
  X(newIndex) \
  X(save_start_conformation) \
  X(save_end_conformation) \
  X(start_index_in_nodes_position) \
  X(direction) \
  X(spinValue) \
  \
  /* MC move bookkeeping */ \
  X(accept_move) \
  X(flipMoveType) \
  X(d_E_1) \
  X(J_chain)
  
template<class ExecSpace, class T>
void upload_all(const DeviceData<Kokkos::HostSpace, T>& h, DeviceData<ExecSpace, T>& d) {
#define X(name) realloc_like_and_copy(d.name, h.name, #name);
  DATA_FIELDS(X)
#undef X
}

template<class ExecSpace, class T>
void download_all(const DeviceData<ExecSpace, T>& d, DeviceData<Kokkos::HostSpace, T>& h) {
#define X(name) realloc_like_and_copy(h.name, d.name, #name);
  DATA_FIELDS(X)
#undef X
}


template<class T>
void SAW_model<T>::HostDataInit() {
    hostdata.map_of_contacts_int = lattice->map_of_contacts_int;
    hostdata.inverse_steps       = lattice->inverse_steps;

    hostdata.L = L_host;
    hostdata.lattice_side_device = lattice->lattice_side_host;

}

template<class T>
void SAW_model<T>::DeviceDataInit() {
    upload_all<Kokkos::CudaSpace, T>(hostdata, devicedata);
}

#endif