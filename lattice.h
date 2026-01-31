#ifndef INTERACTION_SAW_MODELS_LATTICE_H
#define INTERACTION_SAW_MODELS_LATTICE_H

#include <valarray>
#include <vector>

//#ifdef GPU
#include<Kokkos_Core.hpp>
//#endif

class Lattice {
    public:

    Lattice(int max_seq_size = 10) { lattice_side = max_seq_size; };
    int lattice_size() {return lattice_side;};
    int NumberOfNodes () {return number_of_nodes;};

    virtual int ndim2() = 0;

    virtual void create_lattice() = 0 ;
     
public:
    int lattice_side;
    int number_of_nodes;
    Kokkos::View<int*, Kokkos::HostSpace> map_of_contacts_int;
    Kokkos::View<int*, Kokkos::HostSpace> inverse_steps;
};

class Lattice_2D : public Lattice {
    public:
    Lattice_2D(int max_seq_size = 0);
    int ndim2()  {return 4;};

    void create_lattice();
};

class Lattice_3D : public Lattice {
    public:
    Lattice_3D(int max_seq_size = 0);

    int ndim2()  {return 6;};

    void create_lattice();
};

#endif