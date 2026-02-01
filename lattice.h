#ifndef INTERACTION_SAW_MODELS_LATTICE_H
#define INTERACTION_SAW_MODELS_LATTICE_H

#include <valarray>
#include <vector>

//#ifdef GPU
#include<Kokkos_Core.hpp>
//#endif

class Lattice {
    public:

    Lattice(int max_seq_size = 10) { 
        lattice_side = max_seq_size;
        lattice_side_host = Kokkos::View<int, Kokkos::HostSpace>("lattice_side");
        lattice_side_host() = lattice_side;
    };
    int lattice_size() {return lattice_side;};
    int NumberOfNodes () {return number_of_nodes;};

    virtual int ndim2() = 0;

    virtual void create_lattice() = 0 ;

    virtual float radius(const int& start, const int& end) = 0;
     
public:
    int lattice_side;
    int number_of_nodes;
    Kokkos::View<int, Kokkos::HostSpace> lattice_side_host;
    Kokkos::View<int*, Kokkos::HostSpace> map_of_contacts_int;
    Kokkos::View<int*, Kokkos::HostSpace> inverse_steps;
};

class Lattice_2D : public Lattice {
    public:
    Lattice_2D(int max_seq_size = 0);
    int ndim2()  {return 4;};

    void create_lattice();
    float radius(const int& start, const int& end);
};

class Lattice_3D : public Lattice {
    public:
    Lattice_3D(int max_seq_size = 0);

    int ndim2()  {return 6;};

    void create_lattice();
    float radius(const int& start, const int& end);
};


KOKKOS_INLINE_FUNCTION
float radius_sq_3d(const int& start, const int& end, const int& lattice_side) {
    int start_x = start % lattice_side;
    int start_y = (start % (lattice_side * lattice_side)) /lattice_side;
    int start_z = start / (lattice_side * lattice_side);
    int end_x = end % lattice_side;
    int end_y = (end % (lattice_side * lattice_side)) /lattice_side;
    int end_z = end / (lattice_side * lattice_side);

    //torus distance;
    float xdiff = abs(end_x - start_x);
    if (xdiff > (lattice_side/2))
        xdiff = lattice_side - xdiff;

    float ydiff = abs(end_y - start_y);
    if (ydiff > (lattice_side / 2))
        ydiff = lattice_side - ydiff;

    float zdiff = abs(end_z - start_z);
    if (zdiff > (lattice_side / 2))
        zdiff = lattice_side - zdiff;

    float r = xdiff *xdiff  + ydiff*ydiff + zdiff*zdiff;

    return r;
}

KOKKOS_INLINE_FUNCTION
float radius_sq_2d(const int& start, const int& end, const int& lattice_side) {
    int start_x = start % lattice_side;
    int start_y = start / lattice_side;
    int end_x = end % lattice_side;
    int end_y = end / lattice_side;

    //torus distance;
    float xdiff = abs(end_x - start_x);
    if (xdiff > (lattice_side/2))
        xdiff = lattice_side - xdiff;

    float ydiff = abs(end_y - start_y);
    if (ydiff > (lattice_side / 2))
        ydiff = lattice_side - ydiff;

    float r = xdiff *xdiff  + ydiff*ydiff ;

    return r;
}

#endif