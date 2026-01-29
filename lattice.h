#ifndef INTERACTION_SAW_MODELS_LATTICE_H
#define INTERACTION_SAW_MODELS_LATTICE_H

#include <valarray>
#include <vector>

#ifdef GPU
#include<Kokkos_Core.hpp>
#endif

class Lattice {
    public:

    Lattice(int max_seq_size = 10) { lattice_side = max_seq_size; };
    int lattice_size() {return lattice_side;};
    int NumberOfNodes () {return number_of_nodes;};

    // #ifdef GPU
    // KOKKOS_INLINE_FUNCTION 
    // #endif
    virtual int ndim2() = 0;
     
    private:
    int lattice_side;
    int number_of_nodes;

};

class Lattice_2D : public Lattice {
    // #ifdef GPU
    // KOKKOS_INLINE_FUNCTION 
    // #endif
    int ndim2()  {return 4;};

};

class Lattice_3D : public Lattice {
    // #ifdef GPU
    // KOKKOS_INLINE_FUNCTION 
    // #endif
    int ndim2()  {return 6;};
};

#endif