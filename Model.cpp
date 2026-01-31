#include"Model.h"


//used to increase length of SAWs for lattice side
#ifndef OUT_Length
#define OUT_Length 4
#endif
//used to define lattice nodes without spins (SAW does not go over this node)
#ifndef NO_SAW_NODE
#define NO_SAW_NODE -1
#endif
//used to define lattice nodes without XY spins
#ifndef NO_XY_SPIN
#define NO_XY_SPIN -5
#endif

template class SAW_model<float>;
template class SAW_model<int>;

Model::Model(int L) : L(L) {
    #ifdef REGIME_2D
    lattice = new Lattice_2D(2 * L + OUT_Length);
#else
    lattice = new Lattice_3D(0.75*L+OUT_Length);
#endif

    L_host = Kokkos::View<int, Kokkos::HostSpace>("L");
    L_host() = L;
};

template<class T>
SAW_model<T>::SAW_model(int L) : Model(L) {
    geometry_initialization_arrays();

    geometry_initialization_stick();

}

template<class T>
void SAW_model<T>::geometry_initialization_arrays(){
    hostdata.next_monomers = Kokkos::View<int **, Kokkos::HostSpace> ("next_monomers", N_CHAINS, lattice-> NumberOfNodes());
    hostdata.previous_monomers = Kokkos::View<int **, Kokkos::HostSpace> ("previous_monomers", N_CHAINS, lattice-> NumberOfNodes());
    hostdata.directions = Kokkos::View<int **, Kokkos::HostSpace> ("directions", N_CHAINS, lattice-> NumberOfNodes());
    hostdata.lattice_nodes_positions = Kokkos::View<int **, Kokkos::HostSpace> ("lattice_nodes_positions", N_CHAINS, lattice-> NumberOfNodes());

    hostdata.start_conformation = Kokkos::View<int *, Kokkos::HostSpace> ("start_conformation", N_CHAINS);
    hostdata.end_conformation = Kokkos::View<int *, Kokkos::HostSpace> ("end_conformation", N_CHAINS);

}
 
 
template<class T>
void SAW_model<T>::geometry_initialization_stick() {

    for (int chain = 0; chain < N_CHAINS; chain++) {
        hostdata.start_conformation(chain) = 0;
        hostdata.end_conformation(chain) = L - 1;

        hostdata.lattice_nodes_positions(chain, 0) = hostdata.start_conformation(chain);
        hostdata.lattice_nodes_positions(chain, L - 1) = hostdata.end_conformation(chain);

        for (int i = 1; i < L - 1; i++) {
            hostdata.previous_monomers(chain, i) = i - 1;
            hostdata.next_monomers(chain, i) = i + 1;
            hostdata.lattice_nodes_positions(chain, i) = i;
        }
        hostdata.next_monomers(chain, 0) = 1;
        hostdata.previous_monomers(chain, L - 1) = L - 2;
        for (int i = 0; i < L - 1; i++) {
            hostdata.directions(chain, i) = 0; //all directions_h are the right moves
        }
    }


}