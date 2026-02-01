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
#define NO_XY_SPIN -100
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
    scalars_MC_preparation();

}

template<class T>
void SAW_model<T>::scalars_MC_preparation(){
    hostdata.E                  = Kokkos::View<float*, Kokkos::HostSpace>("E", N_CHAINS);
    hostdata.newE               = Kokkos::View<float*, Kokkos::HostSpace>("newE", N_CHAINS);
    
    hostdata.oldspin            = Kokkos::View<T*, Kokkos::HostSpace>("oldspin", N_CHAINS);
    hostdata.oldIndex           = Kokkos::View<int*, Kokkos::HostSpace>("oldIndex", N_CHAINS);
    hostdata.newIndex           = Kokkos::View<int*, Kokkos::HostSpace>("newIndex", N_CHAINS);
    
    hostdata.save_start_conformation        = Kokkos::View<int*, Kokkos::HostSpace>("save_start_conformation", N_CHAINS);
    hostdata.save_end_conformation          = Kokkos::View<int*, Kokkos::HostSpace>("save_end_conformation", N_CHAINS);
    hostdata.start_index_in_nodes_position  = Kokkos::View<int*, Kokkos::HostSpace>("start_index_in_nodes_position", N_CHAINS);
    
    hostdata.direction          = Kokkos::View<int*, Kokkos::HostSpace>("direction", N_CHAINS);
    hostdata.spinValue          = Kokkos::View<T*, Kokkos::HostSpace>("spinValue", N_CHAINS);

    hostdata.accept_move = Kokkos::View<int*,   Kokkos::HostSpace>("accept_move", N_CHAINS);
    hostdata.flipMoveType= Kokkos::View<float*, Kokkos::HostSpace>("flipMoveType", N_CHAINS);
    hostdata.d_E_1       = Kokkos::View<float*, Kokkos::HostSpace>("d_E_1", N_CHAINS);
    hostdata.J_chain     = Kokkos::View<float*, Kokkos::HostSpace>("J_chain", N_CHAINS);

    int N_pairs = L*(L-1)/2;
    hostdata.N_pairs = Kokkos::View<int, Kokkos::HostSpace>("N_pairs");
    hostdata.N_pairs() = N_pairs;

    hostdata.i_index = Kokkos::View<int*, Kokkos::HostSpace>("i_index", N_pairs);
    hostdata.j_index = Kokkos::View<int*, Kokkos::HostSpace>("j_index", N_pairs);

    int i_pair = 0 ;
    for (int i =0; i < L; i++) {
        for (int j = i + 1; j < L; j++) {
            hostdata.i_index(i_pair) = i;
            hostdata.j_index(i_pair) = j;
            i_pair += 1;
        }
    }
}

template<class T>
void SAW_model<T>::geometry_initialization_arrays(){
    hostdata.next_monomers = Kokkos::View<int **, Kokkos::HostSpace> ("next_monomers", N_CHAINS, lattice-> NumberOfNodes());
    hostdata.previous_monomers = Kokkos::View<int **, Kokkos::HostSpace> ("previous_monomers", N_CHAINS, lattice-> NumberOfNodes());
    hostdata.directions = Kokkos::View<int **, Kokkos::HostSpace> ("directions", N_CHAINS, lattice-> NumberOfNodes());
    hostdata.lattice_nodes_positions = Kokkos::View<int **, Kokkos::HostSpace> ("lattice_nodes_positions", N_CHAINS, lattice-> NumberOfNodes());

    //Fill emoty nodes in lattice 
    Kokkos::deep_copy(hostdata.next_monomers, NO_SAW_NODE);
    Kokkos::deep_copy(hostdata.previous_monomers, NO_SAW_NODE);
    Kokkos::deep_copy(hostdata.directions, NO_SAW_NODE);

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

XY_LI::XY_LI (int L) : SAW_model<float>(L) {
    spin_init_random();
};

void XY_LI::spin_init_random() {

    std::uniform_real_distribution<float> distribution_theta(0, 2.0*PI);
    std::mt19937 generators_theta;
    generators_theta.seed(std::chrono::steady_clock::now().time_since_epoch().count());

    hostdata.sequence_on_lattice = Kokkos::View<float**, Kokkos::HostSpace>("sequence_on_lattice", N_CHAINS, lattice-> NumberOfNodes());
    Kokkos::deep_copy(hostdata.sequence_on_lattice, NO_XY_SPIN);

    for (int chain = 0; chain < N_CHAINS; chain++) { 

        for (int i = 0; i < L; i++) {
            hostdata.sequence_on_lattice(chain, i) =  distribution_theta(generators_theta);
        }

    }
}

void XY_LI::start_kernel_energy_init() {
    auto d = this->devicedata;

    using team_policy = Kokkos::TeamPolicy<Kokkos::Cuda>;
    team_policy policy(N_CHAINS, Kokkos::AUTO());

    Kokkos::parallel_for("MCMC_Start", policy,
        StartFunctor{d}
      );

      Kokkos::fence();
}