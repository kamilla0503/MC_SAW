#include"Model.h"


template class SAW_model<float, 2>;
template class SAW_model<float, 3>;
template class SAW_model<int, 2>;
template class SAW_model<int, 3>;
template class XY_LI<2>;
template class XY_LI<3>;
template class XY_SI<2>;
template class XY_SI<3>;
template class Ising_SI<2>;
template class Ising_SI<3>;
template class Homopolymer<2>;
template class Homopolymer<3>;

template<int Dim>
Model<Dim>::Model(int L) : L(L) {
 //   #ifdef REGIME_2D
 //   lattice = new Lattice_2D(2 * L + OUT_Length);
//#else
    if constexpr (Dim == 3) {
        lattice = new Lattice_3D(0.55*L+OUT_Length);
    }
    else if constexpr (Dim == 2) {
        lattice = new Lattice_2D(0.95*L+OUT_Length);
    }
    else {
        lattice = new Lattice_2D(0.95*L+OUT_Length);
    }
    //lattice = new Lattice_3D(1.05*L+OUT_Length);
//#endif

    L_host = Kokkos::View<int, Kokkos::HostSpace>("L");
    L_host() = L;
};

template<class T, int Dim>
SAW_model<T, Dim>::SAW_model(int L,  float Jmin, float Jmax) : Model<Dim>(L) {
    HostDataInit();
    geometry_initialization_arrays();
    //geometry_initialization_stick();
    geometry_initialization_half();
    scalars_MC_preparation(Jmin, Jmax);

}

template<class T, int Dim>
void SAW_model<T, Dim>::scalars_MC_preparation(float Jmin, float Jmax){
    hostdata.E                  = Kokkos::View<float*, Kokkos::HostSpace>("E", N_CHAINS);
    hostdata.newE               = Kokkos::View<float*, Kokkos::HostSpace>("newE", N_CHAINS);
    
    hostdata.oldspin            = Kokkos::View<T*, Kokkos::HostSpace>("oldspin", N_CHAINS);
    hostdata.oldIndex           = Kokkos::View<int*, Kokkos::HostSpace>("oldIndex", N_CHAINS);
    hostdata.newIndex           = Kokkos::View<int*, Kokkos::HostSpace>("newIndex", N_CHAINS);
    
    hostdata.save_start_conformation        = Kokkos::View<int*, Kokkos::HostSpace>("save_start_conformation", N_CHAINS);
    hostdata.save_end_conformation          = Kokkos::View<int*, Kokkos::HostSpace>("save_end_conformation", N_CHAINS);
   // hostdata.start_index_in_nodes_position  = Kokkos::View<int*, Kokkos::HostSpace>("start_index_in_nodes_position", N_CHAINS);
    
    hostdata.direction          = Kokkos::View<int*, Kokkos::HostSpace>("direction", N_CHAINS);
    hostdata.spinValue          = Kokkos::View<T*, Kokkos::HostSpace>("spinValue", N_CHAINS);

    hostdata.accept_move = Kokkos::View<int*,   Kokkos::HostSpace>("accept_move", N_CHAINS);
    hostdata.flipMoveType= Kokkos::View<float*, Kokkos::HostSpace>("flipMoveType", N_CHAINS);
    hostdata.d_E_1       = Kokkos::View<float*, Kokkos::HostSpace>("d_E_1", N_CHAINS);
    hostdata.J_chain     = Kokkos::View<float*, Kokkos::HostSpace>("J_chain", N_CHAINS);

    int N_pairs = this->L*(this->L-1)/2;
    hostdata.N_pairs = Kokkos::View<int, Kokkos::HostSpace>("N_pairs");
    hostdata.N_pairs() = N_pairs;

    hostdata.i_index = Kokkos::View<int*, Kokkos::HostSpace>("i_index", N_pairs);
    hostdata.j_index = Kokkos::View<int*, Kokkos::HostSpace>("j_index", N_pairs);

    int i_pair = 0 ;
    for (int i =0; i < this->L; i++) {
        for (int j = i + 1; j < this->L; j++) {
            hostdata.i_index(i_pair) = i;
            hostdata.j_index(i_pair) = j;
            i_pair += 1;
        }
    }

    for (int c = 0; c < N_CHAINS; ++c) {
        hostdata.J_chain(c) = Jmin + (Jmax - Jmin) * (float(c) / (N_CHAINS - 1));
    }
 
}

template<class T, int Dim>
void SAW_model<T, Dim>::geometry_initialization_arrays(){
    hostdata.next_monomers = Kokkos::View<int **, Kokkos::HostSpace> ("next_monomers", N_CHAINS, this->lattice-> NumberOfNodes());
    hostdata.previous_monomers = Kokkos::View<int **, Kokkos::HostSpace> ("previous_monomers", N_CHAINS, this->lattice-> NumberOfNodes());
    hostdata.directions = Kokkos::View<int **, Kokkos::HostSpace> ("directions", N_CHAINS, this->lattice-> NumberOfNodes());
    hostdata.lattice_nodes_positions = Kokkos::View<int **, Kokkos::HostSpace> ("lattice_nodes_positions", N_CHAINS,this->L);

    //Fill emoty nodes in lattice 
    Kokkos::deep_copy(hostdata.next_monomers, NO_SAW_NODE);
    Kokkos::deep_copy(hostdata.previous_monomers, NO_SAW_NODE);
    Kokkos::deep_copy(hostdata.directions, NO_SAW_NODE);

    hostdata.start_index_in_nodes_position  = Kokkos::View<int*, Kokkos::HostSpace>("start_index_in_nodes_position", N_CHAINS);
    Kokkos::deep_copy(hostdata.start_index_in_nodes_position, 0);

    hostdata.start_conformation = Kokkos::View<int *, Kokkos::HostSpace> ("start_conformation", N_CHAINS);
    hostdata.end_conformation = Kokkos::View<int *, Kokkos::HostSpace> ("end_conformation", N_CHAINS);

}
 
template<class T, int Dim>
void SAW_model<T, Dim>::geometry_initialization_half() {
 
    auto nbr = [&](int node, int dir) -> int {
        return hostdata.map_of_contacts_int(hostdata.ndim2() * node + dir);
      };

    const int Lm = this->L;   

    for (int chain = 0; chain < N_CHAINS; ++chain) {

        int middle = (Lm / 2) - 1;

        // start at 0
        hostdata.start_conformation(chain) = 0;
        hostdata.start_index_in_nodes_position(chain) = 0;
        hostdata.lattice_nodes_positions(chain, 0) = 0;
        hostdata.previous_monomers(chain, 0) = NO_SAW_NODE;
        hostdata.directions(chain, 0) = 0;
        for (int i = 1; i < middle; ++i) {
            hostdata.previous_monomers(chain, i) = nbr(i, 1);
            hostdata.next_monomers(chain, i) = nbr(i, 0);
            hostdata.directions(chain, i) = 0;
            hostdata.lattice_nodes_positions(chain, i) = i;
        }

        int i_pos = middle;
        hostdata.previous_monomers(chain, middle) = nbr(middle, 1);
        hostdata.next_monomers(chain, middle) = nbr(middle, 2);
        hostdata.directions(chain, middle) = 2;   // Go Up
        hostdata.lattice_nodes_positions(chain, i_pos) = middle;
        i_pos += 1;

        middle = hostdata.next_monomers(chain, middle);

        hostdata.previous_monomers(chain, middle) = nbr(middle, 3);
        hostdata.next_monomers(chain, middle) = nbr(middle, 1);
        hostdata.directions(chain, middle) = 1;
        hostdata.lattice_nodes_positions(chain, i_pos) = middle;
        i_pos += 1;
        middle = hostdata.next_monomers(chain, middle);
        hostdata.lattice_nodes_positions(chain, i_pos) = middle;
        for (int pos = (Lm / 2) + 2; pos < Lm; ++pos) {
            hostdata.previous_monomers(chain, middle) = nbr(middle, 0);
            hostdata.next_monomers(chain, middle) = nbr(middle, 1);
            hostdata.directions(chain, middle) = 1;
      
            middle = hostdata.next_monomers(chain, middle);
            hostdata.lattice_nodes_positions(chain, pos) = middle;
          }

          hostdata.end_conformation(chain) = middle;
     
        hostdata.next_monomers(chain, 0) = nbr(0, 0);
        hostdata.previous_monomers(chain, hostdata.end_conformation(chain)) = nbr(hostdata.end_conformation(chain), 0);

        // terminate list at end (good hygiene for SAW linked list)
        hostdata.next_monomers(chain, hostdata.end_conformation(chain)) = NO_SAW_NODE;
        hostdata.directions(chain, hostdata.end_conformation(chain)) = NO_SAW_NODE;

        // ensure last position is end
        hostdata.lattice_nodes_positions(chain, Lm - 1) = hostdata.end_conformation(chain);
    

    }
 
 }
 
template<class T, int Dim>
void SAW_model<T, Dim>::geometry_initialization_stick() {

    for (int chain = 0; chain < N_CHAINS; chain++) {
        hostdata.start_conformation(chain) = 0;
        hostdata.end_conformation(chain) = this->L - 1;

        hostdata.lattice_nodes_positions(chain, 0) = hostdata.start_conformation(chain);
        hostdata.lattice_nodes_positions(chain, this->L - 1) = hostdata.end_conformation(chain);

        for (int i = 1; i < this->L - 1; i++) {
            hostdata.previous_monomers(chain, i) = i - 1;
            hostdata.next_monomers(chain, i) = i + 1;
            hostdata.lattice_nodes_positions(chain, i) = i;
        }
        hostdata.next_monomers(chain, 0) = 1;
        hostdata.previous_monomers(chain, this->L - 1) = this->L - 2;
        for (int i = 0; i < this->L - 1; i++) {
            hostdata.directions(chain, i) = 0; //all directions_h are the right moves
        }
    }
}


template<int Dim>
XY_LI<Dim>::XY_LI (int L, float Jmin, float Jmax) : SAW_model<float, Dim>(L, Jmin, Jmax) {
    spin_init_random();
};

template<int Dim>
XY_SI<Dim>::XY_SI (int L, float Jmin, float Jmax) : SAW_model<float, Dim>(L, Jmin, Jmax) {
    spin_init_random();
};

template<int Dim>
void XY_LI<Dim>::spin_init_random() {

    std::uniform_real_distribution<float> distribution_theta(0, 2.0*PI);
    std::mt19937 generators_theta;
    generators_theta.seed(std::chrono::steady_clock::now().time_since_epoch().count());

    hostdata.sequence_on_lattice = Kokkos::View<float**, Kokkos::HostSpace>("sequence_on_lattice", N_CHAINS, lattice-> NumberOfNodes());
    Kokkos::deep_copy(hostdata.sequence_on_lattice, NO_XY_SPIN);

    for (int chain = 0; chain < N_CHAINS; chain++) { 

        for (int i = 0; i < this->L; i++) {
            hostdata.sequence_on_lattice(chain, hostdata.lattice_nodes_positions(chain, i)) =  distribution_theta(generators_theta);
        }

    }
}


template<int Dim>
void XY_SI<Dim>::spin_init_random() {

    std::uniform_real_distribution<float> distribution_theta(0, 2.0*PI);
    std::mt19937 generators_theta;
    generators_theta.seed(std::chrono::steady_clock::now().time_since_epoch().count());

    hostdata.sequence_on_lattice = Kokkos::View<float**, Kokkos::HostSpace>("sequence_on_lattice", N_CHAINS, lattice-> NumberOfNodes());
    Kokkos::deep_copy(hostdata.sequence_on_lattice, NO_XY_SPIN);

    for (int chain = 0; chain < N_CHAINS; chain++) { 

        for (int i = 0; i < this->L; i++) {
            hostdata.sequence_on_lattice(chain, hostdata.lattice_nodes_positions(chain, i)) =  distribution_theta(generators_theta);
        }

    }
}


template<int Dim>
void XY_LI<Dim>::start_kernel_energy_init() {
   //auto d = this->devicedata;
   XY_LI_EnergyOp op{ this->devicedata };
   this->energy_init(op);
}

template<int Dim>
void XY_SI<Dim>::start_kernel_energy_init() {
   //auto d = this->devicedata;
   XY_SI_EnergyOp op{ this->devicedata };
   this->energy_init(op);
}


template<int Dim>
Ising_SI<Dim>::Ising_SI (int L, float Jmin, float Jmax) : SAW_model<int, Dim>(L, Jmin, Jmax) {
    spin_init_random();
};

template<int Dim>
void Ising_SI<Dim>::spin_init_random() {

    std::mt19937 gen((unsigned)std::chrono::steady_clock::now().time_since_epoch().count());
    std::uniform_int_distribution<int> bit(0, 1);

    hostdata.sequence_on_lattice = Kokkos::View<int**, Kokkos::HostSpace>("sequence_on_lattice", N_CHAINS, lattice-> NumberOfNodes());
    Kokkos::deep_copy(hostdata.sequence_on_lattice, NO_XY_SPIN);

    for (int chain = 0; chain < N_CHAINS; chain++) { 
        for (int i = 0; i < this->L; i++) {
            hostdata.sequence_on_lattice(chain, hostdata.lattice_nodes_positions(chain, i)) = bit(gen) ? +1 : -1;
        }
    }   
}

template<int Dim>
void Ising_SI<Dim>::start_kernel_energy_init() {
   Ising_SI_EnergyOp op{ this->devicedata };
   this->energy_init(op);
}



template<int Dim>
Homopolymer<Dim>::Homopolymer(int L, float Jmin, float Jmax)
  : SAW_model<int, Dim>(L, Jmin, Jmax)
{
  spin_init_random();
}

template<int Dim>
void Homopolymer<Dim>::spin_init_random() {
  hostdata.sequence_on_lattice =
      Kokkos::View<int**, Kokkos::HostSpace>(
          "sequence_on_lattice", N_CHAINS, lattice->NumberOfNodes());

  Kokkos::deep_copy(hostdata.sequence_on_lattice, NO_XY_SPIN);

  for (int chain = 0; chain < N_CHAINS; ++chain) {
    for (int i = 0; i < this->L; ++i) {
      const int pos = hostdata.lattice_nodes_positions(chain, i);
      hostdata.sequence_on_lattice(chain, pos) = 1;
    }
  }
}

template<int Dim>
void Homopolymer<Dim>::start_kernel_energy_init() {
  Homopolymer_EnergyOp op{ this->devicedata };
  this->energy_init(op);
}



// Explicit instantiations
template class XY_LI_normalize<2>;
template class XY_LI_normalize<3>;

template<int Dim>
XY_LI_normalize<Dim>::XY_LI_normalize(int L, float Jmin, float Jmax, float power)
    : SAW_model<float, Dim>(L, Jmin, Jmax),
      power_ij(power)
{
    spin_init_random();
}

template<int Dim>
void XY_LI_normalize<Dim>::spin_init_random() {
    std::uniform_real_distribution<float> distribution_theta(0.0f, 2.0f * PI);
    std::mt19937 generators_theta;
    generators_theta.seed(
        std::chrono::steady_clock::now().time_since_epoch().count());

    hostdata.sequence_on_lattice = Kokkos::View<float**, Kokkos::HostSpace>(
        "sequence_on_lattice", N_CHAINS, lattice->NumberOfNodes());
    Kokkos::deep_copy(hostdata.sequence_on_lattice, NO_XY_SPIN);

    for (int chain = 0; chain < N_CHAINS; ++chain) {
        for (int i = 0; i < this->L; ++i) {
            hostdata.sequence_on_lattice(
                chain, hostdata.lattice_nodes_positions(chain, i)) =
                distribution_theta(generators_theta);
        }
    }
}

template<int Dim>
void XY_LI_normalize<Dim>::start_kernel_energy_init() {
    XY_LI_normalize_EnergyOp op{ this->devicedata, this->power_ij };
    this->energy_init(op);
}