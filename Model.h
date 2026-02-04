#ifndef INTERACTION_SAW_MODELS_MODEL_H
#define INTERACTION_SAW_MODELS_MODEL_H

#include<iostream>
#include <random>
#include <fstream>
#include <chrono>

#include <Kokkos_Random.hpp>

#include"common.h"
#include"lattice.h"

#ifndef N_CHAINS
#define N_CHAINS 16
#endif

template<class ExecSpace, class T>
struct DeviceData {

  using Layout = Kokkos::LayoutRight; 

  //LATTICE: 
    Kokkos::View<int*, ExecSpace> map_of_contacts_int;
    Kokkos::View<int*, ExecSpace> inverse_steps;
    Kokkos::View<int, ExecSpace> ndim2;


  //MODEL SCALARS:
    Kokkos::View<int, ExecSpace> L;
    Kokkos::View<int, ExecSpace> lattice_side_device;
    

  //MODEl ARRAYS 
  Kokkos::View<T **, Layout, ExecSpace> sequence_on_lattice;
  Kokkos::View<int **, Layout, ExecSpace> next_monomers;
  Kokkos::View<int **, Layout, ExecSpace> previous_monomers;
  Kokkos::View<int **, Layout, ExecSpace> directions;
  Kokkos::View<int **, Layout, ExecSpace> lattice_nodes_positions;

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

  //helpers ?
  Kokkos::View<int*, ExecSpace> i_index;
  Kokkos::View<int*, ExecSpace> j_index;
  Kokkos::View<int, ExecSpace> N_pairs;
};

template<class ExecSpace, class T, class EnergyOp, class SpinProposalOp, int Dim>
struct MetropolisKernel;
template<class ExecSpace, class T, class EnergyOp, int Dim>
struct EnergyInitKernel;


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

template<class T, int Dim>
class SAW_model : public Model {
public:
    SAW_model<T, Dim>(int L,  float Jmin = 0.25, float Jmax = 0.26);

    // SAW_model(int L) : Model(L) {};
    virtual void spin_init_random() = 0;
    virtual void start_kernel_energy_init() = 0;

    void geometry_initialization_arrays();

    void geometry_initialization_stick();

    void scalars_MC_preparation(float Jmin, float Jmax);


    void HostDataInit();
    void DeviceDataInit ();

    //HostData hostdata;
    DeviceData<Kokkos::CudaSpace, T> devicedata;
    DeviceData<Kokkos::HostSpace, T> hostdata;


    template<class ExecSpace, class EnergyOp, class SpinProposalOp>
struct MetropolisKernel {
  using member_type = typename Kokkos::TeamPolicy<ExecSpace>::member_type;
  DeviceData<typename ExecSpace::memory_space, T> d;
  EnergyOp energy;       // model-specific
  SpinProposalOp propose_spin;        // model-specific
  Kokkos::Random_XorShift64_Pool<ExecSpace> pool;

  long long n_iters;
  long long epoch1;

  KOKKOS_INLINE_FUNCTION
  void operator()(const member_type& team) const {
    const int c = team.league_rank();

    for (long long step = 1; step <= n_iters; ++step) {
      Kokkos::single(Kokkos::PerTeam(team), [&](){
        d.flipMoveType(c) = rand_chain_step(12345, c, step * epoch1, 0);
      });
      team.team_barrier();

      // Propose geometry (common)
      if (d.flipMoveType(c) < 0.5f) {
        hierarchicalFlipMoveAddEnd(team, d, c, pool, propose_spin);
      } else {
        hierarchicalFlipMoveAddStart(team, d, c, pool, propose_spin);
      }
      team.team_barrier();
      const int geom_ok = d.accept_move(c);

      if (geom_ok) {
        // Model-specific ΔE (your “only varying piece”)
        energy.delta_energy(team, d, c); // write d.d_E_1(c) or return dE
      }
      team.team_barrier();

      if (geom_ok) {
        const double u = rand_chain_step(12345, c, step * epoch1, 3);

        // Accept/reject & commit (common)
        if (d.flipMoveType(c) < 0.5f) {
          hierarchicalOneKernel_AddEnd_FirstPart(team, d, c, pool, u);
        } else {
          hierarchicalOneKernel_AddStart_FirstPart(team, d, c, pool, u);
        }
      }
      team.team_barrier();          
    }

    SAW_model<T, Dim>::template hierarchicalOneKernel_Reconnect(team, d, c, pool);
    team.team_barrier();
  }
};


    template<class EnergyOp>
    void energy_init(EnergyOp energy) {
      using ExecSpace   = Kokkos::Cuda;
      using team_policy = Kokkos::TeamPolicy<ExecSpace>;

      //team_policy policy(N_CHAINS, Kokkos::AUTO());
      team_policy policy(N_CHAINS, 500, 1);

      auto d = devicedata;

      Kokkos::parallel_for("EnergyInit",
        policy,
        EnergyInitKernel<ExecSpace, T, EnergyOp, Dim>{d, energy}
      );
      Kokkos::fence();

      
    }

    
    KOKKOS_INLINE_FUNCTION
    static void hierarchicalOneKernel_Reconnect (
      const Kokkos::TeamPolicy<Kokkos::Cuda>::member_type& team_member,
      const DeviceData<Kokkos::CudaSpace, float>& flip_data_local,
      int chain, 
      const Kokkos::Random_XorShift64_Pool<Kokkos::Cuda> & pool) {
        Kokkos::single(Kokkos::PerTeam(team_member), [&]() {
          auto rand_gen =  pool.get_state(); //flip_data_local.rand_pool.get_state();
          flip_data_local.direction(chain)  = rand_gen.urand64() % 6;
          pool.free_state(rand_gen);
  
          int  step_coord = flip_data_local.map_of_contacts_int(flip_data_local.ndim2() * flip_data_local.end_conformation(chain) + flip_data_local.direction(chain) );
  
          int c = 0; //
   
          // test self avoidance condition
          if (flip_data_local.sequence_on_lattice(chain, step_coord) == NO_XY_SPIN ||
              flip_data_local.next_monomers(chain, step_coord) == NO_SAW_NODE ||
              step_coord == flip_data_local.previous_monomers(chain, flip_data_local.end_conformation(chain))) {
              return;
          }
  
      int new_end = flip_data_local.next_monomers(chain, step_coord);
      flip_data_local.next_monomers(chain, step_coord) = flip_data_local.end_conformation(chain);
      //need to check inverse steps 
      flip_data_local.directions(chain, step_coord) = flip_data_local.inverse_steps(flip_data_local.direction(chain));
      c = flip_data_local.end_conformation(chain);
      int new_c;
      while (c != new_end) {
          new_c = flip_data_local.previous_monomers(chain, c);
          flip_data_local.next_monomers(chain, c) = flip_data_local.previous_monomers(chain, c);
          flip_data_local.directions(chain, c) = flip_data_local.inverse_steps(flip_data_local.directions(chain, new_c) );
          c = new_c;
      }
      int temp_prev_next = flip_data_local.next_monomers(chain, new_end);
      flip_data_local.previous_monomers(chain, flip_data_local.end_conformation(chain)) = step_coord;
      c = flip_data_local.end_conformation(chain);
      while (c != new_end) {
          new_c = flip_data_local.next_monomers(chain, c);
          flip_data_local.previous_monomers(chain, new_c) = c;
          c = new_c;
      }
      flip_data_local.end_conformation(chain) = new_end;
      flip_data_local.previous_monomers(chain, new_end) = temp_prev_next;
      flip_data_local.next_monomers(chain, new_end) = NO_SAW_NODE;
      flip_data_local.directions(chain, new_end) = NO_SAW_NODE;
  
      flip_data_local.lattice_nodes_positions(chain, 0) = flip_data_local.start_conformation(chain);
      c = flip_data_local.next_monomers(chain, flip_data_local.start_conformation(chain));
      for (int i = 1; i < flip_data_local.L(); i++) {
          flip_data_local.lattice_nodes_positions(chain, i) = c;
          c = flip_data_local.next_monomers(chain, c);
      }
      flip_data_local.start_index_in_nodes_position(chain) = 0;
      //Redefine positions in array now 
      });
      }
    

    template<class SpinProposalOp>
    KOKKOS_INLINE_FUNCTION
    static void hierarchicalFlipMoveAddEnd(
      const Kokkos::TeamPolicy<Kokkos::Cuda>::member_type& team_member,
      const DeviceData<Kokkos::CudaSpace, float>& flip_data_local,
      int c, 
      const Kokkos::Random_XorShift64_Pool<Kokkos::Cuda> & pool,
      const SpinProposalOp& propose_spin)  {
        Kokkos::single(Kokkos::PerTeam(team_member), [&]() {
          // Example random usage
  
          auto rand_gen =  pool.get_state();  
          int dir = rand_gen.urand64() % 6;
          pool.free_state(rand_gen);
          flip_data_local.direction(c) = dir;
  
          int new_point = flip_data_local.map_of_contacts_int(flip_data_local.ndim2() * flip_data_local.end_conformation(c) + dir);
  
          // Check self-avoid
          if (flip_data_local.sequence_on_lattice(c, new_point) != NO_XY_SPIN) {
              //accept_move = false;
              flip_data_local.accept_move(c) = 0;
              return;  // skip the rest
          }
          flip_data_local.accept_move(c) = 1;
         // auto rand_gen1 = pool.get_state();
          flip_data_local.spinValue(c) = propose_spin(pool);
        //  pool.free_state(rand_gen1);
          flip_data_local.oldspin(c) = flip_data_local.sequence_on_lattice(c, flip_data_local.start_conformation(c));
           
  
          // delete the beginning of SAW
          flip_data_local.save_start_conformation(c) = flip_data_local.start_conformation(c);
          flip_data_local.start_conformation(c) = flip_data_local.next_monomers(c, flip_data_local.start_conformation(c));
          flip_data_local.next_monomers(c, flip_data_local.save_start_conformation(c)) = NO_SAW_NODE;
          flip_data_local.previous_monomers(c, flip_data_local.start_conformation(c)) = NO_SAW_NODE;
          flip_data_local.sequence_on_lattice(c, flip_data_local.save_start_conformation(c)) = NO_XY_SPIN;
  
  
          flip_data_local.oldIndex(c) = flip_data_local.save_start_conformation(c);
          flip_data_local.newIndex(c) = new_point;
  
          //add the new monomer at the end of SAW
          flip_data_local.next_monomers(c, flip_data_local.end_conformation(c)) = new_point;
          flip_data_local.sequence_on_lattice(c, new_point) = flip_data_local.spinValue(c); //new spin value
          flip_data_local.previous_monomers(c, new_point) = flip_data_local.end_conformation(c);
          flip_data_local.end_conformation(c) = new_point;
  
          int position_new = flip_data_local.start_index_in_nodes_position(c) ;
  
          flip_data_local.lattice_nodes_positions(c, position_new) = flip_data_local.end_conformation(c);
      });
    }


    template<class SpinProposalOp>
    KOKKOS_INLINE_FUNCTION
    static void hierarchicalFlipMoveAddStart(
      const Kokkos::TeamPolicy<Kokkos::Cuda>::member_type& team_member,
      const DeviceData<Kokkos::CudaSpace, float>& flip_data_local,
      int c, 
      const Kokkos::Random_XorShift64_Pool<Kokkos::Cuda> & pool,
      const SpinProposalOp& propose_spin) {
        Kokkos::single(Kokkos::PerTeam(team_member), [&]() {
          // Example random usage
          auto rand_gen =  pool.get_state();  
          flip_data_local.direction(c)  = rand_gen.urand64() % 6;
          pool.free_state(rand_gen);
  
          int new_point = flip_data_local.map_of_contacts_int(flip_data_local.ndim2() * flip_data_local.start_conformation(c) + flip_data_local.direction(c) );
          flip_data_local.oldspin(c) = flip_data_local.sequence_on_lattice(c, flip_data_local.end_conformation(c));
  
          if (flip_data_local.sequence_on_lattice(c, new_point) != NO_XY_SPIN)  {
              flip_data_local.accept_move(c) = 0; // Set the flag to indicate rejection
              return;
          }
          flip_data_local.accept_move(c) = 1;
      //    auto rand_gen1 = pool.get_state();
          flip_data_local.spinValue(c) = propose_spin(pool);
       //   pool.free_state(rand_gen1);
          //delete end
          flip_data_local.save_end_conformation(c) = flip_data_local.end_conformation(c);
          flip_data_local.end_conformation(c) = flip_data_local.previous_monomers(c, flip_data_local.end_conformation(c));
          flip_data_local.previous_monomers(c, flip_data_local.save_end_conformation(c)) = NO_SAW_NODE;
          flip_data_local.next_monomers(c, flip_data_local.end_conformation(c)) = NO_SAW_NODE;
          flip_data_local.sequence_on_lattice(c, flip_data_local.save_end_conformation(c)) = NO_XY_SPIN;
  
  
          flip_data_local.oldIndex(c) = flip_data_local.save_end_conformation(c);
          flip_data_local.newIndex(c) =  new_point;
  
          //add the new beginning
          flip_data_local.previous_monomers(c, flip_data_local.start_conformation(c)) = new_point;
          flip_data_local.sequence_on_lattice(c, new_point) = flip_data_local.spinValue(c); //выбор спина
          flip_data_local.next_monomers(c, new_point) = flip_data_local.start_conformation(c);
          flip_data_local.start_conformation(c) = new_point;
  
          int position_new = (flip_data_local.start_index_in_nodes_position(c) + flip_data_local.L() - 1) % flip_data_local.L() ;
          //if (position_new == -1 ) position_new = flip_data_local.L - 1;
          flip_data_local.lattice_nodes_positions(c, position_new) = flip_data_local.start_conformation(c);
  
      });

    }

    KOKKOS_INLINE_FUNCTION
    static void hierarchicalOneKernel_AddEnd_FirstPart(
      const Kokkos::TeamPolicy<Kokkos::Cuda>::member_type& team_member,
      const DeviceData<Kokkos::CudaSpace, float>& flip_data_local,
      int c, 
      const Kokkos::Random_XorShift64_Pool<Kokkos::Cuda> & pool, 
      float q_ifaccept)
    {
      Kokkos::single(Kokkos::PerTeam(team_member), [&]() {
 
        float p1 = exp( -(flip_data_local.J_chain(c) * (  flip_data_local.d_E_1(c) )) );
      
        float p_metropolis = (p1 < 1.0) ? p1 : 1.0;

        if (q_ifaccept < p_metropolis) {

            flip_data_local.sequence_on_lattice(c, flip_data_local.save_start_conformation(c)) = NO_XY_SPIN;
            flip_data_local.directions(c, flip_data_local.save_start_conformation(c)) = NO_SAW_NODE;
            flip_data_local.directions(c, flip_data_local.previous_monomers(c, flip_data_local.end_conformation(c))) = flip_data_local.direction(c);
            flip_data_local.start_index_in_nodes_position(c) = (flip_data_local.start_index_in_nodes_position(c) + 1) % flip_data_local.L();

            flip_data_local.E(c) += flip_data_local.d_E_1(c);

            //if (c==2) printf("newE = %f; d_E = %f\n",  flip_data_local.E(c),  flip_data_local.d_E_1(c));
        } else {
            // reject => revert
            int del = flip_data_local.end_conformation(c);
            flip_data_local.end_conformation(c) = flip_data_local.previous_monomers(c, flip_data_local.end_conformation(c));
            flip_data_local.next_monomers(c, flip_data_local.end_conformation(c)) = NO_SAW_NODE;
            flip_data_local.previous_monomers(c, del) = NO_SAW_NODE;
            flip_data_local.sequence_on_lattice(c, del) = NO_XY_SPIN;

            //add the previous beginning
            flip_data_local.previous_monomers(c, flip_data_local.start_conformation(c)) = flip_data_local.save_start_conformation(c);
            flip_data_local.next_monomers(c, flip_data_local.save_start_conformation(c)) = flip_data_local.start_conformation(c);
            flip_data_local.start_conformation(c) = flip_data_local.save_start_conformation(c);
            flip_data_local.sequence_on_lattice(c, flip_data_local.start_conformation(c)) = flip_data_local.oldspin(c);

            flip_data_local.lattice_nodes_positions(c, flip_data_local.start_index_in_nodes_position(c)) = flip_data_local.start_conformation(c);

        }
    });


  }



  
    KOKKOS_INLINE_FUNCTION
    static void hierarchicalOneKernel_AddStart_FirstPart(
      const Kokkos::TeamPolicy<Kokkos::Cuda>::member_type& team_member,
      const DeviceData<Kokkos::CudaSpace, float>& flip_data_local,
      int c, 
      const Kokkos::Random_XorShift64_Pool<Kokkos::Cuda> & pool, 
      float q_ifaccept) 
    {


      Kokkos::single(Kokkos::PerTeam(team_member), [&]() {
        float p1 = exp(-(flip_data_local.J_chain(c) * (  flip_data_local.d_E_1(c)   )));
        float p_metropolis = Kokkos::min(1.0f, p1);
        if (q_ifaccept < p_metropolis) {
            flip_data_local.sequence_on_lattice(c, flip_data_local.save_end_conformation(c)) = NO_XY_SPIN;
            flip_data_local.directions(c, flip_data_local.end_conformation(c)) = NO_SAW_NODE;
            flip_data_local.directions(c, flip_data_local.start_conformation(c)) = flip_data_local.inverse_steps(flip_data_local.direction(c));
            // new start is the new added value
            int position_new = (flip_data_local.start_index_in_nodes_position(c) + flip_data_local.L () - 1) % flip_data_local.L() ;
            flip_data_local.start_index_in_nodes_position(c) = position_new;

            flip_data_local.E(c) += flip_data_local.d_E_1(c);

           // if (c==2) printf("newE = %f; d_E = %f\n",  flip_data_local.E(c),  flip_data_local.d_E_1(c));

        }
        else {
            //reject the new state
            //delete starte
            int del = flip_data_local.start_conformation(c);
            flip_data_local.start_conformation(c) = flip_data_local.next_monomers(c, flip_data_local.start_conformation(c));
            flip_data_local.previous_monomers(c, flip_data_local.start_conformation(c)) = NO_SAW_NODE;
            flip_data_local.next_monomers(c, del) = NO_SAW_NODE;
            flip_data_local.sequence_on_lattice(c, del) = NO_XY_SPIN;

            //readd the end of the saw
            flip_data_local.next_monomers(c, flip_data_local.end_conformation(c)) = flip_data_local.save_end_conformation(c);
            flip_data_local.previous_monomers(c, flip_data_local.save_end_conformation(c)) = flip_data_local.end_conformation(c);
            flip_data_local.end_conformation(c) = flip_data_local.save_end_conformation(c);
            flip_data_local.sequence_on_lattice(c, flip_data_local.end_conformation(c)) = flip_data_local.oldspin(c);

            int position_new = (flip_data_local.start_index_in_nodes_position(c) + flip_data_local.L() - 1) % flip_data_local.L() ;

            flip_data_local.lattice_nodes_positions(c, position_new) = flip_data_local.end_conformation(c);
        }

    });


  }


    void parallel_tempering_swap()
    {
      static long long exch_id = 0;
    
      // 1) Copy J_chain to host and sort indices by J
      auto J_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), devicedata.J_chain);
    
      std::vector<int> ord(N_CHAINS);
      std::iota(ord.begin(), ord.end(), 0);
      std::sort(ord.begin(), ord.end(),
                [&](int a, int b){ return J_host(a) < J_host(b); });
    
      // 2) Build disjoint neighbor pairs in that sorted order and attempt exchanges
      auto do_parity = [&](int parity) {
        const int first = parity ? 1 : 0;
        const int num_pairs = (N_CHAINS - first) / 2;
        if (num_pairs <= 0) return;
    
        // Host buffers of replica indices to swap
        std::vector<int> h_i(num_pairs), h_j(num_pairs);
        for (int k = 0; k < num_pairs; ++k) {
          h_i[k] = ord[first + 2*k];
          h_j[k] = ord[first + 2*k + 1];
        }
    
        // Copy pairs to device
        Kokkos::View<int*, Kokkos::CudaSpace> d_i("pt_pair_i", num_pairs);
        Kokkos::View<int*, Kokkos::CudaSpace> d_j("pt_pair_j", num_pairs);
    
        auto hdi = Kokkos::create_mirror_view(d_i);
        auto hdj = Kokkos::create_mirror_view(d_j);
        for (int k = 0; k < num_pairs; ++k) { hdi(k) = h_i[k]; hdj(k) = h_j[k]; }
        Kokkos::deep_copy(d_i, hdi);
        Kokkos::deep_copy(d_j, hdj);
    
        attempt_exchanges_on_pairs(devicedata, d_i, d_j, exch_id++);
        Kokkos::fence();
      };
    
      do_parity(0);
      do_parity(1);
    }



    static void attempt_exchanges_on_pairs(
      DeviceData<Kokkos::CudaSpace, T> d,
      Kokkos::View<int*, Kokkos::CudaSpace> pair_i,
      Kokkos::View<int*, Kokkos::CudaSpace> pair_j,
      long long exch_id)
  {
    using ExecSpace = Kokkos::Cuda;
    const int num_pairs = pair_i.extent_int(0);

    Kokkos::parallel_for(
      "PT_exchange_pairs",
      Kokkos::RangePolicy<ExecSpace>(0, num_pairs),
      KOKKOS_LAMBDA(const int k) {
        const int i = pair_i(k);
        const int j = pair_j(k);

        const float Ei = d.E(i);        // energy WITHOUT J
        const float Ej = d.E(j);
        const float Ji = d.J_chain(i);  // beta (or coupling)
        const float Jj = d.J_chain(j);

        // Standard replica-exchange acceptance:
        // acc = min(1, exp((Ji - Jj)*(Ei - Ej)))
        const float expo = (Ji - Jj) * (Ei - Ej);
        const float acc  = (expo >= 0.f) ? 1.f : expf(expo);

        // Deterministic RNG like your old code
        const double u = rand_chain_step(77777ull, i, exch_id, /*stream*/ 7);

        if (u < acc) {
          d.J_chain(i) = Jj;
          d.J_chain(j) = Ji;
        }
      }
    );
  }




  template<class EnergyOp, class SpinProposalOp>
  void runMCMCOnDevice_impl(EnergyOp energy, SpinProposalOp propose, long long MC_STEPS, long long epoch) {
    using ExecSpace   = Kokkos::Cuda;
    using team_policy = Kokkos::TeamPolicy<ExecSpace>;

    static long long epoch2 = 2;
    static bool pool_initialized = false;
    static Kokkos::Random_XorShift64_Pool<ExecSpace> my_pool(12345);

    if (!pool_initialized) {
      my_pool.init(12345, 10 * N_CHAINS);
      pool_initialized = true;
    }

    //team_policy policy(N_CHAINS, Kokkos::AUTO());
    team_policy policy(N_CHAINS, 500, 1);

    auto d = devicedata;
    auto pool = my_pool;

    Kokkos::parallel_for("MCMC_on_device", policy,
      MetropolisKernel<ExecSpace, EnergyOp, SpinProposalOp>{d, energy, propose, pool, MC_STEPS, epoch2 }
    );
    Kokkos::fence();
    epoch2 += 1;
  }


  void update_host_for_output() {
    Kokkos::deep_copy(hostdata.J_chain, devicedata.J_chain);
    Kokkos::deep_copy(hostdata.E, devicedata.E);
    Kokkos::deep_copy(hostdata.start_index_in_nodes_position, devicedata.start_index_in_nodes_position);

    Kokkos::deep_copy(hostdata.lattice_nodes_positions, devicedata.lattice_nodes_positions);
    Kokkos::deep_copy(hostdata.sequence_on_lattice, devicedata.sequence_on_lattice);
  }
  void out_angle_data(std::ostream &out, long long n_steps) {

    update_host_for_output();

    for (int c = 0; c < N_CHAINS; ++c) {
      out << n_steps << " " << hostdata.J_chain(c) << " " << hostdata.E(c) << " ";

      for (int e = 0; e < L; ++e) {
        const int pos = hostdata.lattice_nodes_positions(c, e);
        out << hostdata.sequence_on_lattice(c, pos) << " ";
      }
      out << "\n";
    }
  }  


  void out_dir_data(std::ostream &out, long long n_steps) {
    update_host_for_output();

    const int ls = lattice->lattice_size();

    for (int c = 0; c < N_CHAINS; ++c) {
      out << n_steps << " " << hostdata.J_chain(c) << " "
          << hostdata.start_index_in_nodes_position(c) << " "
          << hostdata.E(c) << " ";

      for (int i = 0; i < L; ++i) {
        const int pos = hostdata.lattice_nodes_positions(c, i);

        const int x = pos % ls;

        if constexpr (Dim == 2) {
          const int y = pos / ls;
          out << x << " " << y << " ";
        } else { // Dim == 3
          const int y = (pos / ls) % ls;
          const int z = pos / (ls * ls);
          out << x << " " << y << " " << z << " ";
        }
      }
      out << "\n";
    }
  }

  // Convenience: append to file (angles)
  void append_angle_file(const std::string& filename, long long n_steps) {
    std::ofstream out(filename, std::ios::app);
    if (!out) throw std::runtime_error("Cannot open angles log file: " + filename);
    out_angle_data(out, n_steps);
  }

  // Convenience: append to file (dirs)
  void append_dir_file(const std::string& filename, long long n_steps) {
    std::ofstream out(filename, std::ios::app);
    if (!out) throw std::runtime_error("Cannot open dirs log file: " + filename);
    out_dir_data(out, n_steps);
  }
 
 
};

template<int Dim>
class XY_LI : public SAW_model<float, Dim> {
public:
    using Base = SAW_model<float, Dim>;
    using Base::hostdata;
    using Base::devicedata;
    using Base::lattice;

    XY_LI (int L,float Jmin = 0.25, float Jmax = 0.26);

    void spin_init_random();
    void start_kernel_energy_init() override;

    struct XY_LI_EnergyOp {
      using ExecSpace = Kokkos::Cuda;
      using member_type = typename Kokkos::TeamPolicy<ExecSpace>::member_type;
    
      DeviceData<Kokkos::CudaSpace, float> d;

      KOKKOS_INLINE_FUNCTION float r2(int a, int b, int side) const {
        if constexpr (Dim == 2) return radius_sq_2d(a,b,side);
        else                   return radius_sq_3d(a,b,side);
      }
    
      KOKKOS_INLINE_FUNCTION
      void delta_energy(const member_type& team,
                         const DeviceData<Kokkos::CudaSpace, float>& flip_data,
                         int c) const
      {
        int pos_j  = flip_data.oldIndex(c);
        float theta_j = flip_data.oldspin(c);
        int pos_k  = flip_data.newIndex(c);
        float theta_k = flip_data.sequence_on_lattice(c, pos_k);
        Kokkos::parallel_reduce(
                Kokkos::TeamThreadRange(team,  flip_data.L() ),
                [&](const int i, float &H_total) {
    
                    int pos_i = flip_data.lattice_nodes_positions(c, i);
                    if ((pos_i ==  flip_data.oldIndex(c)) || (pos_i == flip_data.newIndex(c)) ) return; 
    
                        float theta_i  = flip_data.sequence_on_lattice(c, pos_i);
                        float r_val   = r2(pos_i, pos_j, flip_data.lattice_side_device());
                        r_val = Kokkos::sqrt(r_val) * r_val; 
                        H_total += Kokkos::cos(theta_i - theta_j) / r_val;
    
                        r_val   = r2(pos_i, pos_k , flip_data.lattice_side_device());
                        r_val = Kokkos::sqrt(r_val) * r_val;
                        H_total -= Kokkos::cos(theta_i - theta_k) / r_val;
    
                },
               flip_data.d_E_1(c)
        );
      }

      KOKKOS_INLINE_FUNCTION
      void energy(const member_type& team,
                         const DeviceData<Kokkos::CudaSpace, float>& flip_data,
                         int c) const {
              Kokkos::parallel_reduce(
                Kokkos::TeamThreadRange(team,  flip_data.N_pairs() ),
                [&](const int ind, float &H_total) {
                        int i = flip_data.i_index(ind);
                        int j = flip_data.j_index(ind);
                          int pos_i   = flip_data.lattice_nodes_positions(c, i);
                        float theta_i  = flip_data.sequence_on_lattice(c, pos_i);
                          int pos_j  = flip_data.lattice_nodes_positions(c, j);
                        float theta_j = flip_data.sequence_on_lattice(c, pos_j);
                        float r_val   = r2(pos_i, pos_j, flip_data.lattice_side_device());
                        r_val = Kokkos::sqrt(r_val) * Kokkos::sqrt(r_val) * Kokkos::sqrt(r_val);
                        H_total -= Kokkos::cos(theta_i - theta_j) / r_val;
                },
                flip_data.newE(c)
        );
      }


    };

    void runMCMCOnDevice(long long MC_STEPS = 10000, long long epoch = 1000) {
      XY_LI_EnergyOp op{ this->devicedata };
      XYSpinProposal  propose{}; 
      this->runMCMCOnDevice_impl(op, propose, MC_STEPS, epoch);
      //this->parallel_tempering_swap();
    } 
};

template<int Dim>
class XY_SI : public SAW_model<float, Dim> {
public:

};

template<int Dim>
class Ising_SI : public SAW_model<int, Dim> {
public:

};

template<int Dim>
class Ising_LI : public SAW_model<int, Dim> {
public:

};

// Single source of truth list (add fields once, reuse everywhere)
#define DATA_FIELDS(X) \
  X(map_of_contacts_int) \
  X(inverse_steps) \
  X(ndim2) \
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
  X(J_chain) \
  X(i_index) \
  X(j_index) \
  X(N_pairs)
  
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


template<class T, int Dim>
void SAW_model<T,Dim>::HostDataInit() {
    hostdata.map_of_contacts_int = lattice->map_of_contacts_int;
    hostdata.inverse_steps       = lattice->inverse_steps;

    hostdata.L = L_host;
    hostdata.lattice_side_device = lattice->lattice_side_host;

    hostdata.ndim2 = Kokkos::View<int, Kokkos::HostSpace>("ndim2");
    hostdata.ndim2() = lattice->ndim2();

}

template<class T, int Dim>
void SAW_model<T,Dim>::DeviceDataInit() {
    upload_all<Kokkos::CudaSpace, T>(hostdata, devicedata);

    start_kernel_energy_init();

    auto E_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), devicedata.E);
    auto L_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), devicedata.L);
    auto Np_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), devicedata.N_pairs);
    auto side_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), devicedata.lattice_side_device);
    auto ndim2_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), devicedata.ndim2);
    
    std::cout << "DEVICE scalars: L=" << L_h()
              << " N_pairs=" << Np_h()
              << " side=" << side_h()
              << " ndim2=" << ndim2_h() << "\n";
    
    for (int c = 0; c < N_CHAINS; ++c) {
      std::cout << "E[" << c << "]=" << E_h(c) << "\n";
    }    
}

template<class ExecSpace, class T, class EnergyOp, int Dim>
struct EnergyInitKernel {
  using member_type = typename Kokkos::TeamPolicy<ExecSpace>::member_type;
  DeviceData<typename ExecSpace::memory_space, T> d;
  EnergyOp energy;

  KOKKOS_INLINE_FUNCTION
  void operator()(const member_type& team) const {
    const int c = team.league_rank();
    d.newE(c) = 0.0f;
    energy.energy(team, d, c);
    d.E(c) = d.newE(c);
  }
};

template<>
struct SpinProposalTraits<float> {
  using type = XYSpinProposal;
};


#endif