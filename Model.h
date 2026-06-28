#ifndef INTERACTION_SAW_MODELS_MODEL_H
#define INTERACTION_SAW_MODELS_MODEL_H

#include<iostream>
#include <random>
#include <fstream>
#include <chrono>

#include <Kokkos_Random.hpp>

#include"common.h"
#include"lattice.h"

enum {
  CHK_OK = 0,
  CHK_OOB = 1,
  CHK_DUP = 2,
  CHK_ADJ_MAP = 3,
  CHK_NEXT_PREV = 4,
  CHK_DIR = 5,
  CHK_ADJ_COORD = 6,
  CHK_WRAP_STEP = 7
};

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


// template class Model<2>;
// template class Model<3>;

template<int Dim>
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


static inline long long nsites_from_ls(int ls, int Dim) {
  long long n = 1;
  for (int k = 0; k < Dim; ++k) n *= (long long)ls;
  return n;
}

template<class T, int Dim>
class SAW_model : public Model<Dim> {
public:
    SAW_model<T, Dim>(int L,  float Jmin = 0.25, float Jmax = 0.26);

    // SAW_model(int L) : Model(L) {};
    virtual void spin_init_random() = 0;
    virtual void start_kernel_energy_init() = 0;

    void geometry_initialization_arrays();

    void geometry_initialization_stick();

    void geometry_initialization_half();

    void scalars_MC_preparation(float Jmin, float Jmax);


    void HostDataInit();
    void DeviceDataInit ();

    //HostData hostdata;
    DeviceData<Kokkos::CudaSpace, T> devicedata;
    DeviceData<Kokkos::HostSpace, T> hostdata;

    KOKKOS_INLINE_FUNCTION static float r2(int a, int b, int side) {
      if constexpr (Dim == 2) return radius_sq_2d(a,b,side);
      else                   return radius_sq_3d(a,b,side);
    }

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
        d.flipMoveType(c) = rand_chain_step(12345, c, n_iters * epoch1 + step, 0);
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
        const double u = rand_chain_step(12345, c, n_iters * epoch1 + step, 3);

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
      const DeviceData<Kokkos::CudaSpace, T>& flip_data_local,
      int chain, 
      const Kokkos::Random_XorShift64_Pool<Kokkos::Cuda> & pool) {
        Kokkos::single(Kokkos::PerTeam(team_member), [&]() {
          auto rand_gen =  pool.get_state(); //flip_data_local.rand_pool.get_state();
          flip_data_local.direction(chain)  = rand_gen.urand64() % flip_data_local.ndim2();
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
      const DeviceData<Kokkos::CudaSpace, T>& flip_data_local,
      int c, 
      const Kokkos::Random_XorShift64_Pool<Kokkos::Cuda> & pool,
      const SpinProposalOp& propose_spin)  {
        Kokkos::single(Kokkos::PerTeam(team_member), [&]() {
          // Example random usage
  
          auto rand_gen =  pool.get_state();  
          int dir = rand_gen.urand64() % flip_data_local.ndim2();
          pool.free_state(rand_gen);
          flip_data_local.direction(c) = dir;
  
          int to_remove = flip_data_local.start_conformation(c);  // old start
          int new_point = flip_data_local.map_of_contacts_int(flip_data_local.ndim2() * flip_data_local.end_conformation(c) + dir);
  
          // Check self-avoid
          if (new_point != to_remove && flip_data_local.sequence_on_lattice(c, new_point) != NO_XY_SPIN) {
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
      const DeviceData<Kokkos::CudaSpace, T>& flip_data_local,
      int c, 
      const Kokkos::Random_XorShift64_Pool<Kokkos::Cuda> & pool,
      const SpinProposalOp& propose_spin) {
        Kokkos::single(Kokkos::PerTeam(team_member), [&]() {
          // Example random usage
          auto rand_gen =  pool.get_state();  
          flip_data_local.direction(c)  = rand_gen.urand64() % flip_data_local.ndim2();
          pool.free_state(rand_gen);
  
          int to_remove = flip_data_local.end_conformation(c);    // old end
          int new_point = flip_data_local.map_of_contacts_int(flip_data_local.ndim2() * flip_data_local.start_conformation(c) + flip_data_local.direction(c) );
          flip_data_local.oldspin(c) = flip_data_local.sequence_on_lattice(c, flip_data_local.end_conformation(c));
  
          if (new_point != to_remove && flip_data_local.sequence_on_lattice(c, new_point) != NO_XY_SPIN)  {
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
      const DeviceData<Kokkos::CudaSpace, T>& flip_data_local,
      int c, 
      const Kokkos::Random_XorShift64_Pool<Kokkos::Cuda> & pool, 
      float q_ifaccept)
    {
      Kokkos::single(Kokkos::PerTeam(team_member), [&]() {
 
        float p1 = exp( -(flip_data_local.J_chain(c) * (  flip_data_local.d_E_1(c) )) );
      
        float p_metropolis = (p1 < 1.0) ? p1 : 1.0;

        if (q_ifaccept < p_metropolis) {
            const bool recycled = (flip_data_local.newIndex(c) == flip_data_local.oldIndex(c));

            if (!recycled) {
              flip_data_local.sequence_on_lattice(c, flip_data_local.save_start_conformation(c)) = NO_XY_SPIN;
              flip_data_local.directions(c, flip_data_local.save_start_conformation(c)) = NO_SAW_NODE;
            }
            // still do:
            
            flip_data_local.directions(c, flip_data_local.previous_monomers(c, flip_data_local.end_conformation(c))) = flip_data_local.direction(c);
            flip_data_local.start_index_in_nodes_position(c) = (flip_data_local.start_index_in_nodes_position(c) + 1) % flip_data_local.L();

            flip_data_local.E(c) += flip_data_local.d_E_1(c);

          } 
          else {
            // reject => revert
            const bool recycled = (flip_data_local.newIndex(c) == flip_data_local.oldIndex(c));


            int del = flip_data_local.end_conformation(c);
            flip_data_local.end_conformation(c) = flip_data_local.previous_monomers(c, flip_data_local.end_conformation(c));
            flip_data_local.next_monomers(c, flip_data_local.end_conformation(c)) = NO_SAW_NODE;
            flip_data_local.next_monomers(c, del)     = NO_SAW_NODE;
            flip_data_local.directions(c, flip_data_local.end_conformation(c)) = NO_SAW_NODE;
            flip_data_local.previous_monomers(c, del) = NO_SAW_NODE;
            flip_data_local.sequence_on_lattice(c, del) = NO_XY_SPIN;
            // only wipe direction on del if it's not the recycled start node
            if (!recycled) {
              flip_data_local.directions(c, del) = NO_SAW_NODE;
            }

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
      const DeviceData<Kokkos::CudaSpace, T>& flip_data_local,
      int c, 
      const Kokkos::Random_XorShift64_Pool<Kokkos::Cuda> & pool, 
      float q_ifaccept) 
    {


      Kokkos::single(Kokkos::PerTeam(team_member), [&]() {
        float p1 = exp(-(flip_data_local.J_chain(c) * (  flip_data_local.d_E_1(c)   )));
        float p_metropolis = Kokkos::min(1.0f, p1);
        if (q_ifaccept < p_metropolis) {

            const bool recycled = (flip_data_local.newIndex(c) == flip_data_local.oldIndex(c));

            if (!recycled) {
              flip_data_local.sequence_on_lattice(c, flip_data_local.save_end_conformation(c)) = NO_XY_SPIN;
              flip_data_local.directions(c, flip_data_local.save_end_conformation(c)) = NO_SAW_NODE; // optional hygiene
            }
            
            flip_data_local.directions(c, flip_data_local.start_conformation(c)) = flip_data_local.inverse_steps(flip_data_local.direction(c));
            // new start is the new added value
            int position_new = (flip_data_local.start_index_in_nodes_position(c) + flip_data_local.L () - 1) % flip_data_local.L() ;
            flip_data_local.start_index_in_nodes_position(c) = position_new;

            flip_data_local.E(c) += flip_data_local.d_E_1(c);
 
        }
        else {
            //reject the new state
            //delete starte
            int del = flip_data_local.start_conformation(c);
            flip_data_local.start_conformation(c) = flip_data_local.next_monomers(c, flip_data_local.start_conformation(c));
            flip_data_local.previous_monomers(c, flip_data_local.start_conformation(c)) = NO_SAW_NODE;
            flip_data_local.next_monomers(c, del) = NO_SAW_NODE;

            flip_data_local.sequence_on_lattice(c, del) = NO_XY_SPIN;
            flip_data_local.directions(c, del) = NO_SAW_NODE;
      
          //  const bool recycled = (flip_data_local.newIndex(c) == flip_data_local.oldIndex(c));

          //   if (!recycled) {
          //     flip_data_local.sequence_on_lattice(c, flip_data_local.save_end_conformation(c)) = NO_XY_SPIN;
          //     // directions cleanup is fine too, but keep it consistent with your representation
          //   }

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

      for (int e = 0; e < this->L; ++e) {
        const int pos = hostdata.lattice_nodes_positions(c, e);
        out << hostdata.sequence_on_lattice(c, pos) << " ";
      }
      out << "\n";
    }
  }  


  void out_dir_data(std::ostream &out, long long n_steps) {
    update_host_for_output();

    const int ls = this->lattice->lattice_size();

    for (int c = 0; c < N_CHAINS; ++c) {
      out << n_steps << " " << hostdata.J_chain(c) << " "
          << hostdata.start_index_in_nodes_position(c) << " "
          << hostdata.E(c) << " ";

      for (int i = 0; i < this->L; ++i) {
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



  // Main entry: load state into hostdata from your two text files
  long long restart_from_files(const std::string& angles_file,
    const std::string& dirs_file,
    long long target_step = -1)
  {
  const int ls = this->lattice->lattice_size();
  const int nsites = pow_int(ls, Dim);

  //allocate_state_views_host_if_needed(nsites);

  // Clear fields
  Kokkos::deep_copy(hostdata.sequence_on_lattice, (T)NO_XY_SPIN);
  Kokkos::deep_copy(hostdata.next_monomers, NO_SAW_NODE);
  Kokkos::deep_copy(hostdata.previous_monomers, NO_SAW_NODE);
  Kokkos::deep_copy(hostdata.directions, NO_SAW_NODE);

  // Read blocks
  auto A = read_angles_block(angles_file, this->L, target_step);
  auto D = read_dirs_block<Dim>(dirs_file, this->L, ls, target_step);

  // Basic sanity: steps should match (if not, pick dirs as truth)
  const long long step_loaded = (D.step >= 0) ? D.step : A.step;

  // Fill per chain
  for (int c = 0; c < N_CHAINS; ++c) {
  // J and E (prefer dirs for E/J consistency with geometry; you can flip preference)
  hostdata.J_chain(c) = D.J[c];
  hostdata.E(c)       = D.E[c];
  hostdata.newE(c)    = 0.0f;

  // Restore ring buffer
  hostdata.start_index_in_nodes_position(c) = D.start_idx[c];
  for (int e = 0; e < this->L; ++e) {
  hostdata.lattice_nodes_positions(c, e) = D.pos[c][e];
  }

  // Fill spins at the stored positions (same indexing convention you wrote out)
  for (int e = 0; e < this->L; ++e) {
  const int pos = D.pos[c][e];
  hostdata.sequence_on_lattice(c, pos) = (T)A.spin[c][e];
  }

  // Reconstruct start/end from ring buffer + start_index
  const int sidx = hostdata.start_index_in_nodes_position(c);
  const int start_pos = D.pos[c][sidx];
  const int end_pos   = D.pos[c][(sidx + this->L - 1) % this->L];
  hostdata.start_conformation(c) = start_pos;
  hostdata.end_conformation(c)   = end_pos;

  // Reconstruct chain order from start -> end
  std::vector<int> chain_pos(this->L);
  for (int t = 0; t < this->L; ++t) {
  chain_pos[t] = D.pos[c][(sidx + t) % this->L];
  }

  // Link next/prev and directions along the chain
  hostdata.previous_monomers(c, chain_pos[0]) = NO_SAW_NODE;
  for (int t = 0; t < this->L - 1; ++t) {
  const int p = chain_pos[t];
  const int q = chain_pos[t + 1];

  hostdata.next_monomers(c, p)     = q;
  hostdata.previous_monomers(c, q) = p;

  const int dir = find_dir_host(hostdata.map_of_contacts_int, hostdata.ndim2(), p, q);
  hostdata.directions(c, p) = dir;
  }
  hostdata.next_monomers(c, chain_pos[this->L - 1]) = NO_SAW_NODE;
  hostdata.directions(c, chain_pos[this->L - 1])    = NO_SAW_NODE;
  }

  return step_loaded;
  }  
  
 
  struct ConfigCheckResult {
    std::vector<int> code;      // size N_CHAINS
    std::vector<int> at;        // index in chain where it failed (or -1)
    std::vector<int> extra;     // e.g. duplicated index or neighbor mismatch info (or -1)
  };
  

ConfigCheckResult check_config_device_extended(bool check_nextprev,
                                               bool check_directions,
                                               bool check_coord_adjacency,
                                               bool detect_wrap_steps)
{
  using ExecSpace = Kokkos::Cuda;

  const int ls = this->lattice->lattice_size();
  const long long nsites = nsites_from_ls(ls, Dim);
  const int Lloc = this->L;

  Kokkos::View<int*, ExecSpace> d_code("chk_code", N_CHAINS);
  Kokkos::View<int*, ExecSpace> d_at  ("chk_at",   N_CHAINS);
  Kokkos::View<int*, ExecSpace> d_ex  ("chk_ex",   N_CHAINS);

  auto d = devicedata;

  Kokkos::parallel_for(
    "check_config_extended",
    Kokkos::RangePolicy<ExecSpace>(0, N_CHAINS),
    KOKKOS_LAMBDA(const int c) {

      int code = CHK_OK, at = -1, ex = -1;
      const int ndim2 = d.ndim2();
      const int s = d.start_index_in_nodes_position(c);

      auto pos_at = [&](int t) -> int {
        // t is chain-order index: 0..L-1 from start to end
        return d.lattice_nodes_positions(c, (s + t) % Lloc);
      };

      auto decode = [&](int pos, int &x, int &y, int &z) {
        x = pos % ls;
        if constexpr (Dim == 2) {
          y = pos / ls;
          z = 0;
        } else {
          y = (pos / ls) % ls;
          z = pos / (ls * ls);
        }
      };

      // 1) OOB check
      for (int t = 0; t < Lloc; ++t) {
        const int p = pos_at(t);
        if (p < 0 || (long long)p >= nsites) { code = CHK_OOB; at = t; ex = p; break; }
      }

      // 2) duplicate check (O(L^2), fine for L~100)
      if (code == CHK_OK) {
        for (int t = 0; t < Lloc; ++t) {
          const int pt = pos_at(t);
          for (int u = t + 1; u < Lloc; ++u) {
            const int pu = pos_at(u);
            if (pt == pu) { code = CHK_DUP; at = u; ex = t; break; }
          }
          if (code) break;
        }
      }

      // 3) adjacency via neighbor map (cheap and matches your move logic)
      if (code == CHK_OK) {
        for (int t = 0; t < Lloc - 1; ++t) {
          const int p = pos_at(t);
          const int q = pos_at(t + 1);
          bool ok = false;
          const int base = ndim2 * p;
          for (int dir = 0; dir < ndim2; ++dir) {
            if (d.map_of_contacts_int(base + dir) == q) { ok = true; break; }
          }
          if (!ok) { code = CHK_ADJ_MAP; at = t; ex = q; break; }
        }
      }

      // 4) next/prev consistency (finds corruption fast)
      if (check_nextprev && code == CHK_OK) {
        for (int t = 0; t < Lloc - 1; ++t) {
          const int p = pos_at(t);
          const int q = pos_at(t + 1);
          if (d.next_monomers(c, p) != q) { code = CHK_NEXT_PREV; at = t; ex = d.next_monomers(c, p); break; }
          if (d.previous_monomers(c, q) != p) { code = CHK_NEXT_PREV; at = t+1; ex = d.previous_monomers(c, q); break; }
        }
      }

      // 5) directions consistency (this is what you asked for)
      if (check_directions && code == CHK_OK) {
        // end must have NO_SAW_NODE direction
        {
          const int endp = pos_at(Lloc - 1);
          if (d.directions(c, endp) != NO_SAW_NODE) {
            code = CHK_DIR; at = Lloc - 1; ex = d.directions(c, endp);
          }
        }
        // for each link p->q, directions(p) must point to q
        if (code == CHK_OK) {
          for (int t = 0; t < Lloc - 1; ++t) {
            const int p = pos_at(t);
            const int q = pos_at(t + 1);
            const int dirp = d.directions(c, p);
            if (dirp < 0 || dirp >= ndim2) { code = CHK_DIR; at = t; ex = dirp; break; }
            const int neigh = d.map_of_contacts_int(ndim2 * p + dirp);
            if (neigh != q) { code = CHK_DIR; at = t; ex = neigh; break; }
          }
        }
      }

      // 6) coordinate adjacency without wrap detection (catches fake-neighbor maps)
      if (check_coord_adjacency && code == CHK_OK) {
        const int side = d.lattice_side_device(); // read inside device
        for (int t = 0; t < Lloc - 1; ++t) {
          const int p = pos_at(t);
          const int q = pos_at(t + 1);
          const float rr = r2(p, q, side);
          if (rr != 1.0f) { code = CHK_ADJ_COORD; at = t; ex = (int)rr; break; }
        }


 
      }

      d_code(c) = code;
      d_at(c)   = at;
      d_ex(c)   = ex;
    }
  );
  Kokkos::fence();

  auto h_code = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_code);
  auto h_at   = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_at);
  auto h_ex   = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_ex);

  ConfigCheckResult r;
  r.code.resize(N_CHAINS);
  r.at.resize(N_CHAINS);
  r.extra.resize(N_CHAINS);
  for (int c = 0; c < N_CHAINS; ++c) {
    r.code[c]  = h_code(c);
    r.at[c]    = h_at(c);
    r.extra[c] = h_ex(c);
  }
  return r;
}

// 

bool check_before_output(bool check_nextprev) {
  auto r = check_config_device_extended(      /*check_nextprev=*/true,
    /*check_directions=*/true,
    /*check_coord_adjacency=*/true,
    /*detect_wrap_steps=*/true);
  for (int c = 0; c < N_CHAINS; ++c) {
    if (r.code[c] != 0) {
      std::cerr << "[CONFIG BAD] chain " << c
                << " code=" << r.code[c]
                << " at=" << r.at[c]
                << " extra=" << r.extra[c] << "\n";
      return false;
    }
  }
  return true;
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
                        float r_val   = SAW_model<float, Dim>::r2(pos_i, pos_j, flip_data.lattice_side_device());
                        r_val = Kokkos::sqrt(r_val) * r_val; 
                        H_total += Kokkos::cos(theta_i - theta_j) / r_val;
    
                        r_val   = SAW_model<float, Dim>::r2(pos_i, pos_k , flip_data.lattice_side_device());
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
                        float r_val   = SAW_model<float, Dim>::r2(pos_i, pos_j, flip_data.lattice_side_device());
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

  using Base = SAW_model<float, Dim>;
  using Base::hostdata;
  using Base::devicedata;
  using Base::lattice;

  XY_SI (int L,float Jmin = 0.25, float Jmax = 0.26);

  void spin_init_random();
  void start_kernel_energy_init() override;

  struct XY_SI_EnergyOp {
    using ExecSpace = Kokkos::Cuda;
    using member_type = typename Kokkos::TeamPolicy<ExecSpace>::member_type;
  
    DeviceData<Kokkos::CudaSpace, float> d;


    KOKKOS_INLINE_FUNCTION
    void delta_energy(const member_type& team,
                      const DeviceData<Kokkos::CudaSpace, float>& flip,
                      int c) const
    {
      Kokkos::single(Kokkos::PerTeam(team), [&]() {
        const int ndim2 = flip.ndim2();
        const int p_old = flip.oldIndex(c);                     
        const int p_new = flip.newIndex(c);                     
        const float s_old = flip.oldspin(c);                      
        const float s_new = flip.spinValue(c); //flip.sequence_on_lattice(c, p_new);  
      //  printf("%d \n", s_new);
        const bool recycled = (p_old == p_new);

        float E_old_local = 0.0f;   // edges touching p_old in OLD state
        float E_new_local = 0.0f;   // edges touching p_new in NEW state
        {
          const int base = ndim2 * p_old;
          for (int dir = 0; dir < ndim2; ++dir) {
            const int nb = flip.map_of_contacts_int(base + dir);
  
            // If not recycled, p_new is occupied NOW but was empty BEFORE -> skip it for OLD.
            if (!recycled && nb == p_new) continue;
  
            const float s_nb = flip.sequence_on_lattice(c, nb);
            if (s_nb == NO_XY_SPIN) continue;   // neighbor empty
            E_old_local += - Kokkos::cos(s_old - s_nb);
          }
        }

        {
          const int base = ndim2 * p_new;
          for (int dir = 0; dir < ndim2; ++dir) {
            const int nb = flip.map_of_contacts_int(base + dir);
  
            // Symmetric safety: p_old is empty NOW but was occupied BEFORE -> skip for NEW if needed.
            // (Normally sequence_on_lattice(p_old) is already NO_XY_SPIN, so this is just hygiene.)
            if (!recycled && nb == p_old) continue;
  
            const float s_nb = flip.sequence_on_lattice(c, nb);
            if (s_nb == NO_XY_SPIN) continue;
            E_new_local += - Kokkos::cos(s_new - s_nb);
          }
        }        
 
 
          flip.d_E_1(c) = E_new_local - E_old_local;
      });



    }


    KOKKOS_INLINE_FUNCTION
    void energy(const member_type& team,
                       const DeviceData<Kokkos::CudaSpace, float>& flip,
                       int c) const
   {
    const int ndim2 = flip.ndim2();
    float H = 0.0f;
        Kokkos::parallel_reduce(
          Kokkos::TeamThreadRange(team, flip.L()),
          [&](const int t, float& sum) {
            const int p   = flip.lattice_nodes_positions(c, t);
            const float s_p = flip.sequence_on_lattice(c, p);
            // (Should never be empty for positions list, but fine)
            if (s_p == NO_XY_SPIN) return;
            const int base = ndim2 * p;
          for (int dir = 0; dir < ndim2; ++dir) {
            const int nb   = flip.map_of_contacts_int(base + dir);
            const float s_nb = flip.sequence_on_lattice(c, nb);
            if (s_nb == NO_XY_SPIN) continue;

            // Count each undirected edge twice (p->nb and nb->p), so multiply by 1/2.
            sum += -0.5f * Kokkos::cos(s_nb  - s_p);
          }

        }, H);

      Kokkos::single(Kokkos::PerTeam(team), [&]() { flip.newE(c) = H; });
    
    }

  };


  void runMCMCOnDevice(long long MC_STEPS = 10000, long long epoch = 1000) {
    XY_SI_EnergyOp op{ this->devicedata };
    XYSpinProposal  propose{}; 
    this->runMCMCOnDevice_impl(op, propose, MC_STEPS, epoch);
    //this->parallel_tempering_swap();
  } 

};

template<int Dim>
class Ising_SI : public SAW_model<int, Dim> {
public:
  using Base = SAW_model<int, Dim>;
  using Base::hostdata;
  using Base::devicedata;
  using Base::lattice;

  Ising_SI (int L,float Jmin = 0.25, float Jmax = 0.26);

  void spin_init_random();
  void start_kernel_energy_init() override;

  struct Ising_SI_EnergyOp {
    using ExecSpace   = Kokkos::Cuda;
    using member_type = typename Kokkos::TeamPolicy<ExecSpace>::member_type;
    DeviceData<Kokkos::CudaSpace, int> d;


    KOKKOS_INLINE_FUNCTION
    void delta_energy(const member_type& team,
                      const DeviceData<Kokkos::CudaSpace, int>& flip,
                      int c) const
    {

      Kokkos::single(Kokkos::PerTeam(team), [&]() {
        const int ndim2 = flip.ndim2();
        const int p_old = flip.oldIndex(c);                     
        const int p_new = flip.newIndex(c);                     
        const int s_old = flip.oldspin(c);                      
        const int s_new = flip.spinValue(c); //flip.sequence_on_lattice(c, p_new);  
      //  printf("%d \n", s_new);
        const bool recycled = (p_old == p_new);

        float E_old_local = 0.0f;   // edges touching p_old in OLD state
        float E_new_local = 0.0f;   // edges touching p_new in NEW state
        {
          const int base = ndim2 * p_old;
          for (int dir = 0; dir < ndim2; ++dir) {
            const int nb = flip.map_of_contacts_int(base + dir);
  
            // If not recycled, p_new is occupied NOW but was empty BEFORE -> skip it for OLD.
            if (!recycled && nb == p_new) continue;
  
            const int s_nb = flip.sequence_on_lattice(c, nb);
            if (s_nb == NO_XY_SPIN) continue;   // neighbor empty
            E_old_local += -float(s_old * s_nb);
          }
        }

        {
          const int base = ndim2 * p_new;
          for (int dir = 0; dir < ndim2; ++dir) {
            const int nb = flip.map_of_contacts_int(base + dir);
  
            // Symmetric safety: p_old is empty NOW but was occupied BEFORE -> skip for NEW if needed.
            // (Normally sequence_on_lattice(p_old) is already NO_XY_SPIN, so this is just hygiene.)
            if (!recycled && nb == p_old) continue;
  
            const int s_nb = flip.sequence_on_lattice(c, nb);
            if (s_nb == NO_XY_SPIN) continue;
            E_new_local += -float(s_new * s_nb);
          }
        }        

          // //смотрим потери
          // for (int j = 0; j < lattice.ndim2(); j++) {
          // step = lattice.map_of_contacts_int[lattice.ndim2() * temp + j];
          // if (sequence_on_lattice[step] != 0) {
          //     hh = hh - sequence_on_lattice[temp] * sequence_on_lattice[step];
          // }
          // }

          // //смотрим выигрыш
          // for (int j = 0; j < lattice.ndim2(); j++) {
          // step = lattice.map_of_contacts_int[lattice.ndim2() * end_conformation + j];
          // if (sequence_on_lattice[step] != 0) {
          //     hh = hh + sequence_on_lattice[end_conformation] * sequence_on_lattice[step];
          // }
          // } 

          flip.d_E_1(c) = E_new_local - E_old_local;



          // Inside delta_energy, add:
//printf("chain %d: p_old=%d p_new=%d s_old=%d s_new=%d E_old=%.3f E_new=%.3f dE=%.3f\n",
//  c, p_old, p_new, s_old, s_new, E_old_local, E_new_local, E_new_local - E_old_local);

      });

    }



    KOKKOS_INLINE_FUNCTION
    void energy(const member_type& team,
                const DeviceData<Kokkos::CudaSpace, int>& flip,
                int c) const
    {
      const int ndim2 = flip.ndim2();
      float H = 0.0f;

      Kokkos::parallel_reduce(
        Kokkos::TeamThreadRange(team, flip.L()),
        [&](const int t, float& sum) {
          const int p   = flip.lattice_nodes_positions(c, t);
          const int s_p = flip.sequence_on_lattice(c, p);
          // (Should never be empty for positions list, but fine)
          if (s_p == NO_XY_SPIN) return;
          const int base = ndim2 * p;
        for (int dir = 0; dir < ndim2; ++dir) {
          const int nb   = flip.map_of_contacts_int(base + dir);
          const int s_nb = flip.sequence_on_lattice(c, nb);
          if (s_nb == NO_XY_SPIN) continue;

          // Count each undirected edge twice (p->nb and nb->p), so multiply by 1/2.
          sum += -0.5f * float(s_p * s_nb);
        }

      }, H);

      Kokkos::single(Kokkos::PerTeam(team), [&]() { flip.newE(c) = H; });
      
    }

  };


  void runMCMCOnDevice(long long MC_STEPS = 10000, long long epoch = 1000) {
    Ising_SI_EnergyOp op {this->devicedata};
    IsingSpinProposal  propose{}; 
    this->runMCMCOnDevice_impl(op, propose, MC_STEPS, epoch);
    //this->runMCMCOnDevice_impl(op, propose, MC_STEPS, epoch);
    //this->parallel_tempering_swap();

    // XY_LI_EnergyOp op{ this->devicedata };
    // XYSpinProposal  propose{}; 
    // this->runMCMCOnDevice_impl(op, propose, MC_STEPS, epoch);
  } 
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
    hostdata.map_of_contacts_int = this->lattice->map_of_contacts_int;
    hostdata.inverse_steps       = this->lattice->inverse_steps;

    hostdata.L = this->L_host;
    hostdata.lattice_side_device = this->lattice->lattice_side_host;

    hostdata.ndim2 = Kokkos::View<int, Kokkos::HostSpace>("ndim2");
    hostdata.ndim2() = this->lattice->ndim2();

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
    team.team_barrier();
    d.E(c) = d.newE(c);
  }
};

template<>
struct SpinProposalTraits<float> {
  using type = XYSpinProposal;
};




template<int Dim>
class Homopolymer : public SAW_model<int, Dim> {
public:
  using Base = SAW_model<int, Dim>;
  using Base::hostdata;
  using Base::devicedata;
  using Base::lattice;

  Homopolymer(int L, float Jmin = 0.25f, float Jmax = 0.26f);

  void spin_init_random() override;
  void start_kernel_energy_init() override;

  struct Homopolymer_EnergyOp {
    using ExecSpace   = Kokkos::Cuda;
    using member_type = typename Kokkos::TeamPolicy<ExecSpace>::member_type;

    DeviceData<Kokkos::CudaSpace, int> d;

    KOKKOS_INLINE_FUNCTION
    void delta_energy(const member_type& team,
                      const DeviceData<Kokkos::CudaSpace, int>& flip,
                      int c) const
    {
      Kokkos::single(Kokkos::PerTeam(team), [&]() {
        const int ndim2 = flip.ndim2();

        const int p_old = flip.oldIndex(c);   // removed endpoint
        const int p_new = flip.newIndex(c);   // added endpoint
        const bool recycled = (p_old == p_new);

        // Which move was proposed?
        const bool add_end = (flip.flipMoveType(c) < 0.5f);

        // Old bonded neighbor of removed endpoint in the OLD state:
        // after proposal, the chain has already been updated, so:
        //   add_end   -> removed monomer was old start, bonded to current start
        //   add_start -> removed monomer was old end,   bonded to current end
        const int old_bonded =
            add_end ? flip.start_conformation(c)
                    : flip.end_conformation(c);

        // New bonded neighbor of added endpoint in the NEW state:
        //   add_end   -> new endpoint is bonded to its previous monomer
        //   add_start -> new endpoint is bonded to its next monomer
        const int new_bonded =
            add_end ? flip.previous_monomers(c, p_new)
                    : flip.next_monomers(c, p_new);

        float E_old_local = 0.0f;
        float E_new_local = 0.0f;

        // Contacts lost by removing p_old
        {
          const int base = ndim2 * p_old;
          for (int dir = 0; dir < ndim2; ++dir) {
            const int nb = flip.map_of_contacts_int(base + dir);

            // Exclude the covalent bond in the OLD configuration
            if (nb == old_bonded) continue;

            // If not recycled, p_new is occupied now but was empty before
            if (!recycled && nb == p_new) continue;

            if (flip.sequence_on_lattice(c, nb) == NO_XY_SPIN) continue;

            E_old_local += -1.0f;
          }
        }

        // Contacts gained by adding p_new
        {
          const int base = ndim2 * p_new;
          for (int dir = 0; dir < ndim2; ++dir) {
            const int nb = flip.map_of_contacts_int(base + dir);

            // Exclude the covalent bond in the NEW configuration
            if (nb == new_bonded) continue;

            if (flip.sequence_on_lattice(c, nb) == NO_XY_SPIN) continue;

            E_new_local += -1.0f;
          }
        }

        flip.d_E_1(c) = E_new_local - E_old_local;
      });
    }

    KOKKOS_INLINE_FUNCTION
    void energy(const member_type& team,
                const DeviceData<Kokkos::CudaSpace, int>& flip,
                int c) const
    {
      const int ndim2 = flip.ndim2();
      float H = 0.0f;

      Kokkos::parallel_reduce(
        Kokkos::TeamThreadRange(team, flip.L()),
        [&](const int t, float& sum) {
          const int p = flip.lattice_nodes_positions(c, t);
          if (flip.sequence_on_lattice(c, p) == NO_XY_SPIN) return;

          const int p_next = flip.next_monomers(c, p);
          const int p_prev = flip.previous_monomers(c, p);
          const int base   = ndim2 * p;

          for (int dir = 0; dir < ndim2; ++dir) {
            const int nb = flip.map_of_contacts_int(base + dir);

            if (flip.sequence_on_lattice(c, nb) == NO_XY_SPIN) continue;

            // Exclude covalent neighbors along the polymer backbone
            if (nb == p_next || nb == p_prev) continue;

            // Each undirected contact is seen twice, so divide by 2
            sum += -0.5f;
          }
        },
        H
      );

      Kokkos::single(Kokkos::PerTeam(team), [&]() {
        flip.newE(c) = H;
      });
    }
  };

  void runMCMCOnDevice(long long MC_STEPS = 10000, long long epoch = 1000) {
    Homopolymer_EnergyOp op{ this->devicedata };
    HomopolymerSpinProposal propose{};
    this->runMCMCOnDevice_impl(op, propose, MC_STEPS, epoch);
  }
};





template<int Dim>
class XY_LI_normalize : public SAW_model<float, Dim> {
public:
    using Base = SAW_model<float, Dim>;
    using Base::hostdata;
    using Base::devicedata;
    using Base::lattice;

    XY_LI_normalize(int L, float Jmin = 0.25f, float Jmax = 0.26f, float power = 3.0f);

    void spin_init_random();
    void start_kernel_energy_init() override;

    // Exponent in the long-range coupling 1/r^power_ij and in the
    // normalization sum S = sum_{i<j} 1/r_ij^power_ij.
    float power_ij;

    struct XY_LI_normalize_EnergyOp {
        using ExecSpace   = Kokkos::Cuda;
        using member_type = typename Kokkos::TeamPolicy<ExecSpace>::member_type;

        DeviceData<Kokkos::CudaSpace, float> d;
        float power;

        // ------- delta_energy: full recompute of H_norm in the new geometry -------
        // Returns d_E_1(c) = H_norm_new - H_norm_old, where
        //   H_norm = ( -sum_{i<j} cos(theta_i-theta_j) / r_ij^power ) / ( sum_{i<j} 1/r_ij^power ).
        // This is O(L^2) per move but parallelized across the team.
        KOKKOS_INLINE_FUNCTION
        void delta_energy(const member_type& team,
                          const DeviceData<Kokkos::CudaSpace, float>& flip,
                          int c) const
        {
            double E_new = 0.0;   // -sum cos/r^p over all pairs in NEW geometry
            double S_new = 0.0;   //  sum  1 /r^p over all pairs in NEW geometry

            Kokkos::parallel_reduce(
                Kokkos::TeamThreadRange(team, flip.N_pairs()),
                [&](const int ind, double& E_acc, double& S_acc) {
                    const int i      = flip.i_index(ind);
                    const int j      = flip.j_index(ind);
                    const int pos_i  = flip.lattice_nodes_positions(c, i);
                    const int pos_j  = flip.lattice_nodes_positions(c, j);
                    const float ti   = flip.sequence_on_lattice(c, pos_i);
                    const float tj   = flip.sequence_on_lattice(c, pos_j);

                    const float r2_v = SAW_model<float, Dim>::r2(
                        pos_i, pos_j, flip.lattice_side_device());
                    const float r    = Kokkos::sqrt(r2_v);
                    // Fast path for the canonical case power == 3 (matches XY_LI's idiom)
                    const float inv_rp = (power == 3.0f)
                        ? (1.0f / (r * r2_v))
                        : (1.0f / Kokkos::pow(r, power));

                    E_acc += -static_cast<double>(Kokkos::cos(ti - tj)) * inv_rp;
                    S_acc +=  static_cast<double>(inv_rp);
                },
                E_new, S_new
            );

            Kokkos::single(Kokkos::PerTeam(team), [&]() {
                const float scale = static_cast<float>(flip.L()); 
                const float H_norm_new = scale * static_cast<float>(E_new / S_new);
                const float H_norm_old = flip.E(c);   // E stores the cumulative H_norm
                flip.d_E_1(c) = H_norm_new - H_norm_old;
                // Note: the Metropolis kernel will do E(c) += d_E_1(c) on accept,
                // so on accept E(c) becomes H_norm_new automatically.
            });
        }

        // ------- energy: full H_norm of the current configuration -------
        // Used by EnergyInitKernel; writes the result into newE(c), then the
        // base class copies newE -> E at init time.
        KOKKOS_INLINE_FUNCTION
        void energy(const member_type& team,
                    const DeviceData<Kokkos::CudaSpace, float>& flip,
                    int c) const
        {
            double E_total = 0.0;
            double S_total = 0.0;

            Kokkos::parallel_reduce(
                Kokkos::TeamThreadRange(team, flip.N_pairs()),
                [&](const int ind, double& E_acc, double& S_acc) {
                    const int i      = flip.i_index(ind);
                    const int j      = flip.j_index(ind);
                    const int pos_i  = flip.lattice_nodes_positions(c, i);
                    const int pos_j  = flip.lattice_nodes_positions(c, j);
                    const float ti   = flip.sequence_on_lattice(c, pos_i);
                    const float tj   = flip.sequence_on_lattice(c, pos_j);

                    const float r2_v = SAW_model<float, Dim>::r2(
                        pos_i, pos_j, flip.lattice_side_device());
                    const float r    = Kokkos::sqrt(r2_v);
                    const float inv_rp = (power == 3.0f)
                        ? (1.0f / (r * r2_v))
                        : (1.0f / Kokkos::pow(r, power));

                    E_acc += -static_cast<double>(Kokkos::cos(ti - tj)) * inv_rp;
                    S_acc +=  static_cast<double>(inv_rp);
                },
                E_total, S_total
            );

            Kokkos::single(Kokkos::PerTeam(team), [&]() {
                const float scale = static_cast<float>(flip.L()); 
                flip.newE(c) = static_cast<float>(E_total / (S_total) * scale);
            });
        }
    };

    void runMCMCOnDevice(long long MC_STEPS = 10000, long long epoch = 1000) {
        XY_LI_normalize_EnergyOp op{ this->devicedata, this->power_ij };
        XYSpinProposal           propose{};
        this->runMCMCOnDevice_impl(op, propose, MC_STEPS, epoch);
        // this->parallel_tempering_swap();
    }
};

#endif