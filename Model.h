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
  //LATTICE: 
    Kokkos::View<int*, ExecSpace> map_of_contacts_int;
    Kokkos::View<int*, ExecSpace> inverse_steps;
    Kokkos::View<int, ExecSpace> ndim2;


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
    SAW_model<T, Dim>(int L);

    // SAW_model(int L) : Model(L) {};
    virtual void spin_init_random() = 0;
    virtual void start_kernel_energy_init() = 0;

    void geometry_initialization_arrays();

    void geometry_initialization_stick();

    void scalars_MC_preparation();


    void HostDataInit();
    void DeviceDataInit ();

    //HostData hostdata;
    DeviceData<Kokkos::CudaSpace, T> devicedata;
    DeviceData<Kokkos::HostSpace, T> hostdata;

    template<class EnergyOp>
    void energy_init(EnergyOp energy) {
      using ExecSpace   = Kokkos::Cuda;
      using team_policy = Kokkos::TeamPolicy<ExecSpace>;

      team_policy policy(N_CHAINS, Kokkos::AUTO());

      auto d = devicedata;

      Kokkos::parallel_for("EnergyInit",
        policy,
        EnergyInitKernel<ExecSpace, T, EnergyOp, Dim>{d, energy}
      );
      Kokkos::fence();
    }

    
    KOKKOS_INLINE_FUNCTION
    void hierarchicalOneKernel_Reconnect (
      const Kokkos::TeamPolicy<Kokkos::Cuda>::member_type& team_member,
      const DeviceData<Kokkos::CudaSpace, float>& flip_data_local,
      int chain, 
      const Kokkos::Random_XorShift64_Pool<Kokkos::Cuda> & pool) const{
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
    

    template<class ExecSpace, class SpinProposalOp>
    KOKKOS_INLINE_FUNCTION
    void hierarchicalFlipMoveAddEnd(
      const Kokkos::TeamPolicy<Kokkos::Cuda>::member_type& team_member,
      const DeviceData<Kokkos::CudaSpace, float>& flip_data_local,
      int c, 
      const Kokkos::Random_XorShift64_Pool<Kokkos::Cuda> & pool,
      const SpinProposalOp& propose_spin) const{
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
          auto rand_gen1 = pool.get_state();
          flip_data_local.spinValue(c) = propose_spin(T{}, pool);
          pool.free_state(rand_gen1);
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


    template<class ExecSpace, class SpinProposalOp>
    KOKKOS_INLINE_FUNCTION
    void hierarchicalFlipMoveAddStart(
      const Kokkos::TeamPolicy<Kokkos::Cuda>::member_type& team_member,
      const DeviceData<Kokkos::CudaSpace, float>& flip_data_local,
      int c, 
      const Kokkos::Random_XorShift64_Pool<Kokkos::Cuda> & pool,
      const SpinProposalOp& propose_spin) const{
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
          auto rand_gen1 = pool.get_state();
          flip_data_local.spinValue(c) = propose_spin(T{}, pool);
          pool.free_state(rand_gen1);
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
    void hierarchicalOneKernel_AddEnd_FirstPart(
      const Kokkos::TeamPolicy<Kokkos::Cuda>::member_type& team_member,
      const DeviceData<Kokkos::CudaSpace, float>& flip_data_local,
      int c, 
      const Kokkos::Random_XorShift64_Pool<Kokkos::Cuda> & pool, 
      float q_ifaccept) const
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
    void hierarchicalOneKernel_AddStart_FirstPart(
      const Kokkos::TeamPolicy<Kokkos::Cuda>::member_type& team_member,
      const DeviceData<Kokkos::CudaSpace, float>& flip_data_local,
      int c, 
      const Kokkos::Random_XorShift64_Pool<Kokkos::Cuda> & pool, 
      float q_ifaccept) const
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

    team_policy policy(N_CHAINS, Kokkos::AUTO());

    auto d = devicedata;
    auto pool = my_pool;

    Kokkos::parallel_for("MCMC_on_device", policy,
      MetropolisKernel<ExecSpace, T, EnergyOp, SpinProposalOp, Dim>{d, energy, propose, pool, MC_STEPS, epoch2 }
    );
    Kokkos::fence();
    epoch2 += 1;
  }

};

template<int Dim>
class XY_LI : public SAW_model<float, Dim> {
public:
    using Base = SAW_model<float, Dim>;
    using Base::hostdata;
    using Base::devicedata;
    using Base::lattice;

    XY_LI (int L);

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
    }
    

    // void run_one_metropolis_sweep() {
    //   XY_LI_EnergyOp op{ this->devicedata};
    //   this->metropolis_step(op);
    // }

 
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

template<class ExecSpace, class T, class EnergyOp, class SpinProposalOp, int Dim>
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

    hierarchicalOneKernel_Reconnect(team, d, c, pool);
    team.team_barrier();
  }
};


#endif