#ifndef INTERACTION_SAW_MODELS_COMMON_H
#define INTERACTION_SAW_MODELS_COMMON_H

#include <Kokkos_Core.hpp>
#include <string>
#include <type_traits>
#include <sstream>
#include <fstream>

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

#ifndef N_CHAINS
#define N_CHAINS 16
#endif


const float PI = std::atan(1.0)*4; 

template<class DstView, class SrcView>
void realloc_like_and_copy(DstView& dst, const SrcView& src, const std::string& label) {
  static_assert(int(DstView::rank) == int(SrcView::rank), "rank mismatch");
  static_assert(std::is_same_v<typename DstView::value_type, typename SrcView::value_type>,
                "value_type mismatch");

  // Allocate/resize dst to match src
  if constexpr (DstView::rank == 0) {
    // scalar View: constructor takes only label
    if (!dst.is_allocated()) 
      dst = DstView(label);
  } else if constexpr (DstView::rank == 1) {
    if (!dst.is_allocated() || dst.extent(0) != src.extent(0)) {
      dst = DstView(label, src.extent(0));
    }
  } else if constexpr (DstView::rank == 2) {
    if (!dst.is_allocated() || dst.extent(0) != src.extent(0) || dst.extent(1) != src.extent(1)) {
      dst = DstView(label, src.extent(0), src.extent(1));
    }
  } else if constexpr (DstView::rank == 3) {
    if (!dst.is_allocated() || dst.extent(0) != src.extent(0) || dst.extent(1) != src.extent(1) ||
        dst.extent(2) != src.extent(2)) {
      dst = DstView(label, src.extent(0), src.extent(1), src.extent(2));
    }
  } else {
    static_assert(DstView::rank <= 3, "extend realloc_like_and_copy for higher rank");
  }

  Kokkos::deep_copy(dst, src);
}

KOKKOS_INLINE_FUNCTION
uint64_t splitmix64(uint64_t x) {
  x += 0x9E3779B97F4A7C15ull;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  return x ^ (x >> 31);
}

KOKKOS_INLINE_FUNCTION
double u01(uint64_t x) {               // [0,1)
  return (x >> 11) * (1.0/9007199254740992.0);
}

KOKKOS_INLINE_FUNCTION
double rand_chain_step(uint64_t seed, int chain, long long step, int stream) {
  uint64_t key = seed
    ^ (0xA24BAED4963EE407ull * (uint64_t)(chain+1))
    ^ (0x9E3779B97F4A7C15ull * (uint64_t)(step*4 + stream));
  return u01(splitmix64(key));
}

struct IsingSpinProposal {
  template<class Pool>
  KOKKOS_INLINE_FUNCTION
  int operator()(Pool& pool) const {
    auto state = pool.get_state();
    const double u = state.drand();   // [0,1)
    pool.free_state(state);
    return (u < 0.5) ? -1 : +1;
  }
};

struct XYSpinProposal {
  static constexpr float two_pi = 6.283185307179586f;

  template<class Pool>
  KOKKOS_INLINE_FUNCTION
  float operator()(Pool& pool) const {
    auto state = pool.get_state();
    const double u = state.drand();   // [0,1)
    pool.free_state(state);
    return static_cast<float>(u) * two_pi;
  }
};

struct HomopolymerSpinProposal {
  template<class Pool>
  KOKKOS_INLINE_FUNCTION
  int operator()(Pool& /*pool*/) const {
    return 1;
  }
};
 
template<class T>
struct SpinProposalTraits;

template<>
struct SpinProposalTraits<int> {
  using type = IsingSpinProposal;
};




struct RestartBlockAngles {
  long long step = -1;
  std::vector<float> J, E;                 // size N_CHAINS
  std::vector<std::vector<float>> spin;    // [chain][L] in ring-buffer index order
};

struct RestartBlockDirs {
  long long step = -1;
  std::vector<float> J, E;                 // size N_CHAINS
  std::vector<int> start_idx;              // size N_CHAINS
  std::vector<std::vector<int>> pos;       // [chain][L] lattice indices in ring-buffer index order
};

static inline int pow_int(int a, int p) {
  int r = 1; for (int i=0;i<p;i++) r *= a; return r;
}

// Reads the last complete block (N_CHAINS lines with same step) OR the last block matching target_step.
static inline RestartBlockAngles read_angles_block(const std::string& fname, int L, long long target_step = -1) {
  std::ifstream in(fname);
  if (!in) throw std::runtime_error("Cannot open angles file: " + fname);

  RestartBlockAngles best;
  RestartBlockAngles cur;
  cur.J.assign(N_CHAINS, 0.f);
  cur.E.assign(N_CHAINS, 0.f);
  cur.spin.assign(N_CHAINS, std::vector<float>(L, 0.f));

  long long step;
  float J, E;
  int chain_in_block = 0;
  long long cur_step = LLONG_MIN;

  while (true) {
    if (!(in >> step >> J >> E)) break;

    std::vector<float> spins(L);
    for (int i = 0; i < L; ++i) in >> spins[i];

    if (step != cur_step) {
      cur_step = step;
      chain_in_block = 0;
    }

    if (chain_in_block < N_CHAINS) {
      cur.step = cur_step;
      cur.J[chain_in_block] = J;
      cur.E[chain_in_block] = E;
      cur.spin[chain_in_block] = std::move(spins);
      chain_in_block++;
    }

    if (chain_in_block == N_CHAINS) {
      if (target_step < 0 || cur.step == target_step) {
        best = cur;
      }
      // next line should start a new step block
    }
  }

  if (best.step < 0) throw std::runtime_error("No complete angles block found in " + fname);
  return best;
}

// Dirs format: step J start_idx E then L*(x y z)
/*
static inline RestartBlockDirs read_dirs_block(const std::string& fname, int L, int ls, long long target_step = -1) {
  std::ifstream in(fname);
  if (!in) throw std::runtime_error("Cannot open dirs file: " + fname);

  RestartBlockDirs best;
  RestartBlockDirs cur;
  cur.J.assign(N_CHAINS, 0.f);
  cur.E.assign(N_CHAINS, 0.f);
  cur.start_idx.assign(N_CHAINS, 0);
  cur.pos.assign(N_CHAINS, std::vector<int>(L, 0));

  long long step;
  float J, E;
  int start_idx;
  int chain_in_block = 0;
  long long cur_step = LLONG_MIN;

  while (true) {
    if (!(in >> step >> J >> start_idx >> E)) break;

    std::vector<int> posbuf(L);
    for (int i = 0; i < L; ++i) {
      int x, y, z;
      in >> x >> y >> z; // you always wrote x y z
      const int idx = x + ls * (y + ls * z);
      posbuf[i] = idx;
    }

    if (step != cur_step) {
      cur_step = step;
      chain_in_block = 0;
    }

    if (chain_in_block < N_CHAINS) {
      cur.step = cur_step;
      cur.J[chain_in_block] = J;
      cur.E[chain_in_block] = E;
      cur.start_idx[chain_in_block] = start_idx;
      cur.pos[chain_in_block] = std::move(posbuf);
      chain_in_block++;
    }

    if (chain_in_block == N_CHAINS) {
      if (target_step < 0 || cur.step == target_step) {
        best = cur;
      }
    }
  }

  if (best.step < 0) throw std::runtime_error("No complete dirs block found in " + fname);
  return best;
}
*/

template<int Dim>
static inline RestartBlockDirs read_dirs_block(const std::string& fname,
                                               int L,
                                               int ls,
                                               long long target_step = -1) {
  std::ifstream in(fname);
  if (!in) throw std::runtime_error("Cannot open dirs file: " + fname);

  RestartBlockDirs best;
  RestartBlockDirs cur;
  cur.J.assign(N_CHAINS, 0.f);
  cur.E.assign(N_CHAINS, 0.f);
  cur.start_idx.assign(N_CHAINS, 0);
  cur.pos.assign(N_CHAINS, std::vector<int>(L, 0));

  long long step;
  float J, E;
  int start_idx;
  int chain_in_block = 0;
  long long cur_step = LLONG_MIN;

  while (true) {
    if (!(in >> step >> J >> start_idx >> E)) break;

    std::vector<int> posbuf(L);
    for (int i = 0; i < L; ++i) {
      int idx = 0;

      if constexpr (Dim == 2) {
        int x, y;
        if (!(in >> x >> y)) {
          throw std::runtime_error("Malformed 2D dirs record in " + fname);
        }
        idx = x + ls * y;
      } else if constexpr (Dim == 3) {
        int x, y, z;
        if (!(in >> x >> y >> z)) {
          throw std::runtime_error("Malformed 3D dirs record in " + fname);
        }
        idx = x + ls * (y + ls * z);
      } else {
        static_assert(Dim == 2 || Dim == 3, "read_dirs_block only supports Dim=2 or Dim=3");
      }

      posbuf[i] = idx;
    }

    if (step != cur_step) {
      cur_step = step;
      chain_in_block = 0;
    }

    if (chain_in_block < N_CHAINS) {
      cur.step = cur_step;
      cur.J[chain_in_block] = J;
      cur.E[chain_in_block] = E;
      cur.start_idx[chain_in_block] = start_idx;
      cur.pos[chain_in_block] = std::move(posbuf);
      chain_in_block++;
    }

    if (chain_in_block == N_CHAINS) {
      if (target_step < 0 || cur.step == target_step) {
        best = cur;
      }
    }
  }

  if (best.step < 0) {
    throw std::runtime_error("No complete dirs block found in " + fname);
  }
  return best;
}

// Find direction id such that neighbor(pos,dir) == pos_next
KOKKOS_INLINE_FUNCTION
static int find_dir_host(const Kokkos::View<int*, Kokkos::HostSpace>& map_of_contacts_int,
                         int ndim2, int pos, int pos_next)
{
  for (int dir = 0; dir < ndim2; ++dir) {
    if (map_of_contacts_int(ndim2 * pos + dir) == pos_next) return dir;
  }
  return NO_SAW_NODE;
}

#endif
