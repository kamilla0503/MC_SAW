#ifndef INTERACTION_SAW_MODELS_COMMON_H
#define INTERACTION_SAW_MODELS_COMMON_H

#include <Kokkos_Core.hpp>
#include <string>
#include <type_traits>

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
  template<class Pool, class TeamMember>
  KOKKOS_INLINE_FUNCTION
  int propose(const TeamMember& team, Pool& pool, int c, int site) const {
    auto state = pool.get_state();
    const double u = state.drand();        // [0,1)
    pool.free_state(state);
    return (u < 0.5) ? -1 : +1;
  }
};

struct XYSpinProposal {
  float two_pi;

  XYSpinProposal() : two_pi(6.283185307179586f) {}

  template<class Pool, class TeamMember>
  KOKKOS_INLINE_FUNCTION
  float propose(const TeamMember& team, Pool& pool, int c, int site) const {
    auto state = pool.get_state();
    const double u = state.drand();        // [0,1)
    pool.free_state(state);
    return float(u) * two_pi;
  }
};

template<class T>
struct SpinProposalTraits;

template<>
struct SpinProposalTraits<int> {
  using type = IsingSpinProposal;
};

#endif