// =====================================================================
//  process_saw_logs.cpp
//
//  XY-on-SAW post-processor.
//
//  Original observables (preserved):
//      magnetization, end-to-end R^2, mean pairwise r^2, factors,
//      gyration radius, topological contacts.
//
//  NEW observables (added for BKT diagnostics):
//      * Along-chain spin correlation C(d) = <cos(theta_i - theta_{i+d})>
//      * Helicity-modulus building blocks K_alpha, I_alpha per Cartesian
//        axis alpha = x, y[, z], computed for
//            - backbone bonds only           (bb_*)
//            - backbone + topological contacts (full_*)
//
//      IMPORTANT: for an open SAW blob this is NOT a chain-end twist.
//     These are spatial-twist building blocks on embedded lattice bonds.
//     The correct reduced stiffness tensor is obtained AFTER averaging:
//
//        betaY_ab = (1/A_eff) [
//             J <K_ab> - J^2 ( <I_a I_b> - <I_a><I_b> )
//        ]
//
//     If your J is already beta*coupling, betaY_ab is the dimensionless
//     stiffness. If you need physical Upsilon_ab, multiply betaY_ab by T.
//     Do not average the per-sample uncentered betaY pieces and call it a day;
//     that drops the <I_a><I_b> covariance correction, because apparently
//     even CSV files enjoy sabotaging science.
//
//     For open geometry, a true helicity modulus should still be interpreted
//     as a geometric boundary-twist response. The K/I/II variables here are
//     the correct local building blocks, not endpoint-twist variables.
//
//  Spatial correlations:
//      Binned by integer Euclidean distance d = round(sqrt(dx^2+dy^2+dz^2)).
//      One column per integer d in [1, rmax]. Per-sample mean C_sp(d) and
//      pair count N(d) are written; the count column lets you reconstruct
//      <S>/<N> later if you ever need ratio-of-means.
//
//      DO NOT use Manhattan distance for BKT power-law fits: it smears pairs
//      at different Euclidean distances into the same bin and renormalizes
//      eta(T) in a way that looks plausible but is wrong.
//
//  Build:
//      g++ -O3 -std=c++17 -fopenmp process_saw_logs.cpp -o process_saw_logs
//
//  Usage:
//      ./process_saw_logs --angles angles.txt --dirs dirs.txt \
//                         --L 100 --dim 3 --side 64 --outdir ./out \
//                         [--chain -1] [--threads 8] [--dmax -1] [--rmax -1] \
//                         [--periodic-lattice] [--no-spatial-corr] \
//                         [--no-angles] [--no-dirs] [--no-joint]
// =====================================================================

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <filesystem>
#include <limits>
#include <algorithm>
#include <array>

#ifdef _OPENMP
  #include <omp.h>
#endif

// ====================== FastScanner (unchanged) ======================
class FastScanner {
public:
  explicit FastScanner(const std::string& path, size_t buf_sz = (1u << 20))
    : in_(path, std::ios::binary), buf_(buf_sz), i_(0), n_(0) {
    if (!in_) throw std::runtime_error("Cannot open file: " + path);
  }

  bool readLongLong(long long& out) { return readSignedIntegral<long long>(out); }
  bool readInt(int& out)            { return readSignedIntegral<int>(out); }

  bool readDouble(double& out) {
    skipSpaces();
    char c = peek();
    if (!c) return false;

    bool neg = false;
    if (c == '-') { neg = true; get(); }
    else if (c == '+') { get(); }

    double val = 0.0;
    bool any = false;
    while (true) {
      c = peek();
      if (c >= '0' && c <= '9') {
        any = true;
        val = val * 10.0 + double(get() - '0');
      } else break;
    }

    c = peek();
    if (c == '.') {
      get();
      double place = 1.0;
      while (true) {
        c = peek();
        if (c >= '0' && c <= '9') {
          any = true;
          place *= 0.1;
          val += place * double(get() - '0');
        } else break;
      }
    }
    if (!any) return false;

    c = peek();
    if (c == 'e' || c == 'E') {
      get();
      bool eneg = false;
      c = peek();
      if (c == '-') { eneg = true; get(); }
      else if (c == '+') { get(); }

      int expv = 0; bool eany = false;
      while (true) {
        c = peek();
        if (c >= '0' && c <= '9') {
          eany = true;
          expv = expv * 10 + int(get() - '0');
        } else break;
      }
      if (eany) {
        double pow10 = std::pow(10.0, double(expv));
        val = eneg ? (val / pow10) : (val * pow10);
      }
    }

    out = neg ? -val : val;
    return true;
  }

private:
  std::ifstream in_;
  std::vector<char> buf_;
  size_t i_, n_;

  inline bool refill() {
    if (i_ < n_) return true;
    in_.read(buf_.data(), std::streamsize(buf_.size()));
    n_ = size_t(in_.gcount());
    i_ = 0;
    return n_ != 0;
  }
  inline char peek() { if (!refill()) return 0; return buf_[i_]; }
  inline char get()  { if (!refill()) return 0; return buf_[i_++]; }
  inline void skipSpaces() {
    while (true) {
      char c = peek();
      if (!c) return;
      if (c==' '||c=='\n'||c=='\r'||c=='\t'||c=='\v'||c=='\f') get();
      else return;
    }
  }
  template<class T>
  bool readSignedIntegral(T& out) {
    skipSpaces();
    char c = peek(); if (!c) return false;
    bool neg = false;
    if (c == '-') { neg = true; get(); }
    else if (c == '+') { get(); }
    using U = std::make_unsigned_t<T>;
    U val = 0; bool any = false;
    while (true) {
      c = peek();
      if (c >= '0' && c <= '9') {
        any = true;
        val = val * 10 + U(get() - '0');
      } else break;
    }
    if (!any) return false;
    if (neg) out = T(0) - T(val); else out = T(val);
    return true;
  }
};

// ====================== Torus helpers (unchanged) ======================
static inline int min_image_abs(int a, int b, int side) {
  int d = std::abs(b - a);
  int half = side / 2;
  if (d > half) d = side - d;
  return d;
}
static inline int min_image_signed(int a, int b, int side) {
  int d = b - a;
  int half = side / 2;
  if (d >  half) d -= side;
  if (d < -half) d += side;
  return d;
}
struct Vec3i  { int x=0,y=0,z=0; };
struct Vec3ll { long long x=0,y=0,z=0; };


struct CoordKey {
  int x=0, y=0, z=0;
  bool operator==(const CoordKey& o) const noexcept {
    return x == o.x && y == o.y && z == o.z;
  }
};
struct CoordKeyHash {
  size_t operator()(const CoordKey& k) const noexcept {
    // SplitMix-style integer hashing. Works for negative unwrapped coordinates too.
    auto mix = [](uint64_t x) {
      x += 0x9e3779b97f4a7c15ULL;
      x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
      x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
      return x ^ (x >> 31);
    };
    uint64_t hx = mix((uint64_t)(int64_t)k.x);
    uint64_t hy = mix((uint64_t)(int64_t)k.y + 0x9e3779b97f4a7c15ULL);
    uint64_t hz = mix((uint64_t)(int64_t)k.z + 0xbf58476d1ce4e5b9ULL);
    return (size_t)(hx ^ (hy << 1) ^ (hz << 2));
  }
};
static inline CoordKey coord_key(const Vec3i& p) { return CoordKey{p.x, p.y, p.z}; }

// The walk may be stored modulo a large torus even when the physical object is open.
// This reconstructs the open embedded coordinates by following the backbone with
// minimum-image steps. For a nearest-neighbor SAW this turns a stored jump like
// x = side-1 -> 0 into an ordinary +1 step in the unwrapped embedding.
static inline void unwrap_chain_from_torus_storage(const std::vector<Vec3i>& wrapped,
                                                   int side, int dim,
                                                   std::vector<Vec3i>& unwrapped) {
  const int L = (int)wrapped.size();
  unwrapped.assign((size_t)L, Vec3i{});
  if (L == 0) return;
  unwrapped[0] = { wrapped[0].x, wrapped[0].y, (dim == 3 ? wrapped[0].z : 0) };
  for (int i = 1; i < L; ++i) {
    const int dx = min_image_signed(wrapped[i-1].x, wrapped[i].x, side);
    const int dy = min_image_signed(wrapped[i-1].y, wrapped[i].y, side);
    const int dz = (dim == 3) ? min_image_signed(wrapped[i-1].z, wrapped[i].z, side) : 0;
    unwrapped[i].x = unwrapped[i-1].x + dx;
    unwrapped[i].y = unwrapped[i-1].y + dy;
    unwrapped[i].z = unwrapped[i-1].z + dz;
  }
}

static inline double direct_r2(const Vec3i& a, const Vec3i& b, int dim) {
  const long long dx = (long long)b.x - (long long)a.x;
  const long long dy = (long long)b.y - (long long)a.y;
  const long long dz = (dim == 3) ? ((long long)b.z - (long long)a.z) : 0LL;
  return double(dx*dx + dy*dy + dz*dz);
}

static inline long long site_key_2d(int x, int y, int side) {
  return (long long)x + (long long)side * (long long)y;
}
static inline long long site_key_3d(int x, int y, int z, int side) {
  return (long long)x + (long long)side * ((long long)y + (long long)side * (long long)z);
}
static inline int wrap(int a, int side) {
  int r = a % side; if (r < 0) r += side; return r;
}

// ====================== Existing metrics (unchanged) ======================
static inline double torus_r2(const Vec3i& a, const Vec3i& b, int side, int dim) {
  int dx = min_image_abs(a.x, b.x, side);
  int dy = min_image_abs(a.y, b.y, side);
  int dz = (dim == 3) ? min_image_abs(a.z, b.z, side) : 0;
  return double(dx*dx + dy*dy + dz*dz);
}

static inline void pairwise_stats_exact(const std::vector<Vec3i>& chain, int side, int dim,
                                        bool periodic_lattice,
                                        double& mean_r2, double& factors_divN) {
  const int L = (int)chain.size();
  long long count = 0;
  double sum_r2 = 0.0;
  double sum_inv_r3 = 0.0;

  #ifdef _OPENMP
  #pragma omp parallel for reduction(+:sum_r2,sum_inv_r3,count) schedule(static)
  #endif
  for (int i = 0; i < L - 1; ++i) {
    for (int j = i + 1; j < L; ++j) {
      double r2 = periodic_lattice ? torus_r2(chain[i], chain[j], side, dim)
                                   : direct_r2(chain[i], chain[j], dim);
      if (r2 > 0.0) {
        sum_r2 += r2;
        sum_inv_r3 += 1.0 / (std::sqrt(r2) * r2);
        count += 1;
      }
    }
  }
  mean_r2 = (count > 0) ? (sum_r2 / double(count)) : 0.0;
  factors_divN = (L > 0) ? (sum_inv_r3 / double(L)) : 0.0;
}

static inline void gyration_radius_unwrapped(const std::vector<Vec3i>& chain, int side, int dim,
                                             double& Rg2, double& Rg) {
  const int L = (int)chain.size();
  if (L == 0) { Rg2 = 0.0; Rg = 0.0; return; }
  std::vector<Vec3ll> u(L);
  u[0] = { chain[0].x, chain[0].y, (dim==3 ? chain[0].z : 0) };
  for (int i = 1; i < L; ++i) {
    int dx = min_image_signed(chain[i-1].x, chain[i].x, side);
    int dy = min_image_signed(chain[i-1].y, chain[i].y, side);
    int dz = (dim==3) ? min_image_signed(chain[i-1].z, chain[i].z, side) : 0;
    u[i].x = u[i-1].x + dx; u[i].y = u[i-1].y + dy; u[i].z = u[i-1].z + dz;
  }
  long double mx=0, my=0, mz=0;
  for (int i = 0; i < L; ++i) { mx += u[i].x; my += u[i].y; mz += u[i].z; }
  mx /= (long double)L; my /= (long double)L; mz /= (long double)L;
  long double sum = 0;
  for (int i = 0; i < L; ++i) {
    long double dx = (long double)u[i].x - mx;
    long double dy = (long double)u[i].y - my;
    long double dz = (long double)u[i].z - mz;
    if (dim == 2) dz = 0;
    sum += dx*dx + dy*dy + dz*dz;
  }
  Rg2 = double(sum / (long double)L);
  Rg  = std::sqrt(Rg2);
}

static inline long long topological_contacts(const std::vector<Vec3i>& chain, int side, int dim,
                                             bool periodic_lattice) {
  const int L = (int)chain.size();
  if (L < 3) return 0;

  std::unordered_map<CoordKey, int, CoordKeyHash> occ;
  occ.reserve((size_t)L * 2);
  for (int i = 0; i < L; ++i) occ[coord_key(chain[i])] = i;

  const int dirs2d[4][3] = { {+1,0,0},{-1,0,0},{0,+1,0},{0,-1,0} };
  const int dirs3d[6][3] = { {+1,0,0},{-1,0,0},{0,+1,0},{0,-1,0},{0,0,+1},{0,0,-1} };

  long long contacts = 0;
  for (int i = 0; i < L; ++i) {
    const Vec3i& p = chain[i];
    const int nnb = (dim == 2) ? 4 : 6;
    for (int k = 0; k < nnb; ++k) {
      const int dx = (dim == 2) ? dirs2d[k][0] : dirs3d[k][0];
      const int dy = (dim == 2) ? dirs2d[k][1] : dirs3d[k][1];
      const int dz = (dim == 2) ? 0          : dirs3d[k][2];
      Vec3i q;
      q.x = p.x + dx;
      q.y = p.y + dy;
      q.z = (dim == 3) ? (p.z + dz) : 0;
      if (periodic_lattice) {
        q.x = wrap(q.x, side);
        q.y = wrap(q.y, side);
        if (dim == 3) q.z = wrap(q.z, side);
      }
      auto it = occ.find(coord_key(q));
      if (it == occ.end()) continue;
      const int j = it->second;
      // Exclude backbone neighbors and count every non-bonded NN pair once.
      if (j > i + 1) contacts += 1;
    }
  }
  return contacts;
}

// ============================================================
// Spin-spin correlations
// ============================================================
//
// Contour-distance correlator (along the SAW):
//
//   C_chain(d) = (1/(L-d)) sum_i cos(theta_{i+d} - theta_i)
//
// This is useful for polymer/contour physics, but it is NOT the usual spatial
// XY correlation used in BKT analysis.
//
// Spatial correlator, binned by integer Euclidean distance
//   d = round(sqrt(dx^2 + dy^2 + dz^2)):
//
//   N(d) = number of pairs with this rounded distance
//   S(d) = sum over those pairs of cos(theta_j - theta_i)
//   C_sp(d) = S(d) / N(d)        (per-sample mean)
//
// Reasoning for integer Euclidean (rather than per-r^2-shell):
//   * BKT predicts C(r) ~ r^{-eta(T)} in EUCLIDEAN distance, so the bin label
//     should approximate Euclidean r. round(sqrt(r2)) gives r within +-0.5
//     of the label and groups dense neighboring r^2 shells, which kills the
//     "few-pair, noisy" shells without distorting the power law.
//   * Manhattan distance |dx|+|dy| is the wrong choice: it pools (3,0) and
//     (2,1) (Euclidean 3 and sqrt(5)) into the same bin, renormalizing eta.
//
// We save C_sp(d) and N(d) per sample. Per-sample C_sp = S/N is unambiguous;
// keeping N as well lets you form the ratio-of-means <S>/<N> across samples
// later if you want to (relevant only when N(d) fluctuates strongly with
// chain geometry, i.e. at d near rmax).
//
static void compute_chain_correlations(const std::vector<double>& angles_chain,
                                       std::vector<double>& Cd, int dmax) {
  const int L = (int)angles_chain.size();
  const int D = std::min(dmax, L - 1);
  Cd.assign(D, 0.0);

  #ifdef _OPENMP
  #pragma omp parallel for schedule(static)
  #endif
  for (int d = 1; d <= D; ++d) {
    double sum = 0.0;
    int count = 0;
    for (int i = 0; i + d < L; ++i) {
      sum += std::cos(angles_chain[i + d] - angles_chain[i]);
      ++count;
    }
    Cd[d - 1] = (count > 0) ? sum / double(count) : 0.0;
  }
}

static void compute_spatial_correlations_radial(
    const std::vector<Vec3i>& chain,
    const std::vector<double>& angles_chain,
    int side, int dim, int rmax,
    std::vector<double>& mean_per_d,
    std::vector<long long>& count_per_d,
    bool periodic_lattice)
{
  const int L = (int)chain.size();
  const size_t nb = (size_t)rmax + 1;       // indices 1..rmax used; index 0 unused
  std::vector<double> sum_per_d(nb, 0.0);
  count_per_d.assign(nb, 0);
  mean_per_d.assign(nb, 0.0);
  if (rmax <= 0 || L < 2) return;

#ifdef _OPENMP
  const int nthreads = omp_get_max_threads();
  std::vector<std::vector<double>>    ls((size_t)nthreads, std::vector<double>(nb, 0.0));
  std::vector<std::vector<long long>> lc((size_t)nthreads, std::vector<long long>(nb, 0));

  #pragma omp parallel
  {
    const int tid = omp_get_thread_num();
    auto& s = ls[(size_t)tid];
    auto& c = lc[(size_t)tid];

    #pragma omp for schedule(static)
    for (int i = 0; i < L - 1; ++i) {
      for (int j = i + 1; j < L; ++j) {
        const int dx = periodic_lattice ? min_image_signed(chain[i].x, chain[j].x, side)
                                        : (chain[j].x - chain[i].x);
        const int dy = periodic_lattice ? min_image_signed(chain[i].y, chain[j].y, side)
                                        : (chain[j].y - chain[i].y);
        const int dz = (dim == 3)
                       ? (periodic_lattice ? min_image_signed(chain[i].z, chain[j].z, side)
                                           : (chain[j].z - chain[i].z))
                       : 0;
        const long long r2 = (long long)dx*dx + (long long)dy*dy + (long long)dz*dz;
        if (r2 <= 0) continue;
        const int d = (int)std::lround(std::sqrt((double)r2));
        if (d <= 0 || d > rmax) continue;
        s[(size_t)d] += std::cos(angles_chain[j] - angles_chain[i]);
        c[(size_t)d] += 1;
      }
    }
  }

  for (int t = 0; t < nthreads; ++t) {
    for (size_t d = 1; d < nb; ++d) {
      sum_per_d[d]   += ls[(size_t)t][d];
      count_per_d[d] += lc[(size_t)t][d];
    }
  }
#else
  for (int i = 0; i < L - 1; ++i) {
    for (int j = i + 1; j < L; ++j) {
      const int dx = periodic_lattice ? min_image_signed(chain[i].x, chain[j].x, side)
                                      : (chain[j].x - chain[i].x);
      const int dy = periodic_lattice ? min_image_signed(chain[i].y, chain[j].y, side)
                                      : (chain[j].y - chain[i].y);
      const int dz = (dim == 3)
                     ? (periodic_lattice ? min_image_signed(chain[i].z, chain[j].z, side)
                                         : (chain[j].z - chain[i].z))
                     : 0;
      const long long r2 = (long long)dx*dx + (long long)dy*dy + (long long)dz*dz;
      if (r2 <= 0) continue;
      const int d = (int)std::lround(std::sqrt((double)r2));
      if (d <= 0 || d > rmax) continue;
      sum_per_d[(size_t)d]   += std::cos(angles_chain[j] - angles_chain[i]);
      count_per_d[(size_t)d] += 1;
    }
  }
#endif

  for (size_t d = 1; d < nb; ++d) {
    mean_per_d[d] = (count_per_d[d] > 0)
                  ? sum_per_d[d] / double(count_per_d[d])
                  : std::numeric_limits<double>::quiet_NaN();
  }
}

// ============================================================
// Helicity-modulus building blocks
// ============================================================
//
// For a reduced Hamiltonian/action
//
//      S = -J * sum_b cos(theta_i - theta_j)
//
// where J is the coupling used in the Boltzmann weight, the reduced stiffness
// tensor is
//
//      beta Upsilon_ab = (1/N) [
//          J <K_ab> - J^2 ( <I_a I_b> - <I_a><I_b> )
//      ],
//
// with
//
//      K_ab = sum_bonds dr_a dr_b cos(dtheta)
//      I_a  = sum_bonds dr_a sin(dtheta).
//
// The previous code stored only the diagonal K_xx,K_yy,K_zz and I_x,I_y,I_z.
// That is enough only for diagonal components and only after post-processing
// I_a I_b moments. Here we store the full K tensor and write per-sample
// I_a I_b and uncentered beta*Upsilon observables too.
//
// For your present NN square-lattice bonds, K_xy is usually exactly zero
// because dx*dy=0 per bond. We still write it, because future-you will
// otherwise rediscover this trap with long-range bonds at 3 a.m.
//
struct HelicityTerms {
  double K[3][3] = {{0.0, 0.0, 0.0},
                    {0.0, 0.0, 0.0},
                    {0.0, 0.0, 0.0}};
  double I[3] = {0.0, 0.0, 0.0};
  int n_bonds = 0;
};

static inline void accumulate_bond(HelicityTerms& t,
                                   int dx, int dy, int dz, double dth,
                                   double weight = 1.0) {
  const double c = std::cos(dth);
  const double s = std::sin(dth);
  const double dr[3] = { double(dx), double(dy), double(dz) };

  for (int a = 0; a < 3; ++a) {
    t.I[a] += weight * dr[a] * s;
    for (int b = 0; b < 3; ++b) {
      t.K[a][b] += weight * dr[a] * dr[b] * c;
    }
  }
  t.n_bonds += 1;
}

static void compute_helicity_terms(const std::vector<Vec3i>& chain,
                                   const std::vector<double>& angles_chain,
                                   int side, int dim,
                                   HelicityTerms& bb_only,
                                   HelicityTerms& bb_plus_contacts,
                                   bool periodic_lattice) {
  const int L = (int)chain.size();
  bb_only = HelicityTerms{};

  // --- backbone bonds (chain order) ---
  for (int i = 0; i + 1 < L; ++i) {
    int dx = periodic_lattice ? min_image_signed(chain[i].x, chain[i + 1].x, side) : (chain[i + 1].x - chain[i].x);
    int dy = periodic_lattice ? min_image_signed(chain[i].y, chain[i + 1].y, side) : (chain[i + 1].y - chain[i].y);
    int dz = (dim == 3) ? (periodic_lattice ? min_image_signed(chain[i].z, chain[i + 1].z, side) : (chain[i + 1].z - chain[i].z)) : 0;
    double dth = angles_chain[i + 1] - angles_chain[i];
    accumulate_bond(bb_only, dx, dy, dz, dth);
  }

  // --- backbone + contacts ---
  bb_plus_contacts = bb_only;

  if (L >= 3) {
    std::unordered_map<CoordKey, int, CoordKeyHash> occ;
    occ.reserve((size_t)L * 2);
    for (int i = 0; i < L; ++i) occ[coord_key(chain[i])] = i;

    const int dirs2d[4][3] = { {+1,0,0},{-1,0,0},{0,+1,0},{0,-1,0} };
    const int dirs3d[6][3] = { {+1,0,0},{-1,0,0},{0,+1,0},{0,-1,0},{0,0,+1},{0,0,-1} };

    for (int i = 0; i < L; ++i) {
      const Vec3i& p = chain[i];
      const int nnb = (dim == 2) ? 4 : 6;
      for (int k = 0; k < nnb; ++k) {
        int dx = (dim == 2) ? dirs2d[k][0] : dirs3d[k][0];
        int dy = (dim == 2) ? dirs2d[k][1] : dirs3d[k][1];
        int dz = (dim == 2) ? 0            : dirs3d[k][2];
        Vec3i q;
        q.x = p.x + dx;
        q.y = p.y + dy;
        q.z = (dim == 3) ? (p.z + dz) : 0;
        if (periodic_lattice) {
          q.x = wrap(q.x, side);
          q.y = wrap(q.y, side);
          if (dim == 3) q.z = wrap(q.z, side);
        }
        auto it = occ.find(coord_key(q));
        if (it == occ.end()) continue;
        int j = it->second;
        // Only contact bonds: exclude backbone (j == i+1, j == i-1) and
        // count each pair once via the j > i+1 rule.
        if (j > i + 1) {
          double dth = angles_chain[j] - angles_chain[i];
          accumulate_bond(bb_plus_contacts, dx, dy, dz, dth);
        }
      }
    }
  }
}

// ====================== Argument parsing ======================
struct Args {
  std::string angles_file;
  std::string dirs_file;
  std::string outdir = ".";
  int L = -1;
  int dim = 3;
  int side = -1;
  int chain = -1;
  int threads = -1;
  int dmax = -1;            // max contour distance for C_chain(d), -1 -> L-1
  int rmax = -1;            // max spatial distance for C_sp(d), -1 -> min(side/2, 200)
  bool run_angles = true;
  bool run_dirs   = true;
  bool run_joint  = true;
  bool run_spatial_corr = true;
  bool periodic_lattice = false; // default: open embedded SAW; use --periodic-lattice only for torus data
};

static inline void usage_and_exit() {
  std::cerr <<
    "Usage:\n"
    "  process_saw_logs --angles <file> --dirs <file> --L <L> --dim <2|3> --side <side>\n"
    "                   [--outdir <dir>] [--chain <k|-1>] [--threads <n>]\n"
    "                   [--dmax <m>] [--rmax <r>] [--periodic-lattice] [--no-spatial-corr]\n"
    "                   [--no-angles] [--no-dirs] [--no-joint]\n\n"
    "Notes:\n"
    "  C_chain(d) is contour distance along the SAW.\n"
    "  C_sp(d) is spatial correlator binned by integer Euclidean distance d = round(sqrt(r^2)).\n"
    "  Helicity output includes K_ab, I_a, I_a I_b building blocks; centered stiffness must be formed after averaging.\n";
  std::exit(1);
}

static inline Args parse_args(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    std::string s = argv[i];
    auto need = [&](const char* name) {
      if (i + 1 >= argc) { std::cerr << "Missing value for " << name << "\n"; usage_and_exit(); }
      return std::string(argv[++i]);
    };
    if      (s == "--angles")     a.angles_file = need("--angles");
    else if (s == "--dirs")       a.dirs_file   = need("--dirs");
    else if (s == "--outdir")     a.outdir      = need("--outdir");
    else if (s == "--L")          a.L           = std::stoi(need("--L"));
    else if (s == "--dim")        a.dim         = std::stoi(need("--dim"));
    else if (s == "--side")       a.side        = std::stoi(need("--side"));
    else if (s == "--chain")      a.chain       = std::stoi(need("--chain"));
    else if (s == "--threads")    a.threads     = std::stoi(need("--threads"));
    else if (s == "--dmax")       a.dmax        = std::stoi(need("--dmax"));
    else if (s == "--rmax")       a.rmax        = std::stoi(need("--rmax"));
    else if (s == "--no-angles")  a.run_angles  = false;
    else if (s == "--no-dirs")    a.run_dirs    = false;
    else if (s == "--no-joint")   a.run_joint   = false;
    else if (s == "--no-spatial-corr") a.run_spatial_corr = false;
    else if (s == "--periodic-lattice") a.periodic_lattice = true;
    else { std::cerr << "Unknown arg: " << s << "\n"; usage_and_exit(); }
  }
  if (a.angles_file.empty() || a.dirs_file.empty() || a.L <= 0
      || (a.dim != 2 && a.dim != 3) || a.side <= 0) {
    usage_and_exit();
  }
  return a;
}

static inline std::string file_source_tag(const std::string& path) {
  std::filesystem::path p(path);
  auto parent = p.parent_path();
  if (parent.empty()) return "";
  return parent.filename().string();
}

// ====================== Original: angles ======================
static void process_angles(const Args& args) {
  std::filesystem::create_directories(args.outdir);
  const std::string out_path =
      (std::filesystem::path(args.outdir) /
       ("Data_TimeSeries_" + std::to_string(args.L) + ".csv")).string();
  std::ofstream out(out_path);
  if (!out) throw std::runtime_error("Cannot open output: " + out_path);
  out << "L,J,mag,E,step,source\n";

  FastScanner fs(args.angles_file);
  const std::string source = file_source_tag(args.angles_file);

  long long step = 0; double J = 0.0, E = 0.0;
  long long prev_step = std::numeric_limits<long long>::min();
  int line_in_block = 0;

  while (true) {
    if (!fs.readLongLong(step)) break;
    if (!fs.readDouble(J)) break;
    if (!fs.readDouble(E)) break;

    if (step != prev_step) { prev_step = step; line_in_block = 0; }
    int chain_id = line_in_block++;

    double sin_sum = 0.0, cos_sum = 0.0;
    for (int i = 0; i < args.L; ++i) {
      double th = 0.0;
      if (!fs.readDouble(th)) th = 0.0;
      sin_sum += std::sin(th);
      cos_sum += std::cos(th);
    }
    if (args.chain >= 0 && chain_id != args.chain) continue;

    const double mx = sin_sum / args.L, my = cos_sum / args.L;
    const double mag = std::sqrt(mx*mx + my*my);
    out << args.L << "," << J << "," << mag << "," << E << "," << step << "," << source << "\n";
  }
  std::cerr << "Wrote: " << out_path << "\n";
}

// ====================== Original: dirs ======================
static void process_dirs(const Args& args) {
  std::filesystem::create_directories(args.outdir);
  const std::string out_path =
      (std::filesystem::path(args.outdir) /
       ("Data_TimeSeries_dir_" + std::to_string(args.L) + ".csv")).string();
  std::ofstream out(out_path);
  if (!out) throw std::runtime_error("Cannot open output: " + out_path);
  out << "L,J,R2,mean_R_ij,factors,Rg2,Rg,contacts,start_idx,E,step,source\n";

  FastScanner fs(args.dirs_file);
  const std::string source = file_source_tag(args.dirs_file);

  long long step = 0; double J = 0.0, E = 0.0; int start_idx = 0;
  long long prev_step = std::numeric_limits<long long>::min();
  int line_in_block = 0;
  std::vector<Vec3i> ring(args.L);

  while (true) {
    if (!fs.readLongLong(step)) break;
    if (!fs.readDouble(J)) break;
    if (!fs.readInt(start_idx)) break;
    if (!fs.readDouble(E)) break;

    if (step != prev_step) { prev_step = step; line_in_block = 0; }
    int chain_id = line_in_block++;

    for (int i = 0; i < args.L; ++i) {
      int x=0,y=0,z=0;
      if (!fs.readInt(x)) x=0;
      if (!fs.readInt(y)) y=0;
      if (args.dim == 3) { if (!fs.readInt(z)) z=0; } else z = 0;
      ring[i] = { x, y, z };
    }
    if (args.chain >= 0 && chain_id != args.chain) continue;

    const int s = ((start_idx % args.L) + args.L) % args.L;
    std::vector<Vec3i> chain(args.L), chain_geom(args.L);
    for (int t = 0; t < args.L; ++t) chain[t] = ring[(s + t) % args.L];

    if (args.periodic_lattice) chain_geom = chain;
    else unwrap_chain_from_torus_storage(chain, args.side, args.dim, chain_geom);

    const Vec3i& startp = chain_geom[0];
    const Vec3i& endp   = chain_geom[args.L - 1];
    const double R2 = args.periodic_lattice ? torus_r2(chain[0], chain[args.L - 1], args.side, args.dim)
                                            : direct_r2(startp, endp, args.dim);

    double mean_r2 = 0.0, factors = 0.0;
    pairwise_stats_exact(chain_geom, args.side, args.dim, args.periodic_lattice, mean_r2, factors);
    double Rg2 = 0.0, Rg = 0.0;
    gyration_radius_unwrapped(chain, args.side, args.dim, Rg2, Rg);
    const long long contacts = topological_contacts(chain_geom, args.side, args.dim, args.periodic_lattice);

    out << args.L << "," << J << "," << R2 << "," << mean_r2 << "," << factors << ","
        << Rg2 << "," << Rg << "," << contacts << ","
        << start_idx << "," << E << "," << step << "," << source << "\n";
  }
  std::cerr << "Wrote: " << out_path << "\n";
}

// ============================================================
// NEW: joint processor (correlations + helicity)
// ============================================================
//
// Both files are read in lockstep, one chain at a time. We assume both
// were emitted at the same MC samples and in the same chain order.
// Angles and positions share the same "ring-buffer" indexing; start_idx
// (logged in the dirs file) rotates BOTH into chain order.
//
static void write_helicity_terms(std::ofstream& out,
                                 const HelicityTerms& h,
                                 double J,
                                 double norm,
                                 int naxes) {
  // Backward-compatible diagonal columns: K_x,K_y[,K_z], I_x,I_y[,I_z]
  for (int a = 0; a < naxes; ++a) out << "," << h.K[a][a];
  for (int a = 0; a < naxes; ++a) out << "," << h.I[a];

  // Full symmetric tensor K_ab, current products I_a I_b,
  // and uncentered per-sample beta*Y_ab observable.
  for (int a = 0; a < naxes; ++a) {
    for (int b = a; b < naxes; ++b) out << "," << h.K[a][b];
  }
  for (int a = 0; a < naxes; ++a) {
    for (int b = a; b < naxes; ++b) out << "," << (h.I[a] * h.I[b]);
  }
  for (int a = 0; a < naxes; ++a) {
    for (int b = a; b < naxes; ++b) {
      const double betaYpiece_uncentered = (J * h.K[a][b] - J * J * h.I[a] * h.I[b]) / norm;
      out << "," << betaYpiece_uncentered;
    }
  }
}

static void process_joint(const Args& args) {
  std::filesystem::create_directories(args.outdir);

  const std::string hel_path =
      (std::filesystem::path(args.outdir) /
       ("Data_Helicity_" + std::to_string(args.L) + ".csv")).string();
  const std::string cor_chain_path =
      (std::filesystem::path(args.outdir) /
       ("Data_CorrChain_" + std::to_string(args.L) + ".csv")).string();
  const std::string cor_sp_path =
      (std::filesystem::path(args.outdir) /
       ("Data_CorrSpatial_" + std::to_string(args.L) + ".csv")).string();

  std::ofstream out_hel(hel_path);
  std::ofstream out_chain(cor_chain_path);
  std::ofstream out_sp;
  if (args.run_spatial_corr) out_sp.open(cor_sp_path);
  if (!out_hel)   throw std::runtime_error("Cannot open: " + hel_path);
  if (!out_chain) throw std::runtime_error("Cannot open: " + cor_chain_path);
  if (args.run_spatial_corr && !out_sp) throw std::runtime_error("Cannot open: " + cor_sp_path);

  const int dmax = (args.dmax > 0) ? std::min(args.dmax, args.L - 1) : (args.L - 1);
  const int rmax_default = std::min(args.side / 2, 200);
  const int rmax = (args.rmax > 0) ? std::min(args.rmax, args.side / 2) : rmax_default;

  const int naxes = (args.dim == 2) ? 2 : 3;
  const char* axes2[2] = { "x", "y" };
  const char* axes3[3] = { "x", "y", "z" };
  auto axis = [&](int a) -> const char* { return (args.dim == 2) ? axes2[a] : axes3[a]; };

  auto pair_name = [&](int a, int b) -> std::string {
    return std::string(axis(a)) + axis(b);
  };

  // --- helicity CSV header ---
  out_hel << "L,J,step,source,n_bb_bonds,n_contact_bonds,norm_spins,E";
  for (const char* flavor : {"bb", "full"}) {
    for (int a = 0; a < naxes; ++a) out_hel << ",K_" << flavor << "_" << axis(a);       // old diagonal name
    for (int a = 0; a < naxes; ++a) out_hel << ",I_" << flavor << "_" << axis(a);       // old current name
    for (int a = 0; a < naxes; ++a)
      for (int b = a; b < naxes; ++b) out_hel << ",Ktensor_" << flavor << "_" << pair_name(a,b);
    for (int a = 0; a < naxes; ++a)
      for (int b = a; b < naxes; ++b) out_hel << ",II_" << flavor << "_" << pair_name(a,b);
    for (int a = 0; a < naxes; ++a)
      for (int b = a; b < naxes; ++b) out_hel << ",betaYpiece_uncentered_" << flavor << "_" << pair_name(a,b);
  }
  out_hel << "\n";

  // --- contour correlation CSV header ---
  out_chain << "L,J,step,source,E";
  for (int d = 1; d <= dmax; ++d) out_chain << ",Cchain_d" << d;
  out_chain << "\n";

  // --- spatial correlation CSV header ---
  // One column per integer Euclidean distance d in [1, rmax]:
  //   Csp_d<d>  -> per-sample mean correlator C_sp(d) = S(d)/N(d)
  //   Ncnt_d<d> -> per-sample pair count N(d) (lets you form <S>/<N> later)
  if (args.run_spatial_corr) {
    out_sp << "L,J,step,source,E,rmax";
    for (int d = 1; d <= rmax; ++d) out_sp << ",Csp_d"  << d;
    for (int d = 1; d <= rmax; ++d) out_sp << ",Ncnt_d" << d;
    out_sp << "\n";
  }

  FastScanner fs_a(args.angles_file);
  FastScanner fs_d(args.dirs_file);
  const std::string source = file_source_tag(args.dirs_file);

  std::vector<Vec3i>   ring(args.L),  chain(args.L), chain_geom(args.L);
  std::vector<double>  angles_ring(args.L), angles_chain(args.L);
  std::vector<double>  Cd;
  std::vector<double>    mean_d;
  std::vector<long long> cnt_d;

  long long prev_step = std::numeric_limits<long long>::min();
  int line_in_block = 0;

  long long step_a = 0, step_d = 0;
  double    J_a = 0.0, J_d = 0.0, E_a = 0.0, E_d = 0.0;
  int       start_idx = 0;

  long long n_step_mismatch = 0;
  long long n_J_mismatch = 0;

  while (true) {
    // --- angles block ---
    if (!fs_a.readLongLong(step_a)) break;
    if (!fs_a.readDouble(J_a))      break;
    if (!fs_a.readDouble(E_a))      break;
    for (int i = 0; i < args.L; ++i) {
      if (!fs_a.readDouble(angles_ring[i])) angles_ring[i] = 0.0;
    }

    // --- dirs block ---
    if (!fs_d.readLongLong(step_d)) {
      std::cerr << "WARN: dirs file shorter than angles file (at angles step " << step_a << ")\n";
      break;
    }
    if (!fs_d.readDouble(J_d))   break;
    if (!fs_d.readInt(start_idx)) break;
    if (!fs_d.readDouble(E_d))   break;
    for (int i = 0; i < args.L; ++i) {
      int x=0,y=0,z=0;
      if (!fs_d.readInt(x)) x=0;
      if (!fs_d.readInt(y)) y=0;
      if (args.dim == 3) { if (!fs_d.readInt(z)) z=0; } else z = 0;
      ring[i] = { x, y, z };
    }

    if (step_a != step_d) {
      if (n_step_mismatch < 5) {
        std::cerr << "WARN: step mismatch angles=" << step_a
                  << " dirs=" << step_d << " -- check that logs are aligned\n";
      }
      ++n_step_mismatch;
    }
    if (std::abs(J_a - J_d) > 1e-12) {
      if (n_J_mismatch < 5) {
        std::cerr << "WARN: J mismatch angles=" << J_a
                  << " dirs=" << J_d << " -- using dirs J in output\n";
      }
      ++n_J_mismatch;
    }

    if (step_d != prev_step) { prev_step = step_d; line_in_block = 0; }
    const int chain_id = line_in_block++;
    if (args.chain >= 0 && chain_id != args.chain) continue;

    // Rotate by start_idx so that index 0 is the chain's start.
    const int s = ((start_idx % args.L) + args.L) % args.L;
    for (int t = 0; t < args.L; ++t) {
      const int idx = (s + t) % args.L;
      chain[t]        = ring[idx];
      angles_chain[t] = angles_ring[idx];
    }

    if (args.periodic_lattice) chain_geom = chain;
    else unwrap_chain_from_torus_storage(chain, args.side, args.dim, chain_geom);

    // --- contour-distance spin correlation ---
    compute_chain_correlations(angles_chain, Cd, dmax);

    // --- spatial spin correlation (radial bins, integer Euclidean d) ---
    if (args.run_spatial_corr) {
      compute_spatial_correlations_radial(chain_geom, angles_chain,
                                          args.side, args.dim, rmax,
                                          mean_d, cnt_d, args.periodic_lattice);
    }

    // --- helicity building blocks and per-sample reduced stiffness observables ---
    HelicityTerms hel_bb, hel_full;
    compute_helicity_terms(chain_geom, angles_chain, args.side, args.dim, hel_bb, hel_full, args.periodic_lattice);
    const double norm_spins = double(args.L);

    // --- write helicity row ---
    out_hel << args.L << "," << J_d << "," << step_d << "," << source << ","
            << hel_bb.n_bonds << ","
            << (hel_full.n_bonds - hel_bb.n_bonds) << ","
            << norm_spins << ","
            << E_d;
    write_helicity_terms(out_hel, hel_bb,   J_d, norm_spins, naxes);
    write_helicity_terms(out_hel, hel_full, J_d, norm_spins, naxes);
    out_hel << "\n";

    // --- write contour correlation row ---
    out_chain << args.L << "," << J_d << "," << step_d << "," << source << "," << E_d;
    for (size_t d = 0; d < Cd.size(); ++d) out_chain << "," << Cd[d];
    out_chain << "\n";

    // --- write spatial correlation row ---
    if (args.run_spatial_corr) {
      out_sp << args.L << "," << J_d << "," << step_d << "," << source << "," << E_d << "," << rmax;
      for (int d = 1; d <= rmax; ++d) out_sp << "," << mean_d[(size_t)d];
      for (int d = 1; d <= rmax; ++d) out_sp << "," << cnt_d[(size_t)d];
      out_sp << "\n";
    }
  }

  if (n_step_mismatch > 0) {
    std::cerr << "WARN: total step mismatches = " << n_step_mismatch << "\n";
  }
  if (n_J_mismatch > 0) {
    std::cerr << "WARN: total J mismatches = " << n_J_mismatch << "\n";
  }
  std::cerr << "Wrote: " << hel_path << "\n";
  std::cerr << "Wrote: " << cor_chain_path << "\n";
  if (args.run_spatial_corr) std::cerr << "Wrote: " << cor_sp_path << "\n";
}

// ====================== main ======================
int main(int argc, char** argv) {
  Args args = parse_args(argc, argv);

  #ifdef _OPENMP
  if (args.threads > 0) omp_set_num_threads(args.threads);
  #else
  (void)args.threads;
  #endif

  try {
    if (!args.periodic_lattice) {
      std::cerr << "INFO: using open embedded-lattice displacements for joint spatial correlations/helicity. Use --periodic-lattice only for torus logs.\n";
    }
    if (args.run_angles) process_angles(args);
    if (args.run_dirs)   process_dirs(args);
    if (args.run_joint)  process_joint(args);
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 2;
  }
  return 0;
}