
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#include <filesystem>

#ifdef _OPENMP
  #include <omp.h>
#endif

// ============================================================================
// process_saw_xy_vortices.cpp
//
// Reads one angles log and one coordinates/dirs log IN LOCKSTEP.
// Produces per-configuration geometry, vortex, pair-separation, and twist stats.
//
// Assumed input formats:
//
// angles line:
//   step J E theta_0 theta_1 ... theta_{L-1}
//
// dirs line:
//   step J start_idx E x_0 y_0 [z_0] x_1 y_1 [z_1] ... x_{L-1} y_{L-1} [z_{L-1}]
//
// The coordinates may be stored on a large technical torus. For physical
// analysis the chain is unwrapped from start_idx, so the output treats the
// configuration as an open embedded object.
//
// For 2D vortex detection, the code uses occupied elementary square plaquettes:
//   (x,y), (x+1,y), (x+1,y+1), (x,y+1)
// and computes the wrapped spin-angle winding around each valid plaquette.
//
// Compile:
//   g++ -O3 -std=c++17 -fopenmp process_saw_xy_vortices.cpp -o process_saw_xy_vortices
//
// Example:
//   ./process_saw_xy_vortices --angles angles.txt --dirs dirs.txt --L 4096 --dim 2 --side 10000
//       --outdir out --angle-order ring --twist-mode allpairs-power --power-alpha 3 --free-rc 4
// ============================================================================

// ------------------------ FastScanner ------------------------
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

      int expv = 0;
      bool eany = false;
      while (true) {
        c = peek();
        if (c >= '0' && c <= '9') {
          eany = true;
          expv = expv * 10 + int(get() - '0');
        } else break;
      }
      if (eany) {
        const double pow10 = std::pow(10.0, double(expv));
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

  inline char peek() {
    if (!refill()) return 0;
    return buf_[i_];
  }

  inline char get() {
    if (!refill()) return 0;
    return buf_[i_++];
  }

  inline void skipSpaces() {
    while (true) {
      char c = peek();
      if (!c) return;
      if (c == ' ' || c == '\n' || c == '\r' || c == '\t' || c == '\v' || c == '\f') get();
      else return;
    }
  }

  template<class T>
  bool readSignedIntegral(T& out) {
    skipSpaces();
    char c = peek();
    if (!c) return false;

    bool neg = false;
    if (c == '-') { neg = true; get(); }
    else if (c == '+') { get(); }

    using U = std::make_unsigned_t<T>;
    U val = 0;
    bool any = false;
    while (true) {
      c = peek();
      if (c >= '0' && c <= '9') {
        any = true;
        val = val * 10 + U(get() - '0');
      } else break;
    }

    if (!any) return false;
    out = neg ? (T(0) - T(val)) : T(val);
    return true;
  }
};

// ------------------------ Basic helpers ------------------------
static constexpr double PI = 3.141592653589793238462643383279502884;

static inline int wrap_int(int a, int side) {
  int r = a % side;
  if (r < 0) r += side;
  return r;
}

static inline int min_image_signed(int a, int b, int side) {
  int d = b - a;
  const int half = side / 2;
  if (d >  half) d -= side;
  if (d < -half) d += side;
  return d;
}

static inline double wrap_angle(double dtheta) {
  return std::atan2(std::sin(dtheta), std::cos(dtheta)); // (-pi, pi]
}

static inline bool nearly_same(double a, double b, double tol = 1e-8) {
  return std::abs(a - b) <= tol * (1.0 + std::max(std::abs(a), std::abs(b)));
}

struct Vec3i {
  int x=0, y=0, z=0;
};

struct Vec3ll {
  long long x=0, y=0, z=0;
};

static inline long double dist2_ll(const Vec3ll& a, const Vec3ll& b, int dim) {
  const long double dx = (long double)b.x - (long double)a.x;
  const long double dy = (long double)b.y - (long double)a.y;
  const long double dz = (dim == 3) ? ((long double)b.z - (long double)a.z) : 0.0L;
  return dx*dx + dy*dy + dz*dz;
}

struct CoordKey {
  long long x=0, y=0, z=0;
  bool operator==(const CoordKey& o) const {
    return x == o.x && y == o.y && z == o.z;
  }
};

struct CoordHash {
  static uint64_t splitmix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
  }

  size_t operator()(const CoordKey& c) const {
    uint64_t h = splitmix64((uint64_t)c.x);
    h ^= splitmix64((uint64_t)c.y + 0x9e3779b97f4a7c15ULL) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    h ^= splitmix64((uint64_t)c.z + 0xbf58476d1ce4e5b9ULL) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    return (size_t)h;
  }
};

static inline CoordKey key_from_vec(const Vec3ll& p, int dim) {
  return {p.x, p.y, (dim == 3 ? p.z : 0)};
}

static inline std::string file_source_tag(const std::string& path) {
  std::filesystem::path p(path);
  auto parent = p.parent_path();
  if (parent.empty()) return "";
  return parent.filename().string();
}

static inline double mean_of(const std::vector<double>& v) {
  if (v.empty()) return -1.0;
  long double s = 0.0L;
  for (double x : v) s += x;
  return double(s / (long double)v.size());
}

static inline double quantile_sorted(const std::vector<double>& sorted, double q) {
  if (sorted.empty()) return -1.0;
  if (q <= 0.0) return sorted.front();
  if (q >= 1.0) return sorted.back();
  const double pos = q * double(sorted.size() - 1);
  const size_t lo = (size_t)std::floor(pos);
  const size_t hi = (size_t)std::ceil(pos);
  const double t = pos - double(lo);
  return sorted[lo] * (1.0 - t) + sorted[hi] * t;
}

struct DistSummary {
  long long count = 0;
  double mean = -1.0;
  double median = -1.0;
  double p90 = -1.0;
  double max = -1.0;
};

static inline DistSummary summarize_distances(std::vector<double> v) {
  DistSummary s;
  s.count = (long long)v.size();
  if (v.empty()) return s;
  std::sort(v.begin(), v.end());
  s.mean = mean_of(v);
  s.median = quantile_sorted(v, 0.5);
  s.p90 = quantile_sorted(v, 0.9);
  s.max = v.back();
  return s;
}

// ------------------------ Arguments ------------------------
struct Args {
  std::string angles_file;
  std::string dirs_file;
  std::string outdir = ".";

  int L = -1;
  int dim = 2;
  int side = -1;
  int chain = -1;
  int threads = -1;

  // "ring": theta_i corresponds to coordinate slot i in the dirs file.
  // "chain": theta_i corresponds to unwrapped chain order from start_idx.
  std::string angle_order = "ring";

  // Vortex/pair parameters.
  double free_rc = 4.0;
  double charge_tol = 0.25;
  bool write_defects = true;
  bool write_pairs = true;

  // Greedy matched-pair construction can be expensive if there are many vortices.
  // If N_plus * N_minus exceeds this, matched stats are skipped.
  long long max_match_candidates = 5000000LL;

  // Twist observables.
  // "none", "allpairs-power", "contacts"
  std::string twist_mode = "allpairs-power";
  double power_alpha = 3.0;
  bool contact_twist_exclude_bonds = false;
};

static void usage_and_exit() {
  std::cerr <<
    "Usage:\n"
    "  process_saw_xy_vortices --angles <angles_file> --dirs <dirs_file> --L <L> --dim <2|3> --side <side>\n"
    "                         [--outdir <dir>] [--chain <k|-1>] [--threads <n>]\n"
    "                         [--angle-order <ring|chain>]\n"
    "                         [--twist-mode <none|allpairs-power|contacts>] [--power-alpha <a>]\n"
    "                         [--free-rc <r>] [--charge-tol <tol>]\n"
    "                         [--max-match-candidates <n>]\n"
    "                         [--no-write-defects] [--no-write-pairs]\n\n"
    "Important:\n"
    "  * The dirs file is unwrapped from start_idx, so the technical torus is not treated as physical periodicity.\n"
    "  * For 2D vortices this code uses only valid occupied elementary plaquettes.\n"
    "  * If your angles were written in the same ring-buffer order as coords, keep --angle-order ring.\n"
    "  * If your angles were written already from start_idx to end, use --angle-order chain.\n";
  std::exit(1);
}

static Args parse_args(int argc, char** argv) {
  Args a;

  for (int i = 1; i < argc; ++i) {
    std::string s = argv[i];
    auto need = [&](const char* name) -> std::string {
      if (i + 1 >= argc) {
        std::cerr << "Missing value for " << name << "\n";
        usage_and_exit();
      }
      return std::string(argv[++i]);
    };

    if      (s == "--angles") a.angles_file = need("--angles");
    else if (s == "--dirs") a.dirs_file = need("--dirs");
    else if (s == "--outdir") a.outdir = need("--outdir");
    else if (s == "--L") a.L = std::stoi(need("--L"));
    else if (s == "--dim") a.dim = std::stoi(need("--dim"));
    else if (s == "--side") a.side = std::stoi(need("--side"));
    else if (s == "--chain") a.chain = std::stoi(need("--chain"));
    else if (s == "--threads") a.threads = std::stoi(need("--threads"));
    else if (s == "--angle-order") a.angle_order = need("--angle-order");
    else if (s == "--free-rc") a.free_rc = std::stod(need("--free-rc"));
    else if (s == "--charge-tol") a.charge_tol = std::stod(need("--charge-tol"));
    else if (s == "--twist-mode") a.twist_mode = need("--twist-mode");
    else if (s == "--power-alpha") a.power_alpha = std::stod(need("--power-alpha"));
    else if (s == "--max-match-candidates") a.max_match_candidates = std::stoll(need("--max-match-candidates"));
    else if (s == "--contact-twist-exclude-bonds") a.contact_twist_exclude_bonds = true;
    else if (s == "--no-write-defects") a.write_defects = false;
    else if (s == "--no-write-pairs") a.write_pairs = false;
    else {
      std::cerr << "Unknown argument: " << s << "\n";
      usage_and_exit();
    }
  }

  if (a.angles_file.empty() || a.dirs_file.empty() || a.L <= 0 || a.side <= 0 ||
      (a.dim != 2 && a.dim != 3)) {
    usage_and_exit();
  }

  if (a.angle_order != "ring" && a.angle_order != "chain") {
    std::cerr << "--angle-order must be ring or chain\n";
    usage_and_exit();
  }

  if (a.twist_mode != "none" && a.twist_mode != "allpairs-power" && a.twist_mode != "contacts") {
    std::cerr << "--twist-mode must be none, allpairs-power, or contacts\n";
    usage_and_exit();
  }

  return a;
}

// ------------------------ Input record readers ------------------------
struct AngleRecord {
  long long step = 0;
  double J = 0.0;
  double E = 0.0;
  int chain_id = 0;
  std::vector<double> theta_raw;
};

struct DirRecord {
  long long step = 0;
  double J = 0.0;
  int start_idx = 0;
  double E = 0.0;
  int chain_id = 0;
  std::vector<Vec3i> ring;
};

class AngleReader {
public:
  AngleReader(const std::string& path, int L)
    : fs_(path), L_(L), prev_step_(std::numeric_limits<long long>::min()), line_in_block_(0) {}

  bool read(AngleRecord& r) {
    if (!fs_.readLongLong(r.step)) return false;
    if (!fs_.readDouble(r.J)) throw std::runtime_error("Angles file ended while reading J");
    if (!fs_.readDouble(r.E)) throw std::runtime_error("Angles file ended while reading E");

    if (r.step != prev_step_) {
      prev_step_ = r.step;
      line_in_block_ = 0;
    }
    r.chain_id = line_in_block_++;

    r.theta_raw.assign((size_t)L_, 0.0);
    for (int i = 0; i < L_; ++i) {
      if (!fs_.readDouble(r.theta_raw[(size_t)i])) {
        throw std::runtime_error("Angles file ended while reading theta values");
      }
    }
    return true;
  }

private:
  FastScanner fs_;
  int L_;
  long long prev_step_;
  int line_in_block_;
};

class DirReader {
public:
  DirReader(const std::string& path, int L, int dim)
    : fs_(path), L_(L), dim_(dim), prev_step_(std::numeric_limits<long long>::min()), line_in_block_(0) {}

  bool read(DirRecord& r) {
    if (!fs_.readLongLong(r.step)) return false;
    if (!fs_.readDouble(r.J)) throw std::runtime_error("Dirs file ended while reading J");
    if (!fs_.readInt(r.start_idx)) throw std::runtime_error("Dirs file ended while reading start_idx");
    if (!fs_.readDouble(r.E)) throw std::runtime_error("Dirs file ended while reading E");

    if (r.step != prev_step_) {
      prev_step_ = r.step;
      line_in_block_ = 0;
    }
    r.chain_id = line_in_block_++;

    r.ring.assign((size_t)L_, Vec3i{});
    for (int i = 0; i < L_; ++i) {
      int x=0, y=0, z=0;
      if (!fs_.readInt(x)) throw std::runtime_error("Dirs file ended while reading x");
      if (!fs_.readInt(y)) throw std::runtime_error("Dirs file ended while reading y");
      if (dim_ == 3) {
        if (!fs_.readInt(z)) throw std::runtime_error("Dirs file ended while reading z");
      }
      r.ring[(size_t)i] = {x, y, z};
    }
    return true;
  }

private:
  FastScanner fs_;
  int L_;
  int dim_;
  long long prev_step_;
  int line_in_block_;
};

// ------------------------ Configuration construction ------------------------
struct Config {
  std::vector<Vec3ll> chain_pos;   // unwrapped, start_idx -> end
  std::vector<double> theta_chain; // same order as chain_pos

  std::unordered_map<CoordKey, int, CoordHash> occ; // coordinate -> chain index

  long long duplicate_sites = 0;

  long long min_x=0, max_x=0, min_y=0, max_y=0, min_z=0, max_z=0;
  long long bbox_area_2d = 0;
};

static Config build_config(const Args& args, const AngleRecord& ar, const DirRecord& dr) {
  const int L = args.L;
  const int s = ((dr.start_idx % L) + L) % L;

  std::vector<Vec3i> ring = dr.ring;

  Config cfg;
  cfg.chain_pos.assign((size_t)L, Vec3ll{});
  cfg.theta_chain.assign((size_t)L, 0.0);

  // Unwrap coordinates in physical chain order.
  cfg.chain_pos[0] = {
    ring[(size_t)s].x,
    ring[(size_t)s].y,
    (args.dim == 3 ? ring[(size_t)s].z : 0)
  };

  for (int t = 1; t < L; ++t) {
    const int prev_ring = (s + t - 1) % L;
    const int curr_ring = (s + t) % L;

    const int dx = min_image_signed(ring[(size_t)prev_ring].x, ring[(size_t)curr_ring].x, args.side);
    const int dy = min_image_signed(ring[(size_t)prev_ring].y, ring[(size_t)curr_ring].y, args.side);
    const int dz = (args.dim == 3)
      ? min_image_signed(ring[(size_t)prev_ring].z, ring[(size_t)curr_ring].z, args.side)
      : 0;

    cfg.chain_pos[(size_t)t].x = cfg.chain_pos[(size_t)t-1].x + dx;
    cfg.chain_pos[(size_t)t].y = cfg.chain_pos[(size_t)t-1].y + dy;
    cfg.chain_pos[(size_t)t].z = cfg.chain_pos[(size_t)t-1].z + dz;
  }

  // Align spin angles with the unwrapped chain order.
  if (args.angle_order == "ring") {
    for (int t = 0; t < L; ++t) {
      const int ring_idx = (s + t) % L;
      cfg.theta_chain[(size_t)t] = ar.theta_raw[(size_t)ring_idx];
    }
  } else {
    for (int t = 0; t < L; ++t) {
      cfg.theta_chain[(size_t)t] = ar.theta_raw[(size_t)t];
    }
  }

  cfg.occ.reserve((size_t)L * 2);

  cfg.min_x = cfg.max_x = cfg.chain_pos[0].x;
  cfg.min_y = cfg.max_y = cfg.chain_pos[0].y;
  cfg.min_z = cfg.max_z = cfg.chain_pos[0].z;

  for (int t = 0; t < L; ++t) {
    const Vec3ll& p = cfg.chain_pos[(size_t)t];

    cfg.min_x = std::min(cfg.min_x, p.x); cfg.max_x = std::max(cfg.max_x, p.x);
    cfg.min_y = std::min(cfg.min_y, p.y); cfg.max_y = std::max(cfg.max_y, p.y);
    if (args.dim == 3) {
      cfg.min_z = std::min(cfg.min_z, p.z); cfg.max_z = std::max(cfg.max_z, p.z);
    }

    CoordKey k = key_from_vec(p, args.dim);
    auto result = cfg.occ.emplace(k, t);
    if (!result.second) {
      // A true SAW should not do this after unwrapping.
      cfg.duplicate_sites++;
      result.first->second = t; // keep the last one to avoid crashing later
    }
  }

  cfg.bbox_area_2d = (cfg.max_x - cfg.min_x + 1) * (cfg.max_y - cfg.min_y + 1);
  return cfg;
}

// ------------------------ Geometry metrics ------------------------
struct GeometryStats {
  double mag = 0.0;
  double mx = 0.0;
  double my = 0.0;

  double R2 = 0.0;
  double mean_Rij = 0.0;
  double factors_1_over_r3_divN = 0.0;
  double Rg2 = 0.0;
  double Rg = 0.0;
  long long contacts = 0;
};

static GeometryStats compute_geometry_stats(const Args& args, const Config& cfg) {
  const int L = args.L;
  GeometryStats g;

  long double sx = 0.0L, sy = 0.0L;
  for (double th : cfg.theta_chain) {
    sx += std::cos(th);
    sy += std::sin(th);
  }
  g.mx = double(sx / (long double)L);
  g.my = double(sy / (long double)L);
  g.mag = std::sqrt(g.mx*g.mx + g.my*g.my);

  g.R2 = double(dist2_ll(cfg.chain_pos.front(), cfg.chain_pos.back(), args.dim));

  long double sum_r2 = 0.0L;
  long double sum_inv_r3 = 0.0L;
  long long count = 0;

  #ifdef _OPENMP
  #pragma omp parallel for reduction(+:sum_r2,sum_inv_r3,count) schedule(static)
  #endif
  for (int i = 0; i < L - 1; ++i) {
    for (int j = i + 1; j < L; ++j) {
      const long double r2 = dist2_ll(cfg.chain_pos[(size_t)i], cfg.chain_pos[(size_t)j], args.dim);
      if (r2 > 0.0L) {
        sum_r2 += r2;
        sum_inv_r3 += 1.0L / (std::sqrt(r2) * r2);
        count += 1;
      }
    }
  }

  g.mean_Rij = (count > 0) ? double(sum_r2 / (long double)count) : 0.0;
  g.factors_1_over_r3_divN = (L > 0) ? double(sum_inv_r3 / (long double)L) : 0.0;

  long double cx = 0.0L, cy = 0.0L, cz = 0.0L;
  for (const auto& p : cfg.chain_pos) {
    cx += p.x;
    cy += p.y;
    cz += p.z;
  }
  cx /= (long double)L;
  cy /= (long double)L;
  cz /= (long double)L;

  long double rg_sum = 0.0L;
  for (const auto& p : cfg.chain_pos) {
    const long double dx = (long double)p.x - cx;
    const long double dy = (long double)p.y - cy;
    const long double dz = (args.dim == 3) ? ((long double)p.z - cz) : 0.0L;
    rg_sum += dx*dx + dy*dy + dz*dz;
  }

  g.Rg2 = double(rg_sum / (long double)L);
  g.Rg = std::sqrt(g.Rg2);

  // Count nearest-neighbor occupied lattice contacts, excluding chain bonds.
  long long contacts = 0;
  const int dirs2[4][3] = {{+1,0,0},{0,+1,0},{-1,0,0},{0,-1,0}};
  const int dirs3[6][3] = {{+1,0,0},{0,+1,0},{0,0,+1},{-1,0,0},{0,-1,0},{0,0,-1}};

  for (int i = 0; i < L; ++i) {
    const Vec3ll& p = cfg.chain_pos[(size_t)i];
    const int ndirs = (args.dim == 2) ? 4 : 6;

    for (int k = 0; k < ndirs; ++k) {
      const int* d = (args.dim == 2) ? dirs2[k] : dirs3[k];
      CoordKey nk{p.x + d[0], p.y + d[1], p.z + (args.dim == 3 ? d[2] : 0)};
      auto it = cfg.occ.find(nk);
      if (it == cfg.occ.end()) continue;

      const int j = it->second;
      if (j > i + 1) contacts++;
    }
  }
  g.contacts = contacts;

  return g;
}

// ------------------------ Vortex detection and pair stats ------------------------
struct Defect {
  int q = 0;
  double x = 0.0;
  double y = 0.0;
};

struct PairRow {
  std::string kind; // nearest or matched
  int id1 = -1;
  int id2 = -1;
  int q1 = 0;
  int q2 = 0;
  double x1 = 0.0;
  double y1 = 0.0;
  double x2 = 0.0;
  double y2 = 0.0;
  double d = -1.0;
};

struct VortexStats {
  long long valid_plaquettes = 0;
  long long ambiguous_plaquettes = 0;
  long long high_charge_plaquettes = 0;

  long long vortex_plus = 0;
  long long vortex_minus = 0;
  long long vortex_abs = 0;
  long long vortex_net = 0;

  double vortex_density = -1.0;

  double free_fraction = -1.0;
  long long free_count = 0;
  long long defect_count = 0;

  DistSummary nearest_summary;
  DistSummary matched_summary;
  bool matched_skipped = false;

  std::vector<Defect> defects;
  std::vector<PairRow> pair_rows;
};

static inline double defect_dist(const Defect& a, const Defect& b) {
  const double dx = b.x - a.x;
  const double dy = b.y - a.y;
  return std::sqrt(dx*dx + dy*dy);
}

struct PairCandidate {
  int ip = -1;
  int im = -1;
  double d = 0.0;
};

static VortexStats compute_vortex_stats(const Args& args, const Config& cfg) {
  VortexStats vs;

  if (args.dim != 2) {
    return vs;
  }

  // Find vortices on occupied elementary plaquettes.
  for (const auto& kv : cfg.occ) {
    const CoordKey bl = kv.first; // bottom-left
    const CoordKey br{bl.x + 1, bl.y, 0};
    const CoordKey tr{bl.x + 1, bl.y + 1, 0};
    const CoordKey tl{bl.x, bl.y + 1, 0};

    auto it_bl = cfg.occ.find(bl);
    auto it_br = cfg.occ.find(br);
    auto it_tr = cfg.occ.find(tr);
    auto it_tl = cfg.occ.find(tl);

    if (it_bl == cfg.occ.end() || it_br == cfg.occ.end() ||
        it_tr == cfg.occ.end() || it_tl == cfg.occ.end()) {
      continue;
    }

    vs.valid_plaquettes++;

    const double th_bl = cfg.theta_chain[(size_t)it_bl->second];
    const double th_br = cfg.theta_chain[(size_t)it_br->second];
    const double th_tr = cfg.theta_chain[(size_t)it_tr->second];
    const double th_tl = cfg.theta_chain[(size_t)it_tl->second];

    const double sum =
      wrap_angle(th_br - th_bl) +
      wrap_angle(th_tr - th_br) +
      wrap_angle(th_tl - th_tr) +
      wrap_angle(th_bl - th_tl);

    const double q_real = sum / (2.0 * PI);
    const int q = (int)std::llround(q_real);

    if (std::abs(q_real - double(q)) > args.charge_tol) {
      vs.ambiguous_plaquettes++;
      continue;
    }

    if (q == 0) continue;

    if (std::abs(q) > 1) {
      vs.high_charge_plaquettes++;
      // Keep the charge in net counts, but for pair stats it is safer to split
      // only unit vortices. Multiple charge on one elementary square is unusual.
    }

    vs.vortex_net += q;
    vs.vortex_abs += std::llabs((long long)q);

    if (q == +1) vs.vortex_plus++;
    if (q == -1) vs.vortex_minus++;

    if (q == +1 || q == -1) {
      vs.defects.push_back(Defect{q, double(bl.x) + 0.5, double(bl.y) + 0.5});
    }
  }

  if (vs.valid_plaquettes > 0) {
    vs.vortex_density = double(vs.vortex_abs) / double(vs.valid_plaquettes);
  }

  vs.defect_count = (long long)vs.defects.size();

  // Indices of plus/minus defects in vs.defects.
  std::vector<int> plus, minus;
  plus.reserve(vs.defects.size());
  minus.reserve(vs.defects.size());

  for (int i = 0; i < (int)vs.defects.size(); ++i) {
    if (vs.defects[(size_t)i].q == +1) plus.push_back(i);
    else if (vs.defects[(size_t)i].q == -1) minus.push_back(i);
  }

  // Nearest opposite-charge distance for every defect.
  std::vector<double> nearest_distances;
  nearest_distances.reserve(vs.defects.size());

  long long free_count = 0;

  for (int id = 0; id < (int)vs.defects.size(); ++id) {
    const Defect& a = vs.defects[(size_t)id];
    const std::vector<int>& candidates = (a.q == +1) ? minus : plus;

    if (candidates.empty()) {
      free_count++;
      continue;
    }

    double best = std::numeric_limits<double>::infinity();
    int best_id = -1;

    for (int oid : candidates) {
      const double d = defect_dist(a, vs.defects[(size_t)oid]);
      if (d < best) {
        best = d;
        best_id = oid;
      }
    }

    if (std::isfinite(best)) {
      nearest_distances.push_back(best);
      if (best > args.free_rc) free_count++;

      if (args.write_pairs) {
        const Defect& b = vs.defects[(size_t)best_id];
        vs.pair_rows.push_back(PairRow{
          "nearest", id, best_id, a.q, b.q, a.x, a.y, b.x, b.y, best
        });
      }
    }
  }

  vs.free_count = free_count;
  if (!vs.defects.empty()) {
    vs.free_fraction = double(free_count) / double(vs.defects.size());
  }

  vs.nearest_summary = summarize_distances(nearest_distances);

  // Greedy unique matching of + and - defects by shortest distances.
  const long long candidate_count = (long long)plus.size() * (long long)minus.size();

  if (candidate_count > args.max_match_candidates) {
    vs.matched_skipped = true;
  } else {
    std::vector<PairCandidate> candidates;
    candidates.reserve((size_t)candidate_count);

    for (int ip = 0; ip < (int)plus.size(); ++ip) {
      for (int im = 0; im < (int)minus.size(); ++im) {
        const double d = defect_dist(vs.defects[(size_t)plus[(size_t)ip]],
                                     vs.defects[(size_t)minus[(size_t)im]]);
        candidates.push_back(PairCandidate{ip, im, d});
      }
    }

    std::sort(candidates.begin(), candidates.end(),
              [](const PairCandidate& a, const PairCandidate& b) { return a.d < b.d; });

    std::vector<char> used_plus(plus.size(), 0);
    std::vector<char> used_minus(minus.size(), 0);
    std::vector<double> matched_distances;

    for (const auto& c : candidates) {
      if (used_plus[(size_t)c.ip] || used_minus[(size_t)c.im]) continue;

      used_plus[(size_t)c.ip] = 1;
      used_minus[(size_t)c.im] = 1;

      const int idp = plus[(size_t)c.ip];
      const int idm = minus[(size_t)c.im];
      matched_distances.push_back(c.d);

      if (args.write_pairs) {
        const Defect& p = vs.defects[(size_t)idp];
        const Defect& m = vs.defects[(size_t)idm];
        vs.pair_rows.push_back(PairRow{
          "matched", idp, idm, p.q, m.q, p.x, p.y, m.x, m.y, c.d
        });
      }
    }

    vs.matched_summary = summarize_distances(matched_distances);
  }

  return vs;
}

// ------------------------ Twist observables ------------------------
struct TwistStats {
  long long pairs = 0;
  double Cx = 0.0;
  double Cy = 0.0;
  double Sx = 0.0;
  double Sy = 0.0;
  double Sx2 = 0.0;
  double Sy2 = 0.0;
};

static TwistStats compute_twist_stats(const Args& args, const Config& cfg) {
  TwistStats ts;
  if (args.twist_mode == "none") return ts;
  if (args.dim != 2) return ts; // this implementation stores x/y stiffness only

  const int L = args.L;

  if (args.twist_mode == "allpairs-power") {
    long double Cx = 0.0L, Cy = 0.0L, Sx = 0.0L, Sy = 0.0L;
    long long pairs = 0;

    #ifdef _OPENMP
    #pragma omp parallel for reduction(+:Cx,Cy,Sx,Sy,pairs) schedule(static)
    #endif
    for (int i = 0; i < L - 1; ++i) {
      for (int j = i + 1; j < L; ++j) {
        const long double dx = (long double)cfg.chain_pos[(size_t)j].x - (long double)cfg.chain_pos[(size_t)i].x;
        const long double dy = (long double)cfg.chain_pos[(size_t)j].y - (long double)cfg.chain_pos[(size_t)i].y;
        const long double r2 = dx*dx + dy*dy;
        if (r2 <= 0.0L) continue;

        const long double r = std::sqrt(r2);
        const long double w = 1.0L / std::pow(r, (long double)args.power_alpha);

        const double dtheta = cfg.theta_chain[(size_t)j] - cfg.theta_chain[(size_t)i];
        const long double co = std::cos(dtheta);
        const long double si = std::sin(dtheta);

        Cx += w * co * dx * dx;
        Cy += w * co * dy * dy;
        Sx += w * si * dx;
        Sy += w * si * dy;
        pairs++;
      }
    }

    ts.pairs = pairs;
    ts.Cx = double(Cx);
    ts.Cy = double(Cy);
    ts.Sx = double(Sx);
    ts.Sy = double(Sy);
    ts.Sx2 = ts.Sx * ts.Sx;
    ts.Sy2 = ts.Sy * ts.Sy;
    return ts;
  }

  if (args.twist_mode == "contacts") {
    long double Cx = 0.0L, Cy = 0.0L, Sx = 0.0L, Sy = 0.0L;
    long long pairs = 0;

    const int dirs[4][2] = {{+1,0},{0,+1},{-1,0},{0,-1}};

    for (int i = 0; i < L; ++i) {
      const Vec3ll& p = cfg.chain_pos[(size_t)i];

      for (const auto& d : dirs) {
        CoordKey nk{p.x + d[0], p.y + d[1], 0};
        auto it = cfg.occ.find(nk);
        if (it == cfg.occ.end()) continue;

        const int j = it->second;
        if (j <= i) continue; // count undirected pair once
        if (args.contact_twist_exclude_bonds && std::abs(j - i) == 1) continue;

        const long double dx = (long double)cfg.chain_pos[(size_t)j].x - (long double)p.x;
        const long double dy = (long double)cfg.chain_pos[(size_t)j].y - (long double)p.y;

        const double dtheta = cfg.theta_chain[(size_t)j] - cfg.theta_chain[(size_t)i];
        const long double co = std::cos(dtheta);
        const long double si = std::sin(dtheta);

        Cx += co * dx * dx;
        Cy += co * dy * dy;
        Sx += si * dx;
        Sy += si * dy;
        pairs++;
      }
    }

    ts.pairs = pairs;
    ts.Cx = double(Cx);
    ts.Cy = double(Cy);
    ts.Sx = double(Sx);
    ts.Sy = double(Sy);
    ts.Sx2 = ts.Sx * ts.Sx;
    ts.Sy2 = ts.Sy * ts.Sy;
    return ts;
  }

  return ts;
}

// ------------------------ Main processing ------------------------
static void write_summary_header(std::ofstream& out) {
  out
    << "L,dim,J,E_angles,E_dirs,step,chain_id,start_idx,"
    << "mag,Mx,My,"
    << "R2,mean_Rij,factors_1_over_r3_divN,Rg2,Rg,contacts,"
    << "duplicate_sites,bbox_min_x,bbox_max_x,bbox_min_y,bbox_max_y,bbox_area_2d,"
    << "valid_plaquettes,ambiguous_plaquettes,high_charge_plaquettes,"
    << "vortex_plus,vortex_minus,vortex_abs,vortex_net,vortex_density,"
    << "free_rc,free_count,free_fraction,"
    << "nearest_count,nearest_mean,nearest_median,nearest_p90,nearest_max,"
    << "matched_count,matched_mean,matched_median,matched_p90,matched_max,matched_skipped,"
    << "twist_mode,twist_pairs,Cx,Cy,Sx,Sy,Sx2,Sy2,"
    << "angle_order,source_angles,source_dirs\n";
}

static void write_defects_header(std::ofstream& out) {
  out << "L,J,step,chain_id,defect_id,q,x,y\n";
}

static void write_pairs_header(std::ofstream& out) {
  out << "L,J,step,chain_id,kind,id1,id2,q1,q2,x1,y1,x2,y2,distance\n";
}

static void process_all(const Args& args) {
  std::filesystem::create_directories(args.outdir);

  const std::string summary_path =
    (std::filesystem::path(args.outdir) / ("Data_ConfigStats_" + std::to_string(args.L) + ".csv")).string();

  const std::string defects_path =
    (std::filesystem::path(args.outdir) / ("Data_VortexDefects_" + std::to_string(args.L) + ".csv")).string();

  const std::string pairs_path =
    (std::filesystem::path(args.outdir) / ("Data_VortexPairs_" + std::to_string(args.L) + ".csv")).string();

  std::ofstream summary(summary_path);
  if (!summary) throw std::runtime_error("Cannot open output: " + summary_path);
  summary << std::setprecision(17);
  write_summary_header(summary);

  std::ofstream defects;
  if (args.write_defects) {
    defects.open(defects_path);
    if (!defects) throw std::runtime_error("Cannot open output: " + defects_path);
    defects << std::setprecision(17);
    write_defects_header(defects);
  }

  std::ofstream pairs;
  if (args.write_pairs) {
    pairs.open(pairs_path);
    if (!pairs) throw std::runtime_error("Cannot open output: " + pairs_path);
    pairs << std::setprecision(17);
    write_pairs_header(pairs);
  }

  AngleReader ar(args.angles_file, args.L);
  DirReader dr(args.dirs_file, args.L, args.dim);

  const std::string source_angles = file_source_tag(args.angles_file);
  const std::string source_dirs = file_source_tag(args.dirs_file);

  AngleRecord angle_rec;
  DirRecord dir_rec;

  long long processed = 0;
  long long read_records = 0;

  while (true) {
    const bool ok_a = ar.read(angle_rec);
    const bool ok_d = dr.read(dir_rec);

    if (!ok_a && !ok_d) break;
    if (ok_a != ok_d) {
      throw std::runtime_error("Angles and dirs files have different number of records");
    }

    read_records++;

    if (angle_rec.step != dir_rec.step || angle_rec.chain_id != dir_rec.chain_id ||
        !nearly_same(angle_rec.J, dir_rec.J, 1e-7)) {
      std::cerr << "Mismatch at record " << read_records << "\n"
                << "  angles: step=" << angle_rec.step << " J=" << angle_rec.J
                << " chain_id=" << angle_rec.chain_id << "\n"
                << "  dirs:   step=" << dir_rec.step << " J=" << dir_rec.J
                << " chain_id=" << dir_rec.chain_id << "\n";
      throw std::runtime_error("Angles/dirs streams are not synchronized");
    }

    if (args.chain >= 0 && angle_rec.chain_id != args.chain) continue;

    Config cfg = build_config(args, angle_rec, dir_rec);

    GeometryStats gs = compute_geometry_stats(args, cfg);
    VortexStats vs = compute_vortex_stats(args, cfg);
    TwistStats ts = compute_twist_stats(args, cfg);

    summary
      << args.L << "," << args.dim << ","
      << angle_rec.J << "," << angle_rec.E << "," << dir_rec.E << ","
      << angle_rec.step << "," << angle_rec.chain_id << "," << dir_rec.start_idx << ","
      << gs.mag << "," << gs.mx << "," << gs.my << ","
      << gs.R2 << "," << gs.mean_Rij << "," << gs.factors_1_over_r3_divN << ","
      << gs.Rg2 << "," << gs.Rg << "," << gs.contacts << ","
      << cfg.duplicate_sites << ","
      << cfg.min_x << "," << cfg.max_x << "," << cfg.min_y << "," << cfg.max_y << "," << cfg.bbox_area_2d << ","
      << vs.valid_plaquettes << "," << vs.ambiguous_plaquettes << "," << vs.high_charge_plaquettes << ","
      << vs.vortex_plus << "," << vs.vortex_minus << "," << vs.vortex_abs << "," << vs.vortex_net << "," << vs.vortex_density << ","
      << args.free_rc << "," << vs.free_count << "," << vs.free_fraction << ","
      << vs.nearest_summary.count << "," << vs.nearest_summary.mean << "," << vs.nearest_summary.median << ","
      << vs.nearest_summary.p90 << "," << vs.nearest_summary.max << ","
      << vs.matched_summary.count << "," << vs.matched_summary.mean << "," << vs.matched_summary.median << ","
      << vs.matched_summary.p90 << "," << vs.matched_summary.max << "," << (vs.matched_skipped ? 1 : 0) << ","
      << args.twist_mode << "," << ts.pairs << ","
      << ts.Cx << "," << ts.Cy << "," << ts.Sx << "," << ts.Sy << "," << ts.Sx2 << "," << ts.Sy2 << ","
      << args.angle_order << "," << source_angles << "," << source_dirs << "\n";

    if (args.write_defects) {
      for (int i = 0; i < (int)vs.defects.size(); ++i) {
        const Defect& d = vs.defects[(size_t)i];
        defects << args.L << "," << angle_rec.J << "," << angle_rec.step << ","
                << angle_rec.chain_id << "," << i << ","
                << d.q << "," << d.x << "," << d.y << "\n";
      }
    }

    if (args.write_pairs) {
      for (const PairRow& pr : vs.pair_rows) {
        pairs << args.L << "," << angle_rec.J << "," << angle_rec.step << ","
              << angle_rec.chain_id << "," << pr.kind << ","
              << pr.id1 << "," << pr.id2 << ","
              << pr.q1 << "," << pr.q2 << ","
              << pr.x1 << "," << pr.y1 << ","
              << pr.x2 << "," << pr.y2 << ","
              << pr.d << "\n";
      }
    }

    processed++;

    if (processed % 100 == 0) {
      std::cerr << "Processed " << processed << " configurations\r" << std::flush;
    }
  }

  std::cerr << "\nWrote: " << summary_path << "\n";
  if (args.write_defects) std::cerr << "Wrote: " << defects_path << "\n";
  if (args.write_pairs) std::cerr << "Wrote: " << pairs_path << "\n";
  std::cerr << "Processed configurations: " << processed << "\n";
}

int main(int argc, char** argv) {
  Args args = parse_args(argc, argv);

  #ifdef _OPENMP
  if (args.threads > 0) omp_set_num_threads(args.threads);
  #else
  (void)args;
  #endif

  try {
    process_all(args);
  } catch (const std::exception& e) {
    std::cerr << "\nError: " << e.what() << "\n";
    return 2;
  }

  return 0;
}
