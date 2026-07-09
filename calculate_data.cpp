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

#ifdef _OPENMP
  #include <omp.h>
#endif

// ------------------------ FastScanner (buffered numeric parser) ------------------------
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

    // integer part
    double val = 0.0;
    bool any = false;
    while (true) {
      c = peek();
      if (c >= '0' && c <= '9') {
        any = true;
        val = val * 10.0 + double(get() - '0');
      } else break;
    }

    // fractional
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

    // exponent
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
      if (c == ' ' || c == '\n' || c == '\r' || c == '\t' || c == '\v' || c == '\f') {
        get();
      } else {
        return;
      }
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

    if (neg) out = T(0) - T(val);
    else     out = T(val);
    return true;
  }
};

// ------------------------ Torus helpers ------------------------
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

struct Vec3i { int x=0,y=0,z=0; };
struct Vec3ll { long long x=0,y=0,z=0; };

static inline long long site_key_2d(int x, int y, int side) {
  return (long long)x + (long long)side * (long long)y;
}
static inline long long site_key_3d(int x, int y, int z, int side) {
  return (long long)x + (long long)side * ((long long)y + (long long)side * (long long)z);
}

static inline int wrap(int a, int side) {
  int r = a % side;
  if (r < 0) r += side;
  return r;
}

// ------------------------ Metrics ------------------------
struct DirMetrics {
  long long step = 0;
  double J = 0.0;
  int start_idx = 0;
  double E = 0.0;

  double R2 = 0.0;
  double mean_Rij = 0.0;   // mean r^2 over pairs
  double factors = 0.0;    // (sum 1/r^3) / N
  double Rg2 = 0.0;
  double Rg  = 0.0;
  long long contacts = 0;
};

static inline double torus_r2(const Vec3i& a, const Vec3i& b, int side, int dim) {
  int dx = min_image_abs(a.x, b.x, side);
  int dy = min_image_abs(a.y, b.y, side);
  int dz = (dim == 3) ? min_image_abs(a.z, b.z, side) : 0;
  return double(dx*dx + dy*dy + dz*dz);
}

// Compute pairwise mean(r^2) and sum(1/r^3)/N exactly (O(L^2)).
static inline void pairwise_stats_exact(const std::vector<Vec3i>& chain, int side, int dim,
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
      double r2 = torus_r2(chain[i], chain[j], side, dim);
      // r2 should never be 0 for a SAW, but guard anyway.
      if (r2 > 0.0) {
        sum_r2 += r2;
        sum_inv_r3 += 1.0 / (std::sqrt(r2) * r2); // 1 / r^3 where r = sqrt(r2)
        count += 1;
      }
    }
  }

  mean_r2 = (count > 0) ? (sum_r2 / double(count)) : 0.0;
  factors_divN = (L > 0) ? (sum_inv_r3 / double(L)) : 0.0; // matches your Python
}

// Unwrap chain (start->end) to compute Rg^2 on a torus.
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
    u[i].x = u[i-1].x + dx;
    u[i].y = u[i-1].y + dy;
    u[i].z = u[i-1].z + dz;
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

// Count non-bonded nearest-neighbor contacts on a torus, no directions needed.
// A "contact" here = lattice adjacency (r^2 = 1) between monomers i and j with |i-j|>1.
static inline long long topological_contacts(const std::vector<Vec3i>& chain, int side, int dim) {
  const int L = (int)chain.size();
  if (L < 3) return 0;

  std::unordered_map<long long, int> occ;
  occ.reserve((size_t)L * 2);

  auto key = [&](const Vec3i& p) -> long long {
    if (dim == 2) return site_key_2d(p.x, p.y, side);
    return site_key_3d(p.x, p.y, p.z, side);
  };

  for (int i = 0; i < L; ++i) {
    occ[key(chain[i])] = i;
  }

  const int dirs2d[4][3] = { {+1,0,0},{-1,0,0},{0,+1,0},{0,-1,0} };
  const int dirs3d[6][3] = { {+1,0,0},{-1,0,0},{0,+1,0},{0,-1,0},{0,0,+1},{0,0,-1} };

  long long contacts = 0;

  for (int i = 0; i < L; ++i) {
    const Vec3i& p = chain[i];

    const int nnb = (dim == 2) ? 4 : 6;
    for (int k = 0; k < nnb; ++k) {
      int dx = (dim == 2) ? dirs2d[k][0] : dirs3d[k][0];
      int dy = (dim == 2) ? dirs2d[k][1] : dirs3d[k][1];
      int dz = (dim == 2) ? 0          : dirs3d[k][2];

      Vec3i q;
      q.x = wrap(p.x + dx, side);
      q.y = wrap(p.y + dy, side);
      q.z = (dim == 3) ? wrap(p.z + dz, side) : 0;

      auto it = occ.find(key(q));
      if (it == occ.end()) continue;
      int j = it->second;

      // Count each contact once, and exclude bonds (i,i+1)
      if (j > i + 1) contacts += 1;
    }
  }

  return contacts;
}

// ------------------------ Argument parsing ------------------------
struct Args {
  std::string angles_file;
  std::string dirs_file;
  std::string outdir = ".";
  int L = -1;
  int dim = 3;
  int side = -1;
  int chain = -1;   // -1 = all
  int threads = -1; // OpenMP threads
};

static inline void usage_and_exit() {
  std::cerr <<
    "Usage:\n"
    "  process_saw_logs --angles <angles_file> --dirs <dirs_file> --L <L> --dim <2|3> --side <side>\n"
    "                   [--outdir <dir>] [--chain <k|-1>] [--threads <n>]\n\n"
    "Notes:\n"
    "  * --chain filters the k-th line inside each step block (0-based). Use -1 for all lines.\n"
    "  * --side should be the torus side length used to write coords.\n";
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

    if (s == "--angles") a.angles_file = need("--angles");
    else if (s == "--dirs") a.dirs_file = need("--dirs");
    else if (s == "--outdir") a.outdir = need("--outdir");
    else if (s == "--L") a.L = std::stoi(need("--L"));
    else if (s == "--dim") a.dim = std::stoi(need("--dim"));
    else if (s == "--side") a.side = std::stoi(need("--side"));
    else if (s == "--chain") a.chain = std::stoi(need("--chain"));
    else if (s == "--threads") a.threads = std::stoi(need("--threads"));
    else { std::cerr << "Unknown arg: " << s << "\n"; usage_and_exit(); }
  }

  if (a.angles_file.empty() || a.dirs_file.empty() || a.L <= 0 || (a.dim != 2 && a.dim != 3) || a.side <= 0) {
    usage_and_exit();
  }

  return a;
}

static inline std::string file_source_tag(const std::string& path) {
  // mimic Python "file.parent.stem"
  std::filesystem::path p(path);
  auto parent = p.parent_path();
  if (parent.empty()) return "";
  return parent.filename().string();
}

// ------------------------ Processing: angles ------------------------
static void process_angles(const Args& args) {
  std::filesystem::create_directories(args.outdir);

  const std::string out_path = (std::filesystem::path(args.outdir) / ("Data_TimeSeries_" + std::to_string(args.L) + ".csv")).string();
  std::ofstream out(out_path);
  if (!out) throw std::runtime_error("Cannot open output: " + out_path);

  out << "L,J,mag,E,step,source\n";

  FastScanner fs(args.angles_file);

  const std::string source = file_source_tag(args.angles_file);

  long long step = 0;
  double J = 0.0, E = 0.0;

  long long prev_step = std::numeric_limits<long long>::min();
  int line_in_block = 0;

  while (true) {
    if (!fs.readLongLong(step)) break;
    if (!fs.readDouble(J)) break;
    if (!fs.readDouble(E)) break;

    if (step != prev_step) { prev_step = step; line_in_block = 0; }
    int chain_id = line_in_block++;

    double sin_sum = 0.0;
    double cos_sum = 0.0;

    // read L angles and compute mag on the fly
    for (int i = 0; i < args.L; ++i) {
      double th = 0.0;
      if (!fs.readDouble(th)) th = 0.0;
      // if you ever store NO_XY_SPIN on chain nodes, you probably want to skip it
      // but your logs should only contain valid chain spins.
      sin_sum += std::sin(th);
      cos_sum += std::cos(th);
    }

    if (args.chain >= 0 && chain_id != args.chain) continue;

    const double mag2 = (sin_sum / args.L) * (sin_sum / args.L) + (cos_sum / args.L) * (cos_sum / args.L);
    const double mag = std::sqrt(mag2);

    out << args.L << "," << J << "," << mag << "," << E << "," << step << "," << source << "\n";
  }

  std::cerr << "Wrote: " << out_path << "\n";
}

// ------------------------ Processing: dirs ------------------------
static void process_dirs(const Args& args) {
  std::filesystem::create_directories(args.outdir);

  const std::string out_path = (std::filesystem::path(args.outdir) / ("Data_TimeSeries_dir_" + std::to_string(args.L) + ".csv")).string();
  std::ofstream out(out_path);
  if (!out) throw std::runtime_error("Cannot open output: " + out_path);

  out << "L,J,R2,mean_R_ij,factors,Rg2,Rg,contacts,start_idx,E,step,source\n";

  FastScanner fs(args.dirs_file);
  const std::string source = file_source_tag(args.dirs_file);

  long long step = 0;
  double J = 0.0, E = 0.0;
  int start_idx = 0;

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

    // read coords in ring-buffer order
    for (int i = 0; i < args.L; ++i) {
      int x=0,y=0,z=0;
      if (!fs.readInt(x)) x=0;
      if (!fs.readInt(y)) y=0;
      if (args.dim == 3) {
        if (!fs.readInt(z)) z=0;
      } else {
        z = 0;
      }
      ring[i] = { x, y, z };
    }

    if (args.chain >= 0 && chain_id != args.chain) continue;

    // rotate to chain order: start -> end using start_idx
    const int s = ((start_idx % args.L) + args.L) % args.L;

    std::vector<Vec3i> chain(args.L);
    for (int t = 0; t < args.L; ++t) {
      chain[t] = ring[(s + t) % args.L];
    }

    // R2 between start and end on torus (matches your Python: end_idx = (idx-1)%N)
    const Vec3i& startp = ring[s];
    const Vec3i& endp   = ring[(s + args.L - 1) % args.L];
    const double R2 = torus_r2(startp, endp, args.side, args.dim);

    double mean_r2 = 0.0;
    double factors = 0.0;
    pairwise_stats_exact(chain, args.side, args.dim, mean_r2, factors);

    double Rg2 = 0.0, Rg = 0.0;
    gyration_radius_unwrapped(chain, args.side, args.dim, Rg2, Rg);

    const long long contacts = topological_contacts(chain, args.side, args.dim);

    out << args.L << "," << J << "," << R2 << "," << mean_r2 << "," << factors << ","
        << Rg2 << "," << Rg << "," << contacts << ","
        << start_idx << "," << E << "," << step << "," << source << "\n";
  }

  std::cerr << "Wrote: " << out_path << "\n";
}

int main(int argc, char** argv) {
  Args args = parse_args(argc, argv);

  #ifdef _OPENMP
  if (args.threads > 0) omp_set_num_threads(args.threads);
  #else
  (void)args.threads;
  #endif

  try {
    process_angles(args);
    process_dirs(args);
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 2;
  }
  return 0;
}