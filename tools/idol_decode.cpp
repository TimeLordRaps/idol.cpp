// idol_decode: runs the real Idol decoders (speridlabs/idol) on the CPU.
//
// Usage:
//   idol_decode <dir> --check [--case DIR] [--level N] [--threads N] [--octree-only | --bench ROWS --repeat N]
//   idol_decode <dir> --latent FILE --out FILE.ply [--gaussians N] [--level N] [--seed N] [--threads N]
//   idol_decode <dir> --random-latent TOKENS --out FILE.ply [...]
//
// dir is an export from tools/export_idol.py. --check runs the reference case stored in it (anchors.bin, cond.bin,
// ref_oct.bin, ref_gs.bin) through both decoders and prints how far the outputs are from PyTorch's, as the largest
// absolute difference and the error relative to the reference's own size. The exit status is 1 when either relative
// error exceeds --tolerance (default 1e-4). The other forms decode a latent (float32, tokens x 16, raw) to a 3DGS PLY.
// --level is the octree depth (default 5 for --check, which matches the export, and 8 otherwise, as the model uses);
// --gaussians defaults to 262144.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "cpu_time.hpp"
#include "idol/decoders.hpp"
#include "octree_compare.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

std::vector<float> read_floats(const std::string& path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) throw std::runtime_error("cannot open " + path);
    const std::streamoff bytes = in.tellg();
    if (bytes < 0 || bytes % sizeof(float) != 0) throw std::runtime_error("invalid float file size: " + path);
    std::vector<float> v((std::size_t)bytes / sizeof(float));
    in.seekg(0);
    if (!in.read(reinterpret_cast<char*>(v.data()), bytes)) throw std::runtime_error("cannot read " + path);
    return v;
}

double seconds_since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
}

struct Diff {
    double max_abs = 0.0, relative = 0.0;
};

Diff compare(const std::vector<float>& got, const std::vector<float>& ref) {
    if (got.size() != ref.size() || ref.empty()) throw std::runtime_error("reference size mismatch");
    double err2 = 0.0, ref2 = 0.0;
    Diff d;
    for (std::size_t i = 0; i < ref.size(); ++i) {
        if (!std::isfinite(got[i]) || !std::isfinite(ref[i])) throw std::runtime_error("non-finite decoder output or reference");
        const double e = (double)got[i] - ref[i];
        d.max_abs = std::max(d.max_abs, std::fabs(e));
        err2 += e * e;
        ref2 += (double)ref[i] * ref[i];
    }
    d.relative = std::sqrt(err2 / std::max(ref2, 1e-300));
    return d;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        std::string dir, latent, out, case_dir;
        int threads = 0, level = 0, gaussians = 262144, random_tokens = 0, bench_rows = 0, repeat = 5;
        unsigned long long seed = 0;
        double tolerance = 1e-4;
        bool check = false, octree_only = false;
        for (int i = 1; i < argc; ++i) {
            const std::string a = argv[i];
            auto next = [&]() -> std::string {
                if (i + 1 >= argc) throw std::invalid_argument(a + " needs a value");
                return argv[++i];
            };
            if (a == "--check") check = true;
            else if (a == "--octree-only") octree_only = true;
            else if (a == "--case") case_dir = next();
            else if (a == "--bench") bench_rows = std::stoi(next());
            else if (a == "--repeat") repeat = std::stoi(next());
            else if (a == "--latent") latent = next();
            else if (a == "--random-latent") random_tokens = std::stoi(next());
            else if (a == "--out") out = next();
            else if (a == "--gaussians") gaussians = std::stoi(next());
            else if (a == "--level") level = std::stoi(next());
            else if (a == "--seed") seed = std::stoull(next());
            else if (a == "--threads") threads = std::stoi(next());
            else if (a == "--tolerance") tolerance = std::stod(next());
            else if (!a.empty() && a[0] == '-') throw std::invalid_argument("unknown option " + a);
            else dir = a;
        }
        if (dir.empty() || (!check && out.empty()) || (!check && latent.empty() && random_tokens <= 0)) {
            std::fprintf(stderr,
                         "usage: idol_decode <dir> --check | --latent FILE --out FILE.ply [--gaussians N] [--level N] "
                         "[--seed N] [--threads N]\n");
            return 2;
        }
        if (level < 0 || level > 20) throw std::invalid_argument("level must be between 1 and 20");
        if (!std::isfinite(tolerance) || tolerance < 0) throw std::invalid_argument("tolerance must be finite and nonnegative");
        if (repeat < 1 || bench_rows < 0) throw std::invalid_argument("repeat must be positive and bench must be nonnegative");
        if (octree_only && (!check || bench_rows > 0)) throw std::invalid_argument("--octree-only requires --check without --bench");
#ifdef _OPENMP
        if (threads > 0) omp_set_num_threads(threads);
#endif
        auto t0 = std::chrono::steady_clock::now();
        idol::Decoders dec(dir);
        const int C = dec.config().cond;
        std::printf("loaded %s in %.1f s; resident weights %.1f MB\n", dir.c_str(), seconds_since(t0),
                    dec.weight_bytes() / 1e6);

        if (check) {
            const std::string cases = case_dir.empty() ? dir : case_dir;
            const std::vector<float> anchors = read_floats(cases + "/anchors.bin"), cond = read_floats(cases + "/cond.bin"),
                                     ref_oct = read_floats(cases + "/ref_oct.bin"), ref_gs = read_floats(cases + "/ref_gs.bin");
            const int tokens = (int)(cond.size() / C);
            const int lv = level > 0 ? level : 5;
            if (anchors.empty() || anchors.size() % 3 || cond.empty() || cond.size() % C)
                throw std::runtime_error("invalid anchors or conditioning dimensions");
            const std::size_t case_rows = anchors.size() / 3;
            if (ref_oct.size() != case_rows * 8 || ref_gs.size() != case_rows * dec.config().per_anchor * 15)
                throw std::runtime_error("reference size mismatch");
            for (float v : anchors) if (!std::isfinite(v)) throw std::runtime_error("non-finite anchor");
            for (float v : cond) if (!std::isfinite(v)) throw std::runtime_error("non-finite conditioning");
            if (bench_rows > 0) {
                // Timing only: the first bench_rows anchors, one warm-up run, then the median of --repeat runs.
                const int n = std::min(bench_rows, (int)(anchors.size() / 3));
                std::vector<float> o((std::size_t)n * 8), g((std::size_t)n * dec.config().per_anchor * 15);
                auto median_of = [&](auto&& run) {
                    run();
                    std::vector<double> t;
                    for (int i = 0; i < repeat; ++i) {
                        const auto start = std::chrono::steady_clock::now();
                        run();
                        t.push_back(seconds_since(start));
                    }
                    std::sort(t.begin(), t.end());
                    const std::size_t mid = t.size() / 2;
                    return t.size() % 2 ? t[mid] : (t[mid - 1] + t[mid]) / 2;
                };
                const double t_o = median_of([&] { dec.octree_logits(anchors.data(), n, (float)(1 << lv), cond.data(), tokens, o.data()); });
                const double t_g = median_of([&] { dec.gaussian_features(anchors.data(), n, cond.data(), tokens, g.data()); });
                std::printf("BENCH rows=%d tokens=%d oct_s=%.6f gs_s=%.6f resident_mb=%.2f peak_mb=%.1f cpu_s=%.2f\n", n, tokens, t_o, t_g,
                            dec.weight_bytes() / 1e6, idol::tools::peak_memory_mb(), idol::tools::process_cpu_ms() / 1e3);
                return 0;
            }
            const int rows = (int)(anchors.size() / 3);
            std::vector<float> oct((std::size_t)rows * 8), gs((std::size_t)rows * dec.config().per_anchor * 15);
            t0 = std::chrono::steady_clock::now();
            dec.octree_logits(anchors.data(), rows, (float)(1 << lv), cond.data(), tokens, oct.data());
            const double t_oct = seconds_since(t0);
            const Diff a = compare(oct, ref_oct);
            const auto probabilities = idol::detail::compare_octree_probabilities(ref_oct, oct);
            if (octree_only) {
                std::printf("OCT_RESULT rows=%d tokens=%d oct_rel=%.6e oct_prob_mean=%.6e oct_prob_max=%.6e "
                            "resident_mb=%.2f\n", rows, tokens, a.relative, probabilities.mean_total_variation,
                            probabilities.max_total_variation, dec.weight_bytes() / 1e6);
                return a.relative > tolerance ? 1 : 0;
            }
            t0 = std::chrono::steady_clock::now();
            dec.gaussian_features(anchors.data(), rows, cond.data(), tokens, gs.data());
            const double t_gs = seconds_since(t0);
            const Diff b = compare(gs, ref_gs);
            std::printf("octree decoder   (%d cells, %d latent tokens): %.2f s; max |diff| %.3e, relative %.3e\n", rows,
                        tokens, t_oct, a.max_abs, a.relative);
            std::printf("gaussian decoder (%d anchors, %d latent tokens): %.2f s; max |diff| %.3e, relative %.3e\n", rows,
                        tokens, t_gs, b.max_abs, b.relative);
            std::printf("octree child probabilities: mean total variation %.3e, worst %.3e\n",
                        probabilities.mean_total_variation, probabilities.max_total_variation);
            // What the error means for the splats: both outputs turned into activated Gaussians, then compared.
            const idol::GaussianCloud want = dec.to_gaussians(anchors.data(), rows, ref_gs.data()),
                                      have = dec.to_gaussians(anchors.data(), rows, gs.data());
            double pos2 = 0.0, col2 = 0.0, opa = 0.0, ls2 = 0.0;
            for (std::size_t i = 0; i < want.size(); ++i) {
                for (int c = 0; c < 3; ++c) {
                    const double dp = (double)have.xyz[i * 3 + c] - want.xyz[i * 3 + c];
                    const double dc = (double)have.f_dc[i * 3 + c] - want.f_dc[i * 3 + c];
                    const double ds = std::log((double)have.scale[i * 3 + c]) - std::log((double)want.scale[i * 3 + c]);
                    pos2 += dp * dp, col2 += dc * dc, ls2 += ds * ds;
                }
                opa += std::fabs((double)have.opacity[i] - want.opacity[i]);
            }
            const double n = (double)want.size();
            const double pos = std::sqrt(pos2 / (3 * n)), col = std::sqrt(col2 / (3 * n)), mae = opa / n, lsc = std::sqrt(ls2 / (3 * n));
            std::printf("gaussians: position rms %.3e (object spans ~1), colour rms %.3e, opacity mae %.3e, log-scale rms %.3e\n",
                        pos, col, mae, lsc);
            std::printf("peak memory %.1f MB\n", idol::tools::peak_memory_mb());
            std::printf("RESULT oct_rel=%.6e gs_rel=%.6e oct_s=%.3f gs_s=%.3f resident_mb=%.2f peak_mb=%.1f pos=%.6e col=%.6e "
                        "opa=%.6e lsc=%.6e oct_prob_mean=%.6e oct_prob_max=%.6e\n",
                        a.relative, b.relative, t_oct, t_gs, dec.weight_bytes() / 1e6, idol::tools::peak_memory_mb(), pos, col,
                        mae, lsc, probabilities.mean_total_variation, probabilities.max_total_variation);
            return (a.relative > tolerance || b.relative > tolerance) ? 1 : 0;
        }

        std::vector<float> cond;
        if (!latent.empty()) {
            cond = read_floats(latent);
        } else {
            std::mt19937_64 rng(seed);
            std::normal_distribution<float> normal;
            cond.resize((std::size_t)random_tokens * C);
            for (float& v : cond) v = normal(rng);
        }
        if (cond.empty() || cond.size() % C) throw std::runtime_error("the latent is not a nonempty whole number of tokens of " + std::to_string(C) + " floats");
        for (float v : cond) if (!std::isfinite(v)) throw std::runtime_error("non-finite conditioning");
        const int tokens = (int)(cond.size() / C);
        t0 = std::chrono::steady_clock::now();
        const idol::GaussianCloud cloud = dec.decode(cond.data(), tokens, gaussians, level > 0 ? level : 8, seed);
        std::printf("decoded %zu Gaussians from %d latent tokens in %.1f s; peak memory %.1f MB\n", cloud.size(), tokens,
                    seconds_since(t0), idol::tools::peak_memory_mb());
        idol::write_gaussian_ply(cloud, out);
        std::printf("wrote %s\n", out.c_str());
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "idol_decode: %s\n", e.what());
        return 2;
    }
}
