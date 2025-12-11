#include "trajectories.hpp"
#include <chrono>
#include <filesystem>
#include <algorithm>
#include <fstream>

namespace fs = std::filesystem;

int extract_n(const std::string &s)
{
    std::smatch m;
    std::regex re("_n(\\d+)");
    if (std::regex_search(s, m, re))
        return std::stoi(m.str(1));
    return 0;
}

static inline void print_complex_res(const cuDoubleComplex &z)
{
    std::cout << "(" << std::fixed << std::setprecision(12)
              << z.x << (z.y >= 0 ? "+" : "") << z.y << "i)";
}

int main(int argc, char *argv[])
{
    using clock = std::chrono::steady_clock;
    std::string dir_path = "benchmarks";

    if (argc < 4)
    {
        std::cerr << "Usage: " << argv[0] << " <prefix> <output_file> <0/1> [optional: log_shots]\n";
        return 1;
    }

    std::string prefix = argv[1];
    std::string out_file = argv[2];
    int show_results = std::stoi(argv[3]);
    int fixed_log = (argc == 5) ? std::stoi(argv[4]) : -1;

    if (!fs::exists(dir_path))
    {
        std::cerr << "Error: Directory not found.\n";
        return 1;
    }

    // Scan and Sort Files
    std::vector<fs::path> files;
    for (const auto &e : fs::directory_iterator(dir_path))
    {
        if (e.path().extension() == ".qasm" && e.path().filename().string().find(prefix) == 0)
        {
            files.push_back(e.path());
        }
    }

    std::sort(files.begin(), files.end(), [](const fs::path &a, const fs::path &b)
              { 
        int na = extract_n(a.string());
        int nb = extract_n(b.string()); 
        return na != nb ? na < nb : a.string() < b.string(); });

    std::ofstream outfile(out_file);
    outfile << "Filename,Qubits,TotalRuntime(s)\n";

    // Benchmarking Loop
    for (const auto &fpath : files)
    {
        std::string fname = fpath.string();
        std::string sname = fpath.filename().string();
        std::cout << ">>> Benchmarking: " << sname << " ... \n";

        try
        {
            std::map<std::string, int> tm;
            int nq = Trajectories::get_qubits_and_map_from_qasm(fname, tm);
            int ns = (fixed_log != -1) ? (1 << fixed_log) : (1 << std::min(30, nq + 5));

            auto t0 = clock::now();
            {
                Trajectories circ(fname, ns);
                circ.run_qasm_file(fname);

                if (show_results)
                {
                    auto stats = circ.measure_stats();
                    std::vector<Trajectories::MeasurementStat> kept;

                    for (auto &s : stats)
                    {
                        if (s.prob_raw >= 1e-4)
                            kept.push_back(s);
                    }
                    if (kept.empty() && !stats.empty())
                    {
                        kept.push_back(*std::max_element(stats.begin(), stats.end(), [](auto &a, auto &b)
                                                         { return a.prob_raw < b.prob_raw; }));
                    }
                    std::sort(kept.begin(), kept.end(), [](auto &a, auto &b)
                              { return a.prob_raw > b.prob_raw; });

                    std::cout << "\n    --- Results ---\n";
                    for (size_t i = 0; i < std::min((size_t)10, kept.size()); ++i)
                    {
                        std::cout << "    " << std::setw(nq + 4) << std::left << kept[i].ket
                                  << ": Prob=" << kept[i].prob_raw
                                  << " avgW=";
                        print_complex_res(kept[i].avg_w);
                        std::cout << "\n";
                    }
                }
            }
            double dur = std::chrono::duration<double>(clock::now() - t0).count();
            std::cout << "    Done (" << dur << " s)\n";
            outfile << sname << "," << nq << "," << dur << "\n";
            outfile.flush();
        }
        catch (const std::exception &e)
        {
            std::cerr << "    [FAILED]: " << e.what() << "\n";
        }
    }
    return 0;
}