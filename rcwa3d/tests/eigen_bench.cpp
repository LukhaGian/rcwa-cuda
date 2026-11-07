#include <chrono>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include <Eigen/Dense>

// CPU baseline for the cusolverDnXgeev benchmark (rcwa3d_cuda/tests/geev_bench.cu).
// Usage: ./eigen_bench <eigen|zgeev> [n1 n2 ...]   (default sizes: 500 1000 2000 3000)
//   eigen: Eigen::ComplexEigenSolver, as used by SMatrixLayer
//   zgeev: LAPACKE_zgeev, eigenvalues and right eigenvectors
// Threads: set OMP_NUM_THREADS / OPENBLAS_NUM_THREADS (e.g. 1, then all cores).
// Output: one CSV line per run; run 0 is a warm-up and should be ignored.
// The matrices use the same generator and seed as geev_bench, so equal size lists give equal matrices.

using Complex = std::complex<double>;
using Matrix = Eigen::Matrix<Complex, Eigen::Dynamic, Eigen::Dynamic>;
using Vector = Eigen::Matrix<Complex, Eigen::Dynamic, 1>;

int main(int argc, char** argv)
{
    /*
    Times the eigensolve for each size, reports CPU cores used and the relative residual
    ||A V - V diag(W)||_F / (||A||_F ||V||_F).
    */
    if (argc < 2 || (std::string(argv[1]) != "eigen" && std::string(argv[1]) != "zgeev"))
    {
        std::cerr << "usage: " << argv[0] << " <eigen|zgeev> [n1 n2 ...]\n";
        return 1;
    }
    const std::string method = argv[1];
    std::vector<int64_t> sizes{500, 1000, 2000, 3000};
    if (argc > 2)
    {
        sizes.clear();
        for (int a = 2; a < argc; ++a) sizes.push_back(std::atoll(argv[a]));
    }
    const int reps = 2; // timed runs per size, after one warm-up

    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    std::cout << "method,n,run,wall_s,cores_used,info,residual\n";

    for (const int64_t n : sizes)
    {
        // Random complex matrix, entries uniform in [-1, 1] for both parts, filled column by column
        Matrix A0(n, n);
        for (int64_t k = 0; k < n * n; ++k) A0.data()[k] = Complex(dist(rng), dist(rng));

        for (int run = 0; run <= reps; ++run)
        {
            Matrix A = A0; // zgeev overwrites its input
            Vector W(n);
            Matrix V(n, n);
            int info = 0;

            const auto wall_start = std::chrono::steady_clock::now();
            const std::clock_t cpu_start = std::clock(); // CPU time summed over all threads of the process

            if (method == "eigen")
            {
                Eigen::ComplexEigenSolver<Matrix> solver(A);
                info = (solver.info() == Eigen::Success) ? 0 : 1;
                W = solver.eigenvalues();
                V = solver.eigenvectors();
            }
            else
            {
                Matrix VL(1, 1); // not referenced with jobvl = 'N'
                info = LAPACKE_zgeev(LAPACK_COL_MAJOR, 'N', 'V', static_cast<lapack_int>(n),
                                     reinterpret_cast<lapack_complex_double*>(A.data()), static_cast<lapack_int>(n),
                                     reinterpret_cast<lapack_complex_double*>(W.data()),
                                     reinterpret_cast<lapack_complex_double*>(VL.data()), 1,
                                     reinterpret_cast<lapack_complex_double*>(V.data()), static_cast<lapack_int>(n));
            }

            const double wall_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - wall_start).count();
            const double cpu_s = static_cast<double>(std::clock() - cpu_start) / CLOCKS_PER_SEC;

            double residual = -1.0; // only computed on the last run
            if (run == reps && info == 0)
            {
                const Matrix R = A0 * V - V * W.asDiagonal();
                residual = R.norm() / (A0.norm() * V.norm());
            }

            std::cout << method << ',' << n << ',' << run << ',' << wall_s << ',' << cpu_s / wall_s << ','
                      << info << ',' << residual << std::endl;
        }
    }
    return 0;
}
