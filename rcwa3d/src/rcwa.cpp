// #include "../include/rcwa.hpp"
#include "rcwa.hpp"
#include <unsupported/Eigen/FFT>
#include <iostream>

Matrix ConvMat(const std::vector<Complex>& field, int layer, int Nx, int Ny, int Nx_harmonics, int Ny_harmonics)
{                 
    /*
    Function that returns the 2D convolution matrix for a given field for a specific layer.
    The input field is assumed to be a 2D array of size Ny x Nx, and the output convolution matrix will be of size
    PQ x PQ, where P = (2 * Nx_harmonics + 1) and Q = (2 * Ny_harmonics + 1)
    -> the convolution matrix is of size (2 * Nx_harmonics + 1) * (2 * Ny_harmonics + 1) x (2 * Nx_harmonics + 1) * (2 * Ny_harmonics + 1)
    */

    // Map the input field to a row-major matrix of size Ny x Nx
    using RowMajorMatrixXdMap = Eigen::Map<const Eigen::Matrix<Complex, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
    //                                  raw pointer of data + move it to the specific layer, then map it to a row-major matrix of size Ny x Nx
    RowMajorMatrixXdMap realRowMajorMap(field.data() + layer * Nx * Ny, Ny, Nx);
    // cast the real row-major matrix to a complex column-major matrix of size Ny x Nx (default Eigen matrix is column-major)
    //Matrix F = realRowMajorMap.cast<Complex>();
    Matrix F = realRowMajorMap; // no need to cast, data is already Complex

    // FFT computation
    Eigen::FFT<Real> fft;
    // By row
    Vector row_fft;
    for (int iy = 0; iy < Ny; ++iy)
    {
        fft.fwd(row_fft, F.row(iy));
        F.row(iy) = row_fft;
    }
    // By column
    Vector col_fft;
    for (int ix = 0; ix < Nx; ++ix)
    {
        fft.fwd(col_fft, F.col(ix));
        F.col(ix) = col_fft;
    }

    // Normalization of the FFT
    F /= static_cast<Real>(Nx * Ny);

    // FFT shift to center the zero frequency component
    Matrix F_shift(Ny, Nx);
    int half_Nx = Nx /2;
    int half_Ny = Ny /2;
    // F = [Q1 Q2]
    //     [Q3 Q4]
    // flip blocks -> Q1 to Q4, Q2 to Q3, Q3 to Q2, Q4 to Q1
    F_shift.block(half_Ny, half_Nx, Ny-half_Ny, Nx-half_Nx) = F.block(0, 0, Ny-half_Ny, Nx-half_Nx); // Q1 to Q4
    F_shift.block(0,0, half_Ny, half_Nx) = F.block(Ny-half_Ny, Nx-half_Nx, half_Ny, half_Nx); // Q4 to Q1
    F_shift.block(half_Ny, 0, Ny-half_Ny, half_Nx) = F.block(0, Nx-half_Nx, Ny-half_Ny, half_Nx); // Q2 to Q3
    F_shift.block(0, half_Nx, half_Ny, Nx-half_Nx) = F.block(Ny-half_Ny, 0, half_Ny, Nx-half_Nx); // Q3 to Q2

    // Toeplitz matrix assembly of size PQ x PQ
    int P = 2 * Nx_harmonics + 1;
    int Q = 2 * Ny_harmonics + 1;
    int PQ = P * Q;
    Matrix C = Matrix::Zero(PQ, PQ);
    
    for (int m = 0; m < PQ; ++m)
    {
        int py_m = m / P - Ny_harmonics; // y index of m
        int px_m = m % P - Nx_harmonics; // x index of m

        for (int n = 0; n < PQ; ++n)
        {
            int py_n = n / P - Ny_harmonics; // y index of n
            int px_n = n % P - Nx_harmonics; // x index of n

            int kx_idx = (px_m - px_n) + half_Nx; // index in the FFT shifted matrix along x
            int ky_idx = (py_m - py_n) + half_Ny; // index in the FFT shifted matrix along y
            if (kx_idx >= 0 && kx_idx < Nx && ky_idx >= 0 && ky_idx < Ny)
            {
                C(m, n) = F_shift(ky_idx, kx_idx);
            }
        }
    }

    return C;

}


void ComputeWaveVectors(const Device& device, const Source& source, const RCWAParams& params, std::vector<Complex>& k_inc, Vector& Kx, Vector& Ky, Vector& Kz_ref, Vector& Kz_trn)
{
    /*
    Function that computes the Wave Vector Expansion.
    */
    // incidence vector
    Complex n_inc = std::sqrt(params.er_ref * params.ur_ref); // refractive index, added supprt for non lossless materials
    std::cout << n_inc << '\n';
    Real sin_theta = std::sin(source.theta);
    Real cos_theta = std::cos(source.theta);
    Real sin_phi = std::sin(source.phi);
    Real cos_phi = std::cos(source.phi);
    Real delta = 1e-8; // infinitesimal angle shift

    k_inc[0] = n_inc * sin_theta * cos_phi + delta; // infinitesimal angle shift for handling degenerate cases
    std::cout << "k_inc[0]" << '\n';
    std::cout << k_inc[0] << '\n';
    k_inc[1] = n_inc * sin_theta * sin_phi + delta; // infinitesimal angle shift
    k_inc[2] = n_inc * cos_theta;

    // Wave vector components
    Vector k_x_tilde = Vector::LinSpaced(2*params.Nx_harmonics + 1, -params.Nx_harmonics, params.Nx_harmonics);
    Vector k_y_tilde = Vector::LinSpaced(2*params.Ny_harmonics + 1, -params.Ny_harmonics, params.Ny_harmonics);
    std::cout << '\n';
    //std::cout << k_y_tilde << '\n';
    k_x_tilde *= -(source.lambda0 / device.Lx);
    //k_x_tilde *= -(2 * M_PI / (source.k0 * device.Lx));
    k_x_tilde.array() += k_inc[0];
    k_y_tilde *= -(source.lambda0 / device.Ly);
    //k_y_tilde *= -(2 * M_PI / (source.k0 * device.Ly));
    k_y_tilde.array() += k_inc[1];

    Matrix Kx_tilde = Matrix::Zero(k_y_tilde.size(), k_x_tilde.size());
    Matrix Ky_tilde = Matrix::Zero(k_y_tilde.size(), k_x_tilde.size());
    MeshGrid(k_x_tilde, k_y_tilde, Kx_tilde, Ky_tilde);
    std::cout << Kx_tilde << '\n';
    std::cout << '\n';
    std::cout << Ky_tilde << '\n';
    std::cout << '\n';
    // Longitudinal vector components in the reflection and transmission region
    // NOTE: Added conjugate operation for non lossless materials (in general complex values)
    //Matrix k_z_ref = -(std::conj(params.ur_ref) * std::conj(params.er_ref) - Kx_tilde.array().square() - Ky_tilde.array().square()).sqrt().conjugate();
    //Matrix k_z_trn = (std::conj(params.ur_trn) * std::conj(params.er_trn) - Kx_tilde.array().square() - Ky_tilde.array().square()).sqrt().conjugate();
    std::cout << std::conj(params.ur_ref) << '\n';

    /*
    Sign convention: e^{-jkz} for forward propagating waves
    --> k_z_ref: Re < 0 for propagating, Im > 0 for evanescent
    custom correct branch of sqrt enforced via unsigned_sqrt
    */
    Matrix k_z_ref = -(unsigned_sqrt(std::conj(params.ur_ref) * std::conj(params.er_ref) - Kx_tilde.array().square() - Ky_tilde.array().square()).conjugate());
    Matrix k_z_trn = unsigned_sqrt(std::conj(params.ur_trn) * std::conj(params.er_trn) - Kx_tilde.array().square() - Ky_tilde.array().square()).conjugate();
    //std::cout << Kx_tilde.rows() << " " << Kx_tilde.cols() << '\n';
    //std::cout << k_z_ref.rows() << " " << k_z_ref.cols() << '\n';
    //std::cout << k_z_ref << '\n';
    //std::cout << k_z_trn << '\n';
    std::cout << '\n';
    std::cout << "k_z_ref" << '\n';
    std::cout << k_z_ref << '\n';
    Kx = Kx_tilde.reshaped();
    Ky = Ky_tilde.reshaped();
    Kz_ref = k_z_ref.reshaped(); // a little bit of inconsistency
    Kz_trn = k_z_trn.reshaped();
    //std::cout << k_z_ref << '\n';
    //std::cout << k_z_ref.rows() << " " << k_z_ref.cols() << '\n';
    //std::cout << Kz_ref << '\n';
    std::cout << "Kx" << '\n';
    std::cout << '\n';
    std::cout << Kx << '\n';
    std::cout << '\n';
    std::cout << "Ky" << '\n';
    std::cout << '\n';
    std::cout << Ky << '\n';


    // NOTE: REMEMBER TO USE THE MATRICES/VECTORS Kx, Ky, Kz_ref AND Kz_trn WITH THE CALL .asDiagonal()
}


void GapMedium(const Vector& Kx, const Vector& Ky, Matrix& W0, Matrix& V0)
{
    /*
    Compute Eigenmodes of the Gap medium.
    */
    Vector Kz = unsigned_sqrt(1.0 - Kx.array().square() - Ky.array().square()).conjugate();
    std::cout << "Kz" << '\n';
    std::cout << '\n';
    std::cout << Kz << '\n';

    int block_size = Kz.size();

    Vector Kx_Ky = Kx.array() * Ky.array();
    //Vector main_diag(Kx_Ky.size()*2);
    //main_diag << Kx_Ky, -Kx_Ky;
    //Matrix top_left_Mat = Kx_Ky.asDiagonal();
    //std::cout << top_left_Mat << '\n';
    //std::cout << '\n';
    //std::cout << top_left_Mat.rows() << " " << top_left_Mat.cols() << '\n';
    Vector top_right = 1.0 - Kx.array().square();
    //Matrix top_right_Mat = top_right.asDiagonal();
    Vector bottom_left = Ky.array().square() - 1.0;
    //Matrix bottom_left_Mat = bottom_left.asDiagonal();
    //Matrix bottom_right_Mat = - top_left_Mat;
    Matrix Q(2*block_size, 2*block_size);

    Q.block(0, 0, block_size, block_size) = Kx_Ky.asDiagonal();
    Q.block(0, block_size, block_size, block_size) = top_right.asDiagonal();
    Q.block(block_size, 0, block_size, block_size) = bottom_left.asDiagonal();
    Q.block(block_size, block_size, block_size, block_size) = (- Kx_Ky).asDiagonal();
    //std::cout << Q <<'\n';

    W0 = Matrix::Identity(2*block_size, 2*block_size);

    Vector lambda(2*block_size);
    lambda << 1i*Kz, 1i*Kz;
    //std::cout << '\n';
    //std::cout << lambda << '\n';
    // Possible instability for lambda?
    Matrix Test = Q * lambda.array().inverse().matrix().asDiagonal(); // check if .matrix is really needed
   
    //V0 = Q.array().rowwise() / lambda.array();
    // check if it's the best way to do this
    for (int j = 0; j < Q.cols(); ++j)
    {
        V0.col(j) = Q.col(j) / lambda(j); // are we potentially dividing for 0? Yeah....
    }
    //std::cout << V0 << '\n';
    if (V0.isApprox(Test))
        std::cout << "Similar" << '\n';
    //std::cout << V0 << '\n';

}


ScatteringMatrix SMatrixInit(int Nx_harmonics, int Ny_harmonics)
{
    /*
    Function that initialize the device Scattering matrix. The various components are interpreted as:
    S11 = 0 --> reflection at the first interface (init as zeros)
    S12 = I --> transmission in the forward direction
    S21 = I --> transmission in the backward direction
    S22 = 0 --> reflection at the last interface
    */
    int P = 2 * Nx_harmonics + 1;
    int Q = 2 * Ny_harmonics + 1;
    int PQ = P * Q;
    return ScatteringMatrix(Matrix::Zero(2*PQ, 2*PQ), Matrix::Identity(2*PQ, 2*PQ), Matrix::Identity(2*PQ, 2*PQ), Matrix::Zero(2*PQ, 2*PQ));
}



ScatteringMatrix SMatrixLayer(int layer, const Device& device, const Source& source, const RCWAParams& params, const Vector& Kx, const Vector& Ky, const Matrix& W0, const Matrix& V0)
{
    /*
    Function that computes the layer scattering matrix for the given ith layer. TO BE TESTED
    */
    int P = 2 * params.Nx_harmonics + 1;
    int Q = 2 * params.Ny_harmonics + 1;
    int PQ = P * Q;
    // Build eigenvalue problem
    // P_i
    Matrix P_i = Matrix::Zero(2*PQ, 2*PQ);
    //                             TO BE SUBSTITUTED WITH NO BOUND CHECK [], faster
    //                                                  view of Kx creates a temporary full dense matrix ==> A bit of a waste!!!!
    Matrix erc_inv_Kx = device.erc.at(layer).lu().solve(Kx.asDiagonal().toDenseMatrix());
    Matrix erc_inv_Ky = device.erc.at(layer).lu().solve(Ky.asDiagonal().toDenseMatrix());
    // P_i
    P_i.block(0, 0, PQ, PQ) = Kx.asDiagonal() * erc_inv_Ky; // top left
    P_i.block(0, PQ, PQ, PQ) = device.urc.at(layer) - Kx.asDiagonal() * erc_inv_Kx; // top right
    P_i.block(PQ, 0, PQ, PQ) = Ky.asDiagonal() * erc_inv_Ky - device.urc.at(layer); // bottom left
    P_i.block(PQ, PQ, PQ, PQ) = (-Ky).asDiagonal() * erc_inv_Kx; // bottom right
    // Q_i
    Matrix Q_i = Matrix::Zero(2*PQ, 2*PQ);
    Matrix urc_inv_Kx = device.urc.at(layer).lu().solve(Kx.asDiagonal().toDenseMatrix());
    Matrix urc_inv_Ky = device.urc.at(layer).lu().solve(Ky.asDiagonal().toDenseMatrix());
    Q_i.block(0, 0, PQ, PQ) = Kx.asDiagonal() * urc_inv_Ky;
    Q_i.block(0, PQ, PQ, PQ) = device.erc.at(layer) - Kx.asDiagonal() * urc_inv_Kx;
    Q_i.block(PQ, 0, PQ, PQ) = Ky.asDiagonal() * urc_inv_Ky - device.erc.at(layer);
    Q_i.block(PQ, PQ, PQ, PQ) = (-Ky).asDiagonal() * urc_inv_Kx;

    Matrix Omega2 = P_i * Q_i;

    // solve eigenvalue problem Omega2 * x = lambda * x
    // In general EigenSolver<Matrix> solver(A)
    // solver.eigenvalues()
    // solver.eigenvectors()
    // if A = A^T --> use SelfAdjointEigenSolver<Matrix> solver(A)
    // expA = A.matrixExponential();
    Eigen::ComplexEigenSolver<Matrix> solver(Omega2);

    if (solver.info() != Eigen::Success) abort();

    std::cout << solver.eigenvectors() << '\n';
    std::cout << solver.eigenvalues() << '\n';

    auto [lambda2, W_i] = extraction(std::move(solver));

    std::cout << lambda2 << '\n';
    std::cout << W_i << '\n';

    // Check a more elegant way of computing it =======================
    Matrix V_i = Q_i * W_i * unsigned_sqrt(lambda2.array()).inverse().matrix().asDiagonal(); // is .array() really needed? lazy expr...
    Matrix W_i_inv_W0 = W_i.lu().solve(W0);
    Matrix V_i_inv_V0 = V_i.lu().solve(V0);
    Matrix A_i0 = W_i_inv_W0  + V_i_inv_V0;
    Matrix B_i0 = W_i_inv_W0 - V_i_inv_V0;
    Matrix X_i = (-source.k0 * device.t[layer] * unsigned_sqrt(lambda2.array())).exp().matrix().asDiagonal(); // Matrix Exponential

    // Assemble S matrix
    // Lazy implementation, enable NRVO next
    auto A_i0_lu = A_i0.lu();
    auto partial_prod_lu = (A_i0 - X_i * B_i0 * A_i0_lu.solve(X_i) * B_i0).lu();
    Matrix S11 = partial_prod_lu.solve(X_i * B_i0 * A_i0_lu.solve(X_i) * A_i0 - B_i0);
    Matrix S12 = partial_prod_lu.solve(X_i) * (A_i0 - B_i0 * A_i0_lu.solve(B_i0));

    return ScatteringMatrix(S11, S12, S12, S11);
}



ScatteringMatrix RedhefferProduct(const ScatteringMatrix& A, const ScatteringMatrix& B)
{
    /*
    Function that computes the Redheffer star product of two scattering matrices A and B.
    The Redheffer star product is defined as follows:
    S = A x B = [S11, S12; S21, S22] where
    S11 = B11 * (I - A12 * B21)^(-1) * A11
    S12 = B12 + B11 * (I - A12 * B21)^(-1) * A12 * B22
    S21 = A21 + A22 * (I - B21 * A12)^(-1) * B21 * A11
    S22 = A22 * (I - B21 * A12)^(-1) * B22,
    where Xnm = X.Snm, and I is the identity matrix of appropriate size.
    */

    // To be merged for efficiency
    Matrix I_A12_B21 = Matrix::Identity(A.S12.rows(), B.S21.cols()) - A.S12 * B.S21;
    Matrix I_B21_A12 = Matrix::Identity(B.S21.rows(), A.S12.cols()) - B.S21 * A.S12;

    auto lu_I_A12_B21 = I_A12_B21.transpose().lu();
    auto lu_I_B21_A12 = I_B21_A12.transpose().lu();

    Matrix shared_first_row_block = lu_I_A12_B21.solve(B.S11.transpose()).transpose();
    Matrix shared_second_row_block = lu_I_B21_A12.solve(A.S22.transpose()).transpose();
    

    return ScatteringMatrix(shared_first_row_block * A.S11, B.S12 + shared_first_row_block * A.S12 * B.S22,
                            A.S21 + shared_second_row_block * B.S21 * A.S11, shared_second_row_block * B.S22);

}

EigenvalSolverResults extraction(Eigen::ComplexEigenSolver<Matrix>&& solver)
{
        // Return values are constructed directly in the caller's memory frame
    return EigenvalSolverResults{solver.eigenvalues(), solver.eigenvectors()};
}