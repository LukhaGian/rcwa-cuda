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
        int px_m = m / Q - Nx_harmonics;  // x is SLOW index
        int py_m = m % Q - Ny_harmonics;  // y is FAST index

        for (int n = 0; n < PQ; ++n)
        {
            int px_n = n / Q - Nx_harmonics;
            int py_n = n % Q - Ny_harmonics;

            int kx_idx = (px_m - px_n) + half_Nx;
            int ky_idx = (py_m - py_n) + half_Ny;
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
    Complex n_inc = std::sqrt(params.er_ref * params.ur_ref); // refractive index, added support for non lossless materials
    std::cout << n_inc << '\n';
    Real sin_theta = std::sin(source.theta);
    Real cos_theta = std::cos(source.theta);
    Real sin_phi = std::sin(source.phi);
    Real cos_phi = std::cos(source.phi);
    //Real delta = 0; // infinitesimal angle shift

    k_inc[0] = n_inc * sin_theta * cos_phi;
    //std::cout << "k_inc[0]" << '\n';
    //std::cout << k_inc[0] << '\n';
    k_inc[1] = n_inc * sin_theta * sin_phi;
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

    //std::cout << std::conj(params.ur_ref) << '\n';

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
    
    /*
    std::cout << "Gap Medium" << '\n';
    std::cout << '\n';
    std::cout << - Kx.array().square() - Ky.array().square() << '\n';
    std::cout << '\n';
    std::cout << 1.0 - Kx.array().square() - Ky.array().square() << '\n';
    std::cout << '\n';
    std::cout << unsigned_sqrt(1.0 - Kx.array().square() - Ky.array().square()) << '\n';
    std::cout << '\n';
    */
    Vector Kz = unsigned_sqrt(1.0 - Kx.array().square() - Ky.array().square()).conjugate();
    //std::cout << "Kz" << '\n';
    //std::cout << '\n';
    //std::cout << Kz << '\n';

    int block_size = Kz.size();

    Vector Kx_Ky = Kx.array() * Ky.array();
    //std::cout << '\n';
    //std::cout << Kx_Ky << '\n';
    //std::cout << '\n';

    Vector top_right = 1.0 - Kx.array().square();
    Vector bottom_left = Ky.array().square() - 1.0;

    Matrix Q = Matrix::Zero(2*block_size, 2*block_size);
    Q.block(0, 0, block_size, block_size) = Kx_Ky.asDiagonal();
    //std::cout << Q.block(0, 0, block_size, block_size) << '\n';
    Q.block(0, block_size, block_size, block_size) = top_right.asDiagonal();
    //std::cout << '\n';
    //std::cout << Q.block(0, block_size, block_size, block_size) << '\n';
    Q.block(block_size, 0, block_size, block_size) = bottom_left.asDiagonal();
    //std::cout << '\n';
    //std::cout << Q.block(block_size, 0, block_size, block_size) << '\n';
    Q.block(block_size, block_size, block_size, block_size) = (- Kx_Ky).asDiagonal();
    //std::cout << '\n';
    //std::cout << Q.block(block_size, block_size, block_size, block_size) << '\n';
    //std::cout << Q <<'\n';
    //std::cout << "Q " << double((Q.array() != 0.0).count()) / Q.size() << '\n';
    
    W0 = Matrix::Identity(2*block_size, 2*block_size);
    //std::cout <<'\n';
    //std::cout << W0 << '\n';

    Vector lambda(2*block_size);
    lambda << 1i*Kz, 1i*Kz;
    //std::cout << '\n';
    //std::cout << "lambda" << '\n';
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
    //std::cout << '\n';
    //std::cout << "V0" << '\n';
    //std::cout << V0.block(block_size, 0, block_size, block_size) << '\n';

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
    Function that computes the layer scattering matrix for the given ith layer.
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
    std::cout << "erc_inv_Kx " << double((erc_inv_Kx.array() != 0.0).count()) / erc_inv_Kx.size() << '\n';
    //std::cout << "Test1" << '\n';
    Matrix erc_inv_Ky = device.erc.at(layer).lu().solve(Ky.asDiagonal().toDenseMatrix());
    std::cout << "erc_inv_Ky " << double((erc_inv_Ky.array() != 0.0).count()) / erc_inv_Ky.size() << '\n';
    // P_i
    P_i.block(0, 0, PQ, PQ) = Kx.asDiagonal() * erc_inv_Ky; // top left
    P_i.block(0, PQ, PQ, PQ) = device.urc.at(layer) - Kx.asDiagonal() * erc_inv_Kx; // top right
    P_i.block(PQ, 0, PQ, PQ) = Ky.asDiagonal() * erc_inv_Ky - device.urc.at(layer); // bottom left
    P_i.block(PQ, PQ, PQ, PQ) = (-Ky).asDiagonal() * erc_inv_Kx; // bottom right
    std::cout << "P_i " << double((P_i.array() != 0.0).count()) / P_i.size() << '\n';
    // Q_i
    Matrix Q_i = Matrix::Zero(2*PQ, 2*PQ);
    Matrix urc_inv_Kx = device.urc.at(layer).lu().solve(Kx.asDiagonal().toDenseMatrix());
    Matrix urc_inv_Ky = device.urc.at(layer).lu().solve(Ky.asDiagonal().toDenseMatrix());
    Q_i.block(0, 0, PQ, PQ) = Kx.asDiagonal() * urc_inv_Ky;
    Q_i.block(0, PQ, PQ, PQ) = device.erc.at(layer) - Kx.asDiagonal() * urc_inv_Kx;
    Q_i.block(PQ, 0, PQ, PQ) = Ky.asDiagonal() * urc_inv_Ky - device.erc.at(layer);
    Q_i.block(PQ, PQ, PQ, PQ) = (-Ky).asDiagonal() * urc_inv_Kx;
    std::cout << "Q_i " << double((Q_i.array() != 0.0).count()) / Q_i.size() << '\n';
    //std::cout << "Test2" << '\n';
    Matrix Omega2 = P_i * Q_i;
    std::cout << "Omega2 " << double((Omega2.array() != 0.0).count()) / Omega2.size() << '\n';


    // solve eigenvalue problem Omega2 * x = lambda * x
    // In general EigenSolver<Matrix> solver(A)
    // solver.eigenvalues()
    // solver.eigenvectors()
    // if A = A^T --> use SelfAdjointEigenSolver<Matrix> solver(A)
    // expA = A.matrixExponential();
    Eigen::ComplexEigenSolver<Matrix> solver(Omega2);

    if (solver.info() != Eigen::Success) abort();
    //std::cout << "Test3" << '\n';

    //std::cout << solver.eigenvectors() << '\n';
    //std::cout << solver.eigenvalues() << '\n';

    auto [lambda2, W_i] = extraction(std::move(solver));
    //std::cout << "Test4" << '\n';

    //std::cout << "lambda2" << '\n';
    //std::cout << lambda2 << '\n';
    //std::cout << '\n';
    //std::cout << W_i << '\n';
    //if (lambda2.isApprox(solver.eigenvalues()) && W_i.isApprox(solver.eigenvectors()))
    //    std::cout << "Similar" << '\n';

    //std::cout << (unsigned_sqrt(lambda2.array())).array().square() << '\n';
    auto lambda = unsigned_sqrt(lambda2.array());
    //std::cout << lambda.inverse() << '\n';
    //std::cout << '\n';
    //std::cout << unsigned_sqrt(lambda2.array()) * unsigned_sqrt(lambda2.array()).inverse() << '\n';
    //std::cout << lambda * lambda.inverse() << '\n';
    //std::cout << lambda.inverse().matrix() << '\n';

    // Check a more elegant way of computing it =======================
    Matrix Test = Q_i * W_i * lambda.inverse().matrix().asDiagonal();
    Matrix V_i = Q_i * W_i;
    for (int j = 0; j < V_i.cols(); ++j)
    {
        V_i.col(j) /= lambda(j);
    }
    //std::cout << V_i << '\n';
    if (V_i.isApprox(Test))
        std::cout << "Similar" << '\n';

    Matrix W_i_inv_W0 = W_i.lu().solve(W0);
    Matrix V_i_inv_V0 = V_i.lu().solve(V0);
    Matrix A_i0 = W_i_inv_W0  + V_i_inv_V0;
    Matrix B_i0 = W_i_inv_W0 - V_i_inv_V0;
    Matrix X_i = (-source.k0 * device.t.at(layer) * lambda).exp().matrix().asDiagonal(); // Matrix Exponential, THIS IS A FULL MATRIX --> INEFFICIENT
    //std::cout << X_i << '\n';
    //std::cout << "Test5" << '\n';

    // Assemble S matrix
    // Lazy implementation, enable NRVO next
    auto A_i0_lu = A_i0.lu();
    auto partial_prod_lu = (A_i0 - X_i * B_i0 * A_i0_lu.solve(X_i) * B_i0).lu();
    Matrix S11 = partial_prod_lu.solve(X_i * B_i0 * A_i0_lu.solve(X_i) * A_i0 - B_i0);
    Matrix S12 = partial_prod_lu.solve(X_i) * (A_i0 - B_i0 * A_i0_lu.solve(B_i0));
    //std::cout << "Test6" << '\n';

    return ScatteringMatrix(S11, S12, S12, S11);
}

ScatteringMatrix SMatrixReflection(const RCWAParams& params, const Vector& Kx, const Vector& Ky, Vector& Kz_ref, const Matrix& W0, const Matrix& V0, Matrix& W_ref)
{
    /*
    Function that computes the Reflection side Connection S-Matrix
    */
    int block_size = Kz_ref.size();

    Vector Kx_Ky = Kx.array() * Ky.array(); // since Kx and Ky are both diagonal, Kx * Ky = Ky * Kx
    Vector top_right = params.ur_ref * params.er_ref - Kx.array().square(); ///////// CONJUGATE?
    Vector bottom_left = Ky.array().square() - params.ur_ref * params.er_ref; ////// CONJUGATE?

    Matrix Q_ref(2*block_size, 2*block_size);

    Q_ref.block(0, 0, block_size, block_size) = Kx_Ky.asDiagonal();
    Q_ref.block(0, block_size, block_size, block_size) = top_right.asDiagonal();
    Q_ref.block(block_size, 0, block_size, block_size) = bottom_left.asDiagonal();
    Q_ref.block(block_size, block_size, block_size, block_size) = (- Kx_Ky).asDiagonal(); // see comment above
    //std::cout << Q <<'\n';
    Q_ref /= params.ur_ref;
    std::cout << "Q_ref " << double((Q_ref.array() != 0.0).count()) / Q_ref.size() << '\n';
    W_ref = Matrix::Identity(2*block_size, 2*block_size);

    Vector lambda_ref(2*block_size);
    lambda_ref << -1i * Kz_ref, -1i * Kz_ref;

    Matrix V_ref(2*block_size, 2*block_size);

    Matrix Test = Q_ref * lambda_ref.array().inverse().matrix().asDiagonal(); // check if .matrix is really needed
   
    // check if it's the best way to do this
    for (int j = 0; j < Q_ref.cols(); ++j)
    {
        V_ref.col(j) = Q_ref.col(j) / lambda_ref(j); // are we potentially dividing for 0? Yeah....
    }
    //std::cout << V_ref << '\n';
    if (V_ref.isApprox(Test))
        std::cout << "Similar" << '\n';

    Matrix W0_inv_W_ref = W0.lu().solve(W_ref);
    //std::cout << "W_ref " << W0_inv_W_ref.isApprox(W_ref) << '\n';
    Matrix V0_inv_V_ref = V0.lu().solve(V_ref);
    //std::cout << "V0_inv_V_ref " << V0_inv_V_ref << '\n';
    Matrix A_i1 = W0_inv_W_ref + V0_inv_V_ref;
    Matrix B_i1 = W0_inv_W_ref - V0_inv_V_ref;

    
    Matrix A_i1_inv_B_i1 = A_i1.lu().solve(B_i1);

    std::cout << '\n';
    std::cout << A_i1_inv_B_i1 << '\n';

    Matrix S11 = - A_i1_inv_B_i1;
    Matrix S12 = A_i1.lu().solve(2 * Matrix::Identity(A_i1.rows(), A_i1.cols()));
    Matrix S21 = 0.5 * (A_i1 - B_i1 * A_i1_inv_B_i1);
    Matrix S22 = (A_i1.transpose().lu().solve(B_i1.transpose())).transpose();

    return ScatteringMatrix(S11, S12, S21, S22);
}


ScatteringMatrix SMatrixTransmission(const RCWAParams& params, const Vector& Kx, const Vector& Ky, Vector& Kz_trn, const Matrix& W0, const Matrix& V0, Matrix& W_trn)
{
    /*
    Function that computes the Transmission side Connection S-Matrix.
    */
    int block_size = Kz_trn.size();

    Vector Kx_Ky = Kx.array() * Ky.array(); // since Kx and Ky are both diagonal, Kx * Ky = Ky * Kx
    Vector top_right = params.ur_trn * params.er_trn - Kx.array().square(); ///////// CONJUGATE?
    Vector bottom_left = Ky.array().square() - params.ur_trn * params.er_trn; ///////// CONJUGATE?

    Matrix Q_trn(2*block_size, 2*block_size);

    Q_trn.block(0, 0, block_size, block_size) = Kx_Ky.asDiagonal();
    Q_trn.block(0, block_size, block_size, block_size) = top_right.asDiagonal();
    Q_trn.block(block_size, 0, block_size, block_size) = bottom_left.asDiagonal();
    Q_trn.block(block_size, block_size, block_size, block_size) = (- Kx_Ky).asDiagonal(); // see comment above
    //std::cout << Q <<'\n';
    Q_trn /= params.ur_trn;
    std::cout << "Q_trn " << double((Q_trn.array() != 0.0).count()) / Q_trn.size() << '\n';
    W_trn = Matrix::Identity(2*block_size, 2*block_size);

    Vector lambda_trn(2*block_size);
    lambda_trn << 1i * Kz_trn, 1i * Kz_trn;

    Matrix V_trn(2*block_size, 2*block_size);

    Matrix Test = Q_trn * lambda_trn.array().inverse().matrix().asDiagonal(); // check if .matrix is really needed
   
    // check if it's the best way to do this
    for (int j = 0; j < Q_trn.cols(); ++j)
    {
        V_trn.col(j) = Q_trn.col(j) / lambda_trn(j); // are we potentially dividing for 0? Yeah....
    }
    //std::cout << V0 << '\n';
    if (V_trn.isApprox(Test))
        std::cout << "Similar" << '\n';
    //std::cout << V_trn << '\n';

    Matrix W0_inv_W_trn = W0.lu().solve(W_trn);
    Matrix V0_inv_V_trn = V0.lu().solve(V_trn);
    Matrix A_i2 = W0_inv_W_trn + V0_inv_V_trn;
    Matrix B_i2 = W0_inv_W_trn - V0_inv_V_trn;

    Matrix A_i2_inv_B_i2 = A_i2.lu().solve(B_i2);

    Matrix S11 = (A_i2.transpose().lu().solve(B_i2.transpose())).transpose();
    Matrix S12 = 0.5 * (A_i2 - B_i2 * A_i2_inv_B_i2);
    Matrix S21 = A_i2.lu().solve(2 * Matrix::Identity(A_i2.rows(), A_i2.cols()));
    Matrix S22 = - A_i2_inv_B_i2;

    return ScatteringMatrix(S11, S12, S21, S22);
}


void ComputeSourceModeCoeff(const Source& source, const RCWAParams& params, const std::vector<Complex>& k_inc, const Matrix& W_ref, Vector& csrc)
{
    /*
    Function that computes the Source parameters and the mode coefficents of the source, csrc
    */

    // Compute directions of TE and TM polarization
    // a_TE = (n x k_inc) / |n x k_inc|,
    // a_TM = (k_inc x a_TE) / |k_inc x a_TE|,
    // where n = a_z = (0, 0, 1).
    // In general, a x b = (a_1*b_2 - a_2*b_1, a_2*b0 - a_0*b_2, a_0*b_1 - a_1*b_0) , with z = (z_0, z_1, z_2).
    // They are 3D, perhaps better to use fixed dimension declaration and allocation
    Vec_3d a_TE{}; // 0 initialization via brace initialization

    if (std::abs(source.theta) < 1e-6)
        a_TE << 0.0 + 0.0i, 1.0 + 0.0i, 0.0 + 0.0i;
    else    
        a_TE << -k_inc.at(1), k_inc.at(0), 0.0 + 0.0i;
    a_TE.normalize();
    std::cout << "a_TE" << '\n';
    std::cout << a_TE << '\n';
    Vec_3d a_TM{};
    a_TM << k_inc.at(1) * a_TE(2) - k_inc.at(2) * a_TE(1), k_inc.at(2) * a_TE(0) - k_inc.at(0) * a_TE(2), k_inc.at(0) * a_TE(1) - k_inc.at(1) * a_TE(0);
    a_TM.normalize();

    // Compute Polarization Vector
    Vec_3d Pol = source.pte * a_TE + source.ptm * a_TM;
    std::cout << "POLARIZATION VECTOR" <<'\n';
    std::cout << Pol <<'\n';
    Pol.normalize(); // Ensure normalization of the vector
    std::cout << '\n';
    std::cout << "POLARIZATION VECTOR 2" <<'\n';
    std::cout << Pol <<'\n';
    // Construct delta vector
    int P = 2 * params.Nx_harmonics + 1;
    int Q = 2 * params.Ny_harmonics + 1;
    int PQ = P * Q;
    //Vector delta_0 = Vector::Zero(PQ);
    //delta_0(PQ/2) = Complex(1.0, 0.0);

    //std::cout << delta_0 << '\n';

    // Build directly Source Field esrc
    Vector esrc = Vector::Zero(2*PQ);
    esrc(PQ/2) = Pol(0); // x component of electric field polarization vector
    esrc(PQ/2 + PQ) = Pol(1); // y component of electric field polarization vector
    std::cout << '\n';
    std::cout << esrc << '\n';
    std::cout << '\n';

    // mode coefficients of the source
    csrc = W_ref.lu().solve(esrc);

    std::cout << csrc << '\n';
}


void ComputeReflectedField(const RCWAParams& params, const ScatteringMatrix& S_global, const Vector& csrc, const Vector& Kx, const Vector& Ky, const Vector& Kz_ref, const Matrix& W_ref, Vector& r)
{
    /*
    Function that computes the reflected field r.
    */
    int P = 2 * params.Nx_harmonics + 1;
    int Q = 2 * params.Ny_harmonics + 1;
    int PQ = P * Q;

    r(Eigen::seqN(0, 2*PQ)) = W_ref * S_global.S11 * csrc;
    std::cout << "Size of r " << r.size() << '\n';
    std::cout << "Size of Kx " << Kx.rows() << " " << Kx.cols() << '\n';
    std::cout << "Size of Ky " << Ky.rows() << " " << Ky.cols() << '\n';
    std::cout << "Size of Kz_ref " << Kz_ref.size() << '\n';

    r(Eigen::seqN(2*PQ, PQ)) = -(Kx.array() * (r(Eigen::seqN(0, PQ))).array() + Ky.array() * (r(Eigen::seqN(PQ, PQ))).array()).array() / (Kz_ref.array());
}

void ComputeTransmittedField(const RCWAParams& params, const ScatteringMatrix& S_global, const Vector& csrc, const Vector& Kx, const Vector& Ky, const Vector& Kz_trn, const Matrix& W_trn, Vector& t)
{
    /*
    Function that computes the transmitted field t.
    */
    int P = 2 * params.Nx_harmonics + 1;
    int Q = 2 * params.Ny_harmonics + 1;
    int PQ = P * Q;

    t(Eigen::seqN(0, 2*PQ)) = W_trn * S_global.S21 * csrc;
    /*
    std::cout << "Size of t " << t.size() << '\n';
    std::cout << "Size of Kx " << Kx.rows() << " " << Kx.cols() << '\n';
    std::cout << "Size of Ky " << Ky.rows() << " " << Ky.cols() << '\n';
    std::cout << "Size of Kz_ref " << Kz_trn.size() << '\n';
    */
    t(Eigen::seqN(2*PQ, PQ)) = -(Kx.array() * (t(Eigen::seqN(0, PQ))).array() + Ky.array() * (t(Eigen::seqN(PQ, PQ))).array()).array() / (Kz_trn.array());
}


Results ComputeDiffractionEfficiencies(const RCWAParams& params, const Vector& r, const Vector& t, const std::vector<Complex>& k_inc, const Vector& Kz_ref, const Vector& Kz_trn)
{
    /*
    Function that computes the diffraction efficiencies for reflected and transmitted fields.
    */
    int P = 2 * params.Nx_harmonics + 1;
    int Q = 2 * params.Ny_harmonics + 1;
    int PQ = P * Q;
    std::cout << "ComputeDiffractionEfficiencies" <<'\n';
    // Reflected power
    Real_Vector r_ref_modes = r(Eigen::seqN(0, PQ)).cwiseAbs2() + r(Eigen::seqN(PQ, PQ)).cwiseAbs2() + r(Eigen::seqN(2*PQ, PQ)).cwiseAbs2();
    //std::cout << r_ref_modes <<'\n';
    Real_Matrix R = ((((- Kz_ref / params.ur_ref).real()) / (k_inc.at(2) / params.ur_ref).real()).array() * r_ref_modes.array()).reshaped(Q, P);
    Real R_tot = R.sum();
    // Transmitted power
    Real_Vector t_trn_modes = t(Eigen::seqN(0, PQ)).cwiseAbs2() + t(Eigen::seqN(PQ, PQ)).cwiseAbs2() + t(Eigen::seqN(2*PQ, PQ)).cwiseAbs2();
    //std::cout << '\n';
    //std::cout << t_trn_modes <<'\n';
    Real_Matrix T = ((((Kz_trn / params.ur_trn).real()) / (k_inc.at(2) / params.ur_ref).real()).array() * t_trn_modes.array()).reshaped(Q, P);
    Real T_tot = T.sum();
    return Results(R, T, R_tot, T_tot);
}   


ScatteringMatrix RedhefferProduct(const ScatteringMatrix& A, const ScatteringMatrix& B)
{
    /*
    Function that computes the Redheffer star product of two scattering matrices A and B.
    The Redheffer star product is defined as follows:
    S = A x B = [S11, S12; S21, S22] where
    S11 = A11 + A12 * (I - B11 * A22)^(-1) * B11 * A21
    S12 = A12 * (I - B11 * A22)^(-1) * B12
    S21 = B21 * (I - A22 * B11)^(-1) * A21
    S22 = B22 + B21 * (I - A22 * B11)^(-1) * A22 * B12,
    where Xnm = X.Snm, and I is the identity matrix of appropriate size.
    */

    // To be merged for efficiency
    Matrix I_B11_A22 = Matrix::Identity(B.S11.rows(), A.S22.cols()) - B.S11 * A.S22;
    Matrix I_A22_B11 = Matrix::Identity(A.S22.rows(), B.S11.cols()) - A.S22 * B.S11;

    auto lu_I_B11_A22 = I_B11_A22.lu();
    auto lu_I_A22_B11 = I_A22_B11.lu();

    Matrix S11 = A.S11 + A.S12 * lu_I_B11_A22.solve(B.S11) * A.S21;
    Matrix S12 = A.S12 * lu_I_B11_A22.solve(B.S12);
    Matrix S21 = B.S21 * lu_I_A22_B11.solve(A.S21);
    Matrix S22 = B.S22 + B.S21 * lu_I_A22_B11.solve(A.S22) * B.S12;
    
    return ScatteringMatrix(S11, S12, S21, S22);

}

EigenvalSolverResults extraction(Eigen::ComplexEigenSolver<Matrix>&& solver)
{
        // Return values are constructed directly in the caller's memory frame
    return EigenvalSolverResults{solver.eigenvalues(), solver.eigenvectors()};
}