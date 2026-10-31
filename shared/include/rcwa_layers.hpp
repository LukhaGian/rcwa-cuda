#pragma once

#include <vector>
#include <complex>
#include <cmath>
#include <algorithm>

using Real = double;
using Complex = std::complex<Real>;

namespace layer
{
    inline std::vector<Complex> uniform(int Nx, int Ny, Complex value)
    {
        // Function that creates a uniform 2D array of size Nx x Ny with a given complex value
        return std::vector<Complex>(Nx * Ny, value);

    }


    inline std::vector<Complex> x_pattern(int Nx, int Ny, double Lx, double Ly, Complex inside_value, Complex outside_value)
    {
        std::vector<Complex> out(Nx * Ny, Complex(0.0, 0.0));
        for (int iy = 0; iy < Ny; ++iy)
        {
        double y = (iy + 0.5) * Ly / Ny;  // pyhsical [0, Ly]
        for (int ix = 0; ix < Nx; ++ix)
            {
                double x = (ix + 0.5) * Lx / Nx;  // pyhsical [0, Lx]
                bool inside_square = (std::abs(x - Lx/2.0) <= 0.25);
                out[iy * Nx + ix] = inside_square ? inside_value : outside_value;
            }
        }
        return out;

    }


    inline std::vector<Complex> y_pattern(int Nx, int Ny, double Lx, double Ly, Complex inside_value, Complex outside_value)
    {
        std::vector<Complex> out(Nx * Ny, Complex(0.0, 0.0));
        for (int iy = 0; iy < Ny; ++iy)
        {
        double y = (iy + 0.5) * Ly / Ny;  // pyhsical [0, Ly]
        for (int ix = 0; ix < Nx; ++ix)
            {
                double x = (ix + 0.5) * Lx / Nx;  // pyhsical [0, Lx]
                bool inside_square = (std::abs(y - Ly/2.0) <= 0.25);
                out[iy * Nx + ix] = inside_square ? inside_value : outside_value;
            }
        }
        return out;

    }


    inline std::vector<Complex> triangle_pattern(int Nx, int Ny, double Lx, double Ly, Complex inside_value, Complex outside_value)
    {
        std::vector<Complex> out(Nx * Ny, Complex(0.0, 0.0));
        for (int iy = 0; iy < Ny; ++iy)
        {
            double y = (iy + 0.5) * Ly / Ny;  // physical y coordinate
            for (int ix = 0; ix < Nx; ++ix)
                {
                    double x = (ix + 0.5) * Lx / Nx;  // physical x coordinate
                    // Triangle centred at (Lx/2, Ly/2) with base w = 0.7*Ly
                    double w = 0.70 * Ly;
                    double xc = Lx / 2.0;
                    double yc = Ly / 2.0;
                    // Triangle vertices (centred)
                    double y_rel = y - (yc - w/2.0);  // relative to triangle base
                    double half_base = (w/2.0) * (y_rel/w);  // apex at top, base at bottom                
                    bool inside_triangle = (y_rel >= 0) && (y_rel <= w) && (std::abs(x - xc) <= half_base);
                    out[iy * Nx + ix] = inside_triangle ? inside_value : outside_value;
                }
        }
        return out;

    }


    inline std::vector<Complex> triangle_pattern_shifted(int Nx, int Ny, double Lx, double Ly, double x_shift, double y_shift, Complex inside_value, Complex outside_value)
    {
        std::vector<Complex> out(Nx * Ny, Complex(0.0, 0.0));
        for (int iy = 0; iy < Ny; ++iy)
        {
            double y = (iy + y_shift) * Ly / Ny;  // physical y coordinate
            for (int ix = 0; ix < Nx; ++ix)
                {
                    double x = (ix + x_shift) * Lx / Nx;  // physical x coordinate
                    // Triangle centred at (Lx/2, Ly/2) with base w = 0.7*Ly
                    double w = 0.70 * Ly;
                    double xc = Lx / 2.0;
                    double yc = Ly / 2.0;
                    // Triangle vertices (centred)
                    double y_rel = y - (yc - w/2.0);  // relative to triangle base
                    double half_base = (w/2.0) * (y_rel/w);  // apex at top, base at bottom                
                    bool inside_triangle = (y_rel >= 0) && (y_rel <= w) && (std::abs(x - xc) <= half_base);
                    out[iy * Nx + ix] = inside_triangle ? inside_value : outside_value;
                }
        }
        return out;

    }


    inline std::vector<Complex> square_pattern(int Nx, int Ny, Complex inside_value, Complex outside_value) 
    {
    
        // Allocate the single 1D vector representing the entire Ny x Nx grid
        std::vector<Complex> out(Ny * Nx, outside_value);

        // Define center and half-width of the square
        int cy = Ny / 2;
        int cx = Nx / 2;
        int half_size = std::min(Nx, Ny) / 4; // Controls the size of the square

        // Calculate boundary indices
        int y_start = cy - half_size;
        int y_end   = cy + half_size;
        int x_start = cx - half_size;
        int x_end   = cx + half_size;

        // Fill the square values within boundaries
        for (int y = y_start; y < y_end; ++y) {
            for (int x = x_start; x < x_end; ++x) {
                if (y >= 0 && y < Ny && x >= 0 && x < Nx) {
                    // Flattened 1D index mapping for (y, x) row-major order
                    out[y * Nx + x] = inside_value;
                }
            }
        }
        return out;

    }


    inline std::vector<Complex> stack(const std::vector<std::vector<Complex>>& layers)
    {
        size_t total = 0;
        for (const auto& layer : layers)
            total += layer.size();

        std::vector<Complex> out;
        out.reserve(total); // reserve the memory for the total size of all layers

        for (const auto& layer : layers)
            out.insert(out.end(), layer.begin(), layer.end());

        return out;

    }
}