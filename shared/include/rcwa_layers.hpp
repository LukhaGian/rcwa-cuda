#ifdef __CUDACC__ // macro defined by nvcc during compilation of cuda code
#define RCWA_HD __host__ __device__
#else // standard cpp compiler
#define RCWA_HD // empty
#endif

#pragma once

//#include <cuda_runtime.h>
//#include <cstdio> // for default handling case in InsideShape
#include <vector>
#include <complex>
#include <cmath>
#include <algorithm>

using Real = double;
using Complex = std::complex<Real>;

enum ShapeType : int {SHAPE_UNIFORM, SHAPE_X, SHAPE_Y, 
                      SHAPE_TRIANGLE, SHAPE_TRIANGLE_SHIFTED, SHAPE_SQUARE};

// Generalized struct that describes the shape
struct ShapeDesc
{
    int type;
    double p0, p1, p2; // shape parameters, for example x_shift, y_shift
    double dx; // compute once Lx / Nx -> non need to recompute for each thread call
    double dy; // compute once Ly / Ny
    Complex value_inside;
    Complex value_outside;
};

RCWA_HD inline bool InsideShape(const ShapeDesc& s, int idx, int idy, int Nx, int Ny, double Lx, double Ly)
{
    /*
    Keep idx and idy as int, for simplicity --> can be moved to double if easier, loging a little bit of clarity to get some potential benefit in cuda
    */
    switch(s.type)
    {
        case SHAPE_UNIFORM:
            return true;
        case SHAPE_X:
            return (std::abs((idx + 0.5) * Lx/Nx - Lx/2.0) <= 0.25);
        case SHAPE_Y:
            return (std::abs((idy + 0.5) * Ly/Ny - Ly/2.0) <= 0.25);
        case SHAPE_TRIANGLE:
        {
            // s.p0 = w
            double x = (idx + 0.5) * Lx/Nx; // physical x coordinate
            double y = (idy + 0.5) * Ly/Ny;  // physical y coordinate
            double y_rel = y - (Ly/2.0 - s.p0/2.0);
            return ((y_rel >= 0) && (y_rel <= s.p0) && (std::abs(x - (Lx/2.0)) <= (y_rel/2.0)));
        }
        case SHAPE_TRIANGLE_SHIFTED:
        {
            // s.p0 = w
            // s.p1 = x_shift
            // s.p2 = y_shift
            double x = (idx + s.p1) * Lx/Nx; // physical x coordinate
            double y = (idy + s.p2) * Ly/Ny;  // physical y coordinate
            double y_rel = y - (Ly/2.0 - s.p0/2.0);
            return ((y_rel >= 0) && (y_rel <= s.p0) && (std::abs(x - (Lx/2.0)) <= (y_rel/2.0)));
        }
            case SHAPE_SQUARE:
            // s.p0 = half_size = std::min(Nx, Ny) / 4.0;
            return ((idy >= ((Ny/2.0) - s.p0)) && (idy < ((Ny/2.0) + s.p0)) && (idx >= ((Nx/2.0) - s.p0)) && (idx < ((Nx/2.0) + s.p0)));
        default:
            return false;
            // WIP: calling __trap() (device only funct) in __host__ __device__ function is not allowed
            //printf("CUDA Fatal Error: Invalid ShapeType (%d).\n", static_cast<int>(s.type));
            //__trap(); // aborts kernel exec
    }
}


namespace layer
{
    inline std::vector<Complex> uniform(const ShapeDesc& s, int Nx, int Ny, double Lx, double Ly)
    {
        // Function that creates a uniform 2D array of size Nx x Ny with a given complex value
        std::vector<Complex> out(Nx * Ny, Complex(-1.0, -1.0));
        for (int iy = 0; iy < Ny; ++iy)
        {
            for (int ix = 0; ix < Nx; ++ix)
                {
                    bool inside = InsideShape(s, ix, iy, Nx, Ny, Lx, Ly);
                    out[iy * Nx + ix] = inside ? s.value_inside : s.value_outside;
                }
        }
        return out;

    }

    inline std::vector<Complex> uniform_old(int Nx, int Ny, Complex value)
    {
        // Function that creates a uniform 2D array of size Nx x Ny with a given complex value
        return std::vector<Complex>(Nx * Ny, value);

    }
    


    inline std::vector<Complex> x_pattern(const ShapeDesc& s, int Nx, int Ny, double Lx, double Ly)
    {
        std::vector<Complex> out(Nx * Ny, Complex(-1.0, -1.0));
        for (int iy = 0; iy < Ny; ++iy)
        {
            for (int ix = 0; ix < Nx; ++ix)
                {
                    bool inside_square = InsideShape(s, ix, iy, Nx, Ny, Lx, Ly);
                    out[iy * Nx + ix] = inside_square ? s.value_inside : s.value_outside;
                }
        }
        return out;

    }


    inline std::vector<Complex> x_pattern_old(int Nx, int Ny, double Lx, double Ly, Complex inside_value, Complex outside_value)
    {
        std::vector<Complex> out(Nx * Ny, Complex(0.0, 0.0));
        for (int iy = 0; iy < Ny; ++iy)
        {
            for (int ix = 0; ix < Nx; ++ix)
                {
                    double x = (ix + 0.5) * Lx / Nx;  // pyhsical [0, Lx]
                    bool inside_square = (std::abs(x - Lx/2.0) <= 0.25);
                    out[iy * Nx + ix] = inside_square ? inside_value : outside_value;
                }
        }
        return out;

    }


    inline std::vector<Complex> y_pattern(const ShapeDesc& s, int Nx, int Ny, double Lx, double Ly)
    {
        std::vector<Complex> out(Nx * Ny, Complex(-1.0, -1.0));
        for (int iy = 0; iy < Ny; ++iy)
        {
            for (int ix = 0; ix < Nx; ++ix)
                {
                    bool inside_square = InsideShape(s, ix, iy, Nx, Ny, Lx, Ly);
                    out[iy * Nx + ix] = inside_square ? s.value_inside : s.value_outside;
                }
        }
        return out;

    }

    inline std::vector<Complex> y_pattern_old(int Nx, int Ny, double Lx, double Ly, Complex inside_value, Complex outside_value)
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


    inline std::vector<Complex> triangle_pattern(const ShapeDesc& s, int Nx, int Ny, double Lx, double Ly)
    {
        std::vector<Complex> out(Nx * Ny, Complex(-1.0, -1.0));
        for (int iy = 0; iy < Ny; ++iy)
        {
            //double y = (iy + 0.5) * Ly / Ny;  // physical y coordinate !!!!
            for (int ix = 0; ix < Nx; ++ix)
                {
                    //double x = (ix + 0.5) * Lx / Nx;  // physical x coordinate !!!!!
                    // Triangle centred at (Lx/2, Ly/2) with base w = 0.7*Ly
                    //double w = s.p0; //w = 0.70 * Ly;
                    //double xc = s.p1; //xc = Lx / 2.0;
                    //double yc = Ly / 2.0; //yc = Ly / 2.0;
                    // Triangle vertices (centred)
                    //double y_rel = y - (yc - w/2.0);  // relative to triangle base !!!!!!!!
                    //double half_base = (y_rel/2.0);  // apex at top, base at bottom !!!!!!            
                    bool inside_triangle = InsideShape(s, ix, iy, Nx, Ny, Lx, Ly);
                    out[iy * Nx + ix] = inside_triangle ? s.value_inside : s.value_outside;
                }
        }
        return out;

    }

    
    inline std::vector<Complex> triangle_pattern_old(int Nx, int Ny, double Lx, double Ly, Complex inside_value, Complex outside_value)
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


    inline std::vector<Complex> triangle_pattern_shifted(const ShapeDesc& s, int Nx, int Ny, double Lx, double Ly)
    {
        std::vector<Complex> out(Nx * Ny, Complex(-1.0, -1.0));
        double x_shift = s.p1;
        double y_shift = s.p2;
        for (int iy = 0; iy < Ny; ++iy)
        {
            double y = (iy + y_shift) * Ly / Ny;  // physical y coordinate
            for (int ix = 0; ix < Nx; ++ix)
                {
                    double x = (ix + x_shift) * Lx / Nx;  // physical x coordinate
                    // Triangle centred at (Lx/2, Ly/2) with base w = 0.7*Ly
                    double w = s.p0;          //w = 0.70 * Ly;
                    //double xc = Lx / 2.0;
                    double yc = Ly / 2.0;
                    // Triangle vertices (centred)
                    double y_rel = y - (yc - w/2.0);  // relative to triangle base
                    //double half_base = (y_rel/2.0);  // apex at top, base at bottom                
                    bool inside_triangle = InsideShape(s, ix, iy, Nx, Ny, Lx, Ly);
                    out[iy * Nx + ix] = inside_triangle ? s.value_inside : s.value_outside;
                }
        }
        return out;

    }

    inline std::vector<Complex> triangle_pattern_shifted_old(int Nx, int Ny, double Lx, double Ly, double x_shift, double y_shift, Complex inside_value, Complex outside_value)
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


    inline std::vector<Complex> square_pattern(const ShapeDesc& s, int Nx, int Ny, double Lx, double Ly) 
    {
    
        // Allocate the single 1D vector representing the entire Ny x Nx grid
        std::vector<Complex> out(Nx * Ny, Complex(-1.0, -1.0));

        // Define center and half-width of the square
        //int cy = Ny / 2;
        //int cx = Nx / 2;
        //int half_size = std::min(Nx, Ny) / 4; // Controls the size of the square

        // Calculate boundary indices
        //int y_start = cy - half_size; // Ny/2 - s.p0 
        //int y_end   = cy + half_size;
        //int x_start = cx - half_size;
        //int x_end   = cx + half_size;

        // Fill the square values within boundaries
        for (int iy = 0; iy < Ny; ++iy) 
        {
            for (int ix = 0; ix < Nx; ++ix) 
            {
                bool inside_square = InsideShape(s, ix, iy, Nx, Ny, Lx, Ly);
                out[iy * Nx + ix] = inside_square ? s.value_inside : s.value_outside;
            }
        }
        return out;

    }

        inline std::vector<Complex> square_pattern_old(int Nx, int Ny, Complex inside_value, Complex outside_value) 
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