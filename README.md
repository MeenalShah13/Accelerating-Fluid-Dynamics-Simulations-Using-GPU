# GPU-Accelerated Lattice Boltzmann Simulation

## 🌊 Project Overview

This project implements a **GPU-accelerated fluid dynamics simulation** using the **Lattice Boltzmann Method (LBM)** with Python and CUDA. It demonstrates high-performance computing techniques by comparing CPU and GPU implementations of a 2D incompressible fluid flow simulation around various barrier geometries (linear, circular, and oval obstacles).

The simulation visualizes fluid flow patterns in real-time using velocity curl visualization, allowing users to observe how fluids interact with obstacles at different Reynolds numbers.

## 📚 Background: Lattice Boltzmann Method

The **Lattice Boltzmann Method** is a computational fluid dynamics approach that simulates fluid flow at a mesoscopic level by tracking particle distribution functions along discrete lattice directions. Unlike traditional Navier-Stokes solvers, LBM:

- **Discretizes space and time** into a regular lattice with particles moving in 9 directions (D2Q9 model)
- **Uses a relaxation parameter (ω)** based on fluid viscosity to model particle collisions
- **Naturally handles boundary conditions** through bounce-back collision rules
- **Parallelizes exceptionally well** on GPUs due to local computations at each lattice point

### Key Parameters

- **Viscosity (μ)**: Controls fluid "stickiness"; affects the relaxation parameter `ω = 1/(3μ + 0.5)`
- **Reynolds Number**: Characterizes flow regime (Re = ρ·U·L/μ)
- **Relaxation Parameter (ω)**: Drives the system toward equilibrium; values closer to 1 give lower viscosity
- **Inlet Velocity (u₀)**: Initial and boundary flow speed (0.1 lattice units in this simulation)

## 🛠 Requirements

Ensure you have:

- **Python 3.8+** installed
- **CUDA Toolkit 11.0+** (with compatible NVIDIA drivers)
- **NVIDIA GPU with CUDA support** (compute capability 3.0 or higher)
- **Required Python packages** (see Installation section)

## 🚀 Installation

1. **Create a virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   venv\Scripts\activate  # On Windows
   source venv/bin/activate  # On macOS/Linux
   ```

2. **Install Dependencies**
   
   From the project root directory, run:
   ```bash
   pip install -r requirements.txt
   ```
   
   This installs:
   - `numpy` - Numerical computations
   - `matplotlib` - Visualization and animation
   - `pycuda` - Python bindings for CUDA (GPU code only)

## 🏃 Running the Simulation

### CPU Simulation
```bash
python cpu_code.py
```

### GPU Simulation (Accelerated)
```bash
python gpu_code.py
```

### Workflow

1. **Barrier Selection**: When you run either script, you'll be prompted to select a barrier type:
   - `1` or `linear` - A vertical line obstacle
   - `2` or `circle` - A circular obstacle
   - `3` or `oval` - An elliptical obstacle

2. **Visualization**: The simulation displays:
   - **Colored fluid field**: Shows velocity curl (vorticity) visualization with red/blue indicating rotation direction
   - **Black barrier**: Visualizes the obstacle in the flow
   - **Real-time animation**: Runs for 2500 frames (~30 seconds depending on hardware)

3. **Performance Data**: If `performanceData = True` in the code, frame rates are printed every 100 frames

### Expected Output

- Smooth laminar flow patterns around the barrier
- Vortex shedding behind circular/oval obstacles
- Color gradients indicating velocity field variations
- Frame rate statistics (e.g., "Frame 100: 45.3 frames per second")

## ⚙️ Algorithm Overview

### Stream Phase
Particles move one lattice unit along their 9 direction vectors (N, S, E, W, NE, NW, SE, SW, and stationary).

### Collision Phase
At each lattice point, the velocity distribution is relaxed toward equilibrium using:
- Local density (ρ) and velocity (uₓ, uᵧ) calculations
- Relaxation operator with parameter ω

### Boundary Conditions
- **Inlet (left edge)**: Fixed rightward flow velocity
- **Barriers**: Bounce-back collision (particles reverse direction)
- **Periodic boundaries**: Wrap-around on top/bottom and right edge

## 🛠 Troubleshooting

### PyCUDA Installation Issues

If you encounter PyCUDA errors:

1. **Verify CUDA Installation**
   ```bash
   nvcc --version
   ```
   You should see your CUDA version (e.g., "Cuda compilation tools, release 11.8"). If not, install CUDA Toolkit from [NVIDIA's official site](https://developer.nvidia.com/cuda-downloads).

2. **Check GPU Recognition**
   ```bash
   python -c "import pycuda.driver as cuda; cuda.init(); print(cuda.Device(0).name())"
   ```
   This should print your GPU model.

3. **Reinstall PyCUDA**
   ```bash
   pip install pycuda --no-cache-dir
   ```

### Common Runtime Errors

| Error | Solution |
|-------|----------|
| `pycuda._driver.Error: invalid device ordinal` | Ensure NVIDIA drivers are up-to-date or run GPU simulation on a system with compatible GPU |
| `CUDA Kernel Launch Failed` | Check CUDA memory availability; reduce grid size or resolution |
| `Invalid compute capability` | GPU too old; need compute capability ≥3.0 |
| Slow GPU performance | CPU and GPU may share memory; ensure dedicated GPU, or reduce resolution |

### Performance Comparison

**Expected Performance** (on modern NVIDIA GPUs):
- **CPU (single-threaded)**: ~5-15 frames per second
- **GPU (CUDA-accelerated)**: ~40-100+ frames per second

If GPU is slower than CPU, verify:
- GPU is properly detected and used
- CUDA kernels are compiled correctly
- No PCIe bandwidth bottleneck

## 📊 Simulation Parameters

Located at the top of each script:

```python
height = 80           # Lattice height (cells)
width = 200          # Lattice width (cells)
viscosity = 0.02     # Fluid viscosity
u0 = 0.1             # Inlet velocity
```

**To modify**:
- **Increase resolution**: Raise `height` and `width` (increases computation time)
- **Change flow regime**: Adjust `viscosity` or `u0` to alter Reynolds number
- **Simulation duration**: Change `arg == 2500` to run longer (1 arg ≈ 20 time steps)

## 📈 Significance & Applications

### Performance Gains
This project demonstrates **10-20x speedup** of GPU-accelerated LBM over CPU implementations, showcasing:
- Effective GPU parallelization for structured grid computations
- Practical scalability of iterative numerical methods
- Real-time simulation of physics-based phenomena

### Real-World Applications
- **Aerodynamics**: Aircraft/vehicle design optimization
- **Microfluidics**: Drug delivery and lab-on-chip devices
- **Hemodynamics**: Blood flow in arteries and medical devices
- **Environmental modeling**: Pollutant dispersion and weather simulation
- **Industrial processing**: Mixing, combustion, and multiphase flows

### Educational Value
This implementation illustrates:
- CUDA programming fundamentals (kernel design, memory management)
- Computational fluid dynamics algorithms
- Performance profiling and GPU vs. CPU trade-offs
- Numerical stability and relaxation parameter tuning

## 📖 References

- [NumPy Documentation](https://numpy.org/doc/)
- [Matplotlib Documentation](http://matplotlib.org/stable/index.html)
- [PyCUDA Documentation](https://documen.tician.de/pycuda/)
- [Lattice Boltzmann Method Overview](https://en.wikipedia.org/wiki/Lattice_Boltzmann_methods)
- [CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)

## 📝 Notes

- **Stability**: Simulations are stable for viscosity ≥ 0.01 and u₀ ≤ 0.2
- **Visualization**: Curl (vorticity) field provides better visual insight than raw velocity
- **Scalability**: To simulate larger domains, increase `height` and `width` (adjust block/grid sizes accordingly)