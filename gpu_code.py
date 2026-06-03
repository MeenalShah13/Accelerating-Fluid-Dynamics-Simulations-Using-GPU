import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import math, time

# Define constants:
height = np.int32(80)  # lattice dimensions
width = np.int32(200)
viscosity = 0.02  # fluid viscosity
omega = 1 / (3 * viscosity + 0.5)  # "relaxation" parameter
u0 = 0.1  # initial and in-flow speed
four9ths = 4.0 / 9.0  # abbreviations for lattice-Boltzmann weight factors
one9th = 1.0 / 9.0
one36th = 1.0 / 36.0
performanceData = True  # set to True if performance data is desired

# Initialize all the arrays to steady rightward flow:
n0 = four9ths * (np.ones((height,width), dtype=np.float32) - 1.5*u0**2)
nN = one9th * (np.ones((height,width), dtype=np.float32) - 1.5*u0**2)
nS = one9th * (np.ones((height,width), dtype=np.float32) - 1.5*u0**2)
nE = one9th * (np.ones((height, width), dtype=np.float32) + 3 * u0 + 4.5 * u0**2 - 1.5 * u0**2)
nW = one9th * (np.ones((height, width), dtype=np.float32) + 3 * u0 + 4.5 * u0**2 - 1.5 * u0**2)
nNE = one36th * (np.ones((height,width), dtype=np.float32) + 3*u0 + 4.5*u0**2 - 1.5*u0**2)
nSE = one36th * (np.ones((height,width), dtype=np.float32) + 3*u0 + 4.5*u0**2 - 1.5*u0**2)
nNW = one36th * (np.ones((height,width), dtype=np.float32) - 3*u0 + 4.5*u0**2 - 1.5*u0**2)
nSW = one36th * (np.ones((height,width), dtype=np.float32) - 3*u0 + 4.5*u0**2 - 1.5*u0**2)
rho = n0 + nN + nS + nE + nW + nNE + nSE + nNW + nSW		# macroscopic density
ux = (nE + nNE + nSE - nW - nNW - nSW) / rho				# macroscopic x velocity
uy = (nN + nNE + nNW - nS - nSE - nSW) / rho				# macroscopic y velocity

# Initialize barriers:
barrier = np.zeros((height, width), dtype=np.bool_)  # True wherever there's a barrier

barrier_type = input("Enter the name of the barrier type or the corresponding number:\n1 - Linear\n2 - Circle\n3 - Oval\nType your answer:")

if barrier_type == "1" or barrier_type.lower() == "linear":
    # Simple linear barrier
    for y in range(int(height / 2) - 8, int(height / 2) + 9):
        x = round(height / 3)
        barrier[y, x] = True
elif barrier_type == "2" or barrier_type.lower() == "circle":
    # Simple circle barrier
    center_x = round(height / 3)
    center_y = height / 2
    radius = 10
    for y in range(height):
        for x in range(width):
            if (x - center_x)**2 + (y - center_y)**2 <= radius**2:
                barrier[y, x] = True
else:
    # Simple oval barrier
    center_x = round(width / 3)
    center_y = height / 2
    horizontal_radius = 20
    vertical_radius = 5
    for y in range(height):
        for x in range(width):
            if ((x - center_x) / horizontal_radius)**2 + ((y - center_y) / vertical_radius)**2 <= 1:
                barrier[y, x] = True

# Allocate memory on the GPU
n0_gpu = cuda.mem_alloc(n0.nbytes)
nN_gpu = cuda.mem_alloc(nN.nbytes)
nS_gpu = cuda.mem_alloc(nS.nbytes)
nE_gpu = cuda.mem_alloc(nE.nbytes)
nW_gpu = cuda.mem_alloc(nW.nbytes)
nNE_gpu = cuda.mem_alloc(nNE.nbytes)
nSE_gpu = cuda.mem_alloc(nSE.nbytes)
nNW_gpu = cuda.mem_alloc(nNW.nbytes)
nSW_gpu = cuda.mem_alloc(nSW.nbytes)
barrier_gpu = cuda.mem_alloc(barrier.nbytes)
ux_gpu = cuda.mem_alloc(ux.nbytes)
uy_gpu = cuda.mem_alloc(uy.nbytes)

# Copy data to the GPU
cuda.memcpy_htod(n0_gpu, n0)
cuda.memcpy_htod(nN_gpu, nN)
cuda.memcpy_htod(nS_gpu, nS)
cuda.memcpy_htod(nE_gpu, nE)
cuda.memcpy_htod(nW_gpu, nW)
cuda.memcpy_htod(nNE_gpu, nNE)
cuda.memcpy_htod(nSE_gpu, nSE)
cuda.memcpy_htod(nNW_gpu, nNW)
cuda.memcpy_htod(nSW_gpu, nSW)
cuda.memcpy_htod(barrier_gpu, barrier)
cuda.memcpy_htod(ux_gpu, ux)
cuda.memcpy_htod(uy_gpu, uy)

# CUDA kernel code
cuda_code = """
__global__ void stream(int height, int width, 
                       float *nN, float *nS, float *nE, float *nW, 
                       float *nNE, float *nSE, float *nNW, float *nSW, 
                       bool *barrier) {
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    int idxM1 = (y - 1) * width + x;  // Move north
    int idxP1 = (y + 1) * width + x;  // Move south

    // Move all particles by one step along their directions of motion
    if (y > 0) {
        nN[idx] = nN[idxM1];
        nNE[idx] = nNE[idxM1];
        nNW[idx] = nNW[idxM1];
    }
    if (y < height - 1) {
        nS[idx] = nS[idxP1];
        nSE[idx] = nSE[idxP1];
        nSW[idx] = nSW[idxP1];
    }
    __syncthreads();
    if (x > 0) {
        nE[idx] = nE[idx - 1];
        nNE[idx] = nNE[idx - 1];
        nSE[idx] = nSE[idx - 1];
    }
    if (x < width - 1) {
        nW[idx] = nW[idx + 1];
        nNW[idx] = nNW[idx + 1];
        nSW[idx] = nSW[idx + 1];
    }
    __syncthreads();

    // Handle barrier collisions (bounce-back)
    if (barrier[idx]) {
        nN[idxP1]  = nS[idx];
        nS[idxM1]  = nN[idx];
        nE[idx + 1]  = nW[idx];
        nW[idx - 1]  = nE[idx];
        nNE[idxP1 + 1] = nSW[idx];
        nNW[idxP1 - 1] = nSE[idx];
        nSE[idxM1 + 1] = nNW[idx];
        nSW[idxM1 - 1] = nNE[idx];
    }
}

__global__ void collide(int height, int width, float *n0, float *nN, float *nS, float *nE, float *nW, float *nNE, float *nSE, float *nNW, float *nSW, float *ux, float *uy, float omega, float four9ths, float one9th, float one36th) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;

    // Compute macroscopic quantities
    float rho = n0[idx] + nN[idx] + nS[idx] + nE[idx] + nW[idx] + nNE[idx] + nSE[idx] + nNW[idx] + nSW[idx];
    if (rho < 1e-10f) rho = 1.0f;  // Avoid division by zero/negative rho
	ux[idx] = (nE[idx] + nNE[idx] + nSE[idx] - nW[idx] - nNW[idx] - nSW[idx]) / rho;
	uy[idx] = (nN[idx] + nNE[idx] + nNW[idx] - nS[idx] - nSE[idx] - nSW[idx]) / rho;
	float ux2 = ux[idx] * ux[idx];				// pre-compute terms used repeatedly...
	float uy2 = uy[idx] * uy[idx];
	float u2 = ux2 + uy2;
	float omu215 = 1 - 1.5*u2;			// "one minus u2 times 1.5"
	float uxuy = ux[idx] * uy[idx];

    n0[idx] = (1-omega)*n0[idx] + omega * four9ths * rho * omu215;
	nN[idx] = (1-omega)*nN[idx] + omega * one9th * rho * (omu215 + 3*uy[idx] + 4.5*uy2);
	nS[idx] = (1-omega)*nS[idx] + omega * one9th * rho * (omu215 - 3*uy[idx] + 4.5*uy2);
	nE[idx] = (1-omega)*nE[idx] + omega * one9th * rho * (omu215 + 3*ux[idx] + 4.5*ux2);
	nW[idx] = (1-omega)*nW[idx] + omega * one9th * rho * (omu215 - 3*ux[idx] + 4.5*ux2);
	nNE[idx] = (1-omega)*nNE[idx] + omega * one36th * rho * (omu215 + 3*(ux[idx]+uy[idx]) + 4.5*(u2+2*uxuy));
	nNW[idx] = (1-omega)*nNW[idx] + omega * one36th * rho * (omu215 + 3*(-ux[idx]+uy[idx]) + 4.5*(u2-2*uxuy));
	nSE[idx] = (1-omega)*nSE[idx] + omega * one36th * rho * (omu215 + 3*(ux[idx]-uy[idx]) + 4.5*(u2-2*uxuy));
	nSW[idx] = (1-omega)*nSW[idx] + omega * one36th * rho * (omu215 + 3*(-ux[idx]-uy[idx]) + 4.5*(u2+2*uxuy));
}

__global__ void newFluid(int height, int width, float *n0, float *nN, float *nS, float *nE, float *nW, float *nNE, float *nSE, float *nNW, float *nSW, float *ux, float *uy) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;

    if (x == 0) {
        nE[idx] = 0.14777777777777779f;
        nW[idx] = 0.14777777777777779f;
        nNE[idx] = 0.036944444444444446f;
        nSE[idx] = 0.036944444444444446f;
        nNW[idx] = 0.036944444444444446f;
        nSW[idx] = 0.036944444444444446f;
    }
}
"""

# Compile CUDA kernel
mod = SourceModule(cuda_code)
stream = mod.get_function("stream")
collide = mod.get_function("collide")
newFluid = mod.get_function("newFluid")

# Set up CUDA grid and block sizes
block_size = (16, 16, 1)  # 2D block
grid_size = (math.ceil(width / block_size[0]), math.ceil(height / block_size[1]), 1)

# Set up the figure and animation
fig, ax = plt.subplots()
fluidImage = ax.imshow(nE, origin='lower', norm=plt.Normalize(-.1,.1), cmap=plt.get_cmap('jet'), interpolation='none')
bImageArray = np.zeros((height, width, 4), dtype=np.uint8)  # RGBA image
bImageArray[barrier, 3] = 255  # Set alpha=255 only at barrier sites
barrierImage = ax.imshow(bImageArray, origin='lower', interpolation='none')

# Compute curl of the macroscopic velocity field:
def curl(ux, uy):
	return np.roll(uy,-1,axis=1) - np.roll(uy,1,axis=1) - np.roll(ux,-1,axis=0) + np.roll(ux,1,axis=0)

# Function called for each successive animation frame
start_time = time.time()
def nextFrame(arg):
    global start_time, animate
    if (performanceData and arg%100 == 0 and arg > 0):
        endTime = time.time()
        print("Frame", arg, ":", "%1.1f" % (100/(endTime-start_time)), 'frames per second')
        start_time = endTime
    
    if arg == 2500:
        animate.event_source.stop()
        animate = None
        plt.close()
        print("Animation stopped.")

    for i in range(20):
        try:
            stream(height, width, nN_gpu, nS_gpu, nE_gpu, nW_gpu, nNE_gpu, nSE_gpu, nNW_gpu, nSW_gpu, barrier_gpu, block=block_size, grid=grid_size)
            collide(height, width, n0_gpu, nN_gpu, nS_gpu, nE_gpu, nW_gpu, nNE_gpu, nSE_gpu, nNW_gpu, nSW_gpu, ux_gpu, uy_gpu, np.float32(omega), np.float32(four9ths), np.float32(one9th), np.float32(one36th), block=block_size, grid=grid_size)
            newFluid(height, width, n0_gpu, nN_gpu, nS_gpu, nE_gpu, nW_gpu, nNE_gpu, nSE_gpu, nNW_gpu, nSW_gpu, ux_gpu, uy_gpu, block=block_size, grid=grid_size)
        except Exception as e:
            print(f"CUDA Kernel Launch Failed: {e}")

    # Copy the result back to the host
    cuda.memcpy_dtoh(ux, ux_gpu)
    cuda.memcpy_dtoh(uy, uy_gpu)

    # print("ux:", ux)
    # print("uy:", uy)

    curl_data = curl(ux, uy)
    max_val = np.max(np.abs(curl_data))
    if max_val == 0:  # Avoid division by zero
        max_val = 0.1
    fluidImage.set_norm(plt.Normalize(-max_val, max_val))
    fluidImage.set_array(curl_data)
    return fluidImage, barrierImage

# Create the animation
animate = animation.FuncAnimation(fig, nextFrame, interval=1, blit=True)
plt.show()
print("Ended nicely.")