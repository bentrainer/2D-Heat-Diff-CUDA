# 2D-Heat-Diff-CUDA

ELC 5319 MPP Final Project

## Problem Definition

$$
\frac{\partial T}{\partial t} = \alpha \left( \frac{\partial^2 T}{\partial x^2} + \frac{\partial^2 T}{\partial y^2} \right)
$$

In a discretized form, using explicit Euler's method:

$$
\frac{T_{n+1}(i,j) - T_{n}(i,j)}{\Delta t} = \alpha \left[
    \frac{T_{n}(i+1,j) - 2T_{n}(i,j) + T_{n}(i-1,j)}{\Delta x^2} +
    \frac{T_{n}(i,j+1) - 2T_{n}(i,j) + T_{n}(i,j-1)}{\Delta y^2}
\right]
$$

Tidy up to get

$$
\begin{align*}
T_{n+1}(i,j) = \left( 1 - \frac{2\Delta t}{\Delta x^2} - \frac{2\Delta t}{\Delta y^2} \right) &T_{n}(i,j) \\
+ \frac{\Delta t}{\Delta x^2} &T_{n}(i-1,j) \\
+ \frac{\Delta t}{\Delta x^2} &T_{n}(i+1,j) \\
+ \frac{\Delta t}{\Delta y^2} &T_{n}(i,j+1) \\
+ \frac{\Delta t}{\Delta y^2} &T_{n}(i,j-1)
\end{align*}
$$

### Parameters

- Width $L_x$, Height $L_y$

- Sample number in $x$, $y$ direction: $N_x$, $N_y$

- Mesh size $\Delta x = \frac{L_x}{N_x-1}$, $\Delta y = \frac{L_y}{N_y-1}$

- Time step $n$, step size $\Delta t$

- $\alpha=1.0$

## TODO

- [x] naive single-threaded CPU for-loop baseline

- [x] CUDA naive global-memory implementation

- [ ] CUDA shared-memory tiled implementation

- [ ] CUDA thread-coarsened implementation

- [ ] Nsight profile

### Optional comparison

- [ ] oneMKL sparse-matrix implementation