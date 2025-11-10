# FLAM_Assignment - Research and Development / AI
Parameter estimation for a nonlinear model using L1 minimization. This repo includes Python code to fit unknown variables to provided data, utilizing multi-start local/global optimization and residual analysis, with step-by-step instructions and results.

# Project Overview: 
The goal is to fit unknown parameters $\theta$, $M$, and $X$ in a nonlinear model to match provided $(x,y)$ data points using advanced optimization methods.

# Mathematical Model: 
The model being fitted:
### X-coordinate:
$$
x(t) = t \cdot \cos(\theta) - e^{M \cdot |t|} \cdot \sin(0.3t) \cdot \sin(\theta) + X
$$
### Y-coordinate:
$$
y(t) = 42 + t \cdot \sin(\theta) + e^{M \cdot |t|} \cdot \sin(0.3t) \cdot \cos(\theta)
$$

# Optimization Objective: 
Minimize the L1 loss function:

$$
\mathcal{L}(\theta, M, X) = \sum_{i=1}^{n} |x_{\text{pred},i} - x_{\text{true},i}| + |y_{\text{pred},i} - y_{\text{true},i}|
$$

Subject to boundary conditions:
- $\theta \in [0, 50^\circ]$
- $M \in [-0.05, 0.05]$
- $X \in [0, 100]$

# Solution Approach

The parameter estimation follows a multi-stage optimization pipeline to overcome local minima and ensure robust convergence to the global optimum.

```mermaid
graph TD
    A[Raw Data Loading] --> B[Model Definition]
    B --> C[Multi-Start Local Optimization]
    C --> D[Dual Annealing Global Search]
    D --> E[Final Refinement]
    E --> F[Results & Visualization]
    F --> G[Residual Analysis]
