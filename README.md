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

Where:
- $\theta$: Angle parameter (radians)
- $M$: Exponential growth factor
- $X$: Horizontal offset
- $t$: Time parameter
