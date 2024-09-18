# PID Tuning with Genetic Algorithm

## Overview

This project implements a PID (Proportional-Integral-Derivative) controller tuning using a Genetic Algorithm. The goal is to find optimal PID parameters to minimize the Integral of Absolute Error (IAE) by simulating the controller's performance on a given dataset.

The PID controller is widely used in various control systems for its simplicity and effectiveness. The Genetic Algorithm (GA) is employed here to optimize the PID parameters (P, I, and D) by exploring different combinations and evolving over generations to improve performance.

## Features

- **PID Simulation**: Simulates PID controller performance using CUDA for fast computation.
- **Genetic Algorithm**: Uses the DEAP library to evolve and optimize PID parameters.
- **Ziegler-Nichols Tuning**: Provides initial PID parameter estimates based on the Ziegler-Nichols method.

## Prerequisites

To run this code, you need to have the following Python packages installed:

- `numpy`
- `pandas`
- `deap`
- `numba`

You can install the required packages using pip:

```bash
pip install numpy pandas deap numba

