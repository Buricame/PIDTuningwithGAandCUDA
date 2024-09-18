# PID Tuning with Genetic Algorithm

## Overview

This project implements a PID (Proportional-Integral-Derivative) controller tuning using a Genetic Algorithm (GA). The goal is to find optimal PID parameters (P, I, and D) that minimize the Integral of Absolute Error (IAE). The controller's performance is simulated using real flight data extracted from a CSV file, which contains data collected from previous flights.

The Genetic Algorithm is employed to optimize the PID parameters by exploring different combinations and evolving over generations to improve the performance based on the given dataset. This approach is particularly effective when dealing with dynamic systems like UAVs, where optimal tuning can significantly improve control and stability.

## Features

- **PID Simulation**: Simulates PID controller performance using CUDA for fast computation.
- **Genetic Algorithm**: Uses the DEAP library to evolve and optimize PID parameters.
- **Ziegler-Nichols Tuning**: Provides initial PID parameter estimates based on the Ziegler-Nichols method.

## Limitations

Please note that the optimization results may not always be highly accurate. The performance of the PID controller is dependent on various factors, including the quality of the data and the effectiveness of the Genetic Algorithm parameters. In this project, flight data extracted from previous flights may contain noise and other irregularities, which could affect the optimization process. Despite preprocessing efforts, the presence of noisy or incomplete data can lead to suboptimal results.

As a result, the optimized PID parameters might not achieve the best possible performance in all scenarios, especially if the data used in training is not fully representative of the actual operating conditions.

## Prerequisites

To run this code, you need to have the following Python packages installed:

- `numpy`
- `pandas`
- `deap`
- `numba`

You can install the required packages using pip:

```bash
pip install numpy pandas deap numba
