# 2D Finite Element Method Implementation

This repository contains my implementation of the **2D Finite Element Method (FEM)** for solving Poisson-type problems, as part of the WI4205 course *Applied Finite Elements* at *TU Delft*.

The project builds on the 1D FEM implementation and extends it to 2D problems on curvilinear quadrilateral meshes, following the weak Galerkin formulation.



##  Problem Statement

We solve the weak form of the Poisson equation:

\[
- \nabla \cdot \kappa \nabla u = f \quad \text{in } \Omega, \qquad
u = 0 \text{ on } \Gamma_D, \qquad
\nabla u \cdot n = g \text{ on } \Gamma_N
\]

using finite element spaces defined on mapped reference elements.



##  Project Structure

- `implem_2d.py` – main FEM implementation in 2D  
- `1d_fem_for_students.py` – supporting 1D FEM code  
- `data/` – contains geometry `.mat` files provided for the assignment:
  - `star3.mat`, `star4.mat`, `star5.mat`
  - `distressed_robotD.mat`
  - `distressed_robotDN.mat`



##  Dependencies

This project requires Python **3.10** with the following packages:

- `numpy==1.21.6`
- `scipy==1.7.3`
- `matplotlib==3.5.3`
- `pyqt5` (for interactive plotting, optional)

Older versions of NumPy/SciPy/Matplotlib are needed for compatibility with the given FEM data structures and basis evaluations.



## Setup

Using **conda** is recommended:

```bash
conda create -n fem python=3.10
conda activate fem

# install dependencies
pip install numpy==1.21.6 scipy==1.7.3 matplotlib==3.5.3 PyQt5

