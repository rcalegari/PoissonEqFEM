# 2D Finite Element Method Implementation

This repository contains my implementation of the **2D Finite Element Method (FEM)** for solving Poisson-type problems, as part of the WI4205 course *Applied Finite Elements* at *TU Delft*.

The project builds on the 1D FEM implementation and extends it to 2D problems on curvilinear quadrilateral meshes, following the weak Galerkin formulation.



##  Problem Statement

We solve the weak form of the Poisson equation:

<p align="center">
  <!-- Dark mode (white text) -->
  <img src="https://latex.codecogs.com/svg.latex?\color{White}\displaystyle%20-%5Cnabla%5Ccdot%28%5Ckappa%5Cnabla%20u%29%20%3D%20f%2C%20%5Cquad%20u%7C_%7B%5CGamma_D%7D%3D0%2C%20%5Cquad%20%5Cnabla%20u%5Ccdot%20n%7C_%7B%5CGamma_N%7D%3Dg#gh-dark-mode-only" alt="Poisson equation (dark mode)">
  
  <!-- Light mode (black text) -->
  <img src="https://latex.codecogs.com/svg.latex?\displaystyle%20-%5Cnabla%5Ccdot%28%5Ckappa%5Cnabla%20u%29%20%3D%20f%2C%20%5Cquad%20u%7C_%7B%5CGamma_D%7D%3D0%2C%20%5Cquad%20%5Cnabla%20u%5Ccdot%20n%7C_%7B%5CGamma_N%7D%3Dg#gh-light-mode-only" alt="Poisson equation (light mode)">
</p>



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

