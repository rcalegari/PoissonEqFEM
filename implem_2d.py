#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 11:46:13 2024

@author: rosacalegari
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from numpy.polynomial.legendre import leggauss as gaussquad
from scipy.interpolate import _bspl as bspl
import matplotlib.pyplot as plt
import scipy.io as sio
import matplotlib.tri as mtri
from matplotlib.cm import ScalarMappable
from tqdm import tqdm

def main():
    case = 'H' # 'H' or 'I'
    # if case == 'triangle':
    #     fn = '/home/rosa/Desktop/Master/S2/FEM/assignments/ass4/data/star3.mat'
    #     problem_B = problem_B_H
    #     problem_L = problem_L_H
    # elif case == 'square':
    #     fn = '/home/rosa/Desktop/Master/S2/FEM/assignments/ass4/data/star.mat'
    #     problem_B = problem_B_H
    #     problem_L = problem_L_H

    if case == 'H':
        fn = '/home/rosa/Desktop/Master/S2/FEM/assignments/ass4/data/distressed_robotD.mat'
        problem_B = problem_B_H
        problem_L = problem_L_H
    elif case == 'I':
        fn = '/home/rosa/Desktop/Master/S2/FEM/assignments/ass4/data/distressed_robotDN.mat'
        problem_B = problem_B_I
        problem_L = problem_L_I
    
    neval = 20 # number of evaluation points per direction

    # read file
    ref_data, fe_geometry, fe_space = read_file(fn, neval)
    geom_map = create_geometric_map(fe_geometry, ref_data)

    # plot domain and basis function
    '''
    ind is used to say which basis function to plot
    if ind is True, doesn't plot any basis function, just the domain
    if ind is 'all', plots all basis functions
    if ind is an integer, plots the basis function with that index
    '''
    # plot_domain_and_basis_function(fn, True, neval, ref_data, fe_geometry, fe_space)
    
    # assemble fe problem
    A, b = assemble_fe_problem_2d_newnew(fe_geometry, fe_space ,ref_data, geom_map, problem_B, problem_L)
    
    # exclude boundary bases and find coefficients
    u_tilde = reduced_u(fe_space, A, b, save=case)
    
    # plot solution
    plot_solution_2d_new(ref_data, fe_geometry, fe_space, case, u_tilde)



def create_ref_data(neval, deg, integrate=False):
    reference_element = np.array([0, 1])
    if integrate is False:
        x, y = np.meshgrid(np.linspace(0, 1, neval), np.linspace(0, 1, neval))
        vertices = np.hstack([x.reshape(-1, 1), y.reshape(-1, 1)])
        evaluation_points = vertices
        quadrature_weights = np.zeros((0,))
    else:
        x, w = gaussquad(neval)
        evaluation_points = np.zeros((neval * neval, 2))
        for i in range(neval):
            for j in range(neval):
                evaluation_points[i*neval + j] = np.array([0.5*(x[i] + 1), 0.5*(x[j] + 1)])
        quadrature_weights = np.zeros((neval*neval,))
        for i in range(neval):
            for j in range(neval):
                quadrature_weights[i*neval + j] = 0.25*w[i]*w[j]

    knt1 = np.concatenate((np.zeros((deg[0]+1,),dtype=float),np.ones((deg[0]+1,),dtype=float)),axis=0)
    knt2 = np.concatenate((np.zeros((deg[1]+1,),dtype=float),np.ones((deg[1]+1,),dtype=float)),axis=0)

    tmp1 = [bspl.evaluate_all_bspl(knt1, deg[0], evaluation_points[i][0], deg[0], nu=0)
           for i in range(evaluation_points.shape[0])]
    reference_basis1 = np.vstack(tmp1).T
    tmp2 = [bspl.evaluate_all_bspl(knt2, deg[1], evaluation_points[i][1], deg[1], nu=0)
           for i in range(evaluation_points.shape[0])]
    reference_basis2 = np.vstack(tmp2).T

    reference_basis = np.zeros(((deg[0]+1)*(deg[1]+1), evaluation_points.shape[0]))
    for j1 in range(deg[0]+1):
        for j2 in range(deg[1]+1):
            reference_basis[j1 + j2*(deg[0]+1)] = np.multiply(reference_basis1[j1], reference_basis2[j2])

    tmp3 = [bspl.evaluate_all_bspl(knt1, deg[0], evaluation_points[i][0], deg[0], nu=1)
           for i in range(evaluation_points.shape[0])]
    tmp4 = [bspl.evaluate_all_bspl(knt2, deg[1], evaluation_points[i][1], deg[1], nu=1)
           for i in range(evaluation_points.shape[0])]
    reference_basis_derivatives1 = np.vstack(tmp3).T
    reference_basis_derivatives2 = np.vstack(tmp4).T

    reference_basis_derivatives = np.zeros(((deg[0]+1)*(deg[1]+1), evaluation_points.shape[0], 2))
    for i1 in range(deg[0]+1):
        for i2 in range(deg[1]+1):
            reference_basis_derivatives[i1 + i2*(deg[0]+1), :, 0] = np.multiply(reference_basis_derivatives1[i1], reference_basis2[i2])
            reference_basis_derivatives[i1 + i2*(deg[0]+1), :, 1] = np.multiply(reference_basis1[i1], reference_basis_derivatives2[i2])

    reference_data = {'reference_element': reference_element,
                      'evaluation_points': evaluation_points,
                      'quadrature_weights': quadrature_weights,
                      'deg': deg,
                      'reference_basis': reference_basis,
                      'reference_basis_derivatives': reference_basis_derivatives}
    return reference_data

def create_geometric_map(fe_geometry, ref_data):
    # m = int(fe_geometry['m'])
    m = fe_geometry.m
    coeff = fe_geometry.map_coefficients
    nq = np.size(ref_data['evaluation_points'], 0)
    mapp = np.zeros((nq, 2, m))
    map_derivatives = np.zeros((nq, 4, m))
    imap_derivatives = np.zeros((nq, 4, m))

    p1 = ref_data['deg'][0]
    p2 = ref_data['deg'][1]
    reference_basis_derivatives = ref_data['reference_basis_derivatives']
    geom_matrix_ext = coeff
    X1, X2 = geom_matrix_ext[:, 0, :], geom_matrix_ext[:, 1, :]

    # plt.scatter(X1, X2, marker='x')

    N = ref_data['reference_basis']
    for i in range(m):
        mapp[:, 0, i] = np.matmul(X1[:, i], N)
        mapp[:, 1, i] = np.matmul(X2[:, i], N)
        map_derivatives[:, 0, i] = np.matmul(X1[:, i], reference_basis_derivatives[:, :, 0])
        map_derivatives[:, 1, i] = np.matmul(X2[:, i], reference_basis_derivatives[:, :, 1])
        map_derivatives[:, 2, i] = np.matmul(X1[:, i], reference_basis_derivatives[:, :, 0])
        map_derivatives[:, 3, i] = np.matmul(X2[:, i], reference_basis_derivatives[:, :, 1])

        # Compute inverse map derivatives
    for i in (range(nq)):
        for k in range(m):
            phi_der_1_1 = 0
            phi_der_1_2 = 0
            
            phi_der_2_1 = 0
            phi_der_2_2 = 0
            for j in range((p1+1)*(p2+1)):
                phi_der_1_1 += X1[j,k]*\
                    ref_data['reference_basis_derivatives'][j][i][0]
                    
                phi_der_2_1 += X2[j,k]*\
                    ref_data['reference_basis_derivatives'][j][i][0]
                    
                phi_der_1_2 += X1[j,k]*\
                    ref_data['reference_basis_derivatives'][j][i][1]
                
                phi_der_2_2 += X2[j,k]*\
                    ref_data['reference_basis_derivatives'][j][i][1]
                    
            map_derivatives[i][0][k] = phi_der_1_1
            map_derivatives[i][1][k] = phi_der_2_1
            
            map_derivatives[i][2][k] = phi_der_1_2
            map_derivatives[i][3][k] = phi_der_2_2
            
            # CREATE INVERSE MAP DERIVATIVES (imap_der)
            
            det = phi_der_1_1 * phi_der_2_2 - phi_der_2_1 * phi_der_1_2
            detinv = 1 / det 
            
            imap_derivatives[i][0][k] = detinv *  phi_der_2_2
            imap_derivatives[i][1][k] = detinv * -phi_der_2_1
            
            imap_derivatives[i][2][k] = detinv * -phi_der_1_2
            imap_derivatives[i][3][k] = detinv * phi_der_1_1

    geom_map = {'mapp': mapp,
                'map_derivatives': map_derivatives,
                'imap_derivatives': imap_derivatives}
    return geom_map

def fe_basis_2d(ref_data, fe_geometry, fe_space, plot=True):

    nq = len(ref_data["evaluation_points"])
    m = fe_geometry.m
    p1 = ref_data['deg'][0]
    p2 = ref_data['deg'][1]
    N = np.zeros((m, int(fe_space.n), nq))
    geom_map = create_geometric_map(fe_geometry, ref_data)
    mapp = geom_map['mapp']

    for i in range(m): # each subdomain i
        extraction_coeff = fe_space.support_and_extraction[i].extraction_coefficients
        for j, k in enumerate(fe_space.support_and_extraction[i].supported_bases): 
            for l in range((p1+1)*(p2+1)): # only run over the values where it is nonzero
                N[i,k,:] += extraction_coeff[j,l]*ref_data['reference_basis'][l]

    x_axis = geom_map['mapp'].reshape(-1, m).T
    x_axis = x_axis.reshape(m, nq, 2)
    x_axis = x_axis.reshape(-1, x_axis.shape[-1])

    if plot is True:
        return N
    elif plot == 'all':
        plots = range(fe_space.n)
    elif plot-1 in range(fe_space.n):
        plots = [plot-1]
    
    for i in plots:

        y = N[:,i].flatten()

        triang = mtri.Triangulation(x_axis[:,0], x_axis[:,1])

        fig, ax = plt.subplots(1, 2, figsize=(18, 6))
        for i in range(mapp.shape[2]):
            ax[0].scatter(mapp[:, 0, i], mapp[:, 1, i], s=0.2, label=f'Element {i+1}')
        ax[0].set_xlabel('X')
        ax[0].set_ylabel('Y')
        ax[0].set_title('Physical domain')
        
        c = ax[1].tripcolor(triang, y, cmap='viridis')

        sm = ScalarMappable(cmap='viridis')
        sm.set_array(y)

        cbar = plt.colorbar(sm, ax=ax)
        ax[1].set_xlabel('X')
        ax[1].set_ylabel('Y')
        cbar.set_label(f'$N_{plot}$')
        ax[1].set_title(f'Basis Function $N_{plot}$')
        plt.show()

    return N

def read_file(file_name, neval):
    print('reading file: ', file_name)
    data = sio.loadmat(file_name, struct_as_record=False, squeeze_me=True)
    p = data['p'].flatten()
    p1 = p[0]
    p2 = p[1]
    p = (p1, p2) 
    ref_data = create_ref_data(neval, p, 'integrate')
    fe_geometry = data['fe_geometry']
    fe_space = data['fe_space']
    print('p1: ', p1)
    print('p2: ', p2)
    print('neval: ', neval)
    print('number of elements: ', fe_space.n)

    return ref_data, fe_geometry, fe_space

def plot_domain_and_basis_function(fn, ind, neval, ref_data, fe_geometry, fe_space):
    print('Plotting domain and basis functions...')
    N = fe_basis_2d(ref_data, fe_geometry, fe_space, plot=5)
    return

def assemble_fe_problem_2d_newnew(fe_geometry, fe_space, ref_data, geom_map, problem_B, problem_L):
    """
    Assemble the finite element problem matrix A and vector b.

    Parameters:
    fe_space: Object containing the finite element space information.
    ref_data: Dictionary containing reference data for evaluation points and quadrature weights.
    geom_map: Dictionary containing the geometric mapping of elements.
    problem_B: Function defining the integrand for the bilinear form B.
    problem_L: Function defining the integrand for the linear form L.

    Returns:
    A: Assembled matrix A (n x n).
    b: Assembled vector b (n x 1).
    """
    print('Assembling FE problem...')
    # First compute N_i abd N_j and their derivatives
    p1 = ref_data['deg'][0]
    p2 = ref_data['deg'][1]
    # First compute N_i abd N_j and their derivatives
    neval = len(ref_data["evaluation_points"])
    m = fe_geometry.m
    n = fe_space.n

    # Initialize values
    print(type(m), type(n), type(neval))
    
    
    # Initialize A, b

    A = np.zeros((n,n))
    b = np.zeros((n))

    for l in range(m): # in every Omega_i
        print('element = ',l)
        N_ji = np.zeros((n, neval)) # function values of N_j on interval i #HIER BEN IK
        N_ji_der = np.zeros((n, neval, 2)) # derivative values of N_j on interval i
    
        # print('element = ',l)
        supported_bases = fe_space.support_and_extraction[l].supported_bases
        extraction_matrix = fe_space.support_and_extraction[l].extraction_coefficients

        for local_ind, global_ind in enumerate(supported_bases):#for u in range(0, (ref_data["deg"][0] + 1)*(ref_data["deg"][1]+1)): #run over the p support functions
            for q in range((p1+ 1)*(p2+1)):
                N_ji[global_ind,:] += extraction_matrix[local_ind,q] * ref_data['reference_basis'][q] 
                N_ji_der[global_ind,:,0] += extraction_matrix[local_ind,q]*ref_data['reference_basis_derivatives'][q,:,0]*geom_map['imap_derivatives'][:,0,l] + \
                    extraction_matrix[local_ind,q]*ref_data['reference_basis_derivatives'][q,:,1]*geom_map['imap_derivatives'][:,1,l]
                N_ji_der[global_ind,:,1] += extraction_matrix[local_ind,q]*ref_data['reference_basis_derivatives'][q,:,0]*geom_map['imap_derivatives'][:,2,l] + \
                    extraction_matrix[local_ind,q]*ref_data['reference_basis_derivatives'][q,:,1]*geom_map['imap_derivatives'][:,3,l]

    # Now create A and b

        for i in supported_bases:
            for j in supported_bases:
                value = 0
                for r in range(neval):
                    x = r
                    ksi = ref_data["evaluation_points"][r]
                    x = geom_map['mapp'][r,:,l] 
                    der = geom_map["map_derivatives"][r,:,l]
                    det = der[0]*der[3]-der[1]*der[2]
   
                    value += problem_B(geom_map['mapp'][r,:,l],\
                                       N_ji[i,r], \
                                       N_ji_der[i,r], \
                                       N_ji[j,r], \
                                       N_ji_der[j,r]) * \
                            det *\
                            ref_data["quadrature_weights"][r]  #DIT KLOPT NOG NIET!!!
                # print('value:', value)
                A[i,j] += value 

            value = 0
            for r in range(neval):
                x = r
                ksi = ref_data["evaluation_points"][r]
                der = geom_map["map_derivatives"][r,:,l]
                det = der[0]*der[3]-der[1]*der[2]
                value += problem_L(geom_map['mapp'][r,:,l], N_ji[i,r], N_ji_der[i,r]) *det*ref_data["quadrature_weights"][r]               
            b[i] += value

    return A, b

def assemble_fe_problem_2d(fe_geometry, fe_space ,ref_data, geom_map, problem_B, problem_L):
    
    # First compute N_i abd N_j and their derivatives
    neval = len(ref_data["evaluation_points"])
    m = fe_geometry.m
    n = fe_space.n

    # Initialize values
    print(type(m), type(n), type(neval))
    N_ji = np.zeros((m, n, neval)) # function values of N_j on interval i #HIER BEN IK
    N_ji_der = np.zeros((m, n, neval, 2)) # derivative values of N_j on interval i

    # Initialize A, b

    A = np.zeros((n,n))
    b = np.zeros((n))

    for interval in range(m): # in every Omega_i
        print('interval = ',interval)
        supported_bases_interval = fe_space.support_and_extraction[interval].supported_bases
        extraction_matrix_interval = fe_space.support_and_extraction[interval].extraction_coefficients
        
        for u in range(len(supported_bases_interval)):#for u in range(0, (ref_data["deg"][0] + 1)*(ref_data["deg"][1]+1)): #run over the p support functions
            j = supported_bases_interval[u] #set j to be the u-th function that is supported in Omega_i

            for ksi in range(neval):
                val = 0 #function value of N_j on ksi
                dval0 = 0 #derivative value of N_j on ksi
                dval1 = 0
                for k in range((ref_data["deg"][0]+ 1)*(ref_data["deg"][1]+1)):
                    val += extraction_matrix_interval[u,k] * ref_data['reference_basis'][k][ksi] 
                    dval0 += np.dot(geom_map['imap_derivatives'][ksi,:2,interval], ref_data['reference_basis_derivatives'][k,ksi,:]) * extraction_matrix_interval[u,k] 
                    dval1 += np.dot(geom_map['imap_derivatives'][ksi,2:,interval], ref_data['reference_basis_derivatives'][k,ksi,:]) * extraction_matrix_interval[u,k] 

                N_ji[interval][j,ksi] = val
                N_ji_der[interval][j,ksi] = dval0, dval1

    # Now create A and b

        for i in supported_bases_interval:
            for j in supported_bases_interval:
                value = 0
                for r in range(neval):
                    x = r
                    ksi = ref_data["evaluation_points"][r]
                    x = geom_map['mapp'][r,:,interval] 
                    #x_0 = 
                    #x_1 = 
                    der = geom_map["map_derivatives"][r,:,interval]
                    det = der[0]*der[3]-der[1]*der[2]
   
                    value += problem_B(geom_map['mapp'][r,:,interval],\
                                       N_ji[interval][i,r], \
                                       N_ji_der[interval][i,r], \
                                       N_ji[interval][j,r], \
                                       N_ji_der[interval][j,r]) * \
                            det *\
                            ref_data["quadrature_weights"][r]  #DIT KLOPT NOG NIET!!!
                print('value:', value)
                A[i,j] += value 

            value = 0
            for r in range(neval):
                x = r
                ksi = ref_data["evaluation_points"][r]
                der = geom_map["map_derivatives"][r,:,interval]
                det = der[0]*der[3]-der[1]*der[2]
                value += problem_L(geom_map['mapp'][r,:,interval], N_ji[interval][i,r], N_ji_der[interval][i,r]) *det*ref_data["quadrature_weights"][r]               
            b[i] += value

    return A, b

def assemble_fe_problem_2d_new(fe_geometry, fe_space ,ref_data, geom_map, problem_B, problem_L):
    
    p1 = ref_data['deg'][0]
    p2 = ref_data['deg'][1]
    # First compute N_i abd N_j and their derivatives
    neval = len(ref_data["evaluation_points"])
    m = fe_geometry.m
    n = fe_space.n

    # Initialize values
    print(type(m), type(n), type(neval))
    N_ji = np.zeros((m, n, neval)) # function values of N_j on interval i #HIER BEN IK
    N_ji_der = np.zeros((m, n, neval, 2)) # derivative values of N_j on interval i
    
    # Initialize A, b

    A = np.zeros((n,n))
    b = np.zeros((n))

    for l in range(m): # in every Omega_i
        print('element = ',l)
        supported_bases = fe_space.support_and_extraction[l].supported_bases
        extraction_matrix = fe_space.support_and_extraction[l].extraction_coefficients

        for local_ind, global_ind in enumerate(supported_bases):#for u in range(0, (ref_data["deg"][0] + 1)*(ref_data["deg"][1]+1)): #run over the p support functions
            for q in range((p1+ 1)*(p2+1)):
                N_ji[l,global_ind,:] += extraction_matrix[local_ind,q] * ref_data['reference_basis'][q] 
                N_ji_der[l,global_ind,:,0] += extraction_matrix[local_ind,q]*ref_data['reference_basis_derivatives'][q,:,0]*geom_map['imap_derivatives'][:,0,l] + \
                    extraction_matrix[local_ind,q]*ref_data['reference_basis_derivatives'][q,:,1]*geom_map['imap_derivatives'][:,1,l]
                N_ji_der[l,global_ind,:,1] += extraction_matrix[local_ind,q]*ref_data['reference_basis_derivatives'][q,:,0]*geom_map['imap_derivatives'][:,2,l] + \
                    extraction_matrix[local_ind,q]*ref_data['reference_basis_derivatives'][q,:,1]*geom_map['imap_derivatives'][:,3,l]

    # Now create A and b

        for i in supported_bases:
            for j in supported_bases:
                value = 0
                for r in range(neval):
                    x = r
                    ksi = ref_data["evaluation_points"][r]
                    x = geom_map['mapp'][r,:,l] 
                    der = geom_map["map_derivatives"][r,:,l]
                    det = der[0]*der[3]-der[1]*der[2]
   
                    value += problem_B(geom_map['mapp'][r,:,l],\
                                       N_ji[l][i,r], \
                                       N_ji_der[l][i,r], \
                                       N_ji[l][j,r], \
                                       N_ji_der[l][j,r]) * \
                            det *\
                            ref_data["quadrature_weights"][r]  #DIT KLOPT NOG NIET!!!
                print('value:', value)
                A[i,j] += value 

            value = 0
            for r in range(neval):
                x = r
                ksi = ref_data["evaluation_points"][r]
                der = geom_map["map_derivatives"][r,:,l]
                det = der[0]*der[3]-der[1]*der[2]
                value += problem_L(geom_map['mapp'][r,:,l], N_ji[l][i,r], N_ji_der[l][i,r]) *det*ref_data["quadrature_weights"][r]               
            b[i] += value

    return A, b

# question H
def problem_B_H(x,Ni,dNi,Nj,dNj):
    return float(np.dot(dNi,dNj.T))

def problem_L_H(x,Ni,dNi):
    return Ni

#question I
def problem_B_I(x,Ni,dNi,Nj,dNj):
    return float(np.dot(dNi,dNj.T))

def problem_L_I(x,Ni,dNi):
    return np.sin(x[0])*np.sin(x[1])*Ni

def plot_solution_1d(mesh, space ,ref_data, param_map, coeffs=None, offset=None, derivative=False):
    # retrieve data
    nel = mesh['m']
    B = ref_data['reference_basis']
    dB = ref_data['reference_basis_derivatives']
    xi = ref_data['evaluation_points']
    deg = ref_data['deg']
    # colors
    col = ['g', 'r', 'b', 'c', 'm', 'y', 'k']
    ncol = len(col)
    for i in range(nel):
        # geometry information
        x = param_map['map'](xi,mesh['elements'][0][i],mesh['elements'][1][i])
        dx = param_map['map_derivatives'][i]
        dxi = param_map['imap_derivatives'][i]
        E = space['extraction_coefficients'][i]
        N = np.matmul(E,B)
        dN = np.matmul(E,dB)*dxi
        I = space['supported_bases'][i]
        if coeffs is None:
            for j in range(deg+1):
                if derivative is False:
                    plt.plot(x,N[j],col[np.mod(I[j],ncol)] + '-', lw=3)
                else:
                    plt.plot(x,dN[j],col[np.mod(I[j],ncol)] + '-', lw=3)
        else:
            if offset is None:
                if derivative is False:
                    s = np.matmul(coeffs[I],N)
                else:
                    s = np.matmul(coeffs[I],dN)
            else:
                if derivative is False:
                    s = np.matmul(coeffs[I],N) - offset(x)
                else:
                    s = np.matmul(coeffs[I],dN) - offset(x)
            plt.plot(x,s,col[np.mod(i,ncol)] + '-', lw=3)
    plt.grid(True)
    plt.show()

def reduced_u(fe_space, A, b, save=False):
    i_delete = fe_space.boundary_bases   
    A_tilde = np.delete(A, i_delete, axis=0)
    A_tilde = np.delete(A_tilde, i_delete, axis=1)
    b_tilde = np.delete(b, i_delete)

    # save A and b in text file in folder H
    if save=='H':
        np.savetxt('/home/rosa/Desktop/Master/S2/FEM/assignments/ass4/data/H/A_tilde.csv', A_tilde)
        np.savetxt('/home/rosa/Desktop/Master/S2/FEM/assignments/ass4/data/H/b_tilde.csv', b_tilde)
        # save also in text file in folder H
        np.savetxt('/home/rosa/Desktop/Master/S2/FEM/assignments/ass4/data/H/A_tilde.txt', A_tilde)
        np.savetxt('/home/rosa/Desktop/Master/S2/FEM/assignments/ass4/data/H/b_tilde.txt', b_tilde)
    if save=='I':
        np.savetxt('/home/rosa/Desktop/Master/S2/FEM/assignments/ass4/data/I/A_tilde.csv', A_tilde)
        np.savetxt('/home/rosa/Desktop/Master/S2/FEM/assignments/ass4/data/I/b_tilde.csv', b_tilde)
        # save also in text file in folder I
        np.savetxt('/home/rosa/Desktop/Master/S2/FEM/assignments/ass4/data/I/A_tilde.txt', A_tilde)
        np.savetxt('/home/rosa/Desktop/Master/S2/FEM/assignments/ass4/data/I/b_tilde.txt', b_tilde)
        
    u_tilde = np.linalg.solve(A_tilde, b_tilde)
    return u_tilde

def plot_solution_2d(ref_data, fe_geometry, fe_space, u_coeffs=None, offset=None, derivative=False):
    
    nq = len(ref_data["evaluation_points"])
    m = fe_geometry.m
    p1 = ref_data['deg'][0]
    p2 = ref_data['deg'][1]
    N = np.zeros((m, int(fe_space.n), nq))
    dN = np.zeros((m, int(fe_space.n), nq, 2))
    geom_map = create_geometric_map(fe_geometry, ref_data)
    mapp = geom_map['mapp']

    def coeffs_extended(u_coeffs, fe_space):
        """ Extend the coefficients to the boundary bases, puts zero in the index of the boundary bases """
        n = fe_space.n
        boundary_bases = fe_space.boundary_bases
        u_coeffs_extended = np.zeros(n)
        index = 0
        for i in range(n):
            if i in boundary_bases:
                index += 1
            else:
                u_coeffs_extended[i] = u_coeffs[i-index]
        return u_coeffs_extended
    
    u_coeffs = coeffs_extended(u_tilde, fe_space)

    for i in range(m): # each subdomain i
        for j, k in enumerate(fe_space.support_and_extraction[i].supported_bases): 
            if k in fe_space.boundary_bases:
                continue
            extraction_coeff = fe_space.support_and_extraction[i].extraction_coefficients

            for l in range((p1+1)*(p2+1)): # only run over the values where it is nonzero
                N[i,k,:] += extraction_coeff[j,l]*ref_data['reference_basis'][l]
                dN[i,k,:,0] += extraction_coeff[j,l]*ref_data['reference_basis_derivatives'][l,:,0]*geom_map['imap_derivatives'][:,0,i] + \
                    extraction_coeff[j,l]*ref_data['reference_basis_derivatives'][l,:,1]*geom_map['imap_derivatives'][:,1,i]
                dN[i,k,:,1] += extraction_coeff[j,l]*ref_data['reference_basis_derivatives'][l,:,0]*geom_map['imap_derivatives'][:,2,i] + \
                    extraction_coeff[j,l]*ref_data['reference_basis_derivatives'][l,:,1]*geom_map['imap_derivatives'][:,3,i]

    u_h = np.zeros((m, nq))
    u_h_der = np.zeros((m, nq, 2))

    for i in range(m):
        for j in fe_space.support_and_extraction[i].supported_bases:
            if j not in fe_space.boundary_bases:
                u_h[i] += u_coeffs[j]*N[i,j]
                u_h_der[i,:,0] += u_coeffs[j]*dN[i,j,:,0]
                u_h_der[i,:,1] += u_coeffs[j]*dN[i,j,:,1]

    u_h_value = u_h.flatten()
    u_h_der_x = u_h_der[:,:,0].flatten()
    u_u_der_y = u_h_der[:,:,1].flatten()

    u_h_min, u_h_max = np.min(u_h_value), np.max(u_h_value)
    u_h_der_x_min, u_h_der_x_max = np.min(u_h_der_x), np.max(u_h_der_x)
    u_h_der_y_min, u_h_der_y_max = np.min(u_u_der_y), np.max(u_u_der_y)

    # print('u_h: ', u_h[0])
    # print('u_h_min: ', u_h_min)
    # print('u_h_max: ', u_h_max)
    # print('u_h_der_x_min: ', u_h_der_x_min)
    # print('u_h_der_x_max: ', u_h_der_x_max)
    # print('u_h_der_y_min: ', u_h_der_y_min)
    # print('u_h_der_y_max: ', u_h_der_y_max)

    # x_axis = geom_map['mapp'].reshape(-1, m).T
    # x_axis = x_axis.reshape(m, nq, 2)
    # x_axis = x_axis.reshape(-1, x_axis.shape[-1])

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (30, 5))
    
    ax1.title.set_text(f'$u$')
    ax2.title.set_text(f'$du/dx$')
    ax3.title.set_text(f'$du/dy$')

    for l in range(m):
        # Plot
        x_axis = geom_map['mapp'][:,:,l]
        triang = mtri.Triangulation(x_axis[:,0], x_axis[:,1])
        
        #print('x_axis.shape', x_axis.shape)
        #print('u_h_tot[l].shape', u_h_tot[l].shape)
        
        c = ax1.tripcolor(triang, u_h[l], cmap='viridis',          vmin = u_h_min, vmax = u_h_max)
        d = ax2.tripcolor(triang, u_h_der[l][:,0], cmap='viridis', vmin = u_h_der_x_min, vmax = u_h_der_x_max)
        e = ax3.tripcolor(triang, u_h_der[l][:,1], cmap='viridis', vmin = u_h_der_y_min, vmax = u_h_der_y_max)
    
    #sm = ScalarMappable(cmap='viridis')
    #sm.set_array(u_h_tot)
    cbar = plt.colorbar(c, ax=ax1)
    cbar.set_label(f'$u$')

    #sm.set_array(u_h_derx)
    cbar = plt.colorbar(d, ax=ax2)
    cbar.set_label(f'$du/dx$')

    #sm.set_array(u_h_dery)
    cbar = plt.colorbar(e, ax=ax3)
    cbar.set_label(f'$du/dy$')

    ax2.set_xlabel('X')
    ax1.set_ylabel('Y')
    #plt.title(f'Solution $u_h$')
    fig.tight_layout()
    # Show the plot
    plt.show() 
            
    return u_h

def plot_solution_2d_new(ref_data, fe_geometry, fe_space, case, u_coeffs=None, offset=None, derivative=False):
    
    nq = len(ref_data["evaluation_points"])
    m = fe_geometry.m
    u_h = np.zeros((m, nq))
    u_h_der = np.zeros((m, nq, 2))
    p1 = ref_data['deg'][0]
    p2 = ref_data['deg'][1]
    
    geom_map = create_geometric_map(fe_geometry, ref_data)
    mapp = geom_map['mapp']

    def coeffs_extended():
        """ Extend the coefficients to the boundary bases, puts zero in the index of the boundary bases """
        n = fe_space.n
        boundary_bases = fe_space.boundary_bases
        u_coeffs_extended = np.zeros(n)
        index = 0
        for i in range(n):
            if i in boundary_bases:
                index += 1
            else:
                u_coeffs_extended[i] = u_coeffs[i-index]
        return u_coeffs_extended

    u_coeffs = coeffs_extended()

    for i in range(m): # each subdomain i
        N = np.zeros((int(fe_space.n), nq))
        dN = np.zeros((int(fe_space.n), nq, 2))

        supp_bases = fe_space.support_and_extraction[i].supported_bases
        extraction_matrix = fe_space.support_and_extraction[i].extraction_coefficients

        for j, k in enumerate(supp_bases): 
            if k in fe_space.boundary_bases:
                continue
            extraction_coeff = extraction_matrix

            for l in range((p1+1)*(p2+1)): # only run over the values where it is nonzero
                N[k,:] += extraction_coeff[j,l]*ref_data['reference_basis'][l]
                dN[k,:,0] += extraction_coeff[j,l]*ref_data['reference_basis_derivatives'][l,:,0]*geom_map['imap_derivatives'][:,0,i] + \
                    extraction_coeff[j,l]*ref_data['reference_basis_derivatives'][l,:,1]*geom_map['imap_derivatives'][:,1,i]
                dN[k,:,1] += extraction_coeff[j,l]*ref_data['reference_basis_derivatives'][l,:,0]*geom_map['imap_derivatives'][:,2,i] + \
                    extraction_coeff[j,l]*ref_data['reference_basis_derivatives'][l,:,1]*geom_map['imap_derivatives'][:,3,i]

        u_hh = np.zeros((nq))
        u_hh_der = np.zeros((nq, 2))
        count = 0
        for j in supp_bases:
            if j not in fe_space.boundary_bases:
            #     count+=1
            #     continue
            # if j not in fe_space.boundary_bases:
                u_hh += u_coeffs[j]*N[j+count]
                u_hh_der[:,0] += u_coeffs[j]*dN[j,:,0]
                u_hh_der[:,1] += u_coeffs[j]*dN[j,:,1]

        u_h[i] = u_hh
        u_h_der[i] = u_hh_der

    u_h_value = u_h.flatten()
    u_h_der_x = u_h_der[:,:,0].flatten()
    u_u_der_y = u_h_der[:,:,1].flatten()

    u_h_min, u_h_max = np.min(u_h_value), np.max(u_h_value)
    u_h_der_x_min, u_h_der_x_max = np.min(u_h_der_x), np.max(u_h_der_x)
    u_h_der_y_min, u_h_der_y_max = np.min(u_u_der_y), np.max(u_u_der_y)

    # print('u_h: ', u_h[0])
    # print('u_h_min: ', u_h_min)
    # print('u_h_max: ', u_h_max)
    # print('u_h_der_x_min: ', u_h_der_x_min)
    # print('u_h_der_x_max: ', u_h_der_x_max)
    # print('u_h_der_y_min: ', u_h_der_y_min)
    # print('u_h_der_y_max: ', u_h_der_y_max)

    # x_axis = geom_map['mapp'].reshape(-1, m).T
    # x_axis = x_axis.reshape(m, nq, 2)
    # x_axis = x_axis.reshape(-1, x_axis.shape[-1])

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (30, 5))
    
    ax1.title.set_text(f'$u$')
    ax2.title.set_text(f'$du/dx$')
    ax3.title.set_text(f'$du/dy$')

    print('plotting')
    for l in range(m):
        # Plot
        # print('element = ',i)
        x_axis = geom_map['mapp'][:,:,l]
        triang = mtri.Triangulation(x_axis[:,0], x_axis[:,1])
        
        #print('x_axis.shape', x_axis.shape)
        #print('u_h_tot[l].shape', u_h_tot[l].shape)
        
        c = ax1.tripcolor(triang, u_h[l], cmap='viridis',          vmin = u_h_min, vmax = u_h_max)
        d = ax2.tripcolor(triang, u_h_der[l][:,0], cmap='viridis', vmin = u_h_der_x_min, vmax = u_h_der_x_max)
        e = ax3.tripcolor(triang, u_h_der[l][:,1], cmap='viridis', vmin = u_h_der_y_min, vmax = u_h_der_y_max)
    
    #sm = ScalarMappable(cmap='viridis')
    #sm.set_array(u_h_tot)
    cbar = plt.colorbar(c, ax=ax1)
    cbar.set_label(f'$u$')

    #sm.set_array(u_h_derx)
    cbar = plt.colorbar(d, ax=ax2)
    cbar.set_label(f'$du/dx$')

    #sm.set_array(u_h_dery)
    cbar = plt.colorbar(e, ax=ax3)
    cbar.set_label(f'$du/dy$')

    ax2.set_xlabel('X')
    ax1.set_ylabel('Y')
    #plt.title(f'Solution $u_h$')
    fig.tight_layout()
    # save plot
    plt.savefig(f'ass4/data/{case}.png')
    # Show the plot
    plt.show() 
            
    return u_h

if __name__ == '__main__':
    main()

