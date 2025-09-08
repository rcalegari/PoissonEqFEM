import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from numpy.polynomial.legendre import leggauss as gaussquad
from scipy.interpolate import _bspl as bspl
import matplotlib.pyplot as plt
import scipy.io as sio
from matplotlib import cm

def create_ref_data(neval, deg, integrate=False):
    # reference unit domain
    reference_element = np.array([0, 1])
    if integrate is False:
        # point for plotting are equispaced on reference element
        x = np.linspace(reference_element[0], reference_element[1], neval)
        evaluation_points = x
        quadrature_weights = np.zeros((0,))
    else:
        # points (and weights) for integration are computed according to Gauss quadrature
        x, w = gaussquad(neval)
        evaluation_points = 0.5*(x + 1)
        quadrature_weights = w/2
    # knots for defining B-splines
    knt = np.concatenate((np.zeros((deg+1,),dtype=float),np.ones((deg+1,),dtype=float)),axis=0)
    # reference basis function values
    tmp = [bspl.evaluate_all_bspl(knt, deg, evaluation_points[i], deg, nu=0)
           for i in range(evaluation_points.shape[0])]
    reference_basis = np.vstack(tmp).T
    # reference basis function first derivatives
    tmp = [bspl.evaluate_all_bspl(knt, deg, evaluation_points[i], deg, nu=1)
           for i in range(evaluation_points.shape[0])]
    reference_basis_derivatives = np.vstack(tmp).T
    # store all data and return
    reference_data = {'deg': deg,
                      'reference_element': reference_element,
                      'evaluation_points': evaluation_points,
                      'quadrature_weights': quadrature_weights,
                      'reference_basis': reference_basis,
                      'reference_basis_derivatives': reference_basis_derivatives
    }
    return reference_data

def plot_fe_basis_functions(mesh, fe_space, reference_data, param_map):
    fig, axs = plt.subplots(2, 1, figsize=(14, 10))
    deg = reference_data['deg']
    for i, element in enumerate(mesh['elements'].T):
        for local_index in range(deg + 1):
            for j in range(0,local_index):
                print('e_',i, local_index, j, ': ', fe_space['extraction_coefficients'][i][local_index][j])
            global_index = fe_space['supported_bases'][i, local_index]

            # basis functions and derivatives evaluated at all quadrature points
            basis_function = reference_data['reference_basis'][local_index, :] 
            basis_derivative = reference_data['reference_basis_derivatives'][local_index, :]*param_map['imap_derivatives'][i]

            # transform basis functions and derivatives to physical space'
            x_physical = param_map['map'](reference_data['evaluation_points'], element[0], element[1])
            axs[0].plot(x_physical, basis_function, label=f'Basis {global_index+1} on Element {i+1}')
            axs[1].plot(x_physical, basis_derivative, label=f'dBasis {global_index+1} on Element {i+1}')
    axs[0].set_title('Finite Element Basis Functions')
    axs[0].legend()
    axs[1].set_title('Derivatives of Finite Element Basis Functions')
    axs[1].legend()
    plt.tight_layout()
    plt.show()

def create_param_map(mesh):
    def map(ksi, x_i_1, x_i):
        # create mapping 
        output = x_i_1 + ksi * (x_i - x_i_1)
        return output
    
    """ Create derivatives of the map for every element
    map_derivatives is derivative of x_i_1 + ksi * (x_i - x_i_1) w.r.t. ksi
    which is x_i - x_i_1 """
    map_derivatives = mesh['elements'][1] - mesh['elements'][0]
    """ Create derivatives of the inverse map for every element
    inverse map derivative is derivative of (phi_i - x_i_1)/(x_i - x_i_1) w.r.t. phi_i
    which is 1/(x_i - x_i_1) """
    imap_derivatives = 1.0/(mesh['elements'][1] - mesh['elements'][0])
    
    param_map = {
        'map': map,
        'map_derivatives': map_derivatives,
        'imap_derivatives': imap_derivatives
    }
    
    return param_map

# now make create_ref_data for 2d where deg = (p1, p2), evaluation points = matrix(neval x 2), quadrature_weights = vector(neval x 1), reference_basis = matrix(neval x (p1+1)*(p2+1)), reference_basis_derivatives = 3d matrix(neval x (p1+1)*(p2+1) x 2)
def create_ref_data_2d(neval, deg, data_kind, prnt=False):
    # reference unit domain
    reference_element = np.array([0, 1, 0, 1])
    if data_kind == 'plot':
        # point for plotting are equispaced on reference element
        x = np.linspace(reference_element[0], reference_element[1], neval)
        y = np.linspace(reference_element[2], reference_element[3], neval)
        evaluation_points = np.zeros((neval*neval, 2))
        for i in range(neval):
            for j in range(neval):
                evaluation_points[i*neval + j] = np.array([x[i], y[j]])
        quadrature_weights = np.zeros((0,))
    elif data_kind == 'integrate':
        # points (and weights) for integration are computed according to Gauss quadrature
        x, w = gaussquad(neval)
        evaluation_points = np.zeros((neval*neval, 2))
        for i in range(neval):
            for j in range(neval):
                evaluation_points[i*neval + j] = np.array([0.5*(x[i] + 1), 0.5*(x[j] + 1)])
        quadrature_weights = np.zeros((neval*neval,))
        for i in range(neval):
            for j in range(neval):
                quadrature_weights[i*neval + j] = 0.25*w[i]*w[j]
    # knots for defining B-splines
    knt1 = np.concatenate((np.zeros((deg[0]+1,),dtype=float),np.ones((deg[0]+1,),dtype=float)),axis=0)
    knt2 = np.concatenate((np.zeros((deg[1]+1,),dtype=float),np.ones((deg[1]+1,),dtype=float)),axis=0)

    # reference basis function values
    tmp1 = [bspl.evaluate_all_bspl(knt1, deg[0], evaluation_points[i][0], deg[0], nu=0)
           for i in range(evaluation_points.shape[0])]
    
    reference_basis1 = np.vstack(tmp1).T

    tmp2 = [bspl.evaluate_all_bspl(knt2, deg[1], evaluation_points[i][1], deg[1], nu=0)
              for i in range(evaluation_points.shape[0])]
    
    reference_basis2 = np.vstack(tmp2).T

    reference_basis = np.zeros(((deg[0]+1)*(deg[1]+1), evaluation_points.shape[0]))

    for i1 in range(deg[0]+1):
        for i2 in range(deg[1]+1):
            reference_basis[i1 + (i2-1)*(deg[0]+1)] = np.multiply(reference_basis1[i1],reference_basis2[i2])
    if prnt:
        print('reference_basis: ', reference_basis.shape)

    # reference basis function first derivatives
    tmp3 = [bspl.evaluate_all_bspl(knt1, deg[0], evaluation_points[i][0], deg[0], nu=1)
           for i in range(evaluation_points.shape[0])]
    tmp4 = [bspl.evaluate_all_bspl(knt2, deg[1], evaluation_points[i][1], deg[1], nu=1)
                for i in range(evaluation_points.shape[0])]
    reference_basis_derivatives1 = np.vstack(tmp3).T
    reference_basis_derivatives2 = np.vstack(tmp4).T
    reference_basis_derivatives = np.zeros(((deg[0]+1)*(deg[1]+1), evaluation_points.shape[0], 2))
    if print:
        print('reference_basis_derivatives1: ', reference_basis_derivatives1.shape)
        print('reference_basis_derivatives2: ', reference_basis_derivatives2.shape)
        print('reference_basis_derivatives: ', reference_basis_derivatives.shape)

    for i1 in range(deg[0]+1):
        for i2 in range(deg[1]+1):
            reference_basis_derivatives[i1 + (i2-1)*(deg[0]+1), :, 0] = np.multiply(reference_basis_derivatives1[i1],reference_basis2[i2])
            reference_basis_derivatives[i1 + (i2-1)*(deg[0]+1), :, 1] = np.multiply(reference_basis1[i1],reference_basis_derivatives2[i2])

    # store all data and return
    reference_data = {'deg': deg,
                      'reference_element': reference_element,
                      'evaluation_points': evaluation_points,
                      'quadrature_weights': quadrature_weights,
                      'reference_basis': reference_basis,
                      'reference_basis_derivatives': reference_basis_derivatives
    }
    return reference_data

def create_geometric_map(fe_geometry, ref_data, prnt=False):

    p1, p2 = ref_data['deg'][0], ref_data['deg'][1]
    ref_basis = ref_data['reference_basis']
    ref_basis_derivatives = ref_data['reference_basis_derivatives'] # 3d matrix(neval x (p1+1)*(p2+1) x 2))
    
    m = fe_geometry.m
    print('m: ', type(m))
    # m = fe_geometry['m'][0][0][0][0]
    if prnt:
        print('m: ', m)
    coeff = fe_geometry.map_coefficients

    vals = np.zeros((ref_data['evaluation_points'].shape[0], 2, m))
    derivatives = np.zeros((ref_data['evaluation_points'].shape[0], 4, m))
    iderivatives = np.zeros((ref_data['evaluation_points'].shape[0], 4, m))
    
    for l in range(m):
        X1 = coeff[:,0,l]
        X2 = coeff[:,1,l]

        x1 = np.zeros(ref_data['evaluation_points'].shape[0])
        x2 = np.zeros(ref_data['evaluation_points'].shape[0])
        dx11 = np.zeros(ref_data['evaluation_points'].shape[0])
        dx12 = np.zeros(ref_data['evaluation_points'].shape[0])
        dx21 = np.zeros(ref_data['evaluation_points'].shape[0])
        dx22 = np.zeros(ref_data['evaluation_points'].shape[0])

        for j in range((p1+1)*(p2+1)):
            x1 += ref_basis[j]*X1[j]
            x2 += ref_basis[j]*X2[j]

            dx11 += ref_basis_derivatives[j,:,0]*X1[j]
            dx12 += ref_basis_derivatives[j,:,0]*X2[j]
            dx21 += ref_basis_derivatives[j,:,1]*X1[j]
            dx22 += ref_basis_derivatives[j,:,1]*X2[j]
        
        vals[:,0,l] = x1
        vals[:,1,l] = x2

        derivatives[:,0,l] = dx11
        derivatives[:,1,l] = dx12
        derivatives[:,2,l] = dx21
        derivatives[:,3,l] = dx22

        # inverse derivatives are the elements in the inverse of the Jacobian

        iderivatives[:,0,l] = 1.0/(dx11*dx22 - dx12*dx21)*dx22
        iderivatives[:,1,l] = -1.0/(dx11*dx22 - dx12*dx21)*dx12
        iderivatives[:,2,l] = -1.0/(dx11*dx22 - dx12*dx21)*dx21
        iderivatives[:,3,l] = 1.0/(dx11*dx22 - dx12*dx21)*dx11

        # check if there are any NaN values
        if np.isnan(vals).any() or np.isnan(derivatives).any() or np.isnan(iderivatives).any():
            raise ValueError('NaN values in geometric map')
        
    geom_map = {
        'map': vals,
        'map_derivatives': derivatives,
        'imap_derivatives': iderivatives
    }
    return geom_map

def test_ref_data_2d():
    n = 5
    p1 = 2
    p2 = 2
    ref_data = create_ref_data_2d(n, (p1, p2), 'integrate')
    return ref_data

def read_file(file_name):
    
    data = sio.loadmat(file_name, struct_as_record=False, squeeze_me=True)
    print(data.keys())
    # p1 = data['p'][0,0]
    # p2 = data['p'][0,1]
    p = data['p'].flatten()
    p1 = p[0]
    p2 = p[1]
    print('p1: ', p1)
    print('p2: ', p2)
    p = (p1, p2) 
    neval = 40
    ref_data = create_ref_data_2d(neval, p, 'integrate')
    fe_geometry = data['fe_geometry']
    fe_space = data['fe_space']
    # Check if fe_space is structured as expected
    # if isinstance(fe_space, np.ndarray) and fe_space.dtype.names:
    #     # Access structured data here
    #     print('Structured array detected')

    # geom_map = create_geometric_map(fe_geometry, ref_data)

    return ref_data, fe_geometry, fe_space

def plot_basis_functions_2d(fn, ind):
    ref_data, fe_geometry, fe_space = read_file(fn)
    geom_map = create_geometric_map(fe_geometry, ref_data)
    return

ref_data, fe_geometry, fe_space = read_file('/home/rosa/Desktop/Master/S2/FEM/assignments/ass4/data/star3.mat')
geom_map = create_geometric_map(fe_geometry, ref_data)
print('fe_space:\n n: ', fe_space.n, '\nboundary_bases: ',
       fe_space.boundary_bases)
print('# elts in support_and_extraction (should be equal to # elts): ',len(fe_space.support_and_extraction))  
print('data type in support_bases: ', type(fe_space.support_and_extraction[0].supported_bases), fe_space.support_and_extraction[0].supported_bases.shape)
print('data type in extraction_coefficients: ', type(fe_space.support_and_extraction[0].extraction_coefficients), fe_space.support_and_extraction[0].extraction_coefficients.shape)
for l in range(fe_geometry.m):
    print('support bases for elt ', l+1, ': ', fe_space.support_and_extraction[l].supported_bases)
    print('extraction coefficients for elt ', l+1, ': \n', fe_space.support_and_extraction[l].extraction_coefficients)
# print(fe_space['support_and_extraction'] )
# ['supported_bases'], '\nextraction_coefficients: ', fe_space['support_and_extraction']['extraction_coefficients'])

# fe_geometry, fe_space = read_file()
# ref_data = create_ref_data(neval= 20, deg =[2,2], integrate=False)
# geom_map = create_geometric_map(fe_geometry, ref_data)
mapp = geom_map['map']
ev_points = ref_data['evaluation_points']
plt.scatter(mapp[:, 0, 0], mapp[:, 1, 0], s=0.2)
plt.scatter(mapp[:, 0, 1], mapp[:, 1, 1], s=0.2)
plt.scatter(mapp[:, 0, 2], mapp[:, 1, 2], s=0.2)

plt.show()


def create_fe_space(deg, reg, mesh):
    def bezier_extraction(knt, deg):
        # breakpoints
        brk = np.unique(knt)
        # number of elements
        nel = brk.shape[0]-1
        # number of knots
        m = knt.shape[0]
        # assuming an open knotvector, knt[a] is the last repetition of the first knot
        a = deg
        # next knot
        b = a+1
        # Bezier element being processed
        nb = 0
        # first extraction matrix
        C = [np.eye(deg+1,deg+1, dtype=float)]
        # this is where knot-insertion coefficients are saved
        alphas = np.zeros((np.maximum(deg-1,0),),dtype=float)
        while b < m:
            # initialize extraction matrix for next element
            C.append(np.eye(deg+1,deg+1))
            # save index of current knot
            i = b
            # find last occurence of current knot
            while b < m-1 and knt[b+1] == knt[b]:
                b += 1
            # multiplicity of current knot
            mult = b-i+1
            # if multiplicity is < deg, smoothness is at least C0 and extraction may differ from an identity matrix
            if mult < deg:
                numer = knt[b] - knt[a]
                # smoothness of splines
                r = deg - mult
                # compute linear combination coefficients
                for j in range(deg-1,mult-1,-1):
                    alphas[j-mult] = numer / (knt[a+j+1]-knt[a])
                for j in range(r):
                    s = mult+j
                    for k in range(deg,s,-1):
                        alpha = alphas[k-s-1]
                        C[nb][:,k] = alpha*C[nb][:,k] + (1.0-alpha)*C[nb][:,k-1]
                    save = r-j
                    if b < m:
                        C[nb+1][save-1:j+save+1,save-1] = C[nb][deg-j-1:deg+1,deg]
            # increment element index
            nb += 1
            if b < m:
                a = b
                b += 1
            C = C[:nel]
        return np.array(C)
    # number of mesh elements
    nel = mesh['m']
    # unique breakpoints
    if nel == 1:
        brk = mesh['elements'].T[0]
    else:
        brk = np.concatenate((mesh['elements'][0],
                              np.array([mesh['elements'][1][-1]])), axis=0)
    # multiplicity of each breakpoint
    mult = deg - reg
    # knot vector for B-spline definition
    knt = np.concatenate((np.ones((deg+1,), dtype=float) * brk[0],
                          np.ones((deg+1,), dtype=float) * brk[-1],
                          np.repeat(brk[1:-1],mult)), axis=0)
    knt = np.sort(knt)
    # coefficients of linear combination
    C = bezier_extraction(knt, deg)
    # dimension of finite element space
    dim = knt.shape[0]-deg-1
    # connectivity information (i.e., which bases are non-zero on which element)
    econn = np.zeros((nel,deg+1), dtype=int)
    for i in range(nel):
        if i == 0:
            econn[i] = np.arange( deg+1)
        else:
            econn[i] = econn[i-1] + mult
    # save and return
    space = {'n': dim,
             'supported_bases': econn,
             'extraction_coefficients': C
    }
    return space

def create_mesh(brk):
    m = len(brk)-1
    elements = np.zeros((2,m))
    elements[0] = brk[0:-1]
    elements[1] = brk[1:]
    mesh = {
        'm': m,
        'elements': elements
    }
    return mesh


def assemble_fe_problem(mesh, space ,ref_data, param_map, problem_B, problem_L, bc):
    # First compute N_i abd N_j and their derivatives
    neval = len(ref_data["evaluation_points"])

    # Initialize values
    N_ji = np.zeros((mesh["m"], space['n'], neval)) # function values of N_j on interval i
    N_ji_der = np.zeros((mesh["m"], space['n'], neval)) # derivative values of N_j on interval i

    for interval in range(0, mesh["m"]): #in every Omega_i
        
        # for u, j in enumerate(space['supported_bases'][i]) #run over the p support functions
        for u in range(0, ref_data["deg"] + 1): #run over the p support functions
            j = space['supported_bases'][interval][u] #set j to be the u-th function that is supported in Omega_i

            for ksi in range(0, neval):
                val = 0 #function value of N_j on ksi
                dval = 0 #derivative value of N_j on ksi
                for k in range(0, ref_data["deg"]+ 1):
                    val += space['extraction_coefficients'][interval][u][k] * ref_data['reference_basis'][k][ksi]
                    dval += param_map['imap_derivatives'][interval] * space['extraction_coefficients'][interval][u][k] * ref_data['reference_basis_derivatives'][k][ksi]
                    
                N_ji[interval][j][ksi] = val
                N_ji_der[interval][j][ksi] = dval

    # Now create A and b
    A_bar = np.zeros((space["n"], space["n"]))
    b_bar = np.zeros((space["n"]))
    for l in range(mesh["m"]): #in every Omega_l
        x1 = mesh['elements'][0][l] #lower boundary
        x2 = mesh['elements'][1][l] #upper boundary
        
        for i in space['supported_bases'][l]:
            for j in space['supported_bases'][l]:
                value = 0
                for r in range(0, neval):
                    x = r
                    ksi = ref_data["evaluation_points"][r]
                        
                    value += problem_B(param_map["map"](ksi, x1, x2),\
                                       N_ji[l][i][x], \
                                       N_ji_der[l][i][x], \
                                       N_ji[l][j][x], \
                                       N_ji_der[l][j][x]) * \
                            param_map["map_derivatives"][l] *\
                            ref_data["quadrature_weights"][r]
                A_bar[i,j] += value
            value = 0
            for r in range(0, neval):
                x = r
                ksi = ref_data["evaluation_points"][r]
                
                value += problem_L(param_map["map"](ksi, x1, x2), N_ji[l][i][x], N_ji_der[l][i][x]) * \
                            param_map["map_derivatives"][l]*ref_data["quadrature_weights"][r]
            b_bar[i] += value

    #print('Sum of A bar:', sum(A_bar))
    
    A_bar = A_bar[1:-1] #discard first and last row, since those are corresponds to the boundary values that are known
    A = A_bar[:,1:-1]
    b = b_bar[1:-1] - bc[0]*A_bar[:,0] - bc[1]*A_bar[:,-1]
    return A, b

def assemble_mixed_fe_problem_L(mesh, space_list, ref_data_list, param_map, problem_B, problem_L):
    # First compute N_i abd N_j and their derivatives
    neval = len(ref_data_list[0]["evaluation_points"]) #neval is equal for 0 and 1

    n1 = space_list[0]['n'] #dimension of functions
    n2 = space_list[1]['n']

    A_bar = [[np.zeros((n1,n1)), np.zeros((n1,n2))],[np.zeros((n2,n1)), np.zeros((n2,n2))]]
    b_bar = [np.zeros(n1), np.zeros(n2)]

    for interval in range(0, mesh["m"]): #in every Omega_i
        for y in [1,0]:
            # Create N_ji_y and N_ji_der_y
            N_ji_y = np.zeros((mesh["m"], space_list[y]['n'], neval)) # function values of N_j on interval i
            N_ji_der_y = np.zeros((mesh["m"], space_list[y]['n'], neval)) # derivative values of N_j on interval i

            # for u, j in enumerate(space_list['supported_bases'][i]) #run over the p support functions
            for u in range(0, ref_data_list[y]["deg"] + 1): #run over the p support functions
                j = space_list[y]['supported_bases'][interval][u] #set j to be the u-th function that is supported in Omega_i

                for ksi in range(0, neval):
                    val = 0 #function value of N_j on ksi
                    dval = 0 #derivative value of N_j on ksi
                    for k in range(0, ref_data_list[y]["deg"]+ 1):
                        val += space_list[y]['extraction_coefficients'][interval][u][k] * ref_data_list[y]['reference_basis'][k][ksi]
                        dval += param_map['imap_derivatives'][interval] * space_list[y]['extraction_coefficients'][interval][u][k] * ref_data_list[y]['reference_basis_derivatives'][k][ksi]
                        
                    N_ji_y[interval][j][ksi] = val
                    N_ji_der_y[interval][j][ksi] = dval
            for z in [1,0]:
                #Create N_ji_z
                N_ji_z = np.zeros((mesh["m"], space_list[z]['n'], neval)) # function values of N_j on interval i
                N_ji_der_z = np.zeros((mesh["m"], space_list[z]['n'], neval)) # derivative values of N_j on interval i

                # for u, j in enumerate(space_list['supported_bases'][i]) #run over the p support functions
                for u in range(0, ref_data_list[z]["deg"] + 1): #run over the p support functions
                    j = space_list[z]['supported_bases'][interval][u] #set j to be the u-th function that is supported in Omega_i

                    for ksi in range(0, neval):
                        val = 0 #function value of N_j on ksi
                        dval = 0 #derivative value of N_j on ksi
                        for k in range(0, ref_data_list[z]["deg"]+ 1):
                            val += space_list[z]['extraction_coefficients'][interval][u][k] * ref_data_list[z]['reference_basis'][k][ksi]
                            dval += param_map['imap_derivatives'][interval] * space_list[z]['extraction_coefficients'][interval][u][k] * ref_data_list[z]['reference_basis_derivatives'][k][ksi]
                            
                        N_ji_z[interval][j][ksi] = val
                        N_ji_der_z[interval][j][ksi] = dval
                                
                # Now create A and b    
                x1 = mesh['elements'][0][interval] #lower boundary
                x2 = mesh['elements'][1][interval] #upper boundary    

                for i in space_list[y]['supported_bases'][interval]:
                    for j in space_list[z]['supported_bases'][interval]:
                        value = 0
                        for r in range(0, neval):
                            x = r
                            ksi = ref_data_list[y]["evaluation_points"][r]
                                
                            value += problem_B[y][z](param_map["map"](ksi, x1, x2),\
                                            N_ji_y[interval][i][x], \
                                            N_ji_der_y[interval][i][x], \
                                            N_ji_z[interval][j][x], \
                                            N_ji_der_z[interval][j][x]) * \
                                    param_map["map_derivatives"][interval] *\
                                    ref_data_list[y]["quadrature_weights"][r]
                        A_bar[y][z][i,j] += value
                value = 0
            for r in range(0, neval):
                x = r
                ksi = ref_data_list[y]["evaluation_points"][r]
                
                value += problem_L[y](param_map["map"](ksi, x1, x2), N_ji_y[interval][i][x], N_ji_der_y[interval][i][x]) * \
                            param_map["map_derivatives"][interval]*ref_data_list[y]["quadrature_weights"][r]
            #if z == 1:
            b_bar[y][i] += value

    #print('Sum of A bar:', sum(A_bar))
    A = np.bmat( [ [ A_bar[0][0],A_bar[0][1] ] ,[ A_bar[1][0], A_bar[1][1] ] ] )
    b = np.bmat( [ b_bar[0], b_bar[1]  ] )
    return A, b, N_ji_y, N_ji_z
