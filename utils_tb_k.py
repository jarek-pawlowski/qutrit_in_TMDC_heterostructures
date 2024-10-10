#import feast    # need to be imported first!
import os
from ordered_set import OrderedSet

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from scipy import interpolate
from scipy.linalg import inv, eigh

import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import warnings

import pickle


def abs2(c):
    """Calculates absolule value (modulus) of a given complex number.

    Parameters
    ----------
    c : complex
        Input complex number.

    Returns
    -------
    float
        Absolute value of c: |c|^2.
    """
    return c.real**2 + c.imag**2

def halfspace(a, b, delta):
    return lambda x,y : y > a*x + b - delta

def save_npy(filename, arr):
    with open(filename, 'wb') as f:
        np.save(f, arr, allow_pickle=True)

def load_npy(filename):
    with open(filename, 'rb') as f:
        return np.load(f, allow_pickle=True)


class AtomicUnits:
    """Class storing atomic units.

    All variables, arrays in simulations are in atomic units.

    Attributes
    ----------
    Eh : float
        Hartree energy (in meV)
    Ah : float
        Bohr radius (in nanometers)
    Th : float
        time (in picoseconds)
    Bh : float
        magnetic induction (in Teslas)
    """
    # atomic units
    Eh=27211.4 # meV
    Ah=0.05292 # nm
    Th=2.41888e-5 # ps
    Bh=235051.76 # Teslas

au = AtomicUnits()


class TMDCmaterial:
    """ Class containing lattice model parameters.

    Attributes
    ----------
    a0 : float
        MoS2 lattice constant (in nanometers)
    dim : int
        Hilbert space (sub)dimension: no of orbitals x spin degree = 3 x 2,
        dimension of the whole state-space = dim*N, where N is a no of lattice nodes
    dim2 : int
        squared dim: dim2 = dim*dim
    dim12 : int
        halved dim: dim12 = dim/2
    t.. : float
        tight-binding hopping parameters
    e. : float
        tight-binding onsite energies
    lso : float
        intinsic spin-orbit energy (in meV)
    """
    def __init__(self, a0, t0, t1, t2, t11, t12, t22, e0, e2, lso, r):
        self.a0 = a0/au.Ah
        self.dim = 6
        self.dim2 = self.dim*self.dim
        self.dim12 = int(self.dim/2)
        # hoppings
        self.t0 = t0/au.Eh
        self.t1 = t1/au.Eh
        self.t2 = t2/au.Eh
        self.t11 = t11/au.Eh
        self.t12 = t12/au.Eh
        self.t22 = t22/au.Eh
        # onsite energy
        self.e0 = e0/au.Eh
        self.e2 = e2/au.Eh
        self.diag = np.array([self.e0,self.e2,self.e2,self.e0,self.e2,self.e2])
        # intrinsic spin-orbit
        self.lso = lso/au.Eh
        self.rsba = np.array(r)/au.Ah
        
    def average_parameters(self, m1):
        return TMDCmaterial((self.a0+m1.a0)*au.Ah/2, 
                            (self.t0+m1.t0)*au.Eh/2, (self.t1+m1.t1)*au.Eh/2, (self.t2+m1.t2)*au.Eh/2,
                            (self.t11+m1.t11)*au.Eh/2, (self.t12+m1.t12)*au.Eh/2, (self.t22+m1.t22)*au.Eh/2,
                            (self.e0+m1.e0)*au.Eh/2, (self.e2+m1.e2)*au.Eh/2,
                            (self.lso+m1.lso)*au.Eh/2,
                            (self.rsba+m1.rsba)*au.Ah/2)


class TMDCmaterial_new:
    """ Class containing lattice model parameters.

    """
    def __init__(self, a0, dp, odd, Vdps, Vdpp, Vd2s, Vd2p, Vd2d, Vp2s, Vp2p, Ed, Ep1, Ep0, Eodd, lm, lx2, rsba):
        self.a0 = a0/au.Ah
        self.dr = self.a0/np.sqrt(3.)
        self.dp = dp/au.Ah
        self.dd = np.sqrt(self.dr**2+self.dp**2)
        self.dim = 22
        self.dim2 = self.dim*self.dim
        self.dim12 = int(self.dim/2)
        # include odd bands?
        self.odd = odd
        # hoppings
        self.Vdps = Vdps/au.Eh
        self.Vdpp = Vdpp/au.Eh
        self.Vd2s = Vd2s/au.Eh
        self.Vd2p = Vd2p/au.Eh
        self.Vd2d = Vd2d/au.Eh
        self.Vp2s = Vp2s/au.Eh
        self.Vp2p = Vp2p/au.Eh
        # onsite energy
        self.Ed = Ed/au.Eh
        self.Ep1 = Ep1/au.Eh
        self.Ep0 = Ep0/au.Eh
        self.Eodd = Eodd/au.Eh  # correction for odd bands
        self.diag = np.tile(np.array([self.Ed, self.Ed, self.Ed,
                                      self.Ep1, self.Ep0, self.Ep1,
                                      self.Ep1+self.Eodd[0], self.Ep0+self.Eodd[1], self.Ep1+self.Eodd[2], 
                                      self.Ed+self.Eodd[3], self.Ed+self.Eodd[4]]),2)
        # intrinsic spin-orbit
        self.lm = lm/au.Eh
        self.lx2 = lx2/au.Eh
        self.l_diag = np.array([-self.lm, 0.,  self.lm, -self.lx2/2., 0.,  self.lx2/2., -self.lx2/2., 0.,  self.lx2/2., -self.lm/2.,  self.lm/2.,
                                 self.lm, 0., -self.lm,  self.lx2/2., 0., -self.lx2/2.,  self.lx2/2., 0., -self.lx2/2.,  self.lm/2., -self.lm/2.])
        # rashba spin-orbit
        self.rsba = rsba/au.Ah


rashba_l = np.array([[7.5e-5,5.e-3,1.7e-2],
                     [5.e-3, 4.e-3,1.e-3],
                     [1.7e-2,1.e-3,4.e-3]])
MoS2 = TMDCmaterial(0.319, -184., 401., 507., 218., 338., 57., 1046., 2104., 73., rashba_l)
#WS2 = TMDCmaterial(0.319, -206., 567., 536., 286., 384., -61., 1130., 2275., 211., rashba_l*3.94)
WS2 = TMDCmaterial(0.319, -184., 401., 507., 218., 338., 57., 1046., 2104., 73., rashba_l)

# new materials
E_odd = np.array([2000., 7000., 2000., -4000., -4000.])
MoS2_new = TMDCmaterial_new(0.316, 0.158,
                            True,
                            -3390., 1100., -1100., 760., 270., 1190., -830., 
                            -30., -3360., -4780., 
                            E_odd, 
                            133.9/2, 40./2, rashba_l)
E_odd = np.array([-2000., -2000., -2000., -2000., -2000.])
WSe2_new = TMDCmaterial_new(0.19188*np.sqrt(3.), 0.16792,
                            False,
                            -1581.93, 1175.05, -905.01, 1082.3, -105.6, 520.91, -167.75,
                            -81.97, -2024.97, -3470.89, 
                            E_odd, 
                            550./2, 40./2, rashba_l)


class Lattice:

    def __init__(self, BZ_path=None):
        self.lattice_vectors = np.array([[0.,1.], [np.sqrt(3.)/2.,-.5]])
        #self.lattice_vectors = np.array([[1.,0.], [-.5, np.sqrt(3.)/2.]])
        #self.K_points = [np.array([np.pi*4./3., 0.]), np.array([np.pi*4./6., np.pi*2./np.sqrt(3.)])]
        self.K_points = [np.array([np.pi*4./3.,np.pi*4./np.sqrt(3.)]), np.array([np.pi*2./3,np.pi*2./np.sqrt(3.)])]
        # NN/NNN hopping vectors
        # Maciek:
        RB1 = np.array([1.,0.])
        RB2 = np.array([-1.,np.sqrt(3.)])/2.
        RB3 = np.array([-1.,-np.sqrt(3.)])/2.
        # Jarek:
        #RB1 = np.array([0.,-1.])
        #RB2 = np.array([np.sqrt(3.),1.])/2.
        #RB3 = np.array([-np.sqrt(3.),1.])/2.
        #
        RA1 = RB1 - RB3
        RA2 = RB2 - RB3
        RA3 = RB2 - RB1
        RA4 = RB3 - RB1
        RA5 = RB3 - RB2
        RA6 = RB1 - RB2
        self.hoppingsMX = [RB1, RB2, RB3]
        self.hoppingsMM = [RA1, RA2, RA3, RA4, RA5, RA6]
        #
        K = self.K_points[0][1]
        M = K*3./2
        G = K*np.sqrt(3.)/2.
        dk = (M+G)/100.  # magic number to get exactly 101 points at the path
        self.critical_points = [(r'$\Gamma$', 0.), ('K', K), ('M', M), (r'$\Gamma$', M+G)]
        k_GK = [[0., y] for y in np.arange(0, K, dk)] # k varying from Gamma to K point within the BZ
        k_KM = [[0., y] for y in np.arange(K, M, dk)] # k varying from K to M point within the BZ
        k_MG = [[x, M]  for x in np.linspace(0, G, num=int(G/dk), endpoint=True)] # k varying from M to Gamma point within the BZ
        if BZ_path is not None:
            self.BZ_path = BZ_path
            self.BZ_loaded_externally = True
        else:
            self.BZ_path = np.concatenate((k_GK, k_KM, k_MG)) # full path within the BZ
            self.BZ_loaded_externally = False


class Flake:
    """ Collection of methods for creating flake lattice.

    Attributes
    ----------
    m : FlakeParameters
        flake material parameters
    side_size : int
        number of atomic nodes at the flake edge (e.g. hexagon)
    dot_size : float
        supposed radius of quantum dot created in the flake,
        using when calculating dot occupation in calculate_spindenisities() method
    no_of_nodes : int
        total number of nodes
    nodes : ndarray of floats (no_of_nodes, 2)
        array containing pairs of x,y-coordinates for each node
    material : ndarray of int (no_of_nodes)
        array determining node's material id: 0 or 1
    no_of_links : int
        no of all links between the (nearest) nodes within the flake
    links: ndarray of int (no_of_links, 2)
        links between the nodes, described by pair of indices (as in the nodes array)
    potetntial : ndarray of floats (no_of_nodes)
        electrostatic potential at each node within flake
    electric_field : ndarray of floats
        electric field within flake
    bmag : float
        magnetic field within flake
    """
    def __init__(self, lattice_constant, interalayer_distance, side_size, shape='hexagonal', find_neighbours=True):
        self.lattice_constant = lattice_constant
        self.interalayer_distance = interalayer_distance
        self.side_size = side_size
        self.lattice_shape = shape
        self.dot_size = 0.
        assert self.lattice_shape == 'hexagonal' or self.lattice_shape == 'rhombus', "unknown lattice shape"
        if self.lattice_shape == 'hexagonal':
            self.no_of_nodes = (2*self.side_size[0]+self.side_size[0]-2)*(self.side_size[0]-1)+2*self.side_size[0]-1
            self.nodes, self.nodestype = self.create_hexagonal_flake()
        elif self.lattice_shape == 'rhombus':
            self.no_of_nodes = 2*self.side_size[0]*self.side_size[1]
            self.nodes, self.nodestype = self.create_rhombus_flake()
        if find_neighbours:
            self.no_of_links, self.links, self.linkstype = self.find_neighbours()
        self.shift_k = None
        self.potential = None
        self.electric_field = None
        self.bmag = None

    def create_hexagonal_flake(self):
        """ Create array that define the modeled flake of hexagonal shape.

        Defined flake consists hexagonal (triangular) lattice of M atoms (of TMDC) 
        together with translated X_2 dimmers lattice

        Returns
        -------
        nodes : ndarray
            (centered) nodes' coordinates for the created hexagonal flake
        nodetype: ndarray
            with respective type of nodes (1 = M, 2 = X2)
        """
        self.no_of_nodes *= 3 * 2 * 2  # elongated flake
        nodes = np.zeros((self.no_of_nodes,3))
        nodestype = np.zeros(self.no_of_nodes, dtype=np.int8)
        self.dot_size = self.side_size[0]*self.lattice_constant/2
        x = 0
        y = 0
        ir = int(self.side_size[0]*2.2)  # elongated flake
        k = 0
        for j in range(1,2*self.side_size[0]):
            for _ in range(1,ir+1):
                # M up
                nodes[k,0] = x
                nodes[k,1] = y
                nodes[k,2] = self.interalayer_distance/2.
                nodestype[k] = 1  # M
                k = k + 1
                # X2 up
                nodes[k,0] = x + .5*self.lattice_constant
                nodes[k,1] = y + np.sqrt(3.)/6.*self.lattice_constant
                nodes[k,2] = self.interalayer_distance/2.
                nodestype[k] = 2  # X2
                k = k + 1
                # M down
                nodes[k,0] = x + .5*self.lattice_constant
                nodes[k,1] = y + np.sqrt(3.)/6.*self.lattice_constant
                nodes[k,2] = -self.interalayer_distance/2.
                nodestype[k] = 3  # M
                k = k + 1
                # X2 down
                nodes[k,0] = x
                nodes[k,1] = y
                nodes[k,2] = -self.interalayer_distance/2.
                nodestype[k] = 4  # X2
                x = x + self.lattice_constant
                k = k + 1
            if j < self.side_size[0]:
                x = x - ir*self.lattice_constant - .5*self.lattice_constant
                ir = ir + 1
            else:
                x = x - ir*self.lattice_constant + .5*self.lattice_constant
                ir = ir - 1
            y = y + np.sqrt(3.)/2.*self.lattice_constant

        self.no_of_nodes = k
        nodes = nodes[:k,:]
        nodestype = nodestype[:k]
        print("number of flake nodes = " + str(self.no_of_nodes))
        return self.center_flake(nodes), nodestype

    def create_rhombus_flake(self):
        """ Create array that define the modeled flake of rhombus shape.

        Defined flake consists hexagonal (triangular) lattice of M atoms (of TMDC) 
        together with translated X_2 dimmers lattice

        Returns
        -------
        nodes : ndarray
            (centered) nodes' coordinates for the created rhombus flake
        nodetype: ndarray
            with respective type of nodes (1 = M, 2 = X2)
        """
        self.no_of_nodes *= 2  # elongated flake
        nodes = np.zeros((self.no_of_nodes,3))
        nodestype = np.zeros(self.no_of_nodes, dtype=np.int8)
        self.dot_size = np.amax(self.side_size)*self.lattice_constant/2
        x = 0
        y = 0
        k = 0
        '''
        for _ in range(self.side_size[0]):
            for _ in range(self.side_size[1]):
                # X2
                nodes[k,0] = x
                nodes[k,1] = y
                nodestype[k] = 2  # X2
                k = k + 1
                # M
                nodes[k,0] = x + np.sqrt(3.)/6.*self.lattice_constant
                nodes[k,1] = y - .5*self.lattice_constant
                nodestype[k] = 1  # M
                y = y + self.lattice_constant
                k = k + 1
            x += np.sqrt(3.)/2.*self.lattice_constant 
            y -= self.lattice_constant*(self.side_size[1]+.5)
        '''
        for _ in range(self.side_size[0]):
            for _ in range(self.side_size[1]):
                # M up
                nodes[k,0] = x
                nodes[k,1] = y
                nodes[k,2] = self.interalayer_distance/2.
                nodestype[k] = 1  # M
                k = k + 1
                # X2 up
                nodes[k,0] = x + np.sqrt(3.)/3.*self.lattice_constant
                nodes[k,1] = y
                nodes[k,2] = self.interalayer_distance/2.
                nodestype[k] = 2  # X2
                k = k + 1
                # M down
                nodes[k,0] = x + np.sqrt(3.)/3.*self.lattice_constant
                nodes[k,1] = y
                nodes[k,2] = -self.interalayer_distance/2.
                nodestype[k] = 3  # M
                k = k + 1
                # X2 down
                nodes[k,0] = x
                nodes[k,1] = y
                nodes[k,2] = -self.interalayer_distance/2.
                nodestype[k] = 4  # X2
                y = y + self.lattice_constant
                k = k + 1
            x += np.sqrt(3.)/2.*self.lattice_constant 
            y -= self.lattice_constant*(self.side_size[1]+.5)    
        self.no_of_nodes = k
        nodes = nodes[:k,:]
        nodestype = nodestype[:k]
        print("number of flake nodes = " + str(self.no_of_nodes))
        return self.center_flake(nodes), nodestype

    def set_lattice_parameters(self, lattice_parameters):
        self.l = lattice_parameters
        self.lattice_vectors = self.l.lattice_vectors*self.lattice_constant
        self.K_points = [p/self.lattice_constant for p in self.l.K_points]
        self.Q_points = [p/self.lattice_constant for p in self.l.Q_points]
        #self.G_points = [p/self.lattice_constant for p in self.l.G_points]
        #self.hoppingsMM = [h*self.lattice_constant/np.sqrt(3.) for h in self.l.hoppingsMM]
        #self.hoppingsMX = [h*self.lattice_constant/np.sqrt(3.) for h in self.l.hoppingsMX]
        #self.hoppingsMM_nn = [h*self.lattice_constant/np.sqrt(3.) for h in self.l.hoppingsMM_nn]
        self.BZ_path = self.l.BZ_path/self.lattice_constant
        self.critical_points = [(p[0], p[1]/self.lattice_constant) for p in self.l.critical_points]

    def create_reciprocal_lattice(self):
        B = inv(self.lattice_vectors.T)*np.pi*2.
        b1 = B[1]
        b2 = B[0]
        nodes_k = np.zeros((self.side_size[0]*self.side_size[1],2))
        nodes_kij = np.zeros((self.side_size[0]*self.side_size[1],2), dtype=np.int16)
        plaquettes = []
        for m1 in range(self.side_size[0]):
            for m2 in range(self.side_size[1]):
                nodes_k[m1*self.side_size[1]+m2, 0] = m1/self.side_size[0]*b1[0] + m2/self.side_size[1]*b2[0]
                nodes_k[m1*self.side_size[1]+m2, 1] = m1/self.side_size[0]*b1[1] + m2/self.side_size[1]*b2[1]
                nodes_kij[m1*self.side_size[1]+m2, 0] = m1
                nodes_kij[m1*self.side_size[1]+m2, 1] = m2
                # save indices that define subsequent plaquettes 
                if m2 < self.side_size[1]-1 and m1 < self.side_size[0]-1:
                    i_node = m1*self.side_size[1]+m2
                    plaquettes.append([i_node, i_node+1, i_node+self.side_size[1]+1, i_node+self.side_size[1]])       
        self.nodes_k = nodes_k
        self.nodes_kij = nodes_kij
        self.lattice_vectors_k = np.array([b1/self.side_size[0],b2/self.side_size[1]])
        self.lattice_vectors_G = np.array([b1, b2])
        self.lattice_vectors_G_shift = (b1+b2)/2.
        self.plaquettes = np.array(plaquettes)
        
    def modulo_BZ(self, k):
        return ((k + self.lattice_vectors_G_shift).dot(self.lattice_vectors[::-1].T)/np.pi/2. % 1).dot(self.lattice_vectors_G) - self.lattice_vectors_G_shift

    def center_reciprocal_lattice(self):
        self.shift_k = np.array([self.K_points[0][0]+self.K_points[1][0],self.K_points[0][1]+self.K_points[1][1]])/2.
        #shift_k = [np.mean(self.K_points[:,0]), np.mean(self.K_points[:,1])]
        self.nodes_k -= self.shift_k
        self.K_points = [p-self.shift_k for p in self.K_points]
        #self.G_points = [p-self.shift_k for p in self.G_points]
        
    def set_BZ_path(self, BZ_path):
        self.BZ_path = BZ_path
        
    def center_flake(self, nodes):
        """Flake position centering.
        
        Translates flake nodes positions in a way that the flake center is located at (0,0)
        
        Returns
        -------
        nodes : ndarray
            nodes coordinates for the centered flake
        """
        # centering the flake
        xmin = np.amin(nodes[:,0])
        xmax = np.amax(nodes[:,0])
        ymin = np.amin(nodes[:,1])
        ymax = np.amax(nodes[:,1])
        nodes[:,0] = nodes[:,0] - (xmax+xmin)/2
        nodes[:,1] = nodes[:,1] - (ymax+ymin)/2
        return nodes

    def find_neighbours(self):
        """ Collect all nearest neighbours within flake
                
        Returns
        -------
        no_of_links : int
            no of all founded links between the (nearest) nodes
        links: ndarray
            founded links between the nodes, described by pair of indices (as in the nodes array)
        linktype: ndarray
            with respective type of links (1 = M-M, 2 = X2-X2, 3 = M->X2, or 4 = X2->M)
        """
        # links between the nodes
        no_of_links = 0
        for i in range(self.no_of_nodes):
            x = self.nodes[i,0]
            y = self.nodes[i,1]
            for j in range(i+1,self.no_of_nodes):
                x1 = self.nodes[j,0]
                y1 = self.nodes[j,1]
                R = (x - x1)**2 + (y - y1)**2
                if R < 1.1*self.lattice_constant**2: no_of_links += 1
        links = np.zeros((no_of_links,2), dtype=int)
        linkstype = np.zeros(no_of_links, dtype=np.int8)
        k=0
        for i in range(self.no_of_nodes):
            x = self.nodes[i,0]
            y = self.nodes[i,1]
            ntype = self.nodestype[i]
            for j in range(i+1,self.no_of_nodes):
                x1 = self.nodes[j,0]
                y1 = self.nodes[j,1]
                ntype1 = self.nodestype[j]
                R = (x - x1)**2 + (y - y1)**2
                if R < 1.1*self.lattice_constant**2:
                    links[k,0] = i
                    links[k,1] = j
                    if ntype == ntype1:
                        if ntype == 1: linkstype[k] = 1  # M-M 
                        else: linkstype[k] = 2  # X2-X2
                    else:
                        if ntype == 1: linkstype[k] = 3  # M->X2 
                        else: linkstype[k] = 4  # X2->M
                    k = k + 1
        return no_of_links, links, linkstype

    def set_bfield(self, induction_in_teslas):
        self.bmag = induction_in_teslas/au.Bh
        
    def set_efield(self, field_in_mV_nm):
        self.fel = field_in_mV_nm*au.Ah/au.Eh


class Planewaves:

    def __init__(self, flake, model):
        self.flake = flake
        self.subspaces = [self.flake.nodes_k]
        self.model = model
        self.m = self.model.m

    def select_subspace(self, special_points, radius):
        radius *= au.Ah
        subspaces = []
        subspaces_ij = []
        total_size = 0
        for sp in special_points: 
            indices = [i for i, nk in enumerate(self.flake.nodes_k) if (nk[0]-sp[0])**2+(nk[1]-sp[1])**2 < radius**2]
            subspaces.append(self.flake.nodes_k[indices])
            subspaces_ij.append(self.flake.nodes_kij[indices])
            total_size += len(indices)
        self.subspaces = subspaces
        self.subspaces_ij = subspaces_ij
        self.subspaces_total_size = total_size
        
    def select_subspace_half(self, delta=1.e-7):
        subspaces = []
        subspaces_ij = []
        total_size = 0
        # K
        test = halfspace(-np.sqrt(3.), self.flake.K_points[0][1]*3., delta)
        indices = [i for i, nk in enumerate(self.flake.nodes_k) if test(nk[0], nk[1])]
        subspaces.append(self.flake.nodes_k[indices])
        subspaces_ij.append(self.flake.nodes_kij[indices])
        total_size += len(indices)
        # Kp
        test = halfspace(-np.sqrt(3.), -self.flake.K_points[0][1]*3., -delta)        
        indices = [i for i, nk in enumerate(self.flake.nodes_k) if test(-nk[0], -nk[1])]
        subspaces.append(self.flake.nodes_k[indices])
        subspaces_ij.append(self.flake.nodes_kij[indices])
        total_size += len(indices)
        #
        self.subspaces = subspaces
        self.subspaces_ij = subspaces_ij
        self.subspaces_total_size = total_size

    def build_plane_hamiltonian(self, kx, ky):
        return self.model.build_tb_hamiltonian_new(kx,ky)

    def build_basis(self, no_of_bands=[14,15], energy_offset=0.):
        self.no_of_bands = no_of_bands
        basis_energies = []
        basis_amplitudes = []
        basis_spins = []
        basis_k = []
        for subspace in self.subspaces:
            for [kx,ky] in subspace:
                w, v = eigh(self.build_plane_hamiltonian(kx, ky))
                for no in self.no_of_bands:  # CB minimum
                    basis_energies.append(w[no])
                    basis_amplitudes.append(v[:,no])
                    pauli_exp = abs2(v[:,no]).reshape(2,-1).sum(axis=1)
                    basis_spins.append(pauli_exp[0]-pauli_exp[1])
                basis_k.append([kx,ky])
        self.basis_energies = np.array(basis_energies) - energy_offset
        self.basis_amplitudes = np.array(basis_amplitudes)
        self.basis_spins = np.array(basis_spins)
        self.basis_k = np.array(basis_k)
        print("number of basis states = " + str(self.subspaces_total_size))

    def solve_BZ_path(self):
        return np.array([eigh(self.build_plane_hamiltonian(k[0],k[1]), eigvals_only=True) for k in self.model.BZ_path])

    def potential_elements(self, sign=1):
        dist_ij = OrderedSet()
        pot_elements_ij = np.zeros((self.subspaces_total_size,self.subspaces_total_size), dtype=np.int32)
        ss = self.flake.side_size[1]
        i1=0
        for subspace_ij in self.subspaces_ij:
            for [i,j] in subspace_ij:
                j1=0
                for subspace_kl in self.subspaces_ij:
                    for [k,l] in subspace_kl:
                        pot_elements_ij[i1,j1] = dist_ij.add((k-i+ss)*ss*2+l-j+ss)  # encode (k-q)-point
                        j1+=1
                i1+=1       
        #
        print("number of potential matrix elements to calculate = " + str(len(dist_ij)))
        # collect nodes of different types (in unit cell)
        unique_nodes = np.unique(self.flake.nodestype)
        nodes = [self.flake.nodes[self.flake.nodestype==unique_node] for unique_node in unique_nodes]
        potential = [self.flake.potential[self.flake.nodestype==unique_node] for unique_node in unique_nodes]
        #
        aux = np.zeros((unique_nodes.size, len(dist_ij)), dtype=np.complex128)
        kq = np.zeros((len(dist_ij),2))
        b = self.flake.lattice_vectors_k
        # calculate elements:
        for i, d in enumerate(dist_ij):
            # decode (k-q)-point
            ki = int(d/ss/2)
            lj = d - ki*ss*2 - ss
            ki -= ss
            kqx = ki*b[0,0] + lj*b[1,0]
            kqy = ki*b[0,1] + lj*b[1,1]
            kq[i] = [kqx,kqy]
            for l in range(unique_nodes.size):
                aux[l,i] = np.inner(np.exp(1.j*(nodes[l][:,0]*kqx+nodes[l][:,1]*kqy)), potential[l].flatten())
        aux = aux/self.flake.no_of_nodes*unique_nodes.size*sign
        # fill matrix
        pot_elements = np.zeros((unique_nodes.size, self.subspaces_total_size, self.subspaces_total_size), dtype=np.complex128)
        for i1 in range(self.subspaces_total_size):
            for j1 in range(self.subspaces_total_size):
                pot_elements[:,i1,j1] = aux[:, pot_elements_ij[i1,j1]]
        self.pot_elements = pot_elements  # indexed by [q,k]
        self.elements_kmq = [np.concatenate((kq, aux[l,:,np.newaxis]), axis=1) for l in range(unique_nodes.size)]  # for k-q distance

    def build_hamiltonian(self, include_diagonal=True):
        dim_k = self.subspaces_total_size
        dim_r = len(self.no_of_bands)
        ham_qk = np.zeros((dim_k*dim_r, dim_k*dim_r), dtype=np.complex128)
        if include_diagonal: np.fill_diagonal(ham_qk, self.basis_energies)
        pMu = self.pot_elements[0]
        pX2u = self.pot_elements[1]
        pMd = self.pot_elements[2]
        pX2d = self.pot_elements[3]
        #pot_elements_lqk = np.stack((pM,pM,pM,pX2,pX2,pX2,pX2,pX2,pX2,pM,pM,
        #                             pM,pM,pM,pX2,pX2,pX2,pX2,pX2,pX2,pM,pM))
        #pot_elements_lqk = np.stack((pM,pM,pM,pM,pM,pM))
        pot_elements_lqk = np.stack((pMu,pMu,pMu,pX2u,pX2u,pX2u,pMu,pMu,pX2u,pX2u,pX2u,
                                     pMd,pMd,pMd,pX2d,pX2d,pX2d,pMd,pMd,pX2d,pX2d,pX2d,
                                     pMu,pMu,pMu,pX2u,pX2u,pX2u,pMu,pMu,pX2u,pX2u,pX2u,
                                     pMd,pMd,pMd,pX2d,pX2d,pX2d,pMd,pMd,pX2d,pX2d,pX2d))
        for q in range(dim_k):
            for p in range(dim_r):
                for k in range(dim_k):
                    for r in range(dim_r):
                        qp = q*dim_r+p
                        kr = k*dim_r+r
                        ham_qk[qp,kr] += np.vdot(self.basis_amplitudes[qp], self.basis_amplitudes[kr]*pot_elements_lqk[:,q,k])
        return ham_qk
    
    def sum_elements(self, elements, k_point, radius):
        radius *= au.Ah
        indices_up = np.array([i*2 for i, nk in enumerate(self.basis_k) if (nk[0]-k_point[0])**2+(nk[1]-k_point[1])**2 < radius**2])
        indices_down = indices_up + 1
        e_uu = np.sum(elements[indices_up, 0::2], axis=0)  # from up to up
        e_ud = np.sum(elements[indices_up, 1::2], axis=0)  # should be zero
        e_du = np.sum(elements[indices_down, 0::2], axis=0)
        e_dd = np.sum(elements[indices_down, 1::2], axis=0)
        return [e_uu, e_ud, e_du, e_dd]


class LoadPotential:

    def __init__(self, device):
        self.dev = device
        self.nx = self.dev.save_grid_points[0]
        self.ny = self.dev.save_grid_points[1]
        self.nz = self.dev.save_grid_points[2]
        self.fi_total = np.zeros((self.nx+1, self.ny+1, self.nz+1))
        self.fi_electron = np.zeros_like(self.fi_total)
        self.fi_flake = np.zeros((self.nx+1, self.ny+1))
        self.efield_flake = np.zeros_like(self.fi_flake)
        self.err = np.array([0.])
    
    def get_values_at_points(self, values_on_grid, points):
        f = interpolate.interp2d(self.dev.save_grid_x_space - self.dev.box_x_size/2., 
                                 self.dev.save_grid_y_space - self.dev.box_y_size/2., 
                                 values_on_grid, kind='cubic')
        return np.array([f(p[1], p[0])[0] for p in points])  # reversed order -- ad hoc fix

    def get_potential_at_points(self, points):
        return self.get_values_at_points(self.fi_flake, points)   

    def get_electric_field_at_points(self, points):
        return self.get_values_at_points(self.efield_flake, points)  

    def get_potential_at_flake(self, flake):
        return self.get_potential_at_points(flake.nodes)

    def get_electric_field_at_flake(self, flake):
        return self.get_electric_field_at_points(flake.nodes)

    def load_confinement(self, directory, suffix=None):
        with open(os.path.join(directory, 'fi_total'+suffix+'.npy'), 'rb') as f:
            self.fi_total = np.load(f)
        with open(os.path.join(directory, 'fi_electron'+suffix+'.npy'), 'rb') as f:
            self.fi_electron = np.load(f)
        with open(os.path.join(directory, 'fi_flake'+suffix+'.npy'), 'rb') as f:
            self.fi_flake = np.load(f)
        with open(os.path.join(directory, 'efield_flake'+suffix+'.npy'), 'rb') as f:
            self.efield_flake = np.load(f)


class FlakeModel:
    """ Build flake (heterostructure) tight-bindng model

    Attributes
    ----------
    flake : Flake
        flake stucture
    parameters : TMDCmaterial
        flake material
    parameters1 : TMDCmaterial
        2nd flake material in case of heterostructure
    """
    def __init__(self, flake, parameters, **kwargs):
        self.flake = flake
        self.m = parameters
        if self.flake.potential is None:
            self.potential = np.zeros(self.flake.no_of_nodes)
            warnings.warn("Flake potential not defined, assuming zero potential.")
        else:
            self.potential = self.flake.potential
        if self.flake.electric_field is None:
            self.electric_field = np.zeros(self.flake.no_of_nodes)
            warnings.warn("Flake electric field not defined, assuming zero field.")
        else:
            self.electric_field = self.flake.electric_field
        if self.flake.bmag is None:
            self.bmag = 0.
            warnings.warn("Flake magnetic induction not defined, assuming zero induction.")
        else:
            self.bmag = self.flake.bmag

    def bfield(self, x,y):
        # static field
        return self.bmag

    def zeeman(self, x,y):
        """
        B-feild: 1 T = 1/Bh a.u.
        Bohr magneton: 1 uB = 0.5 a.u
        g-factor: g = -0.44 (GaAs)
        H_zeeman = g*uB*B/2
        """
        return 2.*0.5*self.bfield(x,y)/2.  # assumed g-factor = 2.

    def apoty(self, x,y):
        """
        Landau gauge: A = [0,A_y,0]
        y-component of the vector potential A_y(x,y) = int{ dx B_z(x,y) }
        """
        return x*self.bmag

    def hopping_matrix_(self, x, y, x1, y1, linkstype):
        """
        create 22x22 hopping matrix that represents hopping integral within the tight-binding model
        
        orbitals basis = {Dm2, D0, Dp2, PEm1, PE0, PEp1, POm1, PO0, POp1, Dm1, Dp1}

        """
        m = self.m 
        hh_m=np.zeros((m.dim,m.dim), dtype=np.complex128)
        if linkstype == 1:
        # M-M hoppings:
            L = (x1-x)/m.a0
            M = (y1-y)/m.a0
            hh_m[0,0] = (3.*m.Vd2s + 4.*m.Vd2p + m.Vd2d)/8. 
            hh_m[0,1] = (np.sqrt(3./2.)/4.)*(1.j*M+L)**2*(m.Vd2d-m.Vd2s)   
            hh_m[0,2] = (1.j*M+L)**4*(3.*m.Vd2s - 4.*m.Vd2p + m.Vd2d)/8.
            hh_m[1,0] = (np.sqrt(3./2.)/4.)*(-1.j*M+L)**2*(m.Vd2d-m.Vd2s)
            hh_m[1,1] = (3.*m.Vd2d+m.Vd2s)/4.
            hh_m[1,2] = (np.sqrt(3./2.)/4.)*(1.j*M+L)**2*(m.Vd2d-m.Vd2s)
            hh_m[2,0] = (-1.j*M+L)**4*(3.*m.Vd2s - 4.*m.Vd2p + m.Vd2d)/8.  
            hh_m[2,1] = (np.sqrt(3./2.)/4.)*(-1.j*M+L)**2*(m.Vd2d-m.Vd2s) 
            hh_m[2,2] = (3.*m.Vd2s + 4.*m.Vd2p + m.Vd2d)/8.
            if m.odd:
                # odd block
                hh_m[9,9] = (m.Vd2p+m.Vd2d)/2.
                hh_m[9,10] = -(1.j*M+L)**2*(m.Vd2p-m.Vd2d)/2.
                hh_m[10,9] = -(-1.j*M+L)**2*(m.Vd2p-m.Vd2d)/2.
                hh_m[10,10] = (m.Vd2p+m.Vd2d)/2.
        elif linkstype == 2:
        # X2-X2 hoppings
            L = (x1-x)/m.a0
            M = (y1-y)/m.a0
            hh_m[3,3] = (m.Vp2s+m.Vp2p)/2.
            hh_m[3,4] = 0. 
            hh_m[3,5] = -1.*(1.j*M+L)**2*(m.Vp2s-m.Vp2p)/2.  # -1 = Maciek correction
            hh_m[4,3] = 0.
            hh_m[4,4] = m.Vp2p
            hh_m[4,5] = 0. 
            hh_m[5,3] = -1.*(-1.j*M+L)**2*(m.Vp2s-m.Vp2p)/2.  # -1 = Maciek correction
            hh_m[5,4] = 0. 
            hh_m[5,5] = (m.Vp2s+m.Vp2p)/2. 
            if m.odd:
                # odd block
                hh_m[6,6] = (m.Vp2s+m.Vp2p)/2.
                hh_m[6,7] = 0. 
                hh_m[6,8] = -1.*(1.j*M+L)**2*(m.Vp2s-m.Vp2p)/2.  # -1 = Maciek correction
                hh_m[7,6] = 0.
                hh_m[7,7] = m.Vp2p
                hh_m[7,8] = 0. 
                hh_m[8,6] = -1.*(-1.j*M+L)**2*(m.Vp2s-m.Vp2p)/2.  # -1 = Maciek correction
                hh_m[8,7] = 0. 
                hh_m[8,8] = (m.Vp2s+m.Vp2p)/2.
        else:
        # M-X2 or X2-M hoppings
            L = (x1-x)/m.dd
            M = (y1-y)/m.dd
            if linkstype == 4:
            # X2-M hoppings, T(-R) = T^\dag(R)
                L *= -1
                M *= -1
            hh_m[0,3] = (1.j*M+L)*(np.sqrt(3.)/2.*m.Vdps*((m.dp/m.dd)**2-1.)-m.Vdpp*((m.dp/m.dd)**2+1.))/np.sqrt(2.)
            hh_m[0,4] = -(1.j*M+L)**2*(m.dp/m.dd)*(np.sqrt(3.)*m.Vdps-2.*m.Vdpp)/2.
            hh_m[0,5] = -(1.j*M+L)**3*(np.sqrt(3.)/2.*m.Vdps-m.Vdpp)/np.sqrt(2.)*(-1.)  # (-1) = Maciek correction
            hh_m[1,3] = -(-1.j*M+L)*((3.*(m.dp/m.dd)**2-1.)*m.Vdps-2.*np.sqrt(3.)*(m.dp/m.dd)**2*m.Vdpp)/2.
            hh_m[1,4] = -(m.dp/m.dd)*((3.*(m.dp/m.dd)**2-1.)*m.Vdps-2.*np.sqrt(3.)*((m.dp/m.dd)**2-1.)*m.Vdpp)/np.sqrt(2.)
            hh_m[1,5] = -( 1.j*M+L)*((3.*(m.dp/m.dd)**2-1.)*m.Vdps-2.*np.sqrt(3.)*(m.dp/m.dd)**2*m.Vdpp)/2.*(-1.)  # (-1) = Maciek correction 
            hh_m[2,3] = -(-1.j*M+L)**3*(np.sqrt(3.)/2.*m.Vdps-m.Vdpp)/np.sqrt(2.)
            hh_m[2,4] = -(-1.j*M+L)**2*(m.dp/m.dd)*(np.sqrt(3.)*m.Vdps-2.*m.Vdpp)/2. 
            hh_m[2,5] = (-1.j*M+L)*(np.sqrt(3.)/2.*m.Vdps*((m.dp/m.dd)**2-1.)-m.Vdpp*((m.dp/m.dd)**2+1.))/np.sqrt(2.)*(-1.)  # (-1) = Maciek correction
            if m.odd:
                # odd block
                hh_m[9,6] = -(m.dp/m.dd)*((L**2+M**2)*(np.sqrt(3.)*m.Vdps-2.*m.Vdpp)+2.*m.Vdpp)/np.sqrt(2.)
                hh_m[9,7] = -(1.j*M+L)*((m.dp/m.dd)**2*(np.sqrt(3.)*m.Vdps-2.*m.Vdpp)+m.Vdpp)
                hh_m[9,8] = -(m.dp/m.dd)*(1.j*M+L)**2*(np.sqrt(3.)*m.Vdps-2.*m.Vdpp)/np.sqrt(2.)*(-1.)  # (-1) = Maciek correction
                hh_m[10,6] = (m.dp/m.dd)*(-1.j*M+L)**2*(np.sqrt(3.)*m.Vdps-2.*m.Vdpp)/np.sqrt(2.)
                hh_m[10,7] = (-1.j*M+L)*((m.dp/m.dd)**2*(np.sqrt(3.)*m.Vdps-2.*m.Vdpp)+m.Vdpp)
                hh_m[10,8] = (m.dp/m.dd)*((L**2+M**2)*(np.sqrt(3.)*m.Vdps-2.*m.Vdpp)+2.*m.Vdpp)/np.sqrt(2.)*(-1.)  # (-1) = Maciek correction
            if linkstype == 4:    
            # X2-M hoppings, T(-R) = T^\dag(R)
                hh_m[3:6,0:3] = np.conjugate(hh_m[0:3,3:6]).transpose()
                hh_m[0:3,3:6] = 0.
                hh_m[6:9,9:11] = np.conjugate(hh_m[9:11,6:9]).transpose()
                hh_m[9:11,6:9] = 0.
        # spin-down block is the same:
        hh_m[m.dim12:,m.dim12:] = hh_m[:m.dim12,:m.dim12]
        # Rashba
        #ef = (self.electric_field[self.flake.links[il,0]]+self.electric_field[self.flake.links[il,1]])/2. 
        #hh_m[:m.dim12,m.dim12:] += m.rsba*ef*(1j*(y1-y) - (x1-x))/m.a0
        #hh_m[m.dim12:,:m.dim12] += m.rsba*ef*(1j*(y1-y) + (x1-x))/m.a0
        # Peierls
        peierls=(self.apoty(x1,y1)+self.apoty(x,y))*(y1-y)/2.
        return hh_m*np.exp(1.j*peierls)

    def hopping_matrix_6(self, x, y, x1, y1, linkstype):
        """
        create 6x6 hopping matrix that represents hopping integral within the tight-binding model
        """
        m = self.m 
        hh_m=np.zeros((m.dim,m.dim), dtype=np.complex128)
        # which of hopping vector R1...R6??
        # see e.g.: Phys. Rev. B 91, 155410 (2015) or Phys. Rev. B 88, 085433 (2013).
        # R1 / R6
        if abs(x - x1) < 1.e-8:
            if y1 > y: R=1
            if y1 < y: R=6
        # R2 / R3
        if abs((y1 - y) - 0.5 *m.a0) < 1.e-8:
            if x1 > x: R=2
            if x1 < x: R=3
        # R4 / R5
        if abs((y1 - y) + 0.5 *m.a0) < 1.e-8:
            if x1 > x: R=5
            if x1 < x: R=4
        if R == 1:
            hh_m[0,0] = m.t0;  hh_m[0,1] = m.t1;   hh_m[0,2] = m.t2
            hh_m[1,0] = -m.t1; hh_m[1,1] = m.t11;  hh_m[1,2] = m.t12
            hh_m[2,0] = m.t2;  hh_m[2,1] = -m.t12; hh_m[2,2] = m.t22
        if R == 6:
            hh_m[0,0] = m.t0;  hh_m[0,1] = -m.t1;  hh_m[0,2] = m.t2
            hh_m[1,0] = m.t1;  hh_m[1,1] = m.t11;  hh_m[1,2] = -m.t12
            hh_m[2,0] = m.t2;  hh_m[2,1] = m.t12;  hh_m[2,2] = m.t22
        if R == 2:
            hh_m[0,0] = m.t0;                            hh_m[0,1] = 0.5*m.t1 - np.sqrt(3.0)/2*m.t2;                       hh_m[0,2] = -np.sqrt(3.0)/2*m.t1 - 0.5*m.t2
            hh_m[1,0] = -0.5*m.t1 - np.sqrt(3.0)/2*m.t2; hh_m[1,1] = 0.25*m.t11 + 0.75*m.t22;                              hh_m[1,2] = -np.sqrt(3.0)/4*m.t11 - m.t12 + np.sqrt(3.0)/4.0*m.t22
            hh_m[2,0] = np.sqrt(3.0)/2*m.t1 - 0.5*m.t2;  hh_m[2,1] = -np.sqrt(3.0)/4*m.t11 + m.t12 + np.sqrt(3.0)/4*m.t22; hh_m[2,2] = 3.0/4.0*m.t11 + 1.0/4.0*m.t22
        if R == 4:
            hh_m[0,0] = m.t0;                            hh_m[0,1] = -0.5*m.t1 - np.sqrt(3.0)/2*m.t2;                      hh_m[0,2] = np.sqrt(3.0)/2*m.t1 - 0.5*m.t2
            hh_m[1,0] = 0.5*m.t1 - np.sqrt(3.0)/2*m.t2;  hh_m[1,1] = 0.25*m.t11 + 0.75*m.t22;                              hh_m[1,2] = -np.sqrt(3.0)/4*m.t11 + m.t12 + np.sqrt(3.0)/4.0*m.t22
            hh_m[2,0] = -np.sqrt(3.0)/2*m.t1 - 0.5*m.t2; hh_m[2,1] = -np.sqrt(3.0)/4*m.t11 - m.t12 + np.sqrt(3.0)/4*m.t22; hh_m[2,2] = 3.0/4.0*m.t11 + 1.0/4.0*m.t22
        if R == 3:
            hh_m[0,0] = m.t0;                            hh_m[0,1] = 0.5*m.t1 + np.sqrt(3.0)/2*m.t2;                       hh_m[0,2] = np.sqrt(3.0)/2*m.t1 - 0.5*m.t2
            hh_m[1,0] = -0.5*m.t1 + np.sqrt(3.0)/2*m.t2; hh_m[1,1] = 0.25*m.t11 + 0.75* m.t22;                             hh_m[1,2] = np.sqrt(3.0)/4*m.t11 - m.t12 - np.sqrt(3.0)/4.0*m.t22
            hh_m[2,0] = -np.sqrt(3.0)/2*m.t1 - 0.5*m.t2; hh_m[2,1] = np.sqrt(3.0)/4*m.t11 + m.t12 - np.sqrt(3.0)/4*m.t22;  hh_m[2,2] = 3.0/4.0*m.t11 + 1.0/4.0*m.t22
        if R == 5:
            hh_m[0,0] = m.t0;                            hh_m[0,1] = -0.5*m.t1 + np.sqrt(3.0)/2*m.t2;                      hh_m[0,2] = -np.sqrt(3.0)/2*m.t1 - 0.5*m.t2
            hh_m[1,0] = 0.5*m.t1 + np.sqrt(3.0)/2*m.t2;  hh_m[1,1] = 0.25*m.t11 + 0.75*m.t22;                              hh_m[1,2] = np.sqrt(3.0)/4*m.t11 + m.t12 - np.sqrt(3.0)/4.0*m.t22
            hh_m[2,0] = np.sqrt(3.0)/2*m.t1 - 0.5*m.t2;  hh_m[2,1] = np.sqrt(3.0)/4*m.t11 - m.t12 - np.sqrt(3.0)/4*m.t22;  hh_m[2,2] = 3.0/4.0*m.t11 + 1.0/4.0*m.t22
        # spin-down block is the same:
        hh_m[m.dim12:,m.dim12:] = hh_m[:m.dim12,:m.dim12]
        # Rashba
        #ef = (self.electric_field[self.flake.links[il,0]]+self.electric_field[self.flake.links[il,1]])/2. 
        #hh_m[:m.dim12,m.dim12:] += m.rsba*ef*(1.j*(y1-y) - (x1-x))/m.a0
        #hh_m[m.dim12:,:m.dim12] += m.rsba*ef*(1.j*(y1-y) + (x1-x))/m.a0
        # Peierls
        peierls=(self.apoty(x1,y1)+self.apoty(x,y))*(y1-y)/2.
        return hh_m*np.exp(1.j*peierls)

    def hopping_matrix(self, il):
        x = self.flake.nodes[self.flake.links[il,0],0]
        y = self.flake.nodes[self.flake.links[il,0],1]
        x1 = self.flake.nodes[self.flake.links[il,1],0]
        y1 = self.flake.nodes[self.flake.links[il,1],1]
        linkstype = self.flake.linkstype[il]
        return self.hopping_matrix_(x, y, x1, y1, linkstype)

    def full_tb_hamiltonian(self):
        """
        create Hamiltonian for the whole flake lattice
        """
        m = self.m
        dim = self.m.dim
        dim12 = self.m.dim12
        dim2 = self.m.dim2
        # dimmension of arrays reprezenting sparse hamiltonian matrix:
        N_nonzero = self.flake.no_of_nodes*(dim+16) + self.flake.no_of_links*dim2*2  # 16 = no of off-diagonal intrinsic SOI elements
        raw = np.zeros(N_nonzero, dtype=np.complex128)
        rw = np.zeros(N_nonzero, dtype=int)
        rk = np.zeros(N_nonzero, dtype=int)
        ip = 0
        for i in range(self.flake.no_of_links):
            id1 = self.flake.links[i,0]*dim
            id2 = self.flake.links[i,1]*dim
            hh_m = self.hopping_matrix(i)
            hh_m = np.reshape(hh_m,dim2)
            # upper triangular part
            raw[ip:ip+dim2] = hh_m
            for j in range(dim):
                for k in range(dim):
                    # row index
                    rw[ip] = j + id1
                    # column index
                    rk[ip] = k + id2
                    ip = ip + 1
            # lower triangular part
            raw[ip:ip+dim2] = np.conjugate(hh_m)
            for j in range(dim):
                for k in range(dim):
                    # row index
                    rw[ip] = k + id2
                    # column index
                    rk[ip] = j + id1
                    ip = ip + 1
        # diagonal part
        for i in range(self.flake.no_of_nodes):
            x = self.flake.nodes[i,0]
            y = self.flake.nodes[i,1]
            id = i*dim 
            for j in range(dim):
                raw[ip] = m.diag[j] - self.potential[i]
                # zeeman
                if j < dim12:  raw[ip] = raw[ip] + self.zeeman(x,y) + 1.e-8 # some energy to lift spin degeneracy
                if j >= dim12: raw[ip] = raw[ip] - self.zeeman(x,y) - 1.e-8
                # intrinistic spin-orbit coupling -- diagonal part:
                raw[ip] += m.l_diag[j]
                #
                rw[ip] = j + id
                rk[ip] = j + id
                ip = ip + 1
            # intrinistic spin-orbit coupling -- off-diagonal elements:       
            # upper triangle
            raw[ip] =  m.lm; rw[ip] = 0 + id; rk[ip] = 20 + id; ip = ip + 1
            raw[ip] =  m.lm; rw[ip] = 10 + id; rk[ip] = 13 + id; ip = ip + 1
            raw[ip] =  np.sqrt(3./2.)*m.lm; rw[ip] = 1 + id; rk[ip] = 21 + id; ip = ip + 1
            raw[ip] =  np.sqrt(3./2.)*m.lm; rw[ip] = 9 + id; rk[ip] = 12 + id; ip = ip + 1
            raw[ip] =  m.lx2/np.sqrt(2.); rw[ip] = 3 + id; rk[ip] = 18 + id; ip = ip + 1
            raw[ip] =  m.lx2/np.sqrt(2.); rw[ip] = 4 + id; rk[ip] = 19 + id; ip = ip + 1            
            raw[ip] =  m.lx2/np.sqrt(2.); rw[ip] = 6 + id; rk[ip] = 15 + id; ip = ip + 1
            raw[ip] =  m.lx2/np.sqrt(2.); rw[ip] = 7 + id; rk[ip] = 16 + id; ip = ip + 1
            # lower triangle
            raw[ip] =  m.lm; rw[ip] = 20 + id; rk[ip] = 0 + id; ip = ip + 1
            raw[ip] =  m.lm; rw[ip] = 13 + id; rk[ip] = 10 + id; ip = ip + 1
            raw[ip] =  np.sqrt(3./2.)*m.lm; rw[ip] = 21 + id; rk[ip] = 1 + id; ip = ip + 1
            raw[ip] =  np.sqrt(3./2.)*m.lm; rw[ip] = 12 + id; rk[ip] = 9 + id; ip = ip + 1
            raw[ip] =  m.lx2/np.sqrt(2.); rw[ip] = 18 + id; rk[ip] = 3 + id; ip = ip + 1
            raw[ip] =  m.lx2/np.sqrt(2.); rw[ip] = 19 + id; rk[ip] = 4 + id; ip = ip + 1            
            raw[ip] =  m.lx2/np.sqrt(2.); rw[ip] = 15 + id; rk[ip] = 6 + id; ip = ip + 1
            raw[ip] =  m.lx2/np.sqrt(2.); rw[ip] = 16 + id; rk[ip] = 7 + id; ip = ip + 1            
            #
        # return sparse matrix in intel CSR format
        print("full tight-binding Hamiltonian size = " + str(self.flake.no_of_nodes*dim))
        return csr_matrix((raw, (rw, rk)), 
                            shape=(self.flake.no_of_nodes * dim, self.flake.no_of_nodes * dim), 
                            dtype=np.complex128)


class FlakeMethods:
    """ Build flake (heterostructure) tight-bindng model

    Attributes
    ----------
    flake : Flake
        flake stucture
    parameters : TMDCmaterial
        flake material
    nkpoints : integer
        Fourier grid size, nkp*2+1 x nkp*2+1
    """
    def __init__(self, flake, parameters, nkpoints=50, basis_k=None):
        self.m = parameters
        self.flake = flake
        self.nkpoints = nkpoints
        if basis_k is not None:
            self.basis_k = basis_k

    def calculate_spindensities(self, eigenvectors):
        """
        :return: spindensities[i-th state, spin-up/spin-down, j-th node], dot_occupation[i-th state]
        dot_occupation = state density inside the dot area: determines if state is localized within the dot or is edge type
        """
        dim = self.m.dim
        dim12 = self.m.dim12
        no_of_eigenstates = eigenvectors.shape[1]
        spindensities = np.zeros((no_of_eigenstates,2,self.flake.no_of_nodes))
        dot_occupation = np.zeros(no_of_eigenstates)
        for i in range(no_of_eigenstates):
            s = 0.
            for j in range(self.flake.no_of_nodes):
                for k in range(dim12):
                    spindensities[i,0,j] += abs2(eigenvectors[j*dim+k,i])
                    spindensities[i,1,j] += abs2(eigenvectors[j*dim+dim12+k,i])
                if (self.flake.nodes[j,0]**2+self.flake.nodes[j,1]**2) < self.flake.dot_size**2: 
                    s += spindensities[i,0,j] + spindensities[i,1,j]
            dot_occupation[i] = s
        self.spindensities = spindensities
        return self.spindensities, dot_occupation

    def calculate_densities_k(self, eigenvectors, calculate_spin_valley=False, calculate_real_states=False, every_n=1):
        no_of_eigenstates = eigenvectors.shape[1]
        densities = np.zeros((no_of_eigenstates, self.flake.no_of_nodes))
        nodes_k = np.concatenate([subspace for subspace in self.basis_k.subspaces])
        dim_r = len(self.basis_k.no_of_bands)
        nodes_kk = np.repeat(nodes_k, dim_r, axis=0)
        spin_valley = None
        k_max = None
        std_y = None
        states = None
        if calculate_spin_valley:
            spin_valley = np.zeros((no_of_eigenstates, 2))
            k_max = np.zeros((no_of_eigenstates, 2))
            std_y = np.zeros(no_of_eigenstates) 
        if calculate_real_states:
            states = np.zeros((no_of_eigenstates, self.flake.no_of_nodes*self.m.dim), dtype=np.complex128)   
        #
        k1 = self.basis_k.subspaces[0].shape[0]*dim_r  # K-valley range
        k2 = k1 + self.basis_k.subspaces[1].shape[0]*dim_r  # K'-valley range
        for i in range(no_of_eigenstates):
            j = 0
            jj= 0
            S = 0.
            K = 0.
            for [x,y,z], type in zip(self.flake.nodes[::every_n], self.flake.nodestype[::every_n]):
                exp_k = np.repeat(np.exp(1.j*(nodes_k[:,0]*x+nodes_k[:,1]*y)), dim_r)
                if calculate_spin_valley:
                    expAB_Ku = np.matmul(np.multiply(exp_k[0:k1], eigenvectors[0:k1,i]), self.basis_k.basis_amplitudes[0:k1])
                    expAB_Kd = np.matmul(np.multiply(exp_k[k1:k2+1], eigenvectors[k1:k2+1,i]), self.basis_k.basis_amplitudes[k1:k2+1])
                    if type == 1:
                        SKuu = np.sum(expAB_Ku[np.r_[0:22]])
                        SKud = np.sum(expAB_Kd[np.r_[0:22]])
                        SKdu = np.sum(expAB_Ku[np.r_[22:44]])
                        SKdd = np.sum(expAB_Kd[np.r_[22:44]])
                    else:
                        SKuu = np.sum(expAB_Ku[np.r_[0:22]])
                        SKud = np.sum(expAB_Kd[np.r_[0:22]])
                        SKdu = np.sum(expAB_Ku[np.r_[22:44]])
                        SKdd = np.sum(expAB_Kd[np.r_[22:44]])
                    densities[i,j] = abs2(SKuu)+abs2(SKud)+abs2(SKdu)+abs2(SKdd)
                    #densities[i,j] = abs2(SKuu+SKud+SKdu+SKdd)  # equivalent to abs2(z) from below
                    S += abs2(SKuu)+abs2(SKud)-abs2(SKdu)-abs2(SKdd)
                    K += abs2(SKuu)-abs2(SKud)+abs2(SKdu)-abs2(SKdd)
                else:
                    expAB = np.matmul(np.multiply(exp_k, eigenvectors[:,i]), self.basis_k.basis_amplitudes)
                    if type == 1:
                        z = np.sum(expAB[np.r_[0:5]])
                    else:
                        z = 0.
                    densities[i,j] = abs2(z)
                if calculate_real_states:
                    states[i,jj:jj+self.m.dim] = np.matmul(np.multiply(exp_k, eigenvectors[:,i]), self.basis_k.basis_amplitudes)
                j += every_n
                jj += every_n*self.m.dim
            if calculate_spin_valley:
                spin_valley[i] = [S,K]
                k_max[i] = nodes_kk[np.argmax(abs2(eigenvectors[:,i]))]
                std_y[i] = np.sqrt(np.mean(densities[i,:]*self.flake.nodes[:,1]*self.flake.nodes[:,1])
                                    -np.mean(densities[i,:]*self.flake.nodes[:,1])**2)
            print(str(i+1)+"/"+str(no_of_eigenstates))

        self.densities = densities/(self.flake.no_of_nodes/2)
        self.spin_valley = spin_valley
        if calculate_spin_valley:
            self.spin_valley = spin_valley/(self.flake.no_of_nodes/2)
        if calculate_real_states:
            states = states/np.sqrt(self.flake.no_of_nodes)
        self.k_max = k_max
        self.std_y = std_y
        return self.densities, self.spin_valley, self.k_max, self.std_y, states

    def save_densities(self, directory='./results/', filename='densities'):
        dict_to_save = {'densities':self.densities, 'nodes':self.flake.nodes}
        with open(os.path.join(directory, filename+'.pkl'), 'wb') as f:
            pickle.dump(dict_to_save, f)

    def calculate_spinindex(self, eigenvector):
        spindensity = abs2(eigenvector).reshape(-1, self.m.dim)
        return spindensity[:,:self.m.dim12].sum()-spindensity[:,self.m.dim12:].sum()

    def calculate_valleyindex(self, eigenvector):
        """
        # K=(4pi/3/a), k=2pi/a thus for k*1.5 we get K ranging -1...1
        """
        nkpoints = self.nkpoints
        phi = np.zeros((nkpoints*2+1,nkpoints*2+1), dtype=np.float64)
        f_utils.fourierdensity(eigenvector, phi,
                                self.flake.nodes,
                                self.m.a0, self.m.dim,
                                self.flake.no_of_nodes, nkpoints)
        k_indx = 0.
        norm = 0.
        for i in range(nkpoints*2+1):
            for k in range(nkpoints*2+1):
                if(i-nkpoints!=0 or k-nkpoints!=0):
                    x=np.abs(np.arctan2(float(k-nkpoints),float(i-nkpoints)))
                    if(x<=np.pi/6 or x>=np.pi*5./6.):
                        k_indx+=phi[i,k]*(i-nkpoints)/nkpoints*1.5
                        norm+=phi[i,k]
        return k_indx/norm, phi
    
    def calculate_Lz(self, flake_size, eigenvector):
        nodes_k = np.concatenate([subspace for subspace in self.basis_k.subspaces])
        dim_r = len(self.basis_k.no_of_bands)
        psi = np.zeros(self.flake.no_of_nodes*self.m.dim, dtype=np.complex128) 
        j = 0
        for [x,y], type in zip(self.flake.nodes, self.flake.nodestype):
            exp_k = np.repeat(np.exp(1.j*(nodes_k[:,0]*x+nodes_k[:,1]*y)), dim_r)
            psi[j:j+self.m.dim] = np.matmul(np.multiply(exp_k, eigenvector), self.basis_k.basis_amplitudes)
            j += self.m.dim
        psi = psi.reshape((self.flake.no_of_nodes,2,3))
        psi =  np.sum(psi, axis=-1)  # sum over orbitals
        psi =  np.sum(psi, axis=-1)  # sum over spins
        psi /= np.sqrt(self.flake.no_of_nodes/2.)
        x0 = 0.
        y0 = 0.
        for i in range(1,flake_size-1):
            for j in range(1,flake_size-1):
                n0 = i*flake_size*2 + j*2
                n0y = n0 + 2
                n0xu = n0y + flake_size*2
                psi0 = psi[n0]+psi[n0+1]
                y = (self.flake.nodes[n0,1]+self.flake.nodes[n0y,1])/2.
                x = (self.flake.nodes[n0,0]+self.flake.nodes[n0xu+1,0])/2.
                x0 += np.conjugate(psi0)*x*psi0
                y0 += np.conjugate(psi0)*y*psi0
        Lz = 0.
        for i in range(1,flake_size-1):
            for j in range(1,flake_size-1):
                n0 = i*flake_size*2 + j*2
                # y direction:
                n0y = n0 + 2
                # x direction:
                n0xu = n0y + flake_size*2
                n0xd = n0xu-2
                # derivatives
                psi0 = psi[n0]+psi[n0+1]
                psi1 = psi[n0y]+psi[n0y+1]
                psi2 = psi[n0xu]+psi[n0xu+1]
                psi10 = (psi1-psi0)/self.m.a0
                psi20 = (psi2-psi0)/self.m.a0
                psi21 = (psi2-psi1)/self.m.a0
                psiy = psi10
                psix1 = (psi20 - psiy/2.)/0.866
                psix2 = (psi21 + psiy/2.)/0.866
                # nodes
                y = (self.flake.nodes[n0,1]+self.flake.nodes[n0y,1])/2.-y0
                x = (self.flake.nodes[n0,0]+self.flake.nodes[n0xu+1,0])/2.-x0
                # Lz/hbar = i(yDx - xDy)
                Lz += np.conjugate(psi0)*1.j*(y*(psix1+psix2)/2. - x*psiy) 
        return Lz


class EigenSolver:

    def solve_eigenproblem_arpack(self, hamiltonian,
                                no_of_eigenvalues_to_find = 100,
                                energy_ref_level = 0.,
                                calculate_eigenvectors = True):
        # we utilize arpack routine for sparse hermitian matrices
        eigenvalues, eigenvectors = eigsh(hamiltonian,
                                        k = no_of_eigenvalues_to_find,
                                        sigma = energy_ref_level/au.Eh,
                                        return_eigenvectors = calculate_eigenvectors)
        # sort
        idx = eigenvalues.argsort()
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        return eigenvalues, eigenvectors

    def solve_eigenproblem_arpack_dense(self, hamiltonian,
                                subset_by_index = [0,9],
                                reverse = False,
                                calculate_eigenvectors = True):
        if reverse:
            subset_by_index = [hamiltonian.shape[0]-1-subset_by_index[1], 
                               hamiltonian.shape[0]-1-subset_by_index[0]]         
        # we utilize arpack routine for (dense) hermitian matrices
        eigenvalues, eigenvectors = eigh(hamiltonian,
                                         subset_by_index = subset_by_index,
                                         eigvals_only = (not calculate_eigenvectors))
        # sort
        idx = eigenvalues.argsort()
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        return eigenvalues, eigenvectors

    def solve_eigenproblem_feast(self, hamiltonian,
                                eigenvalues_subspace_size = 100,
                                energy_min = 0.,
                                energy_max = 1.,
                                comments=True):
        # we utilize FEAST routine for sparse hermitian matrices
        solver = feast.HSolver(hamiltonian, 
                                M0 = eigenvalues_subspace_size, 
                                Em = energy_min/au.Eh, 
                                Ex = energy_max/au.Eh, 
                                which = 0)
        debug = 1 if comments else 0                        
        eigenvalues, eigenvectors, M, info = solver.eigsh(debug=debug)
        if info == 0:
            print("FEAST sucess!", M, "eigenvalues found.")
        else:
            print("info =", info, "FEAST warning or failed!")
        return eigenvalues, eigenvectors


class PlottingOnFlake:
    def __init__(self, flake, directory=None):
        self.flake = flake
        if directory is not None:
            self.directory = os.path.join('./', directory)
            os.makedirs(directory, exist_ok=True)
        else:
            self.directory = './'
        self.pointsize = 2.

    def plot_flake_lattice(self, filename='flake_lattice.png', plot_links=True):
        _, ax = plt.subplots()
        ax.set_aspect(1.)
        ax.set_xlabel("x (nm)")
        ax.set_ylabel("y (nm)")
        node_color = ['blue','orange','blue','orange']
        node_colors = [node_color[nt-1] for nt in self.flake.nodestype]
        link_colors = ['black','green','brown','brown']
        ax.scatter(x=self.flake.nodes[:,0]*au.Ah, 
                   y=self.flake.nodes[:,1]*au.Ah, 
                   s=self.pointsize/2./self.flake.nodestype, 
                   c=node_colors)
        if plot_links:
            for link, linktype in zip(self.flake.links, self.flake.linkstype):
                ax.plot([self.flake.nodes[link[0],0]*au.Ah, self.flake.nodes[link[1],0]*au.Ah],
                        [self.flake.nodes[link[0],1]*au.Ah, self.flake.nodes[link[1],1]*au.Ah],
                        c=link_colors[linktype-1],
                        zorder=-1)
        plt.savefig(os.path.join(self.directory, filename), bbox_inches='tight', dpi=200)
        plt.close()

    def plot_flake_lattice_k(self, filename='flake_lattice_k.png', subsets=None):
        _, ax = plt.subplots()
        ax.set_aspect(1.)
        ax.set_xlabel(r'$k_x$ (nm$^{-1}$)')
        ax.set_ylabel(r'$k_y$ (nm$^{-1}$)')
        node_colors = ['blue','orange']
        ax.scatter(x=self.flake.nodes_k[:,0]/au.Ah, 
                   y=self.flake.nodes_k[:,1]/au.Ah, 
                   c=node_colors[0],
                   s=self.pointsize/2.)
        if subsets is not None:
            for subset in subsets:
                ax.scatter(x=subset[:,0]/au.Ah, 
                           y=subset[:,1]/au.Ah, 
                           c=node_colors[1],
                           s=self.pointsize/2.)
        plt.savefig(os.path.join(self.directory, filename), bbox_inches='tight', dpi=200)
        plt.close()

    def plot_statedensity(self, density, filename='eigenstate', suffix=None):
        _, ax = plt.subplots()
        ax.set_aspect(1.)
        ax.set_xlabel("x (nm)")
        ax.set_ylabel("y (nm)")
        ax.scatter(x=self.flake.nodes[:,0]*au.Ah, y=self.flake.nodes[:,1]*au.Ah, c=density, s=self.pointsize)
        #ax.text(-20, 10, "{:.2f}".format(time))
        if suffix is not None:
            filename += suffix
        plt.savefig(os.path.join(self.directory, filename+'.png'), bbox_inches='tight', dpi=200)    
        plt.close()

    def plot_potential_flake(self, potential, filename='potential_flake', suffix=None):
        fig, ax = plt.subplots()
        ax.set_aspect(1.)
        ax.set_xlabel("x (nm)")
        ax.set_ylabel("y (nm)")
        potentialonflake = ax.scatter(x=self.flake.nodes[:,0]*au.Ah, y=self.flake.nodes[:,1]*au.Ah, c=potential*au.Eh*-1, s=self.pointsize, 
                                        cmap='coolwarm')  # , vmin=-100., vmax=0.)
        axins = inset_axes(ax, width = "10%", height = "100%", loc = 'lower left',
                   bbox_to_anchor = (1.1, 0., 1, 1), bbox_transform = ax.transAxes,
                   borderpad = 0)
        cbar = fig.colorbar(potentialonflake, cax = axins)  #, ticks=[-100.,-50.,-0.])
        cbar.set_label(r'$\phi$ (mV)')
        #ax.text(-20, 10, "{:.2f}".format(time))
        if suffix is not None:
            filename += suffix
        plt.savefig(os.path.join(self.directory, filename+'.png'), bbox_inches='tight', dpi=200) 
        plt.close()

    def plot_electric_field_flake(self, field, filename='electric_field_flake.png'):
        fig, ax = plt.subplots()
        ax.set_aspect(1.)
        ax.set_xlabel("x (nm)")
        ax.set_ylabel("y (nm)")
        potentialonflake = ax.scatter(x=self.flake.nodes[:,0]*au.Ah, y=self.flake.nodes[:,1]*au.Ah, c=field*au.Eh/au.Ah, s=self.pointsize, cmap='plasma')
        cbar = fig.colorbar(potentialonflake, ax=ax)
        cbar.set_label(r'$E_z$ (mV/nm)')
        plt.savefig(os.path.join(self.directory, filename), bbox_inches='tight', dpi=200) 
        plt.close()


class PlottingBands:
    def __init__(self, flake, directory=None):
        self.flake = flake
        self.grid_k = flake.BZ_path
        self.critical_points = flake.critical_points
        self.critical_points_indices = [(r'$\Gamma$', 0), ('Q', 96), ('K', 61), ('M', 41), (r'$\Gamma$', 130)]
        if directory is not None:
            self.directory = os.path.join('./', directory)
            os.makedirs(directory, exist_ok=True)
        else:
            self.directory = './'
        self.pointsize = 2.

    def plot_eigenvalues(self, eigenvalues, dot_occupation=None, filename='eigenvalues.png'):
        _, ax = plt.subplots()
        ax.set_xlabel("subsequent eigenstates")
        ax.set_ylabel("energy (meV)")
        if dot_occupation is not None:
            colors = dot_occupation
        else:
            colors = 'tab:blue'
        ax.scatter(x=np.arange(1, eigenvalues.size+1), y=eigenvalues, s=5., c=colors)
        plt.savefig(os.path.join(self.directory, filename), bbox_inches='tight', dpi=200)
        plt.close()

    def plot_eigenvalues_sv(self, eigenvalues, spin_valley, filename='eigenvalues_sv.png', pointsize=20.):
        #        
        fx=6.
        fy=5.
        dpi=400
        fig, ax = plt.subplots(figsize=(fx,fy), dpi=dpi)
        ax.set_xlabel("subsequent eigenstates")
        ax.set_ylabel("energy (eV)")        
        y=eigenvalues
        dy=np.sqrt(pointsize)*dpi/72.*(np.amax(y)-np.amin(y))/(fy*dpi)/1.5
        x=np.arange(1, eigenvalues.size+1)
        sv = ax.scatter(x=np.arange(1, eigenvalues.size+1), y=y+dy, 
                        s=pointsize, c=spin_valley[:,0], cmap='coolwarm', vmin=-1.05, vmax=1.05)
        # sv = ax.scatter(x=np.arange(1, eigenvalues.size+1), y=y-dy, 
        #                 s=pointsize, c=spin_valley[:,1], cmap='coolwarm', vmin=-1.05, vmax=1.05)
        fig.colorbar(sv, ax=ax, ticks=[-1,-.5,0,.5,1])
        plt.savefig(os.path.join(self.directory, filename), bbox_inches='tight', dpi=200)
        plt.close()

    def plot_eigenvalues_svk(self, eigenvalues, spin_valley, k_max, filename='eigenvalues_svk.png', pointsize=20.):
        #        
        fx=10.
        fy=5.
        dpi=400
        fig, ax = plt.subplots(figsize=(fx,fy), dpi=dpi)
        ax.set_xlabel('$k_x$ (nm$^{-1}$)')
        ax.set_ylabel("energy (eV)")   
        #ax.set_ylim([-79,-5])     
        y=eigenvalues
        dy=np.sqrt(pointsize)*dpi/72.*(np.amax(y)-np.amin(y))/(fy*dpi)/1.7
        #ax.plot([self.flake.K_points[0][1]/au.Ah,self.flake.K_points[0][1]/au.Ah], [-80,0], '--', c='black', zorder=1)
        #ax.plot([self.flake.K_points[1][1]/au.Ah,self.flake.K_points[1][1]/au.Ah], [-80,0], '--', c='black', zorder=1)
        sv = ax.scatter(x=k_max[:,1]/au.Ah, y=y+dy, 
                        s=pointsize, c=spin_valley[:,0], cmap='coolwarm', vmin=-1.05, vmax=1.05, zorder=2)
        sv = ax.scatter(x=k_max[:,1]/au.Ah, y=y-dy, 
                        s=pointsize, c=spin_valley[:,1], cmap='coolwarm', vmin=-1.05, vmax=1.05, zorder=1)
        axins = inset_axes(ax, width = "3%", height = "100%", loc = 'lower left',
                   bbox_to_anchor = (1.02, 0., 1, 1), bbox_transform = ax.transAxes,
                   borderpad = 0)
        fig.colorbar(sv, cax=axins, ticks=[-1,-.5,0,.5,1])
        plt.savefig(os.path.join(self.directory, filename), bbox_inches='tight', dpi=200)
        plt.close()

    def plot_eigenvalues_svk_(self, eigenvalues, spin_valley, k_max, filename='eigenvalues_svk_.png', pointsize=20.):
        #        
        fx=10.
        fy=5.
        dpi=400
        fig, ax = plt.subplots(figsize=(fx,fy), dpi=dpi)
        ax.set_xlabel('$k_y$ (nm$^{-1}$)')
        ax.set_ylabel("energy (eV)")   
        #ax.set_ylim([-79,-5])     
        y=eigenvalues
        dy=np.sqrt(pointsize)*dpi/72.*(np.amax(y)-np.amin(y))/(fy*dpi)/1.5
        #ax.plot([self.flake.K_points[0][1]/au.Ah,self.flake.K_points[0][1]/au.Ah], [-80,0], '--', c='black', zorder=1)
        #ax.plot([self.flake.K_points[1][1]/au.Ah,self.flake.K_points[1][1]/au.Ah], [-80,0], '--', c='black', zorder=1)
        sv = ax.scatter(x=k_max[:,1]/au.Ah, y=y, 
                        s=pointsize, c=spin_valley[:,0], cmap='coolwarm', vmin=-1.05, vmax=1.05, zorder=1)
        axins = inset_axes(ax, width = "3%", height = "100%", loc = 'lower left',
                   bbox_to_anchor = (1.02, 0., 1, 1), bbox_transform = ax.transAxes,
                   borderpad = 0)
        fig.colorbar(sv, cax=axins, ticks=[-1,-.5,0,.5,1])
        plt.savefig(os.path.join(self.directory, filename), bbox_inches='tight', dpi=200)
        plt.close()
        np.savetxt('./results/eigvals_k.txt', np.c_[k_max[:,1]/au.Ah, y, spin_valley[:,0]])

    def plot_eigenvalues_bnk(self, eigenvalues, std_y, k_max, filename='eigenvalues_bnk.png', pointsize=20.):
        #        
        fx=10.
        fy=5.
        dpi=400
        fig, ax = plt.subplots(figsize=(fx,fy), dpi=dpi)
        ax.set_xlabel('$k_x$ (nm$^{-1}$)')
        ax.set_ylabel("energy (eV)")   
        ax.set_ylim([-79,-5])     
        y=eigenvalues
        ax.plot([self.flake.K_points[0][0]/au.Ah,self.flake.K_points[0][0]/au.Ah], [-80,0], '--', c='black', zorder=1)
        ax.plot([self.flake.K_points[1][0]/au.Ah,self.flake.K_points[1][0]/au.Ah], [-80,0], '--', c='black', zorder=1)
        sv = ax.scatter(x=k_max[:,0]/au.Ah, y=y, 
                        s=pointsize, c=std_y*au.Ah, cmap='coolwarm', zorder=2)
        axins = inset_axes(ax, width = "3%", height = "100%", loc = 'lower left',
                   bbox_to_anchor = (1.02, 0., 1, 1), bbox_transform = ax.transAxes,
                   borderpad = 0)
        fig.colorbar(sv, cax=axins)
        plt.savefig(os.path.join(self.directory, filename), bbox_inches='tight', dpi=200)
        plt.close()

    def plot_energy_surface_k(self, subsets, energy, filename='energy_surface_k.png'):
        fig, ax = plt.subplots()
        ax.set_aspect(1.)
        ax.set_xlabel(r'$k_x$ (nm$^{-1}$)')
        ax.set_ylabel(r'$k_y$ (nm$^{-1}$)')
        subset = np.concatenate(subsets)
        energysurface = ax.scatter(x=subset[:,0]/au.Ah, 
                        y=subset[:,1]/au.Ah, 
                        c=energy,  # *au.Eh,
                        s=self.pointsize/2., cmap='coolwarm')
        cbar = fig.colorbar(energysurface, ax=ax)
        cbar.set_label(r'energy (eV)')
        plt.savefig(os.path.join(self.directory, filename), bbox_inches='tight', dpi=200)
        plt.close()
        
    def plot_spin_surface_k(self, subsets, spin, filename='spin_surface_k.png'):
        fig, ax = plt.subplots()
        ax.set_aspect(1.)
        ax.set_xlabel(r'$k_x$ (nm$^{-1}$)')
        ax.set_ylabel(r'$k_y$ (nm$^{-1}$)')
        subset = np.concatenate(subsets)
        spinsurface = ax.scatter(x=subset[:,0]/au.Ah, 
                        y=subset[:,1]/au.Ah, 
                        c=spin,  # *au.Eh,
                        s=self.pointsize/2., cmap='bwr')
        cbar = fig.colorbar(spinsurface, ax=ax)
        cbar.set_label(r'energy (eV)')
        plt.savefig(os.path.join(self.directory, filename), bbox_inches='tight', dpi=200)
        plt.close()
                
    def plot_energy_surface_kk(self, subset, energy, filename='energy_surface_k.png'):
        fig, ax = plt.subplots()
        ax.set_aspect(1.)
        ax.set_xlabel(r'$k_x$ (nm$^{-1}$)')
        ax.set_ylabel(r'$k_y$ (nm$^{-1}$)')
        energysurface = ax.scatter(x=subset[:,0]/au.Ah, 
                        y=subset[:,1]/au.Ah, 
                        c=energy,  # *au.Eh
                        s=self.pointsize/2., cmap='coolwarm')
        cbar = fig.colorbar(energysurface, ax=ax)
        cbar.set_label(r'energy (eV)')
        plt.savefig(os.path.join(self.directory, filename), bbox_inches='tight', dpi=200)
        plt.close()   
            
    def plot_berry_k(self, field, filename='berry_flake.png'):
        fig, ax = plt.subplots()
        ax.set_aspect(1.)
        ax.set_xlabel(r'$k_x$ (nm$^{-1}$)')
        ax.set_ylabel(r'$k_y$ (nm$^{-1}$)')
        fieldonflake = ax.scatter(x=field[:,0]/au.Ah, y=field[:,1]/au.Ah, c=field[:,2], s=self.pointsize, cmap='bwr')
        cbar = fig.colorbar(fieldonflake, ax=ax)
        cbar.set_label(r'$F$')
        plt.savefig(os.path.join(self.directory, filename), bbox_inches='tight', dpi=200) 
        plt.close()

    def plot_Ek(self, Ek, x_label='k (nm$^{-1}$)', y_label='E (eV)'):
        """ Plots dispersion relations for
        two given lists of bands.

        Parameters
        ----------
        Ek_. : List[array]
            List of arrays of eigenvalues
        x_label : string
            label of x-axis
        y_label : string
            label of y-axis
        """
        pointsize = .5
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.axes.set_aspect(.1)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        k_path = np.linspace(0., 1., num=self.grid_k.shape[0])
        
        for band_idx in range(Ek.shape[1]):
            ax.scatter(k_path, Ek[:,band_idx], s=pointsize, marker='.')

        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right')

        text_shift_x = (ax.get_xlim()[1] - ax.get_xlim()[0])*0.01
        plot_max_y = ax.get_ylim()[1]

        for (name, position_index) in self.critical_points_indices:
             position_k=k_path[position_index]
             ax.annotate(name, xy=(position_k-text_shift_x, plot_max_y), xytext=(position_k-text_shift_x, plot_max_y + 0.1))
             ax.axvline(x=position_k, linestyle='--', color='black')
        filename = 'ek.png'
        plt.savefig(os.path.join(self.directory, filename), bbox_inches='tight', dpi=400)
        plt.close()


class PlottingFourier:
    def __init__(self, flake_solver, directory=None):
        self.nkp = flake_solver.nkpoints
        self.grid_kx = np.arange(-self.nkp,self.nkp+1)/self.nkp*np.pi*2/flake_solver.m.a0
        self.grid_ky = self.grid_kx
        if directory is not None:
            self.directory = os.path.join('./', directory)
            os.makedirs(directory, exist_ok=True)
        else:
            self.directory = './'
        self.pointsize = 10.

    def plot_fourierdensity(self, fourierdensity):
        _, ax = plt.subplots()
        ax.axes.set_aspect('equal')
        ax.set_xlabel('$k_x$ (nm$^{-1}$)')
        ax.set_ylabel('$k_y$ (nm$^{-1}$)')
        fouriergrid = ax.pcolormesh(self.grid_kx/au.Ah, self.grid_ky/au.Ah, 
                                    fourierdensity.transpose(), cmap='viridis')
        cbar = plt.colorbar(fouriergrid)
        cbar.set_label('Fourier density')
        filename = 'fourier_density.png'
        plt.savefig(os.path.join(self.directory, filename), bbox_inches='tight', dpi=200)
        plt.close()
        
    def plot_elements_kmq(self, elements_kmq, l_index=0):  # "k minus q"
        _, ax = plt.subplots()
        ax.axes.set_aspect('equal')
        ax.set_xlabel('$(k-q)_x$ (nm$^{-1}$)')
        ax.set_ylabel('$(k-q)_y$ (nm$^{-1}$)')
        ax.set_xlim([-15,15])
        ax.set_ylim([-15,15])
        value = np.sqrt(abs2(elements_kmq[l_index][:,2]))*au.Eh
        #div = np.sqrt(abs2(elements_kmq[1][:,2]-elements_kmq[0][:,2]))*au.Eh
        elements = ax.scatter(x=elements_kmq[l_index][:,0]/au.Ah, y=elements_kmq[l_index][:,1]/au.Ah, 
                              c=value, s=self.pointsize, cmap='viridis', norm=colors.LogNorm(vmin=1.e-10, vmax=1.e1))  # norm=colors.LogNorm(vmin=1.e-15)
        cbar = plt.colorbar(elements)
        cbar.set_label('|potential elements| (meV)')
        filename = 'potential_elements.png'
        plt.savefig(os.path.join(self.directory, filename), bbox_inches='tight', dpi=200)
        plt.close()
        
    def plot_dressed_elements(self, basis_k, dressed_elements, suffix=None, savetofile=None):  # dressed_elements is q,k indexed
        _, ax = plt.subplots()
        ax.axes.set_aspect('equal')
        ax.set_xlabel('$(k-q)_x$ (nm$^{-1}$)')
        ax.set_ylabel('$(k-q)_y$ (nm$^{-1}$)')
        #ax.set_xlim([-15,15])
        #ax.set_ylim([-15,15])
        value = np.sqrt(abs2(dressed_elements))*au.Eh*au.Ah*au.Ah
        value += 1.e-10
        kx = basis_k[:,0]
        ky = basis_k[:,1]
        if savetofile is not None:
            np.savetxt('./results/'+savetofile, np.c_[kx,ky, value])        
        elements = ax.scatter(x=kx/au.Ah, y=ky/au.Ah, 
                              c=value, s=self.pointsize, cmap='viridis', norm=colors.LogNorm(vmin=1.e-10, vmax=1.e1))  # norm=colors.LogNorm(vmin=1.e-15)
        cbar = plt.colorbar(elements)
        cbar.set_label('|potential elements| (meV nm$^2$)')
        filename = 'dressed_elements.png'
        if suffix is not None:
            filename = 'dressed_elements_'+suffix+'.png'
        plt.savefig(os.path.join(self.directory, filename), bbox_inches='tight', dpi=200)
        plt.close()
