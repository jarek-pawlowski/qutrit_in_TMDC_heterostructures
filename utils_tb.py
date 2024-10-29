import os
import numpy as np
from scipy.linalg import eigh
import datetime
import matplotlib.pyplot as plt


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
    Ehh=Eh/1000. # eV
    Ah=0.05292 # nm
    Th=2.41888e-5 # ps
    Bh=235051.76 # Teslas

au = AtomicUnits()


class TMDCmaterial:
    """ Class containing lattice model parameters.

    """
    def __init__(self, a0, dp, Vdps, Vdpp, Vd2s, Vd2p, Vd2d, Vp2s, Vp2p, Ed, Ep1, Ep0, lm, lx2):
        self.a0 = a0/au.Ah
        self.dr = self.a0/np.sqrt(3.)
        self.dp = dp/au.Ah
        self.dd = np.sqrt(self.dr**2+self.dp**2)
        self.dim = 12
        self.dim2 = self.dim*self.dim
        self.dim12 = int(self.dim/2)
        # hoppings
        self.Vdps = Vdps
        self.Vdpp = Vdpp
        self.Vd2s = Vd2s
        self.Vd2p = Vd2p
        self.Vd2d = Vd2d
        self.Vp2s = Vp2s
        self.Vp2p = Vp2p
        # onsite energy
        self.Ed = Ed
        self.Ep1 = Ep1
        self.Ep0 = Ep0
        self.diag = np.tile(np.array([self.Ed, self.Ed, self.Ed,
                                      self.Ep1, self.Ep0, self.Ep1]),2)
        # intrinsic spin-orbit
        self.lm = lm/au.Eh
        self.lx2 = lx2/au.Eh
        self.l_diag = np.array([-self.lm, 0.,  self.lm, -self.lx2/2., 0.,  self.lx2/2., 
                                 self.lm, 0., -self.lm,  self.lx2/2., 0., -self.lx2/2.])

    def update_parameters(self,  Vdps, Vdpp, Vd2s, Vd2p, Vd2d, Vp2s, Vp2p, Ed, Ep1, Ep0, lm, lx2):
         # hoppings
        self.Vdps = Vdps
        self.Vdpp = Vdpp
        self.Vd2s = Vd2s
        self.Vd2p = Vd2p
        self.Vd2d = Vd2d
        self.Vp2s = Vp2s
        self.Vp2p = Vp2p
        # onsite energy
        self.Ed = Ed
        self.Ep1 = Ep1
        self.Ep0 = Ep0
        self.diag = np.tile(np.array([self.Ed, self.Ed, self.Ed,
                                      self.Ep1, self.Ep0, self.Ep1]),2)
        self.lm = lm
        self.lx2 = lx2
        self.l_diag = np.array([-self.lm, 0.,  self.lm, -self.lx2/2., 0.,  self.lx2/2., 
                                 self.lm, 0., -self.lm,  self.lx2/2., 0., -self.lx2/2.])


class TMDCmaterial3:
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
    def __init__(self, a0, t0, t1, t2, t11, t12, t22, e0, e1, e2, lso):
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
        self.e1 = e1/au.Eh
        self.e2 = e2/au.Eh
        self.diag = np.array([self.e0,self.e1,self.e2,self.e0,self.e1,self.e2])
        # intrinsic spin-orbit
        self.lso = lso/au.Eh

    def update_parameters(self, t0, t1, t2, t11, t12, t22, e0, e1, e2, lso):
         # hoppings
        self.t0 = t0
        self.t1 = t1
        self.t2 = t2
        self.t11 = t11
        self.t12 = t12
        self.t22 = t22
        # onsite energy
        self.e0 = e0
        self.e1 = e1
        self.e2 = e2
        self.diag = np.array([self.e0,self.e1,self.e2,self.e0,self.e1,self.e2])
        self.lso = lso


class Newmaterial:
    
    def __init__(self, lattice_const, d2_up, d2_down, d0,
                 Ed_up,Ep1_up,Ep0_up,Vdp_sigma_up,Vdp_pi_up,Vdd_sigma_up,Vdd_pi_up,Vdd_delta_up,Vpp_sigma_up,Vpp_pi_up,Ep1_odd_up,Ep0_odd_up,Ed_odd_up,lambda_M_up,lambda_X2_up,
                 Ed_down,Ep1_down,Ep0_down,Vdp_sigma_down,Vdp_pi_down,Vdd_sigma_down,Vdd_pi_down,Vdd_delta_down,Vpp_sigma_down,Vpp_pi_down,Ep1_odd_down,Ep0_odd_down,Ed_odd_down,lambda_M_down,lambda_X2_down,
                 Vpp_sigma_inter,Vpp_pi_inter,Vdd_sigma_inter,Vdd_pi_inter,Vdd_delta_inter,Vdp_sigma_inter,Vpd_sigma_inter,Vdp_pi_inter,Vpd_pi_inter,
                 offset):
        
        self.dim = 44
        self.lattice_const = lattice_const/au.Ah
        self.a0 = self.lattice_const
        self.d0 = d0/au.Ah
        self.d2_up = d2_up/au.Ah
        self.d2_down = d2_down/au.Ah
        #self.fel = 0.
        #
        self.Ed_up = Ed_up
        self.Ep1_up = Ep1_up
        self.Ep0_up = Ep0_up
        self.Vdp_sigma_up = Vdp_sigma_up
        self.Vdp_pi_up = Vdp_pi_up
        self.Vdd_sigma_up = Vdd_sigma_up
        self.Vdd_pi_up = Vdd_pi_up
        self.Vdd_delta_up = Vdd_delta_up
        self.Vpp_sigma_up = Vpp_sigma_up
        self.Vpp_pi_up = Vpp_pi_up
        self.Ep1_odd_up = Ep1_odd_up
        self.Ep0_odd_up = Ep0_odd_up
        self.Ed_odd_up = Ed_odd_up
        
        self.Ed_up_0 = self.Ed_up
        self.Ep1_up_0 = self.Ep1_up
        self.Ep0_up_0 = self.Ep0_up
        self.Vdp_sigma_up_0 = self.Vdp_sigma_up
        self.Vdp_pi_up_0 = self.Vdp_pi_up
        self.Vdd_sigma_up_0 = self.Vdd_sigma_up
        self.Vdd_pi_up_0 = self.Vdd_pi_up
        self.Vdd_delta_up_0 = self.Vdd_delta_up
        self.Vpp_sigma_up_0 = self.Vpp_sigma_up
        self.Vpp_pi_up_0 = self.Vpp_pi_up
        self.Ep1_odd_up_0 = self.Ep1_odd_up
        self.Ep0_odd_up_0 = self.Ep0_odd_up
        self.Ed_odd_up_0 = self.Ed_odd_up
        #
        self.lambda_M_up = lambda_M_up
        self.lambda_X2_up = lambda_X2_up
        #
        self.Ed_down = Ed_down
        self.Ep1_down = Ep1_down
        self.Ep0_down = Ep0_down
        self.Vdp_sigma_down = Vdp_sigma_down
        self.Vdp_pi_down = Vdp_pi_down
        self.Vdd_sigma_down = Vdd_sigma_down
        self.Vdd_pi_down = Vdd_pi_down
        self.Vdd_delta_down = Vdd_delta_down
        self.Vpp_sigma_down = Vpp_sigma_down
        self.Vpp_pi_down = Vpp_pi_down
        self.Ep1_odd_down = Ep1_odd_down
        self.Ep0_odd_down = Ep0_odd_down
        self.Ed_odd_down = Ed_odd_down
        
        self.Ed_down_0 = Ed_down
        self.Ep1_down_0 = self.Ep1_down
        self.Ep0_down_0 = self.Ep0_down
        self.Vdp_sigma_down_0 = self.Vdp_sigma_down
        self.Vdp_pi_down_0 = self.Vdp_pi_down
        self.Vdd_sigma_down_0 = self.Vdd_sigma_down
        self.Vdd_pi_down_0 = self.Vdd_pi_down
        self.Vdd_delta_down_0 = self.Vdd_delta_down
        self.Vpp_sigma_down_0 = self.Vpp_sigma_down
        self.Vpp_pi_down_0 = self.Vpp_pi_down
        self.Ep1_odd_down_0 = self.Ep1_odd_down
        self.Ep0_odd_down_0 = self.Ep0_odd_down
        self.Ed_odd_down_0 = self.Ed_odd_down
        #
        self.lambda_M_down = lambda_M_down
        self.lambda_X2_down = lambda_X2_down
        #
        self.Vpp_sigma_inter = Vpp_sigma_inter
        self.Vpp_pi_inter = Vpp_pi_inter
        self.Vdd_sigma_inter = Vdd_sigma_inter
        self.Vdd_pi_inter = Vdd_pi_inter
        self.Vdd_delta_inter = Vdd_delta_inter
        
        self.Vdp_sigma_inter = Vdp_sigma_inter
        self.Vpd_sigma_inter = Vpd_sigma_inter
        self.Vdp_pi_inter = Vdp_pi_inter
        self.Vpd_pi_inter = Vpd_pi_inter

        self.offset = offset
            
    def update_parameters(self, 
                          Ed_up,Ep1_up,Ep0_up,Vdp_sigma_up,Vdp_pi_up,Vdd_sigma_up,Vdd_pi_up,Vdd_delta_up,Vpp_sigma_up,Vpp_pi_up,Ep1_odd_up,Ep0_odd_up,Ed_odd_up,
                          Ed_down,Ep1_down,Ep0_down,Vdp_sigma_down,Vdp_pi_down,Vdd_sigma_down,Vdd_pi_down,Vdd_delta_down,Vpp_sigma_down,Vpp_pi_down,Ep1_odd_down,Ep0_odd_down,Ed_odd_down,
                          Vpp_sigma_inter, Vpp_pi_inter, Vdd_sigma_inter, Vdd_pi_inter, Vdd_delta_inter, Vdp_sigma_inter, Vpd_sigma_inter, Vdp_pi_inter, Vpd_pi_inter,
                          offset):
        
        self.Ed_up = self.Ed_up_0 + Ed_up
        self.Ep1_up = self.Ep1_up_0 + Ep1_up
        self.Ep0_up = self.Ep0_up_0 + Ep0_up
        self.Vdp_sigma_up = self.Vdp_sigma_up_0 + Vdp_sigma_up
        self.Vdp_pi_up = self.Vdp_pi_up_0 + Vdp_pi_up
        self.Vdd_sigma_up = self.Vdd_sigma_up_0 + Vdd_sigma_up
        self.Vdd_pi_up = self.Vdd_pi_up_0 + Vdd_pi_up
        self.Vdd_delta_up = self.Vdd_delta_up_0 + Vdd_delta_up
        self.Vpp_sigma_up = self.Vpp_sigma_up_0 + Vpp_sigma_up
        self.Vpp_pi_up = self.Vpp_pi_up_0 + Vpp_pi_up
        self.Ep1_odd_up = self.Ep1_odd_up_0 + Ep1_odd_up
        self.Ep0_odd_up = self.Ep0_odd_up_0 + Ep0_odd_up
        self.Ed_odd_up = self.Ed_odd_up_0 + Ed_odd_up
        
        self.Ed_down = self.Ed_down_0 + Ed_down
        self.Ep1_down = self.Ep1_down_0 + Ep1_down
        self.Ep0_down = self.Ep0_down_0 + Ep0_down
        self.Vdp_sigma_down = self.Vdp_sigma_down_0 + Vdp_sigma_down
        self.Vdp_pi_down = self.Vdp_pi_down_0 + Vdp_pi_down
        self.Vdd_sigma_down = self.Vdd_sigma_down_0 + Vdd_sigma_down
        self.Vdd_pi_down = self.Vdd_pi_down_0 + Vdd_pi_down
        self.Vdd_delta_down = self.Vdd_delta_down_0 + Vdd_delta_down
        self.Vpp_sigma_down = self.Vpp_sigma_down_0 + Vpp_sigma_down
        self.Vpp_pi_down = self.Vpp_pi_down_0 + Vpp_pi_down
        self.Ep1_odd_down = self.Ep1_odd_down_0 + Ep1_odd_down
        self.Ep0_odd_down = self.Ep0_odd_down_0 + Ep0_odd_down
        self.Ed_odd_down = self.Ed_odd_down_0 + Ed_odd_down
        
        self.Vpp_sigma_inter = Vpp_sigma_inter
        self.Vpp_pi_inter = Vpp_pi_inter
        self.Vdd_sigma_inter = Vdd_sigma_inter
        self.Vdd_pi_inter = Vdd_pi_inter
        self.Vdd_delta_inter = Vdd_delta_inter
        
        self.Vdp_sigma_inter = Vdp_sigma_inter
        self.Vpd_sigma_inter = Vpd_sigma_inter
        self.Vdp_pi_inter = Vdp_pi_inter
        self.Vpd_pi_inter = Vpd_pi_inter
        
        self.offset = offset
        
    def set_parameters(self, 
                       Ed_up,Ep1_up,Ep0_up,Vdp_sigma_up,Vdp_pi_up,Vdd_sigma_up,Vdd_pi_up,Vdd_delta_up,Vpp_sigma_up,Vpp_pi_up,Ep1_odd_up,Ep0_odd_up,Ed_odd_up,
                       Ed_down,Ep1_down,Ep0_down,Vdp_sigma_down,Vdp_pi_down,Vdd_sigma_down,Vdd_pi_down,Vdd_delta_down,Vpp_sigma_down,Vpp_pi_down,Ep1_odd_down,Ep0_odd_down,Ed_odd_down,
                       Vpp_sigma_inter, Vpp_pi_inter, Vdd_sigma_inter, Vdd_pi_inter, Vdd_delta_inter, Vdp_sigma_inter, Vpd_sigma_inter, Vdp_pi_inter, Vpd_pi_inter,
                       offset):
        
        self.Ed_up = Ed_up
        self.Ep1_up = Ep1_up
        self.Ep0_up = Ep0_up
        self.Vdp_sigma_up = Vdp_sigma_up
        self.Vdp_pi_up = Vdp_pi_up
        self.Vdd_sigma_up = Vdd_sigma_up
        self.Vdd_pi_up = Vdd_pi_up
        self.Vdd_delta_up = Vdd_delta_up
        self.Vpp_sigma_up = Vpp_sigma_up
        self.Vpp_pi_up = Vpp_pi_up
        self.Ep1_odd_up = Ep1_odd_up
        self.Ep0_odd_up = Ep0_odd_up
        self.Ed_odd_up = Ed_odd_up
        
        self.Ed_down = Ed_down
        self.Ep1_down = Ep1_down
        self.Ep0_down = Ep0_down
        self.Vdp_sigma_down = Vdp_sigma_down
        self.Vdp_pi_down = Vdp_pi_down
        self.Vdd_sigma_down = Vdd_sigma_down
        self.Vdd_pi_down = Vdd_pi_down
        self.Vdd_delta_down = Vdd_delta_down
        self.Vpp_sigma_down = Vpp_sigma_down
        self.Vpp_pi_down = Vpp_pi_down
        self.Ep1_odd_down = Ep1_odd_down
        self.Ep0_odd_down = Ep0_odd_down
        self.Ed_odd_down = Ed_odd_down
        
        self.Vpp_sigma_inter = Vpp_sigma_inter
        self.Vpp_pi_inter = Vpp_pi_inter
        self.Vdd_sigma_inter = Vdd_sigma_inter
        self.Vdd_pi_inter = Vdd_pi_inter
        self.Vdd_delta_inter = Vdd_delta_inter
        
        self.Vdp_sigma_inter = Vdp_sigma_inter
        self.Vpd_sigma_inter = Vpd_sigma_inter
        self.Vdp_pi_inter = Vdp_pi_inter
        self.Vpd_pi_inter = Vpd_pi_inter
        
        self.offset = offset
        
    def set_efield(self, field_in_mV_nm):
        self.fel = field_in_mV_nm*au.Ah/au.Eh


class Lattice:

    def __init__(self, BZ_path=None):
        self.lattice_vectors = np.array([[0.,1.], [np.sqrt(3.)/2.,-.5]])
        self.K_points = [np.array([np.pi*4./np.sqrt(3.),np.pi*4./3.]), np.array([np.pi*2./np.sqrt(3.),np.pi*4./6.])]
        self.Q_points = [np.array([np.pi*4./np.sqrt(3.),np.pi*2./3.]),  # Q1
                         np.array([np.pi*5./np.sqrt(3.),np.pi*5./3.]),  # Q2
                         np.array([np.pi*3./np.sqrt(3.),np.pi*5./3.]),  # Q3
                         np.array([np.pi*2./np.sqrt(3.),np.pi*4./3.]),  # Q1'
                         np.array([np.pi*1./np.sqrt(3.),np.pi*1./3.]),  # Q2'
                         np.array([np.pi*3./np.sqrt(3.),np.pi*1./3.])]  # Q3'
        # RB1 = np.array([0.,-1.])
        # RB2 = np.array([np.sqrt(3.),1.])/2.
        # RB3 = np.array([-np.sqrt(3.),1.])/2.
        RB1 = np.array([1.,0.])
        RB2 = np.array([-1.,np.sqrt(3.)])/2.
        RB3 = np.array([-1.,-np.sqrt(3.)])/2.
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
        K = self.K_points[0][0]
        M = K*3./2
        G = K*np.sqrt(3.)/2.
        dk = (M+G)/131.  # magic number to get exactly 131 points at the path
        self.critical_points = [(r'$\Gamma$', 0.), ('K', K), ('M', M), (r'$\Gamma$', M+G)]
        self.critical_points_w_names = {"gamma_1": 0., "K": K, "M": M, "gamma_2": M+G}
        k_GK = [[x, 0.] for x in np.arange(0, K, dk)] # k varying from Gamma to K point within the BZ
        k_KM = [[x, 0.] for x in np.arange(K, M, dk)] # k varying from K to M point within the BZ
        k_MG = [[M, y]  for y in np.linspace(0, G, num=int(G/dk), endpoint=True)] # k varying from M to Gamma point within the BZ
        if BZ_path is not None:
            self.BZ_path = BZ_path
            self.BZ_loaded_externally = True
        else:
            self.BZ_path = np.concatenate((k_GK, k_KM, k_MG)) # full path within the BZ
            self.BZ_loaded_externally = False
    
    def select_k_indices(self, distance=5):
        """ 
        select points along BZ path:
        with neighbor points taken at some distance
        """
        d = distance 
        self.k_indices = [0,41,51,61,71,81,91,96,101,111]  # G = 0, M = 41, K = 61, Q = 96, G = 130
        self.ks_indices = [61,80,96]
        self.ks_indices = [61,71,80,91,96]        
        self.critical_points_indices = [(r'$\Gamma$', 0), ('Q', 96), ('K', 61), ('M', 41), (r'$\Gamma$', 130)]

class BandModel:

    def __init__(self, parameters, lattice):
        self.m = parameters
        self.l = lattice
        self.hoppingsMM = [h*self.m.a0/np.sqrt(3.) for h in self.l.hoppingsMM]
        self.hoppingsMX = [h*self.m.a0/np.sqrt(3.) for h in self.l.hoppingsMX]
        if self.l.BZ_loaded_externally:
            self.BZ_path = self.l.BZ_path*au.Ah*10.
        else:
            self.BZ_path = self.l.BZ_path/self.m.a0
        self.critical_points = [(p[0], p[1]/self.m.a0) for p in self.l.critical_points]
        self.K_points = [p/self.m.a0 for p in self.l.K_points]
        self.critical_points_indices = self.l.critical_points_indices

    def hopping_matrix_(self, x, y, x1, y1, linkstype):
        """
        create 6x6 hopping matrix that represents hopping integral within the tight-binding model

        orbitals basis = {Dm2, , Dp2, PEm1, PE0, PEp1}

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
            if linkstype == 4:
            # X2-M hoppings, T(-R) = T^\dag(R)
                hh_m[3:6,0:3] = np.conjugate(hh_m[0:3,3:6]).transpose()
                hh_m[0:3,3:6] = 0.
        # spin-down block is the same:
        hh_m[m.dim12:,m.dim12:] = hh_m[:m.dim12,:m.dim12]
        return hh_m

    def build_tb_hamiltonian(self, kx, ky):
        hh_m = np.zeros((self.m.dim,self.m.dim), dtype=np.complex128)
        diagonal = self.m.diag.copy()
        # intrinistic spin-orbit coupling -- diagonal part:
        diagonal += self.m.l_diag
        np.fill_diagonal(hh_m, diagonal)
        # hoppings
        for h in self.hoppingsMX:
            hh_m += self.hopping_matrix_(0., 0., h[0], h[1], 3)*np.exp(1.j*(kx*h[0]+ky*h[1]))
            hh_m += self.hopping_matrix_(0., 0., h[0], h[1], 4)*np.exp(1.j*(kx*h[0]+ky*h[1]))
        for h in self.hoppingsMM:
            hh_m += self.hopping_matrix_(0., 0., h[0], h[1], 1)*np.exp(1.j*(kx*h[0]+ky*h[1]))
            hh_m += self.hopping_matrix_(0., 0., h[0], h[1], 2)*np.exp(1.j*(kx*h[0]+ky*h[1]))
        return hh_m
    
    def hopping_matrix_6(self, x, y, x1, y1, parameters):
        """
        create 6x6 hopping matrix that represents hopping integral within the tight-binding model
        """
        m = parameters 
        hh_m=np.zeros((m.dim,m.dim), dtype=np.complex128)
        # which of hopping vector R1...R6??
        # see e.g.: Phys. Rev. B 91, 155410 (2015) or Phys. Rev. B 88, 085433 (2013).
        # R1 / R6
        if abs(y - y1) < 1.e-8:
            if x1 > x: R=1
            if x1 < x: R=6
        # R2 / R3
        if abs((x1 - x) - 0.5 *m.a0) < 1.e-8:
            if y1 > y: R=3
            if y1 < y: R=2
        # R4 / R5
        if abs((x1 - x) + 0.5 *m.a0) < 1.e-8:
            if y1 > y: R=4
            if y1 < y: R=5
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
        return hh_m  
    
    def build_tb_hamiltonian3(self, kx, ky):
        hh_m = np.zeros((self.m.dim,self.m.dim), dtype=np.complex128)
        np.fill_diagonal(hh_m, self.m.diag)  
        # intrinistic spin-orbit coupling
        hh_m[1,2] = 1.j*self.m.lso
        hh_m[2,1] = -1.j*self.m.lso
        hh_m[4,5] = -1.j*self.m.lso
        hh_m[5,4] = 1.j*self.m.lso
        # hoppings
        for h in self.hoppingsMM:
            hh_m += self.hopping_matrix_6(0., 0., h[0], h[1], self.m)*np.exp(1.j*(kx*h[0]+ky*h[1]))   
        return hh_m

    def build_tb_hamiltonian_new(self, kx, ky):
        # Geometry Parameters (lattice constanst for MoSe2) -------
        lattice_const_in_A = self.m.lattice_const*au.Ah*10.
        kx /= au.Ah*10.
        ky /= au.Ah*10.
        d1            = lattice_const_in_A/np.sqrt(3.0)
        d2_up         = self.m.d2_up * au.Ah*10.  # to be in A
        d2_down       = self.m.d2_down * au.Ah*10.  # to be in A
        d_up          = np.sqrt(d1**2.0+d2_up**2.0) 
        d_down        = np.sqrt(d1**2.0+d2_down**2.0) 
        layer_dist    = self.m.d0 * au.Ah*10.  # to be in A
        dz_pp         = layer_dist - d2_up - d2_down
        d_pp          = np.sqrt( (lattice_const_in_A**2.0 / 3.0) + (dz_pp**2.0) ) 
        dz_dd         = layer_dist
        d_dd          = np.sqrt( (lattice_const_in_A**2.0 / 3.0) + (dz_dd**2.0) ) 
        dz_dp         = layer_dist - d2_down
        dz_pd         = layer_dist - d2_up
        R1x_pp        = -d1/2.0 
        R1y_pp        = d1*np.sqrt(3.0)/2.0 
        R2x_pp        = -d1/2.0  
        R2y_pp        = -d1*np.sqrt(3.0)/2.0 
        R3x_pp        = d1
        R3y_pp        = 0.0 
        R1x_dd        = d1/2.0 
        R1y_dd        = d1*np.sqrt(3.0)/2.0 
        R2x_dd        = d1/2.0  
        R2y_dd        = -d1*np.sqrt(3.0)/2.0 
        R3x_dd        = -d1
        R3y_dd        = 0.0 
        #
        H_k = np.zeros((self.m.dim+1, self.m.dim+1), dtype=np.complex128)
        # --- Constructing the Hamiltonian ---
        # --- Constructing the Hamiltonian ---
        g0        =  4.0*np.cos(3.0/2.0*kx*d1)*np.cos(np.sqrt(3.0)/2.0*ky*d1) + 2.0*np.cos(np.sqrt(3.0)*ky*d1) 
        g2        =  2.0*np.cos(3.0/2.0*kx*d1+np.sqrt(3.0)/2.0*ky*d1)*np.exp(1.j*np.pi/3.0) + 2.0*np.cos(3.0/2.0*kx*d1-np.sqrt(3.0)/2.0*ky*d1)*np.exp(-1.j*np.pi/3.0) - 2.0*np.cos(np.sqrt(3.0)*ky*d1) 
        g4        =  2.0*np.cos(3.0/2.0*kx*d1+np.sqrt(3.0)/2.0*ky*d1)*np.exp(1.j*2.0*np.pi/3.0) + 2.0*np.cos(3.0/2.0*kx*d1-np.sqrt(3.0)/2.0*ky*d1)*np.exp(-1.j*2.0*np.pi/3.0) + 2.0*np.cos(np.sqrt(3.0)*ky*d1) 
        # for up (rotated) just apply kx -> -kx, ky -> -ky
        f_m1_up   = np.exp(1.j*(-kx)*d1) + np.exp(-1.j*(-kx)*d1/2.0)*np.exp(1.j*np.sqrt(3.0)*(-ky)*d1/2.0)*np.exp(1.j*2.0*np.pi/3.0) + np.exp(-1.j*(-kx)*d1/2.0)*np.exp(-1.j*np.sqrt(3.0)*(-ky)*d1/2.0)*np.exp(-1.j*2.0*np.pi/3.0) 
        f_0_up    = np.exp(1.j*(-kx)*d1) + np.exp(-1.j*(-kx)*d1/2.0)*np.exp(1.j*np.sqrt(3.0)*(-ky)*d1/2.0)*np.exp(-1.j*2.0*np.pi/3.0) + np.exp(-1.j*(-kx)*d1/2.0)*np.exp(-1.j*np.sqrt(3.0)*(-ky)*d1/2.0)*np.exp( 1.j*2.0*np.pi/3.0) 
        f_p1_up   = np.exp(1.j*(-kx)*d1) + np.exp(-1.j*(-kx)*d1/2.0)*np.exp(1.j*np.sqrt(3.0)*(-ky)*d1/2.0) + np.exp(-1.j*(-kx)*d1/2.0)*np.exp(-1.j*np.sqrt(3.0)*(-ky)*d1/2.0)
        # 
        f_m1_down = np.exp(1.j*kx*d1) + np.exp(-1.j*kx*d1/2.0)*np.exp(1.j*np.sqrt(3.0)*ky*d1/2.0)*np.exp(1.j*2.0*np.pi/3.0) + np.exp(-1.j*kx*d1/2.0)*np.exp(-1.j*np.sqrt(3.0)*ky*d1/2.0)*np.exp(-1.j*2.0*np.pi/3.0)  
        f_0_down  = np.exp(1.j*kx*d1) + np.exp(-1.j*kx*d1/2.0)*np.exp(1.j*np.sqrt(3.0)*ky*d1/2.0)*np.exp(-1.j*2.0*np.pi/3.0) + np.exp(-1.j*kx*d1/2.0)*np.exp(-1.j*np.sqrt(3.0)*ky*d1/2.0)*np.exp( 1.j*2.0*np.pi/3.0) 
        f_p1_down = np.exp(1.j*kx*d1) + np.exp(-1.j*kx*d1/2.0)*np.exp(1.j*np.sqrt(3.0)*ky*d1/2.0) + np.exp(-1.j*kx*d1/2.0)*np.exp(-1.j*np.sqrt(3.0)*ky*d1/2.0) 
        # layer up
        V1_up  =  1.0/np.sqrt(2.0)*d1/d_up*( np.sqrt(3.0)/2.0*self.m.Vdp_sigma_up*((d2_up/d_up)**(2.0)-1) - self.m.Vdp_pi_up*((d2_up/d_up)**(2.0)+1) ) 
        V2_up  =  0.5*( np.sqrt(3.0)*self.m.Vdp_sigma_up-2.0*self.m.Vdp_pi_up )*(d2_up/d_up)*(d1/d_up)**(2.0) 
        V3_up  =  1.0/np.sqrt(2.0)*( np.sqrt(3.0)/2.0*self.m.Vdp_sigma_up-self.m.Vdp_pi_up)*(d1/d_up)**(3.0) 
        V4_up  =  0.5*( (3.0*(d2_up/d_up)**(2.0)-1)*self.m.Vdp_sigma_up - (2.0*np.sqrt(3.0)*(d2_up/d_up)**(2.0))*self.m.Vdp_pi_up )*(d1/d_up) 
        V5_up  =  1.0/np.sqrt(2.0)*(d2_up/d_up)*( (3.0*(d2_up/d_up)**(2.0)-1)*self.m.Vdp_sigma_up - (2.0*np.sqrt(3.0)*((d2_up/d_up)**(2.0)-1))*self.m.Vdp_pi_up ) 
        V6_up  =  1.0/np.sqrt(2.0)*(d2_up/d_up)*( ((d1/d_up)**2.0)*(np.sqrt(3.0)*self.m.Vdp_sigma_up-2.0*self.m.Vdp_pi_up)+2.0*self.m.Vdp_pi_up ) 
        V7_up  =  1.0/np.sqrt(2.0)*(d2_up*d1**2.0)/d_up**3.0 * ( np.sqrt(3.0)*self.m.Vdp_sigma_up- 2.0*self.m.Vdp_pi_up ) 
        V8_up  =  d1/d_up * ( ((d2_up/d_up)**2.0)*(np.sqrt(3.0)*self.m.Vdp_sigma_up-2.0*self.m.Vdp_pi_up)+self.m.Vdp_pi_up ) 
        W1_up  =  0.125*(3.0*self.m.Vdd_sigma_up+4*self.m.Vdd_pi_up+self.m.Vdd_delta_up) 
        W2_up  =  0.25*(self.m.Vdd_sigma_up+3.0*self.m.Vdd_delta_up) 
        W3_up  = -np.sqrt(3.0)/(4*np.sqrt(2.0))*(self.m.Vdd_sigma_up-self.m.Vdd_delta_up) 
        W4_up  =  0.125*(3.0*self.m.Vdd_sigma_up-4*self.m.Vdd_pi_up+self.m.Vdd_delta_up) 
        W5_up  =  0.5*(self.m.Vpp_sigma_up+self.m.Vpp_pi_up) 
        W6_up  =  self.m.Vpp_pi_up 
        W7_up  =  0.5*(self.m.Vpp_sigma_up-self.m.Vpp_pi_up) 
        W8_up  =  0.5*(self.m.Vdd_pi_up + self.m.Vdd_delta_up) 
        W9_up  =  0.5*(self.m.Vdd_pi_up - self.m.Vdd_delta_up) 
        # layer down
        V1_down  =  1.0/np.sqrt(2.0)*d1/d_down*( np.sqrt(3.0)/2.0*self.m.Vdp_sigma_down*((d2_down/d_down)**(2.0)-1) - self.m.Vdp_pi_down*((d2_down/d_down)**(2.0)+1) ) 
        V2_down  =  0.5*( np.sqrt(3.0)*self.m.Vdp_sigma_down-2.0*self.m.Vdp_pi_down )*(d2_down/d_down)*(d1/d_down)**(2.0) 
        V3_down  =  1.0/np.sqrt(2.0)*( np.sqrt(3.0)/2.0*self.m.Vdp_sigma_down-self.m.Vdp_pi_down)*(d1/d_down)**(3.0) 
        V4_down  =  0.5*( (3.0*(d2_down/d_down)**(2.0)-1)*self.m.Vdp_sigma_down - (2.0*np.sqrt(3.0)*(d2_down/d_down)**(2.0))*self.m.Vdp_pi_down )*(d1/d_down) 
        V5_down  =  1.0/np.sqrt(2.0)*(d2_down/d_down)*( (3.0*(d2_down/d_down)**(2.0)-1)*self.m.Vdp_sigma_down - (2.0*np.sqrt(3.0)*((d2_down/d_down)**(2.0)-1))*self.m.Vdp_pi_down ) 
        V6_down  =  1.0/np.sqrt(2.0)*(d2_down/d_down)*( ((d1/d_down)**2.0)*(np.sqrt(3.0)*self.m.Vdp_sigma_down-2.0*self.m.Vdp_pi_down)+2.0*self.m.Vdp_pi_down ) 
        V7_down  =  1.0/np.sqrt(2.0)*(d2_down*d1**2.0)/d_down**3.0 * ( np.sqrt(3.0)*self.m.Vdp_sigma_down- 2.0*self.m.Vdp_pi_down ) 
        V8_down  =  d1/d_down * ( ((d2_down/d_down)**2.0)*(np.sqrt(3.0)*self.m.Vdp_sigma_down-2.0*self.m.Vdp_pi_down)+self.m.Vdp_pi_down ) 
        W1_down  =  1.0/8*(3.0*self.m.Vdd_sigma_down+4*self.m.Vdd_pi_down+self.m.Vdd_delta_down) 
        W2_down  =  1.0/4*(self.m.Vdd_sigma_down+3.0*self.m.Vdd_delta_down) 
        W3_down  = -np.sqrt(3.0)/(4*np.sqrt(2.0))*(self.m.Vdd_sigma_down-self.m.Vdd_delta_down) 
        W4_down  =  1.0/8*(3.0*self.m.Vdd_sigma_down-4*self.m.Vdd_pi_down+self.m.Vdd_delta_down) 
        W5_down  =  0.5*(self.m.Vpp_sigma_down+self.m.Vpp_pi_down) 
        W6_down  =  self.m.Vpp_pi_down 
        W7_down  =  0.5*(self.m.Vpp_sigma_down-self.m.Vpp_pi_down) 
        W8_down  =  0.5*(self.m.Vdd_pi_down + self.m.Vdd_delta_down) 
        W9_down  =  0.5*(self.m.Vdd_pi_down - self.m.Vdd_delta_down) 
        # interlayer interactions
        W12    = ((dz_pp/d_pp)**2.0)*self.m.Vpp_sigma_inter + (1.0-(dz_pp/d_pp)**2.0)*self.m.Vpp_pi_inter 
        W13    = self.m.Vdd_sigma_inter * (( 0.5*((d1/d_dd)**2.0) - ((dz_dd/d_dd)**2.0) )**2.0) + self.m.Vdd_pi_inter * ( 3.0*((dz_dd/d_dd)**2.0)*(1.0 - ((dz_dd/d_dd)**2.0)) ) + self.m.Vdd_delta_inter * ( 0.75 * (((dz_dd/d_dd)**2.0)**2.0) )
        W14    = ((d1/d_dd)**2.0) * ( 0.75*self.m.Vdd_sigma_inter + 0.25*((dz_dd/d_dd)**2.0)*self.m.Vdd_delta_inter + 0.5*((dz_dd/d_dd)**2.0)*self.m.Vdd_pi_inter )
        W15    = 2.0*((d1/d_dd)**2.0)*self.m.Vdd_pi_inter + ((dz_dd/d_dd)**2.0)*self.m.Vdd_delta_inter
        W16    = (3.0/16.0)*((d1/d_dd)**4.0) * ( 3.0*self.m.Vdd_sigma_inter + self.m.Vdd_delta_inter - 4.0*self.m.Vdd_pi_inter )
        h1_pp  = np.exp(1.j*kx*R1x_pp)*np.exp(1.j*ky*R1y_pp) + np.exp(1.j*kx*R2x_pp)*np.exp(1.j*ky*R2y_pp) + np.exp(1.j*kx*R3x_pp)*np.exp(1.j*ky*R3y_pp) 
        h1_dd  = np.exp(1.j*kx*R1x_dd)*np.exp(1.j*ky*R1y_dd) + np.exp(1.j*kx*R2x_dd)*np.exp(1.j*ky*R2y_dd) + np.exp(1.j*kx*R3x_dd)*np.exp(1.j*ky*R3y_dd) 
        h2_dd  = (-0.5)*np.exp(1.j*kx*R1x_dd)*np.exp(1.j*ky*R1y_dd) + (-0.5)*np.exp(1.j*kx*R2x_dd)*np.exp(1.j*ky*R2y_dd) + np.exp(1.j*kx*R3x_dd)*np.exp(1.j*ky*R3y_dd) 
        h3_dd  = (1.0+2.j*np.sqrt(3.0)/3.0)*np.exp(1.j*kx*R1x_dd)*np.exp(1.j*ky*R1y_dd) + (1.0-2.j*np.sqrt(3.0)/3.0)*np.exp(1.j*kx*R2x_dd)*np.exp(1.j*ky*R2y_dd)
        # diagonal part
        H_k[1,1]   = self.m.Ed_up  + (W1_up*g0)
        H_k[2,2]   = self.m.Ed_up  + (W2_up*g0)
        H_k[3,3]   = self.m.Ed_up  + (W1_up*g0) 
        H_k[4,4]   = self.m.Ep1_up + (W5_up*g0)
        H_k[5,5]   = self.m.Ep0_up + (W6_up*g0)
        H_k[6,6]   = self.m.Ep1_up + (W5_up*g0)
        H_k[7,7]   = self.m.Ed_odd_up  + (W8_up*g0)
        H_k[8,8]   = self.m.Ed_odd_up  + (W8_up*g0) 
        H_k[9,9]   = self.m.Ep1_odd_up + (W5_up*g0) 
        H_k[10,10] = self.m.Ep0_odd_up + (W6_up*g0)
        H_k[11,11] = self.m.Ep1_odd_up + (W5_up*g0)
        H_k[12,12] = self.m.Ed_down  + (W1_down*g0)
        H_k[13,13] = self.m.Ed_down  + (W2_down*g0)
        H_k[14,14] = self.m.Ed_down  + (W1_down*g0)
        H_k[15,15] = self.m.Ep1_down + (W5_down*g0)
        H_k[16,16] = self.m.Ep0_down + (W6_down*g0)
        H_k[17,17] = self.m.Ep1_down + (W5_down*g0) 
        H_k[18,18] = self.m.Ed_odd_down  + (W8_down*g0)
        H_k[19,19] = self.m.Ed_odd_down  + (W8_down*g0)
        H_k[20,20] = self.m.Ep1_odd_down + (W5_down*g0)
        H_k[21,21] = self.m.Ep0_odd_down + (W6_down*g0)
        H_k[22,22] = self.m.Ep1_odd_down + (W5_down*g0)
        # off-diagonal part W-dependent
        H_k[1,2]   =  W3_up*g2 
        H_k[1,3]   =  W4_up*g4 
        H_k[2,3]   =  W3_up*g2 
        H_k[4,6]   = -W7_up*g2 
        H_k[7,8]   = -W9_up*g2 
        H_k[9,11]  = -W7_up*g2 
        H_k[12,13] =  W3_down*g2 
        H_k[12,14] =  W4_down*g4 
        H_k[13,14] =  W3_down*g2 
        H_k[15,17] = -W7_down*g2 
        H_k[18,19] = -W9_down*g2 
        H_k[20,22] = -W7_down*g2  
        # off-diagonal part V-dependent
        H_k[1,4]   =  V1_up*f_m1_up*(-1.) 
        H_k[1,5]   = -V2_up*f_0_up 
        H_k[1,6]   =  V3_up*f_p1_up*(-1.) 
        H_k[2,4]   = -V4_up*f_0_up*(-1.)
        H_k[2,5]   = -V5_up*f_p1_up 
        H_k[2,6]   =  V4_up*f_m1_up*(-1.) 
        H_k[3,4]   = -V3_up*f_p1_up*(-1.) 
        H_k[3,5]   = -V2_up*f_m1_up 
        H_k[3,6]   = -V1_up*f_0_up*(-1.) 
        H_k[7,9]   = -V6_up*f_p1_up 
        H_k[7,10]  = -V8_up*f_m1_up *(-1.)
        H_k[7,11]  =  V7_up*f_0_up 
        H_k[8,9]   =  V7_up*f_m1_up 
        H_k[8,10]  =  V8_up*f_0_up*(-1.) 
        H_k[8,11]  = -V6_up*f_p1_up 
        # layer down: 
        H_k[12,15] =  V1_down*f_m1_down
        H_k[12,16] = -V2_down*f_0_down
        H_k[12,17] =  V3_down*f_p1_down
        H_k[13,15] = -V4_down*f_0_down
        H_k[13,16] = -V5_down*f_p1_down 
        H_k[13,17] =  V4_down*f_m1_down
        H_k[14,15] = -V3_down*f_p1_down
        H_k[14,16] = -V2_down*f_m1_down 
        H_k[14,17] = -V1_down*f_0_down
        H_k[18,20] = -V6_down*f_p1_down 
        H_k[18,21] = -V8_down*f_m1_down
        H_k[18,22] =  V7_down*f_0_down
        H_k[19,20] =  V7_down*f_m1_down 
        H_k[19,21] =  V8_down*f_0_down
        H_k[19,22] = -V6_down*f_p1_down 
        # layer interactions, old ones
        # H_k[2,13]  = -0.5*W13*h1_dd
        # H_k[5,16]  = -0.5*W12*h1_pp
        # H_k[1,14] = h2_dd*W14 + h1_dd*W15 + h3_dd*W16
        # H_k[3,12] = np.conjugate(H_k[1,14])   
        # --- layer interactions following Mathematica ---
        H_k[1,12]=((np.exp((3*1.j*d1*kx)/2.) + np.exp((1.j*np.sqrt(3)*d1*ky)/2.) + np.exp((1.j*d1*(3*kx + 2*np.sqrt(3)*ky))/2.))*(4*(d1**4 + 2*d1**2*dz_dd**2)*self.m.Vdd_pi_inter + (d1**4 + 8*d1**2*dz_dd**2 + 8*dz_dd**4)*self.m.Vdd_delta_inter + 3*d1**4*self.m.Vdd_sigma_inter))/(8.*(d1**2 + dz_dd**2)**2*np.exp((1.j*d1*(2*kx + np.sqrt(3)*ky))/2.))
        H_k[1,13]=(np.sqrt(1.5)*d1**2*((-1 - 1.j*np.sqrt(3))*np.exp((3*1.j*d1*kx)/2.) + 2*np.exp((1.j*np.sqrt(3)*d1*ky)/2.) + 1.j*(1.j + np.sqrt(3))*np.exp((1.j*d1*(3*kx + 2*np.sqrt(3)*ky))/2.))*(-4*dz_dd**2*self.m.Vdd_pi_inter + (d1**2 + 2*dz_dd**2)*self.m.Vdd_delta_inter - (d1**2 - 2*dz_dd**2)*self.m.Vdd_sigma_inter))/(8.*(d1**2 + dz_dd**2)**2*np.exp((1.j*d1*(2*kx + np.sqrt(3)*ky))/2.))
        H_k[1,14]=(d1**4*((1 - 1.j*np.sqrt(3))*np.exp((3*1.j*d1*kx)/2.) - 2*np.exp((1.j*np.sqrt(3)*d1*ky)/2.) + (1 + 1.j*np.sqrt(3))*np.exp((1.j*d1*(3*kx + 2*np.sqrt(3)*ky))/2.))*(4*self.m.Vdd_pi_inter - self.m.Vdd_delta_inter - 3*self.m.Vdd_sigma_inter))/(16.*(d1**2 + dz_dd**2)**2*np.exp((1.j*d1*(2*kx + np.sqrt(3)*ky))/2.))
        # H_k[1,15]=0
        # H_k[1,16]=0
        # H_k[1,17]=0
        H_k[1,18]=(d1*dz_dd*((1 - 1.j*np.sqrt(3))*np.exp((3*1.j*d1*kx)/2.) - 2*np.exp((1.j*np.sqrt(3)*d1*ky)/2.) + (1 + 1.j*np.sqrt(3))*np.exp((1.j*d1*(3*kx + 2*np.sqrt(3)*ky))/2.))*(-4*dz_dd**2*self.m.Vdd_pi_inter + (3*d1**2 + 4*dz_dd**2)*self.m.Vdd_delta_inter - 3*d1**2*self.m.Vdd_sigma_inter))/(8.*(d1**2 + dz_dd**2)**2*np.exp((1.j*d1*(2*kx + np.sqrt(3)*ky))/2.))
        H_k[1,19]=(d1**3*dz_dd*(np.exp((3*1.j*d1*kx)/2.) + np.exp((1.j*np.sqrt(3)*d1*ky)/2.) + np.exp((1.j*d1*(3*kx + 2*np.sqrt(3)*ky))/2.))*(4*self.m.Vdd_pi_inter - self.m.Vdd_delta_inter - 3*self.m.Vdd_sigma_inter))/(4.*(d1**2 + dz_dd**2)**2*np.exp((1.j*d1*(2*kx + np.sqrt(3)*ky))/2.))
        #H_k[1,20]=0
        #H_k[1,21]=0
        #H_k[1,22]=0
        H_k[2,12]=(np.sqrt(1.5)*d1**2*(1.j*(1.j + np.sqrt(3))*np.exp((3*1.j*d1*kx)/2.) + 2*np.exp((1.j*np.sqrt(3)*d1*ky)/2.) + (-1 - 1.j*np.sqrt(3))*np.exp((1.j*d1*(3*kx + 2*np.sqrt(3)*ky))/2.))*(-4*dz_dd**2*self.m.Vdd_pi_inter + (d1**2 + 2*dz_dd**2)*self.m.Vdd_delta_inter - (d1**2 - 2*dz_dd**2)*self.m.Vdd_sigma_inter))/(8.*(d1**2 + dz_dd**2)**2*np.exp((1.j*d1*(2*kx + np.sqrt(3)*ky))/2.))
        H_k[2,13]=((np.exp((3*1.j*d1*kx)/2.) + np.exp((1.j*np.sqrt(3)*d1*ky)/2.) + np.exp((1.j*d1*(3*kx + 2*np.sqrt(3)*ky))/2.))*(12*d1**2*dz_dd**2*self.m.Vdd_pi_inter + 3*d1**4*self.m.Vdd_delta_inter + (d1**2 - 2*dz_dd**2)**2*self.m.Vdd_sigma_inter))/(4.*(d1**2 + dz_dd**2)**2*np.exp((1.j*d1*(2*kx + np.sqrt(3)*ky))/2.))
        H_k[2,14]=(np.sqrt(1.5)*d1**2*((-1 - 1.j*np.sqrt(3))*np.exp((3*1.j*d1*kx)/2.) + 2*np.exp((1.j*np.sqrt(3)*d1*ky)/2.) + 1.j*(1.j + np.sqrt(3))*np.exp((1.j*d1*(3*kx + 2*np.sqrt(3)*ky))/2.))*(-4*dz_dd**2*self.m.Vdd_pi_inter + (d1**2 + 2*dz_dd**2)*self.m.Vdd_delta_inter - (d1**2 - 2*dz_dd**2)*self.m.Vdd_sigma_inter))/(8.*(d1**2 + dz_dd**2)**2*np.exp((1.j*d1*(2*kx + np.sqrt(3)*ky))/2.))
        # H_k[2,15]=0
        H_k[2,16]=(dz_dp*self.m.Vdp_sigma_inter)/(np.sqrt(2)*np.sqrt(dz_dp**2))
        # H_k[2,17]=0
        H_k[2,18]=(np.sqrt(1.5)*d1*dz_dd*((-1 - 1.j*np.sqrt(3))*np.exp((3*1.j*d1*kx)/2.) + 2*np.exp((1.j*np.sqrt(3)*d1*ky)/2.) + 1.j*(1.j + np.sqrt(3))*np.exp((1.j*d1*(3*kx + 2*np.sqrt(3)*ky))/2.))*(2*(d1 - dz_dd)*(d1 + dz_dd)*self.m.Vdd_pi_inter - d1**2*self.m.Vdd_delta_inter - (d1**2 - 2*dz_dd**2)*self.m.Vdd_sigma_inter))/(4.*(d1**2 + dz_dd**2)**2*np.exp((1.j*d1*(2*kx + np.sqrt(3)*ky))/2.))
        H_k[2,19]=(np.sqrt(1.5)*d1*dz_dd*((1 - 1.j*np.sqrt(3))*np.exp((3*1.j*d1*kx)/2.) - 2*np.exp((1.j*np.sqrt(3)*d1*ky)/2.) + (1 + 1.j*np.sqrt(3))*np.exp((1.j*d1*(3*kx + 2*np.sqrt(3)*ky))/2.))*(2*(d1 - dz_dd)*(d1 + dz_dd)*self.m.Vdd_pi_inter - d1**2*self.m.Vdd_delta_inter - (d1**2 - 2*dz_dd**2)*self.m.Vdd_sigma_inter))/(4.*(d1**2 + dz_dd**2)**2*np.exp((1.j*d1*(2*kx + np.sqrt(3)*ky))/2.))
        #H_k[2,20]=0
        H_k[2,21]=(dz_dp*self.m.Vdp_sigma_inter)/(np.sqrt(2)*np.sqrt(dz_dp**2))
        #H_k[2,22]=0
        H_k[3,12]=(d1**4*((1 + 1.j*np.sqrt(3))*np.exp((3*1.j*d1*kx)/2.) - 2*np.exp((1.j*np.sqrt(3)*d1*ky)/2.) + (1 - 1.j*np.sqrt(3))*np.exp((1.j*d1*(3*kx + 2*np.sqrt(3)*ky))/2.))*(4*self.m.Vdd_pi_inter - self.m.Vdd_delta_inter - 3*self.m.Vdd_sigma_inter))/(16.*(d1**2 + dz_dd**2)**2*np.exp((1.j*d1*(2*kx + np.sqrt(3)*ky))/2.))
        H_k[3,13]=(np.sqrt(1.5)*d1**2*(1.j*(1.j + np.sqrt(3))*np.exp((3*1.j*d1*kx)/2.) + 2*np.exp((1.j*np.sqrt(3)*d1*ky)/2.) + (-1 - 1.j*np.sqrt(3))*np.exp((1.j*d1*(3*kx + 2*np.sqrt(3)*ky))/2.))*(-4*dz_dd**2*self.m.Vdd_pi_inter + (d1**2 + 2*dz_dd**2)*self.m.Vdd_delta_inter - (d1**2 - 2*dz_dd**2)*self.m.Vdd_sigma_inter))/(8.*(d1**2 + dz_dd**2)**2*np.exp((1.j*d1*(2*kx + np.sqrt(3)*ky))/2.))
        H_k[3,14]=((np.exp((3*1.j*d1*kx)/2.) + np.exp((1.j*np.sqrt(3)*d1*ky)/2.) + np.exp((1.j*d1*(3*kx + 2*np.sqrt(3)*ky))/2.))*(4*(d1**4 + 2*d1**2*dz_dd**2)*self.m.Vdd_pi_inter + (d1**4 + 8*d1**2*dz_dd**2 + 8*dz_dd**4)*self.m.Vdd_delta_inter + 3*d1**4*self.m.Vdd_sigma_inter))/(8.*(d1**2 + dz_dd**2)**2*np.exp((1.j*d1*(2*kx + np.sqrt(3)*ky))/2.))
        # H_k[3,15]=0
        # H_k[3,16]=0
        # H_k[3,17]=0
        H_k[3,18]=-0.25*(d1**3*dz_dd*(np.exp((3*1.j*d1*kx)/2.) + np.exp((1.j*np.sqrt(3)*d1*ky)/2.) + np.exp((1.j*d1*(3*kx + 2*np.sqrt(3)*ky))/2.))*(4*self.m.Vdd_pi_inter - self.m.Vdd_delta_inter - 3*self.m.Vdd_sigma_inter))/((d1**2 + dz_dd**2)**2*np.exp((1.j*d1*(2*kx + np.sqrt(3)*ky))/2.))
        H_k[3,19]=(d1*dz_dd*((1 + 1.j*np.sqrt(3))*np.exp((3*1.j*d1*kx)/2.) - 2*np.exp((1.j*np.sqrt(3)*d1*ky)/2.) + (1 - 1.j*np.sqrt(3))*np.exp((1.j*d1*(3*kx + 2*np.sqrt(3)*ky))/2.))*(4*dz_dd**2*self.m.Vdd_pi_inter - (3*d1**2 + 4*dz_dd**2)*self.m.Vdd_delta_inter + 3*d1**2*self.m.Vdd_sigma_inter))/(8.*(d1**2 + dz_dd**2)**2*np.exp((1.j*d1*(2*kx + np.sqrt(3)*ky))/2.))
        #H_k[3,20]=0
        #H_k[3,21]=0
        #H_k[3,22]=0
        # H_k[4,12]=0
        # H_k[4,13]=0
        # H_k[4,14]=0
        H_k[4,15]=((1 + np.exp(1.j*np.sqrt(3)*d1*ky) + np.exp((1.j*d1*(3*kx + np.sqrt(3)*ky))/2.))*((d1**2 + 2*dz_pp**2)*self.m.Vpp_pi_inter + d1**2*self.m.Vpp_sigma_inter))/(4.*(d1**2 + dz_pp**2)*np.exp((1.j*d1*(kx + np.sqrt(3)*ky))/2.))
        H_k[4,16]=(d1*dz_pp*(-1 - 1.j*np.sqrt(3) + 1.j*(1.j + np.sqrt(3))*np.exp(1.j*np.sqrt(3)*d1*ky) + 2*np.exp((1.j*d1*(3*kx + np.sqrt(3)*ky))/2.))*(self.m.Vpp_pi_inter - self.m.Vpp_sigma_inter))/(4.*np.sqrt(2)*(d1**2 + dz_pp**2)*np.exp((1.j*d1*(kx + np.sqrt(3)*ky))/2.))
        H_k[4,17]=(d1**2*(-1 + 1.j*np.sqrt(3) + (-1 - 1.j*np.sqrt(3))*np.exp(1.j*np.sqrt(3)*d1*ky) + 2*np.exp((1.j*d1*(3*kx + np.sqrt(3)*ky))/2.))*(self.m.Vpp_pi_inter - self.m.Vpp_sigma_inter))/(8.*(d1**2 + dz_pp**2)*np.exp((1.j*d1*(kx + np.sqrt(3)*ky))/2.))
        #H_k[4,18]=-((dz_pd*Subscript(V,p*d*Pi))/(np.sqrt(2)*np.sqrt(dz_pd**2)))
        #H_k[4,19]=0
        H_k[4,20]=((1 + np.exp(1.j*np.sqrt(3)*d1*ky) + np.exp((1.j*d1*(3*kx + np.sqrt(3)*ky))/2.))*((d1**2 + 2*dz_pp**2)*self.m.Vpp_pi_inter + d1**2*self.m.Vpp_sigma_inter))/(4.*(d1**2 + dz_pp**2)*np.exp((1.j*d1*(kx + np.sqrt(3)*ky))/2.))
        H_k[4,21]=(d1*dz_pp*(-1 - 1.j*np.sqrt(3) + 1.j*(1.j + np.sqrt(3))*np.exp(1.j*np.sqrt(3)*d1*ky) + 2*np.exp((1.j*d1*(3*kx + np.sqrt(3)*ky))/2.))*(self.m.Vpp_pi_inter - self.m.Vpp_sigma_inter))/(4.*np.sqrt(2)*(d1**2 + dz_pp**2)*np.exp((1.j*d1*(kx + np.sqrt(3)*ky))/2.))
        H_k[4,22]=(d1**2*(-1 + 1.j*np.sqrt(3) + (-1 - 1.j*np.sqrt(3))*np.exp(1.j*np.sqrt(3)*d1*ky) + 2*np.exp((1.j*d1*(3*kx + np.sqrt(3)*ky))/2.))*(self.m.Vpp_pi_inter - self.m.Vpp_sigma_inter))/(8.*(d1**2 + dz_pp**2)*np.exp((1.j*d1*(kx + np.sqrt(3)*ky))/2.))
        # H_k[5,12]=0
        H_k[5,13]=(dz_pd*self.m.Vpd_sigma_inter)/(np.sqrt(2)*np.sqrt(dz_pd**2))
        # H_k[5,14]=0
        H_k[5,15]=(d1*dz_pp*(1 - 1.j*np.sqrt(3) + (1 + 1.j*np.sqrt(3))*np.exp(1.j*np.sqrt(3)*d1*ky) - 2*np.exp((1.j*d1*(3*kx + np.sqrt(3)*ky))/2.))*(self.m.Vpp_pi_inter - self.m.Vpp_sigma_inter))/(4.*np.sqrt(2)*(d1**2 + dz_pp**2)*np.exp((1.j*d1*(kx + np.sqrt(3)*ky))/2.))
        H_k[5,16]=-0.5*((1 + np.exp(1.j*np.sqrt(3)*d1*ky) + np.exp((1.j*d1*(3*kx + np.sqrt(3)*ky))/2.))*(d1**2*self.m.Vpp_pi_inter + dz_pp**2*self.m.Vpp_sigma_inter))/((d1**2 + dz_pp**2)*np.exp((1.j*d1*(kx + np.sqrt(3)*ky))/2.))
        H_k[5,17]=(d1*dz_pp*(-1 - 1.j*np.sqrt(3) + 1.j*(1.j + np.sqrt(3))*np.exp(1.j*np.sqrt(3)*d1*ky) + 2*np.exp((1.j*d1*(3*kx + np.sqrt(3)*ky))/2.))*(self.m.Vpp_pi_inter - self.m.Vpp_sigma_inter))/(4.*np.sqrt(2)*(d1**2 + dz_pp**2)*np.exp((1.j*d1*(kx + np.sqrt(3)*ky))/2.))
        #H_k[5,18]=0
        #H_k[5,19]=0
        H_k[5,20]=(d1*dz_pp*(1 - 1.j*np.sqrt(3) + (1 + 1.j*np.sqrt(3))*np.exp(1.j*np.sqrt(3)*d1*ky) - 2*np.exp((1.j*d1*(3*kx + np.sqrt(3)*ky))/2.))*(self.m.Vpp_pi_inter - self.m.Vpp_sigma_inter))/(4.*np.sqrt(2)*(d1**2 + dz_pp**2)*np.exp((1.j*d1*(kx + np.sqrt(3)*ky))/2.))
        H_k[5,21]=-0.5*((1 + np.exp(1.j*np.sqrt(3)*d1*ky) + np.exp((1.j*d1*(3*kx + np.sqrt(3)*ky))/2.))*(d1**2*self.m.Vpp_pi_inter + dz_pp**2*self.m.Vpp_sigma_inter))/((d1**2 + dz_pp**2)*np.exp((1.j*d1*(kx + np.sqrt(3)*ky))/2.))
        H_k[5,22]=(d1*dz_pp*(-1 - 1.j*np.sqrt(3) + 1.j*(1.j + np.sqrt(3))*np.exp(1.j*np.sqrt(3)*d1*ky) + 2*np.exp((1.j*d1*(3*kx + np.sqrt(3)*ky))/2.))*(self.m.Vpp_pi_inter - self.m.Vpp_sigma_inter))/(4.*np.sqrt(2)*(d1**2 + dz_pp**2)*np.exp((1.j*d1*(kx + np.sqrt(3)*ky))/2.))
        # H_k[6,12]=0
        # H_k[6,13]=0
        # H_k[6,14]=0
        H_k[6,15]=(d1**2*(-1 - 1.j*np.sqrt(3) + 1.j*(1.j + np.sqrt(3))*np.exp(1.j*np.sqrt(3)*d1*ky) + 2*np.exp((1.j*d1*(3*kx + np.sqrt(3)*ky))/2.))*(self.m.Vpp_pi_inter - self.m.Vpp_sigma_inter))/(8.*(d1**2 + dz_pp**2)*np.exp((1.j*d1*(kx + np.sqrt(3)*ky))/2.))
        H_k[6,16]=(d1*dz_pp*(1 - 1.j*np.sqrt(3) + (1 + 1.j*np.sqrt(3))*np.exp(1.j*np.sqrt(3)*d1*ky) - 2*np.exp((1.j*d1*(3*kx + np.sqrt(3)*ky))/2.))*(self.m.Vpp_pi_inter - self.m.Vpp_sigma_inter))/(4.*np.sqrt(2)*(d1**2 + dz_pp**2)*np.exp((1.j*d1*(kx + np.sqrt(3)*ky))/2.))
        H_k[6,17]=((1 + np.exp(1.j*np.sqrt(3)*d1*ky) + np.exp((1.j*d1*(3*kx + np.sqrt(3)*ky))/2.))*((d1**2 + 2*dz_pp**2)*self.m.Vpp_pi_inter + d1**2*self.m.Vpp_sigma_inter))/(4.*(d1**2 + dz_pp**2)*np.exp((1.j*d1*(kx + np.sqrt(3)*ky))/2.))
        #H_k[6,18]=0
        #H_k[6,19]=-((dz_pd*Subscript(V,p*d*Pi))/(np.sqrt(2)*np.sqrt(dz_pd**2)))
        H_k[6,20]=(d1**2*(-1 - 1.j*np.sqrt(3) + 1.j*(1.j + np.sqrt(3))*np.exp(1.j*np.sqrt(3)*d1*ky) + 2*np.exp((1.j*d1*(3*kx + np.sqrt(3)*ky))/2.))*(self.m.Vpp_pi_inter - self.m.Vpp_sigma_inter))/(8.*(d1**2 + dz_pp**2)*np.exp((1.j*d1*(kx + np.sqrt(3)*ky))/2.))
        H_k[6,21]=(d1*dz_pp*(1 - 1.j*np.sqrt(3) + (1 + 1.j*np.sqrt(3))*np.exp(1.j*np.sqrt(3)*d1*ky) - 2*np.exp((1.j*d1*(3*kx + np.sqrt(3)*ky))/2.))*(self.m.Vpp_pi_inter - self.m.Vpp_sigma_inter))/(4.*np.sqrt(2)*(d1**2 + dz_pp**2)*np.exp((1.j*d1*(kx + np.sqrt(3)*ky))/2.))
        H_k[6,22]=((1 + np.exp(1.j*np.sqrt(3)*d1*ky) + np.exp((1.j*d1*(3*kx + np.sqrt(3)*ky))/2.))*((d1**2 + 2*dz_pp**2)*self.m.Vpp_pi_inter + d1**2*self.m.Vpp_sigma_inter))/(4.*(d1**2 + dz_pp**2)*np.exp((1.j*d1*(kx + np.sqrt(3)*ky))/2.))
        H_k[7,12]=(d1*dz_dd*((1 + 1.j*np.sqrt(3))*np.exp((3*1.j*d1*kx)/2.) - 2*np.exp((1.j*np.sqrt(3)*d1*ky)/2.) + (1 - 1.j*np.sqrt(3))*np.exp((1.j*d1*(3*kx + 2*np.sqrt(3)*ky))/2.))*(-4*dz_dd**2*self.m.Vdd_pi_inter + (3*d1**2 + 4*dz_dd**2)*self.m.Vdd_delta_inter - 3*d1**2*self.m.Vdd_sigma_inter))/(8.*(d1**2 + dz_dd**2)**2*np.exp((1.j*d1*(2*kx + np.sqrt(3)*ky))/2.))
        H_k[7,13]=(np.sqrt(1.5)*d1*dz_dd*(1.j*(1.j + np.sqrt(3))*np.exp((3*1.j*d1*kx)/2.) + 2*np.exp((1.j*np.sqrt(3)*d1*ky)/2.) + (-1 - 1.j*np.sqrt(3))*np.exp((1.j*d1*(3*kx + 2*np.sqrt(3)*ky))/2.))*(2*(d1 - dz_dd)*(d1 + dz_dd)*self.m.Vdd_pi_inter - d1**2*self.m.Vdd_delta_inter - (d1**2 - 2*dz_dd**2)*self.m.Vdd_sigma_inter))/(4.*(d1**2 + dz_dd**2)**2*np.exp((1.j*d1*(2*kx + np.sqrt(3)*ky))/2.))
        H_k[7,14]=-0.25*(d1**3*dz_dd*(np.exp((3*1.j*d1*kx)/2.) + np.exp((1.j*np.sqrt(3)*d1*ky)/2.) + np.exp((1.j*d1*(3*kx + 2*np.sqrt(3)*ky))/2.))*(4*self.m.Vdd_pi_inter - self.m.Vdd_delta_inter - 3*self.m.Vdd_sigma_inter))/((d1**2 + dz_dd**2)**2*np.exp((1.j*d1*(2*kx + np.sqrt(3)*ky))/2.))
        #H_k[7,15]=(dz_dp*Subscript(V,d*p*Pi))/(np.sqrt(2)*np.sqrt(dz_dp**2))
        #H_k[7,16]=0
        #H_k[7,17]=0
        H_k[7,18]=((np.exp((3*1.j*d1*kx)/2.) + np.exp((1.j*np.sqrt(3)*d1*ky)/2.) + np.exp((1.j*d1*(3*kx + 2*np.sqrt(3)*ky))/2.))*((d1**4 - d1**2*dz_dd**2 + 2*dz_dd**4)*self.m.Vdd_pi_inter + d1**2*((d1**2 + 2*dz_dd**2)*self.m.Vdd_delta_inter + 3*dz_dd**2*self.m.Vdd_sigma_inter)))/(2.*(d1**2 + dz_dd**2)**2*np.exp((1.j*d1*(2*kx + np.sqrt(3)*ky))/2.))
        H_k[7,19]=(d1**2*((1 + 1.j*np.sqrt(3))*np.exp((3*1.j*d1*kx)/2.) - 2*np.exp((1.j*np.sqrt(3)*d1*ky)/2.) + (1 - 1.j*np.sqrt(3))*np.exp((1.j*d1*(3*kx + 2*np.sqrt(3)*ky))/2.))*((d1**2 - 3*dz_dd**2)*self.m.Vdd_pi_inter - d1**2*self.m.Vdd_delta_inter + 3*dz_dd**2*self.m.Vdd_sigma_inter))/(4.*(d1**2 + dz_dd**2)**2*np.exp((1.j*d1*(2*kx + np.sqrt(3)*ky))/2.))
        #H_k[7,20]=(dz_dp*Subscript(V,d*p*Pi))/(np.sqrt(2)*np.sqrt(dz_dp**2))
        #H_k[7,21]=0
        #H_k[7,22]=0
        H_k[8,12]=(d1**3*dz_dd*(np.exp((3*1.j*d1*kx)/2.) + np.exp((1.j*np.sqrt(3)*d1*ky)/2.) + np.exp((1.j*d1*(3*kx + 2*np.sqrt(3)*ky))/2.))*(4*self.m.Vdd_pi_inter - self.m.Vdd_delta_inter - 3*self.m.Vdd_sigma_inter))/(4.*(d1**2 + dz_dd**2)**2*np.exp((1.j*d1*(2*kx + np.sqrt(3)*ky))/2.))
        H_k[8,13]=(np.sqrt(1.5)*d1*dz_dd*((1 + 1.j*np.sqrt(3))*np.exp((3*1.j*d1*kx)/2.) - 2*np.exp((1.j*np.sqrt(3)*d1*ky)/2.) + (1 - 1.j*np.sqrt(3))*np.exp((1.j*d1*(3*kx + 2*np.sqrt(3)*ky))/2.))*(2*(d1 - dz_dd)*(d1 + dz_dd)*self.m.Vdd_pi_inter - d1**2*self.m.Vdd_delta_inter - (d1**2 - 2*dz_dd**2)*self.m.Vdd_sigma_inter))/(4.*(d1**2 + dz_dd**2)**2*np.exp((1.j*d1*(2*kx + np.sqrt(3)*ky))/2.))
        H_k[8,14]=(d1*dz_dd*((1 - 1.j*np.sqrt(3))*np.exp((3*1.j*d1*kx)/2.) - 2*np.exp((1.j*np.sqrt(3)*d1*ky)/2.) + (1 + 1.j*np.sqrt(3))*np.exp((1.j*d1*(3*kx + 2*np.sqrt(3)*ky))/2.))*(4*dz_dd**2*self.m.Vdd_pi_inter - (3*d1**2 + 4*dz_dd**2)*self.m.Vdd_delta_inter + 3*d1**2*self.m.Vdd_sigma_inter))/(8.*(d1**2 + dz_dd**2)**2*np.exp((1.j*d1*(2*kx + np.sqrt(3)*ky))/2.))
        #H_k[8,15]=0
        #H_k[8,16]=0
        #H_k[8,17]=(dz_dp*Subscript(V,d*p*Pi))/(np.sqrt(2)*np.sqrt(dz_dp**2))
        H_k[8,18]=(d1**2*((1 - 1.j*np.sqrt(3))*np.exp((3*1.j*d1*kx)/2.) - 2*np.exp((1.j*np.sqrt(3)*d1*ky)/2.) + (1 + 1.j*np.sqrt(3))*np.exp((1.j*d1*(3*kx + 2*np.sqrt(3)*ky))/2.))*((d1**2 - 3*dz_dd**2)*self.m.Vdd_pi_inter - d1**2*self.m.Vdd_delta_inter + 3*dz_dd**2*self.m.Vdd_sigma_inter))/(4.*(d1**2 + dz_dd**2)**2*np.exp((1.j*d1*(2*kx + np.sqrt(3)*ky))/2.))
        H_k[8,19]=((np.exp((3*1.j*d1*kx)/2.) + np.exp((1.j*np.sqrt(3)*d1*ky)/2.) + np.exp((1.j*d1*(3*kx + 2*np.sqrt(3)*ky))/2.))*((d1**4 - d1**2*dz_dd**2 + 2*dz_dd**4)*self.m.Vdd_pi_inter + d1**2*((d1**2 + 2*dz_dd**2)*self.m.Vdd_delta_inter + 3*dz_dd**2*self.m.Vdd_sigma_inter)))/(2.*(d1**2 + dz_dd**2)**2*np.exp((1.j*d1*(2*kx + np.sqrt(3)*ky))/2.))
        #H_k[8,20]=0
        #H_k[8,21]=0
        #H_k[8,22]=(dz_dp*Subscript(V,d*p*Pi))/(np.sqrt(2)*np.sqrt(dz_dp**2))
        #H_k[9,12]=0
        #H_k[9,13]=0
        #H_k[9,14]=0
        H_k[9,15]=-0.25*((1 + np.exp(1.j*np.sqrt(3)*d1*ky) + np.exp((1.j*d1*(3*kx + np.sqrt(3)*ky))/2.))*((d1**2 + 2*dz_pp**2)*self.m.Vpp_pi_inter + d1**2*self.m.Vpp_sigma_inter))/((d1**2 + dz_pp**2)*np.exp((1.j*d1*(kx + np.sqrt(3)*ky))/2.))
        H_k[9,16]=(d1*dz_pp*(1 + 1.j*np.sqrt(3) + (1 - 1.j*np.sqrt(3))*np.exp(1.j*np.sqrt(3)*d1*ky) - 2*np.exp((1.j*d1*(3*kx + np.sqrt(3)*ky))/2.))*(self.m.Vpp_pi_inter - self.m.Vpp_sigma_inter))/(4.*np.sqrt(2)*(d1**2 + dz_pp**2)*np.exp((1.j*d1*(kx + np.sqrt(3)*ky))/2.))
        H_k[9,17]=(d1**2*(1 - 1.j*np.sqrt(3) + (1 + 1.j*np.sqrt(3))*np.exp(1.j*np.sqrt(3)*d1*ky) - 2*np.exp((1.j*d1*(3*kx + np.sqrt(3)*ky))/2.))*(self.m.Vpp_pi_inter - self.m.Vpp_sigma_inter))/(8.*(d1**2 + dz_pp**2)*np.exp((1.j*d1*(kx + np.sqrt(3)*ky))/2.))
        #H_k[9,18]=(dz_pd*Subscript(V,p*d*Pi))/(np.sqrt(2)*np.sqrt(dz_pd**2))
        #H_k[9,19]=0
        H_k[9,20]=-0.25*((1 + np.exp(1.j*np.sqrt(3)*d1*ky) + np.exp((1.j*d1*(3*kx + np.sqrt(3)*ky))/2.))*((d1**2 + 2*dz_pp**2)*self.m.Vpp_pi_inter + d1**2*self.m.Vpp_sigma_inter))/((d1**2 + dz_pp**2)*np.exp((1.j*d1*(kx + np.sqrt(3)*ky))/2.))
        H_k[9,21]=(d1*dz_pp*(1 + 1.j*np.sqrt(3) + (1 - 1.j*np.sqrt(3))*np.exp(1.j*np.sqrt(3)*d1*ky) - 2*np.exp((1.j*d1*(3*kx + np.sqrt(3)*ky))/2.))*(self.m.Vpp_pi_inter - self.m.Vpp_sigma_inter))/(4.*np.sqrt(2)*(d1**2 + dz_pp**2)*np.exp((1.j*d1*(kx + np.sqrt(3)*ky))/2.))
        H_k[9,22]=(d1**2*(1 - 1.j*np.sqrt(3) + (1 + 1.j*np.sqrt(3))*np.exp(1.j*np.sqrt(3)*d1*ky) - 2*np.exp((1.j*d1*(3*kx + np.sqrt(3)*ky))/2.))*(self.m.Vpp_pi_inter - self.m.Vpp_sigma_inter))/(8.*(d1**2 + dz_pp**2)*np.exp((1.j*d1*(kx + np.sqrt(3)*ky))/2.))
        #H_k[10,12]=0
        H_k[10,13]=-((dz_pd*self.m.Vpd_sigma_inter)/(np.sqrt(2)*np.sqrt(dz_pd**2)))
        #H_k[10,14]=0
        H_k[10,15]=(d1*dz_pp*(-1 + 1.j*np.sqrt(3) + (-1 - 1.j*np.sqrt(3))*np.exp(1.j*np.sqrt(3)*d1*ky) + 2*np.exp((1.j*d1*(3*kx + np.sqrt(3)*ky))/2.))*(self.m.Vpp_pi_inter - self.m.Vpp_sigma_inter))/(4.*np.sqrt(2)*(d1**2 + dz_pp**2)*np.exp((1.j*d1*(kx + np.sqrt(3)*ky))/2.))
        H_k[10,16]=((1 + np.exp(1.j*np.sqrt(3)*d1*ky) + np.exp((1.j*d1*(3*kx + np.sqrt(3)*ky))/2.))*(d1**2*self.m.Vpp_pi_inter + dz_pp**2*self.m.Vpp_sigma_inter))/(2.*(d1**2 + dz_pp**2)*np.exp((1.j*d1*(kx + np.sqrt(3)*ky))/2.))
        H_k[10,17]=(d1*dz_pp*(1 + 1.j*np.sqrt(3) + (1 - 1.j*np.sqrt(3))*np.exp(1.j*np.sqrt(3)*d1*ky) - 2*np.exp((1.j*d1*(3*kx + np.sqrt(3)*ky))/2.))*(self.m.Vpp_pi_inter - self.m.Vpp_sigma_inter))/(4.*np.sqrt(2)*(d1**2 + dz_pp**2)*np.exp((1.j*d1*(kx + np.sqrt(3)*ky))/2.))
        #H_k[10,18]=0
        #H_k[10,19]=0
        H_k[10,20]=(d1*dz_pp*(-1 + 1.j*np.sqrt(3) + (-1 - 1.j*np.sqrt(3))*np.exp(1.j*np.sqrt(3)*d1*ky) + 2*np.exp((1.j*d1*(3*kx + np.sqrt(3)*ky))/2.))*(self.m.Vpp_pi_inter - self.m.Vpp_sigma_inter))/(4.*np.sqrt(2)*(d1**2 + dz_pp**2)*np.exp((1.j*d1*(kx + np.sqrt(3)*ky))/2.))
        H_k[10,21]=((1 + np.exp(1.j*np.sqrt(3)*d1*ky) + np.exp((1.j*d1*(3*kx + np.sqrt(3)*ky))/2.))*(d1**2*self.m.Vpp_pi_inter + dz_pp**2*self.m.Vpp_sigma_inter))/(2.*(d1**2 + dz_pp**2)*np.exp((1.j*d1*(kx + np.sqrt(3)*ky))/2.))
        H_k[10,22]=(d1*dz_pp*(1 + 1.j*np.sqrt(3) + (1 - 1.j*np.sqrt(3))*np.exp(1.j*np.sqrt(3)*d1*ky) - 2*np.exp((1.j*d1*(3*kx + np.sqrt(3)*ky))/2.))*(self.m.Vpp_pi_inter - self.m.Vpp_sigma_inter))/(4.*np.sqrt(2)*(d1**2 + dz_pp**2)*np.exp((1.j*d1*(kx + np.sqrt(3)*ky))/2.))
        #H_k[11,12]=0
        #H_k[11,13]=0
        #H_k[11,14]=0
        H_k[11,15]=(d1**2*(1 + 1.j*np.sqrt(3) + (1 - 1.j*np.sqrt(3))*np.exp(1.j*np.sqrt(3)*d1*ky) - 2*np.exp((1.j*d1*(3*kx + np.sqrt(3)*ky))/2.))*(self.m.Vpp_pi_inter - self.m.Vpp_sigma_inter))/(8.*(d1**2 + dz_pp**2)*np.exp((1.j*d1*(kx + np.sqrt(3)*ky))/2.))
        H_k[11,16]=(d1*dz_pp*(-1 + 1.j*np.sqrt(3) + (-1 - 1.j*np.sqrt(3))*np.exp(1.j*np.sqrt(3)*d1*ky) + 2*np.exp((1.j*d1*(3*kx + np.sqrt(3)*ky))/2.))*(self.m.Vpp_pi_inter - self.m.Vpp_sigma_inter))/(4.*np.sqrt(2)*(d1**2 + dz_pp**2)*np.exp((1.j*d1*(kx + np.sqrt(3)*ky))/2.))
        H_k[11,17]=-0.25*((1 + np.exp(1.j*np.sqrt(3)*d1*ky) + np.exp((1.j*d1*(3*kx + np.sqrt(3)*ky))/2.))*((d1**2 + 2*dz_pp**2)*self.m.Vpp_pi_inter + d1**2*self.m.Vpp_sigma_inter))/((d1**2 + dz_pp**2)*np.exp((1.j*d1*(kx + np.sqrt(3)*ky))/2.))
        #H_k[11,18]=0
        #H_k[11,19]=(dz_pd*Subscript(V,p*d*Pi))/(np.sqrt(2)*np.sqrt(dz_pd**2))
        H_k[11,20]=(d1**2*(1 + 1.j*np.sqrt(3) + (1 - 1.j*np.sqrt(3))*np.exp(1.j*np.sqrt(3)*d1*ky) - 2*np.exp((1.j*d1*(3*kx + np.sqrt(3)*ky))/2.))*(self.m.Vpp_pi_inter - self.m.Vpp_sigma_inter))/(8.*(d1**2 + dz_pp**2)*np.exp((1.j*d1*(kx + np.sqrt(3)*ky))/2.))
        H_k[11,21]=(d1*dz_pp*(-1 + 1.j*np.sqrt(3) + (-1 - 1.j*np.sqrt(3))*np.exp(1.j*np.sqrt(3)*d1*ky) + 2*np.exp((1.j*d1*(3*kx + np.sqrt(3)*ky))/2.))*(self.m.Vpp_pi_inter - self.m.Vpp_sigma_inter))/(4.*np.sqrt(2)*(d1**2 + dz_pp**2)*np.exp((1.j*d1*(kx + np.sqrt(3)*ky))/2.))
        H_k[11,22]=-0.25*((1 + np.exp(1.j*np.sqrt(3)*d1*ky) + np.exp((1.j*d1*(3*kx + np.sqrt(3)*ky))/2.))*((d1**2 + 2*dz_pp**2)*self.m.Vpp_pi_inter + d1**2*self.m.Vpp_sigma_inter))/((d1**2 + dz_pp**2)*np.exp((1.j*d1*(kx + np.sqrt(3)*ky))/2.))

        # ------------------------------------------------
        # SOC
        H_k[23:45,23:45] = H_k[1:23,1:23]
        # spin1
        # even_up
        H_k[1,1]               = H_k[1,1]               + (-1.0)*self.m.lambda_M_up
        H_k[2,2]               = H_k[2,2]               +   0.0
        H_k[3,3]               = H_k[3,3]               + ( 1.0)*self.m.lambda_M_up
        H_k[4,4]               = H_k[4,4]               + (-1.0/2.0)*self.m.lambda_X2_up
        H_k[5,5]               = H_k[5,5]               +   0.0
        H_k[6,6]               = H_k[6,6]               + ( 1.0/2.0)*self.m.lambda_X2_up
        # odd_up
        H_k[7,7]               = H_k[7,7]               + (-1.0/2.0)*self.m.lambda_M_up
        H_k[8,8]               = H_k[8,8]               + ( 1.0/2.0)*self.m.lambda_M_up
        H_k[9,9]               = H_k[9,9]               + (-1.0/2.0)*self.m.lambda_X2_up
        H_k[10,10]             = H_k[10,10]             +  0.0
        H_k[11,11]             = H_k[11,11]             + (1.0/2.0)*self.m.lambda_X2_up            
        # even_down
        H_k[1+11,1+11]         = H_k[1+11,1+11]         + (-1.0)*self.m.lambda_M_down
        H_k[2+11,2+11]         = H_k[2+11,2+11]         +   0.0
        H_k[3+11,3+11]         = H_k[3+11,3+11]         + ( 1.0)*self.m.lambda_M_down
        H_k[4+11,4+11]         = H_k[4+11,4+11]         + (-1.0/2.0)*self.m.lambda_X2_down
        H_k[5+11,5+11]         = H_k[5+11,5+11]         +   0.0
        H_k[6+11,6+11]         = H_k[6+11,6+11]         + ( 1.0/2.0)*self.m.lambda_X2_down
        # odd_down
        H_k[7+11,7+11]         = H_k[7+11,7+11]         + (-1.0/2.0)*self.m.lambda_M_down
        H_k[8+11,8+11]         = H_k[8+11,8+11]         + ( 1.0/2.0)*self.m.lambda_M_down
        H_k[9+11,9+11]         = H_k[9+11,9+11]         + (-1.0/2.0)*self.m.lambda_X2_down
        H_k[10+11,10+11]       = H_k[10+11,10+11]       +   0.0
        H_k[11+11,11+11]       = H_k[11+11,11+11]       + ( 1.0/2.0)*self.m.lambda_X2_down
        # spin2
        # even_up            
        H_k[1+22,1+22]         = H_k[1+22,1+22]         - (-1.0)*self.m.lambda_M_up
        H_k[2+22,2+22]         = H_k[2+22,2+22]         -   0.0
        H_k[3+22,3+22]         = H_k[3+22,3+22]         - ( 1.0)*self.m.lambda_M_up
        H_k[4+22,4+22]         = H_k[4+22,4+22]         - (-1.0/2.0)*self.m.lambda_X2_up
        H_k[5+22,5+22]         = H_k[5+22,5+22]         -   0.0
        H_k[6+22,6+22]         = H_k[6+22,6+22]         - ( 1.0/2.0)*self.m.lambda_X2_up
        # odd_up
        H_k[7+22,7+22]         = H_k[7+22,7+22]         - (-1.0/2.0)*self.m.lambda_M_up
        H_k[8+22,8+22]         = H_k[8+22,8+22]         - ( 1.0/2.0)*self.m.lambda_M_up
        H_k[9+22,9+22]         = H_k[9+22,9+22]         - (-1.0/2.0)*self.m.lambda_X2_up
        H_k[10+22,10+22]       = H_k[10+22,10+22]       -  0.0
        H_k[11+22,11+22]       = H_k[11+22,11+22]       - (1.0/2.0)*self.m.lambda_X2_up            
        # even_down
        H_k[1+11+22,1+11+22]   = H_k[1+11+22,1+11+22]   - (-1.0)*self.m.lambda_M_down
        H_k[2+11+22,2+11+22]   = H_k[2+11+22,2+11+22]   -   0.0
        H_k[3+11+22,3+11+22]   = H_k[3+11+22,3+11+22]   - ( 1.0)*self.m.lambda_M_down
        H_k[4+11+22,4+11+22]   = H_k[4+11+22,4+11+22]   - (-1.0/2.0)*self.m.lambda_X2_down
        H_k[5+11+22,5+11+22]   = H_k[5+11+22,5+11+22]   -   0.0
        H_k[6+11+22,6+11+22]   = H_k[6+11+22,6+11+22]   - ( 1.0/2.0)*self.m.lambda_X2_down
        # odd_down
        H_k[7+11+22,7+11+22]   = H_k[7+11+22,7+11+22]   - (-1.0/2.0)*self.m.lambda_M_down
        H_k[8+11+22,8+11+22]   = H_k[8+11+22,8+11+22]   - ( 1.0/2.0)*self.m.lambda_M_down
        H_k[9+11+22,9+11+22]   = H_k[9+11+22,9+11+22]   - (-1.0/2.0)*self.m.lambda_X2_down
        H_k[10+11+22,10+11+22] = H_k[10+11+22,10+11+22] -  0.0
        H_k[11+11+22,11+11+22] = H_k[11+11+22,11+11+22] - (1.0/2.0)*self.m.lambda_X2_down  
        # even-odd spin mixing
        # layer up
        H_k[1,7+22]  = self.m.lambda_M_up                   # fixed typo: np.sqrt(3.0/2.0)*self.m.lambda_M_up 
        H_k[2,8+22]  = np.sqrt(3.0/2.0)*self.m.lambda_M_up  # fixed typo: self.m.lambda_M_up
        H_k[4,10+22] = self.m.lambda_X2_up/np.sqrt(2.0)
        H_k[5,11+22] = self.m.lambda_X2_up/np.sqrt(2.0)
        H_k[7,2+22]  = np.sqrt(3.0/2.0)*self.m.lambda_M_up  # fixed typo: self.m.lambda_M_up
        H_k[8,3+22]  = self.m.lambda_M_up                   # fixed typo: np.sqrt(3.0/2.0)*self.m.lambda_M_up
        H_k[9,5+22]  = self.m.lambda_X2_up/np.sqrt(2.0)
        H_k[10,6+22] = self.m.lambda_X2_up/np.sqrt(2.0)
        # layer down; AB stacking
        H_k[2+11,7+22+11]  = self.m.lambda_M_down                   # fixed typo: np.sqrt(3.0/2.0)*self.m.lambda_M_down
        H_k[3+11,8+22+11]  = np.sqrt(3.0/2.0)*self.m.lambda_M_down  # fixed typo: self.m.lambda_M_down
        H_k[4+11,10+22+11] = self.m.lambda_X2_down/np.sqrt(2.0)
        H_k[5+11,11+22+11] = self.m.lambda_X2_down/np.sqrt(2.0)
        H_k[7+11,1+22+11]  = np.sqrt(3.0/2.0)*self.m.lambda_M_down  # fixed typo: self.m.lambda_M_down
        H_k[8+11,2+22+11]  = self.m.lambda_M_down                   # fixed typo: np.sqrt(3.0/2.0)*self.m.lambda_M_down
        H_k[9+11,5+22+11]  = self.m.lambda_X2_down/np.sqrt(2.0)
        H_k[10+11,6+22+11] = self.m.lambda_X2_down/np.sqrt(2.0)  

        # E-field
        Vm = self.m.fel*self.m.d0/2.*au.Ehh  # self.m.fel is in Eh/aB, d0 in in aB => Vm is in eV
        Vt = self.m.fel*(self.m.d0/2.+self.m.d2_up)*au.Ehh # MB change 27Oct2024
        Vb = self.m.fel*(self.m.d0/2.-self.m.d2_up)*au.Ehh # MB change 27Oct2024
        dim4 = int(self.m.dim/4)
        for j in range(4):
            d4j = dim4*j
            for i in range(1,dim4+1):
                H_k[i+d4j,i+d4j] += Vm*(-1)**j
            H_k[4+d4j,9+d4j] += (Vt-Vb)/2.
            H_k[5+d4j,10+d4j] += (Vt-Vb)/2.  # fixed!
            H_k[6+d4j,11+d4j] += (Vt-Vb)/2.  
        #    
        H_kk = H_k[1:,1:]
        H_kk += np.conjugate(np.triu(H_kk, k=1)).T
        np.fill_diagonal(H_kk, H_kk.diagonal() + self.m.offset)
        return H_kk/au.Ehh


class EigenSolver:

    def __init__(self, model):
        self.model = model

    def solve_k(self, k, get_spin=False, get_vec=False):
        hamiltonian = self.model.build_tb_hamiltonian_new(k[0],k[1])
        if get_spin is False:
            if get_vec:
                return eigh(hamiltonian, eigvals_only=False)
            else:
                return eigh(hamiltonian, eigvals_only=True)[11:33]
        else:
            val, vec = eigh(hamiltonian, eigvals_only=False)
            vec2 = np.real(np.conjugate(vec)*vec)
            no_bands = int(vec2.shape[0]/2)
            vec2 = vec2.reshape((2,no_bands,-1))
            spin = np.sum(vec2[0,:,:], axis=0)-np.sum(vec2[1,:,:], axis=0)
            comp = np.sum(vec2, axis=0)
            comp = np.sum(comp[:11], axis=0)-np.sum(comp[11:], axis=0)  # layer composition
            #return val, spin, comp
            if get_vec:
                #return val[11:33], vec[:,11:33], spin[11:33]
                return val, vec, spin  # take all 44 bands
            else:
                #return val[11:33], spin[11:33], comp[11:33]
                return val, spin, comp
        
    def solve_at_points(self, k_points, get_spin=False, get_vec=False):
        if get_spin is False:
            if get_vec:
                vals = []
                vecs = []
                for k in k_points:
                    val, vec = self.solve_k(k, get_vec=True)
                    vals.append(val)
                    vecs.append(vec)
                return np.array(vals), np.array(vecs)
            else:
                return np.array([self.solve_k(k) for k in k_points])
        else:
            vals = []
            spins = []
            comps = []
            if get_vec: 
                vecs = []
                for k in k_points:
                    val, vec, spin = self.solve_k(k, get_spin=True, get_vec=True)
                    vals.append(val)
                    vecs.append(vec)
                    spins.append(spin)
                return np.array(vals), np.array(vecs), np.array(spins)                
            else:           
                for k in k_points:
                    val, spin, comp = self.solve_k(k, get_spin=True)
                    vals.append(val)
                    spins.append(spin)
                    comps.append(comp)
                return np.array(vals), np.array(spins), np.array(comps)

    def solve_BZ_path(self, get_spin=False):
        return self.solve_at_points(self.model.BZ_path, get_spin=get_spin)


class Plotting:
    """ Plotting utils.

    Attributes
    ----------
    grid_k : List[List]
        2d list containing full path within the BZ
    critical_points : List[Tuple]
        list of tuples containing critical points`s names and their coordinates
    """
    def __init__(self, model, directory=None):
        self.grid_k = model.BZ_path
        self.critical_points = model.critical_points
        self.critical_points_indices = model.critical_points_indices
        if directory:
            self.directory = os.path.join('./', directory)
            os.makedirs(directory, exist_ok=True)
        else:
            self.directory = './'

    def plot_Ek(self, Ek, x_label='k (nm$^{-1}$)', y_label='E (meV)'):
        """ Plots dispersion relation.

        Parameters
        ----------
        Ek : List[array]
            List of arrays of eigenvalues
        x_label : string
            label of x-axis
        y_label : string
            label of y-axis
        """
        _, ax = plt.subplots()
        ax.axes.set_aspect(.0035)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        # plot dispersion relation
        Ek = np.array(Ek)
        for band_idx in range(Ek.shape[1]):
            ax.plot((self.grid_k[:,0]+self.grid_k[:,1])/au.Ah ,Ek[:,band_idx]*au.Eh, label='Band' + str(band_idx))

        text_shift_x = (ax.get_xlim()[1] - ax.get_xlim()[0])*0.01
        plot_max_y = ax.get_ylim()[1]

        for (name, position) in self.critical_points:
             position_k=position/au.Ah
             ax.annotate(name, xy=(position_k-text_shift_x, plot_max_y), xytext=(position_k-text_shift_x, plot_max_y + 100))
             ax.axvline(x=position_k, linestyle='--', color='black')
        filename = 'ek.png'
        plt.savefig(os.path.join(self.directory, filename), bbox_inches='tight', dpi=200)
        plt.close()

    def plot_Ek_output_target(self, Ek_target, Ek_output1, plot_name, Ek_output2=None):
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
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.axes.set_aspect(3.5)
        ax.set_xlabel('k (nm$^{-1}$)')
        ax.set_ylabel('E (eV)')

        # plot dispersion relation
        Ek_target = np.array(Ek_target)
        Ek_output1 = np.array(Ek_output1)
        if Ek_output2 is not None:
            Ek_output2 = np.array(Ek_output2)
        for band_idx in range(Ek_target.shape[1]):
            ax.plot((self.grid_k[:,0]+self.grid_k[:,1])/au.Ah,Ek_target[:,band_idx], color='green', label='Target band')
            ax.plot((self.grid_k[:,0]+self.grid_k[:,1])/au.Ah,Ek_output1[:,band_idx], '--', color='blue', label='Fitted band')
            if Ek_output2 is not None:
                ax.plot((self.grid_k[:,0]+self.grid_k[:,1])/au.Ah,Ek_output2[:,band_idx], color='red', label='Decoder output')


        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right')

        text_shift_x = (ax.get_xlim()[1] - ax.get_xlim()[0])*0.01
        plot_max_y = ax.get_ylim()[1]

        for (name, position) in self.critical_points:
             position_k=position/au.Ah
             ax.annotate(name, xy=(position_k-text_shift_x, plot_max_y), xytext=(position_k-text_shift_x, plot_max_y + 0.1))
             ax.axvline(x=position_k, linestyle='--', color='black')
        filename = f'{plot_name}.png'
        plt.savefig(os.path.join(self.directory, "plots", filename), bbox_inches='tight', dpi=200)
        plt.close()

    def plot_Ek_output_target_s(self, target, output, plot_name):
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
        ax.set_xlabel('k (nm$^{-1}$)')
        ax.set_ylabel('E (eV)')

        k_path = np.linspace(0., 1., num=self.grid_k.shape[0])
        # plot dispersion relation
        Ek_target = np.array(target[0])
        spin_target = np.array(target[1])
        Ek_output = np.array(output[0])
        spin_output = np.array(output[1])
        
        for band_idx in range(Ek_target.shape[1]):
            ax.scatter(k_path, Ek_target[:,band_idx], s=pointsize, marker='.', c='k', cmap='bwr',label='Target band')
            ax.scatter(k_path, Ek_output[:,band_idx], s=pointsize, marker='.', c=spin_output[:,band_idx], cmap='bwr', label='Fitted band')

        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right')

        text_shift_x = (ax.get_xlim()[1] - ax.get_xlim()[0])*0.01
        plot_max_y = ax.get_ylim()[1]

        for (name, position_index) in self.critical_points_indices:
             position_k=k_path[position_index]
             ax.annotate(name, xy=(position_k-text_shift_x, plot_max_y), xytext=(position_k-text_shift_x, plot_max_y + 0.1))
             ax.axvline(x=position_k, linestyle='--', color='black')
        filename = f'{plot_name}.png'
        plt.savefig(os.path.join(self.directory, "plots", filename), bbox_inches='tight', dpi=400)
        plt.close()
        
    
    def plot_Ek_output_target_ss(self, target, output, plot_name):
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
        pointsize = 2.
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.axes.set_aspect(.2)
        ax.set_xlabel('k (nm$^{-1}$)')
        ax.set_ylabel('E (eV)')
        ax.set_ylim([-2,2.])

        k_path = np.linspace(0., 1., num=self.grid_k.shape[0])
        # plot dispersion relation
        Ek_target = np.array(target[0])
        spin_target = np.array(target[1])
        
        for band_idx in range(Ek_target.shape[1]):
            ax.scatter(k_path, Ek_target[:,band_idx], s=pointsize, marker='.', c='k', cmap='bwr',label='Target band')
            for out in output:
                Ek_output = np.array(out[0])
                spin_output = np.array(out[1])
                ax.scatter(k_path, Ek_output[:,band_idx], s=pointsize, marker='.', c=spin_output[:,band_idx], cmap='bwr', label='Fitted band')

        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right')

        text_shift_x = (ax.get_xlim()[1] - ax.get_xlim()[0])*0.01
        plot_max_y = ax.get_ylim()[1]

        for (name, position_index) in self.critical_points_indices:
             position_k=k_path[position_index]
             ax.annotate(name, xy=(position_k-text_shift_x, plot_max_y), xytext=(position_k-text_shift_x, plot_max_y + 0.1))
             ax.axvline(x=position_k, linestyle='--', color='black')
        filename = f'{plot_name}.png'
        plt.savefig(os.path.join(self.directory, "plots", filename), bbox_inches='tight', dpi=400)
        plt.close()    
        
    def plot_Ek_output_target0(self, Ek_target, Ek_output1, Ek_output2, Ek_output3):
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
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.axes.set_aspect(10.5)
        ax.set_xlabel('k (nm$^{-1}$)')
        ax.set_ylabel('E (eV)')
        
        
        # plot dispersion relation
        Ek_target = np.array(Ek_target)
        Ek_output1 = np.array(Ek_output1)
        if Ek_output2 is not None:
            Ek_output2 = np.array(Ek_output2)
        for band_idx in range(Ek_target.shape[1]):
            ax.plot((self.grid_k[:,0]+self.grid_k[:,1])/au.Ah,Ek_target[:,band_idx], color='green', label='Target band')
            ax.plot((self.grid_k[:,0]+self.grid_k[:,1])/au.Ah,Ek_output1[:,band_idx], color='orange', label='Fitted best')
            ax.plot((self.grid_k[:,0]+self.grid_k[:,1])/au.Ah,Ek_output2[:,band_idx], color='blue', label='Fitted band')
            ax.plot((self.grid_k[:,0]+self.grid_k[:,1])/au.Ah,Ek_output3[:,band_idx], color='red', label='Decoder output')
            
        
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right')
        
        text_shift_x = (ax.get_xlim()[1] - ax.get_xlim()[0])*0.01         
        plot_max_y = ax.get_ylim()[1]

        for (name, position) in self.critical_points:
             position_k=position/au.Ah
             ax.annotate(name, xy=(position_k-text_shift_x, plot_max_y), xytext=(position_k-text_shift_x, plot_max_y + 0.1))
             ax.axvline(x=position_k, linestyle='--', color='black')
        filename = 'ek_target0.png'
        plt.savefig(os.path.join(self.directory, filename), bbox_inches='tight', dpi=200)
        plt.close()

    def plot_Ek_output_target1(self, Ek_target, Ek_output1, Ek_output2=None):
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
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.axes.set_aspect(10.5)
        ax.set_xlabel('k (nm$^{-1}$)')
        ax.set_ylabel('E (eV)')

        # plot dispersion relation
        Ek_target = np.array(Ek_target)
        Ek_output1 = np.array(Ek_output1)
        if Ek_output2 is not None:
            Ek_output2 = np.array(Ek_output2)
        for band_idx in range(Ek_target.shape[1]):
            ax.plot((self.grid_k[:,0]+self.grid_k[:,1])/au.Ah,Ek_target[:,band_idx], color='green', label='Target band')
            ax.plot((self.grid_k[:,0]+self.grid_k[:,1])/au.Ah,Ek_output1[:,band_idx], color='blue', label='Fitted band')
            if Ek_output2 is not None:
                ax.plot((self.grid_k[:,0]+self.grid_k[:,1])/au.Ah,Ek_output2[:,band_idx], color='red', label='Decoder output')


        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right')

        text_shift_x = (ax.get_xlim()[1] - ax.get_xlim()[0])*0.01
        plot_max_y = ax.get_ylim()[1]

        for (name, position) in self.critical_points:
             position_k=position/au.Ah
             ax.annotate(name, xy=(position_k-text_shift_x, plot_max_y), xytext=(position_k-text_shift_x, plot_max_y + 0.1))
             ax.axvline(x=position_k, linestyle='--', color='black')
        filename = 'ek_target1.png'
        plt.savefig(os.path.join(self.directory, filename), bbox_inches='tight', dpi=200)
        plt.close()


def load_data_tomek(filename):
    return np.roll(np.loadtxt(filename)[::-1,1].reshape(101,12)[:,::2], 2, axis=0)  # roll to move CB minimum into K point

def load_data_kasia(filename):
    data = np.reshape(np.loadtxt(filename), (201,22,-1))[:131]
    return data[:,:,2], data[:,:,4]*2-1.

def load_k_path(filename):
    return np.loadtxt(filename)[:131,:2]

def load_np(filename):
    return np.load(filename)


def normalize(bands, parameters=None):
    min = np.amin(bands)
    max = np.amax(bands)
    if parameters is not None:
        parameters[-3:] -= min  # shift diagonals
        parameters /= max-min  # then scale
    return (bands-min)/(max-min)
