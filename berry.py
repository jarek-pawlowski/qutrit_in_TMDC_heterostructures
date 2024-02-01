import numpy as np
import utils_tb

from utils_tb_k import Flake, PlottingBands


lattice_const = 3.323
# --- Slater-Koster layers parameters
# layer up: MoSe2
Ed_up          = -0.09
Ep1_up         = -5.01
Ep0_up         = -5.30
Vdp_sigma_up   = -3.08
Vdp_pi_up      =  1.08
Vdd_sigma_up   = -0.94
Vdd_pi_up      =  0.75
Vdd_delta_up   =  0.13
Vpp_sigma_up   =  1.39
Vpp_pi_up      = -0.45
Ep1_odd_up     =  Ep1_up
Ep0_odd_up     =  Ep0_up
Ed_odd_up      =  Ed_up
lambda_M_up    =  0.186/2.  # 0.15  # it should be: 0.186d0/2.0d0
lambda_X2_up   =  0.200/2.  # 0.15  # it should be: 0.200d0/2.0d0
# layer down: WSe2
Ed_down        = -0.12
Ep1_down       = -4.17
Ep0_down       = -4.52
Vdp_sigma_down = -3.31
Vdp_pi_down    =  1.16
Vdd_sigma_down = -1.14
Vdd_pi_down    =  0.72
Vdd_delta_down =  0.17
Vpp_sigma_down =  1.16
Vpp_pi_down    = -0.25
Ep1_odd_down   =  Ep1_down
Ep0_odd_down   =  Ep0_down
Ed_odd_down    =  Ed_down 
lambda_M_down  =  0.472/2.  # 0.35  # it should be: 0.472d0/2.0d0
lambda_X2_down = -0.390/2.  # -0.20  # it should be: -0.390d0/2.0d0
# interlayer
Vpp_sigma_inter =  0.5  # positive
Vpp_pi_inter    = -0.5  # negative
Vdd_sigma_inter = -0.3  # negative
Vdd_pi_inter    =  0.3  # positive
Vdd_delta_inter = -0.6  # negative

material = utils_tb.Newmaterial(lattice_const,
                    Ed_up,Ep1_up,Ep0_up,Vdp_sigma_up,Vdp_pi_up,Vdd_sigma_up,Vdd_pi_up,Vdd_delta_up,Vpp_sigma_up,Vpp_pi_up,Ep1_odd_up,Ep0_odd_up,Ed_odd_up,lambda_M_up,lambda_X2_up,
                    Ed_down,Ep1_down,Ep0_down,Vdp_sigma_down,Vdp_pi_down,Vdd_sigma_down,Vdd_pi_down,Vdd_delta_down,Vpp_sigma_down,Vpp_pi_down,Ep1_odd_down,Ep0_odd_down,Ed_odd_down,lambda_M_down,lambda_X2_down,
                    Vpp_sigma_inter,Vpp_pi_inter,Vdd_sigma_inter,Vdd_pi_inter,Vdd_delta_inter, 0.)

k_path = utils_tb.load_k_path('kpointsDFT.dat')
lattice = utils_tb.Lattice(BZ_path=k_path)
lattice.select_k_indices(distance=4)
model = utils_tb.BandModel(material, lattice)
eigen_solver = utils_tb.EigenSolver(model)
plotting = utils_tb.Plotting(eigen_solver.model)
parameters = np.array([0.030411386321679234, -0.06768325945268493, -0.13985445526998835, 0.10354798970965703, 0.09511473252137233, 0.017192102201147058, 0.0009731466399938161, 0.09503724114298144, 0.006517343991442069, -0.01765029376948105, -0.139991453790443, 0.022210650586162457, -0.1399665270159982, 0.1396462854912086, -0.01873315173927232, -0.13965675237116856, 0.0479373707384002, -0.1399412022063997, -0.08320015408731994, 0.13994380829292616, 0.06816633552643377, -0.0807977909651734, -0.11578177646634323, -0.13999999976336608, -0.1395760909761852, -0.1398140425616252, -1.1006155026388882, -0.1546785496916816, -0.4117617630474648, 1.8318187602187703, -0.329984241003078, -0.21562176129413912])
# new Kasia
parameters = np.array([-5.958861367832076E-002, -5.07768325945268, -5.43985445526999, -2.97645201029034, 1.17511473252137, -0.922807897798853, 0.750973146639994, 0.225037241142981, 1.39651734399144, -0.467650293769481, -5.21767471324313, -5.41764380468383, -0.199555140694319, 1.964628549120861E-002, -4.18873315173927, -4.65965675237117, -3.26206262926160, 1.02005879779360, -1.22320015408732, 0.859943808292926, 0.238166335526434, 1.07920220903483, -0.365781776466343, -4.32873315150264, -4.79923284334735, -0.120167757070417, -1.10061550263889, -0.154678549691682, -0.500000000000000, 1.83181876021877, -0.329984241003078])

# define flake    
flake_edge_size = [300,300]  # [800,800] 
flake = Flake(material.a0, material.d0, flake_edge_size, shape='rhombus', find_neighbours=False)
flake.set_bfield(0.)  # in teslas

# build reciprocal lattice
flake.set_lattice_parameters(lattice)
flake.create_reciprocal_lattice()

#material.update_parameters(*parameters)
material.set_parameters(*parameters)
material.set_efield(-250.)

bands, vecs, spins  = eigen_solver.solve_at_points(k_points=flake.nodes_k, get_spin=True, get_vec=True)
np.save('bands.npy', bands)
np.save('vecs.npy', vecs)
np.save('spins.npy', spins)
# bands = np.load('bands.npy')
# vecs = np.load('vecs.npy')  # to index eigenstates: [no_k_point,:,no_eigenstate]
# spins = np.load('spins.npy')

def tensordot(a, b):
    return np.tensordot(a.conj().T, b, axes=([1,0]))

def FukuiU(a, b):
    # (Eg. 16) in https://arxiv.org/pdf/cond-mat/0503172.pdf
    return np.linalg.det(tensordot(a, b))


#flake.center_reciprocal_lattice()
berry = []
no_band = 17
# extended verion of Berry Curvature using (Eq. 1.14) in https://arxiv.org/pdf/0907.2021.pdf
#vec = vecs[:,:,no_band] # whole CB, only 22 elements in |Psi>
#vec2 = vecs[:,:,no_band:no_band+2]  # single CB subband, only 22 elements in |Psi>
#vec2 = vecs[:,:,:no_band]  
vec2 = vecs[:,:,11+17:11+18]  # single CB subband, take all 44 elements in |Psi>
#vec2 = vecs[:,:,11+17:]  # whole CB, take all 44 elements in |Psi>
for plaq in flake.plaquettes:
    # F = np.vdot(vec[plaq[0]], vec[plaq[3]])
    # F *= np.vdot(vec[plaq[1]], vec[plaq[0]])
    # F *= np.vdot(vec[plaq[2]], vec[plaq[1]])
    # F *= np.vdot(vec[plaq[3]], vec[plaq[2]])    
    F = FukuiU(vec2[plaq[0]], vec2[plaq[3]])
    F *= FukuiU(vec2[plaq[1]], vec2[plaq[0]])
    F *= FukuiU(vec2[plaq[2]], vec2[plaq[1]])
    F *= FukuiU(vec2[plaq[3]], vec2[plaq[2]])
    F = np.angle(F) 
    kx = flake.nodes_k[plaq[0],0]+flake.nodes_k[plaq[1],0]+flake.nodes_k[plaq[2],0]+flake.nodes_k[plaq[3],0]
    ky = flake.nodes_k[plaq[0],1]+flake.nodes_k[plaq[1],1]+flake.nodes_k[plaq[2],1]+flake.nodes_k[plaq[3],1]
    berry.append([kx,ky,F])
berry = np.array(berry)
berry[np.abs(berry[:,2])>.001]=0.

plot = PlottingBands(flake, directory='plots')
plot.plot_berry_k(berry)
plot.plot_energy_surface_kk(flake.nodes_k, bands[:,11+17])

bands, spins, _  = eigen_solver.solve_BZ_path(get_spin=True)
plotting.plot_Ek_output_target_s([bands, spins], [bands, spins], plot_name='berry_e')