import numpy as np
import utils_tb

import utils_tb_k as utils

# lets define parameters of full TB model here and cross-check with MB Matlab code
lattice_const = 0.3323 #in nm
d2_up         = 0.1669 #in nm
d2_down       = 0.1680 #in nm
d0            = 0.6400 #in nm

lambda_M_up    =  0.093
lambda_M_down  =  0.236
lambda_X2_up   =  0.175
lambda_X2_down = -0.275

# layer up: MoSe2
Ed_up          = -0.140
Ep1_up         = -5.060
Ep0_up         = -5.268
Vdp_sigma_up   = -2.980
Vdp_pi_up      =  1.073
Vdd_sigma_up   = -1.028
Vdd_pi_up      =  0.838
Vdd_delta_up   =  0.229
Vpp_sigma_up   =  1.490
Vpp_pi_up      = -0.550
Ep1_odd_up     = -5.051
Ep0_odd_up     = -5.236
Ed_odd_up      = -0.112

# layer down: WSe2
Ed_down        = -0.220
Ep1_down       = -4.234
Ep0_down       = -4.620
Vdp_sigma_down = -3.251
Vdp_pi_down    =  1.060
Vdd_sigma_down = -1.240
Vdd_pi_down    =  0.817
Vdd_delta_down =  0.255
Vpp_sigma_down =  1.258
Vpp_pi_down    = -0.350
Ep1_odd_down   = -4.132
Ep0_odd_down   = -4.554
Ed_odd_down    = -0.032

# interlayer
Vpp_sigma_inter =  1.393
Vpp_pi_inter    = -0.012
Vdd_sigma_inter =  0.196
Vdd_pi_inter    = -0.480
Vdd_delta_inter =  0.091

Vdp_sigma_inter  = -0.044;
Vpd_sigma_inter  = -1.344;
Vdp_pi_inter     =  0.411;
Vpd_pi_inter     = -1.780;

offset          = 0.0

material = utils_tb.Newmaterial(lattice_const, d2_up, d2_down, d0,
                    Ed_up,Ep1_up,Ep0_up,Vdp_sigma_up,Vdp_pi_up,Vdd_sigma_up,Vdd_pi_up,Vdd_delta_up,Vpp_sigma_up,Vpp_pi_up,Ep1_odd_up,Ep0_odd_up,Ed_odd_up,lambda_M_up,lambda_X2_up,
                    Ed_down,Ep1_down,Ep0_down,Vdp_sigma_down,Vdp_pi_down,Vdd_sigma_down,Vdd_pi_down,Vdd_delta_down,Vpp_sigma_down,Vpp_pi_down,Ep1_odd_down,Ep0_odd_down,Ed_odd_down,lambda_M_down,lambda_X2_down,
                    Vpp_sigma_inter,Vpp_pi_inter,Vdd_sigma_inter,Vdd_pi_inter,Vdd_delta_inter,Vdp_sigma_inter,Vpd_sigma_inter,Vdp_pi_inter,Vpd_pi_inter,
                    offset)

#material.set_efield(0.)  # in mV/nm
material.set_efield(1000.)  # in mV/nm

k_path = utils_tb.load_k_path('kpointsDFT.dat')
lattice = utils_tb.Lattice(BZ_path=k_path)
lattice.select_k_indices(distance=4)
model = utils_tb.BandModel(material, lattice)
eigen_solver = utils_tb.EigenSolver(model)
plotting = utils_tb.Plotting(eigen_solver.model)

#parameters = np.array([0.030411386321679234, -0.06768325945268493, -0.13985445526998835, 0.10354798970965703, 0.09511473252137233, 0.017192102201147058, 0.0009731466399938161, 0.09503724114298144, 0.006517343991442069, -0.01765029376948105, -0.139991453790443, 0.022210650586162457, -0.1399665270159982, 0.1396462854912086, -0.01873315173927232, -0.13965675237116856, 0.0479373707384002, -0.1399412022063997, -0.08320015408731994, 0.13994380829292616, 0.06816633552643377, -0.0807977909651734, -0.11578177646634323, -0.13999999976336608, -0.1395760909761852, -0.1398140425616252, -1.1006155026388882, -0.1546785496916816, -0.4117617630474648, 1.8318187602187703, -0.329984241003078, -0.21562176129413912])
# new Kasia
#parameters = np.array([-5.958861367832076E-002, -5.07768325945268, -5.43985445526999, -2.97645201029034, 1.17511473252137, -0.922807897798853, 0.750973146639994, 0.225037241142981, 1.39651734399144, -0.467650293769481, -5.21767471324313, -5.41764380468383, -0.199555140694319, 1.964628549120861E-002, -4.18873315173927, -4.65965675237117, -3.26206262926160, 1.02005879779360, -1.22320015408732, 0.859943808292926, 0.238166335526434, 1.07920220903483, -0.365781776466343, -4.32873315150264, -4.79923284334735, -0.120167757070417, -1.10061550263889, -0.154678549691682, -0.500000000000000, 1.83181876021877, -0.329984241003078])
#material.update_parameters(*parameters)
#material.set_parameters(*parameters)

# define flake    
flake_edge_size = [102,102]  # [800,800] 
flake = utils.Flake(material.a0, material.d0, flake_edge_size, shape='rhombus', find_neighbours=False)
flake.set_bfield(0.)  # in teslas

plot_flake = utils.PlottingOnFlake(flake, directory='results')
plot_flake.plot_flake_lattice(plot_links=False)

# build reciprocal lattice
flake.set_lattice_parameters(lattice)
flake.create_reciprocal_lattice()
# build flake model and construct k-space Hamiltonian
print("constructing plane-wave basis...")
basis_k = utils.Planewaves(flake, model)
#basis_k.select_subspace(flake.K_points, 1.5)
basis_k.select_subspace(flake.Q_points, 1.5)  # 14.
#basis_k.select_subspace_half()
plot_flake.plot_flake_lattice_k(subsets=basis_k.subspaces)

# setup confinement potential
flake.potential = np.zeros(flake.no_of_nodes)
# Q2'-Q3' coupling:
angle = 0.  # np.pi/2.
for i, n in enumerate(flake.nodes):
    x = n[0]
    y = n[1]
    r = x*np.cos(angle) + y*np.sin(angle)
    d = np.sqrt(x**2+y**2-r**2)
    #flake.potential[i] = (np.exp(-r**2/(.15/utils.au.Ah)**2/2.)-1.)*300./utils.au.Eh  # type-II barrier amplitude
    #flake.potential[i] += np.exp(-d**2/(3./utils.au.Ah)**2/2.)*1./utils.au.Eh
    #flake.potential[i] -= (np.exp(-((x-6.8/utils.au.Ah)**2+(y+4.6/utils.au.Ah)**2)/(5./utils.au.Ah)**2/2.)-1.)*250./utils.au.Eh
    #flake.potential[i] = (np.exp(-(x**2+y**2)/(5./utils.au.Ah)**2/2.)-1.)*250./utils.au.Eh
    flake.potential[i] = (np.exp(-(x**2+y**2)/(5./utils.au.Ah)**2/2.)-1.)*100./utils.au.Eh

# Q1'-Q3 coupling:
# angle = -np.pi/6.
# for i, n in enumerate(flake.nodes):
#     x = n[0]
#     y = n[1]
#     r = x*np.cos(angle) + y*np.sin(angle)
#     d = np.sqrt(x**2+y**2-r**2)
#     flake.potential[i] = (np.exp(-r**2/(.15/utils.au.Ah)**2/2.)-1.)*300./utils.au.Eh  # type-II barrier amplitude
#     flake.potential[i] += np.exp(-d**2/(3./utils.au.Ah)**2/2.)*1./utils.au.Eh

plot_flake.plot_potential_flake(flake.potential)

# build plane-waves basis and calculate matrix elements
basis_k.build_basis(no_of_bands=[11+17,11+18], energy_offset=0.)  
plot_bands = utils.PlottingBands(flake, directory='results')
plot_bands.plot_Ek(basis_k.solve_BZ_path())
plot_bands.plot_energy_surface_k(subsets=basis_k.subspaces, energy=basis_k.basis_energies[1::2])
plot_bands.plot_spin_surface_k(subsets=basis_k.subspaces, spin=basis_k.basis_spins[1::2])
basis_k.potential_elements(sign=-1)

flake_solver = utils.FlakeMethods(flake, material, basis_k=basis_k)
plot_kq = utils.PlottingFourier(flake_solver, directory='results')
plot_kq.plot_elements_kmq(basis_k.elements_kmq, l_index=0)

"""
# build Hamiltonian in plane-waves basis
print("Calculating coupling elements...")
dressed_elements = basis_k.build_hamiltonian(include_diagonal=False)
summed_elements = basis_k.sum_elements(dressed_elements, flake.Q_points[4], 1.)
plot_kq.plot_dressed_elements(basis_k.basis_k, summed_elements[0], suffix='0', savetofile='Q2_Q3.txt')
plot_kq.plot_dressed_elements(basis_k.basis_k, summed_elements[1], suffix='1')
plot_kq.plot_dressed_elements(basis_k.basis_k, summed_elements[2], suffix='2')
plot_kq.plot_dressed_elements(basis_k.basis_k, summed_elements[3], suffix='3')
"""

# build Hamiltonian in plane-waves basis
print("building Hamiltonian...")
hamiltonian = basis_k.build_hamiltonian()
# solve eigenploblem
print("solving eigenproblem...")
es = utils.EigenSolver()
# calculate eigenvalues with indices in a given range
dim_h = basis_k.subspaces_total_size*len(basis_k.no_of_bands)
eigenvalues, eigenvectors = es.solve_eigenproblem_arpack_dense(hamiltonian,
                                subset_by_index = [0, dim_h-1],
                                reverse = False,
                                calculate_eigenvectors = True)

print(eigenvalues[:10]*utils.au.Ehh)
plot_bands.plot_eigenvalues(eigenvalues)
np.save('./results/eigenvalues', eigenvalues, allow_pickle=True)

# evaluate results
firststates = 20 
flake_solver = utils.FlakeMethods(flake, material, basis_k=basis_k)
densities, spin_valley, k_max, std_y, states = flake_solver.calculate_densities_k(eigenvectors[:,:firststates],
                                                                                  calculate_spin_valley=True, 
                                                                                  calculate_real_states=True, 
                                                                                  every_n=1)
plot_bands.plot_eigenvalues_sv(eigenvalues[:firststates], spin_valley, pointsize=20.)
plot_bands.plot_eigenvalues_svk(eigenvalues[:firststates], spin_valley, k_max, pointsize=20.)
plot_bands.plot_eigenvalues_svk_(eigenvalues[:firststates], spin_valley, k_max, pointsize=20.)
plot_bands.plot_eigenvalues_bnk(eigenvalues[:firststates], std_y, k_max, pointsize=20.)
plot_flake.plot_statedensity(densities[0])
flake_solver.save_densities()