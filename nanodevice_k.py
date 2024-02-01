import numpy as np

import utils as utils
au = utils.au

material = utils.MoS2 
# choose bands
# MoS2: VB_max = [12,13], CB_min = [14,15]
bands = [2,3]
# energy offset
# MoS2: eoff = 40.3/utils.au.Eh

# define flake    
flake_edge_size = [60,60]   # poznie wiekszy platek [240,240]
flake = utils.Flake(material.a0, flake_edge_size, shape='rhombus', find_neighbours=False)
flake.set_bfield(0.)  # in teslas
#
plot_flake = utils.PlottingOnFlake(flake, directory='results')
plot_flake.plot_flake_lattice(plot_links=False)
# setup confinement potential
flake.potential = np.zeros(flake.no_of_nodes)
flake.electric_field = np.zeros(flake.no_of_nodes)

# build reciprocal lattice
flake.set_lattice_parameters(utils.Lattice())
flake.create_reciprocal_lattice()

# build flake model and construct k-space Hamiltonian
print("constructing plane-wave basis...")
flake_model = utils.FlakeModel(flake, material)
basis_k = utils.Planewaves(flake, flake_model)
basis_k.select_subspace(flake.K_points, 12.5)
plot_flake.plot_flake_lattice_k(subsets=basis_k.subspaces)

xg = 0.0/utils.au.Ah  # 0.03/utils.au.Ah = 1 V / 10 nm
yg = 0.0/utils.au.Ah
sigma = 0.319
u0 = -1000.

angle = 0. # np.pi/2.
for i, n in enumerate(flake.nodes):
    x = n[0] + xg
    y = n[1] + yg
    #flake.potential[i] = 0.0001 
    #r = x*np.cos(angle) + y*np.sin(angle) # - 150.
    r = np.sqrt(x*x+y*y)
    #flake.potential[i] = (np.exp(-r**2/(10./utils.au.Ah)**2/2.)-1.)*-100./utils.au.Eh
    #flake.potential[i] = (np.exp(-r**2/(1./utils.au.Ah)**2/2.)-1.)*-1000./utils.au.Eh
    flake.potential[i] = (np.exp(-r**2/(sigma/utils.au.Ah)**2/2.)-1.)*u0/utils.au.Eh
    #if x > 0.:
    #    flake.potential[i] +=100000./utils.au.Eh
            
plot_flake.plot_potential_flake(flake.potential)
plot_flake.plot_electric_field_flake(flake.electric_field)

# build plane-waves basis and calculate matrix elements
basis_k.build_basis(no_of_bands=bands, energy_offset=0.)  
plot_bands = utils.PlottingBands(flake, directory='results')
plot_bands.plot_Ek(basis_k.solve_BZ_path())
plot_bands.plot_energy_surface_k(subsets=basis_k.subspaces, energy=basis_k.basis_energies[1::2])
basis_k.potential_elements(sign=1)

flake_solver = utils.FlakeMethods(flake, material, basis_k=basis_k)
plot_kq = utils.PlottingFourier(flake_solver, directory='results')
plot_kq.plot_elements_kmq(basis_k.elements_kmq, l_index=0)

# build Hamiltonian in plane-waves basis
print("building Hamiltonian...")
hamiltonian = basis_k.build_hamiltonian()
# solve eigenploblem
print("solving eigenproblem...")
es = utils.EigenSolver()
# calculate eigenvalues with indices in a given range
dim_h = basis_k.subspaces_total_size*len(basis_k.no_of_bands)
dim_h = 4
eigenvalues, eigenvectors = es.solve_eigenproblem_arpack_dense(hamiltonian,
                                subset_by_index = [0, dim_h-1],
                                reverse = False,
                                calculate_eigenvectors = True)
#eigenvalues, eigenvectors = es.solve_eigenproblem_feast(hamiltonian,
#                                eigenvalues_subspace_size = k,
#                                energy_min = e0,
#                                energy_max = e1,
#                                comments=True)

startstates = 0
endstates = 4

sv = eigenvalues[startstates:endstates]*au.Eh
Eso = (sv[2]+sv[3])/2-(sv[0]+sv[1])/2.
gvmgs = (sv[1]-sv[0])/0.05788
gvpgs = (sv[3]-sv[2])/0.05788
gs = (gvpgs+gvmgs)/2.
gv = (gvpgs-gvmgs)/2.
print(Eso,gs,gv)

# evaluate results
Lz_0 = flake_solver.calculate_Lz(flake_edge_size[0], eigenvectors[:,0])
Lz_1 = flake_solver.calculate_Lz(flake_edge_size[0], eigenvectors[:,1])
Lz_2 = flake_solver.calculate_Lz(flake_edge_size[0], eigenvectors[:,2])
Lz_3 = flake_solver.calculate_Lz(flake_edge_size[0], eigenvectors[:,3])
gvmgs = (Lz_1-Lz_0)
gvpgs = (Lz_3-Lz_2)
print(Eso, (gvpgs-gvmgs)/2.)  # valley g-factor, by policzyc spiniwy trzebeby obłozyc sigma_z
#print(xg, yg, np.real((gvpgs-gvmgs)/2.))
#gv_collect.append([xg,yg,np.real((gvpgs-gvmgs)/2.)])
print(sigma, u0, np.real((gvpgs-gvmgs)/2.))
#gv_collect.append([sigma, u0, np.real((gvpgs-gvmgs)/2.)])

#gv_collect = np.array(gv_collect)
#np.save('./results/gvs', gv_collect, allow_pickle=True)
#np.savetxt('./results/gvs.txt', gv_collect)

densities, spin_valley, k_max, std_y, states = flake_solver.calculate_densities_k(eigenvectors[:,startstates:endstates],
                                                                                  calculate_spin_valley=True, 
                                                                                  calculate_real_states=False, 
                                                                                  every_n=1)
#                                                                                 
plot_bands.plot_eigenvalues_sv(eigenvalues[startstates:endstates], spin_valley, pointsize=20.)
plot_bands.plot_eigenvalues_svk(eigenvalues[startstates:endstates], spin_valley, k_max, pointsize=20.)
plot_bands.plot_eigenvalues_svk_(eigenvalues[startstates:endstates], spin_valley, k_max, pointsize=20.)
plot_bands.plot_eigenvalues_bnk(eigenvalues[startstates:endstates], std_y, k_max, pointsize=20.)

plot_flake.plot_statedensity(densities[0])
flake_solver.save_densities()

"""
plot_kq = utils.PlottingFourier(flake_solver, directory='results')
#plot_kq.plot_elements_kmq(basis_k.elements_kmq, l_index=0)

# sort single-hole states
states = states[::-1,:]
e_order = eigenvalues[-laststates:][::-1]
spin_valley_order = spin_valley[::-1,:]
k_order = k_max[::-1,:]
for i in range(laststates-1):
    if np.abs(e_order[i]-e_order[i+1])*utils.au.Eh < 1.e-4:
        if k_order[i,1] > k_order[i+1,1]:
            s_copy = np.copy(states[i])
            states[i] = np.copy(states[i+1])
            states[i+1] = s_copy
            sv_copy = np.copy(spin_valley_order[i])
            spin_valley_order[i] = np.copy(spin_valley_order[i+1])
            spin_valley_order[i+1] = sv_copy
            k_copy = np.copy(k_order[i])
            k_order[i] = np.copy(k_order[i+1])
            k_order[i+1] = k_copy
            e_copy = np.copy(e_order[i])
            e_order[i] = np.copy(e_order[i+1])
            e_order[i+1] = e_copy
print(e_order)
print(k_order[:,1])
print(spin_valley_order)

# save stuff to calculate conductance: eigv, S, V, kx_, k_y, std_y
utils.save_npy('landauer.npy', np.stack((eigenvalues[-laststates:],spin_valley[:,0],spin_valley[:,1],k_max[:,0],k_max[:,1],std_y)).T)
# save stuff for Coulomb

np.savetxt('./coulomb_sf/nodes.dat', np.c_[flake.nodes[:,0],flake.nodes[:,1]])
np.savetxt('./coulomb_sf/nodestype.dat', flake.nodestype, fmt='%i')
np.savetxt('./coulomb_sf/base.dat', np.c_[np.real(states.flatten()),np.imag(states.flatten())])
np.savetxt('./coulomb_sf/esk.dat', np.c_[e_order, spin_valley_order[:,0], spin_valley_order[:,1]])
"""
