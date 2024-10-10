import numpy as np
import utils_tb

import utils_tb_k as utils


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

parameters = np.array([0.030411386321679234, -0.06768325945268493, -0.13985445526998835, 0.10354798970965703, 0.09511473252137233, 0.017192102201147058, 0.0009731466399938161, 0.09503724114298144, 0.006517343991442069, -0.01765029376948105, -0.139991453790443, 0.022210650586162457, -0.1399665270159982, 0.1396462854912086, -0.01873315173927232, -0.13965675237116856, 0.0479373707384002, -0.1399412022063997, -0.08320015408731994, 0.13994380829292616, 0.06816633552643377, -0.0807977909651734, -0.11578177646634323, -0.13999999976336608, -0.1395760909761852, -0.1398140425616252, -1.1006155026388882, -0.1546785496916816, -0.4117617630474648, 1.8318187602187703, -0.329984241003078, -0.21562176129413912])
# new Kasia
parameters = np.array([-5.958861367832076E-002, -5.07768325945268, -5.43985445526999, -2.97645201029034, 1.17511473252137, -0.922807897798853, 0.750973146639994, 0.225037241142981, 1.39651734399144, -0.467650293769481, -5.21767471324313, -5.41764380468383, -0.199555140694319, 1.964628549120861E-002, -4.18873315173927, -4.65965675237117, -3.26206262926160, 1.02005879779360, -1.22320015408732, 0.859943808292926, 0.238166335526434, 1.07920220903483, -0.365781776466343, -4.32873315150264, -4.79923284334735, -0.120167757070417, -1.10061550263889, -0.154678549691682, -0.500000000000000, 1.83181876021877, -0.329984241003078])
#material.update_parameters(*parameters)
material.set_parameters(*parameters)
material.set_efield(100.)  # in mV/nm

# define flake    
flake_edge_size = [200,200]  # [800,800] 
flake = utils.Flake(material.a0, material.d0, flake_edge_size, shape='rhombus', find_neighbours=False)
flake.set_bfield(0.)  # in teslas

plot_flake = utils.PlottingOnFlake(flake, directory='results')

# setup confinement potential
flake.potential = np.zeros(flake.no_of_nodes)

for i, n in enumerate(flake.nodes):
    x = n[0]
    y = n[1]
    # well
    angle = 0.
    dz = 11.
    r1 = (x-dz/utils.au.Ah)*np.cos(angle) + y*np.sin(angle)
    angle = -np.pi/3.
    dz = -3.4*np.sqrt(3.)
    r2 = (x-dz/utils.au.Ah)*np.cos(angle) + y*np.sin(angle)
    angle = np.pi/3.    
    dz = -12.6*np.sqrt(3.)
    r3 = (x-dz/utils.au.Ah)*np.cos(angle) + y*np.sin(angle)
    if r1 < 0. and r2 > 0. and r3 > 0.:
        flake.potential[i] -= 50./utils.au.Eh
    else:
        flake.potential[i] += 50./utils.au.Eh
    # junction
    jf = True
    if x < -14./utils.au.Ah or -x/np.sqrt(3) < y-16./utils.au.Ah or x/np.sqrt(3) > y+25./utils.au.Ah:
        jf = False
    if jf:           
        angle = 0.  # np.pi/2.
        dz = 11.
        dd = .2
        rl = (x-(dz-dd)/utils.au.Ah)*np.cos(angle) + y*np.sin(angle)
        rr = (x-(dz+dd)/utils.au.Ah)*np.cos(angle) + y*np.sin(angle)
        flake.potential[i] -= (np.exp(-rl**2/(.2/utils.au.Ah)**2/2.)-1.)*100./utils.au.Eh
        flake.potential[i] += (np.exp(-rr**2/(.2/utils.au.Ah)**2/2.)-1.)*100./utils.au.Eh
        #d = np.sqrt(x**2+y**2-r**2)
        #flake.potential[i] += np.exp(-d**2/(3./utils.au.Ah)**2/2.)*10./utils.au.Eh
        angle = -np.pi/3.
        dz = -3.4*np.sqrt(3.)
        dd = .2*np.sqrt(3.)
        rl = (x-(dz+dd)/utils.au.Ah)*np.cos(angle) + y*np.sin(angle)
        rr = (x-(dz-dd)/utils.au.Ah)*np.cos(angle) + y*np.sin(angle)
        flake.potential[i] -= (np.exp(-rl**2/(.2/utils.au.Ah)**2/2.)-1.)*100./utils.au.Eh
        flake.potential[i] += (np.exp(-rr**2/(.2/utils.au.Ah)**2/2.)-1.)*100./utils.au.Eh   
        #
        angle = np.pi/3.   
        dz = -12.6*np.sqrt(3.)
        dd = .2*np.sqrt(3.)
        rl = (x-(dz+dd)/utils.au.Ah)*np.cos(angle) + y*np.sin(angle)
        rr = (x-(dz-dd)/utils.au.Ah)*np.cos(angle) + y*np.sin(angle)
        flake.potential[i] -= (np.exp(-rl**2/(.2/utils.au.Ah)**2/2.)-1.)*100./utils.au.Eh
        flake.potential[i] += (np.exp(-rr**2/(.2/utils.au.Ah)**2/2.)-1.)*100./utils.au.Eh
    # gaussian
    flake.potential[i] -= (np.exp(-((x-6.8/utils.au.Ah)**2+(y+4.6/utils.au.Ah)**2)/(5./utils.au.Ah)**2/2.)-1.)*250./utils.au.Eh

plot_flake.plot_potential_flake(flake.potential, filename='potential_triangle')
np.savetxt('./results/triangledot.txt', np.c_[flake.nodes[:,0]*utils.au.Ah,flake.nodes[:,1]*utils.au.Ah,  flake.potential*utils.au.Eh])
