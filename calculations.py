import os

from functools import partial
import itertools
import math, time, cmath
from math import cos, exp, sin, log, log10, pi, sqrt
import random
import numpy
from numpy import matrix, array, zeros
from pytriqs.operators import *
from pytriqs.archive import *
from pytriqs.gf.local import *
from pytriqs.arrays import BlockMatrix, BlockMatrixComplex
import pytriqs.utility.mpi as mpi

from copy import deepcopy
############################################## MAIN CODES ###################################
from nested_scripts import *


def nested_calculation( clusters, nested_struct_archive_name = None, flexible_Gweiss=False, sign=-1, sign_up_to=2, 
                        Us = [1.0],
                        Ts = [0.125], 
                        ns = [0.5], fixed_n = True,
                        mutildes = [0.0],
                        dispersion = lambda kx, ky: epsilonk_square(kx,ky, 0.25), ph_symmetry = True,
                        use_cumulant = False, 
                        n_ks = [24], n_k_automatic = False, n_k_rules = [[0.06, 32],[0.03, 48],[0.005, 64],[0.00, 96]],
                        w_cutoff = 20.0,
                        min_its = 5, max_its=25, 
                        mix_Sigma = False, rules = [[0, 0.5], [6, 0.2], [12, 0.65]],                        
                        do_dmft_first = False, 
                        use_cthyb = False,
                        alpha = 0.5, delta = 0.1,  automatic_alpha_and_delta = False,
                        n_cycles=10000000, 
                        max_time_rules= [ [1, 5*60], [2, 20*60], [4, 80*60], [8, 200*60], [16,400*60] ], time_rules_automatic=False, exponent = 0.7, overall_prefactor=1.0, no_timing = False,
                        accuracy = 1e-4, 
                        solver_data_package = None,
                        print_current = 1,
                        insulating_initial = False,
                        initial_guess_archive_name = '', suffix=''):

  if mpi.is_master_node():
    print "WELCOME TO %snested calculation!"%("cumul_" if use_cumulant else "")
    if n_k_automatic: print "n_k automatic!!!"
  if len(n_ks)==0 and n_k_automatic: n_ks=[0]

  if use_cthyb:
    solver_class = solvers.cthyb
  else:
    solver_class = solvers.ctint

  fermionic_struct = {'up': [0]}

  if mpi.is_master_node(): print "nested structure: "
  if not (nested_struct_archive_name is None):
    try:
      nested_scheme = nested_struct.from_file(nested_struct_archive_name)
      if mpi.is_master_node(): print "nested structure loaded from file",nested_struct_archive_name 
    except:  
      nested_scheme = nested_struct(clusters)
      nested_scheme.print_to_file(nested_struct_archive_name) 
      if mpi.is_master_node(): print "nested structure printed to file",nested_struct_archive_name 
  else:
    nested_scheme = nested_struct(clusters)
  if mpi.is_master_node(): print nested_scheme.get_tex()

  impurity_struct = nested_scheme.get_impurity_struct()

  if not time_rules_automatic:
    max_times = {}
    for C in impurity_struct:
      for r in max_time_rules:
        if r[0]<=len(impurity_struct[C]):
          max_times[C] = r[1]
    if mpi.is_master_node(): print "max_times from rules: ",max_times


  beta = 1.0/Ts[0] 
  
  n_iw = int(((w_cutoff*beta)/math.pi-1.0)/2.0)
  if mpi.is_master_node():
    print "PM HUBBARD GW: n_iw: ",n_iw

  if not n_k_automatic:
    n_k = n_ks[0]
    print "n_k = ", n_k
  else:
    n_k = n_k_from_rules(Ts[0], n_k_rules)
    #if mpi.is_master_node(): print "n_k automatic!!!"

  dt = nested_data(  n_iw = n_iw, 
                     n_k = n_k, 
                     beta = beta, 
                     impurity_struct = impurity_struct,
                     fermionic_struct = fermionic_struct,
                     archive_name="so_far_nothing_you_shouldnt_see_this_file"  )
  if use_cumulant:
    dt.__class__ = cumul_nested_data
    dt.promote() 

  if fixed_n:
    ps = itertools.product(n_ks,ns,Us,Ts)
  else:
    ps = itertools.product(n_ks,mutildes,Us,Ts)

  counter = 0
  old_nk = n_k
  old_beta = beta

  for p in ps:    
    #name stuff to avoid confusion   
    nk = (p[0] if (not n_k_automatic) else n_k_from_rules(T, n_k_rules) )
    if fixed_n:
      n = p[1]
    else:
      mutilde = p[1]
      n = None
    U = p[2]
    T = p[3] 
    beta = 1.0/T

    if nk!=old_nk and (not n_k_automatic):
      dt.change_ks(IBZ.k_grid(nk))

    if beta!=old_beta:
      n_iw = int(((w_cutoff*beta)/math.pi-1.0)/2.0)
      if n_k_automatic:
        nk = n_k_from_rules(T, n_k_rules)
        if nk != old_nk: 
          dt.change_ks(IBZ.k_grid(nk))
      dt.change_beta(beta, n_iw)
 
    old_beta = beta
    old_nk = nk
    nested_scheme.set_nk(nk) #don't forget this part

    filename = "result"
    if len(n_ks)>1 and (not n_k_automatic):
      filename += ".nk%s"%nk
    if len(ns)>1 and fixed_n: 
      filename += ".n%s"%n
    if len(mutildes)>1 and not fixed_n:
      filename += ".mutilde%s"%mutilde      
    if len(Us)>1: filename += ".U%s"%U
    if len(Ts)>1: filename += ".T%.4f"%T
    filename += ".h5"
    dt.archive_name = filename

    if mpi.is_master_node():
      if fixed_n:
        print "Working: U: %s T %s n: %s n_k: %s n_iw: %s"%(U,n,T,nk,n_iw)
      else:
        print "Working: U: %s T %s mutilde: %s n_k: %s n_iw: %s"%(U,mutilde,T,nk,n_iw)

    if mpi.is_master_node():
      print "about to fill dispersion. ph-symmetry: ",ph_symmetry 
    for key in dt.fermionic_struct.keys():
      for kxi in range(dt.n_k):
        for kyi in range(dt.n_k):
          dt.epsilonk[key][kxi,kyi] = dispersion(dt.ks[kxi], dt.ks[kyi])

    if not use_cumulant: 
      prepare = prepare_nested
    else: 
      prepare = prepare_cumul_nested
    if flexible_Gweiss:
      prepare( dt, nested_scheme, solver_class, flexible_Gweiss, sign, sign_up_to )
    else:  
      prepare( dt, nested_scheme, solver_class )

    solver_class.initialize_solvers( dt, solver_data_package )
 
    max_times = {}

    if no_timing:
      for C in dt.impurity_struct.keys():
        max_times[C] = -1
      if mpi.is_master_node(): print "no_timing! solvers will run until they perform all the mc steps",max_times        

    if time_rules_automatic and (not no_timing): 
      for C in dt.impurity_struct.keys():
        Nc = len(dt.impurity_struct[C])
        pref = ((dt.beta/8.0)*U*Nc)**exponent #**1.2
        print C
        print "Nc: ",Nc,
        print "U: ", U,  
        print "beta: ",dt.beta,
        print "pref: ",pref 
        max_times[C] = int(overall_prefactor*pref*5*60)
      if mpi.is_master_node(): print "max times automatic: ",max_times        

    identical_pairs_Sigma = nested_scheme.get_identical_pairs()
    identical_pairs_G = nested_scheme.get_identical_pairs_for_G()
    identical_pairs_G_ai = nested_scheme.get_identical_pairs_for_G(across_imps=True)
 
    actions =[  generic_action(  name = "lattice",
                    main = lambda data: nested_mains.lattice(data, n=n, ph_symmetry=ph_symmetry, accepted_mu_range=[-2.0,2.0]),
                    mixers = [], cautionaries = [], allowed_errors = [],    
                    printout = lambda data, it: ( [data.dump_general( quantities = ['Gkw','Gijw'], suffix='-current' ), data.dump_scalar(suffix='-current')
                                                  ] if ((it+1) % print_current==0) else None 
                                                )
                              ),
                generic_action(  name = "pre_impurity",
                    main = lambda data: nested_mains.pre_impurity(data),                       
                    mixers = [], cautionaries = [lambda data,it: ph_symmetric_Gweiss_causal_cautionary( data, ntau=5000 ) ], allowed_errors = [0],        
                    printout = lambda data, it: (data.dump_general( quantities = ['Gweiss_iw_unfit','Gweiss_iw','Delta_iw','Delta_iw_fit','Delta_tau','Delta_tau_fit'], suffix='-%s'%it ) )
                              ),
                generic_action(  name = "impurity",
                    main = (lambda data: nested_mains.impurity(data, U, symmetrize_quantities = True, alpha=alpha, delta=delta, automatic_alpha_and_delta = automatic_alpha_and_delta, 
                                                               n_cycles=n_cycles, max_times = max_times, solver_data_package = solver_data_package ))
                           if (not use_cthyb) else
                           (lambda data: nested_mains.impurity_cthyb(data, U, symmetrize_quantities = True, n_cycles=n_cycles, max_times = max_times, solver_data_package = solver_data_package )),
                    mixers = [], cautionaries = [lambda data,it: local_nan_cautionary(data, data.impurity_struct, Qs = ['Sigma_imp_iw'], raise_exception = True),                                                 
                                                 lambda data,it: ( symmetric_G_and_self_energy_on_impurity(data.G_imp_iw, data.Sigma_imp_iw, data.solvers, 
                                                                                                           identical_pairs_Sigma, identical_pairs_G,
                                                                                                           across_imps=True, identical_pairs_G_ai=identical_pairs_G_ai  )
                                                                   if it>=0 else  
                                                                   symmetrize_cluster_impurity(data.Sigma_imp_iw, identical_pairs_Sigma) )
                                                ], allowed_errors = [1],    
                    printout = lambda data, it: ( [ data.dump_general( quantities = ['Sigma_imp_iw','G_imp_iw'], suffix='-current' ),
                                                    data.dump_solvers(suffix='-current')
                                                  ] if ((it+1) % print_current==0) else None)  ),
                generic_action(  name = "selfenergy",
                    main = lambda data: nested_mains.selfenergy(data), 
                    mixers = [], cautionaries = [lambda data,it: nonloc_sign_cautionary(data.Sigmakw['up'], desired_sign = -1, clip_off = False, real_or_imag = 'imag')], allowed_errors = [0],    
                    printout = lambda data, it: (data.dump_general( quantities = ['Sigmakw','Sigmaijw'], suffix='-current' ) if ((it+1) % print_current==0) else None) ) ]

    if use_cumulant:
      del actions[3] 
      actions.append( generic_action(  name = "cumulant",
                          main = lambda data: cumul_nested_mains.cumulant(data), 
                          mixers = [], cautionaries = [lambda data,it: nonloc_sign_cautionary(data.gkw['up'], desired_sign = -1, clip_off = False, real_or_imag = 'imag')], allowed_errors = [0],    
                          printout = lambda data, it: (data.dump_general( quantities = ['gijw','gkw'], suffix='-current' ) if ((it+1) % print_current==0) else None)  ) 
                    )

    monitors = [ monitor( monitored_quantity = lambda: dt.ns['up'], 
                          h5key = 'n_vs_it', 
                          archive_name = dt.archive_name),
                 monitor( monitored_quantity = lambda: dt.mus['up'], 
                          h5key = 'mu_vs_it', 
                          archive_name = dt.archive_name),
                 monitor( monitored_quantity = lambda: dt.err, 
                          h5key = 'err_vs_it', 
                          archive_name = dt.archive_name) ]#,
#                 monitor( monitored_quantity = lambda: actions[3].errs[0], 
#                          h5key = 'sign_err_vs_it', 
#                          archive_name = dt.archive_name) ]

    if use_cumulant:  
      monitors+= [ monitor( monitored_quantity = lambda: dt.gijw['up'][dt.nw/2,0,0].imag, 
                            h5key = 'Img_00_w0_vs_it', 
                            archive_name = dt.archive_name),
                   monitor( monitored_quantity = lambda: dt.gijw['up'][dt.nw/2,0,0].real, 
                            h5key = 'Reg_00_iw0_vs_it', 
                            archive_name = dt.archive_name),
                   monitor( monitored_quantity = lambda: dt.gkw['up'][dt.nw/2, dt.n_k/2, dt.n_k/2].imag, 
                            h5key = 'Img_pipi_iw0_vs_it', 
                            archive_name = dt.archive_name),
                   monitor( monitored_quantity = lambda: dt.gkw['up'][dt.nw/2, dt.n_k/2, dt.n_k/2].real, 
                            h5key = 'Reg_pipi_iw0_vs_it', 
                            archive_name = dt.archive_name)]
    else:  
      monitors+= [ monitor( monitored_quantity = lambda: dt.Sigma_loc_iw['up'].data[dt.nw/2,0,0].imag, 
                            h5key = 'ImSigma_loc_iw0_vs_it', 
                            archive_name = dt.archive_name),
                   monitor( monitored_quantity = lambda: dt.Sigma_loc_iw['up'].data[dt.nw/2,0,0].real, 
                            h5key = 'ReSigma_loc_iw0_vs_it', 
                            archive_name = dt.archive_name),
                   monitor( monitored_quantity = lambda: dt.Sigmakw['up'][dt.nw/2, dt.n_k/2, dt.n_k/2].imag, 
                            h5key = 'ImSigmakw_pipi_vs_it', 
                            archive_name = dt.archive_name),
                   monitor( monitored_quantity = lambda: dt.Sigmakw['up'][dt.nw/2, dt.n_k/2, dt.n_k/2].real, 
                            h5key = 'ReSigmakw_pipi_vs_it', 
                            archive_name = dt.archive_name)]

    
    convergers = [ converger( monitored_quantity = lambda: dt.G_loc_iw,
                            accuracy=accuracy, 
                            struct=fermionic_struct, 
                            archive_name= dt.archive_name,
                            h5key = 'diffs_G_loc' ),
                   converger( monitored_quantity = lambda: dt.Sigma_loc_iw,
                            accuracy=accuracy, 
                            struct=fermionic_struct, 
                            archive_name= dt.archive_name,
                            h5key = 'diffs_Sigma_loc') ]
    max_dist = 3
    for i in range(max_dist+1):
      for j in range(0,i+1):
        convergers.append( converger( monitored_quantity = lambda i=i, j=j: dt.Gijw['up'][:,i,j],
                                      accuracy=accuracy,
                                      func = converger.check_numpy_array,  
                                      archive_name= dt.archive_name,
                                      h5key = 'diffs_G_%s%s'%(i,j) ) )
    convergers.append( converger( monitored_quantity = lambda: dt.G_imp_iw,
                                  accuracy=accuracy, 
                                  struct=impurity_struct, 
                                  archive_name= dt.archive_name,
                                  h5key = 'diffs_G_imp' ) )

    dmft = generic_loop(
                name = "nested-cluster DMFT", 
                actions = actions,
                convergers = convergers,  
                monitors = monitors )

    if (counter==0): #do the initial guess only once!         
      if initial_guess_archive_name!='':
        if mpi.is_master_node(): print "constructing dt from initial guess in a file: ",initial_guess_archive_name, "suffix: ",suffix
        old_epsilonk = dt.epsilonk
        dt.construct_from_file(initial_guess_archive_name, suffix) 
        if dt.beta != beta:
          dt.change_beta(beta, n_iw)
        if dt.n_k != nk:
          dt.change_ks(IBZ.k_grid(nk))
        if mpi.is_master_node(): print "putting back the old Jq and epsilonk"
        dt.epsilonk = old_epsilonk
      else:
        if not fixed_n:  
          dt.mus['up'] = mutilde
        else:
          dt.mus['up'] = U/2.0
        if 'down' in dt.fermionic_struct.keys(): dt.mus['down'] = dt.mus['up']   #this is not necessary at the moment, but may become
        for C in dt.impurity_struct.keys():
          for l in dt.impurity_struct[C]: #just the local components (but on each site!)         
            dt.Sigma_imp_iw[C].data[:,l,l] = U/2.0-int(insulating_initial)*1j/numpy.array(dt.ws)
        if not use_cumulant:
          for key in fermionic_struct.keys(): 
            dt.Sigmakw[key][:,:,:] = U/2.0    
            numpy.transpose(dt.Sigmakw[key])[:] -= int(insulating_initial)*1j/numpy.array(dt.ws)
        else:
          for key in fermionic_struct.keys(): numpy.transpose(dt.gkw[key])[:] = numpy.array(dt.iws[:])**(-1.0)    
      if not use_cumulant:
        dt.dump_general( quantities = ['Sigmakw','Sigma_imp_iw'], suffix='-initial' )  
      else:
        dt.dump_general( quantities = ['gkw'], suffix='-initial' )  


    if (counter==0) and do_dmft_first:
      assert False, "this part of code needs to be adjusted" 
      #do one short run of dmft before starting nested
      if mpi.is_master_node(): print "================= 20 iterations of DMFT!!!! ================="
      #save the old stuff
      old_impurity_struct = dt.impurity_struct 
      old_name = dmft.name
      #dmft_scheme
      dmft_scheme = nested_scheme([cluster(0,0,1,1)])
      dmft_scheme.set_nk(dt.n_k)
      dt.impurity_struct = dmft_scheme.get_impurity_struct()
      prepare_nested( dt, dmft_scheme, solvers.ctint )     
      dmft.name = "dmft"
      #run dmft
      dmft.run( dt, 
                max_its=20, 
                min_its=15,
                max_it_err_is_allowed = 7,
                print_final=True, 
                print_current = 10000 )
      #move the result
      if mpi.is_master_node():
        cmd = 'mv %s %s'%(filename, filename.replace("result", "dmft")) 
        print cmd
        os.system(cmd)
      #put everything back the way it was
      dmft.name = old_name
      dt.impurity_struct = old_impurity_struct
      prepare( dt, nested_scheme, solver_class )
  

    #run nested!-------------

    if mix_Sigma:
      actions[3].mixers.extend([ mixer( mixed_quantity = lambda: (dt.Sigmakw if (not use_cumulant) else dt.gkw),
                                      rules=rules,
                                      func=mixer.mix_lattice_gf,
                                      initialize = True ) ])
      actions[2].mixers.extend([ mixer( mixed_quantity = lambda: dt.Sigma_imp_iw,
                                      rules=rules,
                                      func=mixer.mix_block_gf,
                                      initialize = True ) ])

    dt.dump_parameters()
    dt.dump_non_interacting() 

    err = dmft.run( dt, 
              max_its=max_its, 
              min_its=min_its,
              max_it_err_is_allowed = 7,
              print_final=True, 
              print_current = 1 )
    if mpi.is_master_node():
      if use_cumulant:
        print "calculating Sigma"  
        dt.get_Sigmakw() 
        dt.get_Sigma_loc()
        dt.dump_general(['Sigmakw','Sigma_loc_iw'],suffix='-final')
      cmd = 'mv %s %s'%(filename, filename.replace("result", "nested")) 
      print cmd
      os.system(cmd)

    if (err==2): 
      print "Cautionary error!!! exiting..."
      solver_data_package['construct|run|exit'] = 2
      if MASTER_SLAVE_ARCHITECTURE and (mpi.size>1): solver_data_package = mpi.bcast(solver_data_package)
      break

    if not MASTER_SLAVE_ARCHITECTURE: mpi.barrier()
    counter += 1
  if not (solver_data_package is None): solver_data_package['construct|run|exit'] = 2
  if MASTER_SLAVE_ARCHITECTURE and (mpi.size>1): solver_data_package = mpi.bcast(solver_data_package)
  return dt, monitors, convergers
