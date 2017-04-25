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


def dca_plus_calculation( dca_scheme, embedded = False, real_space_sc = False, no_convolution = False, Xi_alpha = 1.0, n_RL_iterations = 15,
                          Us = [1.0],
                          Ts = [0.125], 
                          ns = [0.5], fixed_n = True,
                          mutildes = [0.0],
                          dispersion = lambda kx, ky: epsilonk_square(kx,ky, 0.25), ph_symmetry = True,
                          nk = 128, 
                          w_cutoff = 20.0,
                          min_its = 5, max_its=25, 
                          mix_Sigma = False, rules = [[0, 0.5], [6, 0.2], [12, 0.65]],                        
                          do_dmft_first = False, 
                          alpha = 0.5, delta = 0.1,  automatic_alpha_and_delta = False,
                          n_cycles=10000000, 
                          max_time_rules= [ [1, 5*60], [2, 20*60], [4, 80*60], [8, 200*60], [16,400*60] ], time_rules_automatic=False, exponent = 0.7, overall_prefactor=1.0, no_timing = False,
                          accuracy = 1e-4, 
                          solver_data_package = None,
                          print_current = 1,
                          initial_guess_archive_name = '', suffix=''):

  if mpi.is_master_node():
    print "WELCOME TO dca_plus calculation!"

  solver_class = solvers.ctint

  impurity_struct = dca_scheme.get_impurity_struct()
  fermionic_struct = dca_scheme.get_fermionic_struct()

  if mpi.is_master_node(): print "impurity_struct: ", impurity_struct
  if mpi.is_master_node(): print "fermionic_struct: ", fermionic_struct

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

  dt = dca_plus_data( n_iw = n_iw,           
                      n_k = nk,
                      beta = beta, 
                      impurity_struct = impurity_struct,
                      fermionic_struct = fermionic_struct,
                      archive_name="so_far_nothing_you_shouldnt_see_this_file"  )

  if fixed_n:
    ps = itertools.product(ns,Us,Ts)
  else:
    ps = itertools.product(mutildes,Us,Ts)

  counter = 0
  old_beta = beta
  for p in ps:    
    #name stuff to avoid confusion       
    if fixed_n:
      n = p[0]
    else:
      mutilde = p[0]
      n = None
    U = p[1]
    T = p[2] 
    beta = 1.0/T

    if beta!=old_beta:
      n_iw = int(((w_cutoff*beta)/math.pi-1.0)/2.0)
      dt.change_beta(beta, n_iw)
 
    old_beta = beta

    filename = "result"
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
        print "Working: U: %s T %s n: %s "%(U,T,n)
      else:
        print "Working: U: %s T %s mutilde: %s "%(U,T,mutilde)

    prepare_dca_plus( dt, dca_scheme, solver_class, Xi_alpha, n_RL_iterations, embedded, real_space_sc, no_convolution )
    
    solver_class.initialize_solvers( dt, solver_data_package )
 
    if mpi.is_master_node():
      print "about to fill dispersion. ph-symmetry: ",ph_symmetry 
    for kxi,kx in enumerate(dt.ks):
      for kyi,ky in enumerate(dt.ks):
        dt.epsilonk['up'][kxi,kyi] = dispersion(kx, ky)

    if no_timing:
      for C in dt.impurity_struct.keys():
        max_times[C] = -1
      if mpi.is_master_node(): print "no_timing! solvers will run until they perform all the mc steps",max_times        

    if time_rules_automatic and (not no_timing): 
      max_times = {}
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

    identical_pairs = dca_scheme.get_identical_pairs()
 
    actions =[  generic_action(  name = "lattice",
                    main = lambda data: nested_mains.lattice(data, n=n, ph_symmetry=ph_symmetry, accepted_mu_range=[-2.0,2.0]),
                    mixers = [], cautionaries = [], allowed_errors = [],    
                    printout = lambda data, it: ( [data.dump_general( quantities = ['GK_iw','GR_iw','Gkw'], suffix='-current' ), data.dump_scalar(suffix='-current')
                                                  ] if ((it+1) % print_current==0) else None 
                                                )
                              ),
                generic_action(  name = "pre_impurity",
                    main = lambda data: nested_mains.pre_impurity(data),
                    mixers = [], cautionaries = [], allowed_errors = [],    
                    printout = lambda data, it: (data.dump_general( quantities = ['GweissK_iw','GweissR_iw','Gweiss_iw'], suffix='-current' ) if ((it+1) % print_current==0) else None)  ),
                generic_action(  name = "impurity",
                    main = (lambda data: nested_mains.impurity(data, U, symmetrize_quantities = True, alpha=alpha, delta=delta, automatic_alpha_and_delta = automatic_alpha_and_delta, 
                                                               n_cycles=n_cycles, max_times = max_times, solver_data_package = solver_data_package )),
                    mixers = [], cautionaries = [lambda data,it: local_nan_cautionary(data, data.impurity_struct, Qs = ['Sigma_imp_iw'], raise_exception = True),                                                 
                                                 lambda data,it: ( symmetric_G_and_self_energy_on_impurity(data.G_imp_iw, data.Sigma_imp_iw, data.solvers, 
                                                                                                          {'x': identical_pairs}, {'x': identical_pairs} )
                                                                   if it>=5 else  
                                                                   symmetrize_cluster_impurity(data.Sigma_imp_iw, {'x': identical_pairs}) )
                                                ], allowed_errors = [1],    
                    printout = lambda data, it: ( [ data.dump_general( quantities = ['Sigma_imp_iw','G_imp_iw'], suffix='-current' ),
                                                    data.dump_solvers(suffix='-current')
                                                  ] if ((it+1) % print_current==0) else None)  ),
                generic_action(  name = "selfenergy",
                    main = lambda data: dca_plus_mains.selfenergy(data), 
                    mixers = [], cautionaries = [lambda data,it: impose_real_valued_in_imtime_numpy(data.Sigmakw['up'])], allowed_errors = [],    
                    printout = lambda data, it: (data.dump_general( quantities = ['SigmaR_iw','SigmaK_iw','Sigmaimpkw','Sigmakw','XiK_iw','XiR_iw','Xikw'], suffix='-current' ) if ((it+1) % print_current==0) else None)  )  ]

    if mix_Sigma:
      actions[3].mixers.append(mixer( mixed_quantity = lambda: dt.SigmaK_iw,
                                      rules=rules,
                                      func=mixer.mix_block_gf,
                                      initialize = True )) 


    monitors = [ monitor( monitored_quantity = lambda: dt.ns['up'], 
                          h5key = 'n_vs_it', 
                          archive_name = dt.archive_name),
                 monitor( monitored_quantity = lambda: dt.mus['up'], 
                          h5key = 'mu_vs_it', 
                          archive_name = dt.archive_name),
                 monitor( monitored_quantity = lambda: dt.SigmaR_iw['00'].data[dt.nw/2,0,0].imag, 
                          h5key = 'ImSigma00_vs_it', 
                          archive_name = dt.archive_name),
                 monitor( monitored_quantity = lambda: dt.SigmaR_iw['00'].data[dt.nw/2,0,0].real, 
                          h5key = 'ReSigma00_vs_it', 
                          archive_name = dt.archive_name),
                 monitor( monitored_quantity = lambda: dt.GR_iw['00'].data[dt.nw/2,0,0].imag, 
                          h5key = 'ImG00_vs_it', 
                          archive_name = dt.archive_name),
                 monitor( monitored_quantity = lambda: dt.GR_iw['00'].data[dt.nw/2,0,0].real, 
                          h5key = 'ReG00_vs_it', 
                          archive_name = dt.archive_name),
                 monitor( monitored_quantity = lambda: dt.err, 
                          h5key = 'err_vs_it', 
                          archive_name = dt.archive_name) ]#,
#                 monitor( monitored_quantity = lambda: actions[3].errs[0], 
#                          h5key = 'sign_err_vs_it', 
#                          archive_name = dt.archive_name) ]
    
    convergers = [ converger( monitored_quantity = lambda: dt.GR_iw,
                            accuracy=accuracy, 
                            struct=fermionic_struct, 
                            archive_name= dt.archive_name,
                            h5key = 'diffs_GR' ),
                   converger( monitored_quantity = lambda: dt.SigmaR_iw,
                            accuracy=accuracy, 
                            struct=fermionic_struct, 
                            archive_name= dt.archive_name,
                            h5key = 'diffs_SigmaR' ) ]

    dmft = generic_loop(
                name = "nested-cluster DMFT", 
                actions = actions,
                convergers = convergers,  
                monitors = monitors )

    if (counter==0): #do the initial guess only once!         
      if initial_guess_archive_name!='':
        if mpi.is_master_node(): print "constructing dt from initial guess in a file: ",initial_guess_archive_name, "suffix: ",suffix
        dt.construct_from_file(initial_guess_archive_name, suffix) #make sure it is the right dca_scheme
        if dt.beta != beta:
          dt.change_beta(beta, n_iw)
      else:
        if not fixed_n:  
          dt.set_mu(mutilde)
        else:
          dt.set_mu( U/2.0)
        for C in dt.impurity_struct.keys():
          for l in dt.impurity_struct[C]: #just the local components (but on each site!)         
            dt.Sigma_imp_iw[C][l,l] << U/2.0
        dt.Sigmakw['up'][:,:,:] = U/2.0
        for K, sig in dt.SigmaK_iw:
          sig[0,0] << U/2.0
          fit_fermionic_sigma_tail(sig)
      dt.dump_general( quantities = ['Sigmakw','SigmaK_iw','Sigma_imp_iw'], suffix='-initial' )  

    if (counter==0) and do_dmft_first:
      assert False, "not implemented" 

    #run dca!-------------
    dt.dump_parameters()
    dt.dump_non_interacting() 

    err = dmft.run( dt, 
              max_its=max_its, 
              min_its=min_its,
              max_it_err_is_allowed = 7,
              print_final=True, 
              print_current = 1 )
    if mpi.is_master_node():
      cmd = 'mv %s %s'%(filename, filename.replace("result", "dca_plus")) 
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
