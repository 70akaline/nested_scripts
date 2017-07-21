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


def reverse_calculation( clusters, nested_struct_archive_name = None, sign=-1, sign_up_to=2, fit_bath = False, Cs=[],
                        U = 1.0,
                        T = 0.125, 
                        n = 0.5, ph_symmetry = False,
                        n_k = 24,
                        w_cutoff = 20.0,
                        min_its = 5, max_its=25, 
                        mix_G_imp = False, rules = [[0, 0.5], [6, 0.2], [12, 0.65]],                        
                        alpha = 0.5, delta = 0.1,  automatic_alpha_and_delta = False,
                        n_cycles=10000000, 
                        max_time_rules= [ [1, 5*60], [2, 20*60], [4, 80*60], [8, 200*60], [16,400*60] ], time_rules_automatic=False, exponent = 0.7, overall_prefactor=1.0, no_timing = False,
                        accuracy = 1e-4, 
                        solver_data_package = None,
                        print_current = 1,
                        initial_guess_archive_name = '', suffix=''):

  if mpi.is_master_node():
    print "WELCOME TO reverse calculation!"

  beta = 1.0/T
  nk = n_k

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

  max_times = {}

  if not time_rules_automatic:
    for C in impurity_struct:
      for r in max_time_rules:
        if r[0]<=len(impurity_struct[C]):
          max_times[C] = r[1]
    if mpi.is_master_node(): print "max_times from rules: ",max_times

  if no_timing:
    for C in impurity_struct.keys():
      max_times[C] = -1
    if mpi.is_master_node(): print "no_timing! solvers will run until they perform all the mc steps",max_times        

  if time_rules_automatic and (not no_timing): 
    for C in impurity_struct.keys():
      Nc = len(impurity_struct[C])
      pref = ((beta/8.0)*U*Nc)**exponent #**1.2
      print C
      print "Nc: ",Nc,
      print "U: ", U,  
      print "beta: ",beta,
      print "pref: ",pref 
      max_times[C] = int(overall_prefactor*pref*5*60)
    if mpi.is_master_node(): print "max times automatic: ",max_times    


  
  n_iw = int(((w_cutoff*beta)/math.pi-1.0)/2.0)
  if mpi.is_master_node():
    print "PM HUBBARD GW: n_iw: ",n_iw

  dt = nested_data(  n_iw = n_iw, 
                     n_k = n_k, 
                     beta = beta, 
                     impurity_struct = deepcopy(impurity_struct),
                     fermionic_struct = deepcopy(fermionic_struct),
                     archive_name="so_far_nothing_you_shouldnt_see_this_file"  )

  filename = "result.h5"
  dt.archive_name = filename


  nested_scheme.set_nk(nk) #don't forget this part

  prepare = prepare_nested
  flexible_Gweiss = False
  prepare( dt, nested_scheme, solver_class, flexible_Gweiss, sign, sign_up_to )


  if Cs!=[]: 
    old_impurity_struct = deepcopy(dt.impurity_struct)
    print dt.impurity_struct
    for C in dt.impurity_struct.keys():
      if not (C in Cs):
        del dt.impurity_struct[C]
    print dt.impurity_struct      
  solver_class.initialize_solvers( dt, solver_data_package )
  if Cs!=[]: 
    dt.impurity_struct = old_impurity_struct
    print dt.impurity_struct

  if mpi.is_master_node():
      print "Working: U: %s T %s n_k: %s n_iw: %s"%(U,T,nk,n_iw)   

  identical_pairs_Sigma = nested_scheme.get_identical_pairs()
  identical_pairs_G = nested_scheme.get_identical_pairs_for_G()
  identical_pairs_G_ai = nested_scheme.get_identical_pairs_for_G(across_imps=True)

  actions =[  generic_action(  name = "pre_impurity",
                  main = lambda data: nested_mains.pre_impurity(data),
                                      mixers = [], cautionaries = [lambda data,it: (ph_symmetric_Gweiss_causal_cautionary( data, ntau=5000 ) if fit_bath else None) ], allowed_errors = [0] if fit_bath else [],        
                    printout = lambda data, it: ( (data.dump_general( quantities = ['Gweiss_iw_unfit','Gweiss_iw','Delta_iw','Delta_iw_fit','Delta_tau','Delta_tau_fit'], suffix='-%s'%it ) ) if fit_bath else None )
                              ),
              generic_action(  name = "impurity",
                  main = (lambda data: nested_mains.impurity(data, U, symmetrize_quantities = True, alpha=alpha, delta=delta, automatic_alpha_and_delta = automatic_alpha_and_delta, 
                                                             n_cycles=n_cycles, max_times = max_times, solver_data_package = solver_data_package, Cs = Cs )),
                  mixers = [], cautionaries = [lambda data,it: local_nan_cautionary(data, data.impurity_struct, Qs = ['Sigma_imp_iw'], raise_exception = True),                                                 
                                               lambda data,it: ( symmetric_G_and_self_energy_on_impurity(data.G_imp_iw, data.Sigma_imp_iw, data.solvers, 
                                                                                                         identical_pairs_Sigma, identical_pairs_G,
                                                                                                         across_imps=False, identical_pairs_G_ai=identical_pairs_G_ai  )
                                                                 if it>=0 else  
                                                                 symmetrize_cluster_impurity(data.Sigma_imp_iw, identical_pairs_Sigma) )
                                              ], allowed_errors = [1],    
                  printout = lambda data, it: ( [ data.dump_general( quantities = ['Sigma_imp_iw','G_imp_iw'], suffix='-%s'%it ),
                                                  data.dump_solvers(suffix='-%s'%it)
                                                ] if ((it+1) % print_current==0) else None)  )
           ]


   
  monitors =   [ monitor( monitored_quantity = lambda C=C: dt.G_imp_iw[C].data[dt.nw/2,0,0].imag, 
                          h5key = 'ImG^%s_00(iw_0)_vs_it'%C, 
                          archive_name = dt.archive_name) for C in dt.impurity_struct
               ] +\
               [ monitor( monitored_quantity = lambda C=C: dt.solvers[C].G0_shift_iw['up'].data[dt.nw/2,0,0].imag, 
                          h5key = 'ImG0^%s_00(iw_0)_vs_it'%C, 
                          archive_name = dt.archive_name) for C in dt.impurity_struct
               ]

  
  convergers = [ converger( monitored_quantity = lambda: dt.G_imp_iw,
                          accuracy=accuracy, 
                          struct=impurity_struct, 
                          archive_name= dt.archive_name,
                          h5key = 'diffs_G_imp' ),
                 converger( monitored_quantity = lambda: dt.Gweiss_iw,
                          accuracy=accuracy, 
                          struct=impurity_struct, 
                          archive_name= dt.archive_name,
                          h5key = 'diffs_Gweiss') ]
 
  dmft = generic_loop(
              name = "nested-cluster DMFT", 
              actions = actions,
              convergers = convergers,  
              monitors = monitors )

  
  if initial_guess_archive_name!='':
    if mpi.is_master_node(): print "constructing dt from initial guess in a file: ",initial_guess_archive_name, "suffix: ",suffix
    dt.construct_from_file(initial_guess_archive_name, suffix) 
    assert dt.beta == beta, "wrong temperature"
    assert dt.n_k == nk, "wrong nk"
    assert dt.impurity_struct == impurity_struct, "wrong impurity_struct"

    dt.dump_general( quantities = ['Gweiss_iw'], suffix='-initial' )
    #dt.dump_general( quantities = ['Gkw','Gijw','G_loc_iw'], suffix='-requiered' )
    dt.dump_general( quantities = ['Gijw'], suffix='-requiered' )        

  else:
    assert False, "initial guess for Gweiss necessary, and a result for G and epsilonk "

  #run reverse!-------------

  if mix_G_imp:
    actions[1].mixers.extend([ mixer( mixed_quantity = lambda: dt.G_imp_iw,
                                    rules=rules,
                                    func=mixer.mix_block_gf,
                                    initialize = True ) ])

  dt.dump_parameters()
  dt.dump_non_interacting() 

  err = dmft.run( dt, 
            max_its=max_its, 
            min_its=min_its,
            max_it_err_is_allowed = 7,
            print_final=False, 
            print_current = 10000,
            start_from_action_index = 1 )
  if mpi.is_master_node():
    dt.get_Sigmaijw()
    dt.get_Sigmakw()
    dt.get_Sigma_loc()
    nested_mains.lattice(dt, n, ph_symmetry, accepted_mu_range=[-2.0,2.0])
    dt.dump_general( quantities = ['ns','mus','G_loc_iw','Gkw','Gijw','Gweiss_iw','G_imp_iw','Sigmakw','Sigmaijw','Sigma_loc_iw'], suffix='-final' )    
    dt.dump_solvers("-final")
    cmd = 'mv %s %s'%(filename, filename.replace("result", "reverse")) 
    print cmd
    os.system(cmd)

  if (err==2): 
    print "Cautionary error!!!"

  if not MASTER_SLAVE_ARCHITECTURE: mpi.barrier()

  if not (solver_data_package is None): solver_data_package['construct|run|exit'] = 2
  if MASTER_SLAVE_ARCHITECTURE and (mpi.size>1): solver_data_package = mpi.bcast(solver_data_package)
  return dt, monitors, convergers
