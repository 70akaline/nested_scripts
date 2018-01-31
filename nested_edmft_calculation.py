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


def nested_edmft_calculation( clusters, nested_struct_archive_name = None, flexible_Gweiss=False, sign=-1, sign_up_to=2, use_Gweiss_causal_cautionary = False,
                              freeze_Uweiss = False, no_lattice = False,
                              Us = [1.0], decoupling = 'ising', decoupling_alpha = 0.5,
                              Ts = [0.125], 
                              ns = [0.5], fixed_n = True,
                              mutildes = [0.0],
                              dispersion = lambda kx, ky: epsilonk_square(kx,ky, 0.25), ph_symmetry = True,                              
                              n_ks = [64], n_k_automatic = False, n_k_rules = [[0.06, 32],[0.03, 48],[0.005, 64],[0.00, 96]],
                              w_cutoff = 50.0,
                              min_its = 5, max_its=25, 
                              mix_GWlatt = False, rules = [[0, 0.5], [6, 0.2], [12, 0.65]],              
                              mix_Uweiss = False, Uweiss_mix_rules = [[0, 0.5], [6, 0.2], [12, 0.65]],                       
                              use_cthyb = False,
                              alpha = 0.5, delta = 0.1,  automatic_alpha_and_delta = False,
                              n_cycles=10000000, 
                              max_time_rules= [ [1, 5*60], [2, 20*60], [4, 80*60], [8, 200*60], [16,400*60] ], time_rules_automatic=False, exponent = 0.7, overall_prefactor=4.0, no_timing = False,
                              accuracy = 1e-4, 
                              solver_data_package = None,
                              print_current = 1,
                              insulating_initial = False,
                              Wilson_bath_initial = False,
                              bath_initial = False,
                              selfenergy_initial = False,  
                              initial_guess_archive_name = '', suffix=''):

  if mpi.is_master_node():
    print "WELCOME TO nested edmft calculation!"
    if n_k_automatic: print "n_k automatic!!!"
  if len(n_ks)==0 and n_k_automatic: n_ks=[0]

  if use_cthyb:
    assert False, "cthyb usage not implemented"
    solver_class = solvers.cthyb
  else:
    solver_class = solvers.ctint

  fermionic_struct = {'up': [0]}
  bosonic_struct = {'0': [0], '1': [0]}   
  if decoupling=='ising':
    if decoupling_alpha==1.0:
      del bosonic_struct['1']
    if decoupling_alpha==0.0:
      del bosonic_struct['0']
  elif decoupling=='heisenberg':
    if decoupling_alpha==2.0/3.0:
      del bosonic_struct['1']
    if decoupling_alpha==1.0/3.0:
      del bosonic_struct['0']


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

  dt = nested_edmft_data(  n_iw = n_iw, 
                     n_k = n_k, 
                     beta = beta, 
                     impurity_struct = impurity_struct,
                     fermionic_struct = fermionic_struct,
                     bosonic_struct = bosonic_struct,
                     archive_name="so_far_nothing_you_shouldnt_see_this_file"  )

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
      assert False, "changing n_k not implemented" 
      dt.change_ks(IBZ.k_grid(nk))

    if beta!=old_beta:
      assert False, "changing beta not implemented"
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
    if decoupling=='ising':
      for key in dt.bosonic_struct.keys():
        if key=='0': dt.Jq[key][:,:]=decoupling_alpha*U
        if key=='1': dt.Jq[key][:,:]=(decoupling_alpha-1)*U
    elif decoupling=='heisenberg': 
        if key=='0': dt.Jq[key][:,:]=(3.0*decoupling_alpha-1.0)*U
        if key=='1': dt.Jq[key][:,:]=(decoupling_alpha-2.0/3.0)*U
    else: assert False, "unknown decoupling scheme"

    prepare_nested_edmft( dt, nested_scheme, solver_class )

    solver_class.initialize_solvers( dt, solver_data_package, bosonic_measures=True )
 

    if no_timing:
      max_times = {}
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

    identical_pairs_Sigma = nested_scheme.get_identical_pairs()
    identical_pairs_G = nested_scheme.get_identical_pairs_for_G()
    identical_pairs_G_ai = nested_scheme.get_identical_pairs_for_G(across_imps=True)
 
    def do_print(*args):
      for x in args: print x,
      print "" 

    used_Cs = []

    actions =[  generic_action(  name = "lattice",
                    main = ( (lambda data: nested_edmft_mains.lattice(data, n=n, ph_symmetry=ph_symmetry, accepted_mu_range=[-2.0,2.0]))
                             if (not no_lattice) else
                             (lambda data: [ data.copy_imp_to_latt(used_Cs[0]),
                                             do_print("just copying imp",used_Cs[0],"->latt, no_lattice! Gijw000:",data.Gijw['up'][0,0,0],
                                                      [ "Wijnu_"+A+ "000:"+str(data.Wijnu['0'][0,0,0]) for A in data.Wijnu.keys()] ) ]) #TODO generalize for any size cluster
                           ),
                           
                    mixers = [], cautionaries = [], allowed_errors = [],    
                    printout = lambda data, it: ( [data.dump_general( quantities = ['Gkw','Gijw','Wqnu','Wijnu'], suffix='-current' ), data.dump_scalar(suffix='-current')
                                                  ] if ((it+1) % print_current==0) else None 
                                                )
                              ),
                generic_action(  name = "pre_impurity",
                    main = lambda data: nested_edmft_mains.pre_impurity(data, freeze_Uweiss = freeze_Uweiss, Cs= used_Cs),                       
                    mixers = [], cautionaries = [], allowed_errors = [],        
                    printout = lambda data, it: ( data.dump_general( quantities = ['Gweiss_iw','Uweiss_iw','Uweiss_dyn_iw'], suffix='-current' ) )
                              ),
                generic_action(  name = "impurity",
                    main = (lambda data: nested_mains.impurity(data, U, symmetrize_quantities = True, alpha=alpha, delta=delta, automatic_alpha_and_delta = automatic_alpha_and_delta, 
                                                               n_cycles=n_cycles, max_times = max_times, solver_data_package = solver_data_package, bosonic_measures=not freeze_Uweiss, Cs= used_Cs ))
                           if (not use_cthyb) else
                           (lambda data: nested_mains.impurity_cthyb(data, U, symmetrize_quantities = True, n_cycles=n_cycles, max_times = max_times, solver_data_package = solver_data_package )),
                    mixers = [], cautionaries = [lambda data,it: local_nan_cautionary(data, data.impurity_struct, Qs = ['Sigma_imp_iw'], raise_exception = True),                                                 
                                                 lambda data,it: ( symmetric_G_and_self_energy_on_impurity(data.G_imp_iw, data.Sigma_imp_iw, data.solvers, 
                                                                                                           identical_pairs_Sigma, identical_pairs_G,
                                                                                                           across_imps=True, identical_pairs_G_ai=identical_pairs_G_ai  )
                                                                   if it>=10000 else  
                                                                   symmetrize_cluster_impurity(data.Sigma_imp_iw, identical_pairs_Sigma) )
                                                ], allowed_errors = [1],    
                    printout = lambda data, it: ( [ data.dump_general( quantities = ['Sigma_imp_iw','G_imp_iw'], suffix='-current' ),
                                                    data.dump_solvers(suffix='-current')
                                                  ] if ((it+1) % print_current==0) else None)  ),
                generic_action(  name = "post_impurity",
                    main = lambda data: nested_edmft_mains.post_impurity(data, identical_pairs = identical_pairs_Sigma, Cs= used_Cs),#, homogeneous_pairs = identical_pairs_G), 
                    mixers = [], cautionaries = [], allowed_errors = [],    
                    printout = lambda data, it: (data.dump_general( quantities = ['chi_imp_iw','P_imp_iw','W_imp_iw'], suffix='-current' ) if ((it+1) % print_current==0) else None) ),
                generic_action(  name = "selfenergy",
                    main = lambda data: nested_edmft_mains.selfenergy(data), 
                    mixers = [], cautionaries = [lambda data,it: nonloc_sign_cautionary(data.Sigmakw['up'], desired_sign = -1, clip_off = False, real_or_imag = 'imag')], allowed_errors = [0],    
                    printout = lambda data, it: (data.dump_general( quantities = ['Sigmakw','Sigmaijw','Sigma_loc_iw','Pqnu','Pijnu','P_loc_iw'], suffix='-current' ) if ((it+1) % print_current==0) else None) ) ]



    monitors = [ monitor( monitored_quantity = lambda: dt.ns['up'], 
                          h5key = 'n_vs_it', 
                          archive_name = dt.archive_name),
                 monitor( monitored_quantity = lambda: dt.mus['up'], 
                          h5key = 'mu_vs_it', 
                          archive_name = dt.archive_name),
                 monitor( monitored_quantity = lambda: dt.err, 
                          h5key = 'err_vs_it', 
                          archive_name = dt.archive_name) ]

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
                   converger( monitored_quantity = lambda: dt.W_loc_iw,
                            accuracy=accuracy, 
                            struct=bosonic_struct, 
                            archive_name= dt.archive_name,
                            h5key = 'diffs_W_loc' ),
                   converger( monitored_quantity = lambda: dt.P_loc_iw,
                            accuracy=accuracy, 
                            struct=bosonic_struct, 
                            archive_name= dt.archive_name,
                            h5key = 'diffs_P_loc' ),
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
    convergers.append( converger( monitored_quantity = lambda: dt.Gweiss_iw,
                                  accuracy=accuracy, 
                                  struct=impurity_struct, 
                                  archive_name= dt.archive_name,
                                  h5key = 'diffs_Gweiss' ) )
    combo_imp_struct = {}   
    for CA,Uw in dt.Uweiss_iw:
      combo_imp_struct[CA] = impurity_struct[C]   
    convergers.append( converger( monitored_quantity = lambda: dt.Uweiss_iw,
                                  accuracy=accuracy, 
                                  struct=combo_imp_struct, 
                                  archive_name= dt.archive_name,
                                  h5key = 'diffs_Uweiss' ) )


    dmft = generic_loop(
                name = "nested-cluster EDMFT", 
                actions = actions,
                convergers = convergers,  
                monitors = monitors )

    start_from_action_index = 0

    if freeze_Uweiss:
      if mpi.is_master_node(): print "Uweiss frozen! Equivalent to nested DMFT calculation"

    if (counter==0): #do the initial guess only once!         
      if initial_guess_archive_name!='':
        if selfenergy_initial:
          start_from_action_index = 0
          if mpi.is_master_node(): print "Taking Sigma from initial guess in:",initial_guess_archive_name, "suffix: ",suffix
          HDFA = HDFArchive(initial_guess_archive_name,'r')                
          dt.Sigmakw = deepcopy(HDFA['Sigmakw%s'%suffix])
          dt.Sigma_imp_iw << HDFA['Sigma_imp_iw%s'%suffix]
          dt.mus = HDFA['mus%s'%suffix]
          del HDFA
          dt.dump_general( quantities = ['Sigmakw','Sigma_imp_iw'], suffix='-initial' )  
        elif bath_initial:
          start_from_action_index = 2
          if mpi.is_master_node(): print "Taking Gweiss from initial guess in:",initial_guess_archive_name, "suffix: ",suffix
          HDFA = HDFArchive(initial_guess_archive_name,'r')                
          input_blocks = [C for C,gw in HDFA['Gweiss_iw%s'%suffix]]  
          if set(input_blocks)!=set(impurity_struct.keys()): 
            used_Cs[:]=input_blocks[:]
            print "WARNING: input block structure does not correspond to the nested scheme block structure. Running only the impurities in the input block set."
          for C,gw in HDFA['Gweiss_iw%s'%suffix]: 
            dt.Gweiss_iw[C] << gw 
          dt.mus = HDFA['mus%s'%suffix]
          for C in impurity_struct.keys(): #in any case fill Uweiss for P_imp calculation not to crash
            for A in bosonic_struct:
              print "dt.Jq[",A,"][0,0]:",dt.Jq[A][0,0]
              dt.Uweiss_iw[C+'|'+A] << dt.Jq[A][0,0]
              dt.Uweiss_dyn_iw[C+'|'+A] << 0.0
          del HDFA
          dt.dump_general( quantities = ['Gweiss_iw','Uweiss_iw','Uweiss_dyn_iw'], suffix='-initial' )       
      else:
        if not fixed_n:  
          dt.mus['up'] = mutilde
        else:
          dt.mus['up'] = U/2.0
        if 'down' in dt.fermionic_struct.keys(): dt.mus['down'] = dt.mus['up']   #this is not necessary at the moment, but may become         
        if Wilson_bath_initial:
          start_from_action_index = 2
          if impurity_struct.keys()!=["1x1"]: assert False, "Wilson initializer inapplicable!"
          for A in bosonic_struct:
            print "dt.Jq[",A,"][0,0]:",dt.Jq[A][0,0]
            dt.Uweiss_iw['1x1|'+A] << dt.Jq[A][0,0]
            dt.Uweiss_dyn_iw['1x1|'+A] << 0.0
          dt.Gweiss_iw['1x1'] << inverse(iOmega_n+U/2.0-Wilson(0.25))
          dt.dump_general( quantities = ['Gweiss_iw','Uweiss_iw','Uweiss_dyn_iw'], suffix='-initial' )  
        else: 
          for C in dt.impurity_struct.keys():
            for l in dt.impurity_struct[C]: #just the local components (but on each site!)         
              dt.Sigma_imp_iw[C].data[:,l,l] = U/2.0-int(insulating_initial)*1j/numpy.array(dt.ws)
            for A in bosonic_struct:
              CA = C+"|"+A
              dt.Uweiss_iw[CA] << dt.Jq[A][0,0]
              dt.Uweiss_dyn_iw[CA] << 0.0
              dt.P_imp_iw[CA] << 0.0
          for A in bosonic_struct:
            dt.Pqnu[A][:,:,:] = 0.0
            dt.P_loc_iw[A]<< 0.0
            dt.Pijnu[A][:,:,:]= 0.0        
          for key in fermionic_struct.keys(): 
            dt.Sigmakw[key][:,:,:] = U/2.0    
            numpy.transpose(dt.Sigmakw[key])[:] -= int(insulating_initial)*1j/numpy.array(dt.ws)
          dt.dump_general( quantities = ['Sigmakw','Sigma_imp_iw'], suffix='-initial' )  

 

    #run nested!-------------

    if mix_GWlatt:
      actions[0].mixers.extend([ mixer( mixed_quantity = lambda: dt.Gijw,
                                      rules=rules,
                                      func=mixer.mix_lattice_gf,
                                      initialize = True ) ])
      actions[0].mixers.extend([ mixer( mixed_quantity = lambda: dt.Wijnu,
                                      rules=rules,
                                      func=mixer.mix_lattice_gf,
                                      initialize = True ) ])
    if mix_Uweiss:
      print "mixing Uweiss, rules:", Uweiss_mix_rules
      actions[1].mixers.extend([ mixer( mixed_quantity = lambda: dt.Uweiss_iw,
                                      rules=Uweiss_mix_rules,
                                      func=mixer.mix_block_gf,
                                      initialize = True ) ])
      actions[1].mixers.extend([ mixer( mixed_quantity = lambda: dt.Uweiss_dyn_iw,
                                      rules=Uweiss_mix_rules,
                                      func=mixer.mix_block_gf,
                                      initialize = True ) ])


    dt.dump_parameters()
    dt.dump_non_interacting() 

    err = dmft.run( dt, 
              max_its=max_its, 
              min_its=min_its,
              max_it_err_is_allowed = 7,
              print_final=True, 
              print_current = 1,
              start_from_action_index = start_from_action_index  )
    if mpi.is_master_node():
      cmd = 'mv %s %s'%(filename, filename.replace("result", "nested_edmft")) 
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
