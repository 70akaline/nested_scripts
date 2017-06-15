from functools import partial
import math, time, cmath
from math import cos, exp, sin, log, log10, pi, sqrt
import random
import numpy
from numpy import matrix, array, zeros, identity
from numpy.linalg import inv
from pytriqs.operators import *
from pytriqs.archive import *
from pytriqs.gf.local import *
from pytriqs.arrays import BlockMatrix, BlockMatrixComplex
import pytriqs.utility.mpi as mpi

from copy import deepcopy

from first_include import *
from tail_fitters import symmetrize_blockgf
from tail_fitters import fit_and_overwrite_tails_on_Sigma
from tail_fitters import fit_and_overwrite_tails_on_G

#from cthyb_spin import Solver

try:
  from ctint import SolverCore as Solver
except:
  if mpi.is_master_node():
    print "CTINT not installed"

try:
  from pytriqs.applications.impurity_solvers.cthyb import SolverCore as cthybSolver
  #from cthyb import SolverCore as cthybSolver
except:
  if mpi.is_master_node():
    print "CTHYB not installed"

#from selfconsistency.useful_functions import adjust_n_points
#from selfconsistency.provenance import hash_dict

import copy

################################ IMPURITY #########################################

class solvers:
  class ctint:
    @staticmethod
    def initialize_solvers(data, solver_data_package = None):
      if solver_data_package is None: solver_data_package = {}
      
      n_tau = 2000 
      for C in data.impurity_struct.keys():
        if len(data.impurity_struct[C])>16: 
          n_tau=1000
      assert n_tau>2*data.n_iw, "solvers.ctint.initialize_solvers: ERROR! n_tau too small!!" 

      solver_data_package['impurity_struct'] = data.impurity_struct
      solver_data_package['constructor_parameters']={}
      solver_data_package['constructor_parameters']['beta'] = data.beta
      solver_data_package['constructor_parameters']['n_iw'] = data.n_iw
      solver_data_package['constructor_parameters']['n_tau_g0'] = n_tau
      solver_data_package['constructor_parameters']['n_tau_f'] = n_tau
      solver_data_package['constructor_parameters']['n_tau_dynamical_interactions'] = 3
      solver_data_package['constructor_parameters']['n_iw_dynamical_interactions'] = 1
      solver_data_package['constructor_parameters']['n_tau_nnt'] = 5
      solver_data_package['constructor_parameters']['n_tau_g2t'] = 5
      solver_data_package['constructor_parameters']['n_w_f_g2w'] = 2
      solver_data_package['constructor_parameters']['n_w_b_g2w'] = 2
      solver_data_package['constructor_parameters']['n_tau_M4t'] = 5
      solver_data_package['constructor_parameters']['n_w_f_M4w'] = 2
      solver_data_package['constructor_parameters']['n_w_b_M4w'] = 2
      solver_data_package['construct|run|exit'] = 0

      if MASTER_SLAVE_ARCHITECTURE and (mpi.size>1): solver_data_package = mpi.bcast(solver_data_package)

      for C in data.impurity_struct.keys():
        solver_struct = {'up': data.impurity_struct[C], 'dn': data.impurity_struct[C]}  
        solver_data_package['constructor_parameters']['gf_struct'] = solver_struct
        data.solvers[C] = Solver( **solver_data_package['constructor_parameters'] )

    @staticmethod
    def run(data, C, U, symmetrize_quantities=True, alpha=0.5, delta=0.1, n_cycles=20000, max_time = 5*60, solver_data_package = None, only_sign = False ):
      solver = data.solvers[C]

      block_names = [name for name,g in solver.G0_iw]
      N_states = len(solver.G0_iw[block_names[0]].data[0,0,:])
      gf_struct = {block_names[0] : range(N_states), block_names[1] : range(N_states)}

      h_int = U * n(block_names[0],0)*n(block_names[1],0)
      for i in range(1,N_states):
        h_int += U * n(block_names[0],i)*n(block_names[1],i)

      N_s = 2      
      ALPHA = [ [ [ alpha + delta*(-1)**(s+sig) for s in range(N_s)] for i in range(N_states)] for sig in range(2) ]

      if solver_data_package is None:  solver_data_package = {}    

      solver_data_package['which_solver'] = C
      solver_data_package['solve_parameters'] = {}
      solver_data_package['solve_parameters']['U'] = U
      solver_data_package['solve_parameters']['alpha'] = ALPHA
      solver_data_package['solve_parameters']['n_cycles'] = n_cycles
      solver_data_package['solve_parameters']['max_time'] = max_time
      solver_data_package['solve_parameters']['length_cycle'] = 200
      solver_data_package['solve_parameters']['n_warmup_cycles'] = 20000
      solver_data_package['solve_parameters']['only_sign'] = only_sign
      solver_data_package['solve_parameters']['measure_nn'] = True
      solver_data_package['solve_parameters']['measure_nnt'] = False
      solver_data_package['solve_parameters']['measure_chipmt'] = False
      solver_data_package['solve_parameters']['measure_gw'] = False
      solver_data_package['solve_parameters']['measure_Mt'] = True
      solver_data_package['solve_parameters']['measure_ft'] = False
      solver_data_package['solve_parameters']['measure_g2t'] = False
      solver_data_package['solve_parameters']['measure_M4t'] = False
      solver_data_package['solve_parameters']['measure_hist'] = True
      solver_data_package['solve_parameters']['g2t_indep'] = []
      solver_data_package['solve_parameters']['post_process'] = True

      print solver_data_package['solve_parameters']
       
      solver_data_package['G0_iw'] = solver.G0_iw
      solver_data_package['D0_iw'] = solver.D0_iw 
      solver_data_package['Jperp_iw'] = solver.Jperp_iw 

      solver_data_package['construct|run|exit'] = 1

      if MASTER_SLAVE_ARCHITECTURE and (mpi.size>1): 
         if mpi.is_master_node(): print "broadcasting solver_data_package!!"
         solver_data_package = mpi.bcast(solver_data_package)

      if mpi.is_master_node(): print "about to run "
      dct = deepcopy(solver_data_package['solve_parameters'])
      del dct['U']
      solver.solve(h_int = h_int, **dct )
      if mpi.is_master_node(): print "average sign: ",solver.average_sign
      if not only_sign:
        G_iw = deepcopy(solver.G_iw)
        Sigma_iw = deepcopy(solver.Sigma_iw)
        if symmetrize_quantities:
          symmetrize_blockgf(G_iw)
          symmetrize_blockgf(Sigma_iw)

        data.G_imp_iw[C] << G_iw['up']   
        data.Sigma_imp_iw[C] << Sigma_iw['up']
      else:
        return solver.average_sign

    @staticmethod
    def slave_run(solver_data_package, printout=True):
      while True:
        if printout: print "[Node ",mpi.rank,"] waiting for instructions..."

        solver_data_package = mpi.bcast(solver_data_package)

        if printout: print "[Node ",mpi.rank,"] received instructions!!!"

        if solver_data_package is None: 
          if printout: print "[Node ",mpi.rank,"] solver_data_package is None, will exit now. Goodbye."          
          break

        if solver_data_package['construct|run|exit'] == 0:     
          if printout: print "[Node ",mpi.rank,"] constructing solvers!!!"
          solvers = {}
          impurity_struct = solver_data_package['impurity_struct']
          for C in impurity_struct.keys():
            solver_struct = {'up': impurity_struct[C], 'dn': impurity_struct[C]}  
            solver_data_package['constructor_parameters']['gf_struct'] = solver_struct        
            solvers[C] = Solver( **(solver_data_package['constructor_parameters']) )

        if solver_data_package['construct|run|exit'] == 1:     
          if printout: print "[Node ",mpi.rank,"] about to run..."
          solver = solvers[solver_data_package['which_solver']] 
          solver.G0_iw << solver_data_package['G0_iw']
          solver.D0_iw << solver_data_package['D0_iw']
          solver.Jperp_iw << solver_data_package['Jperp_iw']     

          block_names = [name for name,g in solver.G0_iw]
          N_states = len(solver.G0_iw[block_names[0]].data[0,0,:])
          gf_struct = {block_names[0] : range(N_states), block_names[1] : range(N_states)}
          U = solver_data_package['solve_parameters']['U']
          h_int = U * n(block_names[0],0)*n(block_names[1],0)
          for i in range(1,N_states):
            h_int += U * n(block_names[0],i)*n(block_names[1],i)
     
          try:
            dct = deepcopy(solver_data_package['solve_parameters'])
            del dct['U']
            solver.solve(h_int = h_int, **dct )

            if printout: print "[Node ",mpi.rank,"] finished running successfully!"
          except Exception as e:
            print "[Node ",mpi.rank,"] ERROR: crash during running solver" 

        if solver_data_package['construct|run|exit'] == 2: 
          if printout: print "[Node ",mpi.rank,"] received exit signal, will exit now. Goodbye."          
          break

    @staticmethod
    def dump(solver, archive_name, suffix=''):    
      A = HDFArchive(archive_name)
      #stuff from the solver 
      A['D0_iw%s'%suffix] = solver.D0_iw
      A['Jperp_iw%s'%suffix] = solver.Jperp_iw
      A['mc_sign%s'%suffix] = solver.average_sign
      A['G_iw%s'%suffix] = solver.G_iw
      A['Sigma_iw%s'%suffix] = solver.Sigma_iw
      A['nn_tau%s'%suffix] = solver.nn_tau
      A['nn%s'%suffix] = solver.nn
      A['chipm_tau%s'%suffix] = solver.chipm_tau
      A['nn_iw%s'%suffix] = solver.nn_iw
      A['chipm_iw%s'%suffix] = solver.chipm_iw
      A['G0_iw%s'%suffix] = solver.G0_iw
      A['G0_shift_iw%s'%suffix] = solver.G0_shift_iw
      A['hist%s'%suffix] = solver.histogram
      A['M_tau%s'%suffix] = solver.M_tau
      A['M_iw%s'%suffix] = solver.M_iw  
        
        
################################ cthyb #########################################      
  class cthyb:
    @staticmethod
    def initialize_solvers(data, solver_data_package = None):
      if solver_data_package is None: solver_data_package = {}
      
      n_tau = 2000 
      for C in data.impurity_struct.keys():
        if len(data.impurity_struct[C])>16: 
          n_tau=1000
      assert n_tau>2*data.n_iw, "solvers.ctint.initialize_solvers: ERROR! n_tau too small!!" 

      solver_data_package['impurity_struct'] = data.impurity_struct
      solver_data_package['constructor_parameters']={}
      solver_data_package['constructor_parameters']['beta'] = data.beta
      solver_data_package['constructor_parameters']['n_iw'] = data.n_iw
      solver_data_package['constructor_parameters']['n_tau'] = n_tau
      solver_data_package['construct|run|exit'] = 0

      if MASTER_SLAVE_ARCHITECTURE and (mpi.size>1): solver_data_package = mpi.bcast(solver_data_package)

      for C in data.impurity_struct.keys():
        solver_struct = {'up': data.impurity_struct[C], 'dn': data.impurity_struct[C]}  
        solver_data_package['constructor_parameters']['gf_struct'] = solver_struct
        data.solvers[C] = cthybSolver( **solver_data_package['constructor_parameters'] )


    @staticmethod
    def run(data, C, U, symmetrize_quantities=True, n_cycles=20000, max_time = 5*60, solver_data_package = None ):
      solver = data.solvers[C]

      block_names = [name for name,g in solver.G0_iw]

      N_states = len(solver.G0_iw[block_names[0]].data[0,0,:])
      gf_struct = {block_names[0] : range(N_states), block_names[1] : range(N_states)}

      h_int = U * n(block_names[0],0)*n(block_names[1],0)
      for i in range(1,N_states):
        h_int += U * n(block_names[0],i)*n(block_names[1],i)

      QN = [ sum( [n(bl,i) for i in range(N_states)], Operator() ) for bl in block_names ]

      if solver_data_package is None:  solver_data_package = {}        

      solver_data_package['which_solver'] = C
      solver_data_package['solve_parameters'] = {}
      solver_data_package['solve_parameters']['U'] = U
      solver_data_package['solve_parameters']["max_time"] = max_time
      solver_data_package['solve_parameters']["random_name"] = ""
      solver_data_package['solve_parameters']["random_seed"] = 123 * mpi.rank + 567
      solver_data_package['solve_parameters']["length_cycle"] = 50
      solver_data_package['solve_parameters']["n_warmup_cycles"] = 5000
      solver_data_package['solve_parameters']["n_cycles"] = n_cycles
      solver_data_package['solve_parameters']["measure_density_matrix"] = True
      solver_data_package['solve_parameters']["use_norm_as_weight"] = True
      solver_data_package['solve_parameters']["partition_method"] = "quantum_numbers"

      print solver_data_package['solve_parameters']
       
      solver_data_package['G0_iw'] = solver.G0_iw

      solver_data_package['construct|run|exit'] = 1

      if MASTER_SLAVE_ARCHITECTURE and (mpi.size>1): 
         if mpi.is_master_node(): print "broadcasting solver_data_package!!"
         solver_data_package = mpi.bcast(solver_data_package)

      if mpi.is_master_node(): print "about to run "
      dct = deepcopy(solver_data_package['solve_parameters'])
      del dct['U']
      solver.solve(h_int = h_int, quantum_numbers = QN, **dct )

      G_tau = deepcopy(solver.G_tau['up'])     
      if symmetrize_quantities:
        G_tau.data[:,:,:] = 0.0
        for bl in block_names:
          G_tau.data[:,:,:] += solver.G_tau[bl].data[:,:,:]
        G_tau.data[:,:,:]/=len(block_names)   

      data.G_imp_iw[C] << Fourier(G_tau)
      fit_and_overwrite_tails_on_G(data.G_imp_iw[C],starting_iw=7.0)
      G0 = data.Gweiss_iw[C].copy()
      fit_and_overwrite_tails_on_G(G0,starting_iw=7.0)      
      for wi in range(data.nw):   
        data.Sigma_imp_iw[C].data[wi,:,:] = inv(G0.data[wi,:,:]) - inv(data.G_imp_iw[C].data[wi,:,:])
      fit_and_overwrite_tails_on_Sigma(data.Sigma_imp_iw[C],starting_iw=7.0)

    @staticmethod
    def slave_run(solver_data_package, printout=True):
      while True:
        if printout: print "[Node ",mpi.rank,"] waiting for instructions..."

        solver_data_package = mpi.bcast(solver_data_package)

        if printout: print "[Node ",mpi.rank,"] received instructions!!!"

        try:
          if solver_data_package['construct|run|exit'] == 0:     
            if printout: print "[Node ",mpi.rank,"] constructing solvers!!!"
            solvers = {}
            impurity_struct = solver_data_package['impurity_struct']
            for C in impurity_struct.keys():
              solver_struct = {'up': impurity_struct[C], 'dn': impurity_struct[C]}  
              solver_data_package['constructor_parameters']['gf_struct'] = solver_struct        
              solvers[C] = cthybSolver( **(solver_data_package['constructor_parameters']) )

          if solver_data_package['construct|run|exit'] == 1:     
            if printout: print "[Node ",mpi.rank,"] about to run..."
            solver = solvers[solver_data_package['which_solver']] 
            solver.G0_iw << solver_data_package['G0_iw']    

            block_names = [name for name,g in solver.G0_iw]
            N_states = len(solver.G0_iw[block_names[0]].data[0,0,:])
            gf_struct = {block_names[0] : range(N_states), block_names[1] : range(N_states)}
            U = solver_data_package['solve_parameters']['U']
            h_int = U * n(block_names[0],0)*n(block_names[1],0)
            for i in range(1,N_states):
              h_int += U * n(block_names[0],i)*n(block_names[1],i)
            QN = [ sum( [n(bl,i) for i in range(N_states)], Operator() ) for bl in block_names ]
            try:
              dct = deepcopy(solver_data_package['solve_parameters'])
              del dct['U']
              solver.solve(h_int = h_int, quantum_numbers = QN, **dct )

              if printout: print "[Node ",mpi.rank,"] finished running successfully!"
            except Exception as e:
              print "[Node ",mpi.rank,"] ERROR: crash during running solver" 

          if solver_data_package['construct|run|exit'] == 2: 
            if printout: print "[Node ",mpi.rank,"] received exit signal, will exit now. Goodbye."    
            break
        except:
          if printout: print "[Node ",mpi.rank,"] something went wrong. Will exit now! Goodbye."    
          break


    @staticmethod
    def dump(solver, archive_name, suffix=''):    
      A = HDFArchive(archive_name)
      #stuff from the solver 
      A['mc_sign%s'%suffix] = solver.average_sign      
      A['G_tau%s'%suffix] = solver.G_tau
      A['nn%s'%suffix] = solver.density_matrix
      A['G0_iw%s'%suffix] = solver.G0_iw
      A['Delta_tau%s'%suffix] = solver.Delta_tau
      #A['hist%s'%suffix] = solver.get_perturbation_order_total

