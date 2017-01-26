#from pytriqs.operators import *
#from pytriqs.archive import *
#from pytriqs.gf.local import *
#from pytriqs.arrays import BlockMatrix, BlockMatrixComplex
import pytriqs.utility.mpi as mpi
import numpy
from amoeba import *
from impurity_solvers import solvers

class nested_mains:
  @staticmethod 
  def selfenergy(data, mapping = lambda C,x,y: [0,0,0]): #[i,j,coef]
    data.get_Sigmaijw()
    data.get_Sigmakw() 
    data.get_Sigma_loc()  

  @staticmethod 
  def lattice(data, n, ph_symmetry, accepted_mu_range=[-2.0,2.0]):
    if mpi.is_master_node(): print "GW_mains: lattice:  n: ",n,", ph_symmetry",ph_symmetry, "accepted mu_range: ",accepted_mu_range

    if (n is None) or ((n==0.5) and ph_symmetry):
      if mpi.is_master_node(): print "no mu search to be performed! it is your duty to set the chemical potential to U/2. mu =",data.mus['up']
      data.get_n()
    else:
      def func(var, data):
        mu = var[0]
        dt = data[0]
        #print "func call! mu: ", mu, " n: ",dt.ns['up']
        n= data[1] 
        dt.mus['up'] = mu
        if 'down' in dt.fermionic_struct.keys(): dt.mus['down'] = dt.mus['up']
        dt.get_n()        #print "funcvalue: ",-abs(n - dt.ns['up'])  
        #print "dt.ns: ", dt.ns
          
        val = 1.0-abs(n - dt.ns['up'])  
        if mpi.is_master_node(): print "amoeba func call: val = ",val
        if val != val: return -1e+6
        else: return val

      if mpi.is_master_node(): print "about to do mu search:"

      guesses = [data.mus['up'], 0.0, -0.1, -0.3, -0.4, -0.5, -0.7, 0.3, 0.5, 0.7]
      found = False  
      for l in range(len(guesses)):
        varbest, funcvalue, iterations = amoeba(var=[guesses[l]],
                                              scale=[0.01],
                                              func=func, 
                                              data = [data, n],
                                              itmax=30,
                                              ftolerance=1e-2,
                                              xtolerance=1e-2,
                                              known_max = 1.0,
                                              known_max_accr = 5e-5)
        if (varbest[0]>accepted_mu_range[0] and varbest[0]<accepted_mu_range[1]) and (abs(funcvalue-1.0)<1e-2): #change the bounds for large doping
          found = True 
          func(varbest, [data, n])
          break 
        if l+1 == len(guesses):
          if mpi.is_master_node(): print "mu search FAILED: doing a scan..."

          mu_grid = numpy.linspace(-1.0,0.3,50)
          func_values = [func(var=[mu], data=[data,n]) for mu in mu_grid]
          if mpi.is_master_node(): 
            print "func_values: "
            for i in range(len(mu_grid)):
              print "mu: ",mu_grid[i], " 1-abs(n-n): ", func_values[i]
          mui_max = numpy.argmax(func_values)
          if mpi.is_master_node(): print "using mu: ", mu_grid[mui_max]
          data.mus['up'] = mu_grid[mui_max]
          if 'down' in data.fermionic_struct.keys(): data.mus['down'] = data.mus['up']
          data.get_n()
             
      if mpi.is_master_node() and found:
        print "guesses tried: ", l  
        print "mu best: ", varbest
        print "1-abs(diff n - data.n): ", funcvalue
        print "iterations used: ", iterations

    data.get_Gijw()

  @staticmethod
  def pre_impurity(data):
    data.get_Gweiss()    

  @classmethod
  def optimize_alpha_and_delta(cls, data, C, U, max_time, solver_data_package):
    if mpi.is_master_node(): print "nested_mains.optimize_alpha_and_delta"
    alphas = [0.5]#[0.3,0.4,0.45,0.48, 0.5, 0.52, 0.55,0.6,0.7]
    deltas = [0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8]
    signs = numpy.zeros((len(alphas),len(deltas))) 
    breaker = False     
    for alpha in alphas:
      for delta in deltas:
        if mpi.is_master_node(): print "trying: alpha: %s delta: %s"%(alpha,delta)
        signs[alphas.index(alpha),deltas.index(delta)] = sign = solvers.ctint.run(data, C, U, symmetrize_quantities=False, 
                                                                           alpha=alpha, delta=delta, n_cycles=1000000, max_time=max_time, 
                                                                           solver_data_package=solver_data_package, only_sign = True)
        if mpi.is_master_node(): print "%s >>>> (alpha: %s, delta: %s): sign=%s"%(C,alpha,delta,sign) 
        if sign>0.8:
          breaker = True
          break
      if breaker: break
    ai, di = numpy.unravel_index(numpy.argmax(signs), (len(alphas),len(deltas)))
    max_sign = numpy.amax(signs) 
    return alphas[ai], deltas[di], max_sign 

  @staticmethod
  def impurity(data, U, symmetrize_quantities = True, alpha=0.5, delta=0.1, automatic_alpha_and_delta = False, n_cycles=20000, max_times = {'1x1': 5*60 }, solver_data_package = None, Cs = [] ):
    data.Sigma_imp_iw << 0
    for C in (data.impurity_struct.keys() if Cs==[] else Cs):
      solver_struct = {'up': data.impurity_struct[C], 'dn': data.impurity_struct[C]}        
      for key in solver_struct.keys():
        data.solvers[C].G0_iw[key] << data.Gweiss_iw[C]
      data.solvers[C].Jperp_iw << 0.0
      data.solvers[C].D0_iw << 0.0
      if automatic_alpha_and_delta:
        if mpi.is_master_node(): print "about to optimize alpha and delta for impurity",C
        shorttime = min(600, max(30,int(max_times[C]/100)))
        if mpi.is_master_node(): print "time per alpha,delta", shorttime
        alpha, delta, max_sign = nested_mains.optimize_alpha_and_delta(data, C, U, shorttime, solver_data_package) 
        if mpi.is_master_node(): print "%s >>>> best (alpha=%s,delta=%s): max_sign=%s"%(C,alpha,delta,max_sign)
      if mpi.is_master_node(): print "nested_mains.impurity: launching impurity",C
      solvers.ctint.run(data, C, U, symmetrize_quantities, alpha, delta, n_cycles, max_times[C], solver_data_package)

  @staticmethod
  def impurity_cthyb(data, U, symmetrize_quantities = True, n_cycles=20000, max_times = {'1x1': 5*60 }, solver_data_package = None, Cs = [] ):
    data.Sigma_imp_iw << 0
    for C in (data.impurity_struct.keys() if Cs==[] else Cs):
      solver_struct = {'up': data.impurity_struct[C], 'dn': data.impurity_struct[C]}        
      for key in solver_struct.keys():
        data.solvers[C].G0_iw[key] << data.Gweiss_iw[C]

      if mpi.is_master_node(): print "nested_mains.impurity_cthyb: launching impurity",C
      solvers.cthyb.run(data, C, U, symmetrize_quantities, n_cycles, max_times[C], solver_data_package)
      
#--------------------------------------------- cumul_nested ----------------------------------#
class cumul_nested_mains:
  @staticmethod 
  def cumulant(data, mapping = lambda C,x,y: [0,0,0]): #[i,j,coef]
    data.get_g_imp()
    data.get_gijw()
    data.get_gkw()

  @staticmethod 
  def selfenergy(data, mapping = lambda C,x,y: [0,0,0]): #[i,j,coef]
    data.get_Sigmakw()


