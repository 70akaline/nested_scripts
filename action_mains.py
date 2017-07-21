#from pytriqs.operators import *
#from pytriqs.archive import *
#from pytriqs.gf.local import *
#from pytriqs.arrays import BlockMatrix, BlockMatrixComplex
import pytriqs.utility.mpi as mpi
import numpy
from amoeba import *
from impurity_solvers import solvers
from tail_fitters import fit_bosonic_tail
from action_cautionaries import symmetrize_cluster_impurity_bosonic

class nested_mains:
  @staticmethod 
  def selfenergy(data):
    data.get_Sigmaijw()
    data.get_Sigmakw() 
    data.get_Sigma_loc()  

  @staticmethod 
  def lattice(data, n, ph_symmetry, accepted_mu_range=[-2.0,2.0]):
    if mpi.is_master_node(): print "GW_mains: lattice:  n: ",n,", ph_symmetry",ph_symmetry, "accepted mu_range: ",accepted_mu_range

    if (n is None) or ((n==0.5) and ph_symmetry):
      if mpi.is_master_node(): print "no mu search to be performed! it is your duty to set the chemical potential to U/2. mu =",data.get_mu()
      print 'n on the lattice : ', data.get_n()
    else:
      def func(var, data):
        mu = var[0]
        dt = data[0]
        #print "func call! mu: ", mu, " n: ",dt.ns['up']
        n= data[1] 
        #dt.mus['up'] = mu
        #if 'down' in dt.fermionic_struct.keys(): dt.mus['down'] = dt.mus['up']
        dt.set_mu(mu)
        dtn = dt.get_n()        #print "funcvalue: ",-abs(n - dt.ns['up'])  
        #print "dt.ns: ", dt.ns
          
        val = 1.0-abs(n - dtn)  
        if mpi.is_master_node(): print "amoeba func call: mu: %.2f n: %.2f dtn: %.2f val = "%(mu,n,dtn),val
        if val != val: return -1e+6
        else: return val

      if mpi.is_master_node(): print "about to do mu search:"

      guesses = [data.get_mu(), 0.0, -0.1, -0.3, -0.4, -0.5, -0.7, 0.3, 0.5, 0.7]
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
          #data.mus['up'] = mu_grid[mui_max]
          #if 'down' in data.fermionic_struct.keys(): data.mus['down'] = data.mus['up']
          data.set_mu(mu_grid[mui_max])  
          data.get_n()
             
      if mpi.is_master_node() and found:
        print "guesses tried: ", l  
        print "mu best: ", varbest
        print "1-abs(diff n - data.n): ", funcvalue
        print "iterations used: ", iterations

    data.get_Gijw()

  @staticmethod
  def pre_impurity(data, Cs=[]):
    data.get_Gweiss() 
    for C in (data.impurity_struct.keys() if Cs==[] else Cs):
      solver_struct = {'up': data.impurity_struct[C], 'dn': data.impurity_struct[C]}        
      for key in solver_struct.keys():
        data.solvers[C].G0_iw[key] << data.Gweiss_iw[C]
      data.solvers[C].Jperp_iw << 0.0
      data.solvers[C].D0_iw << 0.0   

  @classmethod
  def optimize_alpha_and_delta(cls, data, C, U, max_time, solver_data_package):
    if mpi.is_master_node(): print "nested_mains.optimize_alpha_and_delta"
    alphas = [0.5]#[0.3,0.4,0.45,0.48, 0.5, 0.52, 0.55,0.6,0.7]
    deltas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
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
  def impurity(data, U, symmetrize_quantities = True, alpha=0.5, delta=0.1, automatic_alpha_and_delta = False, n_cycles=20000, max_times = {'1x1': 5*60 }, solver_data_package = None, Cs = [], bosonic_measures = False ):
    if mpi.is_master_node(): print "nested_mains.impurity. max_times",max_times
    data.Sigma_imp_iw << 0
    for C in (data.impurity_struct.keys() if Cs==[] else Cs):  
      if automatic_alpha_and_delta:
        if mpi.is_master_node(): print "about to optimize alpha and delta for impurity",C
        shorttime = min(600, max(30,int(max_times[C]/100)))
        if mpi.is_master_node(): print "time per alpha,delta", shorttime
        alpha, delta, max_sign = nested_mains.optimize_alpha_and_delta(data, C, U, shorttime, solver_data_package) 
        if mpi.is_master_node(): print "%s >>>> best (alpha=%s,delta=%s): max_sign=%s"%(C,alpha,delta,max_sign)
      if mpi.is_master_node(): print "nested_mains.impurity: launching impurity",C
      solvers.ctint.run(data, C, U, symmetrize_quantities, alpha, delta, n_cycles, max_times[C], solver_data_package, bosonic_measures=bosonic_measures)

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
  def cumulant(data): #[i,j,coef]
    data.get_g_imp()
    data.get_gijw()
    data.get_gkw()

  @staticmethod 
  def selfenergy(data,): #[i,j,coef]
    data.get_Sigmakw()

#--------------------------------------------- cumul_nested ----------------------------------#
class dca_mains:
  @staticmethod 
  def selfenergy(data):
    data.get_SigmaR()
    data.get_SigmaK()  

class dca_plus_mains:
  @staticmethod 
  def selfenergy(data):
    dca_mains.selfenergy(data)
    data.get_XiK()
    data.get_XiR()
    data.get_Xik()
    data.get_Sigmaimpk()
    data.get_Sigmakw()

#--------------------------------------------- nested edmft ----------------------------------#

class nested_edmft_mains:

  @staticmethod 
  def selfenergy(data):
    nested_mains.selfenergy(data)
    data.get_Pijnu()
    data.get_Pqnu() 
    data.get_P_loc()  

  @staticmethod 
  def lattice(data, n, ph_symmetry, accepted_mu_range=[-2.0,2.0]):
    nested_mains.lattice(data, n, ph_symmetry, accepted_mu_range) 
    data.get_Wqnu()
    data.get_Wijnu()
    data.get_W_loc()

  @classmethod
  def extract_chi_imp_from_solver(cls, data, su2_symmetry=False): #this function doesn't belong here. move it to file with formulas
    for C in data.impurity_struct.keys():
      for A in data.bosonic_struct.keys():
        CA = C+"|"+A
        blocks = [name for name, nn_iw in data.solvers[C].nn_iw]
        blocks2 = data.solvers[C].nn.block_names
        assert set(blocks)==set(blocks2), "block structure crap"
        assert 'up|up' in blocks, "wrong solver block structure"
        assert 'dn|dn' in blocks, "wrong solver block structure"
        assert 'up|dn' in blocks, "wrong solver block structure"
        assert 'dn|up' in blocks, "wrong solver block structure"      
        
        if A == '0': 
          sgn = 1
          pref = 1
        if A == '1':
          sgn = -1
          pref = int(not su2_symmetry)
        if A == '+-': assert False, "not implemented"
        data.chi_imp_iw[CA] << data.solvers[C].nn_iw['up|up'] \
                                + data.solvers[C].nn_iw['dn|dn'] \
                                + sgn*data.solvers[C].nn_iw['up|dn'] \
                                + sgn*data.solvers[C].nn_iw['dn|up']
        for i in data.impurity_struct[C]:
          for j in data.impurity_struct[C]:
            data.chi_imp_iw[CA].data[data.nnu/2,i,j] -= pref*data.beta\
                                                           *(data.solvers[C].nn['up|up'][i,i]+sgn*data.solvers[C].nn['dn|dn'][i,i])\
                                                           *(data.solvers[C].nn['up|up'][j,j]+sgn*data.solvers[C].nn['dn|dn'][j,j])

  @classmethod
  def prepare_Jperp_iw(cls, data, Cs=[]): #takes a single block
    for C in (data.impurity_struct.keys() if Cs==[] else Cs): 
      data.solvers[C].Jperp_iw << 0.0
  #  Jperp_iw << Uweiss_iw
  #  fixed_coeff = TailGf(1,1,2,-1) #not general for clusters
  #  fixed_coeff[-1] = array([[0.]])
  #  fixed_coeff[0] = array([[0.]])
  #  nmax = Jperp_iw.mesh.last_index()
  #  nmin = nmax/2
  #  Jperp_iw.fit_tail(fixed_coeff, 5, nmin, nmax, True) #!!!!!!!!!!!!!!!1

  @classmethod
  def prepare_D0_iw(cls, data, Cs=[]):
    print "Cs: ",Cs
    blocks = ['up','dn']
    for C in (data.impurity_struct.keys() if Cs==[] else Cs): 
      for bl1 in blocks:
        for bl2 in blocks:
          d0 =  data.solvers[C].D0_iw[bl1+'|'+bl2] 
          d0 << 0.0
          for A in data.bosonic_struct.keys():
            pref = 1.0
            if (bl1!=bl2) and (A=='1'):
              pref *= -1.0        
            d0 << d0 + pref*data.Uweiss_dyn_iw[C+"|"+A]
          fit_bosonic_tail(d0, no_static=True, overwrite_tail=True, max_order=5)

  @staticmethod
  def pre_impurity(data, Cs=[], freeze_Uweiss=False):
    nested_mains.pre_impurity(data,Cs)  
    if not freeze_Uweiss: data.get_Uweiss()    
    nested_edmft_mains.prepare_D0_iw(data, Cs)
    nested_edmft_mains.prepare_Jperp_iw(data,Cs)

  @staticmethod
  def post_impurity(data, Cs=[], su2_symmetry=False, identical_pairs = [], homogeneous_pairs = []):
    nested_edmft_mains.extract_chi_imp_from_solver(data, su2_symmetry)
    if identical_pairs!=[]: symmetrize_cluster_impurity_bosonic(data.chi_imp_iw, identical_pairs, name="chi_imp_iw")
    data.get_W_imp() 
    if identical_pairs!=[]: symmetrize_cluster_impurity_bosonic(data.W_imp_iw, identical_pairs, name="W_imp_iw")
    if homogeneous_pairs!=[]: symmetrize_cluster_impurity_bosonic(data.W_imp_iw, homogeneous_pairs, name="W_imp_iw")
    data.get_P_imp()
    if identical_pairs!=[]: symmetrize_cluster_impurity_bosonic(data.W_imp_iw, identical_pairs, name="P_imp_iw")

