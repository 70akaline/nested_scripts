import numpy
from functools import partial
import math, time, cmath
from math import cos, exp, sin, log, log10, pi, sqrt
import random
import numpy
import numpy.fft
from numpy import matrix, array, zeros
from pytriqs.operators import *
from pytriqs.archive import *
from pytriqs.gf.local import *
from pytriqs.arrays import BlockMatrix, BlockMatrixComplex
import pytriqs.utility.mpi as mpi
#from glattice_tools.core import *  
#from glattice_tools.multivar import *  
#from trilex.tools import *
#from cthyb_spin import Solver  
#from selfconsistency.useful_functions import adjust_n_points
#from selfconsistency.provenance import hash_dict
import copy

#from impurity_solvers import *

####################################################################################
#  This file deals with data containers. Data classes know about
#       - numerical parameters
#        - choice of containers
#        - choice of discretization schemes
#
#  IBZ is about k discretization, use of symmetry and resampling
#  mats_freq is about matsubara frequencies and resampling (changing the number of 
#     points or interpolating a function to a matsuara grid at a different temp.)
#  function_applicators contain function that fill the containers with given 
#     scalar functions

#--------------------------------------------------------------------------#

class interpolation:
  @staticmethod
  def linear(x, x1, x2, Q1, Q2):
   return Q1 + (Q2 - Q1)*(x - x1)/(x2 - x1)

  @staticmethod
  def bilinear(x,y, x1,x2,y1,y2, Q11, Q12, Q21, Q22):
    return ( Q11*(x2-x)*(y2-y) + Q21*(x-x1)*(y2-y)+ Q12*(x2-x)*(y-y1) + Q22*(x-x1)*(y-y1) ) / ( (x2-x1)*(y2-y1) )


#--------------------------------------------------------------------------#
class mats_freq:
  @staticmethod    
  def fermionic( n, beta): return  ( 2*n + 1 )*pi/beta
  @staticmethod    
  def bosonic( m, beta): return  ( 2*m )*pi/beta

  @staticmethod    
  def fermionic_n_from_w( w, beta): return  int(((w*beta)/math.pi-1.0)/2.0)
  @staticmethod    
  def bosonic_m_from_nu( nu, beta): return  int(((nu*beta)/math.pi)/2.0)


  @staticmethod          
  def change_temperature(Q_old, Q_new, ws_old, ws_new, Q_old_wrapper=lambda iw: 0.0): #can be also used to change the number of points
    j_old = 0
    for i in range(len(ws_new)):     
      for j in range(j_old, len(ws_old)):
        if ( (ws_old[j]>ws_new[i]) and (j==0) ) or ( (ws_old[j]<=ws_new[i]) and (j==len(ws_old)-1) ):
          Q_new[i] = Q_old_wrapper(1j*ws_new[i])          
          j_old = j
          break          
        if (ws_old[j]<=ws_new[i]) and (ws_old[j+1]>ws_new[i]):
          Q_new[i] = interpolation.linear(ws_new[i], ws_old[j], ws_old[j+1], Q_old[j], Q_old[j+1])
          j_old = j
          break

  @staticmethod
  def change_temperature_gf(Q_old, Q_new): #can be used to change the number of points
    n1 = len(Q_old.data[0,:,0])
    n2 = len(Q_old.data[0,0,:])  
    n1_new = len(Q_new.data[0,:,0])
    n2_new = len(Q_new.data[0,0,:])  
    assert  (n1 == n1_new) and (n2 == n2_new), "the new Gf needs to have the same target space as the old Gf!"

    ws_old = [w.imag for w in Q_old.mesh]
    ws_new = [w.imag for w in Q_new.mesh]
    #print "len ws old: ", len(ws_old), "ws_old[-1]:", ws_old[-1]
    #print "len ws new: ", len(ws_new), "ws_new[-1]:", ws_new[-1]

    fixed_coeff = TailGf(n1,n2,1,-1)
    fixed_coeff[-1] = numpy.zeros((n1,n2))
    nmax = Q_old.mesh.last_index()
    nmin = nmax/2
    Q_old.fit_tail(fixed_coeff, 3, nmin, nmax, False)
    for i in range(n1):
      for j in range(n2):
        tail = [Q_old.tail[l][i,j] for l in range(4)]
        wrapper = lambda iw:  tail[0]\
                            + tail[1]/(iw)\
                            + tail[2]/(iw**2.0)\
                            + tail[3]/(iw**3.0)

        mats_freq.change_temperature(Q_old.data[:,i,j], Q_new.data[:,i,j], ws_old, ws_new, wrapper)   

  @staticmethod
  def get_tail_from_numpy_array(Q, beta, statistic, n_iw, positive_only=False): #get a tail for a gf stored in a numpy array
    g = GfImFreq(indices = [0], beta = beta, n_points = n_iw, statistic = statistic)
    if statistic=='Fermion': 
      nw = n_iw*2
      if positive_only: 
        nw = n_iw
        shift = n_iw
    if statistic=='Boson':
      nw = n_iw*2-1
      if positive_only: 
        shift = n_iw-1 
        nw = n_iw      
    for i in range(nw):
      g.data[i+shift,0,0] = Q[i]
      if positive_only: 
        if statistic=='Fermion': 
          g.data[shift-i-1,0,0] = Q[i]
        if statistic=='Boson': 
          g.data[shift-i,0,0] = Q[i]
      fixed_coeff = TailGf(1,1,1,-1) 
      fixed_coeff[-1] = array([[0.]])
      nmax = n_iw-1
      nmin = 3*nmax/4
      g.fit_tail(fixed_coeff,3,nmin,nmax, False) 
      tail = [g.tail[i][0,0] for i in range(4)]
    return tail


#--------------------------------------------------------------------------#
class IBZ:
  @staticmethod
  def k_from_i(i, nk, k_max = 2.0*pi):
    return 1.*i*k_max/nk #1. is a cautionary measure against integer k_max

  @staticmethod
  def k_grid(nk, k_max = 2.0*pi):
    return numpy.array([IBZ.k_from_i(i, nk, k_max) for i in range(nk)])

  @staticmethod
  def multiplicity(kxi, kyi, nk):
    if ( kxi==0 and kyi==0 )or( kxi==nk/2 and kyi==nk/2 ): return 1.0
    if ( kxi==nk/2 and kyi==0 )or( kxi==0 and kyi==nk/2 ): return 2.0
    if ( kxi==nk/2 or kyi==0 or kxi==0 or kyi==nk/2 or kxi==kyi): return 4.0
    return 8.0 

  @staticmethod
  def resample(Q_old, Q_new, ks_old, ks_new, k_max=2.0*math.pi):
    nk_new = len(ks_new)
    nk_old = len(ks_old)
    #print "nk_old: ",nk_old," nk_new: ",nk_new
    dk_old = k_max/nk_old
    #print "dk_old: ",dk_old
    for i in range(nk_new):
      x = ks_new[i]
      #print "x: ", x
      i1 = int(ks_new[i]/dk_old)
      #print "i1: ", i1
      x1 = ks_old[i1]
      if (i1==nk_old-1):
        i2 = 0
        x2 = k_max
      else:
        i2 = i1+1
        x2 = ks_old[i2]
      for j in range(nk_new):
        y = ks_new[j]
        #print "y: ", y
        j1 = int(ks_new[j]/dk_old)
        #print "j1: ", j1
        y1 = ks_old[j1]
        if (j1==nk_old-1):
          j2 = 0
          y2 = k_max
        else: 
          j2 = j1+1
          y2 = ks_old[j2]
        Q_new[i,j] = interpolation.bilinear( x , y, x1, x2, y1, y2, Q_old[i1,j1], Q_old[i1,j2], Q_old[i2,j1], Q_old[i2,j2])

  @staticmethod
  def copy_by_symmetry(Q, nk=None): #nk doesn't do anything anymore
    nk = len(Q[0,:])
    assert len(Q[0,:])==len(Q[:,0]), "discretization must respect full lattice symmetry"
    assert nk%2 == 0, "copy_by_symmetry: nk must be even"
    for kxi in range(nk/2+1): 
      for kyi in range(kxi+1):
         #mirror
         Q[kyi,kxi] = Q[kxi,kyi]
    for kxi in range(nk/2+1): 
      for kyi in range(nk/2+1):
         if (kxi == 0 and kyi==0) or (kxi == nk/2 and kyi==nk/2): continue
         #rotate
         Q[-kyi,kxi] = Q[kxi,kyi]
         Q[kyi,-kxi] = Q[kxi,kyi]
         Q[-kxi,-kyi] = Q[kxi,kyi]

  @staticmethod
  def copy_by_weak_symmetry(Q, nk):
    assert nk%2 == 0, "copy_by_weak_symmetry: nk must be even"
    for kxi in range(nk/2+1): 
      for kyi in range(nk/2+1):
         if (kxi == 0 and kyi==0) or (kxi == nk/2 and kyi==nk/2): continue
         #rotate
         Q[-kxi,kyi] = Q[kxi,kyi]
         Q[kxi,-kyi] = Q[kxi,kyi]
         Q[-kxi,-kyi] = Q[kxi,kyi]

  @staticmethod
  def get_Qkw_on_path(Q, wi, only_positive = False):
      nk = len(Q[0,0,:])
      #print nk
      ks = numpy.linspace(0,2*pi,nk, endpoint=False)
      #print ks
      if only_positive:
        shift = 0  
      else:  
        shift = len(Q[:,0,0])/2
      
      ys= []
      xs= []
      xtcs = [0.0]
      for i in range(nk/2+1):
          ys.append(Q[shift+wi,i,0])
          xs.append(ks[i])
      xtcs.append(xs[-1])    
      for i in range(1,nk/2+1):
          ys.append(Q[shift+wi,nk/2,i])
          xs.append(math.pi+ks[i])        
      counter=1    
      xtcs.append(xs[-1])
      for i in reversed(range(nk/2)):
          ys.append(Q[shift+wi,i,i]) 
          xs.append(2.0*math.pi+counter*math.sqrt(2.0)*math.pi/(nk/2.0))
          counter +=1
      xtcs.append(xs[-1]) 
      #xtcs_labels = [r"$(0,0)$",r"$(0,\pi)$",r"$(\pi,\pi)$",r"$(0,0)$"]   
      return xs,ys, xtcs#, xtcs_labels 

  @staticmethod
  def get_Qkw_on_long_path(Q, wi, only_positive = False, equidistant_checkpoints = False):
      nk = len(Q[0,0,:])
      #print nk
      ks = numpy.linspace(0,2*pi,nk, endpoint=False)

      dk = 2.0*pi/nk
      dk_diag = math.sqrt(2.0)*dk if not equidistant_checkpoints else dk
      #print ks
      if only_positive:
        shift = 0  
      else:  
        shift = len(Q[:,0,0])/2
      
      ys= []
      xs= []           
      path_covered = 0.0
      
      for i in range(nk/2):
          ys.append(Q[shift+wi,i,i]) 
          xs.append(path_covered)
          path_covered += dk_diag
    
      for i in reversed(range(1,nk/2+1)):
          ys.append(Q[shift+wi,i,nk/2])
          xs.append(path_covered)
          path_covered += dk

      for i in range(nk/2):
          ys.append(Q[shift+wi,i,nk/2-i])
          xs.append(path_covered)
          path_covered += dk_diag

      for i in reversed(range(nk/2+1)):
          ys.append(Q[shift+wi,i,0])
          xs.append(path_covered)
          path_covered += dk
      xtcs=[xs[0],xs[nk/2],xs[nk],xs[3*nk/2],xs[-1]]    
      xtcs_labels = [r"$(0,0)$",r"$(\pi,\pi)$",r"$(0,\pi)$",r"$(\pi,0)$",r"$(0,0)$"]
      return xs,ys, xtcs, xtcs_labels


################################ DATA ##########################################

class basic_data:
  def __init__(self, n_iw = 100, 
                     beta = 10.0, 
                     fermionic_struct = {'up': [0]},
                     archive_name="basic.h5"):
    self.archive_name = archive_name

    self.solvers = {}

    self.fermionic_struct = fermionic_struct

    #---------- error control
    self.err = False 

    #---------take the parameters
    self.n_iw = n_iw #number of positive mats freq
    self.nw = 2*n_iw #total number of fermionic mats freq
     
    self.beta = beta   

    g = GfImFreq(indices = [0], beta = self.beta, n_points =self.n_iw, statistic = 'Fermion')
     
    self.ws = []
    self.iws = []
    self.ws = [ w.imag for w in g.mesh ]
    self.iws = [ w for w in g.mesh ]
    assert len(self.ws) == self.nw, "Something wrong with number of points"
     
    #---------initialize containers
    self.mus = {}
    self.ns = {}
    if mpi.is_master_node(): print "fermionic_struct: ", fermionic_struct
    for U in fermionic_struct.keys(): 
      self.mus[U] = 0.0
      self.ns[U] = 0.0

    self.Sz = 0  

    #---------quantity dictionaries
    self.errors = ['err']
    self.parameters = ['n_iw', 'nw', 'beta', 'fermionic_struct','ws', 'iws' ]
    self.scalar_quantities = ['mus', 'ns', 'Sz']
    self.non_interacting_quantities = []
    self.local_quantities = []
    self.non_local_quantities = []

  def fmats_freq(self, n): return mats_freq.fermionic(n, self.beta)
  def bmats_freq(self, m): return mats_freq.bosonic(m, self.beta)

  def dump_general(self, quantities, archive_name=None, suffix=''):
    if archive_name is None:
      archive_name = self.archive_name    
    A = HDFArchive(archive_name)
    for key in quantities:
      #print "dumping ",key
      A['%s%s'%(key,suffix)] = vars(self)[key]
    del A

  def dump_errors(self, archive_name=None, suffix=''):
    self.dump_general(self.errors, archive_name, suffix)

  def dump_parameters(self, archive_name=None, suffix=''):
    self.dump_general(self.parameters, archive_name, suffix)

  def dump_scalar(self, archive_name=None, suffix=''):
    self.dump_general(self.scalar_quantities, archive_name, suffix)

  def dump_non_interacting(self, archive_name=None, suffix=''):
    self.dump_general(self.non_interacting_quantities, archive_name, suffix)

  def dump_local(self, archive_name=None, suffix=''):
    self.dump_general(self.local_quantities, archive_name, suffix)
    
  def dump_non_local(self, archive_name=None, suffix=''):
    self.dump_general(self.non_local_quantities, archive_name, suffix)   
    
  def dump_all(self, archive_name=None, suffix='', parameters_and_non_interacting_without_suffix = True):   
    if archive_name is None:
      archive_name = self.archive_name  #this part because of dump_solver which does not know about data
    try:
      self.dump_solvers(self.solvers, archive_name, suffix)
    except:
      try:
        self.dump_solvers(suffix=suffix)
      except:
        print "solvers cannot be dumped!" 
    if parameters_and_non_interacting_without_suffix:
      suffix2 = ""
    else: suffix2 =suffix

    self.dump_errors(archive_name, suffix)
    self.dump_parameters(archive_name, suffix2)
    self.dump_scalar(archive_name, suffix)
    self.dump_local(archive_name, suffix)
    self.dump_non_interacting(archive_name, suffix2)
    self.dump_non_local(archive_name, suffix)


  def construct_from_file(self, archive_name=None, suffix='', no_suffix_for_parameters_and_non_interacting = True):
    if archive_name is None:
      archive_name = self.archive_name    

    all_quantities =    self.parameters\
                      + self.scalar_quantities\
                      + self.non_interacting_quantities\
                      + self.local_quantities\
                      + self.non_local_quantities
    if mpi.is_master_node():
      A = HDFArchive(archive_name, 'r')
      for key in all_quantities: 
        print "loading ",key          
        try:
          if no_suffix_for_parameters_and_non_interacting and ((key in self.parameters) or (key in self.non_interacting_quantities)):
            vars(self)[key] = copy.deepcopy(A['%s'%(key)])
            #vars(self)[key] = A['%s'%(key)]
          else:
            vars(self)[key] = copy.deepcopy(A['%s%s'%(key,suffix)])
            #vars(self)[key] = A['%s%s'%(key,suffix)]
        except:
          print "WARNING: key ",key," not found in archive!! "  

      del A
    #if mpi.size!=1:
    #  if mpi.is_master_node(): print "mpi.size = ",mpi.size, " will now broadcast all the read quantities"
    #  for key in all_quantities:
    #    vars(self)[key] = copy.deepcopy( mpi.bcast(vars(self)[key]) ) 

#------------------------ local data --------------------------------#
class local_data(basic_data):
  def __init__(self, n_iw = 100, 
                     beta = 10.0, 
                     impurity_struct = {'1x1': [0], '1x2': [0,1], '2x2': [0,1,2,3]},
                     fermionic_struct = {'up': [0]},
                     archive_name="dmft.out.h5"):
    basic_data.__init__(self, n_iw, beta, fermionic_struct, archive_name) 
    local_data.promote(self, impurity_struct)

  def promote(self,  impurity_struct):
    self.impurity_struct = impurity_struct
    self.parameters.extend( ['impurity_struct'] )

    gs = []
    for C in self.impurity_struct.keys(): 
      gs.append ( GfImFreq(indices = self.impurity_struct[C], beta = self.beta, n_points =self.n_iw, statistic = 'Fermion') )     

    self.G_imp_iw = BlockGf(name_list = self.impurity_struct.keys(), block_list = gs, make_copies = True)
    self.Sigma_imp_iw = BlockGf(name_list = self.impurity_struct.keys(), block_list = gs, make_copies = True)
    self.Gweiss_iw = BlockGf(name_list = self.impurity_struct.keys(), block_list = gs, make_copies = True)

    gs = []
    for U in self.fermionic_struct.keys(): 
      gs.append ( GfImFreq(indices = self.fermionic_struct[U], beta = self.beta, n_points =self.n_iw, statistic = 'Fermion') )     

    self.G_loc_iw = BlockGf(name_list = self.fermionic_struct.keys(), block_list = gs, make_copies = True)
    self.Sigma_loc_iw = BlockGf(name_list = self.fermionic_struct.keys(), block_list = gs, make_copies = True)

    self.impurity_fermionic_gfs = [ 'G_imp_iw', 'Sigma_imp_iw', 'Gweiss_iw' ]
    self.local_fermionic_gfs = [ 'G_loc_iw', 'Sigma_loc_iw' ]

    self.local_quantities.extend( self.local_fermionic_gfs + self.impurity_fermionic_gfs ) 

  def w_from_wi(self, wi): return self.fmats_freq(self.n_from_wi(wi))
  def n_to_wi(self, n):    return n+self.nw/2  
  def n_from_wi(self, wi): return wi-self.nw/2  

  def change_beta(self, beta_new, n_iw_new = None, finalize = True):
    if mpi.is_master_node(): print ">>>>>>>> CHANGING BETA!!!!"
    if n_iw_new is None: n_iw_new = self.n_iw
    nw_new = n_iw_new*2
    ntau_new = 5*n_iw_new 
    #---impurity gfs
    gs = []
    for C in self.impurity_struct.keys():
      gs.append ( GfImFreq(indices = self.impurity_struct[C], beta = beta_new, n_points = n_iw_new, statistic = 'Fermion') )
    bgf = BlockGf(name_list = self.impurity_struct.keys(), block_list = gs, make_copies = False)
    ws_new = [w.imag for w in gs[0].mesh] 
    for key in self.impurity_fermionic_gfs:  
      for C in self.impurity_struct.keys():       
        mats_freq.change_temperature_gf(vars(self)[key][C], bgf[C])
      vars(self)[key] = bgf.copy()

    #---lattloc gfs
    gs = []
    for U in self.fermionic_struct.keys():
      gs.append ( GfImFreq(indices = self.fermionic_struct[U], beta = beta_new, n_points = n_iw_new, statistic = 'Fermion') )
    bgf = BlockGf(name_list = self.fermionic_struct.keys(), block_list = gs, make_copies = False)
    ws_new = [w.imag for w in gs[0].mesh] 
    for key in self.local_fermionic_gfs:  
      for U in self.fermionic_struct.keys():       
        mats_freq.change_temperature_gf(vars(self)[key][U], bgf[U])
      vars(self)[key] = bgf.copy()

    self.nw = nw_new
    self.ws = copy.deepcopy(ws_new)
    self.iws = [ 1j*w for w in self.ws ]
    if finalize: 
      self.beta = beta_new
      self.n_iw = n_iw_new
      self.ntau = ntau_new


#-------------------------------k data ---------------------#
class non_local_data(basic_data):
  def __init__(self, n_iw = 100, 
                     n_k = 12, 
                     beta = 10.0, 
                     fermionic_struct = {'up': [0], 'down': [0]},
                     archive_name="dmft.out.h5"):
    basic_data.__init__(self, n_iw, beta, fermionic_struct, archive_name)
    non_local_data.promote(self, n_k)

  def promote(self, n_k):
    print "promoting non_local data. n_k: ", n_k
    self.n_k = n_k  
    self.ks = IBZ.k_grid(n_k)
    self.epsilonk = {}
    self.G0kw = {}

    self.Sigmakw = {}
    self.Gkw = {}

    self.Sigmaijw = {}
    self.Gijw = {}

    for U in self.fermionic_struct.keys():
      if mpi.is_master_node(): print "constructing fermionic_non_local_gfs, block: ", U
      self.G0kw[U] = numpy.zeros((self.nw, n_k, n_k), dtype=numpy.complex_)
      self.epsilonk[U] = numpy.zeros((n_k, n_k), dtype=numpy.complex_)
      self.Sigmakw[U] = numpy.zeros((self.nw, self.n_k, self.n_k), dtype=numpy.complex_)
      self.Gkw[U] = numpy.zeros((self.nw, n_k, n_k), dtype=numpy.complex_)
      self.Sigmaijw[U] = numpy.zeros((self.nw, self.n_k, self.n_k), dtype=numpy.complex_)
      self.Gijw[U] = numpy.zeros((self.nw, n_k, n_k), dtype=numpy.complex_)

    self.parameters.extend(['ks','n_k'])
    self.non_interacting_quantities.extend([ 'epsilonk', 'G0kw'] )

    new_fermionic = [ 'Gkw', 'Sigmakw', 'Gijw', 'Sigmaijw' ]
    self.non_local_fermionic_gfs = new_fermionic + ['G0kw'] 
    self.non_local_quantities.extend( new_fermionic )

  def change_ks(self, ks_new):
    n_k_new = len(ks_new)
    for U in self.fermionic_struct.keys():
      epsilonk_new = numpy.zeros((n_k_new,n_k_new),dtype=numpy.complex_)
      IBZ.resample(self.epsilonk[U], epsilonk_new, self.ks, ks_new)
      self.epsilonk[U] = copy.deepcopy(epsilonk_new)
      for key in self.non_local_fermionic_gfs:
        try:
          npoints = len(vars(self)[key][U][:,0,0])
          g = numpy.zeros((npoints, n_k_new, n_k_new),dtype=numpy.complex_)
          for i in range(npoints):
            IBZ.resample(vars(self)[key][U][i,:,:], g[i,:,:], self.ks, ks_new) 
          vars(self)[key][U] = copy.deepcopy(g)
        except:
          if mpi.is_master_node(): print "WARNING: could not change ks for ",key,"[",U,"]"  
    self.ks = copy.deepcopy(ks_new)
    self.n_k = n_k_new

  def change_beta(self, beta_new, n_iw_new = None, finalize = True):
    if mpi.is_master_node(): print ">>>>>>>> CHANGING BETA!!!!"
    if n_iw_new is None: n_iw_new = self.n_iw
    nw_new = n_iw_new*2

    #---lattice stugff gfs
    for key in self.non_local_fermionic_gfs:
      for U in self.fermionic_struct.keys():  
        try: 
          if mpi.is_master_node(): print "  doing: ",key,"[",U,"]"," keys: ", vars(self)[key].keys()
          if not ( U in vars(self)[key].keys() ):
            if mpi.is_master_node(): print "WARNING: skipping block",U
            continue         
        except:
          print "WARNING: could not change temperature for ",key
          continue
        g = numpy.zeros((nw_new, self.n_k, self.n_k),dtype=numpy.complex_)          
        for kxi in range(self.n_k):
          for kyi in range(self.n_k):
            mats_freq.change_temperature(vars(self)[key][U][:,kxi,kyi], g[:,kxi,kyi], self.ws, ws_new)     
        vars(self)[key][U] = copy.deepcopy(g)
    self.nw = nw_new
    self.ws = copy.deepcopy(ws_new)
    self.iws = [ 1j*w for w in self.ws ]
    if finalize: 
      self.beta = beta_new
      self.n_iw = n_iw_new
      self.ntau = ntau_new


#-------------------------------nested data ------------------------------#
class nested_data(local_data,non_local_data):
  def __init__(self, n_iw = 100, 
                     n_k = 12, 
                     beta = 10.0, 
                     impurity_struct = {'1x1': [0], '1x2': [0,1], '2x2': [0,1,2,3]},
                     fermionic_struct = {'up': [0]},
                     archive_name="dmft.out.h5"):
    basic_data.__init__(self, n_iw, beta, fermionic_struct, archive_name)
    local_data.promote(self, impurity_struct)
    non_local_data.promote(self, n_k)

  def change_beta(self, beta_new, n_iw_new = None, finalize = True):
    local_data.change_beta(self, beta_new, n_iw_new, finalize = False)
    non_local_data.change_beta(self, beta_new, n_iw_new, finalize = True)

#------------------------------------ cumul nested -------------------------#

class cumul_nested_data(nested_data):
  def __init__(self, n_iw = 100, 
                     n_k = 12, 
                     beta = 10.0, 
                     impurity_struct = {'1x1': [0], '1x2': [0,1], '2x2': [0,1,2,3]},
                     fermionic_struct = {'up': [0]},
                     archive_name="dmft.out.h5"):
    nested_data.__init__(self, n_iw, n_k, beta, impurity_struct, fermionic_struct, archive_name)
    cumul_nested_data.promote(self)

  def promote(self):
    self.gkw = {}
    self.gijw = {}
    for U in self.fermionic_struct.keys():
      if mpi.is_master_node(): print "constructing cumulants, block: ", U
      self.gkw[U] = numpy.zeros((self.nw, self.n_k, self.n_k), dtype=numpy.complex_)
      self.gijw[U] = numpy.zeros((self.nw, self.n_k, self.n_k), dtype=numpy.complex_)
    cumulants = ['gkw', 'gijw']
    self.non_local_fermionic_gfs.extend( cumulants ) 
    self.non_local_quantities.extend( cumulants )

    self.g_imp_iw = copy.deepcopy(self.Sigma_imp_iw)
    self.g_imp_iw << 0.0
    self.impurity_fermionic_gfs.append('g_imp_iw')
    self.local_quantities.append('g_imp_iw') 


#-------------------------------cellular data for CDMFT---------------------#

class cellular_data(cumul_nested_data):
  def __init__(self, n_iw = 100, 
                     n_k = 12, 
                     beta = 10.0, 
                     impurity_struct = {'1x2': [0,1]},
                     fermionic_struct = {'up': [0]},
                     archive_name="dmft.out.h5"):
    assert len(impurity_struct.keys())==1, "in cellular dmft we only solve one impurity problem"
    cumul_nested_data.__init__(self, n_iw, n_k, beta, impurity_struct, fermionic_struct, archive_name) 
    cellular_data.promote(self)

  def promote(self):
    self.imp_key = self.impurity_struct.keys()[0]
    self.Nc = len(self.impurity_struct[self.imp_key])

    self.parameters.extend(['imp_key','Nc'])

    self.epsilonijk = {}
    self.G0ijkw = {}

    self.non_interacting_quantities.extend([ 'epsilonijk', 'G0ijkw'] )

    self.Sigmaijkw = {}
    self.Gijkw = {}

    new_fermionic = [ 'Gijkw', 'Sigmaijkw' ]
    self.matrix_non_local_fermionic_gfs = new_fermionic 
    self.matrix_non_local_quantities = new_fermionic 

    #initialize arrays 
    for U in self.fermionic_struct.keys():
      if mpi.is_master_node(): print "constructing fermionic_non_local_gfs, block: ", U
      self.G0ijkw[U] = numpy.zeros((self.nw, self.Nc, self.Nc, self.n_k, self.n_k), dtype=numpy.complex_)
      self.epsilonijk[U] = numpy.zeros((self.Nc, self.Nc, self.n_k, self.n_k), dtype=numpy.complex_)
      self.Sigmaijkw[U] = numpy.zeros((self.nw, self.Nc, self.Nc, self.n_k, self.n_k), dtype=numpy.complex_)
      self.Gijkw[U] = numpy.zeros((self.nw, self.Nc, self.Nc, self.n_k, self.n_k), dtype=numpy.complex_)

    gs = []
    for C in self.impurity_struct.keys(): 
      gs.append ( GfImFreq(indices = self.impurity_struct[C], beta = self.beta, n_points =self.n_iw, statistic = 'Fermion') )     

    self.G_ij_iw = BlockGf(name_list = self.impurity_struct.keys(), block_list = gs, make_copies = True)
    self.Sigma_ij_iw = BlockGf(name_list = self.impurity_struct.keys(), block_list = gs, make_copies = True)

    self.impurity_fermionic_gfs.extend( [ 'G_ij_iw', 'Sigma_ij_iw' ] )

  def dump_all(self, archive_name=None, suffix='', parameters_and_non_interacting_without_suffix = True):
    basic_data.dump_all(self, archive_name, suffix, parameters_and_non_interacting_without_suffix)
    basic_data.dump_general(self, self.matrix_non_local_quantities, archive_name, suffix)

  def change_ks(self, ks_new):
    n_k_new = len(ks_new)

    cumul_nested_data.change_ks(self, ks_new) 

    for U in self.fermionic_struct.keys():
      for key in self.matrix_non_local_fermionic_gfs:
        epsilonijk_new = numpy.zeros((self.Nc,self.Nc, n_k_new, n_k_new),dtype=numpy.complex_)
        for i in range(self.Nc):
          for j in range(self.Nc):            
            IBZ.resample(self.epsilonijk[U][i,j,:,:], epsilonk_new[i,j,:,:], self.ks, ks_new)
        self.epsilonk[U] = copy.deepcopy(epsilonk_new)
        try:
          g = numpy.zeros((npoints, self.Nc, self.Nc, n_k_new, n_k_new),dtype=numpy.complex_)
          for wi in range(self.nw):
            for i in range(self.Nc):
              for j in range(self.Nc):            
                IBZ.resample(vars(self)[key][U][wi,i,j,:,:], g[wi,i,j,:,:], self.ks, ks_new) 
          vars(self)[key][U] = copy.deepcopy(g)
        except:
          if mpi.is_master_node(): print "WARNING: could not change ks for ",key,"[",U,"]"
    self.ks = copy.deepcopy(ks_new)
    self.n_k = n_k_new

  def change_beta(self, beta_new, n_iw_new = None, finalize = True):
    if mpi.is_master_node(): print ">>>>>>>> CHANGING BETA!!!!"
    if n_iw_new is None: n_iw_new = self.n_iw
    nw_new = n_iw_new*2

    cumul_nested_data.change_beta(self, beta_new, n_iw_new, finalize = False) 

    #---lattice stugff gfs
    for key in self.matrix_non_local_fermionic_gfs:
      for U in self.fermionic_struct.keys():  
        g = numpy.zeros((nw_new, self.Nc, self.Nc, self.n_k, self.n_k),dtype=numpy.complex_)          
        for i in range(self.Nc):
          for j in range(self.Nc):
            for kxi in range(self.n_k):
              for kyi in range(self.n_k):
                mats_freq.change_temperature(vars(self)[key][U][:,i,j,kxi,kyi], g[:,i,j,kxi,kyi], self.ws, ws_new)     
        vars(self)[key][U] = copy.deepcopy(g)
    if finalize: 
      self.beta = beta_new
      self.n_iw = n_iw_new
      self.ntau = ntau_new
      self.nw = nw_new
      self.ws = copy.deepcopy(ws_new)
      self.iws = [ 1j*w for w in self.ws ]

#================================ DCA ===========================================================#
class dca_data(local_data):
  def __init__(self, n_iw = 100,                     
                     beta = 10.0, 
                     impurity_struct = {'up': range(4)},
                     fermionic_struct = {'0': [0],'1': [0],'2': [0],'3': [0]},
                     archive_name="dmft.out.h5"):
    basic_data.__init__(self, n_iw, beta, fermionic_struct, archive_name)
    local_data.promote(self, impurity_struct)
    dca_data.promote(self)

  def promote(self):
    self.GweissK_iw = copy.deepcopy(self.G_loc_iw)
    self.GweissR_iw = copy.deepcopy(self.G_loc_iw)
    self.GK_iw = copy.deepcopy(self.G_loc_iw)
    self.GR_iw = copy.deepcopy(self.G_loc_iw)
    self.SigmaK_iw = copy.deepcopy(self.G_loc_iw)
    self.SigmaR_iw = copy.deepcopy(self.G_loc_iw)
    del self.G_loc_iw

    new_local = ['GweissK_iw','GweissR_iw','GK_iw','GR_iw','SigmaK_iw','SigmaR_iw']

    self.local_fermionic_gfs.remove('G_loc_iw')
    self.local_fermionic_gfs.extend(new_local)
    self.local_quantities.remove('G_loc_iw')
    self.local_quantities.extend(new_local) 

#================================ DCA+ ===========================================================#
class dca_plus_data(dca_data, non_local_data):
  def __init__(self, n_k = 128,
                     n_iw = 100,                     
                     beta = 10.0, 
                     impurity_struct = {'up': range(4)},
                     fermionic_struct = {'0': [0],'1': [0],'2': [0],'3': [0]},
                     archive_name="dmft.out.h5"):
    dca_data.__init__(self, n_iw, beta, impurity_struct, fermionic_struct, archive_name)
    non_local_data.__init__(self, n_iw, n_k, beta, {'up': [0]}, archive_name)
    #self.non_local_struct = 
    #self.fermionic_struct = self.non_local_struct #just temporarily
    #non_local_data.promote(self, n_k)
    #self.fermionic_struct = fermionic_struct #put it back...
    dca_plus_data.promote(self)
    print numpy.shape(self.Sigmakw['up'])
    self.fermionic_struct = fermionic_struct #put it back...

  def promote(self):
    self.mus = {'up': 0}
    self.ns = {'up': 0}

    self.Sigmaimpkw = copy.deepcopy(self.Sigmakw)
    self.XiR_iw = copy.deepcopy(self.SigmaR_iw)
    self.XiK_iw = copy.deepcopy(self.SigmaK_iw)
    self.Xikw = copy.deepcopy(self.Sigmakw)

    gs = []   
    gs.append ( GfImFreq(indices = {'up': [0]}, beta = self.beta, n_points =self.n_iw, statistic = 'Fermion') )     
    self.G_loc_iw = BlockGf(name_list = ['up'], block_list = gs, make_copies = True)

    new_local = ['XiK_iw','XiR_iw']

    self.local_fermionic_gfs.extend(new_local)    
    self.local_quantities.extend(new_local) 

    new_non_local = [ 'Xikw', 'Sigmaimpkw']
    self.non_local_fermionic_gfs = new_non_local
    self.non_local_quantities.extend( new_non_local )



