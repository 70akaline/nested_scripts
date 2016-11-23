from optimized_latt_ft import spatial_FT
from optimized_latt_ft import temporal_FT
from optimized_latt_ft import spatial_inverse_FT
from optimized_latt_ft import temporal_inverse_FT
from tail_fitters import fit_fermionic_gf_tail


from pytriqs.operators import *
from pytriqs.archive import *
from pytriqs.gf.local import *
from pytriqs.arrays import BlockMatrix, BlockMatrixComplex
import pytriqs.utility.mpi as mpi

import numpy
from numpy.linalg import inv

from math import cos
import cmath

################################ general initializers ##########################################

def sgn(x):
  if x>=0: 
    return 1
  else:
    return -1

#---------- dispersions and bare interactions ----------------------------------------#
def Jq_square(qx, qy, J):
  return 2.0*J*(cos(qx)+cos(qy))

def epsilonk_square(kx,ky,t):
  return Jq_square(kx, ky, t)

def epsilonk_Aoki(kx,ky, t, tp, ts):
  return 2.0*t*(cos(kx) + cos(ky)) + 4.0*tp*cos(kx)*cos(ky) + 2.0*ts*( cos(2.0*kx) + cos(2.0*ky) )

def Jq_square_AFM(qx, qy, J): #should not make any difference when summed over the brillouin zone
  return J*( 1.0 + cmath.exp(1j*qx) + cmath.exp(1j*qy) + cmath.exp(1j*(qx+qy)) )

def X_dwave(kx, ky, X):
  return X*(cos(kx)-cos(ky))

###############################################################################################################################################
#---------------------------------------------------------------------------------------------------------------------------------------------#
def total_inverse_FT(Qkw,
                     beta, ntau, n_iw, n_k,
                     statistic='Fermion', use_IBZ_symmetry = True, fit_tail = False,      
                     N_cores=1):
   Qktau = temporal_inverse_FT(Qkw, beta, ntau, n_iw, n_k, statistic, use_IBZ_symmetry, fit_tail, N_cores=N_cores)
   return spatial_inverse_FT(Qktau, N_cores=N_cores)



#-----------------------------------------------------------------------------------------------------------------------------------------------#
def blockwise_get_Gijtau_from_Gkw(Gkw, 
                                  beta, ntau, n_iw, n_k,
                                  N_cores=1): #only use for Gtilde... for the full G one must fit tail. TODO
   return total_inverse_FT(Gkw, beta, ntau, n_iw, n_k, statistic='Fermion', use_IBZ_symmetry = True, fit_tail = False, N_cores=N_cores)

def blockwise_get_Gijw_from_Gkw(Gkw, N_cores=1): #only use for Gtilde... for the full G one must fit tail. TODO
   return spatial_inverse_FT(Gkw, N_cores=N_cores)

def full_fill_Gijw_from_Gkw(Gijw, Gkw, N_cores=1): #only use for Gtilde... for the full G one must fit tail. TODO
  if mpi.is_master_node(): print "full_fill_Gijw_from_Gkw"
  for U in Gkw.keys():
    Gijw[U][:,:,:] = blockwise_get_Gijw_from_Gkw(Gkw[U], N_cores)

def full_fill_Sigmakw_from_Sigmaijw(Sigmakw, Sigmaijw, N_cores=1): #only use for Gtilde... for the full G one must fit tail. TODO
  if mpi.is_master_node(): print "full_fill_Sigmakw_from_Sigmaijw"
  for U in Sigmaijw.keys():
    Sigmakw[U][:,:,:] = spatial_FT(Sigmaijw[U], N_cores=N_cores)


#-----------------------------------------------------------------------------------------------------------------------------------------------#
def blockwise_get_G_loc_tau_from_G_loc_iw(G_loc_iw,                                          
                                          fit_tail_starting_iw = 14.0, ntau = None):
  if mpi.is_master_node(): print "blockwise_get_G_loc_tau_from_G_loc_iw"

  fit_fermionic_gf_tail(G_loc_iw, starting_iw=fit_tail_starting_iw)       
  if ntau is None:
    ntau = 3*len(G_loc_iw.data[:,0,0])

  G_loc_tau = GfImTime(indices = [0], beta = G_loc_iw.beta, n_points = ntau, statistic = 'Fermion')
  G_loc_tau << InverseFourier(G_loc_iw)  
  return G_loc_tau

#-----------------------------------------------------------------------------------------------------------------------------------------------#
def blockwise_get_n_from_G_loc_iw(G_loc_iw,
                                  fit_tail_starting_iw = 14.0, ntau = None):  
    Gw = G_loc_iw.copy()
    Gtau = blockwise_get_G_loc_tau_from_G_loc_iw(Gw, fit_tail_starting_iw, ntau)
    return -Gtau.data[-1,0,0]

#-----------------------------------------------------------------------------------------------------------------------------------------------#
def full_fill_ns_from_G_loc_iw(ns, G_loc_iw,
                              fit_tail_starting_iw = 14.0, ntau = None):
  for U in [name for name, g in G_loc_iw]:
    Gw = G_loc_iw.copy()
    ns[U] = blockwise_get_n_from_G_loc_iw(G_loc_iw[U], fit_tail_starting_iw, ntau)


#-----------------------------------------------------------------------------------------------------------------------------------------------#
def full_fill_local_from_latt(Q_loc_iw, Qkw): #func here has no purpose
  for key in Qkw.keys():
    nk = len(Qkw[key][0,0,:])
    Q_loc_iw[key].data[:,0,0] = numpy.sum(Qkw[key],axis=(1,2))/nk**2

#-----------------------------------------------------------------------------------------------------------------------------------------------#
def blockwise_add_local_to_lattice(Q_loc_iw, Qkw):
  numpy.transpose(Qkw)[:] += Q_loc_iw.data[:,0,0]

#-----------------------------------------------------------------------------------------------------------------------------------------------#
def blockwise_subtract_local_from_lattice(Q_loc_iw, Qkw):
  numpy.transpose(Qkw)[:] -= Q_loc_iw.data[:,0,0]

#-----------------------------------------------------------------------------------------------------------------------------------------------#
def add_local_to_lattice(Q_loc_iw, Qkw):
  for key in Qkw.keys():
    numpy.transpose(Qkw[key])[:] += Q_loc_iw[key].data[:,0,0]

#-----------------------------------------------------------------------------------------------------------------------------------------------#
def subtract_local_from_lattice(Q_loc_iw, Qkw):
  for key in Qkw.keys():
    numpy.transpose(Qkw[key])[:] -= Q_loc_iw[key].data[:,0,0]

#-----------------------------------------------------------------------------------------------------------------------------------------------#
#                                        lattice_dyson
#-----------------------------------------------------------------------------------------------------------------------------------------------#

def blockwise_get_Gkw_from_iws_mu_epsiolonk_and_Sigmakw(iws,mu,epsilonk,Sigmakw):
  if mpi.is_master_node(): print "blockwise_get_Gkw_from_iws_mu_epsiolonk_and_Sigmakw...",
  Gkw = -Sigmakw[:,:,:]
  numpy.transpose(Gkw)[:] += iws[:]
  Gkw[:,:,:] += mu
  Gkw[:,:,:] -= epsilonk[:,:]
  Gkw **= -1.0
  if mpi.is_master_node(): print "done!"
  return Gkw 

#-----------------------------------------------------------------------------------------------------------------------------------------------#
def full_fill_Gkw_from_iws_mus_epsiolonk_and_Sigmakw(Gkw, iws, mus, epsilonk, Sigmakw):
  if mpi.is_master_node(): print "full_fill_Gkw_from_iws_mus_epsiolonk_and_Sigmakw"
  for U in Gkw.keys():
    Gkw[U] = blockwise_get_Gkw_from_iws_mu_epsiolonk_and_Sigmakw(iws,mus[U],epsilonk[U],Sigmakw[U])
  if mpi.is_master_node(): print "done!"

def full_fill_Gkw_from_epsiolonk_and_gkw(Gkw, epsilonk, gkw):
  if mpi.is_master_node(): print "full_fill_Gkw_from_epsiolonk_and_gkw"
  for U in Gkw.keys():
    invG = gkw[U]**(-1.0)
    invG -= epsilonk[U]
    Gkw[U][:,:,:] = invG**(-1.0)


#-----------------------------------------------------------------------------------------------------------------------------------------------#
#                                        weiss_field
#-----------------------------------------------------------------------------------------------------------------------------------------------#


def blockwise_fill_Gweiss_iw_from_Gijw_and_Sigma_imp_iw(Gweiss_iw,Gijw,Sigma_imp_iw, mapping = lambda i,j: [0,0]):
  if mpi.is_master_node(): print "blockwise_fill_Gweiss_iw_from_Gijw_and_Sigma_imp_iw"
  n_sites = len(Gweiss_iw.data[0,0,:])
  nw = len(Gweiss_iw.data[:,0,0])
  Gtemp = numpy.zeros((nw,n_sites,n_sites), dtype=numpy.complex_)
  for i in range(n_sites):
    for j in range(n_sites):
       Gtemp[:,i,j] = Gijw[:,mapping(i,j)[0], mapping(i,j)[1]] 

  invGtemp = Gtemp[:,:,:]
  for wi in range(nw):
    invGtemp[wi,:,:] = inv(Gtemp[wi,:,:])
    Gweiss_iw.data[wi,:,:] = inv(invGtemp[wi,:,:] + Sigma_imp_iw.data[wi,:,:])

  fit_fermionic_gf_tail(Gweiss_iw)

def full_fill_Gweiss_iw_from_Gijw_and_Sigma_imp_iw(Gweiss_iw,Gijw,Sigma_imp_iw, mapping = lambda C,i,j: [0,0]):      
  if mpi.is_master_node(): print "full_fill_Gweiss_iw_from_Gijw_and_Sigma_imp_iw"
  assert 'up' in Gijw.keys(), "this assumes there is only one block in lattice functions. should be generalized for magnetized calculations" 
  block_names = [name for name,g in Gweiss_iw]   
  for C in block_names:
    blockwise_fill_Gweiss_iw_from_Gijw_and_Sigma_imp_iw(Gweiss_iw[C],Gijw['up'],Sigma_imp_iw[C], mapping= lambda i,j: mapping(C,i,j) ) 
  if mpi.is_master_node(): print "done!"

#-----------------------------------------------------------------------------------------------------------------------------------------------#
#                                        nested specific
#-----------------------------------------------------------------------------------------------------------------------------------------------#
  
def full_fill_Sigmaijw_from_Sigma_imp_iw(Sigmaijw, Sigma_imp_iw, mapping):
  if mpi.is_master_node(): print "full_fill_Sigmaijw_from_Sigma_imp_iw"
  #impurity_blocks = [name for name, g in Sigma_imp_iw] don't need this
  for U in Sigmaijw.keys():
    nk = len(Sigmaijw[U][0,0,:])
    Sigmaijw[U][:,:,:] = 0.0
    for x in range(nk):
      for y in range(nk):
          mps = mapping(x,y)
          for mp in mps:
            C = mp[0]
            pref = mp[1]
            i = mp[2]
            j = mp[3]
            Sigmaijw[U][:,x,y] += pref * Sigma_imp_iw[C].data[:,i,j]


#-----------------------------------------------------------------------------------------------------------------------------------------------#
#                                        cumulant
#-----------------------------------------------------------------------------------------------------------------------------------------------#

def full_fill_g_imp_iw_from_Sigma_imp_iw(g_imp_iw, mu, Sigma_imp_iw):
  if mpi.is_master_node(): print "full_fill_g_imp_iw_from_Sigma_imp_iw"
  impurity_blocks = [name for name, g in Sigma_imp_iw] 
  iws = numpy.array([ w for w in Sigma_imp_iw[impurity_blocks[0]].mesh ])
  nw = len(iws)
  for C in impurity_blocks:
    Nc = len(g_imp_iw[C].data[0,0,:])
    for l in range(nw):
      g_imp_iw[C].data[l,:,:] = inv( (iws[l]+mu)*numpy.identity(Nc) - Sigma_imp_iw[C].data[l,:,:] )

def full_fill_Sigmakw_from_gkw(Sigmakw, ws, mu, gkw):
  if mpi.is_master_node(): print "full_fill_Sigmakw_from_gkw"
  for U in Sigmakw.keys():
     invSigma = -gkw[U]**(-1.0)
     numpy.transpose(invSigma)[:] += 1j*numpy.array(ws[:])+mu
     Sigmakw[U][:,:,:] = invSigma


