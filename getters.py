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

from copy import deepcopy

from math import cos, pi
import cmath
from cmath import exp


################################ general initializers ##########################################

def sgn(x):
  if x>=0: 
    return 1
  else:
    return -1


def initCubicTBH(Nx, Ny, Nz, eps, t, cyclic=True):
  H = [[0 for j in range(Nx*Ny*Nz)] for i in range(Nx*Ny*Nz)]  
  for i in range(Nx*Ny*Nz):
    H[i][i]=eps    
  for i in range(Nx):
    for j in range(Ny):
      for k in range(Nz): 
        if Nx>1:
          if i+1==Nx:
            if cyclic: H [ k*Nx*Ny+j*Nx+i ]  [ k*Nx*Ny + j*Nx ] = t
          else:
            H [ k*Nx*Ny+j*Nx+i ]  [ k*Nx*Ny + j*Nx + i+1 ] = t
        
          if i==0:
            if cyclic: H [ k*Nx*Ny+j*Nx+i ]  [ k*Nx*Ny + j*Nx + Nx-1] = t
          else:
            H [ k*Nx*Ny+j*Nx+i ]  [ k*Nx*Ny + j*Nx + i-1] = t  
            
        if Ny>1:
          if j+1==Ny:
            if cyclic: H [ k*Nx*Ny+j*Nx+i ]  [ k*Nx*Ny + i ] = t
          else:  
            H [ k*Nx*Ny+j*Nx+i ]  [ k*Nx*Ny + (j+1)*Nx + i ] = t
        
          if j==0:
            if cyclic: H [ k*Nx*Ny+j*Nx+i ]  [ k*Nx*Ny + (Ny-1)*Nx + i ] = t
          else:
            H [ k*Nx*Ny+j*Nx+i ]  [ k*Nx*Ny + (j-1)*Nx + i ] = t

        if Nz>1:
          if (k+1==Nz): 
            if cyclic: H [ k*Nx*Ny+j*Nx+i ]  [ j*Nx + i ] = t
          else:
            H [ k*Nx*Ny+j*Nx+i ]  [ (k+1)*Nx*Ny + j*Nx + i ] = t 
            
          if k==0:
            if cyclic: H [ k*Nx*Ny+j*Nx+i ]  [ (Nz-1)*Nx*Ny + j*Nx + i ] = t
          else:
            H [ k*Nx*Ny+j*Nx+i ]  [ (k-1)*Nx*Ny + j*Nx + i ] = t
    
  return H 


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
def full_decompose(Q):
    nw,nk,nk = numpy.shape(Q)
    ReQ, ImQ = numpy.real(Q), numpy.imag(Q)
    ReQloc,ImQloc = numpy.sum(ReQ,axis=(1,2))/nk**2,numpy.sum(ImQ,axis=(1,2))/nk**2
    ReQtilde = numpy.transpose(numpy.transpose(ReQ)[:,:,:]-ReQloc)
    ImQtilde = numpy.transpose(numpy.transpose(ImQ)[:,:,:]-ImQloc)
    return ReQ, ImQ, ReQloc, ImQloc, ReQtilde, ImQtilde

def decompose(Q):
    nw,nk,nk = numpy.shape(Q)
    Qloc = numpy.sum(ReQ,axis=(1,2))
    Qtilde = numpy.transpose(numpy.transpose(Q)[:,:,:]-Qloc)
    return Qloc, Qtilde

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

#-------------------------------------------------------------------------------------------------------------------------------------------------#
def get_Delta_from_Gweiss_and_H0(Gweiss, H0, mu):
    Delta = Gweiss.copy()    
    Delta << 0.0
    iws = numpy.array([iw for iw in Gweiss.mesh])
    nw,Nsites,dummy = numpy.shape(Gweiss.data)
    assert numpy.shape(H0) == (Nsites,Nsites)
    for i in range(Nsites):
        Delta.data[:,i,i] = iws[:]+mu
    Delta.data[:,:,:] -= H0
    Delta -= inverse(Gweiss)
    return Delta

def get_Gweiss_from_Delta_and_H0(Delta, H0, mu):
    Gweiss = Delta.copy()    
    Gweiss << 0.0
    iws = numpy.array([iw for iw in Delta.mesh])
    nw,Nsites,dummy = numpy.shape(Delta.data)
    assert numpy.shape(H0) == (Nsites,Nsites)
    for i in range(Nsites):
        Gweiss.data[:,i,i] = iws[:]+mu
    Gweiss.data[:,:,:] -= H0
    Gweiss -= Delta
    #fit_fermionic_gf_tail(Gweiss)
    Gweiss << inverse(Gweiss)
    return Gweiss

#-----------------------------------------------------------------------------------------------------------------------------------------------#
def blockwise_get_G_loc_tau_from_G_loc_iw(G_loc_iw,                                          
                                          fit_tail_starting_iw = 14.0, ntau = None):
  if mpi.is_master_node(): print "blockwise_get_G_loc_tau_from_G_loc_iw"

  fit_fermionic_gf_tail(G_loc_iw, starting_iw=fit_tail_starting_iw)       
  if ntau is None:
    ntau = 3*len(G_loc_iw.data[:,0,0])

  Nc = len(G_loc_iw.data[0,0,:])
  G_loc_tau = GfImTime(indices = range(Nc), beta = G_loc_iw.beta, n_points = ntau, statistic = 'Fermion')
  G_loc_tau << InverseFourier(G_loc_iw)  
  return G_loc_tau

#-----------------------------------------------------------------------------------------------------------------------------------------------#
def blockwise_get_n_from_G_loc_iw(G_loc_iw,
                                  fit_tail_starting_iw = 14.0, ntau = None,
                                  site_index = 0):  
    Gw = G_loc_iw.copy()
    Gtau = blockwise_get_G_loc_tau_from_G_loc_iw(Gw, fit_tail_starting_iw, ntau)
    return -Gtau.data[-1,site_index,site_index]

#-----------------------------------------------------------------------------------------------------------------------------------------------#
def full_fill_ns_from_G_loc_iw(ns, G_loc_iw,
                              fit_tail_starting_iw = 14.0, ntau = None):
  if mpi.is_master_node(): print "full_fill_ns_from_G_loc_iw"
  for U in [name for name, g in G_loc_iw]:
    Gw = G_loc_iw.copy()
    if U not in ns.keys():       
      key = 'up' #careful with this!!!!
    else:
      key = U
    ns[key] = blockwise_get_n_from_G_loc_iw(G_loc_iw[U], fit_tail_starting_iw, ntau)
    #print "ns: ", ns  


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

def blockwise_flexible_Gweiss_iw_from_Gweiss_iw_Gijw_and_G_imp_iw(Gweiss_iw,Gijw,G_imp_iw,
                                                                  mapping, sign, sign_up_to ):
  if mpi.is_master_node(): print "blockwise_flexible_Gweiss_iw_from_Gweiss_iw_Gijw_and_G_imp_iw"
  n_sites = len(Gweiss_iw.data[0,0,:])
  nw = len(Gweiss_iw.data[:,0,0])
  Gtemp = numpy.zeros((nw,n_sites,n_sites), dtype=numpy.complex_)
  for i in range(n_sites):
    for j in range(n_sites):
       Gtemp[:,i,j] = Gijw[:,mapping(i,j)[0], mapping(i,j)[1]] 

  invGtemp = Gtemp[:,:,:]
  for wi in range(nw):
    n_mats = wi-nw/2    
    if sign_up_to == -1: sign_up_to = nw
    if ( n_mats>=0 and n_mats<sign_up_to) \
        or n_mats<0 and abs(n_mats)-1<sign_up_to:
        sgn = sign
    else: sgn= -sign    
    invGtemp[wi,:,:] = inv(Gtemp[wi,:,:])
    Gweiss_iw.data[wi,:,:] = inv( inv(Gweiss_iw.data[wi,:,:]) + sgn*(invGtemp[wi,:,:] - inv(G_imp_iw.data[wi,:,:])) )

  fit_fermionic_gf_tail(Gweiss_iw)

def flexible_Gweiss_iw_from_Gweiss_iw_Gijw_and_G_imp_iw(Gweiss_iw, Gijw, G_imp_iw, mapping = lambda C,i,j: [0,0], 
                                                        sign=1, sign_up_to=-1):      
  if mpi.is_master_node(): print "flexible_Gweiss_iw_from_Gweiss_iw_Gijw_and_G_imp_iw"
  assert 'up' in Gijw.keys(), "this assumes there is only one block in lattice functions. should be generalized for magnetized calculations" 
  block_names = [name for name,g in Gweiss_iw]   
  for C in block_names:
    blockwise_flexible_Gweiss_iw_from_Gweiss_iw_Gijw_and_G_imp_iw(
        Gweiss_iw[C],Gijw['up'],G_imp_iw[C], mapping= lambda i,j: mapping(C,i,j), sign=sign, sign_up_to=sign_up_to ) 
  if mpi.is_master_node(): print "done!"



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


#-----------------------------------------------------------------------------------------------------------------------------------------------#
#                                        dca specific
#-----------------------------------------------------------------------------------------------------------------------------------------------#

def full_fill_SigmaR_iw_from_Sigma_imp_iw(SigmaR_iw, Sigma_imp_iw, i_to_ij):
  if mpi.is_master_node(): print "full_fill_SigmaR_iw_from_Sigma_imp_iw"
  block = [name for name,s in Sigma_imp_iw]
  assert len(block)==1, "in dca there is only one impurity problem"
  for R, sig in SigmaR_iw:
    i,j = i_to_ij(int(R))
    sig << Sigma_imp_iw[block[0]][i,j]

def full_fill_SigmaK_iw_from_SigmaR_iw(SigmaK_iw, SigmaR_iw, P, Pinv):
  if mpi.is_master_node(): print "full_fill_SigmaK_iw_from_Sigma_imp_iw"
  SigmaK_iw.zero()
  dim = len([name for name,g in SigmaK_iw])  
  for i in range(dim):
    for l in range(dim):
      SigmaK_iw["%02d"%i] += dim * Pinv[i,0] * SigmaR_iw["%02d"%l] * P[l,i]
  
def full_fill_GweissK_iw_from_Dyson(GweissK_iw, GK_iw, SigmaK_iw):
  for K,g0 in GweissK_iw:
    #print K
    #nw = len(GK_iw[K].data[:,0,0])
    #print "GK_iw[K].data[nw/2,0,0]:",GK_iw[K].data[nw/2,0,0]
    #print "SigmaK_iw[K].data[nw/2,0,0]:",SigmaK_iw[K].data[nw/2,0,0]
    #try:  
    g0 << inverse(inverse(GK_iw[K]) + SigmaK_iw[K])
    #except:
    #  pass

def full_fill_GweissR_iw_from_GweissK_iw(GweissR_iw, GweissK_iw, P, Pinv, l_list = []):
  if mpi.is_master_node(): print "full_fill_GweissR_iw_from_GweissK_iw"
  GweissR_iw.zero()
  dim = len([name for name,g in GweissR_iw])  
  for l in (range(dim) if l_list==[] else l_list):
    for i in range(dim):
      GweissR_iw["%02d"%l] += P[0,i] * GweissK_iw["%02d"%i] * Pinv[i,l]

def full_fill_Gweiss_iw_from_GweissR_iw(Gweiss_iw, GweissR_iw, ij_to_0i):
  if mpi.is_master_node(): print "full_fill_Gweiss_iw_from_GweissR_iw"
  dim = len([name for name,g in GweissR_iw])  
  block = [name for name,s in Gweiss_iw]
  assert len(block)==1, "in dca there is only one impurity problem"
  for i in range(dim):
    for j in range(dim):
        Gweiss_iw[block[0]][i,j] << GweissR_iw["%02d"%ij_to_0i[i,j]]

def full_fill_GK_iw(GK_iw, SigmaK_iw, mu, dca_patches):
  if mpi.is_master_node(): print "full_fill_GK_iw...",
  for p in dca_patches:
    GK_iw[p.name] = p.ht(Sigma = SigmaK_iw[p.name], mu = mu)
  if mpi.is_master_node(): print "DONE"


def get_Sigmakw_from_SigmaK_iw(SigmaK_iw, dca_scheme, nk=64):
    ks = numpy.linspace(0,2*pi,nk, endpoint=False)
    nw = len(SigmaK_iw['00'].data[:,0,0])
    Sigmakw = numpy.zeros((nw,nk,nk),dtype=numpy.complex_)            
    full_k_points = list(dca_scheme.k_points)
    full_k_points += list(dca_scheme.k_points+[0,2*pi])
    full_k_points += list(dca_scheme.k_points+[2*pi,0])
    full_k_points += list(dca_scheme.k_points+[2*pi,2*pi])
                    
    for kxi in range(nk):
        for kyi in range(nk):
            min_dist,min_l = 2*pi, 0
            for l in range(len(full_k_points)):
                dist = numpy.linalg.norm(numpy.array([ks[kxi],ks[kyi]]) - numpy.array(full_k_points[l]))
                if dist<min_dist: min_dist, min_l = dist, l
            while min_l>=dca_scheme.dim: min_l-=dca_scheme.dim
            Sigmakw[:,kxi,kyi] = SigmaK_iw['%02d'%min_l].data[:,0,0]   

    return Sigmakw, ks  

#-----------------------------------------------------------------------------------------------------------------------------------------------#
#                                        dca_plus specific
#-----------------------------------------------------------------------------------------------------------------------------------------------#

def fill_XiK_from_SigmaK(XiK_iw,SigmaK_iw, alpha=1):
    assert alpha > 0.0, 'alpha must be positive'
    XiK_iw << inverse(SigmaK_iw - 1j*alpha)

def blockwise_Sigmak_from_Xik(Sigmakw, Xikw, alpha=1):
    assert alpha > 0.0, 'alpha must be positive'
    Sigmakw[:,:,:] = Xikw[:,:,:]**(-1) + 1j*alpha
    
def IBZ_convolution(Q, nK):
    nw = len(Q[:,0,0])
    nk = len(Q[0,:,0])    
    assert len(Q[0,:,0])==len(Q[0,0,:]), "must be nkx=nky"
    assert nk % nK ==0, "must be divisible with the coarse graining"
    if (nk/nK) % 2 == 0:
        n_k_in_K = nk/nK        
        D = n_k_in_K/2    
        #print "n_k_in_K: ", n_k_in_K
        #print "D: ", D        
        Qp = deepcopy(Q)    
        Qp[:,:,:] = 0.0
        Qlarge = numpy.zeros((nw,3*nk,3*nk))
        for i in range(3):
            for j in range(3):
                Qlarge[:,i*nk:(i+1)*nk,j*nk:(j+1)*nk] = Q[:,:,:]
        for kxi in range(nk):
            for kyi in range(nk):                            
                lxi =kxi+nk
                lyi =kyi+nk
                for shftx in [0,1]:
                    for shfty in [0,1]:
                        Qp[:,kxi,kyi] += numpy.sum(Qlarge[:,lxi-D+shftx:lxi+D+shftx,lyi-D+shfty:lyi+D+shfty],axis=(1,2))
                Qp[:,kxi,kyi] *= 0.25   

        return Qp/n_k_in_K**2
    else:
        n_k_in_K = nk/nK        
        D = (n_k_in_K-1)/2    
        #print "n_k_in_K: ", n_k_in_K
        #print "D: ", D        
        Qp = deepcopy(Q)    
        Qp[:,:,:] = 0.0
        Qlarge = numpy.zeros((nw,3*nk,3*nk))
        for i in range(3):
            for j in range(3):
                Qlarge[:,i*nk:(i+1)*nk,j*nk:(j+1)*nk] = Q[:,:,:]
        for kxi in range(nk):
            for kyi in range(nk):                            
                lxi =kxi+nk
                lyi =kyi+nk
                Qp[:,kxi,kyi] += numpy.sum(Qlarge[:,lxi-D:lxi+D+1,lyi-D:lyi+D+1],axis=(1,2))
        return Qp/n_k_in_K**2
   
from data_containers import IBZ
def Richardson_Lucy(Qtarget, Q, nK, n_iterations = 5, accr=1e-5):
    print "----------- Richardson Lucy --------------"
    nk = len(Q[0,:,0])
    ReQ, ImQ, ReQloc, ImQloc, ReQtilde, ImQtilde = full_decompose(Q)
    ReQt, ImQt, ReQtloc, ImQtloc, ReQttilde, ImQttilde = full_decompose(Qtarget)
        
    for q, qt in [(ReQ, ReQt), (ImQ,ImQt)]:        
        print "----------- Doing %s part"%("real" if q is ReQ else "imag")  
        for it in range(n_iterations):
            print "------ it: ",it
            Qp = IBZ_convolution(q, nK)
            if numpy.amin(numpy.abs(Qp))<1e-12:
              print "encountering zero!"
              break
            Qw = qt/Qp
            Qpp = IBZ_convolution(Qw, nK)
            q *= Qpp
            IBZ.copy_by_symmetry(numpy.transpose(q)[:,:,:])
            diff = numpy.linalg.norm(IBZ_convolution(q, nK)-qt)/numpy.linalg.norm(qt)            
            print "diff: ", diff
            if diff < accr: 
              print "----converged!"
              break             
  
    Q[:,:,:] = ReQ+1j*ImQ
    Qloc = numpy.sum(Q,axis=(1,2))/nk**2
    numpy.transpose(Q)[:,:,:] += ReQtloc + 1j*ImQtloc - Qloc #enforce the correct local part


#-----------------------------------------------------------------------------------------------------------------------------------------------#
#                                        cellular specific
#-----------------------------------------------------------------------------------------------------------------------------------------------#

def blockwise_get_Gijkw(iws, mu, epsilonijk, Sigmaijkw):
  if mpi.is_master_node(): print "blockwise_get_Gijkw"
  Nc = len(epsilonijk[:,0,0,0])
  nk = len(epsilonijk[0,0,:,0])
  nw = len(iws)
  
  Gijkw = -Sigmaijkw
  iom = numpy.zeros((Nc,Nc,nw), dtype=numpy.complex_)
  for i in range(Nc): iom[i,i,:] = numpy.array(iws[:])+mu
  numpy.transpose(Gijkw)[:,:,:] += iom[:,:,:]
  Gijkw -= epsilonijk
  for wi in range(nw):
    for kxi in range(nk):
      for kyi in range(nk):
        Gijkw[wi,:,:,kxi,kyi] = inv(Gijkw[wi,:,:,kxi,kyi])
  return Gijkw

def full_fill_Sigmaijkw(Sigmaijkw, Sigma_imp_iw):
    if mpi.is_master_node(): print "full_fill_Sigmaijkw"
    assert len(Sigmaijkw.keys())==1, "must be only one block in Sigmaijkw"
    impkeys = [name for name,g in Sigma_imp_iw]    
    assert len(impkeys)==1, "must be only one block in Sigma_imp_iw"      
    numpy.transpose(Sigmaijkw[Sigmaijkw.keys()[0]])[:,:,:] = numpy.transpose(Sigma_imp_iw[impkeys[0]].data)[:,:,:]

def full_fill_Sigmaijkw_periodized(Sigmaijkw, Sigma_imp_iw, ks):
    if mpi.is_master_node(): print "full_fill_Sigmaijkw_periodized"
    assert len(Sigmaijkw.keys())==1, "must be only one block in Sigmaijkw"
    impkeys = [name for name,g in Sigma_imp_iw]    
    assert len(impkeys)==1, "must be only one block in Sigma_imp_iw"      
    impkey = impkeys[0]
    numpy.transpose(Sigmaijkw[Sigmaijkw.keys()[0]])[:,:,:] = numpy.transpose(Sigma_imp_iw[impkey].data)[:,:,:]    
       
    s = Sigma_imp_iw[impkey].data[:,0,1]
    sp = Sigma_imp_iw[impkey].data[:,0,3]    
    z = deepcopy(s)
    z[:] = 0    
    for kxi, kx in enumerate(ks):
      for kyi, ky in enumerate(ks):
        skx = s*exp(1j*kx)
        sky = s*exp(1j*ky)
        cskx = s*exp(-1j*kx)
        csky = s*exp(-1j*ky)

        B =       [[z,    skx,  sky,  z  ],
                   [cskx, z,    z,    sky],
                   [csky, z,    z,    skx],
                   [z,    csky, cskx, z  ]]
        spkAD = sp*numpy.conj(  exp(-1j*kx)+exp(-1j*ky) + exp(-1j*(kx+ky))  )
        spkBC = sp*numpy.conj(  exp(1j*kx)+exp(-1j*ky) + exp(1j*(kx-ky))  )
        spkDA = sp*(  exp(-1j*kx)+exp(-1j*ky) + exp(-1j*(kx+ky))  )
        spkCB = sp*(  exp(1j*kx)+exp(-1j*ky) + exp(1j*(kx-ky))  )

        C =       [[z,     z,     z,     spkAD ],
                   [z,     z,     spkBC, z     ],
                   [z,     spkCB, z,     z     ],
                   [spkDA, z,     z,     z     ]]

        BC = numpy.array(B)+numpy.array(C)
        numpy.transpose(Sigmaijkw[Sigmaijkw.keys()[0]])[kyi,kxi,:,:,:] += BC
       

def full_fill_Gijkw(Gijkw, iws, mus, epsilonijk, Sigmaijkw):
  for U in Gijkw.keys():
    Gijkw[U][:,:,:,:,:] = blockwise_get_Gijkw(iws, mus[U], epsilonijk[U], Sigmaijkw[U])

def full_fill_G_ij_iw(G_ij_iw, Gijkw):
  if mpi.is_master_node(): print "full_fill_G_ij_iw"
  nk = len(Gijkw['up'][0,0,0,0,:])
  for name,g in G_ij_iw:
    g.data[:,:,:] = numpy.sum(Gijkw['up'], axis=(3,4))/nk**2.0

def full_fill_Gweiss_iw(Gweiss_iw, G_ij_iw, Sigma_imp_iw):
  if mpi.is_master_node(): print "full_fill_Gweiss_iw"
  for name,g in Gweiss_iw:
    nw = len(g.data[:,0,0])
    for wi in range(nw):
      g.data[wi,:,:] = inv( inv(G_ij_iw[name].data[wi,:,:]) + Sigma_imp_iw[name].data[wi,:,:] )
    fit_fermionic_gf_tail(g)

def cellular_latt_to_imp_mapping(x,y,nk,Lx,Ly):  
  X,Y = abs(x),abs(y)
  if X>nk/2+1: X = nk - X
  if Y>nk/2+1: Y = nk - Y
  if Y>X: X,Y = Y,X
  if (X>=Lx) or (Y>=Ly): return None   
  i = 0
  j = Lx*Y + X
  return [i,j]

def periodize_cumul(Gkw, Sigmakw, gkw, gijw, g_imp_iw, iws, mus, epsilonk, Sigma_imp_iw, Lx, Ly):
  if mpi.is_master_node(): print "periodize_cumul"
  full_fill_g_imp_iw_from_Sigma_imp_iw(g_imp_iw, mus['up'], Sigma_imp_iw)
  imp_key = [name for name,g in g_imp_iw ][0] 
  Nc = len(g_imp_iw[imp_key].data[0,0,:])
  nk = len(epsilonk['up'][0,:])
  for U in gijw.keys():
    for x in range(nk):
      for y in range(nk):
        ij = cellular_latt_to_imp_mapping(x,y,nk,Lx,Ly)        
        if ij is None:
          gijw[U][:,x,y] = 0.0
        else:
          i = ij[0]
          j = ij[1]
          print "x,y,i,j: ", x,y,i,j
          gijw[U][:,x,y] = g_imp_iw[imp_key].data[:,i,j]
    gkw[U][:,:,:] = spatial_FT(gijw[U], N_cores=1)
  full_fill_Sigmakw_from_gkw(Sigmakw, numpy.array(iws).imag, mus['up'], gkw)
  full_fill_Gkw_from_epsiolonk_and_gkw(Gkw, epsilonk, gkw)

def periodize_selfenergy(Gkw, Sigmakw, Sigmaijw, iws, mus, epsilonk, Sigma_imp_iw, Lx, Ly):  
  if mpi.is_master_node(): print "periodize_selfenergy"
  imp_key = [name for name,g in Sigma_imp_iw ][0] 
  Nc = len(Sigma_imp_iw[imp_key].data[0,0,:])
  nk = len(epsilonk['up'][0,:])
  for U in Sigmaijw.keys():
    for x in range(nk):
      for y in range(nk):
        ij = cellular_latt_to_imp_mapping(x,y,nk,Lx,Ly)
        if ij is None:
          Sigmaijw[U][:,x,y] = 0.0
        else:
          i = ij[0]
          j = ij[1]
          Sigmaijw[U][:,x,y] = Sigma_imp_iw[imp_key].data[:,i,j]
    Sigmakw[U][:,:,:] = spatial_FT(Sigmaijw[U], N_cores=1)
  full_fill_Gkw_from_iws_mus_epsiolonk_and_Sigmakw(Gkw, iws, mus, epsilonk, Sigmakw)

def matrix_dispersion(Nc, t,tp, kx, ky):
  if Nc==2:
    #  ABAB
    #  BABA 
    #  ABAB
    A =       [[0,t],
               [t,0]]

    tk = t*(exp(1j*kx)+2.0*cos(ky))

    B =       [[0,             tk],
               [numpy.conj(tk),0 ]]

    tpk = tp*cos(kx)*cos(ky)

    C =       [[tpk,0],
               [0, tpk]]

    return numpy.array(A) + numpy.array(B) + numpy.array(C)
  elif Nc==4:
    #  CD-CD-CD   <----o
    #  AB-AB-AB   X    |
    #  ||X||X||        | 
    #  CD-CD-CD        |
    #  AB-AB-AB        v Y
    A =       [[0,  t,  t,  tp],
               [t,  0,  tp, t ],
               [t,  tp, 0,  t ],
               [tp, t,  t,  0 ]]

    tkx = t*exp(-1j*kx)
    tky = t*exp(-1j*ky)
    ctkx = t*exp(1j*kx)
    ctky = t*exp(1j*ky)

    B =       [[0,    tkx,  tky,  0  ],
               [ctkx, 0,    0,    tky],
               [ctky, 0,    0,    tkx],
               [0,    ctky, ctkx, 0  ]]

    tpkAD = tp*(  exp(-1j*kx)+exp(-1j*ky) + exp(-1j*(kx+ky))  )
    tpkBC = tp*(  exp(1j*kx)+exp(-1j*ky) + exp(1j*(kx-ky))  )
    tpkDA = numpy.conj(tpkAD)
    tpkCB = numpy.conj(tpkBC)

    C =       [[0,     0,     0,     tpkAD ],
               [0,     0,     tpkBC, 0     ],
               [0,     tpkCB, 0,     0     ],
               [tpkDA, 0,     0,     0     ]]

    return numpy.array(A) + numpy.array(B) + numpy.array(C)
  else:
    assert False, "not yet implemented"

#-----------------------------------------------------------------------------------------------------------------------------------------------#
#                                        triangular specific
#-----------------------------------------------------------------------------------------------------------------------------------------------#

def triangular_identical_pair_sets(Lx,Ly):
  if Lx==2 and Ly==2: #this is not true symmetry of the cluster. deviations are expected. in the periodized scheme, this is how we're doing it. it's a bit ad hoc. do nested.
    return [ [ [0,0],[1,1],[2,2],[3,3] ],
             [ [0,1],[1,0],[0,2],[2,0],[1,2],[2,1],[2,3],[3,2],[1,3],[3,1] ],
             [ [0,3],[3,0] ]
           ]
  elif Lx==2 and Ly==1: 
    return [ [ [0,0],[1,1] ],
             [ [0,1],[1,0] ] 
           ]
  else: 
    assert False, "not implemented!"

def triangular_cellular_latt_to_imp_mapping(x,y,nk,Lx,Ly): 
  if Lx==2 and Ly==2: 
    X,Y = x,y
    if X>nk/2+1: X -= nk
    if Y>nk/2+1: Y -= nk
    if (abs(X)>=Lx) or (abs(Y)>=Ly): return None   
    i = 0
    if X<0: i=abs(X)
    if Y<0: i += Lx*abs(Y)  
    j = i+ Lx*Y + X
    return [i,j]
  elif Lx==2 and Ly==1: 
    if [x,y] in [[0,0]]:
      return [0,0]
    elif [x,y] in [ [0,1],[1,0],[nk-1,1],[1,nk-1],[0,nk-1],[nk-1,0] ]:
      return [0,1] 

def triangular_full_fill_Sigmaijkw_periodized(Sigmaijkw, Sigma_imp_iw, ks):
    if mpi.is_master_node(): print "full_fill_Sigmaijkw_periodized"
    assert len(Sigmaijkw.keys())==1, "must be only one block in Sigmaijkw"
    impkeys = [name for name,g in Sigma_imp_iw]    
    assert len(impkeys)==1, "must be only one block in Sigma_imp_iw"      
    impkey = impkeys[0]
    numpy.transpose(Sigmaijkw[Sigmaijkw.keys()[0]])[:,:,:] = numpy.transpose(Sigma_imp_iw[impkey].data)[:,:,:]    

    if Nc==2:
      s = Sigma_imp_iw[impkey].data[:,0,1]
      z = deepcopy(s)
      z[:] = 0    
      for kxi, kx in enumerate(ks):
        for kyi, ky in enumerate(ks):
          sk = s*(cos(kx+ky)+exp(1j*(ky-kx)))
          csk = numpy.conj(sk)

          B =       [[z,    skx  ],
                     [cskx, z    ]]
          skAA = s*cos(ky)
          skBB = skAA

          C =       [[skAA,  z   ],
                     [z,     skBB]]

          BC = numpy.array(B)+numpy.array(C)
          numpy.transpose(Sigmaijkw[Sigmaijkw.keys()[0]])[kyi,kxi,:,:,:] += BC

    elif Nc==4:   
      s = Sigma_imp_iw[impkey].data[:,0,1]
      sp = Sigma_imp_iw[impkey].data[:,0,3]    
      z = deepcopy(s)
      z[:] = 0    
      for kxi, kx in enumerate(ks):
        for kyi, ky in enumerate(ks):
          skx = s*exp(1j*kx)
          sky = s*exp(1j*ky)
          cskx = s*exp(-1j*kx)
          csky = s*exp(-1j*ky)

          B =       [[z,    skx,  sky,  z  ],
                     [cskx, z,    z,    sky],
                     [csky, z,    z,    skx],
                     [z,    csky, cskx, z  ]]
          spkAD = sp*numpy.conj(  exp(-1j*kx)+exp(-1j*ky) + exp(-1j*(kx+ky))  )
          spkDA = sp*(  exp(-1j*kx)+exp(-1j*ky) + exp(-1j*(kx+ky))  )
          skBC = s*numpy.conj(  exp(1j*kx)+exp(-1j*ky) + exp(1j*(kx-ky))  )
          skCB = s*(  exp(1j*kx)+exp(-1j*ky) + exp(1j*(kx-ky))  )

          C =       [[z,     z,     z,     spkAD ],
                     [z,     z,     skBC,  z     ],
                     [z,     skCB,  z,     z     ],
                     [spkDA, z,     z,     z     ]]

          BC = numpy.array(B)+numpy.array(C)
          numpy.transpose(Sigmaijkw[Sigmaijkw.keys()[0]])[kyi,kxi,:,:,:] += BC

def triangular_matrix_dispersion(Nc, t, kx, ky):
  if Nc==2:
    #  ABAB   \ky /kx
    #  BABA    \ /
    #  ABAB     V
    A =       [[0,t],
               [t,0]]

    tk = t*(exp(1j*(ky-kx))+2.0*cos(kx+ky))

    B =       [[0,             tk],
               [numpy.conj(tk),0 ]]

    tpk = t*cos(ky)

    C =       [[tpk,0],
               [0, tpk]]

    return numpy.array(A) + numpy.array(B) + numpy.array(C)
  elif Nc==4:
    #  CD-CD-CD   <----o
    #  AB-AB-AB   X    |
    #  ||X||X||        | 
    #  CD-CD-CD        |
    #  AB-AB-AB        v Y
    A =       [[0,  t,  t,  0],
               [t,  0,  t,  t ],
               [t,  t,  0,  t ],
               [0,  t,  t,  0 ]]

    tkx = t*exp(-1j*kx)
    tky = t*exp(-1j*ky)
    ctkx = t*exp(1j*kx)
    ctky = t*exp(1j*ky)

    B =       [[0,    tkx,  tky,  0  ],
               [ctkx, 0,    0,    tky],
               [ctky, 0,    0,    tkx],
               [0,    ctky, ctkx, 0  ]]

    tpkAD = tp*(  exp(-1j*kx)+exp(-1j*ky) + exp(-1j*(kx+ky))  )
    #tpkBC = tp*(  exp(1j*kx)+exp(-1j*ky) + exp(1j*(kx-ky))  )
    tpkDA = numpy.conj(tpkAD)
    #tpkCB = numpy.conj(tpkBC)

    C =       [[0,     0,     0,     tpkAD ],
               [0,     0,     0,     0     ],
               [0,     0,     0,     0     ],
               [tpkDA, 0,     0,     0     ]]

    return numpy.array(A) + numpy.array(B) + numpy.array(C)
  else:
    assert False, "not yet implemented"
