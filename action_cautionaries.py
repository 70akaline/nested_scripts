import numpy
from numpy.linalg import inv

from pytriqs.operators import *
from pytriqs.archive import *
from pytriqs.gf.local import *
from pytriqs.arrays import BlockMatrix, BlockMatrixComplex
import pytriqs.utility.mpi as mpi


from copy import deepcopy
from tail_fitters import symmetrize_blockgf

def impose_real_valued_in_imtime_numpy(Q):
  #print "impose_real_valued_in_imtime_numpy"
  Q[:,:,:] += numpy.conjugate(Q[::-1,:,:])
  Q /= 2.0

def impose_real_valued_in_imtime(Q):
  impose_real_valued_in_imtime_numpy(Q.data)

def impose_particle_hole_symmetry(Qimpdata, tail=0.0): #set to U/2 for sigma
  print "imposing particle hole symmetry..."
  sha = numpy.shape(Qimpdata)
  nsites = sha[2]
  L = int(numpy.sqrt(nsites))
  print "nsites:",nsites," L:",L,"tail:",tail
  for i in range(nsites):
    Qimpdata[:,i,i] -= Qimpdata[:,i,i].real
    Qimpdata[:,i,i] += tail
  
  for i in range(nsites-1):
    for j in range(i+1,nsites):                
      rx = abs(i%L - j%L)
      ry = abs(i/L - j/L)
      #print "i,j:",i,j,"rx,ry:",rx,ry,"(rx+ry)%2",(rx+ry)%2 
      if (rx+ry)%2==0: XX = numpy.real
      else: XX = numpy.imag  
      Qimpdata[:,i,j] -= XX(Qimpdata[:,i,j])    
      Qimpdata[:,j,i] -= XX(Qimpdata[:,j,i])
  impose_real_valued_in_imtime_numpy(Qimpdata)

def nonloc_sign_cautionary(Q, clip_value = 0.0, desired_sign = -1, clip_off = False, real_or_imag = 'imag'):
      clip_off = clip_off and not (real_or_imag=='imag')
      clipped = False
      if real_or_imag == 'real':
        res2 = numpy.greater(desired_sign*Q[:,:,:].real, 0.0 )
      elif real_or_imag == 'imag': 
        nw = len(Q[:,0,0])
        res2 = numpy.greater(desired_sign*Q[nw/2:,:,:].imag, 0.0 )          
      if not numpy.all(res2):
        clipped = True
        if mpi.is_master_node(): print "nonloc_sign_cautionary: wrong sign!!!%s"%((" Clipping off to %s"%clip_value) if clip_off else "")
        if clip_off: 
          Q[:,:,:] = Q[:,:,:]*(1.0 - res2[:,:,:]) + res2[:,:,:]*clip_value*desired_sign             
      return clipped

  
def local_nan_cautionary(data, struct, Qs = [], raise_exception = True):
      nan_found = False
      for Q in Qs:
        for block in struct.keys():
          if numpy.any(numpy.isnan(vars(data)[Q][block].data[:,:,:])):
            nan_found=True
            if mpi.is_master_node(): print "local_nan_cautionary: nan in ",Q,"[",block,"]"
      if nan_found and raise_exception:  
        raise Exception('No nans please')

def symmetrize_cluster_impurity(Sigma_imp_iw, identical_pairs, name="quantity"):
  err = False
  for C in identical_pairs.keys():     
    for ips in identical_pairs[C]:
      total = deepcopy(Sigma_imp_iw[C].data[:,0,0])
      total[:] = 0.0   
      counter = 0
      for ip in ips:  
        i,j = ip[0],ip[1]
        if (i==-1) or (j==-1): continue
        total += Sigma_imp_iw[C].data[:,i,j]
        counter += 1 
      total[:] /= counter 
      for ip in ips:  
        i,j = ip[0],ip[1]
        if (i==-1) or (j==-1): continue
        if numpy.any(numpy.greater(numpy.abs(Sigma_imp_iw[C].data[:,i,j]-total[:]), 5e-3)):
          err = True  
          if mpi.is_master_node():
            print "symmetrize_cluster_impurity: WARNING!! %s[%s][:,%s,%s] far from average"%(name,C,i,j)
        Sigma_imp_iw[C].data[:,i,j] = total[:]
  return err

def symmetrize_cluster_impurities(X_imp_iw, identical_pairs_across_impurities, name="quantity"):
  err = False
  for rxry in identical_pairs_across_impurities.keys():
    Cdummy = identical_pairs_across_impurities[rxry][0][0]
    #print identical_pairs_across_impurities[rxry][0]
    #print Cdummy 
    total = deepcopy(X_imp_iw[Cdummy].data[:,0,0])
    total[:] = 0.0   
    counter = 0
    for ip in identical_pairs_across_impurities[rxry]:
        C,i,j = ip[0],ip[1],ip[2]
        if (i==-1) or (j==-1): continue
        total += X_imp_iw[C].data[:,i,j]
        counter += 1 
    total[:] /= counter 
    for ip in identical_pairs_across_impurities[rxry]:
        C,i,j = ip[0],ip[1],ip[2]
        if (i==-1) or (j==-1): continue
        if numpy.any(numpy.greater(numpy.abs(X_imp_iw[C].data[:,i,j]-total[:]), 5e-3)):
          err = True  
          if mpi.is_master_node():
            print "symmetrize_cluster_impurities: WARNING!! %s[%s][:,%s,%s] far from average"%(name,C,i,j)
        X_imp_iw[C].data[:,i,j] = total[:]
  return err


def symmetric_G_and_self_energy_on_impurity(G_imp_iw, Sigma_imp_iw, solvers, identical_pairs_Sigma, identical_pairs_G, across_imps = False, identical_pairs_G_ai = []):
  if (identical_pairs_Sigma is None) or (identical_pairs_G is None): return False
  M_imp_dict = {}
  blocks = [name for name, g in solvers[solvers.keys()[0]].G0_iw]
  for b in blocks:
    M_imp_dict[b] = deepcopy(Sigma_imp_iw)

  for C in solvers.keys():  
    for b in blocks:
      M_imp_dict[b][C] << solvers[C].M_iw[b]
  for b in blocks:  
    symmetrize_cluster_impurity(M_imp_dict[b], identical_pairs_Sigma, "M_imp")

  G_imp_dict = deepcopy(M_imp_dict)
  for C in solvers.keys():
    for b in blocks:  
      G_imp_dict[b][C] << solvers[C].G0_shift_iw[b] + solvers[C].G0_shift_iw[b]*M_imp_dict[b][C]*solvers[C].G0_shift_iw[b]
  for b in blocks:  
    symmetrize_cluster_impurity(G_imp_dict[b], identical_pairs_G, "G_imp")
    if across_imps: symmetrize_cluster_impurities(G_imp_dict[b], identical_pairs_G_ai, "G_imp")

  shift = {}      
  for C in solvers.keys():
    shift[C] = {} 
    for b in blocks:
      nw = len(solvers[C].G0_shift_iw[b].data[:,0,0])
      shift[C][b] = inv(solvers[C].G0_shift_iw[b].data[nw/2,:,:]) - inv(solvers[C].G0_iw[b].data[nw/2,:,:]) 

    G_imp = deepcopy(solvers[C].G_iw)
    for b in blocks:
      G_imp[b] << G_imp_dict[b][C]
    symmetrize_blockgf(G_imp) 
    G_imp_iw[C] << G_imp[blocks[0]]

  Sigma_imp_dict = deepcopy(M_imp_dict)
  for C in solvers.keys():
    for b in blocks:
      Sigma_imp_dict[b][C] << M_imp_dict[b][C]*solvers[C].G0_shift_iw[b]*inverse(G_imp_iw[C]) - shift[C][b]
    symmetrize_cluster_impurity(Sigma_imp_dict[b], identical_pairs_Sigma)
    Sigma_imp = deepcopy(solvers[C].Sigma_iw)
    for b in blocks:
      Sigma_imp[b] << Sigma_imp_dict[b][C]
    symmetrize_blockgf(Sigma_imp) 
    Sigma_imp_iw[C] << Sigma_imp[blocks[0]]

  return False
  


#===================================================================================================================================#


