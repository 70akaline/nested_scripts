import numpy
from numpy.linalg import inv

from pytriqs.operators import *
from pytriqs.archive import *
from pytriqs.gf.local import *
from pytriqs.arrays import BlockMatrix, BlockMatrixComplex
import pytriqs.utility.mpi as mpi


from copy import deepcopy
from tail_fitters import symmetrize_blockgf

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
      shift[C][b] = inv(solvers[C].G0_shift_iw[b].data[nw/2+10,:,:]) - inv(solvers[C].G0_iw[b].data[nw/2+10,:,:]) 

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

def second_derivative(a, dx=1, axis=0):
    sha = numpy.shape(a)
    sha = list(sha)
    sha[axis]-=2
    sha = tuple(sha)
    d2a = numpy.zeros(sha)
    for xi in range(sha[axis]):
        d2a[xi,:,:] = a[xi,:,:]+a[xi+2,:,:]-2.0*a[xi+1,:,:]
    d2a/=dx**2.0 
    return d2a

def patch_range(a, rng, desired_sign=-1):   
    nl = len(a)
    l0 = rng[0]
    l1 = rng[1]    
    for dl in range(1000):        
        if l0>0: l0-=1
        if l1<nl-1: l1+=1                        
        a0 = a[l0]
        a1 = a[l1]      
        for l in range(l0,l1):
            a[l] = a0+(a1-a0)*(l-l0)/(l1-l0)        
        fixed = True                              
        if l0>0:
            d2al0 = a[l0-1]+a[l0+1]-2.0*a[l0]    
            fixed = fixed and (d2al0*desired_sign>=0)    
        if l1<nl-1:
            d2al1 = a[l1-1]+a[l1+1]-2.0*a[l1]    
            fixed = fixed and (d2al1*desired_sign>=0)
        if fixed:
            print "fixed it","dl: ",dl,"l0,l1: ",l0,l1
            return
    print "couldn't fix it!"    
            
            
def fix_second_derivative(a, dx, desired_sign=-1, axis=0, max_counter=4, tolerance = 1e-2):
    assert axis == 0, "not general"
    old_a = deepcopy(a)
    d2a = second_derivative(a, dx, axis)
    old_d2a = deepcopy(d2a)
    sha = numpy.shape(d2a)
    sha = list(sha)  
    assert sha[1]==sha[2]
    #print "sha: ", sha
    #fix diagonal elements first
    been_fixing = False
    
    for i in range(sha[1]):        
        ranges = []               
        for l in range(sha[0]):
            d = d2a[l,i,i]
            if d*desired_sign<-tolerance:
                if len(ranges)==0: ranges.append([l])
                if len(ranges[-1])!=1: ranges.append([l])
            if d*desired_sign>=0:
                if len(ranges)>0:
                    if len(ranges[-1])==1: ranges[-1].append(l)                
        for rng in ranges:
            print "fixing range",rng,"in diagonal element",i
            patch_range(a[:,i,i], list(numpy.array(rng)+1), desired_sign)    
        if len(ranges)!=0:
            been_fixing = True

    #fix eigenvalues    
    print "checking eigenvalues"    
    
    for counter in range(max_counter):
        d2a = second_derivative(a, dx, axis)
        not_good = False
        for l in range(sha[0]):
            w,v = numpy.linalg.eig(d2a[l,:,:])        
            if len([ww for ww in w if ((ww*desired_sign)<-10*tolerance)])!= 0:                
                #print sorted(w)
                print "max w: ", numpy.amax(w)
                not_good = True
                break
        if not_good:                 
            print "reducing off-diagonal elements to get proper eigenvalues"
            been_fixing = True
            for i in range(sha[1]):
                for j in range(sha[1]):
                    if i==j: continue
                    a[:,i,j] *= 0.95            
        if not not_good: 
            if counter==0:
              print "nothing to fix"
            else:
              print "fixed eigenvalues, counter:",counter
            break
    if counter==max_counter-1: print "can't fix eigenvalues"       
    return been_fixing         

from tail_fitters import fit_fermionic_gf_tail
def Gweiss_causal_cautionary(Gweiss_iw, ntau=1000, max_counter=4, tolerance=1e-2):   
  print "######### Checking causality of Gweiss ########"
  for name, g in Gweiss_iw:
    print ">>>>>>>>>>>>>>>>>> block ",name
    nsites = len(g.data[0,:,0])
    g_tau = GfImTime(indices = range(nsites), beta = g.beta, n_points = ntau, statistic = 'Fermion')
    fit_fermionic_gf_tail(g)
    g_tau << InverseFourier(g)  

    dtau=g.beta/(ntau-1)       
    been_fixing = fix_second_derivative(g_tau.data.real, dtau, max_counter=max_counter, tolerance = tolerance)
    if been_fixing: print ">>>>>>>>>>>>>>>>>> block ",name," has been fixed!"
    else: print ">>>>>>>>>>>>>>>>>> block ",name," is fine."

    g << Fourier(g_tau)
  print "#########           DONE!!!            ########"  
  return False

