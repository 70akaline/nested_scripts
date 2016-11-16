import numpy

from pytriqs.operators import *
from pytriqs.archive import *
from pytriqs.gf.local import *
from pytriqs.arrays import BlockMatrix, BlockMatrixComplex
import pytriqs.utility.mpi as mpi

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






