#from pytriqs.lattice.tight_binding import *
from pytriqs.lattice.bz_patch import *
from pytriqs.dos.hilbert_transform import *
import pytriqs.utility.mpi as mpi

from pytriqs.plot.mpl_interface import *

from matplotlib.patches import Polygon

import numpy as np
from numpy import pi
import numpy.linalg
from scipy.spatial import Voronoi



from data_containers import IBZ

class dca_struct:
    def __init__(self, n1,m1,n2,m2, TB,
                 ex = np.array([1,0]),
                 ey = np.array([0,1])):
        self.n1 = n1
        self.m1 = m1
        self.n2 = n2
        self.m2 = m2    
        self.Lmax = 5 * max(abs(n1),abs(m1),abs(n2),abs(m2))
                        
        self.ex = ex
        self.ey = ey                
        assert abs(numpy.linalg.norm(ex)-1)<1e-5, "ex not unit vector"
        assert abs(numpy.linalg.norm(ey)-1)<1e-5, "ey not unit vector"
        
        self.TB = TB
        
        self.dim = abs(n1*m2 - m1*n2)

        self.eps = 0.001
        self.R1, self.R2, self.d1, self.d2, self.BZsize = self.get_auxiliary()
        self.r_points = self.get_r_points()
        self.k_unit, self.k_points = self.get_k_points()
        self.dca_patches = self.get_dca_patches()
        self.ij_to_0i = self.get_ij_to_0i_map()
        self.P, self.Pinv = self.get_FT_matrices()
        
        if mpi.rank == 0:
            print 'dca_scheme:'
            print "   ex,ey = ", ex,ey
            print "   Nc = ", self.dim
            print "   n1,m1,n2,m2 = ", n1,m1,n2,m2
            print "   Lmax: ", self.Lmax  
            print "   r0 = ",self.get_r0()

        
    def get_auxiliary(self):
        n1, n2, m1, m2, ex, ey = self.n1, self.n2, self.m1, self.m2, self.ex, self.ey

        BZsize = [ 1.0/max(ex[0],ey[0]),1.0/max(ex[1],ey[1]) ]
        
        R1 = n1 * ex + m1 * ey
        R2 = n2 * ex + m2 * ey

        R1x = R1[0] 
        R1y = R1[1] 
        R2x = R2[0] 
        R2y = R2[1] 
        d1 = np.array([R2y,-R2x])/float(R1x*R2y-R1y*R2x)
        d2 = np.array([R1y,-R1x])/float(R1y*R2x-R1x*R2y)        

        return R1, R2, d1, d2, BZsize    
       
    def get_r_points(self):
        eps = self.eps
        ex, ey, R1, R2, d1, d2, L = self.ex, self.ey, self.R1, self.R2, self.d1, self.d2, self.Lmax
        
        # Get the direct lattice points
        r_points = []
        for x in range(-L,L):
            for y in range(-L,L):                
                v = x * ex + y * ey
                #print x,y,v
                if (np.dot(v,d1)>-eps) and (np.dot(v,d2)>-eps) and (np.dot(v,d1)<(1-eps)) and (np.dot(v,d2)<(1-eps)):
                    r_points.append(v)
        r_points = np.array(r_points)
        
        assert(len(r_points) == self.dim)
        return r_points

    def get_k_points(self): 
        dim, eps = self.dim, self.eps
        R1, R2, d1, d2, BZsize, L = self.R1, self.R2, self.d1, self.d2, self.BZsize, self.Lmax

        # Get the reciprocal lattice points
        k_unit = []
        for u in range(-L, L):
            for v in range(-L, L):
                kv = u * d1 + v * d2
                if (kv[0]>-eps) and (kv[1]>-eps) and (kv[0]<(BZsize[0]-eps)) and (kv[1]<(BZsize[1]-eps)):
                    k_unit.append(kv)
        pi = np.arccos(-1)
        k_unit = np.array(k_unit)
        k_points = 2*pi*k_unit
                
        assert(len(k_unit) == dim)
        return k_unit, k_points
    
    @classmethod
    def get_wigner_seitz(cls, k1, k2, L=10):
        kpts = []
        for u in range(-L,L):
            for v in range(-L,L):
                kpts.append(u*k1+v*k2)
        V = Voronoi(np.array(kpts))
        res = np.array([ V.vertices[n] for n in V.regions[V.point_region[2*L*L+L]] ])
        return res   
    
    def get_voronoi(self):
        k1,k2,k_unit,L = self.d1, self.d2, self.k_unit, 10        
        kpts = []
        for u in range(-L,L):
            for v in range(-L,L):
                kpts.append(u*k1+v*k2)
        return Voronoi(np.array(kpts))

    def get_dca_patches(self):
        d1,d2,k_unit,TB = self.d1, self.d2, self.k_unit, self.TB

        # Prepare the reciprocal lattice vectors of the original lattice
        a2xa3 = numpy.array([self.ey[1],-self.ey[0]])
        a3xa1 = numpy.array([-self.ex[1],self.ex[0]])
        b1 = a2xa3 / numpy.dot(self.ex, a2xa3)
        b2 = a3xa1 / numpy.dot(self.ex, a2xa3)
            
        #print "b1:",b1
        #print "b2:",b2
        # Prepare the transformation matrix
        P = numpy.transpose([b1,b2])
        Pinv = numpy.linalg.inv(P)

        # Define the patches
        dca_patches = []
        ws = self.__class__.get_wigner_seitz(d1,d2)
        for i, kv in enumerate(k_unit):
            p = ws + kv
            #print "old p: ",p
            # Change the basis of the vectors
            for ppi, pp in enumerate(p):
                p[ppi] = numpy.dot(Pinv,pp)
            #print "new p: ",p
            dca_patches += [ BZPatch(name = '%02d'%i, polygons = [p]) ]

        # Hilbert transforms
        for p in dca_patches:
          p.ht = HilbertTransform(p.dos(TB, 101, 400))
        return dca_patches        
    
    def get_ij_to_0i_map(self):
        # Fill the map (i,j) --> (0,i)
        dim, eps, d1, d2, R1, R2, r_points = self.dim, self.eps, self.d1, self.d2, self.R1, self.R2, self.r_points
        r0 = self.get_r0()
        ij_to_0i = np.zeros((dim,dim), np.int32)
        for i in range(dim):
          for j in range(dim):
            v = r_points[r0] + r_points[j] - r_points[i]
            if (np.dot(v,d1)<-eps): v += R1
            if (np.dot(v,d1)>(1-eps)): v -= R1
            if (np.dot(v,d2)<-eps): v += R2
            if (np.dot(v,d2)>(1-eps)): v -= R2
            for ind, rv in enumerate(r_points):
                if (np.linalg.norm(v-rv) < eps):
                  ij_to_0i[i,j] = ind
                  break
        return ij_to_0i  

    def i_to_ij(self,ind):
        for i in range(self.dim):
            for j in range(self.dim):
                if self.ij_to_0i[i,j] == ind:
                    return i,j
        assert False, "dca i_to_ij: not found!!!!"

    def get_QR_from_Q_imp(self, QR_iw, Q_imp_iw):
      dim = len([name for name,q in QR_iw])  
      assert dim == self.dim, "wrong size of QR container" 
      block = [name for name,q in Q_imp_iw]
      assert len(block)==1, "in dca there is only one impurity problem"
      for R, q in QR_iw:
        i,j = self.i_to_ij(int(R))
        q << Q_imp_iw[block[0]][i,j]
    
    def get_Q_imp_from_QR(self, Q_imp_iw, QR_iw):
      if mpi.is_master_node(): print "full_fill_Gweiss_iw_from_GweissR_iw"
      dim = len([name for name,q in QR_iw])  
      assert dim == self.dim, "wrong size of QR container" 
      block = [name for name,q in Q_imp_iw]
      assert len(block)==1, "in dca there is only one impurity problem"
      for i in range(dim):
        for j in range(dim):
            Q_imp_iw[block[0]][i,j] << QR_iw["%02d"%(self.ij_to_0i[i,j])]

    def get_identical_pairs(self):
        dim, ij_to_0i = self.dim, self.ij_to_0i
        identical_pairs = []
        for i in range(dim):
            for j in range(dim):
                found = False
                for l,ip in enumerate(identical_pairs):                    
                    if ij_to_0i[i,j] == ij_to_0i[ip[0][0],ip[0][1]]:
                        identical_pairs[l].append([i,j])
                        found = True
                        break
                if not found:
                    identical_pairs.append([[i,j]])                    
        return identical_pairs

    def get_FT_matrices(self):
        dim, r_points, k_points = self.dim, self.r_points, self.k_points
        # Basis change matrices
        P = np.zeros([dim,dim], np.complex)
        for j in range(dim):
          for k in range(dim):
            P[j,k] = np.exp(1j * np.dot(k_points[k], r_points[j]))

        Pinv = np.linalg.inv(P)
        return P, Pinv

    def get_QK_from_QR(self, QK_iw, QR_iw):
      r0 = self.get_r0()
      QK_iw.zero()
      dim = self.dim
      P, Pinv = self.P, self.Pinv
      for i in range(dim):
        for l in range(dim):
          QK_iw["%02d"%i] += dim * Pinv[i,r0] * QR_iw["%02d"%l] * P[l,i]

    def get_QR_from_QK(self, QR_iw, QK_iw, l_list = []):
      r0 = self.get_r0()
      QR_iw.zero()
      dim = self.dim
      P, Pinv = self.P, self.Pinv
      for l in (range(dim) if l_list==[] else l_list):
        for i in range(dim):
          QR_iw["%02d"%l] += P[r0,i] * QK_iw["%02d"%i] * Pinv[i,l]

    def get_independent_r_point_groups(self):
        if not ( self.m1==0 and self.n2==0 ):
          print 'inapplicable to general clusters, returning None'
          return None
        n1,m2 = self.n1,self.m2
        
        indep_r_groups = []
        for rp in self.r_points:
            if rp[0]>n1/2 or rp[1]>m2/2 or rp[1]>rp[0]: continue
            indep_r_groups.append([])
            for l in range(n1*m2):
                p = self.r_points[l,:].copy()
                if p[0]>n1/2: p[0]=n1-p[0]
                if p[1]>m2/2: p[1]=m2-p[1]
                if p[1]>p[0]: p[0], p[1] = p[1], p[0]
                if p[0] == rp[0] and p[1] == rp[1]:
                    indep_r_groups[-1].append(l)
        return indep_r_groups    

    def symmetrize_QR(self, QR):
        irs = self.get_independent_r_point_groups()
        for ir in irs:
            tot = 0.0
            for l in ir:
                tot += QR["%.2d"%l].data[:,:,:]
            tot/=len(ir)
            for l in ir:
                QR["%.2d"%l].data[:,:,:] = tot

    def get_Qkw_from_QK_iw(self, QK_iw, nk=64): #piecewise constant interpolation, voronoi patches
        ks = numpy.linspace(0,2*pi,nk, endpoint=False)
        nw = len(QK_iw['00'].data[:,0,0])
        Qkw = numpy.zeros((nw,nk,nk),dtype=numpy.complex_)            
        full_k_points = list(self.k_points)
        full_k_points += list(self.k_points+[0,2*pi])
        full_k_points += list(self.k_points+[2*pi,0])
        full_k_points += list(self.k_points+[2*pi,2*pi])
                        
        for kxi in range(nk):
            for kyi in range(nk):
                min_dist,min_l = 2*pi, 0
                for l in range(len(full_k_points)):
                    dist = numpy.linalg.norm(numpy.array([ks[kxi],ks[kyi]]) - numpy.array(full_k_points[l]))
                    if dist<min_dist: min_dist, min_l = dist, l
                while min_l>=self.dim: min_l-=self.dim
                Qkw[:,kxi,kyi] = QK_iw['%02d'%min_l].data[:,0,0]   

        return Qkw, ks 

    def get_Qk_from_QR_embedded(self, Qkw, QR_iw, ks):
        assert self.m1==0 and self.n2==0, 'inapplicable to general clusters'
        n1,m2 = self.n1,self.m2
        r_points = self.r_points
        indep_r_groups = self.get_independent_r_point_groups()
        Qkw[:,:,:] = 0.0
        for rg in indep_r_groups:
            key = "%.2d"%rg[0]
            r = r_points[rg[0]]
            rx,ry = r[0],r[1]
            #print key, r, rx,ry
            if rx == 0 and ry == 0:
                numpy.transpose(Qkw)[:,:,:] += QR_iw[key].data[:,0,0]
                continue
            pref = lambda kx,ky: 0
            for x in ([rx,-rx] if rx!=0 else [0]):
                for y in ([ry,-ry] if ry!=0 else [0]):
                    #print x,y
                    pref = lambda kx,ky,pref=pref,x=x,y=y: pref(kx,ky) + numpy.exp(-1j*(kx*x+ky*y))
                    if rx!=ry: 
                        pref = lambda kx,ky,pref=pref,x=x,y=y: pref(kx,ky) + numpy.exp(-1j*(kx*y+ky*x))
                        #print y,x       
            for kxi, kx in enumerate(ks):
                for kyi, ky in enumerate(ks):
                    Qkw[:,kxi,kyi] += pref(kx,ky)*QR_iw[key].data[:,0,0]                    

    def get_Qk_from_QR(self, Qkw, QR_iw, ks, symmetrize = True):
        assert self.m1==0 and self.n2==0, 'inapplicable to general clusters'
        n1,m2 = self.n1,self.m2
        r_points = self.r_points
        Qkw[:,:,:] = 0.0    
        for l, r in enumerate(r_points):
            key = "%.2d"%l
            rx,ry = r[0],r[1]
            #print l, r, rx,ry        
            if rx>n1/2: rx=rx-n1
            if ry>m2/2: ry=ry-m2
            #print "after:", rx,ry                
            pref = lambda kx,ky: 0
            for sgnx in [1,-1]:
                for sgny in [1,-1]:
                    for flip in [True,False]:
                        if flip:
                            pref = lambda kx,ky, pref=pref, sgnx=sgnx, sgny=sgny: pref(kx,ky) + (numpy.exp(-1j*(sgnx*ky*rx+sgny*kx*ry)))
                        else:
                            pref = lambda kx,ky, pref=pref, sgnx=sgnx, sgny=sgny: pref(kx,ky) + (numpy.exp(-1j*(sgnx*kx*rx+sgny*ky*ry)))
            pref = lambda kx,ky, pref=pref: 0.125*pref(kx,ky)        
            for kxi, kx in enumerate(ks):
                for kyi, ky in enumerate(ks):
                    Qkw[:,kxi,kyi] += pref(kx,ky)*QR_iw[key].data[:,0,0]
        

    def QK_iw_to_QKw(self, QKw, QK_iw):           
        assert self.m1==0 and self.n2==0, 'inapplicable to general clusters'
        assert self.n1==self.m2, 'must be'
        nK = self.n1
        for l in range(self.dim):
            QKw[:,l/nK,l%nK] = QK_iw["%.2d"%l].data[:,0,0]
            
    def QKw_to_QK_iw(self, QK_iw, QKw):    
        assert self.m1==0 and self.n2==0, 'inapplicable to general clusters'
        assert self.n1==self.m2, 'must be'
        nK = self.n1
        for l in range(self.dim):
            QK_iw["%.2d"%l].data[:,0,0] = QKw[:,l/nK,l%nK]

    def Qkw_to_QK_iw(self, QK_iw, Qkw):    
        print "Qkw_to_QK_iw"
        nk = len(Qkw[0,:,0])
        assert self.m1==0 and self.n2==0, 'inapplicable to general clusters'
        assert self.n1==self.m2, 'must be'
        nK = self.n1
        assert nk % nK == 0, "has to be divisible by nK"
        D = nk/nK    
        for l in range(self.dim):  
            #print "filling K: ", l        
            QK_iw["%.2d"%l].data[:,0,0] = Qkw[:,(l/nK)*D,(l%nK)*D]

    def Qrw_to_QR_iw(self, QR_iw, Qrw):    
        assert self.m1==0 and self.n2==0, 'inapplicable to general clusters'
        n1,m2 = self.n1,self.m2
        nk = len(Qrw[0,:,0])
        for l, r in enumerate(self.r_points):
          rx,ry = r[0],r[1]
          #print "l, r, rx,ry: ",l,r,rx,ry  
          if rx>n1/2: rx=rx-n1
          if ry>m2/2: ry=ry-m2
          #print "after:", rx,ry    
          QR_iw["%.2d"%l].data[:,0,0] = Qrw[:,rx,ry]

    def get_impurity_struct(self):
        return {'x': range(self.dim)}

    def get_fermionic_struct(self):
        fermionic_struct = {}
        for l in range(self.dim): fermionic_struct["%02d"%l]=[0]
        return fermionic_struct 

    def get_r0(self):
        for r0 in range(self.dim):
            if list(self.r_points[r0]) == [0,0]: return r0
        assert found, "there has to be a zero real space vector" 
    
    def plot_r_points(self, plt):        
        r_points, R1, R2, d1, d2 = self.r_points, self.R1, self.R2, self.d1, self.d2
        
        x_min = min(0, R1[0], R2[0], (R1+R2)[0])
        x_max = max(0, R1[0], R2[0], (R1+R2)[0])
        y_min = min(0, R1[1], R2[1], (R1+R2)[1])
        y_max = max(0, R1[1], R2[1], (R1+R2)[1])
        
        plt.plot(x_min,y_min,'.')
        plt.plot(x_max,y_max,'D')

        all_points = numpy.array([n*self.ex+m*self.ey for n in range(-10,10) for m in range(-10,10)])
        plt.plot(all_points[:,0],all_points[:,1],'o',color='lightgray')
        
        plt.plot([0,R1[0]],[0,R1[1]],'x-', label='R1')
        plt.plot([0,R2[0]],[0,R2[1]],'x-', label='R2')
        plt.plot([0,d1[0]],[0,d1[1]],'d-', label='d1')
        plt.plot([0,d2[0]],[0,d2[1]],'d-', label='d2')
        plt.plot([R1[0],(R1+R2)[0]],[R1[1],(R1+R2)[1]],'-')
        plt.plot([R2[0],(R1+R2)[0]],[R2[1],(R1+R2)[1]],'-')        
        plt.plot(r_points[:,0],r_points[:,1],'o')
        
        plt.axes().set_aspect('equal')
        plt.xlim(x_min-1,x_max+1)
        plt.ylim(y_min-1,y_max+1)
        plt.legend(bbox_to_anchor=(1.5,1.0))
        plt.show() 

def plot_k_points(self, plt):      
    fig, ax = plt.subplots()

    # Prepare the reciprocal lattice vectors of the original lattice
    a2xa3 = numpy.array([self.ey[1],-self.ey[0]])
    a3xa1 = numpy.array([-self.ex[1],self.ex[0]])
    b1 = a2xa3 / numpy.dot(self.ex, a2xa3)
    b2 = a3xa1 / numpy.dot(self.ex, a2xa3)

    # Prepare the transformation matrix
    P = numpy.transpose([b1,b2])
    Pinv = numpy.linalg.inv(P)
    
    nk = 100
    ks = numpy.linspace(-pi,3.0*pi,nk,endpoint=False)

    epsk = numpy.zeros((nk,nk))

    for kxi, kx in enumerate(ks):
        for kyi, ky in enumerate(ks):
            k = numpy.dot(Pinv,numpy.array([kx/(2.0*pi),ky/(2.0*pi)])) #change basis to d1,d2 to put in TB            
            epsk[kxi,kyi] = energies_on_bz_path (TB, k, k, 1)              

    CP = plt.contourf(ks,ks, numpy.transpose(epsk), levels=numpy.linspace(numpy.amin(epsk),numpy.amax(epsk),100))
    plt.colorbar(CP)
    plt.plot()    
    
    BZsize = 2.0*pi*numpy.array(self.BZsize)
    ax.add_artist(Polygon([[0,0],[BZsize[0],0],[BZsize[0],BZsize[1]],[0,BZsize[1]]],
                          True, color='gray', alpha=0.2, lw=2,ec='black'))                            

    colors = ['blue','green','red','cyan','magenta','yellow','gray','white']
    counter = 0
    for patch in self.dca_patches:        
        trgls = numpy.array(patch._triangles)
        for trgli,trgl in enumerate(trgls):
            trgls[trgli] = 2.0*pi*numpy.dot(P,trgl)
        ax.add_artist(Polygon(trgls, True, color=colors[counter], alpha=0.5, lw=0))            
        counter += 1
        if counter>=len(colors): counter=0        

    k_points, k_unit = self.k_points, self.k_unit

    all_points = numpy.array([2.0*numpy.pi*(n*self.d1+m*self.d2) for n in range(-10,10) for m in range(-10,10)])
    plt.plot(all_points[:,0],all_points[:,1],'o',color='lightgray')

    plt.plot(k_points[:,0],k_points[:,1],'o', color='white')
    
    plt.xlim(-pi,3*pi)
    plt.ylim(-pi,3*pi)
    plt.axes().set_aspect('equal')        
    plt.show()

    for Ki,patch in enumerate(self.dca_patches):
        dos = patch.dos(TB, 101, 1000)
        plt.plot(dos.eps,dos.rho, label=r"$\mathbf{K}=(%.2f,%.2f)$"%(dca.k_points[Ki][0],dca.k_points[Ki][1]))
    plt.legend(bbox_to_anchor=(1.5,1.0))
    plt.xlabel(r"$\epsilon$")
    plt.ylabel(r"$\rho(\mathbf{K};\epsilon)$")
    plt.show()
        
