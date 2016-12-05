import math
from copy import copy
from copy import deepcopy
from operator import itemgetter
import itertools
from numpy import pi
from pytriqs.archive import *

####################################################################################################################
#---------------------------- symmetry operations to figure out equivalent links ----------------------------------#

def Translate_X(a, dx):
    a[0]+=dx

def Translate_Y(a, dy):
    a[1]+=dy
        
def Translate_Z(a, dz):
    a[2]+=dz

def Rotate_2D(a, phi):
    R = [[math.cos(phi),-math.sin(phi)],[math.sin(phi),math.cos(phi)]]
    b = [sum([R[j][i]*a[j] for j in range(2)]) for i in range(2)]
    for i in range(2): a[i] = b[i]    
    
def Rotate_Z(a, phi):
    R = [[math.cos(phi),-math.sin(phi),0],[math.sin(phi),math.cos(phi),0],[0,0,1]]
    b = [sum([R[j][i]*a[j] for j in range(3)]) for i in range(3)]
    for i in range(3): a[i] = b[i]
        
def Rotate_X(a, phi):
    R = [[1,0,0],[0,math.cos(phi),-math.sin(phi)],[0, math.sin(phi),math.cos(phi),0]]
    b = [sum([R[j][i]*a[j] for j in range(3)]) for i in range(3)]
    for i in range(3): a[i] = b[i]

def Rotate_Y(a, phi):
    R = [[math.cos(phi),0,-math.sin(phi)],[0,1,0],[math.sin(phi),0,math.cos(phi)]]
    b = [sum([R[j][i]*a[j] for j in range(3)]) for i in range(3)]
    for i in range(3): a[i] = b[i]
    
def Mirror(a,ms):    
    for i in ms:
      a[i]= -a[i]
    
def Cycle(a,cp):    
    if a[cp[0]]> cp[1]+1e-5:
       a[cp[0]]-= cp[2] 
    if a[cp[0]]< -cp[1]-1e-5:
       a[cp[0]]+= cp[2] 
    
def compare_using_symmetries(A,B, syms):
  if len(A)!=len(B): 
    print "Error"
    return False
    
  lists = [syms[i][1] for i in range(len(syms))]
  ps = itertools.product(*lists) 

  for p in ps:
    Acp = deepcopy(A) #make a copy of A
    #apply transformations to all vectors in the copy of A  
    for a in Acp:
      for i in range(len(syms)):    
         syms[i][0](a,p[i])
    #check if copy of A same as B
    same = True
    for j in range(len(B)):
      for i in range(3):
        if abs(Acp[j][i]-B[j][i])>1e-5:
          same=False
          break
      if not same: break
    
    if same: 
      #print "FOUND:"  
      #print "   p:",p
      #print "   A:",A, "   Acp:",Acp,  "   B:",B
      return True
  return False


#---------------------------- get equivalent links for a solid cluster ----------------------------------#
def get_identical_pair_sets(Nx,Ny):
    N=Nx*Ny
        
    def get_xyz(i):
      x = i%Nx - (Nx-1)/2.0
      y = i/Nx - (Ny-1)/2.0
      z = 0
      return [x, y, z]

    #embed the cluster in a physical space

    #always include neutral transformation
    #dsx = range(0,Nx)
    #dsy = range(0,Ny) 
    #phis = [0.0, 0.5*math.pi, math.pi, 1.5*math.pi]
    #ms = [[],[0],[1]] #mirrorx*mirrory is just rotation for pi, so is not needed here
    #cx = [0,0.5*(Nx-1),1.0*Nx]
    #cy = [1,0.5*(Ny-1),1.0*Ny]

    #make a list of symmetry operations

    #cyclic square cluster
    #syms = [ [ Rotate_Z, phis ],
    #         [ Mirror, ms ],
    #         [ Translate_X, dsx ],
    #         [ Translate_Y, dsy ],
    #         [ Cycle, [cx] ],
    #         [ Cycle, [cy] ]
    #       ]
    
    if Nx==Ny:
        #square but non-cyclic
        phis = [0.0, 0.5*math.pi, math.pi, 1.5*math.pi]
        ms = [[],[0],[1]] #mirrorx*mirrory is just rotation for pi, so is not needed here
    
        syms = [ [Rotate_Z,phis],
                 [Mirror,ms]]
    else:        
        #non-square
        phis = [0.0, math.pi]
        ms = [[],[0],[1]] #mirrorx*mirrory is just rotation for pi, so is not needed here
    
        syms = [ [Rotate_Z,phis],
                 [Mirror,ms]]

    #GET PAIRS
    #general way to do it
    identical_pair_sets = []    
    for i in range(N):
      #print "Working i: ",i
      for j in range(N): 
         #print "Working i,j: ",i,",",j
         found=False
         for k in range(len(identical_pair_sets)):
            k0 = identical_pair_sets[k][0][0]
            k1 = identical_pair_sets[k][0][1]
            A = [get_xyz(i),get_xyz(j)]
            B = [get_xyz(k0),get_xyz(k1)]
            if compare_using_symmetries(A,B, syms):
                identical_pair_sets[k].append([i,j]) 
                found = True
                break
         if not found:
            identical_pair_sets.append([]);
            identical_pair_sets[len(identical_pair_sets)-1].append([i,j]);

    #print "len: ",len(identical_pair_sets)
    
    for k1 in range(len(identical_pair_sets)-1):        
        for k2 in range(k1+1,len(identical_pair_sets),1):            
            #print "k1, k2: ", k1, k2
            if identical_pair_sets[k2] == []: continue
            if [identical_pair_sets[k2][0][1],identical_pair_sets[k2][0][0]] in identical_pair_sets[k1]:
                identical_pair_sets[k1] += deepcopy(identical_pair_sets[k2])
                identical_pair_sets[k2] = []

    identical_pair_sets = [ips for ips in identical_pair_sets if ips != []]
        
    #for k in range(len(identical_pair_sets)):
    #    print  identical_pair_sets[k]#[0] 
    return identical_pair_sets

####################################################################################################################

#--------------------------------------------- clsuter -------------------------------------------------------------#
class cluster:
    #if cluster is not solid, it must preserve symmetries of the included links. 
    #if a single link (only 2 sites) is included, nothing to worry about
    def __init__(self, x,y,Lx,Ly=None, sites_contained = [], prefactor = None): 
        #empty sites_contained means all sites are contained
        #otherwise, specify all sites with coordinates relative to x,y
        self.x = copy(x)
        self.y = copy(y)
        self.Lx = copy(Lx)
        self.Ly = (copy(Ly) if (not (Ly is None)) else copy(Lx))
        self.sites_contained = sorted(sites_contained, key=itemgetter(0,1))
        self.already_used = []
        self.prefactor = prefactor 

    @classmethod
    def from_abs_sites(cls, abs_sites_contained): #provide sites contained 
        if abs_sites_contained == []: return cls(0,0,0,0,[])
        xs = [asc[0] for asc in abs_sites_contained]
        ys = [asc[1] for asc in abs_sites_contained]
        minx = min(xs)
        miny = min(ys)
        maxx = max(xs)
        maxy = max(ys)
        Lx = maxx - minx + 1
        Ly = maxy - miny + 1
        sites_contained = deepcopy(abs_sites_contained)
        if len(sites_contained) == Lx*Ly:
            sites_contained = []
        else:
            for sc in sites_contained:
                sc[0] -= minx
                sc[1] -= miny
        return cls(minx,miny,Lx,Ly,sites_contained)    
        
    def __str__(self):
        return "x: %s y: %s Lx: %s Ly: %s %s %s\n"%(self.x, self.y, self.Lx, self.Ly, 
                                                 "" if (self.sites_contained==[]) else self.sites_contained,
                                                 "" if self.prefactor is None else "pref: %s"%self.prefactor)
    def __repr__(self):
        return self.__str__()
    
    def make_zero(self):
        self.x = 0
        self.y = 0
        self.Lx = 0
        self.Ly = 0
        self.sites_contained = []
        self.already_used = []
        
    
    def is_zero(self):
        return (self.Lx==0) or (self.Ly==0)
    
    def encompasses(self,x,y): #x,y are absolute
        if self.sites_contained==[]:
            return ((x>=self.x) and (x<self.x+self.Lx) and (y>=self.y) and (y<self.y+self.Ly))    
        else:
            return ([x-self.x,y-self.y] in self.sites_contained)
    
    def is_LxLy_ordered(self):
        return (self.Lx>=self.Ly)
    
    def is_00_ordered(self):
        if self.sites_contained == []: return True
        return (([0,0] in self.sites_contained) or self.sites_contained==[])
        
    def is_totalxy_ordered(self, axis):
        if self.sites_contained == []: return True
        sumx = sum([s[0] for s in self.sites_contained])
        sumy = sum([s[1] for s in self.sites_contained])
        test_sites = deepcopy(self.sites_contained)
        mirror_sites_and_shift_to_positive(test_sites,axis=axis)
        if not ([0,0] in test_sites): return True
        test_sumx = sum([s[0] for s in test_sites])
        test_sumy = sum([s[1] for s in test_sites])        
        return (test_sumx<=sumx) and (test_sumy>=sumy)
        
    def is_ordered(self):
        return self.is_LxLy_ordered()\
               and self.is_00_ordered()\
               and self.is_totalxy_ordered(axis=[0])\
               and self.is_totalxy_ordered(axis=[1])
    
    def get_ordered(self, point=None):     #point is auxiliary variable - get's transformed together with sites_contained
        #print "get_ordered: self.Lx: ",self.Lx," self.Ly: ",self.Ly
        ordered_sites_contained = deepcopy(self.sites_contained)
        maxes = [[0,1],[0],[1]]
        lx,ly = copy(self.Lx), copy(self.Ly)
        x,y = copy(self.x),copy(self.y)
        if not self.is_00_ordered():
            #print "not 00 ordered!"
            mirror_sites_and_shift_to_positive(ordered_sites_contained,axis=[0])
            #maxes = [[1]] #if it is necessary to mirror x to get [0,0], don't mirro back to satisfy max total x
            if not (point is None): point[0] = 2*x+lx-point[0]-1
        #print ordered_sites_contained        
        if not self.is_LxLy_ordered(): 
            #print "not ordered!"      
            #print "not LxLy ordered!"
            for sc in ordered_sites_contained:
                sc[0],sc[1] = sc[1],sc[0]
            lx,ly = ly,lx
            x,y = y,x
            if not (point is None): point[0],point[1] = point[1],point[0]
        for ma in maxes:                
            #assert (ma==0) or (ma==1), "we're in 2D"
            if not cluster(0,0,lx,ly, ordered_sites_contained).is_totalxy_ordered(axis=ma):
                #print "not totalxy ordered!", ma
                mirror_sites_and_shift_to_positive(ordered_sites_contained,axis=ma)
                #print ordered_sites_contained    
                if not (point is None):
                    if 0 in ma: 
                        L = lx
                        r = x
                        point[0] = 2*r+L-point[0]-1
                    if 1 in ma: 
                        L = ly
                        r = y
                        point[1] = 2*r+L-point[1]-1
                    
        return cluster(x,y,lx,ly, ordered_sites_contained)
    
    def label_within(self,x0,y0):
        #print "x0,y0: ", x0, y0, "self.x, self.y: ",self.x,self.y
        counter = 0
        found = False
        for y in range(self.Ly): #y and x are relative to self.x and self.y
            for x in range(self.Lx):
                if (self.sites_contained==[]) or ([x,y] in self.sites_contained):
                    if (self.x+x == x0) and (self.y+y==y0): 
                        found = True
                        break            
                    counter +=1
            if found: break                           
        #print "counter: ",counter        
        if not found: return -1
        else: return counter

    def label_within_solid(self, i):
        if self.sites_contained==[]: return copy(i)
        assert len(self.sites_contained)>i, "there's not so many sites"
        x0 = self.sites_contained[i][0]
        y0 = self.sites_contained[i][1]
        return cluster(0,0,self.Lx,self.Ly, sites_contained=[]).label_within(x0,y0)
    
    def label_within_non_solid(self, i):
        if self.sites_contained == []: return copy(i)
        x0 = i % self.Lx
        y0 = i / self.Lx
        for l in range(len(self.sites_contained)):
            if self.sites_contained[l] == [x0,y0]: return l
        return -1
        
    def label_within_ordered(self,x0,y0):
        point = [copy(x0), copy(y0)]
        #print "point before ordering: ", point
        B = self.get_ordered(point=point)
        #print "point after ordering: ", point
        return B.label_within(point[0],point[1])            

    def get_abs_sites_contained(self):
        if self.sites_contained != []:
            abs_sites_contained = [[sc[0]+self.x,sc[1]+self.y] for sc in self.sites_contained]
        else:
            abs_sites_contained = []
            for y in range(self.Ly): #y and x are relative to self.x and self.y
                for x in range(self.Lx):
                    abs_sites_contained.append([x+self.x,y+self.y])
        return abs_sites_contained                                                          
    
    def is_supercluster_to(self, A):
        return       (A.x>=self.x)\
                and  (A.y>=self.y)\
                and  (A.x+A.Lx<=self.x + self.Lx)\
                and  (A.y+A.Ly<=self.y + self.Ly)

#--------------------------------------------- contribution -------------------------------------------------------------#

class contribution():
    def __init__(self,prefactor,i,j,Lx,Ly, sites_contained=[]):
        assert Lx>= Ly, "contribution must be ordered Lx: %s Ly: %s"%(Lx,Ly)
        self.prefactor = copy(prefactor)
        self.i = copy(i)
        self.j = copy(j)
        self.Lx = copy(Lx)
        self.Ly = copy(Ly)
        self.sites_contained = sorted(sites_contained, key=itemgetter(0,1))         
   
    def name(self):
        if self.sites_contained==[]:
            return "%sx%s"%(self.Lx, self.Ly)
        else:
            s = "%s-"%len(self.sites_contained)
            for sc in self.sites_contained:
                if sc != [0,0]:
                    s += "(%s,%s)"%(sc[0],sc[1])
            return s        
                
    def N_sites(self):
        if self.sites_contained==[]:
            return self.Lx*self.Ly
        else:
            return len(self.sites_contained)
    
    def get_tex(self):
        if self.prefactor == 0: return ""
        if abs(self.prefactor)!=1: absp = "%s"%abs(self.prefactor)
        else: absp = ""    
        if self.prefactor<0: sgnp = "-"
        else: sgnp = "+"
            
        if self.sites_contained==[]:
            spscr ="\mathrm{imp}(%s\\times %s)"%(self.Lx, self.Ly)
        else:
            spscr = "\mathrm{imp}"
            for s in self.sites_contained:
                spscr += "(%s,%s)"%(s[0],s[1])
        return "%s%s\Sigma_{%s%s}^{%s}"%(sgnp,absp,self.i, self.j, spscr )
        
    def __str__(self):
        tex = self.get_tex()
        if tex=="": return ""
        else:
          return r"$%s$"%tex
    def __repr__(self):
        return self.__str__()
    
    def is_identical_to(self,c):
        return     (self.Lx==c.Lx) and (self.Ly==c.Ly)\
               and (self.i==c.i) and (self.j==c.j)\
               and (self.sites_contained == c.sites_contained)

#--------------------------------------------- routines -------------------------------------------------------------#
               
def get_unique_non_zero_contributions(contributions):
    unzcs = []
    for c in contributions:
        if c.prefactor == 0: continue
        found = False
        for c2 in unzcs:
            if c.is_identical_to(c2): 
                found = True
                break  
        if not found: 
            unzcs.append(c)
    return unzcs

def clean_zero(clusters):    
    clusters[:] = [c for c in clusters if not (c.is_zero())]        
        
def clean_identical(clusters):    
    for i in range(len(clusters)-1):
        for j in range(i+1,len(clusters)):
            if are_identical(clusters[i],clusters[j]):
                clusters[j].make_zero()                
    clean_zero(clusters)    
    
def clean_subclusters(clusters):
    for i in range(len(clusters)):
        for j in range(len(clusters)):
            if i==j: continue
            if clusters[i].is_supercluster_to(clusters[j]):
                clusters[j].make_zero()                
    clean_zero(clusters)    
    
def shift_sites_to_positive(sites):
    xs = [s[0] for s in sites]
    ys = [s[1] for s in sites]
    minx = min(xs)
    miny = min(ys)
    sites[:] = [[s[0]-minx,s[1]-miny] for s in sites]
    
def rotate_sites(sites, phi):
    for rsc in sites:
        Rotate_2D(rsc,phi)
        #print "rsc before: ", rsc
        rsc[0]=int(round(rsc[0]))
        rsc[1]=int(round(rsc[1]))
        #print "rsc after: ", rsc   
        
def rotate_sites_and_shift_to_positive(sites,phi):
    rotate_sites(sites, phi)
    shift_sites_to_positive(sites)
    
def mirror_sites(sites, axis=[0]):
    for rsc in sites:
        Mirror(rsc,axis)
        
def mirror_sites_and_shift_to_positive(sites,axis=[0]):
    mirror_sites(sites, axis)
    shift_sites_to_positive(sites)    

def are_identical(A,B):
    return    (A.x == B.x)\
           and(A.y == B.y)\
           and(A.Lx == B.Lx)\
           and(A.Ly == B.Ly)\
           and(A.sites_contained == B.sites_contained)     
    
def are_equivalent(A,B): #can be used also for contributions   
    if A.__class__ == cluster:        
        a = A.get_ordered()
    else: a=A    
    if B.__class__ == cluster:        
        b = B.get_ordered()
    else: b=B    
    return    (a.Lx == b.Lx)\
           and(a.Ly == b.Ly)\
           and(a.sites_contained == b.sites_contained) 

def contrib_to_cluster(C):
    return cluster(0,0,C.Lx,C.Ly,C.sites_contained,C.prefactor)

def contribs_to_clusters(contribs):       
    return [cluster(0,0,C.Lx,C.Ly,C.sites_contained,C.prefactor) for C in contribs]   

def get_coords_from_label(C,i):
    if C.sites_contained==[]:
        return [C.x + i % C.Lx, C.y + i / C.Lx]     
    else:
        return [ C.x + sites_contained[i][0], C.y + sites_contained[i][1] ]

def get_rel_coords_from_label(C,i):
    if C.sites_contained==[]:
        return [i % C.Lx,  i / C.Lx]     
    else:
        return [ sites_contained[i][0], sites_contained[i][1] ]   
    
def get_overlap(A,B):
    if A.sites_contained!=[] or B.sites_contained!=[]:
        Asites = A.get_abs_sites_contained()
        Bsites = B.get_abs_sites_contained()
        overlap_sites = [a for a in Asites if a in Bsites]
        return cluster.from_abs_sites(overlap_sites)
    else:   
        x = max(A.x,B.x)
        y = max(A.y,B.y)
        Lx = min(A.x+A.Lx,B.x+B.Lx)-x
        if Lx<0: Lx=0
        Ly = min(A.y+A.Ly,B.y+B.Ly)-y
        if Ly<0: Ly=0            
        return cluster(x,y,Lx,Ly)

#   square (Lx=Ly) | solid (sites_contained=[])  | rotations
#        X         |            X                | 0
#                  |            X                | 0,pi/2
#        X         |                             | 0, pi/2, pi, 3/pi/2, mirror
#                  |                             | 0, pi, and specially treated pi/2, 3pi/2 + mirror
    
    
def get_all_shifts_and_rotations(x1,y1,A):
    #here we assume x0=y0=0         

    cluster_list = []         
    # all shifts + all shifts_rotated by pi
    if A.sites_contained!=[]:  
        phis = [0,pi/2,pi,3*pi/2]
        mirror_axes = [[],[0],[1]]
    else:    
        if A.Lx==A.Ly:
          phis = [0]
        else:
          phis = [0,pi/2]  
        mirror_axes = [[]]
    
    for ma in mirror_axes:    
        mirrored_sites = deepcopy(A.sites_contained)        
        if ma!=[]:
            mirror_sites_and_shift_to_positive(mirrored_sites)
        for phi in phis:        
            rotated_sites = deepcopy(mirrored_sites)
            if phi!=0 and rotated_sites!=[]:                
                rotate_sites_and_shift_to_positive(rotated_sites,phi)         
            if phi==pi/2 or phi==3*pi/2:    
                lx, ly = copy(A.Ly),copy(A.Lx)
            else:
                lx, ly = copy(A.Lx),copy(A.Ly)
            for x in range(-lx+1,1):
                for y in range(-ly+1,1):
                    B = cluster(x,y,lx,ly, rotated_sites)
                    #in case cluster is not solid, we need to check for 0,0 as well
                    if B.encompasses(x1,y1) and B.encompasses(0,0): 
                        B.prefactor = 1
                        cluster_list.append(B)
    clean_identical(cluster_list)                    
    return cluster_list

def get_all_overlaps(cluster_list, mask=None):
    n = len(cluster_list)
    #print "cluster list n:",n
    overlap_list = []
    for j in range(1,n):
        for i in range(0,j):        
            #print "cluster list i,j:",i,j
            C = get_overlap(cluster_list[i],cluster_list[j])
            if mask!=None:
                C = get_overlap(C,mask)
            if not C.is_zero():
                #print "found overlap with prefactor -1"
                C.prefactor = -1
                overlap_list.append(C)
            for au in cluster_list[i].already_used:       
                D = get_overlap(C,au)    
                if not D.is_zero():
                    #print "found overlap with prefactor 1"
                    D.prefactor = 1
                    overlap_list.append(D)                   
            overlap_list.extend(get_all_overlaps(cluster_list[i].already_used, mask = deepcopy(C)))       
            if not C.is_zero():  
                is_already_used = False
                for c in cluster_list[j].already_used:
                    if are_identical(C,c):
                        is_already_used = True
                if not is_already_used: cluster_list[j].already_used.append(deepcopy(C))
            #print "already_used: ", cluster_list[j].already_used        
            clean_subclusters(cluster_list[j].already_used)             
            #print "already_used after cleaning: ", cluster_list[j].already_used        
    return overlap_list

def get_contributions_from_all_subclusters(x1, y1, shifts_and_overlaps):
    contributions = []
    for A in shifts_and_overlaps:
        #find the indices of sites within ordered cluster (Lx>=Ly), here we assume symmetry x <-> y
        i = copy(A.label_within_ordered(0,0))
        j = copy(A.label_within_ordered(x1,y1) )
        #print A, "i: ", i, "j: ", j, "x1,y1: ",x1, y1
        assert (i>=0) and (j>=0), "must be found in cluster"
        #now we use the lowest i index by applying cluster symmetries
        B = A.get_ordered()                
        assert B.is_ordered(), "B is not ordered, something fishy: A:"+str(A)+" B: "+str(B)
        #print "B.Lx: ",B.Lx," B.Ly: ",B.Ly
        #H_0 = initCubicTBH(copy(B.Lx), copy(B.Ly), 1, -1, -0.25, cyclic=False)
        #print H_0
        
        ips = get_identical_pair_sets(copy(B.Lx), copy(B.Ly))        
        #print ips
        found = False
        for ip in ips:
            if [B.label_within_solid(i),B.label_within_solid(j)] in ip:
                i,j = B.label_within_non_solid(ip[0][0]),B.label_within_non_solid(ip[0][1])
                assert (i!=-1) and (j!=-1), "must be found in the cluster"
                found = True
        assert found, "something fishy, should be in pair list"        
        #add contribution
        #if A in overlaps: pref = -1
        #if A in shifts: pref = 1
        found = False        
        for c in contributions:
            if are_equivalent(c,B) and c.i==i and c.j==j:
                c.prefactor += A.prefactor
                found = True
                break
        if not found: 
            contributions.append(contribution(A.prefactor,i,j,B.Lx, B.Ly, B.sites_contained))       
    return get_unique_non_zero_contributions(contributions)      

def get_all_contributions(x1,y1, clusters):    
    shifts = []
    for cluster in clusters:
      shifts.extend(get_all_shifts_and_rotations(x1,y1, cluster))
    overlaps = get_all_overlaps(shifts)
    #print overlaps
    return get_contributions_from_all_subclusters(x1, y1, shifts+overlaps)

def is_contained_in_any(C, clusters):
    for c in clusters:
        if c.is_supercluster_to(C): return True
    return False

def smart_get_all_contributions(x1,y1, clusters):     
    maxLx, maxLy = 0, 0
    for C in clusters:
        if C.Lx > maxLx: maxLx = C.Lx
        if C.Ly > maxLy: maxLy = C.Ly
    assert maxLx>=maxLy, "please order your clusters"        
    #maxL = max(maxLx, maxLy)            
    
    shifts_per_LxLy = []
    for Lx in reversed(range(1,maxLx+1)):
        for Ly in reversed(range(1,Lx+1)):  
            C = cluster(0,0,Lx,Ly)
            if not is_contained_in_any(C, clusters): continue
            all_shifts = get_all_shifts_and_rotations(x1,y1,C)
            shifts_per_LxLy.append(all_shifts)            

    NLxLys = len(shifts_per_LxLy)
    for l in range(1,NLxLys):
        for C in shifts_per_LxLy[l]:
            total_prefactor = 0        
            for lp in range(l):
                for D in shifts_per_LxLy[lp]:
                    if D.is_supercluster_to(C):
                        total_prefactor+=D.prefactor        
            C.prefactor = 1-total_prefactor

    shifts_and_overlaps = []
    for shifts in shifts_per_LxLy:
        shifts_and_overlaps.extend(shifts)

    return get_contributions_from_all_subclusters(x1,y1, shifts_and_overlaps)

#############################################################################################################################
            
#--------------------------------------------- nested_structure -------------------------------------------------------------#                

class nested_struct:
    def __init__(self,clusters, use_smart_algorithm = False):
        self.all_contribs = []
        self.contribs = {}
        maxLx = 0
        for c in clusters:
            if c.Lx > maxLx:
                maxLx = copy(c.Lx)        
        self.maxLx = maxLx
        for x in range(maxLx):
            for y in range(0,x+1):                
                if use_smart_algorithm:
                  c = smart_get_all_contributions(x,y,clusters)
                else:
                  c = get_all_contributions(x,y,clusters)
                #print "x,y:",c
                self.contribs["%s|%s"%(x,y)] = c
                self.all_contribs.extend(c)                
        #print self.all_contribs

    def print_to_file(self, archive_name):
        A=HDFArchive(archive_name,"w")        
        allctrbs = []
        for c in self.all_contribs:
          allctrbs.append(c.__dict__)
        A['all_contribs'] = allctrbs
        ctrbs = {}
        for C in self.contribs.keys():
           ctrbsC = []
           for c in self.contribs[C]:
             ctrbsC.append(c.__dict__)
           ctrbs[C] = ctrbsC                            
        A['contribs'] = ctrbs
        A['maxLx'] = self.maxLx
        del A

    @classmethod  
    def from_file(cls, archive_name):
        ns = cls([])
        A=HDFArchive(archive_name,"r")        
        allctrbs = A['all_contribs']
        ns.all_contribs = []
        for cdict in allctrbs:
          ns.all_contribs.append(contribution(**cdict))

        ctrbs = A['contribs']
        ns.contribs = {}
        for C in ctrbs.keys():
           ns.contribs[C] = []
           for cdict in ctrbs[C]:
             ns.contribs[C].append(contribution(**cdict))
        ns.maxLx = A['maxLx']
        del A
        return ns


    
    def get_tex(self):
        lbl = ""
        print "self.maxLx: ", self.maxLx
        for x in range(self.maxLx):
            for y in range(0,x+1):                
                key = "%s|%s"%(x,y)
                if key in self.contribs.keys(): 
                    lbl += "\Sigma^\mathrm{latt}_{\mathbf{r}=(%s,%s)} &=& "%(x,y)                
                    for c in self.contribs[key]:                         
                        lbl += c.get_tex()
                    lbl += "\\\\ \\nonumber \n"    
        return lbl
    
    def set_nk(self,nk):
        assert nk % 2==0, "works only for even nk"  
        self.nk = copy(nk)
    
    def get_impurity_struct(self):
        impurity_struct = {}
        for c in self.all_contribs:
            impurity_struct[c.name()] = range(c.N_sites())
        return impurity_struct    
    
    def get_latt_to_imp_mapping(self):
        return (lambda x,y: self.latt_to_imp_mapping(x,y) )
    
    def latt_to_imp_mapping(self, x,y):
        X,Y = abs(x),abs(y)
        if X>self.nk/2+1: X = self.nk - X
        if Y>self.nk/2+1: Y = self.nk - Y
        if Y>X: X,Y = Y,X ## careful here with the order of commands!
        mps = []
        key = "%s|%s"%(X,Y)
        if key in self.contribs.keys(): 
            for c in self.contribs[key]:                         
                mps.append( [c.name(), c.prefactor, c.i, c.j] )
        return mps
    
    def get_imp_to_latt_mapping(self):
        return (lambda C,i,j: self.imp_to_latt_mapping(C,i,j) )
    
    def imp_to_latt_mapping(self,C,i,j):
        for c in self.all_contribs:
            if c.name() == C: break
        if c.sites_contained == []:            
            rx1 = i % c.Lx
            ry1 = i / c.Lx
            rx2 = j % c.Lx
            ry2 = j / c.Lx            
        else:
            rx1 = c.sites_contained[i][0]
            ry1 = c.sites_contained[i][1]
            rx2 = c.sites_contained[j][0]
            ry2 = c.sites_contained[j][1]
        return [rx1-rx2,ry1-ry2] 
    
    def get_identical_pairs(self):
        identical_pairs = {}
        for c in self.all_contribs:
            if c.name() in identical_pairs.keys(): continue
            ipss = get_identical_pair_sets(c.Lx,c.Ly)
            if c.sites_contained != []:
                C = cluster(0,0, c.Lx, c.Ly, sites_contained = c.sites_contained) 
                for ips in ipss:
                    for ip in ips:
                        ip[0] = C.label_within_non_solid(ip[0]) 
                        ip[1] = C.label_within_non_solid(ip[1])             
            identical_pairs[c.name()] = ipss
        return identical_pairs

