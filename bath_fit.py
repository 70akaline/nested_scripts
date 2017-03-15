import random

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
def G_from_params(G, P, epss, V):
    nw,Nsites,dummy = numpy.shape(G.data)
    iws = [iw for iw in G.mesh]
    Nstates = len(epss)
    assert numpy.shape(V) == (Nsites,Nstates), "something wrong "+str(V)+str(epss)
    G << 0.0
    for a in range(Nsites):
        for b in range(Nsites):
            for c,eps in enumerate(epss):
                 G.data[:,a,b] += P * V[a,c] * numpy.conj(V[b,c]) / (iws[:]-eps)  
                    
def add_one_peak(G, k, W, V):
    nw,Nsites,dummy = numpy.shape(G.data)
    iws = [iw for iw in G.mesh]
    Nstates = len(W)
    assert numpy.shape(V) == (Nsites,Nstates), "something wrong"+str(V)    
    for a in range(Nsites):
        for b in range(Nsites):
            for c,eps in enumerate(W):
                 G.data[:,a,b] = (k*G.data[:,a,b] + V[a,c] * numpy.conj(V[b,c]) / (iws[:]-eps) )/(k+1)
                    

          
def ph_symm_V_from_params(epss, Vs, kinds, Nsites=4):    
    Nstates = len(epss)
    assert len(epss) == len(Vs), "something wrong"
    assert len(epss) == len(kinds), "something wrong"
    
    def forms_from_kind(kind):
        if Nsites==4:
            if kind==0:
                return [1,1,1,1], [1,-1,-1,1]
            elif kind==1:
                return [1,1,-1,-1], [1,-1,1,-1]
            else: assert False, "something wrong"
        elif Nsites==2:
            return [1,1],[1,-1]
        elif Nsites==1:
            return [1],[1]
    
    
    V = []
    W = []
    
    for i, eps, v, kind in zip(range(Nstates),epss,Vs,kinds):
        W.extend([eps,-eps])
        if kind==1 and Nsites==4: W.extend([-eps,eps])
        form1,form2 = forms_from_kind(kind)
        for l in ( range(kind+1) if (Nsites==4) else range(1) ):
            V.append(v*numpy.array(form1))
            V.append(v*numpy.array(form2))
        
    W = numpy.array(W)
    V = numpy.transpose(numpy.array(V))
    #print "W:",W
    #print "V:",V
    Ntot = (((Nstates+sum(kinds))*2) if (Nsites==4) else (2*Nstates))
    
    assert numpy.shape(V) == (Nsites,Ntot), "something wrong"+str(Ntot)+str(Nstates)
    assert len(W) == Ntot, "something wrong Ntot: "+str(Ntot)+" W: "+str(W)+" sh: "+str(numpy.shape(W))
    return W,V


        
def diff(G1, G2, n_mats, alpha = 1.0):
    nw,Nsites,dummy = numpy.shape(G1.data)
    ws = [iw.imag for iw in G1.mesh]
    assert numpy.shape(G1.data) == numpy.shape(G2.data), "something wrong"
   
    G1inv = deepcopy(G1.data)
    G2inv = deepcopy(G2.data)
    try:
        for wi in range(nw/2,nw/2+n_mats):
            G1inv[wi,:,:] = numpy.linalg.inv(G1inv[wi,:,:])
            G2inv[wi,:,:] = numpy.linalg.inv(G2inv[wi,:,:])
    except:
        print G2inv[wi,:,:]
        raise
    
    return (sum([numpy.sum(numpy.abs(G1inv[wi,:,:]-G2inv[wi,:,:]))/numpy.abs(ws[wi])**alpha for wi in range(nw/2,nw/2+n_mats)]))

def stochastic_minimize(func, x0, T=0.01, max_its=2000, printout=None, Nit_print=2000):    
    Nc = len(x0)
    f_old = func(x0)
    
    accepted=0.
    avg_diff=0.
    for it in range(1,max_its+1):
        x = deepcopy(x0)
        c = random.randint(0,Nc-1)
        sgn = (-1)**random.randint(0,1)
        x[c] *= sgn*(0.95+0.1*random.random())

        f_new = func(x)
        
        avg_diff += abs(f_new-f_old)
        
        if f_new < f_old:
            x0[:] = x
            f_old = f_new
            accepted+=1
        else:
            r = exp(-(f_new-f_old)/T)
            if random.random()<r:
                x0[:] = x
                f_old = f_new
        if it % Nit_print == 0:
            print it, ":", f_old 
            print "acc_ratio: ", accepted/it            
            if not (printout is None): printout()
            #T=avg_diff/200
            print "T: ", T
            avg_diff=0
            
def stochastic_complete(Delta, Delta_init, V_range, eps_range, accepted_init,
                        T = 0.01, max_its=2000, n_mats = 30, printout=None, Nit_print=2000):        
    Nsites = numpy.shape(Delta.data)[1]
    accepted=deepcopy(accepted_init)
    f_old = diff(Delta, Delta_init, n_mats)   
    print "f_old:", f_old
    
    kinds = ( [0,1] if Nsites==4 else [0] )
    npeaks = ( 2 if Nsites==4 else 1 )
    for it in range(1,max_its+1):        
        Vs = []
        epss = []
        for X,the_range in zip([epss,Vs],[eps_range,V_range]):
            mn = the_range[0]
            rng = the_range[1]-the_range[0]
            X.extend([mn+random.random()*rng for l in range(npeaks) ])
            
        W,V = ph_symm_V_from_params(epss, Vs, kinds, Nsites)
        
        V *= accepted
        #print W,V
            
        Delta_temp = Delta_init.copy()
        
        #print "accepted: ", accepted
        add_one_peak(Delta_temp, accepted, W, V)
        f_new = diff(Delta, Delta_temp, n_mats)                                   
        #print "f_new:", f_new
        #print "f_diff: ", f_new-f_old
        #for i in [0,1,3]:
        #        title("0 %s"%i)
        #        oplot(Delta_temp[0,i])
        #        oplot(Delta[0,i])
        #        show()
        
            
        do_accept = False
        if f_new < f_old:
            do_accept = True
        else:
            r = exp(-(f_new-f_old)/T)
            if random.random()<r: do_accept = True
        if do_accept:
            Delta_init << Delta_temp
            f_old = f_new
            accepted+=1        
        if it % Nit_print == 0:            
            print it, ":", f_old 
            print "acc_ratio: ", (accepted-accepted_init)/(it+0.)
            print "total accepted: ",accepted-accepted_init
            if not (printout is None): printout()
            #T=avg_diff/200
            print "T: ", T
            avg_diff=0            
        

def fit(G, epss_init, Vs_init, kinds_init, P_init=1.0, n_mats = 5, max_its = 100, T=0.1 ):
    nw,Nsites,dummy = numpy.shape(G.data)
    assert Nsites == dummy, "something wrong"
    
    print "Nsites: ", Nsites
    
    if P_init is None: P = 1.0
    else: P = P_init
    print "P: ", P    
        
    Nstates = len(epss_init)
    assert len(Vs_init)== Nstates
    assert len(kinds_init)== Nstates
    print "Nstates: ", Nstates
    
    def x_from_epssVs(epss,Vs):
        x = list(epss) + list(Vs)
        return x    
    
    def epssVs_from_x(x):
        epss = x[:Nstates]
        Vs = x[Nstates:]
        return epss, Vs    
    
    def func(x):
        #assert len(x) == 1 + (2.0*Nsites+1)*(Nstates), "wrong size array"
        assert len(x) == 2*Nstates, "wrong size array"
        epss, Vs = epssVs_from_x(x)
        W, V = ph_symm_V_from_params(epss, Vs, kinds, Nsites)
        Gtemp = G.copy()
        G_from_params(Gtemp, P, W, V)        
        d = diff(G, Gtemp, n_mats)           
        return d

    def printout():
        #print "P: ", P
        epss, Vs = epssVs_from_x(x0)
        print "epss: ", epss
        print "Vs: ", Vs

    x0 = x_from_epssVs(epss_init,Vs_init)
           
    #minimize(func, x0, method='Nelder-Mead', options = {'maxiter': 50})
    print "------------ before" 
    print "x0: ", x0
    printout()    
    stochastic_minimize(func, x0, max_its=max_its, printout=printout, T=T) 
    
    return epssVs_from_x(x0)

from tail_fitters import fit_fermionic_gf_tail
def ph_symmetric_Gweiss_causal_cautionary( data, ntau=5000):   
  assert sorted(data.impurity_struct.keys()) == ['1x1','2x1','2x2'], "not applicable to any other scheme than nested plaquette. impurity_struct: "+str(data.impurity_struct.keys())
  print "######### Checking causality of Gweiss ########"

  data.Gweiss_iw_unfit = data.Gweiss_iw.copy()
  data.Delta_iw = data.Gweiss_iw.copy()
  data.Delta_iw_fit = data.Gweiss_iw.copy()    
  
  gs = []
  for C in data.impurity_struct.keys(): 
      gs.append ( GfImTime(indices = data.impurity_struct[C], beta = data.beta, n_points =ntau, statistic = 'Fermion') )     
  data.Delta_tau = BlockGf(name_list = self.impurity_struct.keys(), block_list = gs, make_copies = False)
  data.Delta_tau_fit = data.Delta_tau.copy()
    
  for C, g in data.Gweiss_iw:
    print ">>>>>>>>>>>>>>>>>> block ",C

    nsites = len(g.data[0,:,0])
    if nsites==4: Nx=Ny=2
    elif nsites==2: Nx,Ny=2,1
    elif nsites==1: Nx=Ny=1
    else: assert False, "wrong number of sites"
 
    H0 = initCubicTBH(Nx, Ny, 1, 0, -0.25, cyclic=False)
    print H0
    data.Delta_iw[C] = get_Delta_from_Gweiss_and_H0(g, H0, data.mus['up'])
   
    for i in range(nsites):
      data.Delta_iw[C].data[:,i,i] -= data.Delta_iw[C].data[:,i,i].real    
    if nsites>1:
      for i,j in ([(0,1)] if nsites==2 else [(0,1),(1,3),(3,2),(2,0)]):        
          data.Delta_iw[C].data[:,i,j] -= data.Delta_iw[C].data[:,i,j].imag    
          data.Delta_iw[C].data[:,j,i] -= data.Delta_iw[C].data[:,j,i].imag  
    if nsites>2:
      for i,j in [(0,3),(1,2)]:        
          data.Delta_iw[C].data[:,i,j] -= data.Delta_iw[C].data[:,i,j].real    
          data.Delta_iw[C].data[:,j,i] -= data.Delta_iw[C].data[:,j,i].real      
        
    fit_fermionic_sigma_tail(data.Delta_iw[C])
    data.Delta_tau[C] << InverseFourier(data.Delta_iw[C])

    d2tauDelta = second_derivative(data.Delta_tau[C].data.real,data.beta/ntau)   
    ws = numpy.zeros((ntau-2,nsites))    
    needs_fixing = False
    for taui in range(0,ntau-2):        
        w,v = numpy.linalg.eig(d2tauDelta[taui,:,:].real)
        ws[taui,:] = sorted(w)
        if len([ww for ww in numpy.ravel(ws) if ww>0])>0: needs_fixing = True
    if needs_fixing:
      print "WARNING: Delta_iw[%s] unphysical and needs fixing!!!"

    P = 0.01
    if nsites==4:
      epss=  [0.19668414224046424, -0.242286049378347, -0.4851071850062886, 0.6083462424661663]
      Vs=  [1.111030853547017, 0.95763987021629182, 1.3705081047849843, 0.80333649203756208]
      kinds=[0,1,0,1]
    elif nsites<4:
      epss=  [0.19668414224046424, -0.242286049378347]
      Vs=  [1.111030853547017, 0.95763987021629182]
      kinds=[0,0]

    for T in [0.01,0.001]:    
        epss,Vs = fit(data.Delta_iw[C], epss, Vs, kinds, P, n_mats = 5, max_its = 10000 )

        W,V = ph_symm_V_from_params(epss, Vs, kinds, Nsites=4)
        
        G_from_params(data.Delta_iw_fit[C], P, W, V)
           
        epss.extend([1.0,1.0])    
        Vs.extend([0.001,0.001])    
        kinds.extend([0,1])
                   
    for T in [0.001]:
        stochastic_complete(data.Delta_iw[C], data.Delta_iw_fit[C], [-0.0001,0.0001], [-3.0,3.0], 10000,
                            T = T, max_its=10000, n_mats = 5, printout=None, Nit_print=1000)    

    data.Gweiss[C] = get_Gweiss_from_Delta_and_H0(data.Delta_iw_fit[C], H0, mus['up'])    

  print "#########           DONE!!!            ########"  
    
