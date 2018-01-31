from data_containers import nested_data
from data_containers import cumul_nested_data
from data_containers import cellular_data
from data_containers import dca_data
from data_containers import dca_plus_data
from data_containers import nested_edmft_data
from action_cautionaries import impose_real_valued_in_imtime_numpy
from action_cautionaries import impose_real_valued_in_imtime
from getters import *
from impurity_solvers import solvers

def set_n(n, data): 
  for key in data.ns.keys():
    data.ns[key] = n
  return n

def set_mu(mu, data):
  for key in data.mus.keys():
    data.mus[key] = mu

def is_zero(bg):
  return sum([ numpy.count_nonzero(g.data) for name, g in bg ]) == 0

#----------------------------- nested -----------------------------------------------------------------------#
def prepare_nested( data, nested_scheme, solver_class = solvers.ctint, flexible_Gweiss=False, sign=-1, sign_up_to=2, use_G_proj = False ):
  assert (data.__class__ == nested_data) or (data.__class__ == nested_edmft_data) , "wrong data type"
  assert data.fermionic_struct == {'up': [0]}, "wrong fermionic struct for this calcalation"
  assert data.impurity_struct == nested_scheme.get_impurity_struct(), "wrong impurity struct for this nested scheme" 

  data.get_Sigmaijw = lambda: full_fill_Sigmaijw_from_Sigma_imp_iw(data.Sigmaijw, data.Sigma_imp_iw, nested_scheme.get_latt_to_imp_mapping())
  data.get_Sigmakw = lambda: full_fill_Sigmakw_from_Sigmaijw(data.Sigmakw, data.Sigmaijw)
  data.get_Sigma_loc = lambda: full_fill_local_from_latt(data.Sigma_loc_iw, data.Sigmakw)

  data.get_Gkw = lambda: full_fill_Gkw_from_iws_mus_epsiolonk_and_Sigmakw(data.Gkw, data.iws, data.mus, data.epsilonk, data.Sigmakw)
  data.get_G_loc = lambda: full_fill_local_from_latt(data.G_loc_iw, data.Gkw)
  data.get_n_from_G_loc = lambda: blockwise_get_n_from_G_loc_iw(data.G_loc_iw['up'], fit_tail_starting_iw = 14.0, ntau = None, site_index = 0)

  if use_G_proj:                                  
    data.get_Gijw = lambda: [full_fill_Gijw_from_Gkw(data.Gijw, data.Gkw, N_cores=1), full_fill_G_proj_iw(data.G_proj_iw, data.Gijw, nested_scheme) ]
  else:
    data.get_Gijw = lambda: full_fill_Gijw_from_Gkw(data.Gijw, data.Gkw, N_cores=1)

  data.set_mu = lambda mu: set_mu(mu, data)
  data.get_mu = lambda: data.mus['up']
  data.get_n = lambda: [data.get_Gkw(), data.get_G_loc(), set_n(data.get_n_from_G_loc(),data)][-1]

  if flexible_Gweiss:
    data.get_Gweiss = lambda: ( flexible_Gweiss_iw_from_Gweiss_iw_Gijw_and_G_imp_iw(data.Gweiss_iw, data.Gijw, data.G_imp_iw, 
                                                                                    nested_scheme.get_imp_to_latt_mapping(), sign, sign_up_to)
                                if not is_zero(data.Gweiss_iw) else
                                full_fill_Gweiss_iw_from_Gijw_and_Sigma_imp_iw(data.Gweiss_iw,data.Gijw,data.Sigma_imp_iw, mapping = nested_scheme.get_imp_to_latt_mapping())    
                              )
  elif use_G_proj:
    data.get_Gweiss = lambda: full_full_Gweiss_iw_from_G_proj_iw_and_Sigma_imp_iw(data.Gweiss_iw,data.G_proj_iw,data.Sigma_imp_iw)  
  else:
    data.get_Gweiss = lambda: full_fill_Gweiss_iw_from_Gijw_and_Sigma_imp_iw(data.Gweiss_iw,data.Gijw,data.Sigma_imp_iw, mapping = nested_scheme.get_imp_to_latt_mapping())  

  data.dump_solvers = lambda suffix: [solver_class.dump( data.solvers[C], data.archive_name, suffix='-%s%s'%(C,suffix) ) for C in data.solvers.keys()]

#----------------------------- nested edmft -----------------------------------------------------------------------#
def prepare_nested_edmft( data, nested_scheme, solver_class = solvers.ctint): 
  assert data.__class__ == nested_edmft_data, "wrong data type"
  prepare_nested( data, nested_scheme, solver_class )
  data.get_P_imp = lambda: fill_P_imp_from_chi_imp_W_imp_and_Uweiss(data.P_imp_iw, data.chi_imp_iw, data.W_imp_iw, data.Uweiss_iw)
  data.get_Pijnu = lambda: full_fill_Pijnu_from_P_imp_iw(data.Pijnu, data.P_imp_iw, nested_scheme.get_latt_to_imp_mapping())
  data.get_Pqnu = lambda: full_fill_Sigmakw_from_Sigmaijw(data.Pqnu, data.Pijnu)
  data.get_P_loc = lambda: full_fill_local_from_latt(data.P_loc_iw, data.Pqnu)

  data.get_W_imp = lambda: fill_W_imp_from_chi_imp_and_Uweiss( data.W_imp_iw, data.chi_imp_iw, data.Uweiss_iw)
  data.get_Wqnu = lambda: full_fill_Wqnu_from_Jq_and_Pqnu(data.Wqnu,data.Jq,data.Pqnu)
  data.get_W_loc = lambda: full_fill_local_from_latt(data.W_loc_iw, data.Wqnu)                                    
  data.get_Wijnu = lambda: full_fill_Gijw_from_Gkw(data.Wijnu, data.Wqnu, N_cores=1)

  data.get_Uweiss = lambda: [ full_fill_Uweiss_iw_from_Wijnu_and_P_imp_iw(data.Uweiss_iw,data.Wijnu,data.P_imp_iw, mapping = nested_scheme.get_imp_to_latt_mapping()),
                              fill_Uweiss_dyn_from_Uweiss(data.Uweiss_dyn_iw,data.Uweiss_iw) ]
  #no lattice calc, for reversed etc.
  mp = nested_scheme.get_imp_to_latt_mapping()
  print "nested_scheme.maxLx: ",nested_scheme.maxLx
  print "max  nsites:", nested_scheme.maxLx**2
  def ij_iterator():
    nsites = nested_scheme.maxLx**2
    for i in range(nsites):
      for j in range(nsites):
        yield i,j
  def ijA_iterator():
    for i,j in ij_iterator():
      for A in data.bosonic_struct.keys():
         yield i,j,A
  data.copy_imp_to_latt = lambda C: [ [ numpy.copyto(data.Gijw['up'][:,mp(C,i,j)[0], mp(C,i,j)[1]],data.G_imp_iw[C].data[:,i,j]) for i,j in ij_iterator()], 
                                      [ numpy.copyto(data.Wijnu[A][:,mp(C,i,j)[0], mp(C,i,j)[1]],data.W_imp_iw[C+'|'+A].data[:,i,j]) for i,j,A in ijA_iterator() ] ] 

#----------------------------- cumul_nested -----------------------------------------------------------------------#
def prepare_cumul_nested( data, nested_scheme, solver_class = solvers.ctint  ):
  assert data.__class__ == cumul_nested_data, "wrong data type"
  assert data.fermionic_struct == {'up': [0]}, "wrong fermionic struct for this calcalation"
  assert data.impurity_struct == nested_scheme.get_impurity_struct(), "wrong impurity struct for this nested scheme" 

  data.get_g_imp = lambda: full_fill_g_imp_iw_from_Sigma_imp_iw(data.g_imp_iw, data.mus['up'], data.Sigma_imp_iw)
  data.get_gijw = lambda: full_fill_Sigmaijw_from_Sigma_imp_iw(data.gijw, data.g_imp_iw, nested_scheme.get_latt_to_imp_mapping())
  data.get_gkw = lambda: full_fill_Sigmakw_from_Sigmaijw(data.gkw, data.gijw)
  data.get_Sigmakw = lambda: full_fill_Sigmakw_from_gkw(data.Sigmakw, data.ws, data.mus['up'], data.gkw)
  data.get_Sigma_loc = lambda: full_fill_local_from_latt(data.Sigma_loc_iw, data.Sigmakw)    

  data.get_Gkw = lambda: full_fill_Gkw_from_epsiolonk_and_gkw(data.Gkw, data.epsilonk, data.gkw)
  data.get_G_loc = lambda: full_fill_local_from_latt(data.G_loc_iw, data.Gkw)
  data.get_n_from_G_loc = lambda: blockwise_get_n_from_G_loc_iw(data.G_loc_iw['up'], fit_tail_starting_iw = 14.0, ntau = None, site_index = 0)
  data.get_Gijw = lambda: full_fill_Gijw_from_Gkw(data.Gijw, data.Gkw, N_cores=1)
  
  data.set_mu = lambda mu: set_mu(mu, data)
  data.get_mu = lambda: data.mus['up']
  data.get_n = lambda: [data.get_g_imp(), data.get_gijw(), data.get_gkw(), data.get_Gkw(), data.get_G_loc(), set_n(data.get_n_from_G_loc(),data)][-1]

  data.get_Gweiss = lambda: full_fill_Gweiss_iw_from_Gijw_and_Sigma_imp_iw(data.Gweiss_iw,data.Gijw,data.Sigma_imp_iw, mapping = nested_scheme.get_imp_to_latt_mapping()) 

  data.dump_solvers = lambda suffix: [solver_class.dump( data.solvers[C], data.archive_name, suffix='-%s%s'%(C,suffix) ) for C in data.impurity_struct.keys()]

#----------------------------- dca -----------------------------------------------------------------------#
def prepare_dca( data, dca_scheme, solver_class = solvers.ctint ):
  assert len(data.impurity_struct.keys()) == 1, "in dca only one impurity problem!!" 
  key = data.impurity_struct.keys()[0]
  assert len(data.impurity_struct[key]) == dca_scheme.dim, "wrong impurity struct for the dca calculation!" 
  assert len(data.fermionic_struct.keys()) == len(data.impurity_struct[key]), "fermionic and impurity struct not consistent"
  assert data.__class__ == dca_data, "wrong data type"
 
  r0 = dca_scheme.get_r0()
  r0_key = '%02d'%r0
  data.get_SigmaR = lambda: dca_scheme.get_QR_from_Q_imp(data.SigmaR_iw, data.Sigma_imp_iw)
  data.get_SigmaK = lambda: dca_scheme.get_QK_from_QR(data.SigmaK_iw, data.SigmaR_iw)

  data.get_GK = lambda: [ full_fill_GK_iw(data.GK_iw, data.SigmaK_iw, data.mus[r0_key], dca_scheme.dca_patches), 
                          [impose_real_valued_in_imtime(g) for name,g in data.GK_iw] ]
  data.get_GR0 = lambda:  [ dca_scheme.get_QR_from_QK(data.GR_iw, data.GK_iw, l_list = [r0]), 
                            impose_real_valued_in_imtime(data.GR_iw[r0_key]) ]
  data.get_n_from_GR0 = lambda: blockwise_get_n_from_G_loc_iw(data.GR_iw[r0_key], fit_tail_starting_iw = 14.0, ntau = None, site_index = 0)
  data.get_GR = lambda: [ dca_scheme.get_QR_from_QK(data.GR_iw, data.GK_iw), 
                         [impose_real_valued_in_imtime(g) for name,g in data.GR_iw] ]
  data.get_Gijw = data.get_GR

  data.set_mu = lambda mu: set_mu(mu, data)
  data.get_mu = lambda: data.mus['00']
  data.get_n = lambda: [data.get_GK(), data.get_GR0(), set_n(data.get_n_from_GR0(),data)][-1]

  data.get_GweissK = lambda: full_fill_GweissK_iw_from_Dyson(data.GweissK_iw, data.GK_iw, data.SigmaK_iw)
  data.get_GweissR = lambda: dca_scheme.get_QR_from_QK(data.GweissR_iw, data.GweissK_iw)
  data.get_Gweiss_iw = lambda: dca_scheme.get_Q_imp_from_QR(data.Gweiss_iw, data.GweissR_iw)

  data.get_Gweiss = lambda: [data.get_GweissK(), data.get_GweissR(), data.get_Gweiss_iw(), [impose_real_valued_in_imtime(g) for name,g in data.Gweiss_iw] ]

  data.dump_solvers = lambda suffix: [solver_class.dump( data.solvers[C], data.archive_name, suffix='-%s%s'%(C,suffix) ) for C in data.impurity_struct.keys()]

#----------------------------- dca_plus -----------------------------------------------------------------------#
def prepare_dca_plus( data, dca_scheme, solver_class = solvers.ctint, alpha = 1, n_RL_iterations = 10, embedded = False, real_space_sc = False, no_convolution = False, impose_ph_symmetry = False ):
  assert len(data.impurity_struct.keys()) == 1, "in dca only one impurity problem!!" 
  key = data.impurity_struct.keys()[0]
  assert len(data.impurity_struct[key]) == dca_scheme.dim, "wrong impurity struct for the dca calculation!" 
  assert len(data.fermionic_struct.keys()) == len(data.impurity_struct[key]), "fermionic and impurity struct not consistent"
  assert data.__class__ == dca_plus_data, "wrong data type"
  nK = int(round(numpy.sqrt(dca_scheme.dim)))
  print 'nK: ', nK
  assert dca_scheme.n1 == dca_scheme.n1 and dca_scheme.m1==0 and dca_scheme.n2==0, "not general for now..."  
  assert nK**2 == dca_scheme.dim, "must be n1==m2, n2==m1==0"

  #data.get_SigmaR = lambda: [ full_fill_SigmaR_iw_from_Sigma_imp_iw(data.SigmaR_iw, data.Sigma_imp_iw, lambda i: dca_scheme.i_to_ij(i)), dca_scheme.symmetrize_QR(data.SigmaR_iw) ]
  data.get_SigmaR = lambda: [ dca_scheme.get_QR_from_Q_imp(data.SigmaR_iw, data.Sigma_imp_iw) ]
  data.get_SigmaK = lambda: dca_scheme.get_QK_from_QR(data.SigmaK_iw, data.SigmaR_iw)

  r0 = dca_scheme.get_r0()
  r0_key = '%02d'%r0

  data.get_XiK = lambda: fill_XiK_from_SigmaK(data.XiK_iw, data.SigmaK_iw, alpha)
  data.get_XiR = lambda: dca_scheme.get_QR_from_QK(data.XiR_iw, data.XiK_iw)
  if not embedded:
    data.get_Xik = lambda: dca_scheme.get_Qk_from_QR(data.Xikw['up'], data.XiR_iw, data.ks)
    data.get_Sigmaimpk = lambda: blockwise_Sigmak_from_Xik(data.Sigmaimpkw['up'], data.Xikw['up'], alpha)
    if not no_convolution:
      data.get_Sigmakw = lambda: [ numpy.copyto(data.Sigmakw['up'], data.Sigmaimpkw['up']),
                                   Richardson_Lucy(data.Sigmaimpkw['up'], data.Sigmakw['up'], 
                                                   nK, n_iterations = n_RL_iterations, 
                                                   desired_loc=data.SigmaR_iw[r0_key].data[:,0,0],
                                                   impose_ph_symmetry=impose_ph_symmetry) ]
    else:
      data.get_Sigmakw = lambda: numpy.copyto(data.Sigmakw['up'], data.Sigmaimpkw['up']) 

  else:
    data.get_Xik = lambda: dca_scheme.get_Qk_from_QR_embedded(data.Xikw['up'], data.XiR_iw, data.ks)
    data.get_Sigmaimpk = lambda: None 
    data.get_Sigmakw = lambda: blockwise_Sigmak_from_Xik(data.Sigmakw['up'], data.Xikw['up'], alpha)

  data.get_Gkw = lambda: full_fill_Gkw_from_iws_mus_epsiolonk_and_Sigmakw(data.Gkw, data.iws, data.mus, data.epsilonk, data.Sigmakw)
  data.get_G_loc = lambda: full_fill_local_from_latt(data.G_loc_iw, data.Gkw)
  data.get_n_from_G_loc = lambda: blockwise_get_n_from_G_loc_iw(data.G_loc_iw['up'], fit_tail_starting_iw = 14.0, ntau = None, site_index = 0)
  data.get_GR = lambda: [ dca_scheme.get_QR_from_QK(data.GR_iw, data.GK_iw), dca_scheme.symmetrize_QR(data.GR_iw) ]

  data.set_mu = lambda mu: set_mu(mu, data)
  data.get_mu = lambda: data.mus['up']
  data.get_n = lambda: [data.get_Gkw(), data.get_G_loc(), set_n(data.get_n_from_G_loc(),data)][-1]

  if not real_space_sc:
    data.get_GK = lambda: dca_scheme.Qkw_to_QK_iw(data.GK_iw, IBZ_convolution(data.Gkw['up'].real, nK)+1j*IBZ_convolution(data.Gkw['up'].imag, nK))
    data.get_Gijw = lambda: [data.get_GK(), [fit_fermionic_gf_tail(g) for name,g in data.GK_iw], data.get_GR()]
  else:
    data.get_Gijw = lambda: full_fill_Gijw_from_Gkw(data.Gijw, data.Gkw, N_cores=1)
    data.get_GR = lambda: dca_scheme.Qrw_to_QR_iw(data.GR_iw, data.Gijw)
    data.get_GK = lambda: dca_scheme.get_QK_from_QR(data.GK_iw, data.GR_iw)    

  data.get_GweissK = lambda: full_fill_GweissK_iw_from_Dyson(data.GweissK_iw, data.GK_iw, data.SigmaK_iw)
  data.get_GweissR = lambda: dca_scheme.get_QR_from_QK(data.GweissR_iw, data.GweissK_iw)
  data.get_Gweiss_iw = lambda: dca_scheme.get_Q_imp_from_QR(data.Gweiss_iw, data.GweissR_iw)

  data.get_Gweiss = lambda: [data.get_GweissK(), data.get_GweissR(), dca_scheme.symmetrize_QR(data.GweissR_iw), data.get_Gweiss_iw()]

  data.dump_solvers = lambda suffix: [solver_class.dump( data.solvers[C], data.archive_name, suffix='-%s%s'%(C,suffix) ) for C in data.impurity_struct.keys()]

#----------------------------- celullar -----------------------------------------------------------------------#

def prepare_cellular( data, Lx, Ly, solver_class = solvers.ctint, periodized = False  ):
  print "prepare_cellular"

  assert data.__class__ == cellular_data, "wrong data type"
  assert data.fermionic_struct == {'up': [0]}, "wrong fermionic struct for this calcalation"
  assert len(data.impurity_struct.keys()) == 1, "in celullar we solve only one cluster" 

  if periodized:
    data.get_Sigmaijkw = lambda: full_fill_Sigmaijkw_periodized(data.Sigmaijkw, data.Sigma_imp_iw, data.ks)
  else:
    data.get_Sigmaijkw = lambda: full_fill_Sigmaijkw(data.Sigmaijkw, data.Sigma_imp_iw)
  
  data.get_Gijkw = lambda: full_fill_Gijkw(data.Gijkw, data.iws, data.mus, data.epsilonijk, data.Sigmaijkw)
  data.get_G_ij_loc = lambda: full_fill_G_ij_iw(data.G_ij_iw, data.Gijkw)
  data.get_Gijw = data.get_G_ij_loc #this is needed for the nested_mains.lattice
  print 'imp_key: ', data.imp_key
  data.get_n_from_G_ij_loc = lambda: blockwise_get_n_from_G_loc_iw(data.G_ij_iw[data.imp_key], fit_tail_starting_iw = 14.0, ntau = None, site_index = 0)
                               #full_fill_ns_from_G_loc_iw(data.ns, data.G_ij_iw, fit_tail_starting_iw = 14.0, ntau = None)

  data.set_mu = lambda mu: set_mu(mu, data)
  data.get_mu = lambda: data.mus['up']
  data.get_n = lambda: [data.get_Gijkw(), data.get_G_ij_loc(), set_n(data.get_n_from_G_ij_loc(),data)][-1]

  data.get_Gweiss = lambda: full_fill_Gweiss_iw(data.Gweiss_iw, data.G_ij_iw, data.Sigma_imp_iw)

  data.dump_solvers = lambda suffix: [solver_class.dump( data.solvers[C], data.archive_name, suffix='-%s%s'%(C,suffix) ) for C in data.impurity_struct.keys()]

  data.periodize_cumul = lambda: periodize_cumul(data.Gkw, data.Sigmakw, data.gkw, data.gijw, data.g_imp_iw, data.iws, data.mus, data.epsilonk, data.Sigma_imp_iw, Lx, Ly)
  data.periodize_selfenergy = lambda: periodize_selfenergy(data.Gkw, data.Sigmakw, data.Sigmaijw, data.iws, data.mus, data.epsilonk, data.Sigma_imp_iw, Lx, Ly)

  data.dump_solvers = lambda suffix: [solver_class.dump( data.solvers[C], data.archive_name, suffix='-%s%s'%(C,suffix) ) for C in data.impurity_struct.keys()]

#----------------------------- triangular celullar -----------------------------------------------------------------------#

def prepare_cellular_triangular( data, Lx, Ly, solver_class = solvers.ctint, periodized = False  ):
  print "prepare_cellular_triangular"
  prepare_cellular( data, Lx, Ly, solver_class, periodized )  

  if periodized:
    data.get_Sigmaijkw = lambda: triangular_full_fill_Sigmaijkw_periodized(data.Sigmaijkw, data.Sigma_imp_iw, data.ks)
  
  data.periodize_cumul = lambda: None
  data.periodize_selfenergy = lambda: periodize_selfenergy(data.Gkw, data.Sigmakw, data.Sigmaijw, 
                                                           data.iws, data.mus, data.epsilonk, data.Sigma_imp_iw, 
                                                           Lx, Ly, mapping=triangular_cellular_latt_to_imp_mapping)

