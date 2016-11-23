from data_containers import nested_data
from data_containers import cumul_nested_data
from getters import *
from impurity_solvers import solvers

def prepare_nested( data, nested_scheme ):
  assert data.__class__ == nested_data, "wrong data type"
  assert data.fermionic_struct == {'up': [0]}, "wrong fermionic struct for this calcalation"
  assert data.impurity_struct == nested_scheme.get_impurity_struct(), "wrong impurity struct for this nested scheme" 
  #TODO get mapping for cluster 



  data.get_Sigmaijw = lambda: full_fill_Sigmaijw_from_Sigma_imp_iw(data.Sigmaijw, data.Sigma_imp_iw, nested_scheme.get_latt_to_imp_mapping())
  data.get_Sigmakw = lambda: full_fill_Sigmakw_from_Sigmaijw(data.Sigmakw, data.Sigmaijw)
  data.get_Sigma_loc = lambda: full_fill_local_from_latt(data.Sigma_loc_iw, data.Sigmakw)

  data.get_Gkw = lambda: full_fill_Gkw_from_iws_mus_epsiolonk_and_Sigmakw(data.Gkw, data.iws, data.mus, data.epsilonk, data.Sigmakw)
  data.get_G_loc = lambda: full_fill_local_from_latt(data.G_loc_iw, data.Gkw)
  data.get_n_from_G_loc = lambda: full_fill_ns_from_G_loc_iw(data.ns, data.G_loc_iw, fit_tail_starting_iw = 14.0, ntau = None)
  data.get_Gijw = lambda: full_fill_Gijw_from_Gkw(data.Gijw, data.Gkw, N_cores=1)

  data.get_n = lambda: [data.get_Gkw(), data.get_G_loc(), data.get_n_from_G_loc()]

  data.get_Gweiss = lambda: full_fill_Gweiss_iw_from_Gijw_and_Sigma_imp_iw(data.Gweiss_iw,data.Gijw,data.Sigma_imp_iw, mapping = nested_scheme.get_imp_to_latt_mapping()) 

  data.dump_solvers = lambda suffix: [solvers.ctint.dump( data.solvers[C], data.archive_name, suffix='-%s%s'%(C,suffix) ) for C in data.impurity_struct.keys()]

def prepare_cumul_nested( data, nested_scheme ):
  assert data.__class__ == cumul_nested_data, "wrong data type"
  assert data.fermionic_struct == {'up': [0]}, "wrong fermionic struct for this calcalation"
  assert data.impurity_struct == nested_scheme.get_impurity_struct(), "wrong impurity struct for this nested scheme" 
  #TODO get mapping for cluster 

  data.get_g_imp = lambda: full_fill_g_imp_iw_from_Sigma_imp_iw(data.g_imp_iw, data.mus['up'], data.Sigma_imp_iw)
  data.get_gijw = lambda: full_fill_Sigmaijw_from_Sigma_imp_iw(data.gijw, data.g_imp_iw, nested_scheme.get_latt_to_imp_mapping())
  data.get_gkw = lambda: full_fill_Sigmakw_from_Sigmaijw(data.gkw, data.gijw)
  data.get_Sigmakw = lambda: full_fill_Sigmakw_from_gkw(data.Sigmakw, data.ws, data.mus['up'], data.gkw)
  data.get_Sigma_loc = lambda: full_fill_local_from_latt(data.Sigma_loc_iw, data.Sigmakw)    

  data.get_Gkw = lambda: full_fill_Gkw_from_epsiolonk_and_gkw(data.Gkw, data.epsilonk, data.gkw)
  data.get_G_loc = lambda: full_fill_local_from_latt(data.G_loc_iw, data.Gkw)
  data.get_n_from_G_loc = lambda: full_fill_ns_from_G_loc_iw(data.ns, data.G_loc_iw, fit_tail_starting_iw = 14.0, ntau = None)
  data.get_Gijw = lambda: full_fill_Gijw_from_Gkw(data.Gijw, data.Gkw, N_cores=1)

  data.get_n = lambda: [data.get_g_imp(), data.get_gijw(), data.get_gkw(), data.get_Gkw(), data.get_G_loc(), data.get_n_from_G_loc()]

  data.get_Gweiss = lambda: full_fill_Gweiss_iw_from_Gijw_and_Sigma_imp_iw(data.Gweiss_iw,data.Gijw,data.Sigma_imp_iw, mapping = nested_scheme.get_imp_to_latt_mapping()) 

  data.dump_solvers = lambda suffix: [solvers.ctint.dump( data.solvers[C], data.archive_name, suffix='-%s%s'%(C,suffix) ) for C in data.impurity_struct.keys()]
