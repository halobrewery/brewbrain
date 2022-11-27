
def srm_to_lovibond(srm): return (srm+0.76)/1.3546
def srm_to_ebc(srm):      return srm*1.97
def lovibond_to_srm(lov): return 1.3546*lov - 0.76
def ebc_to_srm(ebc):      return ebc*0.508
def ebc_to_lovibond(ebc): return srm_to_lovibond(ebc_to_srm(ebc))