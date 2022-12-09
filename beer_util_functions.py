import numpy as np

# Colour Conversion
def srm_to_lovibond(srm): return (srm+0.76)/1.3546
def srm_to_ebc(srm):      return srm*1.97
def lovibond_to_srm(lov): return 1.3546*lov - 0.76
def ebc_to_srm(ebc):      return ebc*0.508
def ebc_to_lovibond(ebc): return srm_to_lovibond(ebc_to_srm(ebc))

# Temperature Conversion
def f_to_c(f): return (f - 32.0) * 5.0/9.0
def c_to_f(c): return (c * 9.0/5.0) + 32.0

# Gravity/Sugar and Attenuation Converters and Calculators
def sg_to_plato(sg): return (258.0 - 205.0*(sg - 1.0)) * (sg - 1.0)
def plato_to_sg(p): return (668 - np.sqrt(446224 - 820*(463 + p))) / 410

def oe_ae_re_abv_atten(original_extract, apparent_extract, final_gravity=None):
  """ Return a dict of the various extract and attenuation values given an original and apparent extract with optional final gravity.
  Args:
    original_extract (float): The original extract in Plato (as measured by refractometer before fermentation).
    apparent_extract (float): The apparent extract in Plato (as measured by hydrometer after/during fermentation).
    final_gravity (float, optional): The final gravity in SG (e.g., 1.010), if provided this will be used in place of the apparent extract. Defaults to None.
  Returns:
    dict: Set of calculated and given values for attenuation:
    {
      're': Real extract in Plato, 
      'abv_w': Alcohol by weight fraction, 
      'abv_v': Alcohol by volume fraction, 
      'ae': Apparent extract in Plato, 
      'real_atten': Real attenutation fraction
    }
  """
  final_gravity = plato_to_sg(apparent_extract) if final_gravity is None else final_gravity
  re = (0.1948*original_extract) + (0.8052*apparent_extract)
  abv_wt = (original_extract - re)/(2.0665 - 1.0665*original_extract/100.0)/100.0
  return {
    're'         : re,
    'abv_w'      : abv_wt,
    'abv_v'      : abv_wt*final_gravity/0.7907,
    'ae'         : (original_extract - apparent_extract)/original_extract,
    'real_atten' : ((original_extract - re)/original_extract)*(1.0/(1.0 - 0.005161*re))
  }

def coarsegrindasis_to_ppg(cgai): return cgai * 46.0  # Convert Coarse-Grind-As-Is of a grain addition into Points-Per-Pound-Per-Gallon
def coarsegrindasis_to_pkl(cgai): return cgai * 384.0 # Convert Coarse-Grind-As-Is of a grain addition into Points-Per-Kilogram-Per-Litre
