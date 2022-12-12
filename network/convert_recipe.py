
import numpy as np

EMPTY_INDEX = 0

NO_GRAIN_INDEX  = 0
NUM_GRAIN_SLOTS = 16

def build_grain_type_lookup_maps(grains_db):
  # Create a mapping between the database id and the index for each grain
  grains_dbid_to_idx = {grain.id: i+1 for i,grain in enumerate(grains_db)} # 0 is the "no grain" category
  grains_idx_to_dbid = {i+1: grain.id for i,grain in enumerate(grains_db)}
  grains_idx_to_dbid[NO_GRAIN_INDEX] = None

  return grains_dbid_to_idx, grains_idx_to_dbid

def build_core_grain_type_lookup_maps(core_grains_db):
  # Create a mapping between the database id and the index for each grain
  core_grains_dbid_to_idx = {grain.id: i+1 for i,grain in enumerate(core_grains_db)} # 0 is the "no grain" category
  core_grains_idx_to_dbid = {i+1: grain.id for i,grain in enumerate(core_grains_db)}
  core_grains_idx_to_dbid[NO_GRAIN_INDEX] = None

  return core_grains_dbid_to_idx, core_grains_idx_to_dbid

def convert_recipe_numpy(recipes_db, grains_dbid_to_idx, core_grains_dbid_to_idx):
  recipe_mashes = []
  for recipe in recipes_db:
    grain_specific_type_inds = np.zeros((NUM_GRAIN_SLOTS), dtype=np.int32)
    grain_core_type_inds = np.zeros((NUM_GRAIN_SLOTS), dtype=np.int32)
    grain_amts = np.zeros((NUM_GRAIN_SLOTS), dtype=np.float32)
    total_qty = 0

    # Order the grains based on their quantities highest to lowest
    sorted(recipe.grains, key=lambda x: x.amount, reverse=True)
    idx = 0
    for grainAT in recipe.grains:
      # Certain grains may not be counted because they contribute nothing to the recipe (e.g., Rice Hulls)
      if grainAT.grain.core_grain_id == None: continue
      
      total_qty += grainAT.amount
      grain_amts[idx] = grainAT.amount
      grain_specific_type_inds[idx] = grains_dbid_to_idx[grainAT.grain_id]
      grain_core_type_inds[idx] = core_grains_dbid_to_idx[grainAT.grain.core_grain_id]
      idx += 1

    if total_qty == 0: 
      print(f"Invalid/Empty recipe found (id: {recipe.id})")
      continue

    grain_amts /= total_qty

    recipe_mashes.append({
      'recipe_id': recipe.id,
      'core_types': grain_core_type_inds,
      'specific_types': grain_specific_type_inds,
      'amts': grain_amts
    })
    
  return recipe_mashes
