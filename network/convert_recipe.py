
import numpy as np

NO_GRAIN_INDEX  = 0
NUM_GRAIN_SLOTS = 16

def build_grain_type_lookup_maps(grains_db):
  # Create a mapping between the database id and the index for each grain
  grains_dbid_to_idx = {grain.id: i+1 for i,grain in enumerate(grains_db)} # 0 is the "no grain" category
  grains_idx_to_dbid = {i+1: grain.id for i,grain in enumerate(grains_db)}
  grains_idx_to_dbid[NO_GRAIN_INDEX] = None

  return grains_dbid_to_idx, grains_idx_to_dbid


def convert_recipe_numpy(recipes_db, grains_dbid_to_idx):
  recipe_mashes = []
  for recipe in recipes_db:
    grain_inds = np.zeros((NUM_GRAIN_SLOTS), dtype=np.int32)
    grain_amts = np.zeros((NUM_GRAIN_SLOTS), dtype=np.float32)
    total_qty = 0

    # Order the grains based on their quantities highest to lowest
    sorted(recipe.grains, key=lambda x: x.amount, reverse=True)
    for i, grainAT in enumerate(recipe.grains):
      total_qty += grainAT.amount
      grain_amts[i] = grainAT.amount
      grain_inds[i] = grains_dbid_to_idx[grainAT.grain_id]
    grain_amts /= total_qty

    recipe_mashes.append({
      'type_inds': grain_inds,
      'amts': grain_amts
    })

  return recipe_mashes
