
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

'''
from sqlalchemy.orm import Session
from sqlalchemy import select

from db_scripts.brewbrain_db import RecipeML

class RecipeDataset(torch.utils.data.Dataset):

  def __init__(self, db_engine=None): 
    if db_engine != None: self.load_from_db(db_engine)

  def load_from_db(self, db_engine):
    # Convert the database into numpy arrays as members of this
    with Session(db_engine) as session:
      # Read all the recipes into numpy format
      self._load_recipes(session)


  # Build all the look-up tables for the indices (used by the model/network) to the database IDs
  # NOTE: 0 is the "empty slot" category for all look-ups
  def _load_lookup_tables(self, session):

    # Core Grains
    coregrain_dbids = session.scalars(select(CoreGrain.id)).all()
    self.core_grains_dbid_to_idx = {cgid: i+1 for i, cgid in enumerate(coregrain_dbids)} 
    self.core_grains_idx_to_dbid = {i+1: cgid for i, cgid in enumerate(coregrain_dbids)}
    
    # Core Adjuncts
    # TODO
    
    # Hops
    hop_dbids = session.scalars(select(Hop.id)).all()
    self.hops_dbid_to_idx = {hopid: i+1 for i, hopid in enumerate(hop_dbids)}
    self.hops_idx_to_dbid = {i+1: hopid for i, hopid in enumerate(hop_dbids)}
    
    # Core Microorganisms
    # TODO
    
    # Core Styles
    corestyle_dbids = session.scalars(select(CoreStyle.id)).all()
    self.core_styles_dbid_to_idx = {csid: i+1 for i, csid in enumerate(corestyle_dbids)}
    self.core_styles_idx_to_dbid = {i+1: csid for i, csid in enumerate(corestyle_dbids)}
    
    # Core Miscs
    # TODO

  def _load_recipes(self, session):
    self._load_lookup_tables(session) # Build the set of tables for look-up between indices and database ids
      
    self.recipes = []
    
    # Only load fixed quantities of rows into memory at a time, the recipes table is BIG
    recipe_select_stmt = select(RecipeML).execution_options(yield_per=1024)
    for recipeMLs in session.scalars(recipe_select_stmt):
      
      for recipeML in recipeMLs:
        # Mash steps...
        # > type
        # > time
        # > avg. temperature
        # > infusion amount
        
        # Grains...
        # > amount (%)
        # > core type

        # Adjuncts...
        # > amount (g/L)
        # > core type???

        # Hops...
        # > amount ... g/L? IBUs? ... context dependant???
        # > stage - index... {mash, fw, boil, aroma, whirlpool, dh})
        # > time
        
        # Misc...
        # > type (core type???)
        # > amount
        # > unit... weight or volume?
        # > stage - index... {mash, sparge, boil, whirlpool, primary, secondary}
        
        # Microorganisms
        # > type... core types???
        # > stage - index... {primary, secondary}
        
        # Core Style > index...
        

'''
    
    