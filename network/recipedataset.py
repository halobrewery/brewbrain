import sys
import os

import torch
import numpy as np

from sqlalchemy.orm import Session
from sqlalchemy import select

from db_scripts.brewbrain_db import RecipeML, CoreGrain, CoreAdjunct, Hop, Microorganism, Misc
from db_scripts.brewbrain_db import RecipeMLMiscAT, RecipeMLHopAT, RecipeMLMicroorganismAT

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from beer_util_functions import hop_form_utilization, alpha_acid_mg_per_l

class RecipeDataset(torch.utils.data.Dataset):
  NUM_GRAIN_SLOTS = 16
  NUM_ADJUNCT_SLOTS = 8
  NUM_HOP_SLOTS = 32
  NUM_MISC_SLOTS = 16
  NUM_MICROORGANISM_SLOTS = 8

  def __init__(self, db_engine=None): 
    if db_engine != None: self.load_from_db(db_engine)

  def load_from_db(self, db_engine):
    # Convert the database into numpy arrays as members of this
    with Session(db_engine) as session:
      # Read all the recipes into numpy format
      self._load_recipes(session)


  def _load_recipes(self, session):
    # Build the set of tables for look-up between indices and database ids
    # NOTE: 0 is the "empty slot" category for all look-ups
    
    def _db_idx_lookups(orm_class):
      dbids = session.scalars(select(orm_class.id)).all()
      dbid_to_idx = {id: i+1 for i, id in enumerate(dbids)} 
      idx_to_dbid = {i+1: id for i, id in enumerate(dbids)}
      idx_to_dbid[0] = None
      return (dbid_to_idx, idx_to_dbid)
    
    def _stage_lookup(stage_names):
      stage_name_to_idx = {stage: i+1 for i, stage in enumerate(stage_names)}
      stage_idx_to_name = {i+1: stage for i, stage in enumerate(stage_names)}
      stage_idx_to_name[0] = None
      return (stage_name_to_idx, stage_idx_to_name)
    
    core_grains_dbid_to_idx, core_grains_idx_to_dbid = _db_idx_lookups(CoreGrain) # Core Grains
    core_adjs_dbid_to_idx, core_adjs_idx_to_dbid = _db_idx_lookups(CoreAdjunct) # Core Adjuncts
    hops_dbid_to_idx, hops_idx_to_dbid = _db_idx_lookups(Hop) # Hops
    mos_dbid_to_idx, mos_idx_to_dbid = _db_idx_lookups(Microorganism) # Microorganisms
    miscs_dbid_to_idx, miscs_idx_to_dbid = _db_idx_lookups(Misc) # Miscs
    
    # Sub-enumerations
    # Mash step types (e.g., Infusion, Decoction, Temperature)
    mash_step_name_to_idx, mash_step_idx_to_name = _stage_lookup(_mash_step_types(session))
    # Misc stage (e.g., Mash, Boil, Primary, ...)
    misc_stage_name_to_idx, misc_stage_idx_to_name = _stage_lookup(_misc_stage_types(session))
    # Hop stage (e.g., Mash, Boil, Primary, ...)
    hop_stage_name_to_idx, hop_stage_idx_to_name = _stage_lookup(_hop_stage_types(session))
    # Microorganism stage (e.g., Primary, Secondary)
    mo_stage_name_to_idx, mo_stage_idx_to_name = _stage_lookup(_microorganism_stage_types(session))
    
    # ...Core Styles
    #corestyle_dbids = session.scalars(select(CoreStyle.id)).all()
    #self.core_styles_dbid_to_idx = {csid: i+1 for i, csid in enumerate(corestyle_dbids)}
    #self.core_styles_idx_to_dbid = {i+1: csid for i, csid in enumerate(corestyle_dbids)}
      
    self.recipes = []
    
    # Only load fixed quantities of rows into memory at a time, the recipes table is BIG
    recipe_select_stmt = select(RecipeML).execution_options(yield_per=1024)
    for recipeMLs in session.scalars(recipe_select_stmt):
      
      for recipeML in recipeMLs:
        recipe_data = {
          'dbid': recipeML.id
        }
        
        # Mash steps...
        mash_step_type_inds = np.zeros((RecipeML.MAX_MASH_STEPS), dtype=np.int32)   # mash step type (index)
        mash_step_times = np.zeros((RecipeML.MAX_MASH_STEPS), dtype=np.float32)     # mash step time (mins)
        mash_step_avg_temps = np.zeros((RecipeML.MAX_MASH_STEPS), dtype=np.float32) # mash step avg. temperature (C)
        mash_steps = recipeML.mash_steps()
        for idx, mash_step in enumerate(mash_steps):
          prefix = "mash_step_"+str(idx+1)
          ms_type = mash_step[prefix+"_type"]
          ms_time = mash_step[prefix+"_time"]
          ms_start_temp = mash_step[prefix+"_start_temp"]
          ms_end_temp = mash_step[prefix+"_end_temp"]
          if ms_type == None or ms_time == None or (ms_start_temp == None and ms_end_temp == None): break
          
          mash_step_type_inds[idx] = mash_step_name_to_idx[ms_type]
          mash_step_times[idx] = ms_time
          if ms_start_temp != None:
            if ms_end_temp != None:
              mash_step_avg_temps[idx] = (ms_start_temp + ms_end_temp) / 2.0
            else:
              mash_step_avg_temps[idx] = ms_start_temp
          else:
            mash_step_avg_temps[idx] = ms_end_temp
        recipe_data['mash_step_type_inds'] = mash_step_type_inds
        recipe_data['mash_step_times'] = mash_step_times
        recipe_data['mash_step_avg_temps'] = mash_step_avg_temps
        
        # Grains...
        grain_core_type_inds = np.zeros((self.NUM_GRAIN_SLOTS), dtype=np.int32) # core grain type (index)
        grain_amts = np.zeros((self.NUM_GRAIN_SLOTS), dtype=np.float32)         # amount (as a %)
        total_grain_qty = 0.0
        for idx, grainAT in enumerate(recipeML.grains):
          assert grainAT.grain.core_grain_id != None
          total_grain_qty += grainAT.amount
          grain_amts[idx] = grainAT.amount
          grain_core_type_inds[idx] = core_grains_dbid_to_idx[grainAT.grain.core_grain_id]
        grain_amts /= total_grain_qty
        recipe_data['grain_core_type_inds'] = grain_core_type_inds
        recipe_data['grain_amts'] = grain_amts
        
        # Adjuncts...
        adjunct_core_type_inds = np.zeros((self.NUM_ADJUNCT_SLOTS), dtype=np.int32) # core adjunct type (index)
        adjunct_amts = np.zeros((self.NUM_ADJUNCT_SLOTS), dtype=np.float32)         # amount (in ~(g or ml)/L)
        for idx, adjunctAT in enumerate(recipeML.adjuncts):
          assert adjunctAT.adjunct.core_adjunct_id != None
          adjunct_core_type_inds[idx] = core_adjs_dbid_to_idx[adjunctAT.adjunct.core_adjunct_id]
          adjunct_amts[idx] = adjunctAT.amount / recipeML.fermenter_vol
        recipe_data['adjunct_core_type_inds'] = adjunct_core_type_inds
        recipe_data['adjunct_amts'] = adjunct_amts
        
        # IBU-Valued Hops (boil hops)
        #boil_hop_type_inds = np.zeros((...), dtype=np.int32)  # hop type (index)
        #boil_hop_aa_concs = np.zeros((...), dtype=np.float32) # amount (concentration of alpha acids in mg/L)
        #boil_hop_times = np.zeros((...), dtype=np.float32)    # time (in mins)
        
        # Non-IBU Hops
        # amount (mass in g/L)
        # stage (index)
        # time (in mins)
        
        # Misc...
        misc_type_inds = np.zeros((self.NUM_MISC_SLOTS), dtype=np.int32)  # misc type (index)
        misc_amts = np.zeros((self.NUM_MISC_SLOTS), dtype=np.float32)     # amount (in ~(g or ml)/L)
        misc_stage_inds = np.zeros((self.NUM_MISC_SLOTS), dtype=np.int32) # stage (index)
        
        

        
        
        # Microorganisms
        # > type... core types???
        # > stage - index... {primary, secondary}
        
        self.recipes.append(recipe_data)


def _mash_step_types(session):
  mash_step_types = []
  mash_step_types += session.query(RecipeML.mash_step_1_type).group_by(RecipeML.mash_step_1_type).all()
  mash_step_types += session.query(RecipeML.mash_step_1_type).group_by(RecipeML.mash_step_2_type).all()
  mash_step_types += session.query(RecipeML.mash_step_1_type).group_by(RecipeML.mash_step_3_type).all()
  mash_step_types += session.query(RecipeML.mash_step_1_type).group_by(RecipeML.mash_step_4_type).all()
  mash_step_types += session.query(RecipeML.mash_step_1_type).group_by(RecipeML.mash_step_5_type).all()
  mash_step_types += session.query(RecipeML.mash_step_1_type).group_by(RecipeML.mash_step_6_type).all()
  return list(set(mash_step_types))

def _misc_stage_types(session):
  return [stage[0] for stage in session.query(RecipeMLMiscAT.stage).group_by(RecipeMLMiscAT.stage).all()]

def _hop_stage_types(session):
  return [stage[0] for stage in session.query(RecipeMLHopAT.stage).group_by(RecipeMLHopAT.stage).all()]

def _microorganism_stage_types(session):
  return [stage[0] for stage in session.query(RecipeMLMicroorganismAT.stage).group_by(RecipeMLMicroorganismAT.stage).all()]

def _recipe_vol_at_stage(recipe_ml, stage_name):
  if stage_name == 'mash':
    return recipe_ml.total_infusion_vol()
  elif stage_name == 'sparge' or stage_name == 'boil' or stage_name == 'first wort' or stage_name == 'whirlpool':
    return recipe_ml.postboil_vol
  else:
    assert stage_name == 'dry hop' or stage_name == 'primary' or stage_name == 'secondary' or stage_name == 'other' or stage_name == 'finishing'
    return recipe_ml.fermenter_vol
  