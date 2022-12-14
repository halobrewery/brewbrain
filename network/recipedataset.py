import sys
import os

import torch
import numpy as np

from sqlalchemy.orm import Session
from sqlalchemy import select

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from db_scripts.brewbrain_db import RecipeML, CoreGrain, CoreAdjunct, Hop, Microorganism, Misc
from db_scripts.brewbrain_db import RecipeMLMiscAT, RecipeMLHopAT, RecipeMLMicroorganismAT

from beer_util_functions import hop_form_utilization, alpha_acid_mg_per_l

class RecipeDataset(torch.utils.data.Dataset):
  NUM_GRAIN_SLOTS = 16
  NUM_ADJUNCT_SLOTS = 8
  NUM_HOP_SLOTS = 32
  NUM_MISC_SLOTS = 16
  NUM_MICROORGANISM_SLOTS = 8
  NUM_FERMENT_STAGE_SLOTS = 2

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
    
    def _build_lookup(lookup_values):
      value_to_idx = {value: i+1 for i, value in enumerate(lookup_values)}
      idx_to_value = {i+1: value for i, value in enumerate(lookup_values)}
      idx_to_value[0] = None
      return (value_to_idx, idx_to_value)
    
    def _db_idx_lookups(orm_class):
      return _build_lookup(session.scalars(select(orm_class.id)).all())
    def _db_used_only_idx_lookups(at_id):
      dbids = session.scalars(select(at_id).group_by(at_id)).all()
      return _build_lookup(dbids)
    
    core_grains_dbid_to_idx, core_grains_idx_to_dbid = _db_idx_lookups(CoreGrain) # Core Grains
    core_adjs_dbid_to_idx, core_adjs_idx_to_dbid = _db_idx_lookups(CoreAdjunct)   # Core Adjuncts
    hops_dbid_to_idx, hops_idx_to_dbid = _db_used_only_idx_lookups(RecipeMLHopAT.hop_id) # Hops
    miscs_dbid_to_idx, miscs_idx_to_dbid = _db_idx_lookups(Misc) # Miscs
    mos_dbid_to_idx, mos_idx_to_dbid = _db_used_only_idx_lookups(RecipeMLMicroorganismAT.microorganism_id) # Microorganisms
    
    # Sub-enumerations
    # Mash step types (e.g., Infusion, Decoction, Temperature)
    mash_step_name_to_idx, mash_step_idx_to_name = _build_lookup(_mash_step_types(session))
    # Misc stage (e.g., Mash, Boil, Primary, ...)
    misc_stage_name_to_idx, misc_stage_idx_to_name = _build_lookup(_misc_stage_types(session))
    # Hop stage (e.g., Mash, Boil, Primary, ...)
    hop_stage_name_to_idx, hop_stage_idx_to_name = _build_lookup(_hop_stage_types(session))
    # Microorganism stage (e.g., Primary, Secondary)
    mo_stage_name_to_idx, mo_stage_idx_to_name = _build_lookup(_microorganism_stage_types(session))
    
    # ...Core Styles
    #corestyle_dbids = session.scalars(select(CoreStyle.id)).all()
    #self.core_styles_dbid_to_idx = {csid: i+1 for i, csid in enumerate(corestyle_dbids)}
    #self.core_styles_idx_to_dbid = {i+1: csid for i, csid in enumerate(corestyle_dbids)}
    
    
    self.recipes = []
    
    # Only load fixed quantities of rows into memory at a time, the recipes table is BIG
    recipe_select_stmt = select(RecipeML).execution_options(yield_per=1024)
    for recipeML_partition in session.scalars(recipe_select_stmt).partitions():
      
      for recipeML in recipeML_partition:
        infusion_vol = recipeML.total_infusion_vol()
        recipe_data = {
          'dbid': recipeML.id,
          'mash_ph': recipeML.mash_ph,
          'sparge_temp': recipeML.sparge_temp,
        }
        
        # Mash steps
        mash_step_type_inds = np.zeros((RecipeML.MAX_MASH_STEPS), dtype=np.int32)   # mash step type (index)
        mash_step_times = np.zeros((RecipeML.MAX_MASH_STEPS), dtype=np.float32)     # mash step time (mins)
        mash_step_avg_temps = np.zeros((RecipeML.MAX_MASH_STEPS), dtype=np.float32) # mash step avg. temperature (C)
        mash_steps = recipeML.mash_steps()
        for idx, mash_step in enumerate(mash_steps):
          ms_type = mash_step["_type"]
          ms_time = mash_step["_time"]
          ms_start_temp = mash_step["_start_temp"]
          ms_end_temp = mash_step["_end_temp"]
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
        
        # Fermentation steps
        ferment_stage_times = np.zeros((self.NUM_FERMENT_STAGE_SLOTS), dtype=np.float32) # time (in days)
        ferment_stage_temps = np.zeros((self.NUM_FERMENT_STAGE_SLOTS), dtype=np.float32) # temperature (in C)
        for idx in range(recipeML.num_ferment_stages):
          prefix = "ferment_stage_" + str(idx+1)
          ferment_stage_times[idx] = getattr(recipeML, prefix+"_time")
          ferment_stage_temps[idx] = getattr(recipeML, prefix+"_temp")
        recipe_data['ferment_stage_times'] = ferment_stage_times
        recipe_data['ferment_stage_temps'] = ferment_stage_temps
        
        # Grains
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
        
        # Adjuncts
        adjunct_core_type_inds = np.zeros((self.NUM_ADJUNCT_SLOTS), dtype=np.int32) # core adjunct type (index)
        adjunct_amts = np.zeros((self.NUM_ADJUNCT_SLOTS), dtype=np.float32)         # amount (in ~(g or ml)/L)
        for idx, adjunctAT in enumerate(recipeML.adjuncts):
          assert adjunctAT.adjunct.core_adjunct_id != None
          adjunct_core_type_inds[idx] = core_adjs_dbid_to_idx[adjunctAT.adjunct.core_adjunct_id]
          vol = recipeML.fermenter_vol if adjunctAT.stage == None else _recipe_vol_at_stage(recipeML, infusion_vol, adjunctAT.stage)
          assert vol != None and vol > 0
          adjunct_amts[idx] = (adjunctAT.amount * 1000.0) / vol
        recipe_data['adjunct_core_type_inds'] = adjunct_core_type_inds
        recipe_data['adjunct_amts'] = adjunct_amts
        
        # Hops
        hop_type_inds = np.zeros((self.NUM_HOP_SLOTS), dtype=np.int32)        # hop type (index)
        hop_stage_type_inds = np.zeros((self.NUM_HOP_SLOTS), dtype=np.int32)  # hop use/stage (index)
        hop_times = np.zeros((self.NUM_HOP_SLOTS), dtype=np.float32)          # time (in mins)
        hop_concentrations = np.zeros((self.NUM_HOP_SLOTS), dtype=np.float32) # if this is a boil hop then the amount is a (concentration of alpha acids in g/L), otherwise it's the hop concentration in g/L
        for idx, hopAT in enumerate(recipeML.hops):
          hop_type_inds[idx] = hops_dbid_to_idx[hopAT.hop_id]
          hop_stage_type_inds[idx] = hop_stage_name_to_idx[hopAT.stage]
          hop_times[idx] = hopAT.time
          if hopAT.stage == 'boil':
            hop_concentrations[idx] = hop_form_utilization(hopAT.form) * alpha_acid_mg_per_l(hopAT.alpha / 100.0, hopAT.amount * 1000.0, recipeML.postboil_vol) / 1000.0
          else: 
            hop_concentrations[idx] = (hopAT.amount * 1000.0) / _recipe_vol_at_stage(recipeML, infusion_vol, hopAT.stage)
        recipe_data['hop_type_inds'] = hop_type_inds
        recipe_data['hop_stage_type_inds'] = hop_stage_type_inds
        recipe_data['hop_times'] = hop_times
        recipe_data['hop_concentrations'] = hop_concentrations

        # Miscs
        misc_type_inds = np.zeros((self.NUM_MISC_SLOTS), dtype=np.int32)  # misc type (index)
        misc_amts = np.zeros((self.NUM_MISC_SLOTS), dtype=np.float32)     # amount (in ~(g or ml)/L)
        misc_times = np.zeros((self.NUM_MISC_SLOTS), dtype=np.float32)    # time (in mins)
        misc_stage_inds = np.zeros((self.NUM_MISC_SLOTS), dtype=np.int32) # stage (index)
        for idx, miscAT in enumerate(recipeML.miscs):
          misc_type_inds[idx] = miscs_dbid_to_idx[miscAT.misc_id]
          vol =  _recipe_vol_at_stage(recipeML, infusion_vol, miscAT.stage)
          assert vol != None and vol > 0
          misc_amts[idx] = (miscAT.amount * 1000.0) / vol
          misc_times[idx] = miscAT.time
          misc_stage_inds[idx] = misc_stage_name_to_idx[miscAT.stage]
        recipe_data['misc_type_inds'] = misc_type_inds
        recipe_data['misc_amts'] = misc_amts
        recipe_data['misc_times'] = misc_times
        recipe_data['misc_stage_inds'] = misc_stage_inds
        
        # Microorganisms
        mo_type_inds  = np.zeros((self.NUM_MICROORGANISM_SLOTS), dtype=np.int32)
        mo_stage_inds = np.zeros((self.NUM_MICROORGANISM_SLOTS), dtype=np.int32)
        for idx, moAT in enumerate(recipeML.microorganisms):
          mo_type_inds[idx]  = mos_dbid_to_idx[moAT.microorganism_id]
          mo_stage_inds[idx] = mo_stage_name_to_idx[moAT.stage]
        recipe_data['mo_type_inds'] = mo_type_inds
        recipe_data['mo_stage_inds'] = mo_stage_inds
        
        self.recipes.append(recipe_data)
        
      break # TODO: Remove this - just for debugging

def _mash_step_types(session):
  step_types = set()
  step_types.update(session.scalars(select(RecipeML.mash_step_1_type).group_by(RecipeML.mash_step_1_type)).all())
  step_types.update(session.scalars(select(RecipeML.mash_step_2_type).group_by(RecipeML.mash_step_2_type)).all())
  step_types.update(session.scalars(select(RecipeML.mash_step_3_type).group_by(RecipeML.mash_step_3_type)).all())
  step_types.update(session.scalars(select(RecipeML.mash_step_4_type).group_by(RecipeML.mash_step_4_type)).all())
  step_types.update(session.scalars(select(RecipeML.mash_step_5_type).group_by(RecipeML.mash_step_5_type)).all())
  step_types.update(session.scalars(select(RecipeML.mash_step_6_type).group_by(RecipeML.mash_step_6_type)).all())
  return [type for type in step_types if type != None]

def _misc_stage_types(session):
  return [stage[0] for stage in session.query(RecipeMLMiscAT.stage).group_by(RecipeMLMiscAT.stage).all()]

def _hop_stage_types(session):
  return [stage[0] for stage in session.query(RecipeMLHopAT.stage).group_by(RecipeMLHopAT.stage).all()]

def _microorganism_stage_types(session):
  return [stage[0] for stage in session.query(RecipeMLMicroorganismAT.stage).group_by(RecipeMLMicroorganismAT.stage).all()]

def _recipe_vol_at_stage(recipe_ml, infusion_vol, stage_name):
  if stage_name == 'mash':
    return infusion_vol
  elif stage_name == 'sparge' or stage_name == 'boil' or stage_name == 'first wort' or stage_name == 'whirlpool':
    return recipe_ml.postboil_vol
  else:
    assert stage_name == 'dry hop' or stage_name == 'primary' or stage_name == 'secondary' or stage_name == 'other' or stage_name == 'finishing'
    return recipe_ml.fermenter_vol
  
  

if __name__ == "__main__":
  from sqlalchemy import create_engine
  from db_scripts.brewbrain_db import BREWBRAIN_DB_ENGINE_STR, Base
  engine = create_engine(BREWBRAIN_DB_ENGINE_STR, echo=False, future=True)
  Base.metadata.create_all(engine)
  dataset = RecipeDataset(engine)