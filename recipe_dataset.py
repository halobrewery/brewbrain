import sys
import os
import pickle
import copy
import json

import torch
import numpy as np

from sqlalchemy.orm import Session
from sqlalchemy import select, func

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from brewbrain_db import RecipeML, CoreGrain, CoreAdjunct, Misc, Hop, Microorganism
from brewbrain_db import RecipeMLMiscAT, RecipeMLHopAT, RecipeMLMicroorganismAT

from beer_util_functions import hop_form_utilization, alpha_acid_mg_per_l
from running_stats import RunningStats

RECIPE_DATASET_FILENAME       = "recipe_dataset.pkl"
RECIPE_DATASET_TEST_FILENAME  = "recipe_dataset_test.pkl"
DATASET_MAPPINGS_FILENAME     = "recipe_dataset_mappings.json"

NUM_GRAIN_SLOTS = 16
NUM_ADJUNCT_SLOTS = 8
NUM_HOP_SLOTS = 32
NUM_MISC_SLOTS = 16
NUM_MICROORGANISM_SLOTS = 8
NUM_FERMENT_STAGE_SLOTS = 2
NUM_MASH_STEPS = RecipeML.MAX_MASH_STEPS

EMPTY_TAG = "<EMPTY>"

def load_dataset(filepath=RECIPE_DATASET_FILENAME, verbose=True):
  if verbose: print(f"Loading mappings from {filepath}...")
  # Load the dataset and create a dataloader for it
  with open(filepath, 'rb') as f:
    dataset = pickle.load(f)
  return dataset

def save_dataset(dataset, dataset_filepath=RECIPE_DATASET_FILENAME, mappings_filepath=DATASET_MAPPINGS_FILENAME, save_mappings=True, verbose=True):
  if save_mappings:
    if verbose: print(f"Resaving mappings to {mappings_filepath}...")
    mappings = DatasetMappings()
    mappings.init_from_dataset(dataset)
    mappings.save(mappings_filepath)
  if verbose: print(f"Resaving dataset to {dataset_filepath}...")
  with open(dataset_filepath, 'wb') as f:
    pickle.dump(dataset, f)


class _DatasetMappingsEncoder(json.JSONEncoder):
  def default(self, obj):
    if isinstance(obj, DatasetMappings) or isinstance(obj, RunningStats):
      return vars(obj)
    return super().default(obj)

class DatasetMappings():
  def __init__(self, **kwargs) -> None:
    self.core_grains_idx_to_dbid = {}
    self.core_adjs_idx_to_dbid   = {}
    self.hops_idx_to_dbid        = {}
    self.miscs_idx_to_dbid       = {}
    self.mos_idx_to_dbid         = {}

    # Mash step types (e.g., Infusion, Decoction, Temperature)
    self.mash_step_idx_to_name  = {}
    # Misc stage (e.g., Mash, Boil, Primary, ...)
    self.misc_stage_idx_to_name = {}
    # Hop stage (e.g., Mash, Boil, Primary, ...)
    self.hop_stage_idx_to_name  = {}
    # Microorganism stage (e.g., Primary, Secondary)
    self.mo_stage_idx_to_name   = {}

    # Normalization values for the dataset
    self.normalizers = {}

    for key, value in kwargs.items():
      if key in self.__dict__:
        setattr(self, key, value)
        if key == 'normalizers':
          for n_key, normalizer in self.normalizers.items():
            self.normalizers[n_key] = RunningStats(**normalizer)


  def init_from_dataset(self, dataset):
    for key in vars(self).keys():
      dataset_value = getattr(dataset, key, None)
      if dataset_value == None: continue
      setattr(self, key, copy.deepcopy(dataset_value))

  def save(self, filepath=DATASET_MAPPINGS_FILENAME):
    with open(filepath, 'w') as f:
      json.dump(self, fp=f, cls=_DatasetMappingsEncoder)

  @staticmethod
  def load(filepath=DATASET_MAPPINGS_FILENAME):
    assert os.path.exists(filepath)
    with open(filepath, 'r') as f:
      mappings_json = json.load(f)
    return DatasetMappings(**mappings_json)


class RecipeDataset(torch.utils.data.Dataset):

  def __init__(self, dataset_mappings=None):
    self._VERSION = 2
    
    self.recipes = []
    self.block_size = 128
    self.last_saved_idx = 0
    if dataset_mappings != None:
      self.normalizers = dataset_mappings.normalizers

  def __len__(self):
    return len(self.recipes)
  
  def __getitem__(self, idx):
    if torch.is_tensor(idx):
      idx = idx.item()
    recipe = copy.deepcopy(self.recipes[idx])
    
    # Normalize the relevant recipe data in copied arrays
    for key, stats in self.normalizers.items():
      recipe[key] = ((recipe[key] - stats.mean()) / stats.std()).astype(np.float32)
    
    # Sort ingredient slots based on quantities (higher to lower)
    adjunct_new_inds = np.argsort(recipe['adjunct_amts'])[::-1]
    recipe['adjunct_core_type_inds'] = recipe['adjunct_core_type_inds'][adjunct_new_inds]
    recipe['adjunct_amts'] = recipe['adjunct_amts'][adjunct_new_inds]

    hop_new_inds = np.argsort(recipe['hop_concentrations'])[::-1]
    recipe['hop_type_inds'] = recipe['hop_type_inds'][hop_new_inds]
    recipe['hop_times'] = recipe['hop_times'][hop_new_inds]
    recipe['hop_concentrations'] = recipe['hop_concentrations'][hop_new_inds]
    recipe['hop_stage_type_inds'] = recipe['hop_stage_type_inds'][hop_new_inds]

    misc_new_inds = np.argsort(recipe['misc_amts'])[::-1]
    recipe['misc_type_inds'] = recipe['misc_type_inds'][misc_new_inds]
    recipe['misc_amts'] = recipe['misc_amts'][misc_new_inds]
    recipe['misc_times'] = recipe['misc_times'][misc_new_inds]
    recipe['misc_stage_inds'] = recipe['misc_stage_inds'][misc_new_inds]

    '''
    # Move all non-empty slots to the front of each array for various ingredients,
    # shuffle each first to make sure there's no dependance on ordering
    adjunct_shuffle = np.random.permutation(len(recipe['adjunct_core_type_inds']))
    recipe['adjunct_core_type_inds'] = recipe['adjunct_core_type_inds'][adjunct_shuffle]
    recipe['adjunct_amts'] = recipe['adjunct_amts'][adjunct_shuffle]
    non_empty_adjunct_inds = recipe['adjunct_core_type_inds'] != 0
    adjunct_new_inds = np.argsort(non_empty_adjunct_inds)[::-1]
    recipe['adjunct_core_type_inds'] = recipe['adjunct_core_type_inds'][adjunct_new_inds]
    recipe['adjunct_amts'] = recipe['adjunct_amts'][adjunct_new_inds]
    
    hop_shuffle = np.random.permutation(len(recipe['hop_type_inds']))
    recipe['hop_type_inds'] = recipe['hop_type_inds'][hop_shuffle]
    recipe['hop_times'] = recipe['hop_times'][hop_shuffle]
    recipe['hop_concentrations'] = recipe['hop_concentrations'][hop_shuffle]
    recipe['hop_stage_type_inds'] = recipe['hop_stage_type_inds'][hop_shuffle]
    non_empty_hop_inds = recipe['hop_type_inds'] != 0
    hop_new_inds = np.argsort(non_empty_hop_inds)[::-1]
    recipe['hop_type_inds'] = recipe['hop_type_inds'][hop_new_inds]
    recipe['hop_times'] = recipe['hop_times'][hop_new_inds]
    recipe['hop_concentrations'] = recipe['hop_concentrations'][hop_new_inds]
    recipe['hop_stage_type_inds'] = recipe['hop_stage_type_inds'][hop_new_inds]

    misc_shuffle = np.random.permutation(len(recipe['misc_type_inds']))
    recipe['misc_type_inds'] = recipe['misc_type_inds'][misc_shuffle]
    recipe['misc_amts'] = recipe['misc_amts'][misc_shuffle]
    recipe['misc_times'] = recipe['misc_times'][misc_shuffle]
    recipe['misc_stage_inds'] = recipe['misc_stage_inds'][misc_shuffle]
    non_empty_misc_inds = recipe['misc_type_inds'] != 0
    misc_new_inds = np.argsort(non_empty_misc_inds)[::-1]
    recipe['misc_type_inds'] = recipe['misc_type_inds'][misc_new_inds]
    recipe['misc_amts'] = recipe['misc_amts'][misc_new_inds]
    recipe['misc_times'] = recipe['misc_times'][misc_new_inds]
    recipe['misc_stage_inds'] = recipe['misc_stage_inds'][misc_new_inds]

    mo_shuffle = np.random.permutation(len(recipe['mo_type_inds']))
    recipe['mo_type_inds']  = recipe['mo_type_inds'][mo_shuffle]
    recipe['mo_stage_inds'] = recipe['mo_stage_inds'][mo_shuffle]
    non_empty_mo_inds = recipe['mo_type_inds'] != 0
    mo_new_inds = np.argsort(non_empty_mo_inds)[::-1]
    recipe['mo_type_inds']  = recipe['mo_type_inds'][mo_new_inds]
    recipe['mo_stage_inds'] = recipe['mo_stage_inds'][mo_new_inds]
    '''

    return recipe
  
  # Pickle (dump)...
  def __getstate__(self):
    state = self.__dict__.copy()
    return state
  
  # Unpickle (load)...
  def __setstate__(self, newstate):
    self.__dict__.update(newstate)
    if '_VERSION' not in newstate or newstate['_VERSION'] < 2:
      # Update to version 2 - version 1 had no normalization data
      self._VERSION = 2
      self._calc_normalization()
      
  def _calc_normalization(self):
    self.normalizers = {
      'boil_time':   RunningStats(),
      'mash_ph':     RunningStats(),
      'sparge_temp': RunningStats(),
      'mash_step_times': RunningStats(),
      'mash_step_avg_temps': RunningStats(),
      'ferment_stage_times': RunningStats(),
      'ferment_stage_temps': RunningStats(),
      'grain_amts': RunningStats(1, 0.5, 0.25), # all grain amounts are percentages in [0,1]
      'adjunct_amts': RunningStats(),
      'hop_times': RunningStats(),
      'hop_concentrations': RunningStats(),
      'misc_amts': RunningStats(),
      'misc_times': RunningStats(),
    }
    
    #max_hop_concentration = 0.0
    #max_adjunct_amt = 0.0
    #max_misc_amt = 0.0
    for recipe in self.recipes:
      self.normalizers['boil_time'].add(recipe['boil_time'])
      self.normalizers['mash_ph'].add(recipe['mash_ph'])
      self.normalizers['sparge_temp'].add(recipe['sparge_temp'])
      
      valid_step_inds = recipe['mash_step_type_inds'] != 0
      self.normalizers['mash_step_times'].add(recipe['mash_step_times'][valid_step_inds])
      self.normalizers['mash_step_avg_temps'].add(recipe['mash_step_avg_temps'][valid_step_inds])
      
      valid_ferment_inds = recipe['ferment_stage_times'] != 0
      self.normalizers['ferment_stage_times'].add(recipe['ferment_stage_times'][valid_ferment_inds])
      self.normalizers['ferment_stage_temps'].add(recipe['ferment_stage_temps'][valid_ferment_inds])
      
      valid_adj_inds = recipe['adjunct_core_type_inds'] != 0
      #max_adjunct_amt = max(max_adjunct_amt, np.amax(recipe['adjunct_amts']).item())
      self.normalizers['adjunct_amts'].add(recipe['adjunct_amts'][valid_adj_inds])
      
      valid_hop_inds = recipe['hop_type_inds'] != 0
      self.normalizers['hop_times'].add(recipe['hop_times'][valid_hop_inds])
      #max_hop_concentration = max(max_hop_concentration, np.amax(recipe['hop_concentrations']).item())
      self.normalizers['hop_concentrations'].add(recipe['hop_concentrations'][valid_hop_inds])
      
      valid_misc_inds = recipe['misc_type_inds'] != 0
      #max_misc_amt = max(max_misc_amt, np.amax(recipe['misc_amts']).item())
      self.normalizers['misc_amts'].add(recipe['misc_amts'][valid_misc_inds])
      self.normalizers['misc_times'].add(recipe['misc_times'][valid_misc_inds])
  
    #self.normalizers['adjunct_amts'] = RunningStats(1, 0.0, max_adjunct_amt*max_adjunct_amt)
    #self.normalizers['hop_concentrations'] = RunningStats(1, 0.0, max_hop_concentration*max_hop_concentration)
    #self.normalizers['misc_amts'] = RunningStats(1, 0.0, max_misc_amt*max_misc_amt)

  def load_from_db(self, db_engine, options=None):
    # Convert the database into numpy arrays as members of this
    with Session(db_engine) as session:
      # Read all the recipes into numpy format
      self._load_recipes(session, options)

  def _load_recipes(self, session, options):
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
    
    core_grains_dbid_to_idx, self.core_grains_idx_to_dbid = _db_idx_lookups(CoreGrain) # Core Grains
    core_adjs_dbid_to_idx, self.core_adjs_idx_to_dbid = _db_idx_lookups(CoreAdjunct)   # Core Adjuncts
    hops_dbid_to_idx, self.hops_idx_to_dbid = _db_used_only_idx_lookups(RecipeMLHopAT.hop_id) # Hops
    miscs_dbid_to_idx, self.miscs_idx_to_dbid = _db_idx_lookups(Misc) # Miscs
    mos_dbid_to_idx, self.mos_idx_to_dbid = _db_used_only_idx_lookups(RecipeMLMicroorganismAT.microorganism_id) # Microorganisms
    
    # Sub-enumerations
    # Mash step types (e.g., Infusion, Decoction, Temperature)
    mash_step_name_to_idx, self.mash_step_idx_to_name = _build_lookup(_mash_step_types(session))
    # Misc stage (e.g., Mash, Boil, Primary, ...)
    misc_stage_name_to_idx, self.misc_stage_idx_to_name = _build_lookup(_misc_stage_types(session))
    # Hop stage (e.g., Mash, Boil, Primary, ...)
    hop_stage_name_to_idx, self.hop_stage_idx_to_name = _build_lookup(_hop_stage_types(session))
    # Microorganism stage (e.g., Primary, Secondary)
    mo_stage_name_to_idx, self.mo_stage_idx_to_name = _build_lookup(_microorganism_stage_types(session))
    
    # ...Core Styles
    #corestyle_dbids = session.scalars(select(CoreStyle.id)).all()
    #self.core_styles_dbid_to_idx = {csid: i+1 for i, csid in enumerate(corestyle_dbids)}
    #self.core_styles_idx_to_dbid = {i+1: csid for i, csid in enumerate(corestyle_dbids)}
    
    # Depending on options we may want to only load some recipes
    if options != None and 'select_stmt' in options:
      recipe_select_stmt = options['select_stmt']
    else:
      recipe_select_stmt = select(RecipeML)

    # Only load fixed quantities of rows into memory at a time, the recipes table is BIG
    recipe_select_stmt = recipe_select_stmt.execution_options(yield_per=self.block_size)
    
    
    block_idx = 0
    for recipeML_partition in session.scalars(recipe_select_stmt).partitions():
      if block_idx < self.last_saved_idx:
        block_idx += 1
        continue

      for recipeML in recipeML_partition:
        infusion_vol = recipeML.total_infusion_vol()
        recipe_data = {
          'dbid': recipeML.id, # Allows us to re-lookup the recipe if we need more info about it at some point
          'boil_time': recipeML.boil_time,
          'mash_ph': recipeML.mash_ph,
          'sparge_temp': recipeML.sparge_temp,
        }
        
        # Mash steps
        mash_step_type_inds = np.zeros((NUM_MASH_STEPS), dtype=np.int32)   # mash step type (index)
        mash_step_times = np.zeros((NUM_MASH_STEPS), dtype=np.float32)     # mash step time (mins)
        mash_step_avg_temps = np.zeros((NUM_MASH_STEPS), dtype=np.float32) # mash step avg. temperature (C)
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
        ferment_stage_times = np.zeros((NUM_FERMENT_STAGE_SLOTS), dtype=np.float32) # time (in days)
        ferment_stage_temps = np.zeros((NUM_FERMENT_STAGE_SLOTS), dtype=np.float32) # temperature (in C)
        for idx in range(recipeML.num_ferment_stages):
          prefix = "ferment_stage_" + str(idx+1)
          ferment_stage_times[idx] = getattr(recipeML, prefix+"_time")
          ferment_stage_temps[idx] = getattr(recipeML, prefix+"_temp")
        recipe_data['ferment_stage_times'] = ferment_stage_times
        recipe_data['ferment_stage_temps'] = ferment_stage_temps
        
        # Grains
        grain_core_type_inds = np.zeros((NUM_GRAIN_SLOTS), dtype=np.int32) # core grain type (index)
        grain_amts = np.zeros((NUM_GRAIN_SLOTS), dtype=np.float32)         # amount (as a %)
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
        adjunct_core_type_inds = np.zeros((NUM_ADJUNCT_SLOTS), dtype=np.int32) # core adjunct type (index)
        adjunct_amts = np.zeros((NUM_ADJUNCT_SLOTS), dtype=np.float32)         # amount (in ~(g or ml)/L)
        for idx, adjunctAT in enumerate(recipeML.adjuncts):
          assert adjunctAT.adjunct.core_adjunct_id != None
          adjunct_core_type_inds[idx] = core_adjs_dbid_to_idx[adjunctAT.adjunct.core_adjunct_id]
          vol = recipeML.fermenter_vol if adjunctAT.stage == None else _recipe_vol_at_stage(recipeML, infusion_vol, adjunctAT.stage)
          assert vol != None and vol > 0
          adjunct_amts[idx] = (adjunctAT.amount * 1000.0) / vol
        recipe_data['adjunct_core_type_inds'] = adjunct_core_type_inds
        recipe_data['adjunct_amts'] = adjunct_amts
        
        # Hops
        hop_type_inds = np.zeros((NUM_HOP_SLOTS), dtype=np.int32)        # hop type (index)
        hop_stage_type_inds = np.zeros((NUM_HOP_SLOTS), dtype=np.int32)  # hop use/stage (index)
        hop_times = np.zeros((NUM_HOP_SLOTS), dtype=np.float32)          # time as a multiple of stage time
        hop_concentrations = np.zeros((NUM_HOP_SLOTS), dtype=np.float32) # if this is a boil hop then the amount is a (concentration of alpha acids in g/L), otherwise it's the hop concentration in g/L
        for idx, hopAT in enumerate(recipeML.hops):
          hop_type_inds[idx] = hops_dbid_to_idx[hopAT.hop_id]
          hop_stage_type_inds[idx] = hop_stage_name_to_idx[hopAT.stage]
          assert hopAT.time >= 0
          time_div, max_time = recipe_time_at_stage(recipe_data, hopAT.stage)
          assert time_div > 0
          hop_times[idx] = min(hopAT.time, max_time) / time_div
          #if hopAT.stage == 'boil':
          #  hop_concentrations[idx] = hop_form_utilization(hopAT.form) * alpha_acid_mg_per_l(hopAT.alpha / 100.0, hopAT.amount * 1000.0, recipeML.postboil_vol) / 1000.0
          #else: 
          hop_concentrations[idx] = (hopAT.amount * 1000.0) / _recipe_vol_at_stage(recipeML, infusion_vol, hopAT.stage)
        recipe_data['hop_type_inds'] = hop_type_inds
        recipe_data['hop_stage_type_inds'] = hop_stage_type_inds
        recipe_data['hop_times'] = hop_times
        recipe_data['hop_concentrations'] = hop_concentrations

        # Miscs
        misc_type_inds = np.zeros((NUM_MISC_SLOTS), dtype=np.int32)  # misc type (index)
        misc_amts = np.zeros((NUM_MISC_SLOTS), dtype=np.float32)     # amount (in ~(g or ml)/L)
        misc_times = np.zeros((NUM_MISC_SLOTS), dtype=np.float32)    # time as a multiple of stage time
        misc_stage_inds = np.zeros((NUM_MISC_SLOTS), dtype=np.int32) # stage (index)
        for idx, miscAT in enumerate(recipeML.miscs):
          misc_type_inds[idx] = miscs_dbid_to_idx[miscAT.misc_id]
          vol =  _recipe_vol_at_stage(recipeML, infusion_vol, miscAT.stage)
          assert vol != None and vol > 0
          misc_amts[idx] = (miscAT.amount * 1000.0) / vol
          assert miscAT.time >= 0
          time_div, max_time = recipe_time_at_stage(recipe_data, miscAT.stage)
          assert time_div > 0
          misc_times[idx] = min(miscAT.time, max_time) / time_div
          misc_stage_inds[idx] = misc_stage_name_to_idx[miscAT.stage]
        recipe_data['misc_type_inds'] = misc_type_inds
        recipe_data['misc_amts'] = misc_amts
        recipe_data['misc_times'] = misc_times
        recipe_data['misc_stage_inds'] = misc_stage_inds
        
        # Microorganisms
        mo_type_inds  = np.zeros((NUM_MICROORGANISM_SLOTS), dtype=np.int32)
        mo_stage_inds = np.zeros((NUM_MICROORGANISM_SLOTS), dtype=np.int32)
        for idx, moAT in enumerate(recipeML.microorganisms):
          mo_type_inds[idx]  = mos_dbid_to_idx[moAT.microorganism_id]
          mo_stage_inds[idx] = mo_stage_name_to_idx[moAT.stage]
        recipe_data['mo_type_inds'] = mo_type_inds
        recipe_data['mo_stage_inds'] = mo_stage_inds
        
        self.recipes.append(recipe_data)
        
      block_idx += 1
      if options != None and 'filename' in options:
        if 'write_every_blocks' not in options or block_idx % options['write_every_blocks'] == 0:
          filename = options['filename']
          print(f"Writing/Overwriting file {filename}, at block index: {block_idx}")
          with open(filename, 'wb') as f:
            self.last_saved_idx = block_idx
            pickle.dump(self, f)
    
    # Calculate the normalizers if the options specify it
    if options != None and 'calc_normalizers' in options and options['calc_normalizers'] == True:
      self._calc_normalization()
    
    # Make sure we save the final dataset (if pickle options are enabled)        
    if options != None and 'filename' in options:
      filename = options['filename']
      print(f"Finished loading dataset, writing final cached data to file {filename}")
      with open(filename, 'wb') as f:
        self.last_saved_idx = block_idx
        pickle.dump(self, f)


def _db_table_labels(db_engine, orm_class, idx_to_dbid_lookup):
  result = [EMPTY_TAG]
  with Session(db_engine) as session:
    result += session.scalars(select(orm_class.name).filter(orm_class.id.in_(idx_to_dbid_lookup.values()))).all()
  return result

def core_grain_labels(db_engine, dataset):
  return _db_table_labels(db_engine, CoreGrain, dataset.core_grains_idx_to_dbid)
def core_adjunct_labels(db_engine, dataset):
  return _db_table_labels(db_engine, CoreAdjunct, dataset.core_adjs_idx_to_dbid)
def hop_labels(db_engine, dataset):
  return _db_table_labels(db_engine, Hop, dataset.hops_idx_to_dbid)
def misc_labels(db_engine, dataset):
  return _db_table_labels(db_engine, Misc, dataset.miscs_idx_to_dbid)
def microorganism_labels(db_engine, dataset):
  return _db_table_labels(db_engine, Microorganism, dataset.mos_idx_to_dbid)


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

# Volume in Liters
def _recipe_vol_at_stage(recipe_ml, infusion_vol, stage_name):
  if stage_name == 'mash':
    assert infusion_vol > 0, "Infusion volume must be greater than zero!"
    return infusion_vol
  elif stage_name == 'sparge' or stage_name == 'boil' or stage_name == 'first wort' or stage_name == 'whirlpool':
    return recipe_ml.postboil_vol
  else:
    assert stage_name == 'dry hop' or stage_name == 'primary' or stage_name == 'secondary' or stage_name == 'other' or stage_name == 'finishing'
    return recipe_ml.fermenter_vol

# Reasonable times in minutes for a given stage in a given recipe
# Returns a tuple (reasonable_time, max_possible_time)
def recipe_time_at_stage(recipe, stage_name):
  if stage_name == 'mash':
    mash_time = recipe['mash_step_times'].sum()
    return (mash_time, mash_time)
  elif stage_name == 'finishing':
    return (60, 180)
  elif stage_name == 'boil' or stage_name == 'first wort':
    return (recipe['boil_time'], recipe['boil_time'])
  elif stage_name == 'primary':
    primary_mins = recipe['ferment_stage_times'][0]*1440 # days -> mins
    return (primary_mins, primary_mins)
  elif stage_name == 'secondary':
    secondary_mins = recipe['ferment_stage_times'][1]*1440 # days -> mins
    return (secondary_mins, secondary_mins)
  elif stage_name == 'dry hop' or stage_name == 'other':
    ferment_mins = recipe['ferment_stage_times'].sum()*1440 # days -> mins
    return (ferment_mins, ferment_mins)
  elif stage_name == 'sparge':
    return (60, 180)  # Typical additions to a sparge are within an hour
  elif stage_name == 'whirlpool':
    return (60, 180) # Set the typical whirlpool to be about an hour
  assert False


def load_save_recalc_normalizations():
  dataset = load_dataset()
  print("Recalculating normalizations...")
  dataset._calc_normalization()
  save_dataset(dataset)

def remove_dbids(dataset, session, dbids_to_remove):
  print(f"Removing {len(dbids_to_remove)} recipes...")
  dataset.recipes = [recipe for recipe in dataset.recipes if recipe['dbid'] not in dbids_to_remove]
  print("Recalculating normalizations...")
  dataset._calc_normalization()
  recipes_to_remove = session.scalars(select(RecipeML).filter(RecipeML.id.in_(dbids_to_remove))).all()
  for recipe in recipes_to_remove:
    session.delete(recipe)
  session.commit()

if __name__ == "__main__":
  dataset = load_dataset()
  dbids_to_remove = [r['dbid'] for r in dataset.recipes if np.any(r['misc_amts'] > 200)]#[recipe['dbid'] for recipe in dataset.recipes if np.any(recipe['hop_concentrations'] > 25) or np.any(recipe['adjunct_amts'] > 1000) or np.any(recipe['misc_amts']) > 200]

  from sqlalchemy import create_engine
  from brewbrain_db import BREWBRAIN_DB_ENGINE_STR, Base, RecipeML
  engine = create_engine(BREWBRAIN_DB_ENGINE_STR, echo=False, future=True)
  Base.metadata.create_all(engine)
  with Session(engine) as session:
    '''
    db_recipes = session.scalars(select(RecipeML)).all()
    db_recipes_map = {r.id: r for r in db_recipes}

    # Clean up hop concentrations... fix the boil hops to use g/L
    boil_idx = -1
    for idx, name in dataset.hop_stage_idx_to_name.items():
      if name == 'boil':
        boil_idx = idx
        break
    assert boil_idx != -1
    dbids_to_remove = set()
    for recipe in dataset.recipes:
      # Are there boil hops?
      boil_hop_inds = recipe['hop_stage_type_inds'] == boil_idx
      if not np.any(boil_hop_inds): continue

      # Change boil hop concentrations to g/L
      if recipe['dbid'] not in db_recipes_map:
        dbids_to_remove.add(recipe['dbid'])
        continue

      recipe_ml = db_recipes_map[recipe['dbid']]
      assert recipe_ml != None, f"No recipe found for dbid {recipe['dbid']}!"
      infusion_vol = recipe_ml.total_infusion_vol()
      for idx, hop_at in enumerate(recipe_ml.hops):
        recipe['hop_concentrations'][idx] = (hop_at.amount * 1000.0) / _recipe_vol_at_stage(recipe_ml, infusion_vol, hop_at.stage)
        if recipe['hop_concentrations'][idx] > 25:
          dbids_to_remove.add(recipe['dbid'])
    '''
    remove_dbids(dataset, session, dbids_to_remove) 
    save_dataset(dataset)
    
  exit()

  


  '''
  # Clean up the timings on hop and misc:
  # Remove all recipes with negative times
  dataset.recipes = [recipe for recipe in dataset.recipes if (recipe['hop_times'] >= 0).all() and (recipe['misc_times'] >= 0).all()]
  print(f"After removal of negative timings: {len(dataset.recipes)} recipes.")

  # Set all hop and misc. addition times clamped and set to a multiple of their stage time
  dbids_to_remove = set()
  for recipe in dataset.recipes:
    for i in np.nonzero(recipe['hop_type_inds'])[0]:
      stage = dataset.hop_stage_idx_to_name[recipe['hop_stage_type_inds'][i]]
      div_time, max_time = recipe_time_at_stage(recipe, stage)
      if div_time <= 0:
        dbids_to_remove.add(recipe['dbid'])
        break
      recipe['hop_times'][i] = min(recipe['hop_times'][i], max_time) / div_time
    for i in np.nonzero(recipe['misc_type_inds'])[0]:
      stage = dataset.misc_stage_idx_to_name[recipe['misc_stage_inds'][i]]
      div_time, max_time = recipe_time_at_stage(recipe, stage)
      if div_time <= 0:
        dbids_to_remove.add(recipe['dbid'])
        break
      recipe['misc_times'][i] = min(recipe['misc_times'][i], max_time) / div_time

  # Remove bogus recipes (where we couldn't properly set the hop/misc times)
  dataset.recipes = [recipe for recipe in dataset.recipes if recipe['dbid'] not in dbids_to_remove]
  print(f"After removal of erroroneous recipes: {len(dataset.recipes)} recipes.")

  # Recalculate all the normalization values for the hop and misc. times
  dataset.normalizers['hop_times']  = RunningStats()
  dataset.normalizers['misc_times'] = RunningStats()
  for recipe in dataset.recipes:
    dataset.normalizers['hop_times'].add(recipe['hop_times'][recipe['hop_type_inds'] != 0])
    dataset.normalizers['misc_times'].add(recipe['misc_times'][recipe['misc_type_inds'] != 0])

  print("New normalizations...")
  print(f"Hop Times:  [mean={dataset.normalizers['hop_times'].mean():>10.3f}, std={dataset.normalizers['hop_times'].std():>10.3f}]")
  print(f"Misc Times: [mean={dataset.normalizers['misc_times'].mean():>10.3f}, std={dataset.normalizers['misc_times'].std():>10.3f}]")

  print("Resaving mappings...")
  mappings = DatasetMappings()
  mappings.init_from_dataset(dataset)
  mappings.save()

  print("Resaving dataset...")
  with open(RECIPE_DATASET_FILENAME, 'wb') as f:
    pickle.dump(dataset, f)

  #mappings = DatasetMappings()
  #mappings.init_from_dataset(dataset)
  #mappings.save()
  #loaded_mappings = DatasetMappings.load()
  '''

  # Set of dbids to remove based on network testing for outliers...
  remove_dbids = set(
    [
      138918,196206,215956,221149,133902,211680,215135,223880,231843,23052, 233061,214458,102975,25209, 88871, 290456,21840, 32398, 6709,  44007, 131139,210853,29271, 254751,221338,226719,273031,219908,22601, 20541, 119635,197266,27991, 233555,45011, 219349,50315, 105705,220452,194885,200975,28019, 21844, 249832,40956, 143032,143709,222311,223045,166997,
    ]
  )

  # Remove the ids from the dataset
  dataset.recipes = [recipe for recipe in dataset.recipes if recipe['dbid'] not in remove_dbids]
  print(f"After removal of specified dbids: {len(dataset.recipes)} recipes.")

  print("Recalculating normalizations...")
  dataset._calc_normalization()

  print("Resaving mappings...")
  mappings = DatasetMappings()
  mappings.init_from_dataset(dataset)
  mappings.save()

  print("Resaving dataset...")
  with open(RECIPE_DATASET_FILENAME, 'wb') as f:
    pickle.dump(dataset, f)

  # Remove the ids from the DB
  print("Removing specified dbids from database...")
  from sqlalchemy import create_engine
  from brewbrain_db import BREWBRAIN_DB_ENGINE_STR, Base
  engine = create_engine(BREWBRAIN_DB_ENGINE_STR, echo=False, future=True)
  Base.metadata.create_all(engine)
  with Session(engine) as session:
    recipes_to_remove = session.scalars(select(RecipeML).filter(RecipeML.id.in_(remove_dbids))).all()
    for recipe in recipes_to_remove:
      session.delete(recipe)
    session.commit()

  '''
  from torch.utils.data import DataLoader
  dataloader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=0)
  for batch_idx, batch in enumerate(dataloader):
    if batch_idx == 1:
      print(batch.__dict__)
      break
  


  # Convert the database into a pickled file so that we can quickly load everything into memory for training
  from sqlalchemy import create_engine
  from db_scripts.brewbrain_db import BREWBRAIN_DB_ENGINE_STR, Base
  engine = create_engine(BREWBRAIN_DB_ENGINE_STR, echo=False, future=True)
  Base.metadata.create_all(engine)
    
  options = {
    'calc_normalizers': True,
    'filename': RECIPE_DATASET_FILENAME,
    'write_every_blocks': 1
  }

  # Read in whatever has been saved from the dataset up to this point
  if os.path.exists(options['filename']):
    with open(options['filename'], 'rb') as f:
      dataset = pickle.load(f)
  else:
    dataset = RecipeDataset()

  # Continue loading and writing the dataset to disk
  dataset.load_from_db(engine, options)
  '''
