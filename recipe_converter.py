import torch
import numpy as np

from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy import select

from recipe_dataset import DatasetMappings, recipe_time_at_stage
from brewbrain_db import BREWBRAIN_DB_ENGINE_STR, Base, Hop, Misc, Microorganism, CoreGrain, CoreAdjunct

class RecipeConverter():

  def __init__(self, dataset_mappings: DatasetMappings) -> None:
    self.dataset_mappings = dataset_mappings
    self.db_engine = create_engine(BREWBRAIN_DB_ENGINE_STR, echo=False, future=True)
    Base.metadata.create_all(self.db_engine)


  @torch.no_grad()
  def batch_to_recipes(self, ds_recipes):
    num_recipes = ds_recipes['boil_time'].shape[0]
    recipes = []
    for i in range(num_recipes):
      recipe = {}
      for key, value in ds_recipes.items():
        recipe[key] = value[i].detach().cpu().numpy()
        if key in self.dataset_mappings.normalizers:
          normalizer = self.dataset_mappings.normalizers[key]
          recipe[key] = normalizer.std() * recipe[key] + normalizer.mean()

      # Clean up some of the recipe data...

      # Make the boil time a multiple of 5 mins
      recipe['boil_time'] = np.round(recipe['boil_time'].item()/5) * 5
      # Round the mash pH to the nearest 100ths
      recipe['mash_ph'] = np.round(recipe['mash_ph'].item(), 2)
      # Round the sparge temp to the nearest 10th of a degree
      recipe['sparge_temp'] = np.round(recipe['sparge_temp'].item(), 1)

      # Mash steps should only exist for non-empty steps
      invalid_mash_step_inds = recipe['mash_step_type_inds'] == 0
      recipe['mash_step_times'][invalid_mash_step_inds] = 0
      recipe['mash_step_avg_temps'][invalid_mash_step_inds] = 0
      # Round mash step times to the nearest 5 mins
      recipe['mash_step_times'] = np.round(recipe['mash_step_times']/5) * 5
      # Round mash step temps to the nearest 10th of a degree
      recipe['mash_step_avg_temps'] = np.round(recipe['mash_step_avg_temps'], 1)

      # Grain amounts should only exist for non-empty slots and 
      # are proper percentages that add up to 1
      invalid_grain_inds = recipe['grain_core_type_inds'] == 0
      recipe['grain_amts'][invalid_grain_inds] = 0
      recipe['grain_amts'] /= recipe['grain_amts'].sum()

      # Round fermentation stage time to the nearest day
      recipe['ferment_stage_times'] = np.round(np.clip(recipe['ferment_stage_times'], a_min=0.0, a_max=None))
      # Fermentation stages should only exist for positive times
      invalid_ferment_stage_inds = recipe['ferment_stage_times'] <= 0
      recipe['ferment_stage_temps'][invalid_ferment_stage_inds] = 0
      # Round fermentation temperatures to the nearest 10th of a degree
      recipe['ferment_stage_temps'] = np.round(recipe['ferment_stage_temps'], 1)
      
      # Adjunct amounts should only exist for non-empty slots
      invalid_adj_inds = recipe['adjunct_core_type_inds'] == 0
      recipe['adjunct_amts'][invalid_adj_inds] = 0

      # Hop values should only exist for non-empty slots
      invalid_hop_inds = recipe['hop_type_inds'] == 0
      recipe['hop_stage_type_inds'][invalid_hop_inds] = 0
      recipe['hop_stage_type_names'] = self.hop_stage_type_names(recipe['hop_stage_type_inds'])
      recipe['hop_times'][invalid_hop_inds] = 0
      recipe['hop_concentrations'][invalid_hop_inds] = 0
      # Convert times back into mins
      for i in np.nonzero(recipe['hop_type_inds'])[0]:
        stage = recipe['hop_stage_type_names'][i]
        time_div, _ = recipe_time_at_stage(recipe, stage)
        recipe['hop_times'][i] *= time_div
      
      # Misc. values should only exist for non-empty slots
      invalid_misc_inds = recipe['misc_type_inds'] == 0
      recipe['misc_stage_inds'][invalid_misc_inds] = 0
      recipe['misc_stage_names'] = self.misc_stage_type_names(recipe['misc_stage_inds'])
      recipe['misc_times'][invalid_misc_inds] = 0
      recipe['misc_amts'][invalid_misc_inds] = 0
      # Convert times back into mins
      for i in np.nonzero(recipe['misc_type_inds'])[0]:
        stage = recipe['misc_stage_names'][i]
        time_div, _ = recipe_time_at_stage(recipe, stage)
        recipe['misc_times'][i] *= time_div
      
      # Microorganism values should only exist for non-empty slots
      invalid_mo_inds = recipe['mo_type_inds'] == 0
      recipe['mo_stage_inds'][invalid_mo_inds] = 0

      recipes.append(recipe)

    return recipes

  @torch.no_grad()
  def net_output_to_recipes(self, foots):
    t_recipes = {}
    t_recipes['boil_time'], t_recipes['mash_ph'], t_recipes['sparge_temp'] = torch.chunk(foots.x_hat_toplvl, 3, dim=1)
    num_recipes = t_recipes['boil_time'].shape[0]

    t_recipes['mash_step_type_inds'] = torch.argmax(torch.softmax(foots.dec_mash_step_type_onehot.view(num_recipes, foots.dec_mash_step_times.shape[1], -1), dim=-1), dim=-1, keepdim=False)
    t_recipes['mash_step_times']     = foots.dec_mash_step_times
    t_recipes['mash_step_avg_temps'] = foots.dec_mash_step_avg_temps
    t_recipes['ferment_stage_times'], t_recipes['ferment_stage_temps'] = torch.chunk(foots.x_hat_ferment_stages, 2, dim=1)
    t_recipes['grain_core_type_inds'] = torch.argmax(torch.softmax(foots.dec_grain_type_logits, dim=-1), dim=-1, keepdim=False)
    t_recipes['grain_amts'] = foots.dec_grain_amts
    t_recipes['adjunct_core_type_inds'] = torch.argmax(torch.softmax(foots.dec_adjunct_type_logits, dim=-1), dim=-1, keepdim=False)
    t_recipes['adjunct_amts'] = foots.dec_adjunct_amts
    t_recipes['hop_type_inds'] = torch.argmax(torch.softmax(foots.dec_hop_type_logits, dim=-1), dim=-1, keepdim=False)
    t_recipes['hop_stage_type_inds'] = torch.argmax(torch.softmax(foots.dec_hop_stage_type_onehot.view(num_recipes, foots.dec_hop_times.shape[1], -1), dim=-1), dim=-1, keepdim=False)
    t_recipes['hop_times'] = foots.dec_hop_times
    t_recipes['hop_concentrations'] = foots.dec_hop_concentrations
    t_recipes['misc_type_inds']  = torch.argmax(torch.softmax(foots.dec_misc_type_logits, dim=-1), dim=-1, keepdim=False)
    t_recipes['misc_stage_inds'] = torch.argmax(torch.softmax(foots.dec_misc_stage_type_onehot.view(num_recipes, foots.dec_misc_amts.shape[1], -1), dim=-1), dim=-1, keepdim=False)
    t_recipes['misc_times'] = foots.dec_misc_times
    t_recipes['misc_amts'] = foots.dec_misc_amts
    t_recipes['mo_type_inds'] = torch.argmax(torch.softmax(foots.dec_mo_type_logits, dim=-1), dim=-1, keepdim=False)
    t_recipes['mo_stage_inds'] = torch.argmax(torch.softmax(foots.dec_mo_stage_type_onehot.view(num_recipes, t_recipes['mo_type_inds'].shape[1], -1), dim=-1), dim=-1, keepdim=False)
    
    return self.batch_to_recipes(t_recipes)

  def mash_step_type_names(self, mash_step_type_inds):
    return [self.dataset_mappings.mash_step_idx_to_name[str(idx)] for idx in mash_step_type_inds]
  
  def hop_stage_type_names(self, hop_stage_type_inds):
    return [self.dataset_mappings.hop_stage_idx_to_name[str(idx)] for idx in hop_stage_type_inds]
  
  def misc_stage_type_names(self, misc_stage_type_inds):
    return [self.dataset_mappings.misc_stage_idx_to_name[str(idx)] for idx in misc_stage_type_inds]

  def microorganism_stage_type_names(self, mo_stage_type_inds):
    return [self.dataset_mappings.mo_stage_idx_to_name[str(idx)] for idx in mo_stage_type_inds]

  def _type_names(self, type_inds, idx_to_dbid_map, orm_table):
    with Session(self.db_engine) as session:
      dbids = [idx_to_dbid_map[str(idx)] for idx in type_inds]
      values = session.query(orm_table).with_entities(orm_table.id, orm_table.name).filter(orm_table.id.in_(dbids)).all()
      value_map = {dbid:name for dbid, name in values}
      return [value_map[dbid] for dbid in dbids]

  def grain_type_names(self, grain_type_inds):
    return self._type_names(grain_type_inds, self.dataset_mappings.core_grains_idx_to_dbid, CoreGrain)

  def adjunct_type_names(self, adjunct_type_inds):
    return self._type_names(adjunct_type_inds, self.dataset_mappings.core_adjs_idx_to_dbid, CoreAdjunct)

  def hop_type_names(self, hop_type_inds):
    return self._type_names(hop_type_inds, self.dataset_mappings.hops_idx_to_dbid, Hop)

  def misc_type_names(self, misc_type_inds):
    return self._type_names(misc_type_inds, self.dataset_mappings.miscs_idx_to_dbid, Misc)

  def microorganism_type_names(self, mo_type_inds):
    return self._type_names(mo_type_inds, self.dataset_mappings.mos_idx_to_dbid, Microorganism)
