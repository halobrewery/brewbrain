
import random
from collections import defaultdict
from brewbrain_db import Style, CoreStyle, RecipeML

class StyleDistribution:
  def __init__(self, dbid:int, style_name:str) -> None:
    self.dbid = dbid
    self.style_name = style_name
    self.dist_map = defaultdict(lambda: [])

  def add_dist_value(self, attr, value):
    self.dist_map[attr].append(value)

  def sample_mash_ph(self) -> float:
    return self.sample_named_value('mash_ph')
  def sample_sparge_temp(self) -> float:
    return self.sample_named_value('sparge_temp')
  def sample_num_mash_steps(self) -> int:
    return self.sample_named_value('num_mash_steps')

  def sample_named_value(self, arr_name):
    arr = self.dist_map[arr_name]
    return random.choice(arr) if len(arr) > 0 else None
  
  def sample_named_value_cond(self, arr_name, cond_name, cond_value):
    values = self.sample_named_values_cond([arr_name], cond_name, cond_value)
    return values[0] if len(values) > 0 else None

  def sample_named_values_cond(self, arr_names, cond_name, cond_value):
    idx_choice_tuples = [(i,c) for i, c in enumerate(self.dist_map[cond_name]) if c == cond_value]
    if len(idx_choice_tuples) == 0: return [None for val in arr_names]
    choice = random.choice(idx_choice_tuples)
    return [self.dist_map[val][choice[0]] for val in arr_names]


  def sample_named_values_conds(self, arr_names, conds_name_value_tuples):
    idx_choices = []
    for i, dist in enumerate(list(zip(*[self.dist_map[cond_name] for cond_name, _ in conds_name_value_tuples]))):
      if all([v == conds_name_value_tuples[tuple_idx][1] for tuple_idx, v in enumerate(dist)]):
        idx_choices.append(i)

    idx_choices = [idx for idx in idx_choices if all([idx < len(self.dist_map[name]) for name in arr_names])]
    if len(idx_choices) == 0: return [None for _ in arr_names]
    choice = random.choice(idx_choices)
    return [self.dist_map[val][choice] for val in arr_names]

def _dist_by_style(style_id_name_recipe_db):
  dist_map = {}
  for styleid, stylename, recipe in style_id_name_recipe_db:
    if styleid not in dist_map:
      dist_map[styleid] = StyleDistribution(styleid, stylename)

    curr_dist = dist_map[styleid]
    assert recipe.mash_ph != None, "NULL mash pH found."
    curr_dist.add_dist_value('mash_ph', round(recipe.mash_ph, 2))
    assert recipe.sparge_temp != None, "NULL sparge temperature found."
    curr_dist.add_dist_value('sparge_temp', recipe.sparge_temp)
    assert recipe.num_mash_steps != 0 and recipe.num_mash_steps != None, "Empty number of mash steps found."
    curr_dist.add_dist_value('num_mash_steps', recipe.num_mash_steps)
    assert recipe.preboil_vol != None, "Empty preboil volume found."
    curr_dist.add_dist_value('preboil_vol', recipe.preboil_vol)

    for step_num in range(1,RecipeML.MAX_MASH_STEPS+1):
      prefix = RecipeML.MASH_STEP_PREFIX+str(step_num)
      for postfix in RecipeML.MASH_STEP_POSTFIXES:
        attr_name = prefix+postfix
        value = getattr(recipe, attr_name)
        if value != None:
          curr_dist.add_dist_value(attr_name, value)

  return dist_map

def distributions_by_style_id(session):
  style_values = session.query(Style.id, Style.name, RecipeML).join(RecipeML, RecipeML.style_id == Style.id).all()
  return _dist_by_style(style_values)

def distributions_by_core_style_id(session):
  core_style_values = session.query(CoreStyle.id, CoreStyle.name, RecipeML) \
    .join(Style, RecipeML.style_id == Style.id) \
    .join(CoreStyle, Style.core_style_id == CoreStyle.id).all()
  return _dist_by_style(core_style_values)
