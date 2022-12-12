import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy import select, func
from sqlalchemy import or_, and_

from brewbrain_db import BREWBRAIN_DB_ENGINE_STR, Base, RecipeML, Misc, Adjunct, RecipeMLGrainAT, RecipeMLAdjunctAT, RecipeMLHopAT, RecipeMLMiscAT
from distributions import distributions_by_style_id

def remove_zero_mash_or_ferment_step_recipes(session):
  bad_recipes = session.query(RecipeML).filter(or_(RecipeML.num_mash_steps == 0, RecipeML.num_ferment_stages == 0)).all()
  for recipe in bad_recipes:
    session.delete(recipe)

def clean_up_recipe_mash_steps(recipe, recalc_hash=True):
  DEFAULT_GRAIN_TO_WATER_RATIO = 2.6 # L/Kg
  GRAIN_ABSORB_L_PER_KG = 1.00144835

  # If this recipe is reasonable then the total infusion volume should add up to AT LEAST the grain absorption
  total_infusion_vol = recipe.total_infusion_vol()
  total_grain_mass   = recipe.total_grain_mass()
  grain_absorb_vol   = GRAIN_ABSORB_L_PER_KG * total_grain_mass
  if total_infusion_vol <= grain_absorb_vol:
    # This recipe can't possibly have enough water in its mash... 
    # Try to calculate an appropriate water volume as long as there's only one mash step to deal with
    if recipe.num_mash_steps == 1 and recipe.mash_step_1_start_temp != None and (recipe.mash_step_1_start_temp >= 60 or recipe.mash_step_1_end_temp >= 60):
      recipe.mash_step_1_type = "infusion"
      recipe.mash_step_1_infuse_amt = total_grain_mass * DEFAULT_GRAIN_TO_WATER_RATIO
      recipe.mash_step_1_time = max(45, recipe.mash_step_1_time)
      if recalc_hash:
        recipe.hash = recipe.gen_hash()
      return True
    else:
      return False

  # If the first infuse step is enough to actually soak the grains then we're ok
  if recipe.mash_step_1_infuse_amt != None and recipe.mash_step_1_infuse_amt > grain_absorb_vol:
    return True
  
  # There appears enough water to make the mash happen, we just need to find it 
  # in the other infusion steps and move it into the first infusion step... 
  if recipe.num_mash_steps == 1 and recipe.mash_step_1_type != "infusion":
    return False
  
  # First thing we need to do is to make the initial mash step an infusion (can't do a mash without water!)
  # Heating up the grains is not a relevant step for our purposes
  if recipe.mash_step_1_type != "infusion":
    recipe.mash_step_1_type = "infusion"
    recipe.mash_step_1_infuse_amt = max(0, recipe.mash_step_1_infuse_amt) if recipe.mash_step_1_infuse_amt != None else 0

  def remove_mash_step(step_idx):
    assert step_idx >= 1 and step_idx <= recipe.num_mash_steps
    for idx in range(step_idx, recipe.num_mash_steps+1):
      prefix_prev = RecipeML.MASH_STEP_PREFIX+str(idx)
      if idx == 6:
        for attr in RecipeML.MASH_STEP_POSTFIXES:
          setattr(recipe, prefix_prev+attr, None)
      else:
        prefix_curr = RecipeML.MASH_STEP_PREFIX+str(idx+1)
        for attr in RecipeML.MASH_STEP_POSTFIXES:
          setattr(recipe, prefix_prev+attr, getattr(recipe, prefix_curr+attr))
    recipe.num_mash_steps -= 1

  curr_infuse_amt  = recipe.mash_step_1_infuse_amt or 0
  curr_infuse_start_temp = recipe.mash_step_1_start_temp
  curr_infuse_end_temp = recipe.mash_step_1_end_temp
  curr_infuse_time = recipe.mash_step_1_time
  i = 2
  first_infusion_fixed = False
  steps_inbetween = False
  while i <= recipe.num_mash_steps:
    prefix = RecipeML.MASH_STEP_PREFIX+str(i)
    type = getattr(recipe, prefix+"_type")
    if type == "infusion":
      # If the infusion amount is zero then we need to get rid of this step
      infuse_amt = getattr(recipe, prefix+"_infuse_amt") or 0
      if not first_infusion_fixed:
        # Merge this mash step with the first mash step (infusion) and remove this step
        curr_infuse_amt += infuse_amt
        curr_infuse_start_temp = max(curr_infuse_start_temp, getattr(recipe, prefix+"_start_temp"))
        curr_infuse_end_temp = max(curr_infuse_end_temp, getattr(recipe, prefix+"_end_temp"))
        curr_infuse_time += getattr(recipe, prefix+"_time")
        first_infusion_fixed = curr_infuse_amt > grain_absorb_vol # Update whether there's a reasonable amount of water yet
        remove_mash_step(i)
        continue
      if infuse_amt == None or infuse_amt <= 0:
        # This should probably be a temperature step, not an infuse step.
        if not steps_inbetween:
          # If there were no steps between this one and the first mash step then merge this infusion with the first one
          curr_infuse_start_temp =  max(curr_infuse_start_temp, getattr(recipe, prefix+"_start_temp"))
          curr_infuse_end_temp = max(curr_infuse_end_temp, getattr(recipe, prefix+"_end_temp"))
          curr_infuse_time += getattr(recipe, prefix+"_time")
          remove_mash_step(i)
          continue
        else:
          # Change this step to be a temperature step
          setattr(recipe, prefix+"_type", "temperature")
    else:
      steps_inbetween = True
    i += 1

  if first_infusion_fixed:
    recipe.mash_step_1_time = curr_infuse_time
    recipe.mash_step_1_start_temp = curr_infuse_start_temp
    recipe.mash_step_1_end_temp = curr_infuse_end_temp
    recipe.mash_step_1_infuse_amt = curr_infuse_amt
  
  if recalc_hash:
    recipe.hash = recipe.gen_hash()
  return True


def clean_up_mash_steps(session):
  MAX_L_PER_KG_DECOC = 3.12953
  MIN_L_PER_KG_DECOC = 2.607939
  
  # Find all starting decoctions or temperatures and just remove them
  bad_recipes = session.query(RecipeML).filter(or_(RecipeML.mash_step_1_type == "temperature", RecipeML.mash_step_1_type == "decoction")).all()
  for recipe in bad_recipes:
    session.delete(recipe)
  session.commit()

  bad_recipes = session.query(RecipeML).filter(or_(
    RecipeML.mash_step_1_time > 180, RecipeML.mash_step_2_time > 180, RecipeML.mash_step_3_time > 180,
    RecipeML.mash_step_4_time > 180, RecipeML.mash_step_5_time > 180, RecipeML.mash_step_6_time > 180,
    RecipeML.mash_step_1_start_temp > 78, RecipeML.mash_step_2_start_temp > 78, RecipeML.mash_step_3_start_temp > 78,
    RecipeML.mash_step_4_start_temp > 78, RecipeML.mash_step_5_start_temp > 78, RecipeML.mash_step_6_start_temp > 78,
    RecipeML.mash_step_1_infuse_amt <= 0
  )).all()
  for recipe in bad_recipes:
    session.delete(recipe)
  session.commit()

  # Condense the mash steps for recipes with invalid infusion volumes, 
  # delete them if none of their steps have enough water for the grains 
  # (check against the typical grain absorption volume)
  # TODO
  bad_recipes = session.query(RecipeML).filter(
    and_(
      RecipeML.num_mash_steps > 1, 
      or_(
        RecipeML.mash_step_2_type == "infusion", RecipeML.mash_step_3_type == "infusion",
        RecipeML.mash_step_4_type == "infusion", RecipeML.mash_step_5_type == "infusion", RecipeML.mash_step_6_type == "infusion",
      ),
      or_(
        RecipeML.mash_step_1_infuse_amt < 0.5*RecipeML.preboil_vol
      )
    )
  ).all()

  for recipe in bad_recipes:
    if not clean_up_recipe_mash_steps(recipe):
      session.delete(recipe)
    session.commit()

  #RecipeML.mash_step_2_infuse_amt == None,  RecipeML.mash_step_3_infuse_amt == None, 
  #RecipeML.mash_step_4_infuse_amt == None, RecipeML.mash_step_5_infuse_amt == None,
  #RecipeML.mash_step_6_infuse_amt == None,
  '''
  # Find recipes that have a bad temperature step to start and move up all other steps
  bad_recipes = session.query(RecipeML).filter(
    and_(RecipeML.mash_step_1_type == "temperature", or_(RecipeML.mash_step_1_time == 0, RecipeML.mash_step_1_time == 1))
  ).all()
  postfixes = ["_type", "_time", "_start_temp", "_end_temp", "_infuse_amt"]
  for recipe in bad_recipes:
    if recipe.num_mash_steps == 1:
      # This is just a completely invalid recipe, delete it
      session.delete(recipe)
      session.commit()
      continue
    
    # Remove the first mash step, move up all subsequent steps
    for i in range(1, recipe.num_mash_steps+1):
      prefix_prev = "mash_step_"+str(i)
      if i == 6:
        for attr in postfixes:
          setattr(recipe, prefix_prev+attr, None)
      else:
        prefix_curr = "mash_step_"+str(i+1)
        for attr in postfixes:
          setattr(recipe, prefix_prev+attr, getattr(recipe, prefix_curr+attr))
    recipe.num_mash_steps -= 1
    recipe.hash = recipe.gen_hash()
    session.commit()

  '''

  '''
  # Calculate the infusion water amount for the decoction
  total_grains_kg = 0
  for grain in recipe.grains:
    total_grains_kg += float(grain.amount)
  #total_absorb = total_grains_kg * GRAIN_ABSORB_L_PER_KG
  total_infusion = total_grains_kg * np.random.normal((MAX_L_PER_KG_DECOC-MIN_L_PER_KG_DECOC)/2.0 + MIN_L_PER_KG_DECOC, (MAX_L_PER_KG_DECOC-MIN_L_PER_KG_DECOC)/4.0)
  recipe.mash_step_1_infuse_amt = total_infusion
  '''


def clean_up_bad_mash_ph_recipes(session):
  recipes_to_remove = session.scalars(select(RecipeML).filter(or_(RecipeML.mash_ph < 5.0, RecipeML.mash_ph > 5.8))).all()
  for recipe in recipes_to_remove:
    session.delete(recipe)


def clean_up_no_sparge_temp(session):
  style_dists = distributions_by_style_id(session)
  recipes_to_update = session.scalars(select(RecipeML).filter(RecipeML.sparge_temp == None)).all()
  for recipe in recipes_to_update:
    sparge_temp = style_dists[recipe.style_id].sample_sparge_temp()
    if sparge_temp == None:
      sparge_temp = 75.5555556
    recipe.sparge_temp = sparge_temp

def clean_up_misc(session):
  bad_miscs = session.scalars(select(Misc).filter(Misc.type == "fining")).all()
  for misc in bad_miscs:
    session.delete(misc)

def clean_up_bad_volumes(session):
  bad_recipes = session.scalars(select(RecipeML).filter(RecipeML.preboil_vol < RecipeML.postboil_vol)).all()
  for recipe in bad_recipes:
    session.delete(recipe)
  session.commit()

def remove_zero_amounts(session):
  # Grains...
  grainATs = session.scalars(select(RecipeMLGrainAT).filter(RecipeMLGrainAT.amount <= 0)).all()
  for grainAT in grainATs:
    session.delete(grainAT)
  session.commit()
  # Make sure there are no recipes with zero grains now
  recipes = session.scalars(select(RecipeML).join(RecipeML.grains).group_by(RecipeML.id).having(func.count() == 0)).all()
  for recipe in recipes:
    session.delete(recipe)
  session.commit()
  # Adjuncts...
  adjunctATs = session.scalars(select(RecipeMLAdjunctAT).filter(RecipeMLAdjunctAT.amount <= 0)).all()
  for adjunctAT in adjunctATs:
    session.delete(adjunctAT)
  session.commit()
  # Hops...
  hopATs = session.scalars(select(RecipeMLHopAT).filter(RecipeMLHopAT.amount <= 0)).all()
  for hopAT in hopATs:
    session.delete(hopAT)
  session.commit()
  # Miscs...
  miscATs = session.scalars(select(RecipeMLMiscAT).filter(RecipeMLMiscAT.amount <= 0)).all()
  for miscAT in miscATs:
    session.delete(miscAT)
  session.commit()

def remove_duplicate_malts(session):
  from collections import defaultdict
  recipes = session.scalars(select(RecipeML).join(RecipeML.grains).group_by(RecipeML.id).having(func.count() > 1)).all()
  for i, recipe in enumerate(recipes):
    grain_map = defaultdict(lambda: 0)
    for grainAT in recipe.grains:
      grain_map[grainAT.grain_id] += grainAT.amount
      
    # Check for duplicates...
    if len(grain_map) < len(recipe.grains):
      updated_ids = set()
      num_deleted = 0
      for grainAT in recipe.grains:
        if grainAT.grain_id in updated_ids:
          session.delete(grainAT)
          num_deleted += 1
        else:
          grainAT.amount = grain_map[grainAT.grain_id]
          updated_ids.add(grainAT.grain_id)
          if grainAT.amount <= 0:
            session.delete(grainAT)
            num_deleted += 1
            
      # No grains left?... (totally bogus recipe with 0 for all the amounts)
      if num_deleted == len(recipe.grains):
        session.delete(recipe)
        
      session.commit()


def misc_adjunct_to_adjunct(session):
  from temp_list import misc_to_adj_lookup
  '''
  # Try to find all of the adjuncts in the miscs table in the adjuncts table
  misc_adjs = session.scalars(select(Misc).filter(Misc.type == 'adjunct')).all()
  for misc_adj in misc_adjs:
    if misc_adj.name in misc_to_adj_lookup: continue
    found_adjs = session.scalars(select(Adjunct).filter(Adjunct.name.ilike(f"%{misc_adj.name}%"))).all()
    if len(found_adjs) == 0:
      print(f"No match found for misc:         {misc_adj.name}")
    elif len(found_adjs) > 1:
      print(f"Too many matches found for misc: {misc_adj.name}")
    else:
      print(f"Misc: '{misc_adj.name}' -> adjunct: {found_adjs[0].name}")
  '''
  
  # Convert misc adjuncts into a proper adjunct for all recipes...
  recipes = session.scalars(
    select(RecipeML)
    .join(RecipeMLMiscAT, RecipeMLMiscAT.recipe_ml_id == RecipeML.id)
    .join(Misc, Misc.id == RecipeMLMiscAT.misc_id)
    .filter(Misc.type == 'adjunct')
  ).all()
  
  for recipe in recipes:
    miscATs_to_convert = [miscAT for miscAT in recipe.miscs if miscAT.misc.type == 'adjunct']
    for miscAT in miscATs_to_convert:
      misc = miscAT.misc
      if misc.name in misc_to_adj_lookup:
        adjunct_id = misc_to_adj_lookup[misc.name]
        adjunct = session.scalar(select(Adjunct).filter_by(id=adjunct_id))
      else:
        adjunct = session.scalars(select(Adjunct).filter(Adjunct.name.ilike(f"%{misc.name}%"))).first()
      
      assert adjunct != None
      # Convert the misc to an adjunct and add it to the recipe
      adjunctAT = RecipeMLAdjunctAT(
        adjunct=adjunct,
        amount=miscAT.amount, 
        yield_override=None,
        stage=miscAT.stage,
        time=miscAT.time,
        amount_is_weight=miscAT.amount_is_weight
      )
      recipe.adjuncts.append(adjunctAT)
      
      # Remove the misc assoc table from the database and the recipe
      session.delete(miscAT)
    session.commit()
      

def aroma_to_whirlpool_hops(session):
  hopATs = session.scalars(select(RecipeMLHopAT).filter_by(stage="aroma")).all()
  for hopAT in hopATs:
    hopAT.stage = "whirlpool"
    session.flush()


if __name__ == "__main__":
  engine = create_engine(BREWBRAIN_DB_ENGINE_STR, echo=False, future=True)
  Base.metadata.create_all(engine)

  with Session(engine) as session:
    #clean_up_mash_steps(session)
    #remove_zero_mash_or_ferment_step_recipes(session)
    #clean_up_misc(session)

    #ids_to_remove = [6948, 7056, 22066, 27256, 39472]
    #recipes_to_remove = session.scalars(select(RecipeML).filter(or_(*[RecipeML.id == id for id in ids_to_remove]))).all()
    #for recipe in recipes_to_remove:
    #  session.delete(recipe)
    
    #ids_to_remove = []
    #misc_ats_to_remove = session.scalars(select(RecipeMLMiscAT).filter(or_(*[RecipeMLMiscAT.id == id for id in ids_to_remove]))).all()
    #for misc_at in misc_ats_to_remove:
    #  session.delete(misc_at)
    
    #clean_up_bad_mash_ph_recipes(session)
    #clean_up_no_sparge_temp(session)

    #clean_up_bad_volumes(session)
    #remove_duplicate_malts(session)
    #remove_zero_amounts(session)

    #misc_adjunct_to_adjunct(session)
    aroma_to_whirlpool_hops(session)
    session.commit()