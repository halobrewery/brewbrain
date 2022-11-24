import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy import select
from sqlalchemy import or_, and_

from brewbrain_db import BREWBRAIN_DB_ENGINE_STR, Base, RecipeML, Misc


def remove_zero_mash_or_ferment_step_recipes(session):
  bad_recipes = session.query(RecipeML).filter(or_(RecipeML.num_mash_steps == 0, RecipeML.num_ferment_stages == 0)).all()
  for recipe in bad_recipes:
    session.delete(recipe)

def clean_up_mash_steps(session):
  MAX_L_PER_KG_DECOC = 3.12953
  MIN_L_PER_KG_DECOC = 2.607939
  GRAIN_ABSORB_L_PER_KG = 1.00144835

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
    # If this recipe is reasonable then the total infusion volume should add up to AT LEAST the grain absorption
    total_infusion_vol = recipe.total_infusion_vol()
    total_grain_mass   = recipe.total_grain_mass()
    grain_absorb_vol   = GRAIN_ABSORB_L_PER_KG * total_grain_mass
    if total_infusion_vol <= grain_absorb_vol:
      # This recipe can't possibly have enough water in its mash, delete it
      session.delete(recipe)
      continue

    # If the first infuse step is enough to actually soak the grains then we're ok
    if recipe.mash_step_1_infuse_amt > grain_absorb_vol:
      continue
    
    # There appears enough water to make the mash happen, we just need to find it 
    # in the other infusion steps and move it into the first infusion step... 
    assert recipe.num_mash_steps > 1 and recipe.mash_step_1_type == "infusion"

    postfixes = ["_type", "_time", "_start_temp", "_end_temp", "_infuse_amt"]    
    def remove_mash_step(step_idx):
      assert step_idx >= 1 and step_idx <= recipe.num_mash_steps
      for idx in range(step_idx, recipe.num_mash_steps+1):
        prefix_prev = "mash_step_"+str(idx)
        if idx == 6:
          for attr in postfixes:
            setattr(recipe, prefix_prev+attr, None)
        else:
          prefix_curr = "mash_step_"+str(idx+1)
          for attr in postfixes:
            setattr(recipe, prefix_prev+attr, getattr(recipe, prefix_curr+attr))
      recipe.num_mash_steps -= 1

    curr_infuse_amt  = recipe.mash_step_1_infuse_amt
    curr_infuse_start_temp = recipe.mash_step_1_start_temp
    curr_infuse_end_temp = recipe.mash_step_1_end_temp
    curr_infuse_time = recipe.mash_step_1_time
    i = 2
    first_infusion_fixed = False
    steps_inbetween = False
    while i <= recipe.num_mash_steps:
      prefix = "mash_step_"+str(i)
      type = getattr(recipe, prefix+"_type")
      if type == "infusion":
        # If the infusion amount is zero then we need to get rid of this step
        infuse_amt = getattr(recipe, prefix+"_infuse_amt")
        if not first_infusion_fixed:
          # Merge this mash step with the first mash step (infusion) and remove this step
          curr_infuse_amt += infuse_amt if infuse_amt != None else 0
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
      
    recipe.hash = recipe.gen_hash()
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

def clean_up_misc(session):
  bad_miscs = session.scalars(select(Misc).filter(Misc.type == "fining")).all()
  for misc in bad_miscs:
    session.delete(misc)

if __name__ == "__main__":
  engine = create_engine(BREWBRAIN_DB_ENGINE_STR, echo=True, future=True)
  Base.metadata.create_all(engine)

  with Session(engine) as session:
    #clean_up_mash_steps(session)
    #remove_zero_mash_or_ferment_step_recipes(session)
    #clean_up_misc(session)
    ids_to_remove = [9914]
    recipes_to_remove = session.scalars(select(RecipeML).filter(or_(*[RecipeML.id == id for id in ids_to_remove]))).all()
    for recipe in recipes_to_remove:
      session.delete(recipe)
    
    
    session.commit()