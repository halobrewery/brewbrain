
import os
import glob
import sys
import re

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from pybeerxml.parser import Parser

from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy import select
from sqlalchemy import or_, and_

from brewbrain_db import BREWBRAIN_DB_ENGINE_STR, Base, Style, Hop, Grain, Adjunct, Misc, Microorganism, RecipeML
from brewbrain_db import RecipeMLHopAT, RecipeMLGrainAT, RecipeMLAdjunctAT, RecipeMLMiscAT, RecipeMLMicroorganismAT

from hop_names import build_hop_name_dicts, match_hop_id
from fermentable_names import build_fermentable_name_dicts, match_fermentable_id
from yeast_names import build_yeast_dicts, match_yeast_id
from style_names import build_style_dicts, match_style_id
from misc_names import build_misc_name_dicts, match_misc_id

hop_name_to_id, id_to_hop_names = build_hop_name_dicts()
fermentable_name_to_id, id_to_fermentable_names = build_fermentable_name_dicts()
yeast_name_to_id, _, id_to_yeast_names = build_yeast_dicts()
style_name_to_id, id_to_style_names = build_style_dicts()
misc_name_to_id, id_to_misc_names = build_misc_name_dicts()

# NOTE: BeerXML units are - Weight: Kg, Volume: L, Temperature: C, 
# Time (mins default, unless specified by tag), Pressure: KPa
def read_recipe(session, recipe, filepath):
  if recipe.type.lower() != 'all grain': return
  
  recipe_name   = recipe.name.strip() if isinstance(recipe.name, str) else str(recipe.name)
  boil_time     = recipe.boil_time
  efficiency    = recipe.efficiency/100
  fermenter_vol = recipe.batch_size
  
  equipment = recipe.equipment
  if recipe.boil_size != None and recipe.boil_size != 0:
    preboil_vol = recipe.boil_size
  else:
    if equipment == None:
      raise ValueError(f"No equipment and no boil size, in recipe: {recipe_name}")

    # Preboil volume needs to be calculated from the equipment, 
    # BOIL_SIZE = (BATCH_SIZE – TOP_UP_WATER – TRUB_CHILLER_LOSS) * (1+BOIL_TIME * EVAP_RATE ) 
    preboil_vol = (equipment.batch_size - equipment.top_up_water - equipment.trub_chiller_loss) * (1 + equipment.boil_time/60 * equipment.evap_rate/100)

  if equipment != None and equipment.evap_rate > preboil_vol:
    raise ValueError(f"Invalid evaporation rate for recipe: {recipe_name}")

  if equipment != None and equipment.evap_rate:
    postboil_vol = preboil_vol - (preboil_vol * equipment.evap_rate/100 * boil_time/60)
  else:
    postboil_vol = fermenter_vol
  
  # Fermentation Stages, Bottling, Carb
  aging_time  = recipe.age
  aging_temp  = recipe.age_temp
  carbonation = recipe.carbonation
  num_ferment_stages = 0
  ferment_stage_1_time = ferment_stage_1_temp = None
  ferment_stage_2_time = ferment_stage_2_temp = None
  ferment_stage_3_time = ferment_stage_3_temp = None

  if recipe.fermentation_stages != None:
    num_ferment_stages = int(recipe.fermentation_stages)
    if num_ferment_stages == 0:
      raise ValueError(f"No fermentation stages found in recipe: {recipe_name}")
    if num_ferment_stages > 0:
      ferment_stage_1_time = recipe.primary_age
      ferment_stage_1_temp = recipe.primary_temp
      if ferment_stage_1_time == None or ferment_stage_1_temp == None:
        raise ValueError(f"'None' value(s) found for ferment stage 1 in recipe: {recipe_name}")
      if num_ferment_stages > 1:
        ferment_stage_2_time = recipe.secondary_age
        ferment_stage_2_temp = recipe.secondary_temp
        if ferment_stage_2_time == None or ferment_stage_2_temp == None: num_ferment_stages = 1
        if num_ferment_stages > 2:
          ferment_stage_3_time = recipe.tertiary_age
          ferment_stage_3_temp = recipe.tertiary_temp
          if ferment_stage_3_time == None or ferment_stage_3_temp == None: num_ferment_stages = 2

  # Style
  style = recipe.style
  if hasattr(style, 'style_guide'):
    style_guide = re.sub(r"\s*\d+", "", style.style_guide).lower().strip()
    style_guide_year = re.sub(r"\D+\s*", "", style.style_guide).strip()
    style_guide_year = int(style_guide_year) if len(style_guide_year) > 0 else 2021
  else:
    # This is actually malformed beerxml, but we'll just assume an up-to-date style guide anyway
    style_guide = "bjcp"
    style_guide_year = 2021

  style_letter = None
  if hasattr(style, 'style_letter'):
    style_letter = str(style.style_letter.upper() if isinstance(style.style_letter, str) else int(style.style_letter))
  category_number = None
  if hasattr(style, 'category_number'):
    category_number = str(int(style.category_number) if isinstance(style.category_number, float) else style.category_number)

  # First, try to find the style guide and number/letter of the style
  existing_style = None
  if style_letter != None and category_number != None:
    existing_styles = session.scalars(select(Style).filter_by(guide=style_guide, letter=style_letter, number=category_number)).all()
    if len(existing_styles) == 1:
      existing_style = existing_styles[0]
    else:
      for potential_style in existing_styles:
        if style.name.lower() == potential_style.name.lower():
          existing_style = potential_style
          break

  if existing_style == None:
    if not hasattr(style, 'name') or style.name == None:
      raise ValueError(f"Style has no name for recipe: {recipe_name}")

    existing_style = session.scalars(select(Style).filter(
      and_(Style.name == style.name, Style.guide.ilike(f"{style_guide}%")),
    )).first()
    if existing_style == None:
      existing_style = session.scalars(select(Style).filter(Style.name.ilike(f"%{style.name}%"))).first()
      if existing_style == None:
        # Last ditch effort: Try to match misspellings and variations on the style name
        style_name = style.name.lower().strip()
        style_names = [style_name]
        style_id = match_style_id(style_name, style_name_to_id)
        if style_id != None:
          style_names += id_to_style_names[style_id]
        for style_name in style_names:
          existing_style = session.scalars(select(Style).filter(Style.name.ilike(f"%{style_name}%"))).first()
          if existing_style != None: break

        if existing_style == None:
          # ... couldn't find the style
          raise ValueError(f"Failed to find style '{style.name}' for recipe: {recipe_name}")
      
  style = existing_style

  # Mash/Sparge
  mash = recipe.mash
  if mash == None: raise ValueError(f"No mash found, in recipe: {recipe_name}")

  mash_ph = mash.ph
  if mash_ph == None: raise ValueError(f"No mash pH found, in recipe: {recipe_name}")

  sparge_temp = mash.sparge_temp
  if sparge_temp < 35.0:
    sparge_temp = None

  MAX_MASH_STEPS = 6
  num_mash_steps = len(mash.steps)
  if num_mash_steps > MAX_MASH_STEPS:
    raise ValueError(f"Too many mash steps ({num_mash_steps}), in recipe: {recipe_name}")
  elif num_mash_steps == 0:
    raise ValueError(f"No mash steps found for recipe: {recipe_name}")

  first_step_type = mash.steps[0].type.lower()
  if first_step_type != "infusion":
    raise ValueError(f"Invalid starting mash type '{first_step_type}' found, in recipe: {recipe_name}")

  mash_step_vals = {}
  for i in range(MAX_MASH_STEPS):
    step = i+1
    prefix = 'mash_step_' + str(step)
    mash_step_vals[prefix+'_type'] = mash_step_vals[prefix+'_time'] = mash_step_vals[prefix+'_start_temp'] = mash_step_vals[prefix+'_end_temp'] = mash_step_vals[prefix+'_infuse_amt'] = None
  for i, mash_step in enumerate(mash.steps):
    step = i+1
    prefix = 'mash_step_' + str(step)
    m_step_type = mash_step.type.lower()
    mash_step_vals[prefix+'_type'] = m_step_type # Type of the step {"Infusion", "Temperature", "Decoction"}
    mash_step_vals[prefix+'_time'] = mash_step.step_time
    mash_step_vals[prefix+'_start_temp'] = mash_step.step_temp
    mash_step_vals[prefix+'_end_temp'] = mash_step.end_temp
    if m_step_type == "infusion":
      mash_step_vals[prefix+'_infuse_amt'] = mash_step.infuse_amount # Volume of water for infusion step (L)

  # Hops
  hops_ats = []
  for hop in recipe.hops:
    # Try to find the hop in the database - first by name and origin, then just by name
    # This is tricky because of misspellings and variations on the name - 
    # do a lookup into a long list of aliases and perform a select 'like' query on all those aliases
    if not hasattr(hop, 'name') or hop.name == None:
      raise ValueError(f"No hop name found in recipe: {recipe_name}")

    hop_name = str(hop.name).lower()
    found_hop_id = match_hop_id(hop_name, hop_name_to_id)
    hop_names = [hop_name]
    if found_hop_id != None:
      hop_names += id_to_hop_names[found_hop_id]

    for name in hop_names:
      existing_hop = session.scalars(select(Hop).filter((Hop.name.ilike(f"{name}")) & (Hop.origin.ilike(f"{hop.origin}%")))).first()
      if existing_hop == None:
        existing_hop = session.scalars(select(Hop).filter(Hop.name.ilike(f"{name}"))).first()
        if existing_hop == None:
          existing_hop = session.scalars(select(Hop).filter(Hop.name.ilike(f"%{name}%"))).first()
      if existing_hop != None: break

    if existing_hop == None:
      raise ValueError(f"Failed to find hop '{hop.name}', in recipe: {recipe_name}")
    
    assert existing_hop != None
    hops_ats.append(
      RecipeMLHopAT(
        hop=existing_hop,
        amount=hop.amount,
        stage=hop.use.lower(),
        time=hop.time,
        form=hop.form,
        alpha=hop.alpha,
      )
    )

  # Fermentables (Grains and Adjuncts)
  grains_ats = []
  adjuncts_ats = []
  for fermentable in recipe.fermentables:
    if not hasattr(fermentable, 'name') or fermentable.name == None:
      raise ValueError(f"No fermentable name found in recipe: {recipe_name}")

    f_name  = str(fermentable.name).lower()
    if hasattr(fermentable, 'type') and fermentable.type != None:
      f_type  = fermentable.type.lower()
    else:
      # This is frustrating... malformed beerxml and we're going to have to figure out if it's a grain or not
      # Try a look-up by name and if that fails just ignore this recipe
      existing_something = session.scalars(select(Adjunct).filter(Adjunct.name.ilike(f"%{f_name}%"))).first()
      if existing_something == None:
        existing_something = session.scalars(select(Grain).filter(Grain.name.ilike(f"%{f_name}%"))).first()
        if existing_something == None:
          raise ValueError(f"No fermentable type provided for {f_name}, couldn't find match in database, in recipe {recipe_name}")
        else:
          f_type = "grain"
      else:
        f_type = "adjunct"

    f_amt   = fermentable.amount
    f_yield = fermentable._yield/100

    f_id = match_fermentable_id(f_name, fermentable_name_to_id)
    f_names = [f_name]
    if f_id != None:
      f_names += id_to_fermentable_names[f_id]

    def grain_addition_check_add():
      # Try to find the grain in our database...
      existing_grain = None
      for name in f_names:
        existing_grain = session.scalars(select(Grain).filter(
          (Grain.name.ilike(f"{name}")) & 
          or_(Grain.supplier.ilike(f"{fermentable.supplier}%"), Grain.origin.ilike(f"{fermentable.origin}%"))
        )).first()
        if existing_grain == None:
          existing_grain = session.scalars(select(Grain).filter(Grain.name.ilike(f"{name}"))).first()
          if existing_grain == None:
            existing_grain = session.scalars(select(Grain).filter(Grain.name.ilike(f"%{name}%"))).first()
        if existing_grain != None: break

      if existing_grain == None:
        return False
      else:
        moisture_override = fermentable.moisture/100
        coarse_fine_diff_override = fermentable.coarse_fine_diff/100
        protein_override = fermentable.protein/100
        grains_ats.append(
          RecipeMLGrainAT(
            grain=existing_grain,
            amount=f_amt,
            fgdb_override=f_yield,
            moisture_override=moisture_override,
            coarse_fine_diff_override=coarse_fine_diff_override,
            protein_override=protein_override,
          )
        )
      return True

    def adjunct_addition_check_add():
       # Try to find the adjunct in our database...
      existing_adjunct = None
      for name in f_names:
        existing_adjunct = session.scalars(select(Adjunct).filter(Adjunct.name.ilike(f"{name}"))).first()
        if existing_adjunct == None:
          existing_adjunct = session.scalars(select(Adjunct).filter(Adjunct.name.ilike(f"%{name}%"))).first()
        if existing_adjunct != None: break
      if existing_adjunct == None:
        return False
      else:
        adjuncts_ats.append(RecipeMLAdjunctAT(adjunct=existing_adjunct, amount=f_amt, yield_override=f_yield))
      return True

    if f_type == 'grain':
      if not grain_addition_check_add():
        # Try to find the addition as an adjunct instead
        f_type = 'adjunct'
        if not adjunct_addition_check_add():
          raise ValueError(f"Failed to find grain/adjunct '{f_name}', in recipe: {recipe_name}")
    else:
      if not adjunct_addition_check_add():
        f_type = 'grain'
        if not grain_addition_check_add():
          raise ValueError(f"Failed to find adjunct/grain '{f_name}', in recipe: {recipe_name}")

  # Misc
  miscs_ats = []
  for misc in recipe.miscs:
    m_name = misc.name.lower().strip()
    
    if misc.type != None:
      if misc.type.lower() == 'fining': continue

    existing_misc = session.scalars(select(Misc).filter(Misc.name.ilike(f"%{m_name}%"))).first()
    if existing_misc == None:
      m_id = match_misc_id(m_name, misc_name_to_id)
      if m_id == 'ignore':
        continue
      elif m_id == None:
        raise ValueError(f"Failed to find misc ingredient: {m_name}, in recipe {recipe_name}")
      else:
        m_names = id_to_misc_names[m_id]
        for name in m_names:
          existing_misc = session.scalars(select(Misc).filter(Misc.name.ilike(f"{name}"))).first()
          if existing_misc == None:
            existing_misc = session.scalars(select(Misc).filter(Misc.name.ilike(f"%{name}%"))).first()
          if existing_misc != None: break

    if existing_misc == None:
      raise ValueError(f"Failed to find misc '{m_name}', in recipe: {recipe_name}")

    amount = misc.amount
    amount_is_weight = misc.amount_is_weight
    if misc.use == None: continue
    stage = misc.use.lower()
    time  = misc.time

    assert existing_misc != None
    miscs_ats.append(
      RecipeMLMiscAT(
        misc=existing_misc, 
        amount=amount,
        amount_is_weight=amount_is_weight,
        stage=stage,
        time=time,
    ))

  # Yeast/Microorganisms
  microorganisms_ats = []
  for microorganism in recipe.yeasts:
    m_name = microorganism.name
    m_lab  = microorganism.laboratory
    m_product_code = str(int(microorganism.product_id) if isinstance(microorganism.product_id, float) else microorganism.product_id)

    m_id = match_yeast_id(m_name, yeast_name_to_id)
    m_names = [m_name]
    if m_id != None:
      m_names += id_to_yeast_names[m_id]
    elif len(m_product_code) > 0:
      m_id = match_yeast_id(m_product_code, yeast_name_to_id)
      if m_id != None:
        m_names += id_to_yeast_names[m_id]

    for m_name in m_names:
      existing_microorganism = session.scalars(select(Microorganism).filter(
        (Microorganism.name.ilike(f"{m_name}")) & 
        or_(Microorganism.lab.ilike(f"{m_lab}%"), Microorganism.product_code.ilike(f"{m_product_code}%"))
      )).first()
      if existing_microorganism == None:
        existing_microorganism = session.scalars(select(Microorganism).filter(Microorganism.name.ilike(f"{m_name}"))).first()
        if existing_microorganism == None:
          existing_microorganism = session.scalars(select(Microorganism).filter(Microorganism.name.ilike(f"%{m_name}%"))).first()
        if existing_microorganism == None:
          existing_microorganism = session.scalars(select(Microorganism).filter(Microorganism.product_code.ilike(f"%{m_product_code}%"))).first()
      if existing_microorganism != None: break

    if existing_microorganism == None:
      raise ValueError(f"Failed to find microorganism '{m_name}', in recipe: {recipe_name}")

    assert existing_microorganism != None
    stage = "secondary" if microorganism.add_to_secondary else "primary"
    microorganisms_ats.append(
      RecipeMLMicroorganismAT(
        microorganism=existing_microorganism,
        stage=stage
      )
    )

  session.autoflush = False
  recipe_ml = RecipeML(
    name=recipe_name,
    preboil_vol=preboil_vol,
    postboil_vol=postboil_vol,
    fermenter_vol=fermenter_vol,
    boil_time=boil_time,
    efficiency=efficiency,
    mash_ph=mash_ph,
    sparge_temp=sparge_temp,
    num_mash_steps=num_mash_steps,
    **mash_step_vals,
    num_ferment_stages=num_ferment_stages,
    ferment_stage_1_time=ferment_stage_1_time,
    ferment_stage_1_temp=ferment_stage_1_temp,
    ferment_stage_2_time=ferment_stage_2_time,
    ferment_stage_2_temp=ferment_stage_2_temp,
    ferment_stage_3_time=ferment_stage_3_time,
    ferment_stage_3_temp=ferment_stage_3_temp,
    aging_time=aging_time,
    aging_temp=aging_temp,
    carbonation=carbonation,
    style=style,
    hops=hops_ats,
    grains=grains_ats,
    adjuncts=adjuncts_ats,
    miscs=miscs_ats,
    microorganisms=microorganisms_ats,
  )
  recipe_ml.hash = recipe_ml.gen_hash()
  
  # Check whether the recipe already exists in the database via its hash
  existing_recipe_ml = session.scalars(select(RecipeML).filter_by(hash=recipe_ml.hash)).first()
  if existing_recipe_ml == None:
    session.add(recipe_ml)
    # Need to call commit if any deletes happened to mark all associated cascades as deleted too!
    session.commit()
    print(f"Recipe converted and committed to database: '{recipe_name}', in file {filepath}")
  else:
    session.rollback()
    print(f"Recipe already exists in database '{recipe_name}', in file {filepath}, skipping.")
  session.autoflush = True
  

if __name__ == "__main__":
  engine = create_engine(BREWBRAIN_DB_ENGINE_STR, echo=False, future=True)
  Base.metadata.create_all(engine)

  def session_read_beerxml_files():
    beerxml_parser = Parser()
    datapath = "/Users/callumhay/projects/brewbrain/data"

    with Session(engine) as session:
      #for dirpath, dirnames, files in os.walk(datapath):
      for filepath in glob.glob(os.path.join(datapath, "*.xml")):
        try:
          recipes = beerxml_parser.parse(filepath)
        except Exception as e:
          print(f"Erroneous file found ({filepath}), exception: {e} exiting.")
          session.rollback()
          return

        for recipe in recipes:
          try:
            read_recipe(session, recipe, filepath)
          except ValueError as e:
            print(f"Failed to read recipe: {e}, rolling back and continuing.")
            session.rollback()
            continue
        #print(f"Finished reading recipes from file {filepath}")
      session.commit()

  session_read_beerxml_files()
