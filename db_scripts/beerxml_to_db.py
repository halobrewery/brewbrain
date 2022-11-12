
import os
import glob
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from pybeerxml.parser import Parser

from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy import select
from sqlalchemy import or_

from brewbrain_db import BREWBRAIN_DB_ENGINE_STR, Base, Style, Hop, Grain, Adjunct, Misc, RecipeML
from brewbrain_db import RecipeMLHopAT, RecipeMLGrainAT, RecipeMLAdjunctAT, RecipeMLMiscAT, RecipeMLMicroorganismAT

from hop_names import build_hop_name_dicts, match_hop_id
from fermentable_names import build_fermentable_name_dicts, match_fermentable_id
from yeast_names import build_yeast_dicts, match_yeast_id

if __name__ == "__main__":
  beerxml_parser = Parser()

  datapath = "/Users/callumhay/projects/brewbrain/data"
  filename = "beersmith_recipes1.xml"
  filepath = os.path.join(datapath, filename)
  
  engine = create_engine(BREWBRAIN_DB_ENGINE_STR, echo=True, future=True)
  Base.metadata.create_all(engine)

  hop_name_to_id, id_to_hop_names = build_hop_name_dicts()
  fermentable_name_to_id, id_to_fermentable_names = build_fermentable_name_dicts()
  yeast_name_to_id, _, id_to_yeast_names = build_yeast_dicts()

  with Session(engine) as session:
    try:
      recipes = beerxml_parser.parse(filepath)
    except:
      print("Erroneous file found: " + filepath)
      exit()

    # NOTE: BeerXML units are - Weight: Kg, Volume: L, Temperature: C, Time (mins default, unless specified by tag), Pressure: KPa
    for recipe in recipes:
      if recipe.type.lower() != 'all grain': continue
      
      name = recipe.name
      boil_time     = recipe.boil_time
      efficiency    = recipe.efficiency/100
      fermenter_vol = recipe.batch_size
      
      # If this fails then the preboil volume needs to be calculated from the equipment, 
      # BOIL_SIZE = (BATCH_SIZE – TOP_UP_WATER – TRUB_CHILLER_LOSS) * (1+BOIL_TIME * EVAP_RATE ) 
      assert recipe.boil_size != None and recipe.boil_size != 0

      equipment = recipe.equipment
      if recipe.boil_size != None and recipe.boil_size != 0:
        preboil_vol = recipe.boil_size
      else:
        preboil_vol = (equipment.batch_size - equipment.top_up_water - equipment.trub_chiller_loss) * (1 + equipment.boil_time/60 * equipment.evap_rate/100)

      if equipment.evap_rate:
        postboil_vol = preboil_vol - (preboil_vol * equipment.evap_rate/100 * boil_time/60)
      else:
        postboil_vol = fermenter_vol
      
      # Fermentation Stages, Bottling, Carb
      aging_time  = recipe.age
      aging_temp  = recipe.age_temp
      carbonation = recipe.carbonation
      num_ferment_stages = int(recipe.fermentation_stages)
      ferment_stage_1_time = ferment_stage_1_temp = None
      ferment_stage_2_time = ferment_stage_2_temp = None
      ferment_stage_3_time = ferment_stage_3_temp = None
      if num_ferment_stages > 0:
        ferment_stage_1_time = recipe.primary_age
        ferment_stage_1_temp = recipe.primary_temp
        if num_ferment_stages > 1:
          ferment_stage_2_time = recipe.secondary_age
          ferment_stage_2_temp = recipe.secondary_temp
          if num_ferment_stages > 2:
            ferment_stage_3_time = recipe.tertiary_age
            ferment_stage_3_temp = recipe.tertiary_temp

      # Style
      style = recipe.style
      # First, try to find the style guide and number/letter of the style
      existing_style = session.scalars(select(Style).filter_by(
        guide=style.style_guide.lower(), 
        letter=style.style_letter.upper(), 
        number=str(int(style.category_number))
      )).first()
      if existing_style == None:
        # Style guide doesn't work, try to find a name or category match
        existing_style = session.scalars(select(Style).filter(
          (Style.name == style.name) | (Style.category == style.category)
        )).first()
        if existing_style == None:
          # ... couldn't find the style
          print(f"Failed to find style '{style.name}' for recipe: {filepath}")
          continue
      style = existing_style

      # Mash/Sparge
      mash = recipe.mash
      mash_ph = mash.ph
      sparge_temp = mash.sparge_temp

      MAX_MASH_STEPS = 3
      num_mash_steps = len(mash.steps)
      if num_mash_steps > MAX_MASH_STEPS:
        print(f"Too many mash steps ({num_mash_steps}), in recipe: {filepath}")
        continue

      mash_step_vals = {}
      for i in range(MAX_MASH_STEPS):
        step = i+1
        prefix = 'mash_step_' + str(step)
        mash_step_vals[prefix+'_type'] = mash_step_vals[prefix+'_time'] = mash_step_vals[prefix+'_start_temp'] = mash_step_vals[prefix+'_end_temp'] = mash_step_vals[prefix+'_infuse_amt'] = None
      for i, mash_step in enumerate(mash.steps):
        step = i+1
        prefix = 'mash_step_' + str(step)
        mash_step_vals[prefix+'_type'] = mash_step.type.lower() # Type of the step {"Infusion", "Temperature", "Decoction"}
        mash_step_vals[prefix+'_time'] = mash_step.step_time
        mash_step_vals[prefix+'_start_temp'] = mash_step.step_temp
        mash_step_vals[prefix+'_end_temp'] = mash_step.end_temp
        mash_step_vals[prefix+'_infuse_amt'] = mash_step.infuse_amount # Volume of water for infusion step (L)

      # Hops
      hops_ats = []
      failed = False
      for hop in recipe.hops:
        # Try to find the hop in the database - first by name and origin, then just by name
        # This is tricky because of misspellings and variations on the name - 
        # do a lookup into a long list of aliases and perform a select 'like' query on all those aliases
        hop_name = hop.name.lower()

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
          print(f"Failed to find hop '{hop.name}', in recipe: {filepath}")
          failed = True
          break
        
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

      if failed: continue

      # Fermentables (Grains and Adjuncts)
      grains_ats = []
      adjuncts_ats = []
      for fermentable in recipe.fermentables:
        f_name  = fermentable.name.lower()
        f_type  = fermentable.type.lower()
        f_amt   = fermentable.amount
        f_yield = fermentable._yield/100

        f_id = match_fermentable_id(f_name, fermentable_name_to_id)
        f_names = [f_name]
        if f_id != None:
          f_names += id_to_fermentable_names[f_id]

        if f_type == 'grain':
          # Try to find the grain in our database...
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
            print(f"Failed to find grain '{f_name}', in recipe: {filepath}")
            failed = True
            break

          assert existing_grain != None
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
        else:
          # Try to find the adjunct in our database...
          for name in f_names:
            existing_adjunct = session.scalars(select(Adjunct).filter(Adjunct.name.ilike(f"{name}"))).first()
            if existing_adjunct == None:
              existing_adjunct = session.scalars(select(Adjunct).filter(Adjunct.name.ilike(f"%{name}%"))).first()
            if existing_adjunct != None: break

          if existing_adjunct == None:
            print(f"Failed to find adjunct '{f_name}, in recipe: {filepath}")
            failed = True
            break

          assert existing_adjunct != None
          adjuncts_ats.append(RecipeMLAdjunctAT(adjunct=existing_adjunct, amount=f_amt, yeild_override=f_yield))

      if failed: continue

      # Misc
      miscs_ats = []
      for misc in recipe.miscs:
        m_name = misc.name.lower()
        #m_type = misc.type.lower()

        existing_misc = session.scalars(select(Misc).filter(Misc.name.ilike(f"{name}%"))).first()
        if existing_misc == None:
          print(f"Failed to find misc '{m_name}, in recipe: {filepath}")
          failed = True
          break

        amount = misc.amount
        amount_is_weight = misc.amount_is_weight
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

      if failed: continue

      # Yeast/Microorganisms
      microorganisms_ats = []
      for microorganism in recipe.yeasts:
        m_name = microorganism.name
        m_lab  = microorganism.laboratory
        m_product_code = microorganism.product_id

        


      if failed: continue

      recipe_ml = RecipeML(
        name=name,
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
      )
      recipe_ml.hash = recipe_ml.gen_hash()

      # Check whether the recipe already exists in the database via its hash
      existing_recipe_ml = session.scalars(select(RecipeML).filter_by(hash=recipe_ml.hash)).first()
      if existing_recipe_ml != None:
        session.delete(recipe_ml)

      session.flush() # Need to call this if any deletes happened to mark all associated cascades as deleted too!


    session.commit()
    



