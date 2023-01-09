import os
import datetime
from collections import OrderedDict

import streamlit as st
import pandas as pd
import numpy as np

import torch
from torch.utils.data import DataLoader

from sqlalchemy import create_engine, select, func
from sqlalchemy.orm import Session

from recipe_dataset import RecipeDataset, DatasetMappings
from recipe_converter import RecipeConverter
from recipe_net_args import RecipeNetArgs
from brewbrain import init_rng_seeding, load_dataset
from brewbrain_db import BREWBRAIN_DB_ENGINE_STR, Base, CoreStyle, Style, RecipeML
from model import RecipeNet, MODEL_FILE_KEY_NETWORK, MODEL_FILE_KEY_ARGS

# Basic initialization / global variables for app ****
st.set_page_config(layout="wide")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
init_rng_seeding(42)
db_engine = create_engine(BREWBRAIN_DB_ENGINE_STR, echo=False, future=True)
Base.metadata.create_all(db_engine)
dataset_mappings = DatasetMappings.load()

# Session state keys ****
SESSION_KEY_MODEL_FILEPATH = "model_filepath"
SESSION_KEY_MODEL          = "model"
SESSION_KEY_CORE_STYLE     = "filter_core_style"
SESSION_KEY_NUM_ROWS       = "filter_num_rows"
SESSION_KEY_RECIPE         = "recipe"
SESSION_KEY_RECIPE_OPTS    = "recipe_options"
SESSION_KEY_DATALOADER     = "dataloader"

# Initialize the session state ****
if SESSION_KEY_MODEL not in st.session_state:
  st.session_state[SESSION_KEY_MODEL] = None

# Mapping/List functions for options widgets ****
def get_core_styles(db_engine):
  with Session(db_engine) as session:
    return session.query(CoreStyle).with_entities(CoreStyle.name, CoreStyle.id).all()

MODELS_ROOT_FILEPATH = "./runs"
def get_model_files():
  model_files_list = [("No Model Selected", {'filepath': "", 'datetime': datetime.datetime.now()})]
  for dirpath, _, files in os.walk(MODELS_ROOT_FILEPATH):
    for filename in files:
      if filename.endswith(".chkpt"):
        utc_datetime = datetime.datetime.utcfromtimestamp(int(dirpath.split("_")[-1]))
        time_str = utc_datetime.strftime("%b %d %Y, %I:%M%p")
        #run_step = filename.split(".")[-2].split("_")[-1]
        map_key = time_str + f" [{filename}]"
        model_files_list.append(
          (map_key, {'filepath': os.path.join(dirpath, filename), 'datetime': utc_datetime})
        )

  sorted(model_files_list, key=lambda x: x[1]['datetime'])
  return [(i[0], i[1]['filepath']) for i in model_files_list]

def display_recipe(recipe, converter):

  # High-level recipe features
  st.markdown(f'''
    **Boil Time:** {recipe['boil_time']}<br>
    **Mash pH:**   {recipe['mash_ph']}<br>
    **Sparge Temp:** {recipe['sparge_temp']}<br>
  ''', unsafe_allow_html=True)

  # Mash Steps
  mash_step_inds = recipe['mash_step_type_inds'] != 0
  mash_step_type_names = converter.mash_step_type_names(recipe['mash_step_type_inds'][mash_step_inds])
  mash_step_table = [
    [st for st in mash_step_type_names],
    [str(int(t)) for t in recipe['mash_step_times'][mash_step_inds]],
    [str(t) for t in recipe['mash_step_avg_temps'][mash_step_inds]]
  ]
  st.table(
    pd.DataFrame(
      mash_step_table, 
      columns=[f"Step {i+1}" for i in range(len(mash_step_type_names))], 
      index=["Type", "Time (min)", "Temp (C)"]
    )
  )

  # Grains / Malt Bill
  grain_type_inds = recipe['grain_core_type_inds'] != 0
  grain_names = converter.grain_type_names(recipe['grain_core_type_inds'][grain_type_inds])
  if len(grain_names) == 0:
    # This shouldn't happen
    st.markdown(":red[No Grains Found!]")
    st.error("No grains found in recipe!")
  else:
    grain_table = [(name, str(np.round(recipe['grain_amts'][i]*100,1))+"%") for i, name in enumerate(grain_names)]
    st.table(pd.DataFrame(grain_table, columns=["Grain", "Percentage"], index=[i for i in range(1, len(grain_table)+1)]))

  # Adjuncts
  adjunct_type_inds = recipe['adjunct_core_type_inds'] != 0
  adjunct_names = converter.adjunct_type_names(recipe['adjunct_core_type_inds'][adjunct_type_inds])
  if len(adjunct_names) > 0:
    adjunct_table = [(name, str(recipe['adjunct_amts'][i])) for i, name in enumerate(adjunct_names)]
    st.table(pd.DataFrame(
      adjunct_table, 
      columns=["Adjunct", "Concentration (ml or g / L)"], 
      index=[i for i in range(1, len(adjunct_table)+1)]
    ))

  # Hops
  hop_type_inds = recipe['hop_type_inds'] != 0
  hop_names = converter.hop_type_names(recipe['hop_type_inds'][hop_type_inds])
  if len(hop_names) > 0:
    hop_stage_type_names = converter.hop_stage_type_names(recipe['hop_stage_type_inds'][hop_type_inds])
    hop_table = [(name, hop_stage_type_names[i], str(int(recipe['hop_times'][i])), recipe['hop_concentrations'][i]) for i,name in enumerate(hop_names)]
    st.table(pd.DataFrame(
      hop_table, 
      columns=["Name", "Stage", "Time (min)", "Concentration (AA g/L (boil) or g/L)"], 
      index=[i for i in range(1, len(hop_table)+1)]
    ))

  # Miscs
  misc_type_inds = recipe['misc_type_inds'] != 0
  misc_names = converter.misc_type_names(recipe['misc_type_inds'][misc_type_inds])
  if len(misc_names) > 0:
    misc_stage_type_names = converter.misc_stage_type_names(recipe['misc_stage_inds'][misc_type_inds])
    misc_table = [(name, misc_stage_type_names[i], str((int(recipe['misc_times'][i]))), recipe['misc_amts'][i]) for i, name in enumerate(misc_names)]
    st.table(pd.DataFrame(
      misc_table,
      columns=["Name", "Stage", "Time (min)", "Concentration (g or ml / L)"],
      index=[i for i in range(1, len(misc_table)+1)]
    ))

  # Microorganisms
  mo_type_inds = recipe['mo_type_inds'] != 0
  mo_names = converter.microorganism_type_names(recipe['mo_type_inds'][mo_type_inds])
  if len(mo_names) == 0:
    # This shouldn't happen
    st.markdown(":red[No Microorganism Found!]")
    st.error("No microorganism found in recipe!")
  else:
    mo_stage_type_names = converter.microorganism_stage_type_names(recipe['mo_stage_inds'][mo_type_inds])
    # TODO

  # Fermentation
  primary_ferment_str = "N/A"
  secondary_ferment_str = "N/A"
  if recipe['ferment_stage_times'][0] != 0:
    primary_ferment_str = f"{str(int(recipe['ferment_stage_times'][0]))} days @ {str(recipe['ferment_stage_temps'][0])}C"
  if recipe['ferment_stage_times'][1] != 0:
    secondary_ferment_str = f"{str(int(recipe['ferment_stage_times'][1]))} days @ {str(recipe['ferment_stage_temps'][1])}C"

  st.markdown(f'''
    **Primary Fermentation:** {primary_ferment_str}<br>
    **Secondary Fermentation:** {secondary_ferment_str}<br>
  ''', unsafe_allow_html=True)





# Widget Event functions ****
def on_model_change():
  model_filepath = st.session_state[SESSION_KEY_MODEL_FILEPATH][1]

  if st.session_state[SESSION_KEY_MODEL] != None:
    del st.session_state[SESSION_KEY_MODEL]
    st.session_state[SESSION_KEY_MODEL] = None

  # Quick exit if no model was selected
  if model_filepath == "" or model_filepath == None: return
  
  # Load the new model
  load_str = f"Loading model '{model_filepath}' ..."
  print(load_str)
  with st.spinner(text=load_str):
    model_dict = torch.load(model_filepath, map_location=device)
    args = RecipeNetArgs(model_dict[MODEL_FILE_KEY_ARGS])
    model = RecipeNet(args).to(device)
    model.load_state_dict(model_dict[MODEL_FILE_KEY_NETWORK])
    model.eval()
    del model_dict
    torch.cuda.empty_cache()
    st.session_state[SESSION_KEY_MODEL] = model
  print("Model Loading Complete!")

def on_filter_change():
  core_style_id = st.session_state[SESSION_KEY_CORE_STYLE][1]
  num_rows = st.session_state[SESSION_KEY_NUM_ROWS]

  # Update the recipe options
  with Session(db_engine) as session:
    select_stmt = select(RecipeML) \
      .join(Style, Style.id == RecipeML.style_id) \
      .join(CoreStyle, CoreStyle.id == Style.core_style_id) \
      .filter(CoreStyle.id == core_style_id) \
      .limit(num_rows)
    
    st.session_state[SESSION_KEY_RECIPE_OPTS] = [(recipe.name, recipe.id) for recipe in session.scalars(select_stmt).all()]
    st.session_state[SESSION_KEY_RECIPE] = st.session_state[SESSION_KEY_RECIPE_OPTS][0]
    on_recipe_change()

def on_recipe_change():
  recipe_id = st.session_state[SESSION_KEY_RECIPE][1]
  with st.spinner(text=f"Loading recipe '{st.session_state[SESSION_KEY_RECIPE][0]}'"):
    # Build a new dataloader for the selected recipe
    recipe_dataset = RecipeDataset(dataset_mappings)
    recipe_dataset.load_from_db(db_engine, {'select_stmt': select(RecipeML).filter(RecipeML.id == recipe_id)})
    st.session_state[SESSION_KEY_DATALOADER] = DataLoader(recipe_dataset, batch_size=1)
    print("Recipe loaded.")
  

# Sidebar GUI ****
st.sidebar.subheader("Model")
model_files = get_model_files() # A list of tuples (<option_name>, <filepath_str>) for each available model file
st.sidebar.selectbox(
  "Model File", model_files, format_func=lambda x: x[0], 
  on_change=on_model_change, key=SESSION_KEY_MODEL_FILEPATH
)

st.sidebar.subheader("Recipe")
core_styles = get_core_styles(db_engine)
core_style  = st.sidebar.selectbox("Core Style", core_styles, format_func=lambda x: x[0], key=SESSION_KEY_CORE_STYLE, on_change=on_filter_change)
num_rows    = st.sidebar.slider("Rows", 1, 100, 10, key=SESSION_KEY_NUM_ROWS, on_change=on_filter_change)
if SESSION_KEY_RECIPE_OPTS not in st.session_state or st.session_state[SESSION_KEY_RECIPE_OPTS] == None:
  on_filter_change()

recipe = st.sidebar.selectbox(
  "Recipe", st.session_state[SESSION_KEY_RECIPE_OPTS], 
  format_func=lambda x: x[0], key=SESSION_KEY_RECIPE, on_change=on_recipe_change
)
if SESSION_KEY_DATALOADER not in st.session_state or st.session_state[SESSION_KEY_DATALOADER] == None:
  on_recipe_change()


# Main GUI ****
if st.session_state[SESSION_KEY_MODEL] == None:
  st.write("No model loaded.")
elif st.session_state[SESSION_KEY_DATALOADER] == None:
  st.write("No recipe loaded.")
else:
  # Both a model and a recipe are available, 
  # run the recipe through the network and display the central GUI
  model = st.session_state[SESSION_KEY_MODEL]
  dataloader = st.session_state[SESSION_KEY_DATALOADER]
  recipe_batch = next(iter(dataloader))
  batch = {}
  for key, value in recipe_batch.items():
    if key == 'dbid': continue
    batch[key] = value.cuda()
  heads, foots, mean, logvar, z = model(batch, use_mean=True)

  converter = RecipeConverter(dataset_mappings)

  with st.expander("Latent Space Values (z)", expanded=True):
    st.bar_chart(pd.DataFrame(z[0].detach().cpu().numpy(), columns=["z"]))
  
  original_col, decoded_col = st.columns(2)
  with original_col:
    display_recipe(converter.batch_to_recipes(recipe_batch)[0], converter)
  with decoded_col:
    display_recipe(converter.net_output_to_recipes(foots)[0], converter)

