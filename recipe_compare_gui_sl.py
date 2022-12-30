import os
import datetime

import streamlit as st
import pandas as pd

import torch
from torch.utils.data import DataLoader

from sqlalchemy import create_engine, select, func
from sqlalchemy.orm import Session

from recipe_dataset import RecipeDataset, DatasetMappings
from recipe_net_args import RecipeNetArgs
from brewbrain import init_rng_seeding, load_dataset
from brewbrain_db import BREWBRAIN_DB_ENGINE_STR, Base, CoreStyle, Style, RecipeML
from model import RecipeNet, MODEL_FILE_KEY_NETWORK, MODEL_FILE_KEY_ARGS

#@st.cache(persist=True, allow_output_mutation=True)
#def build_dataloader():
#  return DataLoader(load_dataset(), batch_size=1, shuffle=True, num_workers=0) 

def get_core_styles_map(db_engine):
  with Session(db_engine) as session:
    core_style_tuples = session.query(CoreStyle).with_entities(CoreStyle.name, CoreStyle.id).all()
    return {name: id for name, id in core_style_tuples}

MODELS_ROOT_FILEPATH = "./runs" 
def get_model_files_map():
  model_files_map = {}
  for dirpath, dirnames, files in os.walk(MODELS_ROOT_FILEPATH):
    for filename in files:
      if filename.endswith(".chkpt"):
        utc_datetime = datetime.datetime.utcfromtimestamp(int(dirpath.split("_")[-1]))
        time_str = utc_datetime.strftime("%b %d %Y, %I:%M:%S%p")
        #run_step = filename.split(".")[-2].split("_")[-1]
        map_key = time_str + f" [{filename}]"
        model_files_map[map_key] = os.path.join(dirpath, filename)
  return model_files_map


init_rng_seeding(42)
model_files_map = get_model_files_map()

model  = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

db_engine = create_engine(BREWBRAIN_DB_ENGINE_STR, echo=False, future=True)
Base.metadata.create_all(db_engine)

core_styles_map = get_core_styles_map(db_engine)

# Sidebar GUI ****
st.sidebar.subheader("Model")
model_item = st.sidebar.selectbox("Model File", model_files_map.items(), format_func=lambda x: x[0])
model_filepath = model_item[1]
model_dict = torch.load(model_filepath)
args = RecipeNetArgs(model_dict[MODEL_FILE_KEY_ARGS])
if model != None:
  del model
model = RecipeNet(args).to(device)
model.load_state_dict(model_dict[MODEL_FILE_KEY_NETWORK])
model.eval()

st.sidebar.subheader("Filters")
core_style = st.sidebar.selectbox("Core Style", core_styles_map.keys())
num_rows   = st.sidebar.slider("Rows", 1, 100, 10)
shuffle    = st.sidebar.checkbox("Shuffle Rows", value=True)

st.sidebar.subheader("Recipe Selection")

# Determine the potential recipes from the filters
with Session(db_engine) as session:
  select_stmt = select(RecipeML) \
    .join(Style, Style.id == RecipeML.style_id) \
    .join(CoreStyle, CoreStyle.id == Style.core_style_id) \
    .filter(CoreStyle.id == core_styles_map[core_style]).limit(num_rows)
  if shuffle:
    select_stmt = select_stmt.order_by(func.random())
  
  recipes_map = {}
  for recipe in session.scalars(select_stmt).all():
    recipes_map[recipe.id] = recipe.name
  
  recipe = st.sidebar.selectbox("Recipe", recipes_map.items(), format_func=lambda x: x[1])
  
  dataset_mappings = DatasetMappings.load()
  recipe_dataset = RecipeDataset(dataset_mappings)

  recipe_dataset.load_from_db(db_engine, {'select_stmt': select(RecipeML).filter(RecipeML.id == recipe[0])})
  dataloader = DataLoader(recipe_dataset, batch_size=1)
  #print(recipe_dataset.recipes)

# Main/Central GUI ****
st.title("Recipe Encoder/Decoder/Tweaker")

# Run the selected recipe through the network
recipe_batch = next(iter(dataloader))
batch = {}
for key, value in recipe_batch.items():
  if key == 'dbid': continue
  batch[key] = value.cuda()
heads, foots, mean, logvar, z = model(batch, use_mean=True)

st.bar_chart(pd.DataFrame(z[0].detach().cpu().numpy(), columns=["z"]))


