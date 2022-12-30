import random
import pickle
import argparse
from distutils.util import strtobool

import torch
from torch.utils.data import DataLoader
import numpy as np

from recipe_dataset import RecipeDataset, RECIPE_DATASET_TEST_FILENAME
from recipe_converter import RecipeConverter
from recipe_net_args import RecipeNetArgs
from model import RecipeNet, MODEL_FILE_KEY_NETWORK, MODEL_FILE_KEY_ARGS
from recipe_dataset import DatasetMappings
from brewbrain import init_rng_seeding, build_datasets

'''
def parse_args():
  bool_val_fn = lambda x: bool(strtobool(x.strip()))
  
  parser = argparse.ArgumentParser()
  # Top-level program arguments
  parser.add_argument("--model", type=str, default="", help="Preexisting model to load (.chkpt file)")

  cmd_args = parser.parse_args()
  return cmd_args
'''

NETWORK_MODEL_FILEPATH = "runs/recipe_net__seed42_beta-tc_z32_hidden8096-4096_1672268963/recipe_net_415000.chkpt"

if __name__ == "__main__":
  SEED = 42
  init_rng_seeding(SEED)

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


  # Load the dataset and create a dataloader for it
  dataset, train_dataset, test_dataset = build_datasets()

  dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

  dataset_mappings = DatasetMappings()
  dataset_mappings.init_from_dataset(dataset)
  converter = RecipeConverter(dataset_mappings)

  model_dict = torch.load(NETWORK_MODEL_FILEPATH)
  args = RecipeNetArgs()
  for key, value in model_dict[MODEL_FILE_KEY_ARGS].items():
    setattr(args, key, value)
  model = RecipeNet(args).to(device)
  model.load_state_dict(model_dict[MODEL_FILE_KEY_NETWORK])

  model.eval()

  recipe_batch = next(iter(dataloader))
  batch = {}
  for key, value in recipe_batch.items():
    if key == 'dbid': continue
    batch[key] = value.cuda()
  heads, foots, mean, logvar, z = model(batch, use_mean=True)
  #reconst_loss = model.reconstruction_loss(batch, heads, foots)
  input_recipes  = converter.batch_to_recipes(recipe_batch)
  output_recipes = converter.net_output_to_recipes(foots)

  print("Original:")
  print(recipe_batch)
  print("-------")
  print("Decoded:")
  print(output_recipes[0])
