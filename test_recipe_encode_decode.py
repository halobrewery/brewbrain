import random
import pickle
import argparse
from distutils.util import strtobool

import torch
from torch.utils.data import DataLoader
import numpy as np

from recipe_dataset import RecipeDataset, RECIPE_DATASET_FILENAME, net_output_to_recipe
from recipe_net_args import RecipeNetArgs
from model import RecipeNet, MODEL_FILE_KEY_NETWORK, MODEL_FILE_KEY_ARGS


'''
def parse_args():
  bool_val_fn = lambda x: bool(strtobool(x.strip()))
  
  parser = argparse.ArgumentParser()
  # Top-level program arguments
  parser.add_argument("--model", type=str, default="", help="Preexisting model to load (.chkpt file)")

  cmd_args = parser.parse_args()
  return cmd_args
'''

NETWORK_MODEL_FILEPATH = "runs/recipe_net__seed42_beta-tc_z64_hidden4096-4096_1672243396/recipe_net_45000.chkpt"

if __name__ == "__main__":
  SEED = 42
  random.seed(SEED)
  np.random.seed(SEED)
  torch.manual_seed(SEED)
  torch.backends.cudnn.deterministic = True

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  model_dict = torch.load(NETWORK_MODEL_FILEPATH)
  args = RecipeNetArgs()
  for key, value in model_dict[MODEL_FILE_KEY_ARGS].items():
    setattr(args, key, value)
  model = RecipeNet(args).to(device)
  model.load_state_dict(model_dict[MODEL_FILE_KEY_NETWORK])

  # Load the dataset and create a dataloader for it
  with open(RECIPE_DATASET_FILENAME, 'rb') as f:
    dataset = pickle.load(f)
  dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

  model.eval()

  recipe_batch = next(iter(dataloader))
  batch = {}
  for key, value in recipe_batch.items():
    if key == 'dbid': continue
    batch[key] = value.cuda()
  heads, foots, mean, logvar, z = model(batch)
  #reconst_loss = model.reconstruction_loss(batch, heads, foots)
  net_output_to_recipe(foots, dataset.normalizers)