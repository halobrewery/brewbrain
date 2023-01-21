import os
import time
import pickle
import random
import argparse
from distutils.util import strtobool

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from torch.utils.tensorboard import SummaryWriter

#from sqlalchemy import create_engine
#from brewbrain_db import Base, BREWBRAIN_DB_FILENAME, build_db_str
#from file_utils import find_file_cwd_and_parent_dirs
#from recipe_dataset import core_grain_labels, core_adjunct_labels, hop_labels, misc_labels, microorganism_labels
from recipe_dataset import RecipeDataset, RECIPE_DATASET_FILENAME, load_dataset
from model import RecipeNet, BetaVAELoss, BetaTCVAELoss
from model import MODEL_FILE_KEY_GLOBAL_STEP, MODEL_FILE_KEY_NETWORK, MODEL_FILE_KEY_OPTIMIZER, MODEL_FILE_KEY_NET_TYPE, MODEL_FILE_KEY_SCHEDULER, MODEL_FILE_KEY_ARGS
from recipe_net_args import RecipeNetArgs, dataset_args
from running_stats import RunningStats

def init_rng_seeding(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.backends.cudnn.deterministic = True

def build_datasets(filepath=RECIPE_DATASET_FILENAME, train_percent=0.75, seed=42):
  dataset = load_dataset(filepath)
  # Split the dataset up for training / testing
  dataset_size = len(dataset)
  train_size = int(dataset_size * train_percent)
  # Make sure the split is 'random' but deterministic based on the given seed
  # by providing a generator
  train_dataset, test_dataset = torch.utils.data.random_split(
    dataset, [train_size, dataset_size-train_size],
    generator=torch.Generator().manual_seed(seed)
  )
  return dataset, train_dataset, test_dataset


def load_loss_fn(net_type, device):
  if net_type == "beta":
    return BetaVAELoss(net_args, device)
  elif net_type == "beta-tc":
    return BetaTCVAELoss(net_args)
  else:
    print(f"Invalid VAE network type: '{cmd_args.net_type}', defaulting to Beta-TC VAE.")
    return BetaTCVAELoss(net_args)

def parse_args():
  bool_val_fn = lambda x: bool(strtobool(x.strip()))
  
  parser = argparse.ArgumentParser()
  # Top-level program arguments
  parser.add_argument("--model", type=str, default="", help="Preexisting model to load (.chkpt file)")
  parser.add_argument("--seed", type=int, default=42, help="RNG seed for splitting the dataset")
  parser.add_argument("--cuda", type=bool_val_fn, default=True, help="if toggled, cuda will be enabled by default")
  
  # Network training-specific program arguments
  parser.add_argument("--net-type", type=str, default="beta", help="Type of VAE Network {beta-tc, beta}")
  parser.add_argument("--batch-size", type=int, default=256, help="batch size per training step")
  parser.add_argument("--num-epochs", type=int, default=500, help="number of epochs to train for")
  parser.add_argument("--save-timesteps", type=int, default=5000, help="Global steps between network saves")
  parser.add_argument("--train-percent", type=float, default=0.75, help="Percentage of the dataset to train on")

  cmd_args = parser.parse_args()
  return cmd_args

if __name__ == "__main__":
  cmd_args = parse_args()
  device = torch.device("cuda" if torch.cuda.is_available() and cmd_args.cuda else "cpu")
  
  #init_rng_seeding(cmd_args.seed)

  # Load the dataset and create a dataloader for it
  dataset, train_dataset, test_dataset = build_datasets(train_percent=cmd_args.train_percent, seed=cmd_args.seed)
  train_size = len(train_dataset)
  train_dataloader = DataLoader(train_dataset, batch_size=cmd_args.batch_size, shuffle=True, num_workers=0)
  
  net_args = RecipeNetArgs(dataset_args(dataset))

  # Embedding labels
  #db_filepath = build_db_str(find_file_cwd_and_parent_dirs(BREWBRAIN_DB_FILENAME, os.getcwd()))
  #engine = create_engine(db_filepath, echo=False, future=True)
  #Base.metadata.create_all(engine)
  #grain_type_embedding_labels = core_grain_labels(engine, dataset)
  #adjunct_type_embedding_labels = core_adjunct_labels(engine, dataset)
  #hop_type_embedding_labels = hop_labels(engine, dataset)
  #misc_type_embedding_labels = misc_labels(engine, dataset)
  #microorganism_type_embedding_labels = microorganism_labels(engine, dataset)

  recipe_net = RecipeNet(net_args).to(device)
  optimizer  = torch.optim.AdamW(recipe_net.parameters(), lr=1e-4, weight_decay=0) #, betas=(0.9, 0.999))
  scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1, factor=0.5) #step_size=3, gamma=0.5) # Step size is in epochs
  init_global_step = 1
  global_step = init_global_step
  loss_fn = load_loss_fn(cmd_args.net_type, device)

  # Load the checkpoint/model file if one was provided
  if cmd_args.model is not None and len(cmd_args.model) > 0:
    if os.path.exists(cmd_args.model):
      print(f"Model file '{cmd_args.model}' found, loading...")
      model_dict = torch.load(cmd_args.model, map_location=device)
      load_failed = False
      try:
        recipe_net.load_state_dict(model_dict[MODEL_FILE_KEY_NETWORK], strict=False)
        if MODEL_FILE_KEY_NET_TYPE in model_dict:
          loss_fn = load_loss_fn(model_dict[MODEL_FILE_KEY_NET_TYPE], device)
          cmd_args.net_type = model_dict[MODEL_FILE_KEY_NET_TYPE]
      except RuntimeError as e:
        print("Could not load agent networks:")
        print(e)
        load_failed = True

      if not load_failed:
        try:
          optimizer.load_state_dict(model_dict[MODEL_FILE_KEY_OPTIMIZER])
        except ValueError as e:
          print("Could not load optimizer:")
          print(e)
          load_failed = True
        if not load_failed:
          init_global_step = model_dict[MODEL_FILE_KEY_GLOBAL_STEP]
          global_step      = init_global_step

          if MODEL_FILE_KEY_SCHEDULER in model_dict:
            try:
              scheduler.load_state_dict(model_dict[MODEL_FILE_KEY_SCHEDULER])
            except ValueError as e:
              print("Could not load scheduler:")
              print(e)
        
          print("Model loaded!")
      else:
        print("Model loaded with failures.")
      del model_dict
      torch.cuda.empty_cache()
    else:
      print(f"Could not find/load model file '{cmd_args.model}'")

  run_name = f"recipe_net__seed{cmd_args.seed}_{cmd_args.net_type}_z{net_args.z_size}_hidden{'-'.join([str(l) for l in net_args.hidden_layers])}_{int(time.time())}"
  run_dir = os.path.join("runs", run_name)
  os.makedirs(run_dir, exist_ok=True)

  writer = SummaryWriter(run_dir)
  writer.add_text(
    "hyperparameters",
    "|param|value|\n|-|-|\n" + '\n'.join([f'|{key}|{value}|' for key, value in vars(cmd_args).items() if not isinstance(value, str) or len(value) > 0]) +
    '\n'.join([f'|{key}|{value}|' for key, value in vars(net_args).items()])
  )
  writer.add_text("Model Summary", str(recipe_net).replace("\n", "  \n"))


  # Monitor the recipe network using pytorch hooks and tensorboard
  def hook_lambda(layer_name, update_steps=2500):
    def histogram_hook(layer, _, output):
      weights = layer.weight
      if global_step == 1 or global_step % update_steps == 0:
        writer.add_histogram(f"dist/weights/{layer_name}", weights.detach().flatten().cpu(), global_step)
        writer.add_histogram(f"dist/output/{layer_name}", output.detach().flatten().cpu(), global_step)
      return None
    return histogram_hook

  # Record initialized weights for all the layers and add hooks for recording
  # weights and outputs throughout training
  layer_idx = 0
  for name, layer in recipe_net.encoder.encoder.named_children():
    if isinstance(layer, nn.Linear):
      layer.register_forward_hook(hook_lambda(f"encoder_{name}[{layer_idx}]"))
      layer_idx += 1

  recipe_net.encoder.encode_mean.register_forward_hook(hook_lambda("encoder_mean"))
  recipe_net.encoder.encode_logvar.register_forward_hook(hook_lambda("encoder_logvar"))
  
  layer_idx = 0
  for name, layer in recipe_net.decoder.decoder.named_children():
    if isinstance(layer, nn.Linear):
      layer.register_forward_hook(hook_lambda(f"decoder_{name}[{layer_idx}]"))
      layer_idx += 1
  writer.flush() 

  epoch_running_loss = RunningStats()
  #outlier_loss_window = np.zeros(len(train_dataloader))
  #outlier_window_idx  = 0
  outliers = {}
  KL_WEIGHT = 1.0 #cmd_args.batch_size / train_size
  GRAD_CLIP = 100.0

  recipe_net.train()
  for epoch_idx in range(cmd_args.num_epochs):

    for batch_idx, recipe_batch in enumerate(train_dataloader):
      batch = {}
      for key, value in recipe_batch.items():
        if key == 'dbid': continue
        batch[key] = value.to(device)

      _, x_hat_dict, mean_z, logvar_z, logvar_x, z = recipe_net(batch)
      loss_vals = recipe_net.calc_loss(batch, x_hat_dict, mean_z, logvar_z, logvar_x)
      loss = loss_vals['loss']

      loss_value = loss.detach().cpu().item()
      epoch_running_loss.add(loss_value)

      #outlier_loss_window[outlier_window_idx] = loss_value
      #outlier_window_idx = (outlier_window_idx + 1) % len(outlier_loss_window)

      for key, val in loss_vals.items():
        writer.add_scalar(f"charts/{key}", val.detach().cpu().item(), global_step)
      '''
      if epoch_idx > 0:
        z_score = (loss_value - outlier_loss_window.mean()) / outlier_loss_window.std()
        if z_score > 4.0:
          with torch.no_grad():
            # Go through each item in the batch and see what its loss is
            reconst_loss = recipe_net.reconstruction_loss(batch, heads, foots, 'none')
            reconst_loss_mean = reconst_loss.mean().detach().cpu().item()
            reconst_loss_std  = reconst_loss.std().detach().cpu().item()

            # Find the Database IDs for the recipes with the worst losses:
            # Sort the losses with their indices, highest to lowest
            loss_dict = {idx: loss.item() for idx, loss in enumerate(reconst_loss.detach().cpu())}
            ordered_idx_loss_tuples = sorted(loss_dict.items(), key=lambda x: x[1], reverse=True)

            dbids = recipe_batch['dbid']
            for idx, loss_val in ordered_idx_loss_tuples:
              loss_val_z_score = (loss_val - reconst_loss_mean) / reconst_loss_std
              if loss_val_z_score <= 0: break
              dbid = dbids[idx].item()
              if dbid not in outliers:
                outliers[dbid] = {'loss': np.round(loss_val,2), 'mean': np.round(reconst_loss_mean,2), 'count': 1}
              else:
                outliers[dbid]['count'] += 1
        '''
      optimizer.zero_grad() 
      loss.backward()
      nn.utils.clip_grad_norm_(recipe_net.parameters(), GRAD_CLIP)
      optimizer.step()
      
      global_step += 1

      # Save the network every so often...
      if global_step % cmd_args.save_timesteps == 0:
        print("\r\n", "Saving network...")
        save_dict = {
          MODEL_FILE_KEY_GLOBAL_STEP: global_step,
          MODEL_FILE_KEY_NETWORK    : recipe_net.state_dict(),
          MODEL_FILE_KEY_OPTIMIZER  : optimizer.state_dict(),
          MODEL_FILE_KEY_SCHEDULER  : scheduler.state_dict(),
          MODEL_FILE_KEY_NET_TYPE   : cmd_args.net_type,
          MODEL_FILE_KEY_ARGS       : net_args.__dict__,
        }
        save_path = os.path.join(run_dir, f"recipe_net_{global_step}.chkpt")
        torch.save(save_dict, save_path)
        print("\r\n", f"Network saved ({save_path})")

      loss_str = [f"{key}: {val.item():>10.3f}" for key, val in loss_vals.items()]
      print('\r', "Global Step:", global_step, " ".join(loss_str), "lr:", optimizer.param_groups[0]['lr'], "\t\t\t", end='')
        
    # Send the head encoder's embeddings to tensorboard
    #writer.add_embedding(recipe_net.head_encoder.grain_type_embedding.weight, grain_type_embedding_labels, tag="grain_type")
    #writer.add_embedding(recipe_net.head_encoder.adjunct_type_embedding.weight, adjunct_type_embedding_labels, tag="adjunct_type")
    #writer.add_embedding(recipe_net.head_encoder.hop_type_embedding.weight, hop_type_embedding_labels, tag="hop_type")
    #writer.add_embedding(recipe_net.head_encoder.misc_type_embedding.weight, misc_type_embedding_labels, tag="misc_type")
    #writer.add_embedding(recipe_net.head_encoder.microorganism_type_embedding.weight, microorganism_type_embedding_labels, tag="microorganism_type")        
    #writer.flush()
    
    writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]['lr'], global_step)
    print("\r\n", f"Epoch #{epoch_idx+1} Running Loss: [Mean: {np.around(epoch_running_loss.mean(), 3)}, StdDev: {np.around(epoch_running_loss.std(), 3)}]\t\t\t\t")
    if len(outliers) > 0:
      print("", f"Current outliers: {sorted(sorted(outliers.items(), key=lambda x: x[1]['count'], reverse=True), key=lambda x: x[1]['loss'], reverse=True)[:min(len(outliers),50)]}")
    
    scheduler.step(epoch_running_loss.mean())
    epoch_running_loss.clear()

  writer.close()
