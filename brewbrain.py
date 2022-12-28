import os
import time
import pickle
import random
import argparse
from distutils.util import strtobool

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

from torch.utils.tensorboard import SummaryWriter

from sqlalchemy import create_engine
from brewbrain_db import Base, BREWBRAIN_DB_FILENAME, build_db_str
from file_utils import find_file_cwd_and_parent_dirs
from recipe_dataset import core_grain_labels, core_adjunct_labels, hop_labels, misc_labels, microorganism_labels
from recipe_dataset import RecipeDataset, RECIPE_DATASET_FILENAME
from model import RecipeNet, BetaVAELoss, BetaTCVAELoss
from model import MODEL_FILE_KEY_GLOBAL_STEP, MODEL_FILE_KEY_NETWORK, MODEL_FILE_KEY_OPTIMIZER, MODEL_FILE_KEY_NET_TYPE, MODEL_FILE_KEY_SCHEDULER, MODEL_FILE_KEY_ARGS
from recipe_net_args import RecipeNetArgs, dataset_args
from running_stats import RunningStats

KL_WEIGHT  = 1.0
OUTLIER_MIN_LOSS = 3.5e4
OUTLIER_PER_RECIPE_MIN_LOSS = 1e4

def load_loss_fn(net_type):
  if net_type == "beta":
    return BetaVAELoss(net_args)
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
  parser.add_argument("--seed", type=int, default=42, help="RNG seed of the experiment")
  parser.add_argument("--cuda", type=bool_val_fn, default=True, help="if toggled, cuda will be enabled by default")
  
  # Network training-specific program arguments
  parser.add_argument("--net-type", type=str, default="beta-tc", help="Type of VAE Network {beta-tc, beta}")
  parser.add_argument("--batch-size", type=int, default=256, help="batch size per training step")
  parser.add_argument("--num-epochs", type=int, default=500, help="number of epochs to train for")
  parser.add_argument("--save-timesteps", type=int, default=5000, help="Global steps between network saves")
  parser.add_argument("--train-percent", type=float, default=0.75, help="Percentage of the dataset to train on")

  cmd_args = parser.parse_args()
  return cmd_args

if __name__ == "__main__":
  cmd_args = parse_args()
  device = torch.device("cuda" if torch.cuda.is_available() and cmd_args.cuda else "cpu")
  
  random.seed(cmd_args.seed)
  np.random.seed(cmd_args.seed)
  torch.manual_seed(cmd_args.seed)
  torch.backends.cudnn.deterministic = True

  # Load the dataset and create a dataloader for it
  with open(RECIPE_DATASET_FILENAME, 'rb') as f:
    dataset = pickle.load(f)

  # Split the dataset up for training / testing
  dataset_size = len(dataset)
  train_size = int(dataset_size * cmd_args.train_percent)
  train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, dataset_size-train_size])

  train_dataloader = DataLoader(train_dataset, batch_size=cmd_args.batch_size, shuffle=True, num_workers=0)
  net_args = RecipeNetArgs(dataset_args(dataset))

  # Embedding labels
  db_filepath = build_db_str(find_file_cwd_and_parent_dirs(BREWBRAIN_DB_FILENAME, os.getcwd()))
  engine = create_engine(db_filepath, echo=False, future=True)
  Base.metadata.create_all(engine)
  grain_type_embedding_labels = core_grain_labels(engine, dataset)
  adjunct_type_embedding_labels = core_adjunct_labels(engine, dataset)
  hop_type_embedding_labels = hop_labels(engine, dataset)
  misc_type_embedding_labels = misc_labels(engine, dataset)
  microorganism_type_embedding_labels = microorganism_labels(engine, dataset)

  recipe_net = RecipeNet(net_args).to(device)
  optimizer  = torch.optim.Adam(recipe_net.parameters(), lr=5e-4) #, betas=(0.9, 0.999))
  scheduler  = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=1.0) # Step size is in epochs
  init_global_step = 1
  global_step = init_global_step
  loss_fn = load_loss_fn(cmd_args.net_type)

  # Load the checkpoint/model file if one was provided
  if cmd_args.model is not None and len(cmd_args.model) > 0:
    if os.path.exists(cmd_args.model):
      print(f"Model file '{cmd_args.model}' found, loading...")
      model_dict = torch.load(cmd_args.model)
      load_failed = False
      try:
        recipe_net.load_state_dict(model_dict[MODEL_FILE_KEY_NETWORK], strict=False)
        if MODEL_FILE_KEY_NET_TYPE in model_dict:
          loss_fn = load_loss_fn(model_dict[MODEL_FILE_KEY_NET_TYPE])
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

  running_loss = RunningStats()

  # Monitor the recipe network using hooks and tensorboard
  MONITOR_UPDATE_STEPS = 1000
  def histogram_hook(tag, tensor):
    if global_step % MONITOR_UPDATE_STEPS == 0:
      writer.add_histogram(tag, tensor.flatten().detach().cpu(), global_step)
      writer.flush() 
    return None

  for name, layer in recipe_net.named_children():
    if name == 'encoder':
      encoder_children = list(layer.named_children())
      # Distributions of outputs after the first layer+activation
      first_actfn = encoder_children[1][1]
      first_actfn.register_forward_hook(lambda layer, input, output: histogram_hook("dists/outputs/encoder_first_actfn", output))
      # Distribution of outputs after the last layer+activation (before batchnorm)
      num_hidden_layers = len(net_args.hidden_layers)
      last_layer = encoder_children[num_hidden_layers*2][1]
      last_layer.register_forward_hook(lambda layer, input, output: histogram_hook("dists/outputs/encoder_last_layer_output", output))
      # Distribution of outputs after the encoder (last layer is a batchnorm1D)
      batchnorm = encoder_children[-1][1]
      batchnorm.register_forward_hook(lambda layer, input, output: histogram_hook("dists/outputs/encoder_batchnorm", output))
      # Distributions of weights of the first and last layers
      first_layer = encoder_children[0][1]
      first_layer.register_forward_hook(lambda layer, input, output: histogram_hook("dists/weights/encoder_first_layer_weights", layer.weight))
      last_layer.register_forward_hook(lambda layer, input, output: histogram_hook("dists/weights/encoder_last_layer_weights", layer.weight))
      
    elif name == 'decoder':
      decoder_children = list(layer.named_children())
      first_actfn = decoder_children[1][1]
      first_actfn.register_forward_hook(lambda layer, input, output: histogram_hook("dists/outputs/decoder_first_actfn", output))
      num_hidden_layers = len(net_args.hidden_layers)
      last_layer = decoder_children[num_hidden_layers*2][1]
      last_layer.register_forward_hook(lambda layer, input, output: histogram_hook("dists/outputs/decoder_last_layer_output", output))
      # Distributions of weights of the first and last layers
      first_layer = decoder_children[0][1]
      first_layer.register_forward_hook(lambda layer, input, output: histogram_hook("dists/weights/decoder_first_layer_weights", layer.weight))
      last_layer.register_forward_hook(lambda layer, input, output: histogram_hook("dists/weights/decoder_last_layer_weights", layer.weight))
    elif name == 'head_encoder':
      pass
    else: # name == 'foot_decoder':
      pass


  outliers = {}
  for epoch_idx in range(cmd_args.num_epochs):

    for batch_idx, recipe_batch in enumerate(train_dataloader):
      batch = {}
      for key, value in recipe_batch.items():
        if key == 'dbid': continue
        batch[key] = value.cuda()

      heads, foots, mean, logvar, z = recipe_net(batch)
      reconst_loss = recipe_net.reconstruction_loss(batch, heads, foots)
      loss_vals = loss_fn.calc_loss(
        reconst_loss=reconst_loss, z=z, dataset_size=train_size, 
        mean=mean, logvar=logvar, iter_num=global_step-1
      )
      loss = loss_vals['loss']
      running_loss.add(loss.detach().cpu().item())

      for key, val in loss_vals.items():
        writer.add_scalar(f"charts/{key}", val.detach().cpu().item(), global_step)
      
      ''' 
      if epoch_idx > 0 and loss.item() > OUTLIER_MIN_LOSS:
        with torch.no_grad():
          # Go through each item in the batch and see what its loss is
          reconst_loss = recipe_net.reconstruction_loss(batch, heads, foots, 'none')
          loss_result = loss_fn.calc_loss(reconst_loss=reconst_loss, z=z, dataset_size=train_size, mean=mean, logvar=logvar)

          # Find the Database IDs for the recipes with the worst losses:
          # Sort the losses with their indices, highest to lowest
          loss_dict = {idx: loss.item() for idx, loss in enumerate(loss_result['loss'].detach().cpu())}
          ordered_idx_loss_tuples = sorted(loss_dict.items(), key=lambda x: x[1], reverse=True)

          dbids = recipe_batch['dbid']
          for idx, loss_val in ordered_idx_loss_tuples:
            if loss_val <= OUTLIER_PER_RECIPE_MIN_LOSS: break
            dbid = dbids[idx].item()
            if dbid not in outliers:
              outliers[dbid] = {'loss': loss_val, 'count': 1}
            else:
              outliers[dbid]['count'] += 1
      '''  
      optimizer.zero_grad() 
      loss.backward()
      nn.utils.clip_grad_norm_(recipe_net.parameters(), 100.0)
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
      
    print("\r\n", f"Epoch #{epoch_idx+1} Running Loss: [Mean: {np.around(running_loss.mean(), 3)}, StdDev: {np.around(running_loss.std(), 3)}]\t\t\t\t")
    if len(outliers) > 0:
      print("", f"Current outliers: {sorted(sorted(outliers.items(), key=lambda x: x[1]['count'], reverse=True), key=lambda x: x[1]['loss'], reverse=True)[:min(len(outliers),50)]}")
    running_loss.clear()
    scheduler.step()

  writer.close()
