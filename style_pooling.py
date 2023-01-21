import os
import time
import argparse

import numpy as np
import torch
import torch.nn.functional as F 
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter

import matplotlib as mpl
import matplotlib.pyplot as plt

#from hflayers import HopfieldLayer, HopfieldPooling
from kmeans_pytorch import kmeans, kmeans_predict

from recipe_dataset import RecipeDataset
from brewbrain import init_rng_seeding, build_datasets
from recipe_net_args import RecipeNetArgs, dataset_args
from model import RecipeNet, MODEL_FILE_KEY_NETWORK, MODEL_FILE_KEY_ARGS
from running_stats import RunningStats

def parse_args():
  parser = argparse.ArgumentParser()
  # Top-level program arguments
  parser.add_argument("--model", type=str, default="", help="Preexisting model to load (.chkpt file)")
  parser.add_argument("--style_model", type=str, default="", help="Preexisting style model to load (.chkpt file)")
  parser.add_argument("--seed", type=int, default=42, help="RNG seed of the experiment")

  # Training arguments
  parser.add_argument("--batch-size", type=int, default=256, help="batch size per training step")
  parser.add_argument("--num-epochs", type=int, default=10, help="number of epochs to train for")

  # Style Pooler arguments
  parser.add_argument("--pattern-size", type=int, default=3, help="Pattern/Pooling dimension (should be <= 3 for visualization)")
  parser.add_argument("--num-heads", type=int, default=16, help="Number of attention heads to use for pooling")

  cmd_args = parser.parse_args()
  return cmd_args


if __name__ == "__main__":
  cmd_args = parse_args()
  assert cmd_args.model != None and len(cmd_args.model) > 0 and os.path.exists(cmd_args.model)

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  init_rng_seeding(cmd_args.seed)
  dataset, train_dataset, test_dataset = build_datasets()
  train_size = len(train_dataset)
  train_dataloader = DataLoader(train_dataset, batch_size=cmd_args.batch_size, shuffle=True, num_workers=0)

  print(f"Loading Recipe Encoder from '{cmd_args.model}' ...")
  recipe_model_dict = torch.load(cmd_args.model, map_location=device)
  args = RecipeNetArgs(recipe_model_dict[MODEL_FILE_KEY_ARGS])
  recipe_model = RecipeNet(args).to(device)
  recipe_model.load_state_dict(recipe_model_dict[MODEL_FILE_KEY_NETWORK])
  recipe_encoder = recipe_model.encoder # We're only interested in the encoder, the rest can be purged from memory
  recipe_encoder.eval()
  del recipe_model
  del recipe_model_dict
  torch.cuda.empty_cache()
  print("Encoder loaded.")

  '''
  run_name = f"style_pooler__seed{cmd_args.seed}_z{args.z_size}_{int(time.time())}"
  run_dir = os.path.join("runs", run_name)
  os.makedirs(run_dir, exist_ok=True)

  global_step = 1

  # Setup our pooling model, optimizer, etc.
  style_pooler = HopfieldLayer(
    input_size=args.z_size,
    #hidden_size=cmd_args.hidden_size,
    pattern_size=cmd_args.pattern_size,
    quantity=117,
    #stored_pattern_size=cmd_args.hidden_size, # This doesn't work.
    #pattern_projection_size=args.z_size,
    #pattern_size=1,
    #num_heads=cmd_args.num_heads,
    update_steps_max=3,
    #scaling=8.0,
    #dropout=0.25,
    stored_pattern_as_static=True,
    state_pattern_as_static=True,
  ).to(device)
  optimizer = torch.optim.AdamW(style_pooler.parameters(), lr=1e-3) #, betas=(0.9, 0.999))
  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=100) 
  
  writer = SummaryWriter(run_dir)
  writer.add_text(
    "hyperparameters",
    "|param|value|\n|-|-|\n" + '\n'.join([f'|{key}|{value}|' for key, value in vars(cmd_args).items() if not isinstance(value, str) or len(value) > 0])
  )
  #writer.add_text("Model Summary", str(style_pooler).replace("\n", "  \n"))
  '''
  # Build the dataset of latents
  points = []
  for batch_idx, recipe_batch in enumerate(train_dataloader):
    batch = {}
    for key, value in recipe_batch.items():
      if key == 'dbid': continue
      batch[key] = value.to(device)
    z = recipe_encoder.z(batch, use_mean=True)
    points.append(z.detach().cpu())
  points = torch.concatenate(points, dim=0)

  NUM_CLUSTERS = 25

  # Project all the points into 3D space using PCA
  pU,pS,pV = torch.pca_lowrank(points)
  projected_points = torch.matmul(points, pV[:,:3])
  # K-Means cluster them (hopefully this has some semblance to each point's beer style)
  cluster_ids_x, cluster_centers = kmeans(X=projected_points, num_clusters=NUM_CLUSTERS, distance='euclidean', device=device)

  # Plot the points and the centers of their clusters
  fig = plt.figure(figsize=(4,3), dpi=160)
  ax = fig.add_subplot(projection='3d')

  pxs,pys,pzs = projected_points.detach().cpu().chunk(3,dim=1)
  colours = [mpl.cm.gist_rainbow(i/(NUM_CLUSTERS-1)) for i in cluster_ids_x]
  ax.scatter(pxs.numpy(),pys.numpy(),pzs.numpy(), c=colours, marker='x', linewidths=1, alpha=0.1)

  cxs,cys,czs = cluster_centers.detach().cpu().chunk(3,dim=1)
  ax.scatter(cxs.numpy(),cys.numpy(),czs.numpy(),c='black',marker='o',linewidths=3,alpha=1) # Cluster centers



  plt.show()
  
  #writer.add_mesh("style_cluster_centers", projected_pts, global_step=1)
  #writer.close()


  '''
  epoch_running_loss = RunningStats()

  # Go through the dataset, building latent representations, take those representations and
  # pass them through a hopfield pooler to generate 3-dimensional pools via training based on
  # reconstruction of those representations. The pooler should learn to build a k-means-like
  # clustering of the recipes in 3D space, conceptually this _should_ have some very close
  # similarity to recipe style groupings...
  for epoch_idx in range(cmd_args.num_epochs):
    for batch_idx, recipe_batch in enumerate(train_dataloader):
      batch = {}
      for key, value in recipe_batch.items():
        if key == 'dbid': continue
        batch[key] = value.to(device)

      # Encode the batch to latent 'z'
      z = recipe_encoder.z(batch, use_mean=False)

      # Feed 'z' into the pooler network, which will train a smaller association space based on
      # its reconstructed output 'z_hat'
      z_hat = style_pooler(z.unsqueeze(1))
      z_hat = z_hat.squeeze(1)

      # Simple MSE loss
      loss = F.mse_loss(z_hat, z, reduction='mean')
      
      optimizer.zero_grad() 
      loss.backward()
      #nn.utils.clip_grad_norm_(recipe_net.parameters(), 100.0)
      optimizer.step()
      scheduler.step(loss)

      global_step += 1
      epoch_running_loss.add(loss.item())
      print('\r', "Global Step:", global_step, "Loss:", f"{loss.item():>8.3f}", "lr:", optimizer.param_groups[0]['lr'], "\t\t\t", end='')
    
    print("\r\n", f"Epoch #{epoch_idx+1} Running Loss: [Mean: {np.around(epoch_running_loss.mean(), 3)}, StdDev: {np.around(epoch_running_loss.std(), 3)}]\t\t\t\t")
    epoch_running_loss.clear()
    
    # Build a point cloud of what the 3D values look like coming out of the style pooler
    with torch.no_grad():
      points = []
      for batch_idx, recipe_batch in enumerate(train_dataloader):
        batch = {}
        for key, value in recipe_batch.items():
          if key == 'dbid': continue
          batch[key] = value.to(device)
        z = recipe_encoder.z(batch, use_mean=True)
        pooled = style_pooler.get_projected_pattern_matrix(z.unsqueeze(1)).mean(dim=[1,2])
        points.append(pooled.detach().cpu())

      points = torch.concatenate(points, dim=0).unsqueeze(0)
      writer.add_mesh("style_pooling", points, global_step=global_step)
  '''
