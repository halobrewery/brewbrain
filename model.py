
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from recipe_net_args import RecipeNetArgs

def layer_init_ortho(layer, std=np.sqrt(2)):
  nn.init.orthogonal_(layer.weight, std)
  if layer.bias != None:
    nn.init.constant_(layer.bias, 0.0)
  return layer

def layer_init_xavier(layer, gain):
  nn.init.xavier_normal_(layer.weight, gain)
  if layer.bias != None:
    nn.init.constant_(layer.bias, 0.0)
  return layer

def reparameterize(mu, logvar):
  std = torch.exp(0.5 * logvar)
  eps = torch.randn_like(std)
  return eps * std + mu

class RecipeNetData(object):
  def __init__(self) -> None:
    pass
  
class RecipeNetHeadEncoder(nn.Module):
  def __init__(self, args) -> None:
    super().__init__()
    # Embeddings (NOTE: Any categoricals that don't have embeddings will be one-hot encoded)
    self.grain_type_embedding         = nn.Embedding(args.num_grain_types, args.grain_type_embed_size)
    self.adjunct_type_embedding       = nn.Embedding(args.num_adjunct_types, args.adjunct_type_embed_size) 
    self.hop_type_embedding           = nn.Embedding(args.num_hop_types, args.hop_type_embed_size)
    self.misc_type_embedding          = nn.Embedding(args.num_misc_types, args.misc_type_embed_size)
    self.microorganism_type_embedding = nn.Embedding(args.num_microorganism_types, args.microorganism_type_embed_size)
    self.args = args
    
  def forward(self, x):
    heads = RecipeNetData()
    # Simple top-level heads (high-level recipe parameters)
    heads.x_toplvl = torch.cat((x['boil_time'].unsqueeze(1), x['mash_ph'].unsqueeze(1), x['sparge_temp'].unsqueeze(1)), dim=1) # (B, 3)
    
    # Mash step heads
    # NOTE: Data shape is (B, S=number_of_mash_steps) for the
    # following recipe tensors: {'mash_step_type_inds', 'mash_step_times', 'mash_step_avg_temps'}
    num_mash_step_types = self.args.num_mash_step_types
    heads.enc_mash_step_type_onehot = F.one_hot(x['mash_step_type_inds'].long(), num_mash_step_types).float().flatten(1) # (B, S, num_mash_step_types) -> (B, S*num_mash_step_types) = [B, 24]
    heads.x_mash_steps = torch.cat((heads.enc_mash_step_type_onehot, x['mash_step_times'], x['mash_step_avg_temps']), dim=1) # (B, num_mash_step_types*S+S+S) = [B, 36=(24+6+6)]
    
    # Ferment stage heads
    # NOTE: Data shape is (B, S=2) for the following recipe tensors: {'ferment_stage_times', 'ferment_stage_temps'}
    heads.x_ferment_stages = torch.cat((x['ferment_stage_times'], x['ferment_stage_temps']), dim=1) # (B, S+S)

    # Grain (malt bill) heads
    # NOTE: Data shape is (B, S=num_grain_slots) for the following recipe tensors: {'grain_core_type_inds', 'grain_amts'}
    num_grain_types = self.args.num_grain_types
    heads.enc_grain_type_embed = self.grain_type_embedding(x['grain_core_type_inds']).flatten(1) # (B, S, grain_type_embed_size) -> (B, S*grain_type_embed_size)
    heads.enc_grain_type_onehot = F.one_hot(x['grain_core_type_inds'].long(), num_grain_types).float() # (B, num_grain_slots, num_grain_types)
    heads.x_grains = torch.cat((heads.enc_grain_type_embed, x['grain_amts']), dim=1) # (B, S*grain_type_embed_size+S)
    
    # Adjunct heads
    # NOTE: Data shape is (B, S=num_adjunct_slots) for the following recipe tensors: {'adjunct_core_type_inds', 'adjunct_amts'}
    num_adjunct_types = self.args.num_adjunct_types
    heads.enc_adjunct_type_embed = self.adjunct_type_embedding(x['adjunct_core_type_inds']).flatten(1) # (B, S, adjunct_type_embed_size) -> (B, S*adjunct_type_embed_size)
    heads.enc_adjunct_type_onehot = F.one_hot(x['adjunct_core_type_inds'].long(), num_adjunct_types).float() # (B, num_adjunct_slots, num_adjunct_types)
    heads.x_adjuncts = torch.cat((heads.enc_adjunct_type_embed, x['adjunct_amts']), dim=1) # (B, S*adjunct_type_embed_size+S)
    
    # Hop heads
    # NOTE: Data shape is (B, S=num_hop_slots) for the following recipe tensors: 
    # {'hop_type_inds', 'hop_stage_type_inds', 'hop_times', 'hop_concentrations'}
    num_hop_types = self.args.num_hop_types
    num_hop_stage_types = self.args.num_hop_stage_types
    heads.enc_hop_type_embed = self.hop_type_embedding(x['hop_type_inds']).flatten(1) # (B, S, hop_type_embed_size)
    heads.enc_hop_type_onehot = F.one_hot(x['hop_type_inds'].long(), num_hop_types).float() # (B, num_hop_slots, num_hop_types)
    heads.enc_hop_stage_type_onehot = F.one_hot(x['hop_stage_type_inds'].long(), num_hop_stage_types).float().flatten(1) # (B, S, num_hop_stage_types)
    heads.x_hops = torch.cat((heads.enc_hop_type_embed, heads.enc_hop_stage_type_onehot, x['hop_times'], x['hop_concentrations']), dim=1) # (B, S*hop_type_embed_size + S*num_hop_stage_types + S + S)
    
    # Misc. heads
    # NOTE: Data shape is (B, S=num_misc_slots) for the following recipe tensors:
    # {'misc_type_inds', 'misc_stage_inds', 'misc_times', 'misc_amts'}
    num_misc_types = self.args.num_misc_types
    num_misc_stage_types = self.args.num_misc_stage_types
    heads.enc_misc_type_embed = self.misc_type_embedding(x['misc_type_inds']).flatten(1) # (B, S, misc_type_embed_size)
    heads.enc_misc_type_onehot = F.one_hot(x['misc_type_inds'].long(), num_misc_types).float() # (B, num_misc_slots, num_misc_types)
    heads.enc_misc_stage_type_onehot = F.one_hot(x['misc_stage_inds'].long(), num_misc_stage_types).float().flatten(1) # (B, S, num_misc_stage_types)
    heads.x_miscs = torch.cat((heads.enc_misc_type_embed, heads.enc_misc_stage_type_onehot, x['misc_times'], x['misc_amts']), dim=1) # (B, S*misc_type_embed_size + S*num_misc_stage_types + S + S)
    
    # Microorganism heads
    # NOTE: Data shape is (B, S=num_microorganism_slots) for the following recipe tensors:
    # {'mo_type_inds', 'mo_stage_inds'}
    num_mo_types = self.args.num_microorganism_types
    num_mo_stage_types = self.args.num_mo_stage_types
    heads.enc_mo_type_embed = self.microorganism_type_embedding(x['mo_type_inds']).flatten(1) # (B, S, microorganism_type_embed_size)
    heads.enc_mo_type_onehot = F.one_hot(x['mo_type_inds'].long(), num_mo_types).float() # (B, num_mo_slots, num_mo_types)
    heads.enc_mo_stage_type_onehot = F.one_hot(x['mo_stage_inds'].long(), num_mo_stage_types).float().flatten(1) # (B, S, num_mo_stage_types)
    heads.x_mos = torch.cat((heads.enc_mo_type_embed, heads.enc_mo_stage_type_onehot), dim=1) # (B, S*microorganism_type_embed_size + S*num_mo_stage_types)
    
    # Put all the recipe data together into a flattened tensor
    x = torch.cat((heads.x_toplvl, heads.x_mash_steps, heads.x_ferment_stages, heads.x_grains, heads.x_adjuncts, heads.x_hops, heads.x_miscs, heads.x_mos), dim=1) # (B, num_inputs)
    return x, heads

class RecipeNetFootDecoder(nn.Module):
  def __init__(self, args: RecipeNetArgs) -> None:
    super().__init__()
    gain = args.gain
    self.grain_type_decoder         = layer_init_xavier(nn.Linear(args.grain_type_embed_size, args.num_grain_types, bias=False), gain)
    self.adjunct_type_decoder       = layer_init_xavier(nn.Linear(args.adjunct_type_embed_size, args.num_adjunct_types, bias=False), gain)
    self.hop_type_decoder           = layer_init_xavier(nn.Linear(args.hop_type_embed_size, args.num_hop_types, bias=False), gain)
    self.misc_type_decoder          = layer_init_xavier(nn.Linear(args.misc_type_embed_size, args.num_misc_types, bias=False), gain)
    self.microorganism_type_decoder = layer_init_xavier(nn.Linear(args.microorganism_type_embed_size, args.num_microorganism_types, bias=False), gain)
    
    # [Top-level recipe attributes, Mash steps, Fermentation stages, Grains, Adjuncts, Hops, Misc, Microorganisms]
    self.split_sizes = [
      args.num_toplvl_inputs, args.num_mash_step_inputs, args.num_ferment_stage_inputs, 
      args.num_grain_slot_inputs, args.num_adjunct_slot_inputs, args.num_hop_slot_inputs,
      args.num_misc_slot_inputs, args.num_microorganism_slot_inputs
    ]
    #assert np.sum(se)
    self.args = args
    
  def forward(self, x_hat):
    foots = RecipeNetData()
    
    # The decoded tensor is flat with a shape of (B, num_inputs), we'll need to break it apart
    # so that we can eventually calculate losses appropriately for each head of original data fed to the encoder
    foots.x_hat_toplvl, foots.x_hat_mash_steps, foots.x_hat_ferment_stages, foots.x_hat_grains, foots.x_hat_adjuncts, foots.x_hat_hops, foots.x_hat_miscs, foots.x_hat_mos = torch.split(x_hat, self.split_sizes, dim=1)

    # Mash steps
    num_mash_steps = self.args.num_mash_steps
    enc_mash_step_type_onehot_size = num_mash_steps * self.args.num_mash_step_types
    foots.dec_mash_step_type_onehot, foots.dec_mash_step_times, foots.dec_mash_step_avg_temps = torch.split(
      foots.x_hat_mash_steps, [enc_mash_step_type_onehot_size, num_mash_steps, num_mash_steps], dim=1
    )

    # Grain slots
    num_grain_slots = self.args.num_grain_slots
    grain_type_embed_size = self.args.grain_type_embed_size
    enc_grain_type_embed_size = num_grain_slots * grain_type_embed_size
    foots.dec_grain_type_embed, foots.dec_grain_amts = torch.split(foots.x_hat_grains, [enc_grain_type_embed_size, num_grain_slots], dim=1)
    foots.dec_grain_type_logits = self.grain_type_decoder(foots.dec_grain_type_embed.view(-1, num_grain_slots, grain_type_embed_size)) # (B, num_grain_slots, num_grain_types)

    # Adjunct slots
    num_adjunct_slots = self.args.num_adjunct_slots
    adjunct_type_embed_size = self.args.adjunct_type_embed_size
    enc_adjunct_type_embed_size = num_adjunct_slots * adjunct_type_embed_size
    dec_adjunct_type_embed, foots.dec_adjunct_amts = torch.split(foots.x_hat_adjuncts, [enc_adjunct_type_embed_size, num_adjunct_slots], dim=1)
    foots.dec_adjunct_type_logits = self.adjunct_type_decoder(dec_adjunct_type_embed.view(-1, num_adjunct_slots, adjunct_type_embed_size)) # (B, num_adjunct_slots, num_adjunct_types)
    
    # Hop slots
    num_hop_slots = self.args.num_hop_slots
    hop_type_embed_size = self.args.hop_type_embed_size
    enc_hop_type_embed_size = num_hop_slots * hop_type_embed_size
    enc_hop_stage_type_onehot_size = num_hop_slots * self.args.num_hop_stage_types
    dec_hop_type_embed, foots.dec_hop_stage_type_onehot, foots.dec_hop_times, foots.dec_hop_concentrations = torch.split(
      foots.x_hat_hops, [enc_hop_type_embed_size, enc_hop_stage_type_onehot_size, num_hop_slots, num_hop_slots], dim=1
    )
    foots.dec_hop_type_logits = self.hop_type_decoder(dec_hop_type_embed.view(-1, num_hop_slots, hop_type_embed_size)) # (B, num_hop_slots, num_hop_types)
    
    # Miscellaneous slots
    num_misc_slots = self.args.num_misc_slots
    misc_type_embed_size = self.args.misc_type_embed_size
    enc_misc_type_embed_size = num_misc_slots * misc_type_embed_size
    enc_misc_stage_type_onehot_size = num_misc_slots * self.args.num_misc_stage_types
    dec_misc_type_embed, foots.dec_misc_stage_type_onehot, foots.dec_misc_times, foots.dec_misc_amts = torch.split(
      foots.x_hat_miscs, [enc_misc_type_embed_size, enc_misc_stage_type_onehot_size, num_misc_slots, num_misc_slots], dim=1
    )
    foots.dec_misc_type_logits = self.misc_type_decoder(dec_misc_type_embed.view(-1, num_misc_slots, misc_type_embed_size)) # (B, num_misc_slots, num_misc_types)
    
    # Microorganism slots
    num_mo_slots = self.args.num_microorganism_slots
    mo_type_embed_size = self.args.microorganism_type_embed_size
    enc_mo_type_embed_size = num_mo_slots * mo_type_embed_size
    enc_mo_stage_type_onehot_size = num_mo_slots * self.args.num_mo_stage_types
    dec_mo_type_embed, foots.dec_mo_stage_type_onehot = torch.split(
      foots.x_hat_mos, [enc_mo_type_embed_size, enc_mo_stage_type_onehot_size], dim=1
    )
    foots.dec_mo_type_logits = self.microorganism_type_decoder(dec_mo_type_embed.view(-1, num_mo_slots, mo_type_embed_size)) # (B, num_mo_slots, num_mo_types)
    
    return foots

MODEL_FILE_KEY_GLOBAL_STEP = "global_step"
MODEL_FILE_KEY_NETWORK     = "recipe_net" 
MODEL_FILE_KEY_OPTIMIZER   = "optimizer"
MODEL_FILE_KEY_NET_TYPE    = "net_type"
MODEL_FILE_KEY_SCHEDULER   = "scheduler"
MODEL_FILE_KEY_ARGS        = "args"

class RecipeNet(nn.Module):

  def __init__(self, args) -> None:
    super().__init__()
    
    hidden_layers = args.hidden_layers
    z_size = args.z_size
    activation_fn = args.activation_fn
    gain = args.gain
    
    assert all([num_hidden > 0 for num_hidden in hidden_layers])
    assert args.num_inputs >= 1
    assert len(hidden_layers) >= 1
    assert z_size >= 1 and z_size < args.num_inputs

    # Encoder and decoder networks
    self.encoder = nn.Sequential()
    self.encoder.append(layer_init_xavier(nn.Linear(args.num_inputs, hidden_layers[0]), gain))
    self.encoder.append(activation_fn(**args.activation_fn_params))
    prev_hidden_size = hidden_layers[0]
    for hidden_size in hidden_layers[1:]:
      self.encoder.append(layer_init_xavier(nn.Linear(prev_hidden_size, hidden_size), gain))
      self.encoder.append(activation_fn(**args.activation_fn_params))
      prev_hidden_size = hidden_size
    
    self.encoder.append(layer_init_xavier(nn.Linear(prev_hidden_size, z_size*2, bias=False), gain)) # TODO: bias=False if using batchnorm
    #self.encoder.append(activation_fn(**args.activation_fn_params)) # NOTE: This is determinental to convergence.
    self.encoder.append(nn.BatchNorm1d(z_size*2))

    self.decoder = nn.Sequential()
    self.decoder.append(layer_init_xavier(nn.Linear(z_size, hidden_layers[-1]), gain))
    self.decoder.append(activation_fn(**args.activation_fn_params))
    prev_hidden_size = hidden_layers[-1]
    for hidden_size in reversed(hidden_layers[:-1]):
      self.decoder.append(layer_init_xavier(nn.Linear(prev_hidden_size, hidden_size), gain))
      self.decoder.append(activation_fn(**args.activation_fn_params))
      prev_hidden_size = hidden_size
    self.decoder.append(layer_init_xavier(nn.Linear(hidden_layers[0], args.num_inputs), gain))
    #self.decoder.append(activation_fn(**args.activation_fn_params)) # NOTE: This is determental to convergence.
    
    # Pre-net Encoder (Network 'Heads')
    self.head_encoder = RecipeNetHeadEncoder(args)
    # Post-net Decoder (Network 'Foots')
    self.foot_decoder = RecipeNetFootDecoder(args)

    self.gamma = args.beta_vae_gamma
    self.C_stop_iter = args.beta_vae_C_stop_iter
    self.C_max = torch.Tensor([args.max_beta_vae_capacity])
    
    self.args = args
  
  def _apply(self, fn):
    super()._apply(fn)
    self.C_max = fn(self.C_max)
    return self

  def encode(self, input):
    # Start by breaking the given x apart into all the various heads/embeddings 
    # and concatenate them into a value that can be fed to the encoder network
    x, heads = self.head_encoder(input)
    # Encode to the latent distribution mean and std dev.
    mean, logvar = torch.chunk(self.encoder(x), 2, dim=-1) 
    return heads, mean, logvar
  
  def decode(self, z: torch.Tensor):
    # Decode to the flattened output
    x_hat = self.decoder(z)
    # We need to perform the reverse process on the output from the decoder network:
    # Break apart the output into matching segments similar to the heads (foots!) for use in later loss calculations
    foots = self.foot_decoder(x_hat)
    return foots
    
  def forward(self, input, use_mean=False):
    heads, mean, logvar = self.encode(input)
    # Sample (reparameterize trick) the final latent vector (z)
    z = mean if use_mean else reparameterize(mean, logvar)
    foots = self.decode(z)

    return heads, foots, mean, logvar, z
  
  def reconstruction_loss(self, input, heads, foots, reduction='sum'):
    if reduction == 'none':
      loss_wrap = lambda x: x.sum(dim=1)
    else:
      loss_wrap = lambda x: x

    # TODO: Simplify all this stuff into fewer losses: 
    # Group together all BCELogit and MSE losses into singluar tensors in both x and x_hat
    loss_toplvl = loss_wrap(F.mse_loss(foots.x_hat_toplvl, heads.x_toplvl, reduction=reduction))
    loss_mash_steps = loss_wrap(F.binary_cross_entropy_with_logits(foots.dec_mash_step_type_onehot, heads.enc_mash_step_type_onehot, reduction=reduction)) + \
      loss_wrap(F.mse_loss(foots.dec_mash_step_times, input['mash_step_times'], reduction=reduction)) + \
      loss_wrap(F.mse_loss(foots.dec_mash_step_avg_temps, input['mash_step_avg_temps'], reduction=reduction))
    loss_ferment_stages = loss_wrap(F.mse_loss(foots.x_hat_ferment_stages, heads.x_ferment_stages, reduction=reduction))
    loss_grains = loss_wrap(F.binary_cross_entropy_with_logits(foots.dec_grain_type_logits.flatten(1), heads.enc_grain_type_onehot.flatten(1), reduction=reduction)) + \
      loss_wrap(F.mse_loss(foots.dec_grain_amts, input['grain_amts'], reduction=reduction))
    loss_adjuncts = loss_wrap(F.binary_cross_entropy_with_logits(foots.dec_adjunct_type_logits.flatten(1), heads.enc_adjunct_type_onehot.flatten(1), reduction=reduction)) + \
      loss_wrap(F.mse_loss(foots.dec_adjunct_amts, input['adjunct_amts'], reduction=reduction))
    loss_hops = loss_wrap(F.binary_cross_entropy_with_logits(foots.dec_hop_type_logits.flatten(1), heads.enc_hop_type_onehot.flatten(1), reduction=reduction)) + \
      loss_wrap(F.binary_cross_entropy_with_logits(foots.dec_hop_stage_type_onehot, heads.enc_hop_stage_type_onehot, reduction=reduction)) + \
      loss_wrap(F.mse_loss(foots.dec_hop_times, input['hop_times'], reduction=reduction)) + \
      loss_wrap(F.mse_loss(foots.dec_hop_concentrations, input['hop_concentrations'], reduction=reduction))
    loss_miscs = loss_wrap(F.binary_cross_entropy_with_logits(foots.dec_misc_type_logits.flatten(1), heads.enc_misc_type_onehot.flatten(1), reduction=reduction)) + \
      loss_wrap(F.binary_cross_entropy_with_logits(foots.dec_misc_stage_type_onehot, heads.enc_misc_stage_type_onehot, reduction=reduction)) + \
      loss_wrap(F.mse_loss(foots.dec_misc_times, input['misc_times'], reduction=reduction)) + \
      loss_wrap(F.mse_loss(foots.dec_misc_amts, input['misc_amts'], reduction=reduction))
    loss_mos = loss_wrap(F.binary_cross_entropy_with_logits(foots.dec_mo_type_logits.flatten(1), heads.enc_mo_type_onehot.flatten(1), reduction=reduction)) + \
      loss_wrap(F.binary_cross_entropy_with_logits(foots.dec_mo_stage_type_onehot, heads.enc_mo_stage_type_onehot, reduction=reduction))

    # Add up all our losses for reconstruction of the recipe
    return loss_toplvl + loss_mash_steps + loss_ferment_stages + loss_grains + loss_adjuncts + loss_hops + loss_miscs + loss_mos





class BetaVAELoss():
  def __init__(self, args: RecipeNetArgs) -> None:
    self.gamma = args.beta_vae_gamma
    self.c_stop_iter = args.beta_vae_C_stop_iter
    self.c_max = torch.Tensor([args.max_beta_vae_capacity])

  def calc_loss(self, **kwargs):
    reconst_loss = kwargs['reconst_loss']
    mean = kwargs['mean']
    logvar = kwargs['logvar']
    iter_num = kwargs.get('iter_num', -1)
    kl_weight = kwargs.get('kl_weight', 1)

    # Beta-VAE KL calculation is based on https://arxiv.org/pdf/1804.03599.pdf
    kl_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mean ** 2 - logvar.exp(), dim=1), dim=0)
    if iter_num >= 0:
      C = torch.clamp(self.c_max/self.c_stop_iter * iter_num, 0, self.c_max.data[0])
    else:
      C = self.c_max.data[0]
    loss = reconst_loss + kl_weight * self.gamma * (kl_loss - C).abs()
    return {'loss': loss, 'reconst_loss': reconst_loss, 'kl_loss': kl_loss, 'C': C}


class BetaTCVAELoss():
  def __init__(self, args: RecipeNetArgs) -> None:
    self.alpha = args.beta_tc_vae_alpha
    self.beta  = args.beta_tc_vae_beta
    self.gamma = args.beta_tc_vae_gamma
    self.anneal_steps = args.beta_tc_vae_gamma

  def log_density_gaussian(self, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    Computes the log pdf of the Gaussian with parameters mu and logvar at x
    :param x: (Tensor) Point at whichGaussian PDF is to be evaluated
    :param mu: (Tensor) Mean of the Gaussian distribution
    :param logvar: (Tensor) Log variance of the Gaussian distribution
    :return: The Gaussian PDF result at x.
    """
    norm = -0.5 * (np.log(2 * np.pi) + logvar)
    log_density = norm - 0.5 * ((x - mu) ** 2 * torch.exp(-logvar))
    return log_density

  def calc_loss(self, **kwargs):
    reconst_loss = kwargs['reconst_loss']
    mean = kwargs['mean']
    logvar = kwargs['logvar']
    z = kwargs['z']
    dataset_size = kwargs['dataset_size']
    iter_num = kwargs.get('iter_num', -1)
    kl_weight = kwargs.get('kl_weight', 1)
    
    log_q_zx = self.log_density_gaussian(z, mean, logvar).sum(dim = 1)

    zeros = torch.zeros_like(z)
    log_p_z = self.log_density_gaussian(z, zeros, zeros).sum(dim = 1)

    batch_size, latent_dim = z.shape
    mat_log_q_z = self.log_density_gaussian(
      z.view(batch_size, 1, latent_dim), 
      mean.view(1, batch_size, latent_dim), 
      logvar.view(1, batch_size, latent_dim)
    )

    # References
    # https://github.com/AntixK/PyTorch-VAE/blob/master/models/betatc_vae.py
    # https://github.com/YannDubs/disentangling-vae
    strat_weight = (dataset_size - batch_size + 1) / (dataset_size * (batch_size - 1))
    importance_weights = torch.Tensor(batch_size, batch_size).fill_(1 / (batch_size -1)).to(z.device)
    importance_weights.view(-1)[::batch_size] = 1 / dataset_size
    importance_weights.view(-1)[1::batch_size] = strat_weight
    importance_weights[batch_size - 2, 0] = strat_weight
    log_importance_weights = importance_weights.log()

    mat_log_q_z += log_importance_weights.view(batch_size, batch_size, 1)

    log_q_z = torch.logsumexp(mat_log_q_z.sum(2), dim=1, keepdim=False)
    log_prod_q_z = torch.logsumexp(mat_log_q_z, dim=1, keepdim=False).sum(1)

    idx_loss = (log_q_zx - log_q_z).mean()
    tc_loss  = (log_q_z - log_prod_q_z).mean()
    kld_loss = (log_prod_q_z - log_p_z).mean()

    if iter_num >= 0:
      anneal_rate = min(0 + 1 * iter_num / self.anneal_steps, 1.0)
    else:
      anneal_rate = 1.0

    loss = reconst_loss / batch_size + kl_weight * (self.alpha * idx_loss + self.beta * tc_loss + anneal_rate * self.gamma * kld_loss)
    
    return {
      'loss': loss, 
      'reconst_loss': reconst_loss,
      'idx_loss': idx_loss, 
      'tc_loss': tc_loss,
      'kld_loss': kld_loss
    }

