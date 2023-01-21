
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

#from quantize import VectorQuantizer
from recipe_net_args import RecipeNetArgs, FEATURE_TYPE_CATEGORY

def layer_init_ortho(layer, std=np.sqrt(2)):
  nn.init.orthogonal_(layer.weight, std)
  if layer.bias != None:
    nn.init.constant_(layer.bias, 0.0)
  return layer

def layer_init_xavier(layer, gain, bias=0.0):
  nn.init.xavier_normal_(layer.weight, gain)
  if layer.bias != None:
    nn.init.constant_(layer.bias, bias)
  return layer

def reparameterize(mu, logvar):
  std = torch.exp(0.5 * logvar)
  assert not torch.isnan(std).any()
  eps = torch.randn_like(std)
  return eps * std + mu

  # Old encoder
  '''
    heads = RecipeNetData()
    # Simple top-level heads (high-level recipe parameters)
    heads.x_toplvl = torch.cat((x['boil_time'].unsqueeze(1), x['mash_ph'].unsqueeze(1), x['sparge_temp'].unsqueeze(1)), dim=1) # (B, 3)
    
    # Mash step heads
    # NOTE: Data shape is (B, S=number_of_mash_steps) for the
    # following recipe tensors: {'mash_step_type_inds', 'mash_step_times', 'mash_step_avg_temps'}
    num_mash_step_types = self.args.num_mash_step_types
    heads.enc_mash_step_type_embed  = self.mash_step_type_embedding(x['mash_step_type_inds']).flatten(1) # # (B, S, num_mash_step_types) -> (B, S*num_mash_step_types)
    heads.enc_mash_step_type_onehot = F.one_hot(x['mash_step_type_inds'].long(), num_mash_step_types).float()
    heads.x_mash_steps = torch.cat((heads.enc_mash_step_type_embed, x['mash_step_times'], x['mash_step_avg_temps']), dim=1) # (B, num_mash_step_types*S+S+S) = [B, 36=(24+6+6)]
    
    # Ferment stage heads
    # NOTE: Data shape is (B, S=2) for the following recipe tensors: {'ferment_stage_times', 'ferment_stage_temps'}
    heads.x_ferment_stages = torch.cat((x['ferment_stage_times'], x['ferment_stage_temps']), dim=1) # (B, S+S)

    # Grain (malt bill) heads
    # NOTE: Data shape is (B, S=num_grain_slots) for the following recipe tensors: {'grain_core_type_inds', 'grain_amts'}
    num_grain_types = self.args.num_grain_types
    heads.enc_grain_type_embed  = self.grain_type_embedding(x['grain_core_type_inds']).flatten(1) # (B, S, grain_type_embed_size) -> (B, S*grain_type_embed_size)
    heads.enc_grain_type_onehot = F.one_hot(x['grain_core_type_inds'].long(), num_grain_types).float() # (B, num_grain_slots, num_grain_types)
    heads.x_grains = torch.cat((heads.enc_grain_type_embed, x['grain_amts']), dim=1) # (B, S*grain_type_embed_size+S)
    
    # Adjunct heads
    # NOTE: Data shape is (B, S=num_adjunct_slots) for the following recipe tensors: {'adjunct_core_type_inds', 'adjunct_amts'}
    num_adjunct_types = self.args.num_adjunct_types
    heads.enc_adjunct_type_embed  = self.adjunct_type_embedding(x['adjunct_core_type_inds']).flatten(1) # (B, S, adjunct_type_embed_size) -> (B, S*adjunct_type_embed_size)
    heads.enc_adjunct_type_onehot = F.one_hot(x['adjunct_core_type_inds'].long(), num_adjunct_types).float() # (B, num_adjunct_slots, num_adjunct_types)
    heads.x_adjuncts = torch.cat((heads.enc_adjunct_type_embed, x['adjunct_amts']), dim=1) # (B, S*adjunct_type_embed_size+S)
    
    # Hop heads
    # NOTE: Data shape is (B, S=num_hop_slots) for the following recipe tensors: 
    # {'hop_type_inds', 'hop_stage_type_inds', 'hop_times', 'hop_concentrations'}
    num_hop_types = self.args.num_hop_types
    num_hop_stage_types = self.args.num_hop_stage_types
    heads.enc_hop_type_embed  = self.hop_type_embedding(x['hop_type_inds']).flatten(1) # (B, S, hop_type_embed_size)
    heads.enc_hop_type_onehot = F.one_hot(x['hop_type_inds'].long(), num_hop_types).float() # (B, num_hop_slots, num_hop_types)
    heads.enc_hop_stage_type_embed = self.hop_stage_type_embedding(x['hop_stage_type_inds']).flatten(1) # (B, S, num_hop_stage_types) -> (B, S*num_hop_stage_types)
    heads.enc_hop_stage_type_onehot = F.one_hot(x['hop_stage_type_inds'].long(), num_hop_stage_types).float()
    heads.x_hops = torch.cat((heads.enc_hop_type_embed, heads.enc_hop_stage_type_embed, x['hop_times'], x['hop_concentrations']), dim=1) # (B, S*hop_type_embed_size + S*num_hop_stage_types + S + S)
    
    # Misc. heads
    # NOTE: Data shape is (B, S=num_misc_slots) for the following recipe tensors:
    # {'misc_type_inds', 'misc_stage_inds', 'misc_times', 'misc_amts'}
    num_misc_types = self.args.num_misc_types
    num_misc_stage_types = self.args.num_misc_stage_types
    heads.enc_misc_type_embed  = self.misc_type_embedding(x['misc_type_inds']).flatten(1) # (B, S, misc_type_embed_size)
    heads.enc_misc_type_onehot = F.one_hot(x['misc_type_inds'].long(), num_misc_types).float() # (B, num_misc_slots, num_misc_types)
    heads.enc_misc_stage_type_embed  = self.misc_stage_type_embedding(x['misc_stage_inds']).flatten(1) # (B, S, num_misc_stage_types) -> (B, S*num_misc_stage_types)
    heads.enc_misc_stage_type_onehot = F.one_hot(x['misc_stage_inds'].long(), num_misc_stage_types).float()
    heads.x_miscs = torch.cat((heads.enc_misc_type_embed, heads.enc_misc_stage_type_embed, x['misc_times'], x['misc_amts']), dim=1) # (B, S*misc_type_embed_size + S*num_misc_stage_types + S + S)
    
    # Microorganism heads
    # NOTE: Data shape is (B, S=num_microorganism_slots) for the following recipe tensors:
    # {'mo_type_inds', 'mo_stage_inds'}
    num_mo_types = self.args.num_microorganism_types
    num_mo_stage_types = self.args.num_mo_stage_types
    heads.enc_mo_type_embed  = self.microorganism_type_embedding(x['mo_type_inds']).flatten(1) # (B, S, microorganism_type_embed_size)
    heads.enc_mo_type_onehot = F.one_hot(x['mo_type_inds'].long(), num_mo_types).float() # (B, S, num_mo_types)
    heads.enc_mo_stage_type_embed  = self.mo_stage_type_embedding(x['mo_stage_inds']).flatten(1) # (B, S, num_mo_stage_types)
    heads.enc_mo_stage_type_onehot = F.one_hot(x['mo_stage_inds'].long(), num_mo_stage_types).float()
    heads.x_mos = torch.cat((heads.enc_mo_type_embed, heads.enc_mo_stage_type_embed), dim=1) # (B, S*microorganism_type_embed_size + S*num_mo_stage_types)
    
    # Put all the recipe data together into a flattened tensor
    x = torch.cat((heads.x_toplvl, heads.x_mash_steps, heads.x_ferment_stages, heads.x_grains, heads.x_adjuncts, heads.x_hops, heads.x_miscs, heads.x_mos), dim=1) # (B, num_inputs)
    return x, heads
    '''

  # Old decoder
  '''
    foots = RecipeNetData()
    
    foots.x_hat_toplvl, foots.x_hat_mash_steps, foots.x_hat_ferment_stages, foots.x_hat_grains, foots.x_hat_adjuncts, foots.x_hat_hops, foots.x_hat_miscs, foots.x_hat_mos = torch.split(x_hat, self.split_sizes, dim=1)

    # Mash steps
    num_mash_steps = self.args.num_mash_steps
    num_mash_step_types = self.args.num_mash_step_types
    enc_mash_step_type_embed_size = num_mash_steps * num_mash_step_types
    dec_mash_step_type_embed, foots.dec_mash_step_times, foots.dec_mash_step_avg_temps = torch.split(
      foots.x_hat_mash_steps, [enc_mash_step_type_embed_size, num_mash_steps, num_mash_steps], dim=1
    )
    foots.dec_mash_step_type_logits = self.mash_step_type_decoder(dec_mash_step_type_embed.view(-1, num_mash_steps, num_mash_step_types)) # (B, num_mash_steps, num_mash_step_types)

    # Grain slots
    num_grain_slots = self.args.num_grain_slots
    grain_type_embed_size = self.args.grain_type_embed_size
    enc_grain_type_embed_size = num_grain_slots * grain_type_embed_size
    dec_grain_type_embed, foots.dec_grain_amts = torch.split(foots.x_hat_grains, [enc_grain_type_embed_size, num_grain_slots], dim=1)
    foots.dec_grain_type_logits = self.grain_type_decoder(dec_grain_type_embed.view(-1, num_grain_slots, grain_type_embed_size)) # (B, num_grain_slots, num_grain_types)

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
    num_hop_stage_types = self.args.num_hop_stage_types
    enc_hop_stage_type_embed_size = num_hop_slots * num_hop_stage_types
    dec_hop_type_embed, dec_hop_stage_type_embed, foots.dec_hop_times, foots.dec_hop_concentrations = torch.split(
      foots.x_hat_hops, [enc_hop_type_embed_size, enc_hop_stage_type_embed_size, num_hop_slots, num_hop_slots], dim=1
    )
    foots.dec_hop_type_logits = self.hop_type_decoder(dec_hop_type_embed.view(-1, num_hop_slots, hop_type_embed_size)) # (B, num_hop_slots, num_hop_types)
    foots.dec_hop_stage_type_logits = self.hop_stage_type_decoder(dec_hop_stage_type_embed.view(-1, num_hop_slots, num_hop_stage_types))
    
    # Miscellaneous slots
    num_misc_slots = self.args.num_misc_slots
    misc_type_embed_size = self.args.misc_type_embed_size
    enc_misc_type_embed_size = num_misc_slots * misc_type_embed_size
    num_misc_stage_types = self.args.num_misc_stage_types
    enc_misc_stage_type_embed_size = num_misc_slots * num_misc_stage_types
    dec_misc_type_embed, dec_misc_stage_type_embed, foots.dec_misc_times, foots.dec_misc_amts = torch.split(
      foots.x_hat_miscs, [enc_misc_type_embed_size, enc_misc_stage_type_embed_size, num_misc_slots, num_misc_slots], dim=1
    )
    foots.dec_misc_type_logits = self.misc_type_decoder(dec_misc_type_embed.view(-1, num_misc_slots, misc_type_embed_size)) # (B, num_misc_slots, num_misc_types)
    foots.dec_misc_stage_type_logits = self.misc_stage_type_decoder(dec_misc_stage_type_embed.view(-1, num_misc_slots, num_misc_stage_types))
    
    # Microorganism slots
    num_mo_slots = self.args.num_microorganism_slots
    mo_type_embed_size = self.args.microorganism_type_embed_size
    enc_mo_type_embed_size = num_mo_slots * mo_type_embed_size
    num_mo_stage_types = self.args.num_mo_stage_types
    enc_mo_stage_type_embed_size = num_mo_slots * num_mo_stage_types
    dec_mo_type_embed, dec_mo_stage_type_embed = torch.split(foots.x_hat_mos, [enc_mo_type_embed_size, enc_mo_stage_type_embed_size], dim=1)
    foots.dec_mo_type_logits = self.microorganism_type_decoder(dec_mo_type_embed.view(-1, num_mo_slots, mo_type_embed_size)) # (B, num_mo_slots, num_mo_types)
    foots.dec_mo_stage_type_logits = self.microorganism_stage_type_decoder(dec_mo_stage_type_embed.view(-1, num_mo_slots, num_mo_stage_types))
    
    return foots
    '''

MODEL_FILE_KEY_GLOBAL_STEP = "global_step"
MODEL_FILE_KEY_NETWORK     = "recipe_net" 
MODEL_FILE_KEY_OPTIMIZER   = "optimizer"
MODEL_FILE_KEY_NET_TYPE    = "net_type"
MODEL_FILE_KEY_SCHEDULER   = "scheduler"
MODEL_FILE_KEY_ARGS        = "args"

class RecipeEncoderNet(nn.Module):
  def __init__(self, args: RecipeNetArgs) -> None:
    super().__init__()
    use_batch_norm = args.use_batch_norm
    use_layer_norm = args.use_layer_norm
    fc_prebn_bias = not use_batch_norm
    activation_fn = args.activation_fn
    gain = args.gain
    hidden_layers = args.hidden_layers
    z_size = args.z_size

    # Encoder and decoder networks
    self.encoder = nn.Sequential()
    self.encoder.append(layer_init_xavier(nn.Linear(args.num_inputs, hidden_layers[0], bias=fc_prebn_bias), gain))
    if use_layer_norm:
      self.encoder.append(nn.LayerNorm(hidden_layers[0]))
    self.encoder.append(activation_fn(**args.activation_fn_params))
    if use_batch_norm:
      self.encoder.append(nn.BatchNorm1d(hidden_layers[0]))

    prev_hidden_size = hidden_layers[0]
    for hidden_size in hidden_layers[1:]:
      self.encoder.append(layer_init_xavier(nn.Linear(prev_hidden_size, hidden_size, bias=fc_prebn_bias), gain))
      if use_layer_norm:
        self.encoder.append(nn.LayerNorm(prev_hidden_size))
      self.encoder.append(activation_fn(**args.activation_fn_params))
      if use_batch_norm:
        self.encoder.append(nn.BatchNorm1d(hidden_size))
      prev_hidden_size = hidden_size

    self.encode_mean   = layer_init_xavier(nn.Linear(prev_hidden_size, z_size), gain)
    self.encode_logvar = layer_init_xavier(nn.Linear(prev_hidden_size, z_size), gain)

    # Build the encoder's category embedding weights
    embed_dict = {}
    for key, (feature_type, feature_size, _, embed_size) in args.features.items():
      if feature_type == FEATURE_TYPE_CATEGORY:
        assert embed_size != None and embed_size > 0, f"Invalid embedding size found for value {key}"
        embed_dict[key] = nn.Embedding(feature_size, embed_size, max_norm=1.0)
    self.embedding_dict = nn.ModuleDict(embed_dict)
    self.args = args


  def forward(self, input: dict):
    # Start by breaking the given x apart into all the various heads/embeddings 
    # and concatenate them into a value that can be fed to the encoder network
    input_list = []
    for key, (feature_type, feature_size, num_slots, _) in self.args.features.items():
      feature_x = input[key]
      if feature_type == FEATURE_TYPE_CATEGORY:
        # Category type: All categorical features are embedded
        feat_x_embedded = self.embedding_dict[key](feature_x.long()).flatten(1) # (B, feature_size, num_slots) -> (B, feature_size*num_slots)
        input_list.append(feat_x_embedded)
      else:
        # Real type
        assert feature_size == 1, "All real number features should have a size of 1"
        assert num_slots > 0, f"Invalid number of slots defined for feature {key}"
        if num_slots == 1:
          input_list.append(feature_x.view(-1,1))
        else:
          input_list.append(feature_x)

    x = torch.cat(input_list, dim=1)
    x = self.encoder(x)

    # Encode to the latent distribution mean and std dev.
    mean   = self.encode_mean(x)
    logvar = self.encode_logvar(x)

    return x, mean, logvar

  def z(self, input, use_mean=False):
    _, mean, logvar = self.forward(input)
    # Sample (reparameterize trick) the final latent vector (z)
    return mean if use_mean else reparameterize(mean, logvar)


class RecipeDecoderNet(nn.Module):
  def __init__(self, args: RecipeNetArgs) -> None:
    super().__init__()
    use_batch_norm = args.use_batch_norm
    use_layer_norm = args.use_layer_norm
    fc_prebn_bias = not use_batch_norm
    hidden_layers = args.hidden_layers
    z_size = args.z_size
    activation_fn = args.activation_fn
    gain = args.gain

    self.decoder = nn.Sequential()
    self.decoder.append(layer_init_xavier(nn.Linear(z_size, hidden_layers[-1], bias=fc_prebn_bias), gain))

    if use_layer_norm:
      self.decoder.append(nn.LayerNorm(hidden_layers[-1]))
    self.decoder.append(activation_fn(**args.activation_fn_params))
    if use_batch_norm:
      self.decoder.append(nn.BatchNorm1d(hidden_layers[-1]))

    prev_hidden_size = hidden_layers[-1]
    for hidden_size in reversed(hidden_layers[:-1]):
      self.decoder.append(layer_init_xavier(nn.Linear(prev_hidden_size, hidden_size, bias=fc_prebn_bias), gain))
      if use_layer_norm:
        self.decoder.append(nn.LayerNorm(hidden_size))
      self.decoder.append(activation_fn(**args.activation_fn_params))
      if use_batch_norm:
        self.decoder.append(nn.BatchNorm1d(hidden_size))
      prev_hidden_size = hidden_size
    assert prev_hidden_size == hidden_layers[0]
    #self.decoder.append(layer_init_xavier(nn.Linear(hidden_layers[0], args.num_inputs), gain))
    #self.decoder.append(activation_fn(**args.activation_fn_params)) # NOTE: This is determental to convergence.
    
    # Build the decoder's fully-connected feature decoder heads
    decode_dict = {}
    for key, (feature_type, feature_size, num_slots, embed_size) in args.features.items():
      if feature_type == FEATURE_TYPE_CATEGORY:
        assert embed_size != None and embed_size > 0, f"Invalid embedding size found for value {key}"
        decode_fcs = []
        for _ in range(num_slots):
          decode_fcs.append(layer_init_xavier(nn.Linear(prev_hidden_size, feature_size), args.gain))
        decode_dict[key] = nn.ModuleList(decode_fcs)
      else:
        assert feature_size == 1, "All real number features should have a size of 1"
        decode_dict[key] = layer_init_xavier(nn.Linear(prev_hidden_size, num_slots), args.gain)

    self.decoder_fc_dict = nn.ModuleDict(decode_dict)
    self.log_softmax = nn.LogSoftmax(dim=1)
    self.args = args


  def forward(self, z: torch.Tensor):
    # Decode to the flattened output just before all the decoder/output heads
    x_hat = self.decoder(z)

    # We need to perform the reverse process on the output from the decoder network:
    # The decoded tensor (x_hat) is flat with a shape of (B, num_inputs), we'll need to break it apart
    # so that we can eventually calculate losses appropriately for each head of original data fed to the encoder
    x_hat_dict = {}
    for key, (feature_type, _, num_slots, _) in self.args.features.items():
      if feature_type == FEATURE_TYPE_CATEGORY:
        # Category type: There should be a list of FC layers to produce the logits used to determine
        # each slot's category
        feature_fcs = self.decoder_fc_dict[key]
        assert isinstance(feature_fcs, nn.ModuleList), "Categorical FCs should be a ModuleList, one FC for each slot"
        output_cat_list = []
        for i in range(num_slots):
          output_cat_list.append(self.log_softmax(feature_fcs[i](x_hat)))
        x_hat_dict[key] = output_cat_list
      else:
        # Real type: Just grab the appropriate range of indices from x_hat for all slots
        feature_fc = self.decoder_fc_dict[key]
        x_hat_dict[key] = feature_fc(x_hat)

    return x_hat_dict
    

class RecipeNet(nn.Module):

  def __init__(self, args) -> None:
    super().__init__()

    assert not args.use_batch_norm or not args.use_layer_norm, "You shouldn't be using both layer and batch normalization simultaneously."
    assert all([num_hidden > 0 for num_hidden in args.hidden_layers])
    assert args.num_inputs >= 1
    assert len(args.hidden_layers) >= 1
    assert args.z_size >= 1 and args.z_size < args.num_inputs

    # Encoder and decoder networks
    self.encoder = RecipeEncoderNet(args)
    self.decoder = RecipeDecoderNet(args)
    
    # Log variance of the decoder for real attributes
    #self.register_buffer('logvar_x', torch.zeros(1, args.num_real_features(), dtype=torch.float32))
    self.logvar_x = nn.Parameter(torch.zeros(1, args.num_real_features(), dtype=torch.float32))

    self.args = args
  
  def forward(self, input:dict, use_mean=False):
    # Encode to mean and log variance as well as the recipe tensor representation
    x, mean_z, logvar_z = self.encoder(input)
    # Sample (reparameterization trick) the latent vector (z)
    z = mean_z if use_mean else reparameterize(mean_z, logvar_z)
    # Decode back into recipe representation
    x_hat_dict = self.decoder(z)

    logvar_x = self.logvar_x.clamp(-3,3)

    return x, x_hat_dict, mean_z, logvar_z, logvar_x, z

  @staticmethod
  def softclip(tensor, min):
    return min + F.softplus(tensor - min)

  @staticmethod
  def nll_gauss(gauss_params, input_val_feat, logvar_x):
    #return F.mse_loss(input_val_feat, gauss_params, reduction="none")
    logvar_r = (logvar_x.exp() + 1e-9).log()
    #data_compnt = 0.5 * torch.pow((input_val_feat - gauss_params) / logvar_r.exp(), 2) + logvar_r + 0.5 * np.log(2*np.pi)
    data_compnt = 0.5*logvar_r + (input_val_feat - gauss_params)**2 / (2.* logvar_r.exp() + 1e-9)
    return data_compnt

  @staticmethod
  def nll_category(categ_logp_feat, input_idx_feat):
    return F.nll_loss(categ_logp_feat, input_idx_feat, reduction='none').view(-1,1)
    
  def calc_loss(self, input: dict, x_hat_dict: dict, mean_z: torch.Tensor, logvar_z: torch.Tensor, logvar_x: torch.Tensor) -> dict:
    reconst_loss = torch.zeros(1, dtype=torch.float32).to(logvar_x.device)
    
    real_feat_idx = 0
    # Perform the appropriate calculation for each feature's reconstruction loss
    for key, (feature_type, _, num_slots, _) in self.args.features.items():
      #pi_feat = torch.sigmoid(pi_dict[key]).clamp(1e-6,1-1e-6)
      if feature_type == FEATURE_TYPE_CATEGORY:
        for i in range(num_slots):
          reconst_loss += RecipeNet.nll_category(
            x_hat_dict[key][i], input[key][:,i].long()
          ).sum()
      else:
        reconst_loss += RecipeNet.nll_gauss(
          x_hat_dict[key], input[key].view([-1] + list(x_hat_dict[key].shape[1:])), 
          logvar_x[:,real_feat_idx:real_feat_idx+num_slots]
        ).sum()
        real_feat_idx += 1

    # KL Divergence regularizer on the latent space
    kld_loss = -0.5 * torch.sum(1 + logvar_z - mean_z.pow(2) - logvar_z.exp())

    # NOTE: self.args.alpha_prior is the prior on clean cells (higher values means more likely to be clean)
    # kld regularized on the weights
    #pi_vec = torch.cat([p for p in pi_dict.values()], dim=-1)
    #pi_mtx = torch.sigmoid(pi_vec).clamp(1e-6, 1-1e-6)
    #w_kld_loss = torch.sum(pi_mtx * torch.log(pi_mtx / self.args.alpha_prior) + (1.0-pi_mtx) * torch.log((1.0-pi_mtx) / (1.0-self.args.alpha_prior)))

    loss = reconst_loss + kld_loss #+ w_kld_loss
    return {
      'loss': loss, 
      'reconst_loss': reconst_loss, 
      'kld_loss': kld_loss, 
      #'w_kld_loss': w_kld_loss,
    }


  def _get_pi_exact_category(self, categ_logp_feat:torch.Tensor, feat_size, prior_sig) -> torch.Tensor:
    input_dims = categ_logp_feat.shape
    categ_logp_robust = torch.log(torch.tensor(1.0 / feat_size, dtype=categ_logp_feat.dtype))
    categ_logp_robust = categ_logp_robust * torch.ones(input_dims)
    categ_logp_robust = categ_logp_robust.to(categ_logp_feat.device)
    with torch.no_grad():
      pi = torch.sigmoid(categ_logp_feat - categ_logp_robust + torch.log(prior_sig) - torch.log(1-prior_sig))
    return pi

  def _get_pi_exact_gauss(self, gauss_params:torch.Tensor, input_val_feat:torch.Tensor, logvar, prior_sig, std_0_scale=2.0) -> torch.Tensor:
    mu = gauss_params
    logvar_r = (logvar.exp() + 1e-9).log()
    mu_0 = 0.0
    var_0 = std_0_scale**2
    log_var_0 = np.log(var_0)
    data_compnt   = -(0.5*logvar_r + (input_val_feat - mu)**2 / (2.0* logvar_r.exp() + 1e-9))
    robust_compnt = -(0.5*log_var_0 + (input_val_feat - mu_0)**2 / (2.0 * var_0  + 1e-9))
    with torch.no_grad():
      pi = torch.sigmoid(data_compnt - robust_compnt + torch.log(prior_sig) - torch.log(1-prior_sig))
    return pi

  def get_pi_exact_vec(self, input:dict, x_hat_dict:dict, mean:torch.Tensor, logvar_x:torch.Tensor):
    def logit_fn(x): return (x+1e-9).log()-(1.0-x+1e-9).log()
    prior_sig = torch.tensor(self.args.alpha_prior, dtype=torch.float32).to(mean.device)
    pi_dict = {}
    logvar_x_idx = 0
    for key, (feature_type, feature_size, num_slots, _) in self.args.features.items():
      if feature_type == FEATURE_TYPE_CATEGORY:
        feature_slots = x_hat_dict[key]
        pi_feats = []
        for i in range(num_slots):
          pi_feat = self._get_pi_exact_category(feature_slots[i], feature_size, prior_sig)
          pi_feat = torch.clamp(pi_feat, 1e-6, 1-1e-6)
          pi_feats.append(logit_fn(pi_feat))
        pi_feats = torch.stack(pi_feats, dim=1)
        pi_feats = torch.gather(pi_feats, 2, input[key].unsqueeze(-1).long()).squeeze(-1)
        pi_dict[key] = pi_feats
      else:
        pi_feat = self._get_pi_exact_gauss(x_hat_dict[key], input[key], logvar_x[:,logvar_x_idx:logvar_x_idx+num_slots], prior_sig)
        pi_feat = torch.clamp(pi_feat, 1e-6, 1-1e-6)
        pi_dict[key] = logit_fn(pi_feat)
        logvar_x_idx += 1

    return pi_dict


class BetaVAELoss():
  def __init__(self, args: RecipeNetArgs, device: torch.device) -> None:
    self.gamma = args.beta_vae_gamma
    self.c_stop_iter = args.beta_vae_C_stop_iter
    self.c_max = torch.Tensor([args.max_beta_vae_capacity]).to(device)

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
    self.anneal_steps = args.beta_tc_vae_anneal_steps

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
    is_mss = kwargs.get('is_mss', False)
    
    log_q_zx = self.log_density_gaussian(z, mean, logvar).sum(dim=1)

    zeros = torch.zeros_like(z)
    log_p_z = self.log_density_gaussian(z, zeros, zeros).sum(dim=1)

    batch_size, latent_dim = z.shape
    mat_log_q_z = self.log_density_gaussian(
      z.view(batch_size, 1, latent_dim), 
      mean.view(1, batch_size, latent_dim), 
      logvar.view(1, batch_size, latent_dim)
    )

    # References
    # https://github.com/AntixK/PyTorch-VAE/blob/master/models/betatc_vae.py
    # https://github.com/YannDubs/disentangling-vae
    if is_mss:
      # Minibatch Stratified Sampling
      N = dataset_size
      M = batch_size-1
      strat_weight = (N - M) / (N * M)
      importance_weights = torch.Tensor(batch_size, batch_size).fill_(1 / M).to(z.device)
      importance_weights.view(-1)[::M + 1] = 1 / N
      importance_weights.view(-1)[1::M + 1] = strat_weight
      importance_weights[M - 1, 0] = strat_weight
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

    batch_reconst_loss = reconst_loss / batch_size
    loss = batch_reconst_loss + kl_weight * (self.alpha * idx_loss + self.beta * tc_loss + anneal_rate * self.gamma * kld_loss)
    
    return {
      'loss': loss,
      'batch_reconst_loss': batch_reconst_loss, 
      'idx_loss': idx_loss, 
      'tc_loss': tc_loss,
      'kld_loss': kld_loss
    }

