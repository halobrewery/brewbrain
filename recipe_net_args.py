import torch.nn as nn
from recipe_dataset import NUM_GRAIN_SLOTS, NUM_ADJUNCT_SLOTS, NUM_HOP_SLOTS, NUM_MISC_SLOTS, NUM_MICROORGANISM_SLOTS, NUM_FERMENT_STAGE_SLOTS, NUM_MASH_STEPS


def dataset_args(dataset):
  # NOTE: All types include a "None" (i.e., empty) category
  return {
    'num_grain_types':         len(dataset.core_grains_idx_to_dbid), # Number of (core) grain types (rows in the DB)
    'num_adjunct_types':       len(dataset.core_adjs_idx_to_dbid),   # Number of (core) adjunct types (rows in the DB)
    'num_hop_types':           len(dataset.hops_idx_to_dbid),        # Number of hop types (rows in the DB)
    'num_misc_types':          len(dataset.miscs_idx_to_dbid),       # Number of misc. types (rows in the DB)
    'num_microorganism_types': len(dataset.mos_idx_to_dbid),         # Number of microrganism types (rows in the DB)
    'num_mash_step_types':     len(dataset.mash_step_idx_to_name),   # Number of mash step types (e.g., Infusion, Decoction, Temperature)
    'num_hop_stage_types':     len(dataset.hop_stage_idx_to_name),   # Number of hop stage types (e.g., Mash, Boil, Primary, ...)
    'num_misc_stage_types':    len(dataset.misc_stage_idx_to_name),  # Number of misc stage types (e.g., Mash, Boil, Primary, ...)
    'num_mo_stage_types':      len(dataset.mo_stage_idx_to_name),    # Number of microorganism stage types (e.g., Primary, Secondary)
  }

class RecipeNetArgs:
  def __init__(self, dataset_args) -> None:
    # Recipe-specific constraints ***
    self.num_mash_steps          = NUM_MASH_STEPS
    self.num_grain_slots         = NUM_GRAIN_SLOTS
    self.num_adjunct_slots       = NUM_ADJUNCT_SLOTS
    self.num_hop_slots           = NUM_HOP_SLOTS
    self.num_misc_slots          = NUM_MISC_SLOTS
    self.num_microorganism_slots = NUM_MICROORGANISM_SLOTS
    self.num_ferment_stage_slots = NUM_FERMENT_STAGE_SLOTS
    
    # Embedding sizes ***
    self.grain_type_embed_size         = 63
    self.adjunct_type_embed_size       = 98
    self.hop_type_embed_size           = 256
    self.misc_type_embed_size          = 200
    self.microorganism_type_embed_size = 512
    
    # Loss Notes:
    # - 8096,4096 hidden with z=32 results in a min reconst loss of around ~4500 (batch mean)
    # - 8096,4096 hidden with z=64 results in a min reconst loss of around ~125 (batch mean)
    # - 8096,4096 hidden with z=70 results in a min reconst loss of around ~1.8 (batch mean)
    # - 8096,4096 hidden with z=80 results in a similar loss to z=70
    # - 8096,4096,1024 hidden with z=72 results in ... similar loss (~1.4 batch mean)
    # - 8160,4128 hidden with z=72 ... min reconst loss of around ~1.4 (batch mean)
    # - 9120,5120 hidden with z=128, min reconst loss of ~0.116 (batch mean)
    # - 9248,6144 hidden with z=128, min reconst loss of ~0.107 (batch mean)
    # - 9248,6144,2048 hidden with z=128, min reconst loss of ~0.107 (batch mean)
    # - 9248,6144 hidden with z=128 + layernorm and increased embeddings ~0.081 (batch mean)
    # - 9248,6144 hidden with z=256 + layernorm and increased embeddings ~0.034 (batch mean)

    # Network-specific hyperparameters/constraints ***
    self.hidden_layers = [9248, 6144] # NOTE: A smaller final hidden layer size works better (256 is significantly better than 1024)
    self.z_size = 256 # Latent-bottleneck dimension - NOTE: Don't go smaller than 32
    self.activation_fn = nn.LeakyReLU
    self.activation_fn_params = {}#{'negative_slope': 0.1} # NOTE: The network appears to work better with the default LeakyReLU slope
    self.gain = nn.init.calculate_gain('leaky_relu', 1e-2) # Make sure this corresponds to the activation function!
    self.use_batch_norm = False
    self.use_layer_norm = True

    # VAE-specific hyperparameters ***
    self.beta_vae_gamma = 1000
    self.max_beta_vae_capacity = 30
    self.beta_vae_C_stop_iter = 1e5

    self.beta_tc_vae_alpha = 1.0
    self.beta_tc_vae_beta  = 6.0
    self.beta_tc_vae_gamma = 1.0
    self.beta_tc_vae_anneal_steps = 1e4

    self.num_mash_step_types  = -1
    self.num_hop_stage_types  = -1
    self.num_misc_stage_types = -1
    self.num_mo_stage_types   = -1
    if dataset_args != None:
      for key, value in dataset_args.items():
        setattr(self, key, value)
  
  @property
  def num_toplvl_inputs(self):
    # (boil_time + mash_ph + sparge_temp)
    return 3 
  @property
  def num_mash_step_inputs(self):
     # Mash steps (step_type_index_size + step_time + step_temp) * (number of slots) - ordering assumed [0: step 1, 1: step 2, etc.]
    return self.num_mash_steps*(self.num_mash_step_types + 2)
  @property
  def num_ferment_stage_inputs(self):
    # Fermentation stages (step_time + step_temp) * (number of stages) - ordering assumed [0: primary, 1: secondary]
    return self.num_ferment_stage_slots*(2)
  @property
  def num_grain_slot_inputs(self):
    # Grain/Malt bill slots (grain_type_embed_size + amount) * (number of slots) - no ordering
    return self.num_grain_slots*(self.grain_type_embed_size + 1)
  @property
  def num_adjunct_slot_inputs(self):
    # Adjunct slots (adjunct_type_embed_size + amount) * (number of slots) - no ordering
    return self.num_adjunct_slots*(self.adjunct_type_embed_size + 1)
  @property
  def num_hop_slot_inputs(self):
    # Hop slots (hop_type_embed_size + stage_type_index_size + time + concentration) * (number of slots) - no ordering
    return self.num_hop_slots*(self.hop_type_embed_size + self.num_hop_stage_types + 2)
  @property
  def num_misc_slot_inputs(self):
    # Misc. slots (misc_type_embed_size + stage_type_index_size + time + amounts) * (number of slots) - no ordering
    return self.num_misc_slots*(self.misc_type_embed_size + self.num_misc_stage_types + 2)
  @property
  def num_microorganism_slot_inputs(self):
    # Microorganism slots (mo_type_embed_size + stage_type_index_size) * (number of slots) - no ordering
    return self.num_microorganism_slots*(self.microorganism_type_embed_size + self.num_mo_stage_types)
  
  @property
  def num_inputs(self):
    """Determine the number of inputs to the network.
    Returns:
        int: The total number of network inputs.
    """
    return self.num_toplvl_inputs + self.num_mash_step_inputs + self.num_ferment_stage_inputs + \
           self.num_grain_slot_inputs + self.num_adjunct_slot_inputs + self.num_hop_slot_inputs + \
           self.num_misc_slot_inputs + self.num_microorganism_slot_inputs   