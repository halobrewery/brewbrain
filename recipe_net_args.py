import torch.nn as nn
from recipe_dataset import NUM_GRAIN_SLOTS, NUM_ADJUNCT_SLOTS, NUM_HOP_SLOTS, NUM_MISC_SLOTS, NUM_MICROORGANISM_SLOTS, NUM_FERMENT_STAGE_SLOTS, NUM_MASH_STEPS

FEATURE_TYPE_REAL     = 'real'
FEATURE_TYPE_CATEGORY = 'cate'

def dataset_args(dataset):
  # NOTE: All types include a "None" (i.e., empty) category at index 0
  num_mash_step_types     = len(dataset.mash_step_idx_to_name)
  num_grain_types         = len(dataset.core_grains_idx_to_dbid)
  num_adjunct_types       = len(dataset.core_adjs_idx_to_dbid)
  num_hop_types           = len(dataset.hops_idx_to_dbid)
  num_hop_stage_types     = len(dataset.hop_stage_idx_to_name)
  num_misc_types          = len(dataset.miscs_idx_to_dbid)
  num_misc_stage_types    = len(dataset.misc_stage_idx_to_name)
  num_microorganism_types = len(dataset.mos_idx_to_dbid)
  num_mo_stage_types      = len(dataset.mo_stage_idx_to_name)

  return {
    'num_grain_types':         num_grain_types,         # Number of (core) grain types (rows in the DB)
    'num_adjunct_types':       num_adjunct_types,       # Number of (core) adjunct types (rows in the DB)
    'num_hop_types':           num_hop_types,           # Number of hop types (rows in the DB)
    'num_misc_types':          num_misc_types,          # Number of misc. types (rows in the DB)
    'num_microorganism_types': num_microorganism_types, # Number of microrganism types (rows in the DB)
    'num_mash_step_types':     num_mash_step_types,     # Number of mash step types (e.g., Infusion, Decoction, Temperature)
    'num_hop_stage_types':     num_hop_stage_types,     # Number of hop stage types (e.g., Mash, Boil, Primary, ...)
    'num_misc_stage_types':    num_misc_stage_types,    # Number of misc stage types (e.g., Mash, Boil, Primary, ...)
    'num_mo_stage_types':      num_mo_stage_types,      # Number of microorganism stage types (e.g., Primary, Secondary)
    
    # feature_name: (feature_type, feature_size, number_of_slots, embedding_size)
    'features': {
      # Top-level recipe features
      'boil_time':   (FEATURE_TYPE_REAL, 1, 1, None),
      'mash_ph':     (FEATURE_TYPE_REAL, 1, 1, None),
      'sparge_temp': (FEATURE_TYPE_REAL, 1, 1, None),
      # Mash Steps
      'mash_step_type_inds': (FEATURE_TYPE_CATEGORY, num_mash_step_types, NUM_MASH_STEPS, num_mash_step_types),
      'mash_step_times':     (FEATURE_TYPE_REAL, 1, NUM_MASH_STEPS, None),
      'mash_step_avg_temps': (FEATURE_TYPE_REAL, 1, NUM_MASH_STEPS, None),
      # Grains
      'grain_core_type_inds': (FEATURE_TYPE_CATEGORY, num_grain_types, NUM_GRAIN_SLOTS, 42),
      'grain_amts':           (FEATURE_TYPE_REAL, 1, NUM_GRAIN_SLOTS, None),
      # Adjuncts
      'adjunct_core_type_inds': (FEATURE_TYPE_CATEGORY, num_adjunct_types, NUM_ADJUNCT_SLOTS, 96),
      'adjunct_amts':           (FEATURE_TYPE_REAL, 1, NUM_ADJUNCT_SLOTS, None),
      # Hops
      'hop_type_inds':       (FEATURE_TYPE_CATEGORY, num_hop_types, NUM_HOP_SLOTS, 256),
      'hop_stage_type_inds': (FEATURE_TYPE_CATEGORY, num_hop_stage_types, NUM_HOP_SLOTS, num_hop_stage_types),
      'hop_concentrations':  (FEATURE_TYPE_REAL, 1, NUM_HOP_SLOTS, None),
      'hop_times':           (FEATURE_TYPE_REAL, 1, NUM_HOP_SLOTS, None),
      # Misc.
      'misc_type_inds':  (FEATURE_TYPE_CATEGORY, num_misc_types, NUM_MISC_SLOTS, 196),
      'misc_stage_inds': (FEATURE_TYPE_CATEGORY, num_misc_stage_types, NUM_MISC_SLOTS, num_misc_stage_types),
      'misc_amts':       (FEATURE_TYPE_REAL, 1, NUM_MISC_SLOTS, None),
      'misc_times':      (FEATURE_TYPE_REAL, 1, NUM_MISC_SLOTS, None),
      # Microorganisms
      'mo_type_inds':  (FEATURE_TYPE_CATEGORY, num_microorganism_types, NUM_MICROORGANISM_SLOTS, 320),
      'mo_stage_inds': (FEATURE_TYPE_CATEGORY, num_mo_stage_types, NUM_MICROORGANISM_SLOTS, num_mo_stage_types),
      # Fermentation
      'ferment_stage_times': (FEATURE_TYPE_REAL, 1, NUM_FERMENT_STAGE_SLOTS, None),
      'ferment_stage_temps': (FEATURE_TYPE_REAL, 1, NUM_FERMENT_STAGE_SLOTS, None),
    }
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
    
    # TODO: Remove these
    self.grain_type_embed_size         = 42
    self.adjunct_type_embed_size       = 96
    self.hop_type_embed_size           = 256
    self.misc_type_embed_size          = 196
    self.microorganism_type_embed_size = 320
    
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

    # Loss differences on hidden=[9248, 6144]
    # - z_sizes 256 and 288 have very similar loss numbers... increasing z-dimension no longer making much of a difference


    # Network-specific hyperparameters/constraints ***
    self.hidden_layers = [10272, 4096, 2048]
    self.z_size = 256 # Latent-bottleneck dimension - NOTE: 128 is not enough, 256 gets loss to <= 0.05 (reconst loss batch mean)
    self.alpha_prior = 0.95

    # VQ-VAE parameters
    #self.use_vqvae = True
    #self.embed_width = 128
    #self.n_embed = 256
    #self.embed_dim = 4

    self.activation_fn = nn.LeakyReLU
    self.activation_fn_params = {}#{'negative_slope': 0.1} # NOTE: The network appears to work better with the default LeakyReLU slope
    self.gain = nn.init.calculate_gain('leaky_relu', 1e-2) # Make sure this corresponds to the activation function!
    self.use_batch_norm = False
    self.use_layer_norm = False

    # VAE-specific hyperparameters ***
    self.beta_vae_gamma = 100
    self.max_beta_vae_capacity = 25
    self.beta_vae_C_stop_iter = 1e5

    self.beta_tc_vae_alpha = 1.0
    self.beta_tc_vae_beta  = 6.0
    self.beta_tc_vae_gamma = 1.0
    self.beta_tc_vae_anneal_steps = 1e4

    # Dataset arguements variables...
    self.num_mash_step_types  = -1
    self.num_hop_stage_types  = -1
    self.num_misc_stage_types = -1
    self.num_mo_stage_types   = -1

    self.num_grain_types         = -1
    self.num_adjunct_types       = -1
    self.num_hop_types           = -1
    self.num_misc_types          = -1
    self.num_microorganism_types = -1
    self.num_mash_step_types     = -1
    self.num_hop_stage_types     = -1
    self.num_misc_stage_types    = -1
    self.num_mo_stage_types      = -1
    self.features = {}

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
  
  def num_real_features(self):
    num_real_feats = 0
    for feature_type, _, num_slots, _ in self.features.values():
      num_real_feats += num_slots if feature_type == FEATURE_TYPE_REAL else 0
    return num_real_feats