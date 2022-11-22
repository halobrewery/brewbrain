import torch
import torch.nn as nn
import numpy as np

def layer_init_ortho(layer, std=np.sqrt(2)):
  nn.init.orthogonal_(layer.weight, std)
  nn.init.constant_(layer.bias, 0.0)
  return layer

class DiagonalGaussianDistribution(object):
  def __init__(self, parameters):
    self.parameters = parameters
    self.mean, self.logvar = torch.chunk(parameters, 2, dim=-1)
    self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
    self.std = torch.exp(0.5 * self.logvar)
    self.var = torch.exp(self.logvar)

  def sample(self):
    x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.parameters.device)
    return x

  def kl(self, other=None):
    if other is None:
      return 0.5 * torch.sum(torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar, dim=-1)
    else:
      return 0.5 * torch.sum(torch.pow(self.mean - other.mean, 2) / other.var + self.var / other.var - 1.0 - self.logvar + other.logvar, dim=-1)

  def nll(self, sample):
    LOG_TWO_PI = np.log(2.0 * np.pi)
    return 0.5 * torch.sum(LOG_TWO_PI + self.logvar + torch.pow(sample - self.mean, 2) / self.var, dim=-1)

  def mode(self): return self.mean


class MashVAE(nn.Module):
  def __init__(self, num_grain_types, num_grain_slots, hidden_size, z_size) -> None:
    super().__init__()
    GRAIN_EMBED_SIZE = 128

    self.num_grain_slots = num_grain_slots
    self.num_grain_types = num_grain_types

    self.grain_type_embed = nn.Embedding(num_grain_types, GRAIN_EMBED_SIZE)
    self.input_size = (GRAIN_EMBED_SIZE + 1) * num_grain_slots # type embed (embed size floats) + percentage of the mass (1 float)
    self.encoder = nn.Sequential(
      layer_init_ortho(nn.Linear(self.input_size, hidden_size)),
      nn.ReLU(inplace=True),
      layer_init_ortho(nn.Linear(hidden_size, 2*z_size))
    )
    self.decoder = nn.Sequential(
      layer_init_ortho(nn.Linear(z_size, hidden_size)),
      nn.ReLU(inplace=True),
      layer_init_ortho(nn.Linear(hidden_size, self.input_size)),
    )
    self.type_decoder = layer_init_ortho(nn.Linear(GRAIN_EMBED_SIZE, num_grain_types))

  def encode(self, x):
    h = self.encoder(x)
    posterior = DiagonalGaussianDistribution(h)
    return posterior

  def decode(self, z):
    x_hat = self.decoder(z)
    type_x_hats, amt_x_hats = torch.split(x_hat, (self.num_grain_slots * self.grain_type_embed.embedding_dim, self.num_grain_slots), dim=-1)

    category_x_hats = self.type_decoder(type_x_hats.view(z.shape[0],self.num_grain_slots,-1))
    amt_x_hats = amt_x_hats.view(z.shape[0],self.num_grain_slots,-1)

    return category_x_hats, amt_x_hats

  def forward(self, batch_mashes, sample_posterior=True):
    # batch_mashes is a tuple (type_indices, amount_percentages), shape is ((B, num_grain_slots), (B, num_grain_slots))
    embed_types = self.grain_type_embed(batch_mashes[0]).view(batch_mashes[0].shape[0], -1)
    x = torch.cat([embed_types, batch_mashes[1]], dim=-1)
    posterior = self.encode(x)
    z = posterior.sample() if sample_posterior else posterior.mode()
    category_reconst, amount_reconst = self.decode(z)
    return category_reconst, amount_reconst, posterior

  def loss(self, batch_mashes, category_reconsts, amount_reconsts, posteriors, kl_weight=0.1):
    # batch_mashes is the original input to the network, 
    # it's a tuple (type_indices, amount_percentages), shape is ((B, num_grain_slots), (B, num_grain_slots))
    
    # Break the reconstructions up to get the types and amounts
    one_hot_cats = nn.functional.one_hot(batch_mashes[0], self.num_grain_types).float()

    cat_reconst_loss = nn.functional.binary_cross_entropy_with_logits(category_reconsts, one_hot_cats, reduction='sum')
    #cat_reconst_loss /= category_reconsts.shape[0]
    amt_reconst_loss = nn.functional.mse_loss(nn.functional.softmax(amount_reconsts, dim=1), batch_mashes[1].unsqueeze(-1), reduction='sum')
    #amt_reconst_loss /= amount_reconsts.shape[0]

    # Calculate the KL loss
    kl_loss = posteriors.kl()
    kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

    total_loss = cat_reconst_loss + amt_reconst_loss + kl_weight * kl_loss

    return total_loss, amt_reconst_loss, cat_reconst_loss, kl_loss

'''
class VAELoss(nn.Module):
  def __init__(self, logvar_init=0.0, kl_weight=1.0, pixelloss_weight=1.0):
    super().__init__()
    self.kl_weight = kl_weight
    self.pixel_weight = pixelloss_weight
    self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)
    
  def forward(self, inputs, reconstructions, posteriors):
    rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
    nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
    nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]

    kl_loss = posteriors.kl()
    kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

    return nll_loss + self.kl_weight * kl_loss, nll_loss, kl_loss
'''