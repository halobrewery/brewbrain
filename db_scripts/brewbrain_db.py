import hashlib

from sqlalchemy import Column, Integer, String, Float, Boolean
from sqlalchemy import ForeignKey, UniqueConstraint
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import relationship

BREWBRAIN_DB_ENGINE_STR = "sqlite:///brewbrain.db"

Base = declarative_base()

class Grain(Base):
  __tablename__ = "grains"
  id = Column(Integer, primary_key=True)
  name = Column(String(128))
  origin = Column(String(64))
  supplier = Column(String(64))
  colour_srm = Column(Float(2))
  coarse_fine_diff = Column(Float(4))
  moisture = Column(Float(4))
  protein = Column(Float(4))
  diastatic_power = Column(Float(4))
  is_fermentable = Column(Boolean())
  dbfg = Column(Float(4))

  core_grain_id = Column(Integer(), ForeignKey("core_grains.id"))
  core_grain = relationship("CoreGrain", back_populates="grains")

  def __repr__(self):
    return f"Grain(id={self.id!r}, name={self.name!r}, origin={self.origin!r}, supplier={self.supplier!r})"

class CoreGrain(Base):
  __tablename__ = "core_grains"
  id = Column(Integer, primary_key=True)
  name = Column(String(128), unique=True)
  grains = relationship("Grain", back_populates="core_grain")

  def __repr__(self):
    return f"CoreGrain(id={self.id!r}, name={self.name!r})"


class Adjunct(Base):
  __tablename__ = "adjuncts"
  id = Column(Integer, primary_key=True)
  name = Column(String(128))
  origin = Column(String(64))
  supplier = Column(String(64))
  colour_srm = Column(Float(2))
  is_fermentable = Column(Boolean())
  yield_amt = Column(Float(4))

  def __repr__(self):
    return f"Adjunct(id={self.id!r}, name={self.name!r}, origin={self.origin!r}, supplier={self.supplier!r})"


class Misc(Base):
  __tablename__ = "miscs"
  __table_args__ = (
    UniqueConstraint("name", "type"),
  )

  id = Column(Integer, primary_key=True)
  name = Column(String(128))
  type = Column(String(32)) # {"spice", "fining", "herb", "flavour", "other"}
  
  def __repr__(self):
    return f"Misc(id={self.id!r}, name={self.name!r}, type={self.type!r})"


class Hop(Base):
  __tablename__ = "hops"
  id = Column(Integer, primary_key=True)
  name = Column(String(128))
  origin = Column(String(64))
  avg_alpha = Column(Float(4))
  avg_beta = Column(Float(4))

  def __repr__(self):
    return f"Hop(id={self.id!r}, name={self.name!r}, origin={self.origin!r})"


class Microorganism(Base):
  __tablename__ = "microorganisms"
  id = Column(Integer, primary_key=True)
  name = Column(String(128))
  lab = Column(String(64))
  product_code = Column(String(16))
  flocculation = Column(String(32))
  min_atten = Column(Float(4))
  max_atten = Column(Float(4))
  min_temp_c = Column(Float(4))
  max_temp_c = Column(Float(4))

  def __repr__(self):
    return f"Microorganism(id={self.id!r}, name={self.name!r}, lab={self.lab!r}, product_code={self.product_code!r})"


class Style(Base):
  __tablename__ = "styles"
  id = Column(Integer, primary_key=True)
  name = Column(String(128))
  category = Column(String(128))
  guide = Column(String(128))
  guide_year = Column(Integer(), default=None)
  number = Column(String(16))
  letter = Column(String(16))
  type = Column(String(32))
  min_og = Column(Float(4))
  max_og = Column(Float(4))
  min_fg = Column(Float(4))
  max_fg = Column(Float(4))
  min_ibu = Column(Float(2))
  max_ibu = Column(Float(2))
  min_carb = Column(Float(2))
  max_carb = Column(Float(2))
  min_colour_srm = Column(Float(2))
  max_colour_srm = Column(Float(2))
  min_abv = Column(Float(2))
  max_abv = Column(Float(2))

  UniqueConstraint("name", "category", "guide", "guide_year", "number", "letter")
  recipes = relationship("RecipeML", back_populates="style")

  core_style_id = Column(Integer(), ForeignKey("core_styles.id"))
  core_style = relationship("CoreStyle", back_populates="styles")

  def __repr__(self):
    return f"Style(id={self.id!r}, name={self.name!r}, category={self.category!r})"


# Distilled styles, these are the most 'core' styles possible in beer, no guides/categories/numbers/letters.
# These sum styles across all style guides into the most simplified 'prototypical' styles in all of beer.
class CoreStyle(Base):
  __tablename__ = "core_styles"
  id = Column(Integer, primary_key=True)
  name = Column(String(128), unique=True)
  styles = relationship("Style", back_populates="core_style")


# Association Table (AT) for many-to-many relationship between RecipeMLs and Hops
class RecipeMLHopAT(Base):
  __tablename__= "recipe_hop_at"
  id = Column(Integer, primary_key=True)
  recipe_ml_id = Column(Integer(), ForeignKey("recipes_ml.id"))
  hop_id       = Column(Integer(), ForeignKey("hops.id"))
  
  amount = Column(Float(6))   # Quantity of hops used in kg
  stage  = Column(String(32)) # Stage of use {"boil", "dry hop", "mash", "first wort", "aroma"}
  time   = Column(Integer())  # Amount of time, dependant on use, in mins
  form   = Column(String(32)) # Form of the hops {"pellet", "leaf", "plug", "cryo"}
  alpha  = Column(Float(2))   # Alpha acids of the hop used

  recipe_ml = relationship("RecipeML", back_populates="hops")
  hop       = relationship("Hop")


# Association Table (AT) for many-to-many relationship between RecipeMLs and Grains
class RecipeMLGrainAT(Base):
  __tablename__ = "recipe_grain_at"
  id = Column(Integer, primary_key=True)
  recipe_ml_id = Column(Integer(), ForeignKey("recipes_ml.id"))
  grain_id     = Column(Integer(), ForeignKey("grains.id"))

  amount = Column(Float(6))   # Quantity of grains used in kg
  fgdb_override = Column(Float(2), default=None) # Override for the fine grain, dry basis (percent yield) [0,1]
  moisture_override = Column(Float(2), default=None) # Override for the moisture of the grain [0,1]
  coarse_fine_diff_override = Column(Float(2), default=None) # Override for the coarse/fine diff [0,1]
  protein_override = Column(Float(2), default=None) # Override for the protein content [0,1]
  
  recipe_ml = relationship("RecipeML", back_populates="grains")
  grain     = relationship("Grain")
  
  # Calculate the Dry-Basis Coarse Grind of the grain addition.
  def calc_dbcg(self):
    dbfg = self.fgdb_override if self.fgdb_override != None else self.grain.dbfg
    fgcg_diff = self.grain.coarse_fine_diff
    return dbfg - fgcg_diff
  
  # Calculate the Coarse-Grind-As-Is (accounting for moisture) of the grain addition.
  # This is the number that should be used to calculate the actual ppg or pkl for the addition.
  def calc_coarse_grid_as_is(self):
    dbcg = self.calc_dbcg()
    moisture = self.moisture_override if self.moisture_override != None else self.grain.moisture
    return dbcg * (1.0 - moisture)
  

# Association Table (AT) for many-to-many relationship between RecipeMLs and Adjuncts
class RecipeMLAdjunctAT(Base):
  __tablename__ = "recipe_adjunct_at"
  id = Column(Integer, primary_key=True)
  recipe_ml_id = Column(Integer(), ForeignKey("recipes_ml.id"))
  adjunct_id   = Column(Integer(), ForeignKey("adjuncts.id"))

  amount = Column(Float(6))                       # Quantity of adjunct used in kg
  yield_override = Column(Float(2), default=None) # Override for the yield of the adjunct (percent by weight of sugars)
  
  recipe_ml = relationship("RecipeML", back_populates="adjuncts")
  adjunct   = relationship("Adjunct")

# Association Table (AT) for many-to-many relationship between RecipeMLs and Misc
class RecipeMLMiscAT(Base):
  __tablename__ = "recipe_misc_at"
  id = Column(Integer, primary_key=True)
  recipe_ml_id = Column(Integer(), ForeignKey("recipes_ml.id"))
  misc_id      = Column(Integer(), ForeignKey("miscs.id"))

  amount = Column(Float(6))            # Quantity of miscellaneous ingredient used in kg or L
  amount_is_weight = Column(Boolean()) # If True the amount is in kg, if False then L

  stage = Column(String(32)) # Stage of use {"boil", "mash", "whirlpool", "primary", "secondary", "bottling"}
  time  = Column(Integer())  # Amount of time the ingredient was used, in mins
  
  recipe_ml = relationship("RecipeML", back_populates="miscs")
  misc      = relationship("Misc")

# Association Table (AT) for many-to-many relationship between RecipeMLs and Microorganisms
class RecipeMLMicroorganismAT(Base):
  __tablename__ = "recipe_microorganism_at"
  id = Column(Integer, primary_key=True)
  recipe_ml_id     = Column(Integer(), ForeignKey("recipes_ml.id"))
  microorganism_id = Column(Integer(), ForeignKey("microorganisms.id"))

  stage = Column(String(32)) # The fermentation stage that the microorganism was used {"primary", "secondary", "tertiary", "bottling"}
  
  recipe_ml     = relationship("RecipeML", back_populates="microorganisms")
  microorganism = relationship("Microorganism")


class RecipeML(Base):
  __tablename__ = "recipes_ml"
  id = Column(Integer, primary_key=True)
  hash = Column(String(64), unique=True)
  data_version = Column(Integer, default=None)

  name = Column(String(128)) # Name, for reference / readability
  preboil_vol   = Column(Float(2), default=None) # The pre-boil volume in L
  postboil_vol  = Column(Float(2), default=None) # The post-boil volume in L 
  fermenter_vol = Column(Float(2), default=None) # The volume in L in the fermenter after castout

  boil_time    = Column(Integer(), default=None)  # Boil length in mins
  efficiency   = Column(Float(4),  default=None)  # Expected efficency of the brewhouse (used to estimate the starting gravity of the beer) [0,1]

  mash_ph                = Column(Float(2),  default=None)  # Mash pH
  sparge_temp            = Column(Float(2),  default=None)  # Sparge water temperature in C
  num_mash_steps         = Column(Integer(), default=None)  # Number of steps in the mash (corresponds to which values of mash_step_x are used)
  mash_step_1_type       = Column(String(16), default=None) # Type of mash step {"infusion", "temperature", "decoction"}
  mash_step_1_time       = Column(Integer(),  default=None) # Time in minutes of step 1 of the mash
  mash_step_1_start_temp = Column(Float(2),   default=None) # Starting temperature of step 1 of the mash, in C
  mash_step_1_end_temp   = Column(Float(2),   default=None) # Ending temperature of step 1 of the mash, in C
  mash_step_1_infuse_amt = Column(Float(2),   default=None) # For "infusion" types, this is the volume of infusion water (L) for step 1 of the mash
  mash_step_2_type       = Column(String(16), default=None) # etc...
  mash_step_2_time       = Column(Integer(),  default=None) 
  mash_step_2_start_temp = Column(Float(2),   default=None) 
  mash_step_2_end_temp   = Column(Float(2),   default=None) 
  mash_step_2_infuse_amt = Column(Float(2),   default=None) 
  mash_step_3_type       = Column(String(16), default=None)
  mash_step_3_time       = Column(Integer(),  default=None) 
  mash_step_3_start_temp = Column(Float(2),   default=None) 
  mash_step_3_end_temp   = Column(Float(2),   default=None) 
  mash_step_3_infuse_amt = Column(Float(2),   default=None) 
  mash_step_4_type       = Column(String(16), default=None)
  mash_step_4_time       = Column(Integer(),  default=None) 
  mash_step_4_start_temp = Column(Float(2),   default=None) 
  mash_step_4_end_temp   = Column(Float(2),   default=None) 
  mash_step_4_infuse_amt = Column(Float(2),   default=None) 
  mash_step_5_type       = Column(String(16), default=None)
  mash_step_5_time       = Column(Integer(),  default=None) 
  mash_step_5_start_temp = Column(Float(2),   default=None) 
  mash_step_5_end_temp   = Column(Float(2),   default=None) 
  mash_step_5_infuse_amt = Column(Float(2),   default=None) 
  mash_step_6_type       = Column(String(16), default=None)
  mash_step_6_time       = Column(Integer(),  default=None) 
  mash_step_6_start_temp = Column(Float(2),   default=None) 
  mash_step_6_end_temp   = Column(Float(2),   default=None) 
  mash_step_6_infuse_amt = Column(Float(2),   default=None) 

  num_ferment_stages   = Column(Integer(), default=None)  # Number of fermentation stages (should correspond to which of the primary/secondary/tertiary values are used)
  ferment_stage_1_time = Column(Integer(), default=None)  # Primary fermentation time (days)
  ferment_stage_1_temp = Column(Float(2),  default=None)  # Temp (C) of primary fermentation
  ferment_stage_2_time = Column(Integer(), default=None)  # Secondary fermentation time (days)
  ferment_stage_2_temp = Column(Float(2),  default=None)  # Temp (C) of secondary fermentation
  ferment_stage_3_time = Column(Integer(), default=None)  # Tertiary fermentation time (days)
  ferment_stage_3_temp = Column(Float(2),  default=None)  # Temp (C) of tertiary fermentation

  aging_time = Column(Integer(), default=None)   # Aging time in bottle (days)
  aging_temp = Column(Float(2),  default=None)   # Temp (C) for aging in bottle

  carbonation = Column(Float(2),  default=None)  # Target carb in vols

  hops           = relationship("RecipeMLHopAT", cascade="all, delete-orphan")
  grains         = relationship("RecipeMLGrainAT", cascade="all, delete-orphan")
  adjuncts       = relationship("RecipeMLAdjunctAT", cascade="all, delete-orphan")
  miscs          = relationship("RecipeMLMiscAT", cascade="all, delete-orphan")
  microorganisms = relationship("RecipeMLMicroorganismAT", cascade="all, delete-orphan")

  style_id = Column(Integer, ForeignKey("styles.id"))
  style    = relationship("Style", back_populates="recipes")

  MAX_MASH_STEPS = 6
  MASH_STEP_PREFIX = "mash_step_"
  MASH_STEP_POSTFIXES = ["_type", "_time", "_start_temp", "_end_temp", "_infuse_amt"]

  def mash_steps(self):
    steps = []
    for i in range(self.num_mash_steps):
      prefix = "mash_step_"+str(i+1)
      step = {}
      for postfix in self.MASH_STEP_POSTFIXES:
        step[prefix+postfix] = getattr(self, prefix+postfix)
      steps.append(step)
    return steps

  def total_infusion_vol(self):
    infuse_total = 0
    for i in range(self.num_mash_steps):
      prefix = "mash_step_"+str(i+1)
      infuse_amt = getattr(self, prefix+"_infuse_amt")
      infuse_total += infuse_amt if infuse_amt != None else 0
    return infuse_total
  
  def total_grain_mass(self):
    mass_total = 0
    for grain in self.grains:
      mass_total += grain.amount
    return mass_total   

  # Generates a unique identifier hash for this machine learning recipe 
  # to avoid duplicates of the same recipes in the database
  def gen_hash(self):
    assert self.mash_ph != None and self.num_mash_steps != None
    hash = hashlib.md5()
    hash.update(str(int(self.boil_time)).encode())
    hash.update(str(round(self.mash_ph,2)).encode())
    hash.update(str(int(self.num_mash_steps)).encode())

    for i in range(self.num_mash_steps):
      prefix = self.MASH_STEP_PREFIX + str(i+1)
      step_type = getattr(self, prefix+"_type")
      step_time = getattr(self, prefix+"_time")
      step_start_temp = getattr(self,prefix+"_start_temp")
      step_end_temp   = getattr(self,prefix+"_end_temp")
      assert step_type != None and step_time != None and step_start_temp != None and step_end_temp != None
      hash.update(step_type.encode())
      hash.update(str(int(step_time)).encode())
      hash.update(str(round(step_start_temp,1)).encode())
      hash.update(str(round(step_end_temp,1)).encode())
    
    for i in range(self.num_ferment_stages):
      prefix = "ferment_stage_"+str(i+1)
      stage_time = getattr(self,prefix+"_time")
      stage_temp = getattr(self,prefix+"_temp")
      assert stage_time != None and stage_temp != None
      hash.update(str(int(stage_time)).encode())
      hash.update(str(round(stage_temp,1)).encode())

    for hop_addition in self.hops:
      hop_id    = hop_addition.hop_id
      hop_amt   = hop_addition.amount
      hop_stage = hop_addition.stage
      hop_time  = hop_addition.time
      assert hop_amt != None and hop_stage != None and hop_time != None
      hash.update(str(hop_id).encode())
      hash.update(str(round(hop_amt,6)).encode())
      hash.update(hop_stage.encode())
      hash.update(str(int(hop_time)).encode())

    for grain_addition in self.grains:
      grain_id  = grain_addition.grain_id
      grain_amt = grain_addition.amount
      assert grain_amt != None
      hash.update(str(grain_id).encode())
      hash.update(str(round(grain_amt,6)).encode())

    for adjunct_addition in self.adjuncts:
      adjunct_id = adjunct_addition.adjunct_id
      adjunct_amt = adjunct_addition.amount
      assert adjunct_amt != None
      hash.update(str(adjunct_id).encode())
      hash.update(str(round(adjunct_amt,6)).encode())

    for mo_addition in self.microorganisms:
      mo_id = mo_addition.microorganism_id
      mo_stage = mo_addition.stage
      assert mo_stage != None
      hash.update(str(mo_id).encode())
      hash.update(mo_stage.encode())

    # Who cares about misc. stuff, they are not important enough to differentiate 
    # a recipe from a machine learning perspective

    # Also ignore the style since it's just metadata to the actual recipe 
    # and there may be multiples of the same style across style guides

    return hash.hexdigest()
