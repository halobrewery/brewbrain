import csv
import re

def build_misc_name_dicts():
  # Create a hop name-to-id system
  misc_name_to_id  = {}
  id_to_misc_names = {}
  with open("./data/_db/misc.csv", "r", encoding="utf-8") as f:
      csv_reader = csv.reader(f, delimiter=";")
      for i,row in enumerate(csv_reader):
        names = []
        for j in range(len(row)):
          misc_name = row[j].lower().strip()
          misc_name_to_id[misc_name] = i
          names.append(misc_name)
        id_to_misc_names[i] = names

  return misc_name_to_id, id_to_misc_names

def cleanup_misc_name(misc_name, recipe_version):
  misc_name = misc_name.lower()
  if re.search(r"(colou?ring|diammon|phosphat|ising|cacl|flour|starch|brew(bri|tan)|clearex|gjær|wln\s*1000|prota|polyc|oxygen|biofi|ferm\s?cap|kleer|foam|clari(fy|ty)|delete|floc|flock|(irish|super)\s*moss|nutri|chiller|chiiler|recirculate|pump[^k]|cooler|device|gelat|ph\s*stab|5\.2|hop\s?shot|oysters?|(rice|oat)\s*(hull|husk|shell)s?)", misc_name) != None: return "ignore"
  if re.search(r"\b(camp?den|bacon|yeast|dap|tequila|whiskey|cognac|rum|hops?|vodka|bo?urbon|brandy|daniels?|maker'?s|rye)\b", misc_name) != None: return "ignore"
  if re.search(r"(calcium\s*(sul(ph|f)ate)|caso4|gyp?sum|sulfato(.){2,4}c(á|a)lc)", misc_name) != None: return "gypsum"
  if re.search(r"(junip)", misc_name) != None: return "juniper"
  
  if recipe_version > 0:
    if re.search(r"((dor..|brun|brown)\s*(sugar|fonce|sucr)|cassonad)", misc_name) != None: return "brown sugar"
    if re.search(r"(molas|black\s?str)", misc_name) != None: "molasses"
    if re.search(r"(lactos|milk\s*sugar)", misc_name) != None: return "lactose"
    if re.search(r"(maltodr?ex)", misc_name) != None: return "maltodextrin"
    if re.search(r"(sucr|(raw|table|cane)\s*sug|sugar\s*cane)", misc_name) != None: return "sucrose"
    if re.search(r"(dextro|(corn|prime?ing?)\s*sug)", misc_name) != None: return "dextrose"
    if re.search(r"(molass|black\s*strap)", misc_name) != None: return "molasses"
    if re.search(r"(agave|fruct)", misc_name) != None: return "agave nectar"
    if re.search(r"(cand[iy]\s*s(yr|ug)|\binvert\b)", misc_name) != None: return "candi sugar"
    if re.search(r"(\bpb\b|peanut but)", misc_name) != None: return "powdered peanut butter"
    if re.search(r"(grape\s*conc)", misc_name) != None: return "grape concentrate"
    if re.search(r"(wild\s*grapes?)", misc_name) != None: return "grapes"
    if re.search(r"((cabernet|sauvignon|chardonay|riesling|grape|sauvignon|pinot)\s*(blanc|noir)?\s*(grape)?\s*must)", misc_name) != None: return "grape juice/must"
    if re.search(r"\b(honey)\b", misc_name) != None: return "honey"
    
    if re.search(r"(oak)", misc_name) != None:
      if re.search(r"(cubes?)", misc_name) != None:
        if re.search(r"(french)", misc_name) != None: 
          if re.search(r"(medium)", misc_name) != None:
            if re.search(r"(heavy)", misc_name) != None: return "french oak cubes medium - heavy toast"
            else: return "french oak cubes - medium toast"
          elif re.search(r"(heavy)", misc_name) != None: return "french oak cubes - heavy toast"
        
        elif re.search(r"(americ)", misc_name) != None:
          if re.search(r"(medium)", misc_name) != None:
            if re.search(r"(heavy)", misc_name) != None: return "american oak cubes medium-heavy toast"
            else: return "american oak cubes - medium toast"
          elif re.search(r"(heavy)", misc_name) != None: return "american oak cubes - heavy toast"
          
      elif re.search(r"(hung)", misc_name) != None: return "hungarian oak"
      else: return "oak (general)"
      
    misc_name = re.sub(r"\(?(whole|quartere?d?|half|\b\d+\s?.\s?\d+\b)\)?", "", misc_name)
    
  misc_name = re.sub(r"(berrys)", "berries", misc_name)
  misc_name = re.sub(r"(pickeling)", "pickling", misc_name)
  misc_name = re.sub(r"\(?(stars?|paste?urized?|wyeast|white\s*labs?|cacl2|mgso4|mgcl2|nacl|nahco3|ca\(oh2\)|whole|sparge|mash|boil(ed)?|\d+\s*(g|ml|kg|l))\)?", "", misc_name)
  misc_name = re.sub(r"\,?\(?(loose|frozen|fresh|chopped|cut|five\s*star|raw|natural|toasted|slices?|fried|dry|dried|cracked|belgian|french|ground|bourbon|soaked|\*|\d+%|granulates|granules|crushed|pod|pulp|shred(ded)?)\)?\,?", "", misc_name)
  misc_name = re.sub(r"^\s*\-\s*", "", misc_name.replace(","," "))
  misc_name = re.sub(r"\s+", " ", misc_name).strip()

  return misc_name

def match_misc_id(misc_name, misc_name_to_id, recipe_version=0):
  init_name = misc_name
  misc_name = cleanup_misc_name(misc_name, recipe_version)
  if misc_name == None or misc_name == 'ignore': return misc_name

  if misc_name in misc_name_to_id:
    return misc_name_to_id[misc_name]
  
  best_len_s1 = 0
  best_id_s1  = -1
  best_len_s2 = 0
  best_id_s2  = -1
  for dict_name, dict_id in misc_name_to_id.items():
    name_opts = "(" + dict_name + ")"
    s1 = re.search(name_opts, misc_name)
    if s1 != None and len(s1.group()) > 0:
      group_len = len(s1.group())
      if best_len_s1 < group_len:
        best_len_s1 = group_len
        best_id_s1  = dict_id
    else:
      s2 = re.search(name_opts, init_name)
      if s2 != None and len(s2.group()) > 0:
        group_len = len(s2.group())
        if best_len_s2 < group_len:
          best_len_s2 = group_len
          best_id_s2  = dict_id

  if best_len_s1 < 3 and best_len_s2 < 3: return None

  if best_len_s1 == 0:
    if best_len_s2 == 0:
      return None
    else:
      return best_id_s2
  else:
    return best_id_s1 if best_len_s1 > best_len_s2 else best_id_s2
