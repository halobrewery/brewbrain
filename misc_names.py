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

def cleanup_misc_name(misc_name):
  misc_name = misc_name.lower()
  if re.search(r"(floc|flock|irish\s*moss|campden|nutrient|chiller|chiiler|recirculate|pump[^k]|cooler|device)", misc_name) != None: return "ignore"
  if re.search(r"(calcium\s*(sulfate|sulphate)|caso4|gypsum|sulfato de cálcio)", misc_name) != None: return "gypsum"
  if re.search(r"(juniper)", misc_name) != None: return "juniper"
  
  misc_name = re.sub(r"(berrys)", "berries", misc_name)
  misc_name = re.sub(r"(pickeling)", "pickling", misc_name)
  misc_name = re.sub(r"\(?(stars?|paste?urized?|wyeast|white\s*labs?|cacl2|mgso4|mgcl2|nacl|nahco3|ca\(oh2\)|whole|sparge|mash|boil|\d+\s*(g|ml|kg|l))\)?", "", misc_name)
  misc_name = re.sub(r"\,?\(?(wln1000|fresh|chopped|cut|raw|natural|toasted|fried|dry|dried|cracked|belgian|french|ground|bourbon|soaked|\*|\d+%|granulates|granules|crushed|pod|pulp)\)?\,?", "", misc_name)
  misc_name = re.sub(r"^\s*\-\s*", "", misc_name.replace(","," "))
  misc_name = re.sub(r"\s+", " ", misc_name).strip()

  return misc_name

def match_misc_id(misc_name, misc_name_to_id):
  init_name = misc_name
  misc_name = cleanup_misc_name(misc_name)
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
