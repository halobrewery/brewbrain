
import csv
import re

from beer_util_functions import ebc_to_lovibond

def build_fermentable_name_dicts():
  # Create a fermentable name-to-id system
  fermentable_name_to_id  = {}
  id_to_fermentable_names = {}
  with open("./data/_db/fermentables.csv", "r", encoding="utf-8") as f:
      csv_reader = csv.reader(f, delimiter=";")
      for i,row in enumerate(csv_reader):
          if i == 0: continue
          names = []
          fermentable_name = row[0].lower().strip()
          fermentable_name_to_id[fermentable_name] = i
          names.append(fermentable_name)
          
          for alt_name in row[3].split(",") + row[4].split(","):
              alt_name = alt_name.strip()
              if len(alt_name) > 0:
                  alt_name = alt_name.lower()
                  fermentable_name_to_id[alt_name] = i
                  names.append(alt_name)
                  
          assert len(names) > 0
          id_to_fermentable_names[i] = names
  return fermentable_name_to_id, id_to_fermentable_names

def match_fermentable_id(fermentable_name, fermentable_name_to_id):
  f_name = str(fermentable_name).lower().strip()
  init_name = f_name
  
  # Start with some basic clean-up
  f_name = f_name.replace("marris","maris") \
    .replace("roasted oats","malted oats").replace("torrefied","torrified").replace("specl","special") \
    .replace("carafaz", "carafa").replace("kรถlsch", "kolsch").replace("ch창teau", "chateau") \
    .replace("chã¢teau", "chateau").replace(",","").replace("–", "-")

  f_name = re.sub(r"(Â|®|mdâ|¢|„|â|速|ã|©|aรงucar|รง|MD™)", "", f_name, flags=re.IGNORECASE)
  f_name = re.sub(r"(american|belgian|german)\s*\-?\s*", "", f_name, flags=re.IGNORECASE)
  f_name = re.sub(r"(\s{2,})", " ", f_name)
  f_name = re.sub(r"[\'\"](w|b)[\'\"]", r"\1", f_name, flags=re.IGNORECASE)
  f_name = re.sub(r"\brosted\b", "roasted", f_name, flags=re.IGNORECASE)
  f_name = re.sub(r"\bcarmel\b", "caramel", f_name, flags=re.IGNORECASE)
  f_name = re.sub(r"\bsirup\b", "syrup", f_name, flags=re.IGNORECASE)
  f_name = re.sub(r"\,?\s*steel\s*cut\s*\(?pinhead\s*oats\)?", "", f_name, flags=re.IGNORECASE)
  f_name = re.sub(r"(md)?™","",f_name,flags=re.IGNORECASE)
  f_name = re.sub(r"(.+)\s*\,\s*(flaked|malted|cooked)\s*", r"\1 \2", f_name, flags=re.IGNORECASE)
  f_name = re.sub(r"\s*\-*\s*\((us|de|uk|nz|be|pl|english|british|german|belgian|patent|american|late|addition|mash)\)\s*", " ", f_name, flags=re.IGNORECASE)
  f_name = re.sub(r"\s*(english|german|lund)\s*", "", f_name, flags=re.IGNORECASE)
  f_name = re.sub(r"(cara\s+|crystal\s*/\s*caramel|caramel\s*/\s*crystal|caramel|crystal)(\s+malts?\s+\-?)?","crystal ", f_name, flags=re.IGNORECASE)
  f_name = re.sub(r"\-*\s*\(?(malt=|bob'?s\s*mills|clear\s*choice|muntons|homemade|malting|cal\s*rose|cargill|((green|black|gold)\s?)?swaen|munton'?s?|warminster|biscoff|patagonia|global malt|(thomas)?\s*fawcett?|crisp|best\s*(malz|maltz)|simpson'?s?|valley\s*(malt)?|briess|bairds|gladfield'?s?|château|chateau|castle|dingemans|brewers\s+malt|rahr|organic|dehusked|weyermann?)\)?\s*\-*\s*", " ", f_name, flags=re.IGNORECASE)
  f_name = re.sub(r"\(?(\d+|\d+\-\d+)\s+(ebc|lovibond)\)?\:?", "", f_name, flags=re.IGNORECASE)
  f_name = re.sub(r"\s+", " ", f_name).strip()
  f_name = f_name.replace(",","")
  
  if re.search(r"pale\s*2-row$", f_name) != None:
    f_name = "2-row"
  elif re.search(r"crystal\s+\d+\s?l?", f_name) != None:
    crystal_num = re.sub(r"crystal\s+(\d+\s?l?)", r"\1", f_name, flags=re.IGNORECASE)
    if 'l' in crystal_num:
      f_name = re.sub(r".*(crystal\s+\d+\s?l?).*", r"\1", f_name, flags=re.IGNORECASE)
    else:
      # The number could be ebc... this is a bit tricky, try to be reasonable, if the number is quite large then it is very likely EBC
      crystal_num = int(crystal_num)
      if crystal_num >= 100:
        crystal_num = int((ebc_to_lovibond(crystal_num) // 5) * 5)
        f_name = "crystal " + str(crystal_num) + "l"

  if f_name in fermentable_name_to_id:
    return fermentable_name_to_id[f_name]
      
  # Try adding "malt" to the end
  f_name_w_malt = f_name+" malt" 
  if f_name_w_malt in fermentable_name_to_id:
    return fermentable_name_to_id[f_name_w_malt]

  # Try removing "malt" from the end
  f_name_w_malt = re.sub(r"\s*malt", "", f_name, flags=re.IGNORECASE)
  if f_name_w_malt in fermentable_name_to_id:
    return fermentable_name_to_id[f_name_w_malt]
  
  best_len_s1 = 0
  best_id_s1  = -1
  best_len_s2 = 0
  best_id_s2  = -1
  for dict_name, dict_id in fermentable_name_to_id.items():
    name_opts = "(" + dict_name + ")"
    s1 = re.search(name_opts, f_name)
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

  if best_len_s1 == 0:
    if best_len_s2 == 0:
      return None
    else:
      return best_id_s2
  else:
    return best_id_s1 if best_len_s1 > best_len_s2 else best_id_s2
  
