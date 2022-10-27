import csv
import re

def build_yeast_dicts(yeast_db_filepath="./data/_db/yeasts.csv"):
  yeast_name_to_id = {}
  brand_to_ids = {}
  id_to_yeast_names = {}
  with open(yeast_db_filepath, "r", encoding="utf-8") as f:
    csv_reader = csv.reader(f, delimiter=";")
    for i,row in enumerate(csv_reader):
      if i == 0: continue
      names = []
      yeast_name = row[0].lower().strip()
      yeast_name_to_id[yeast_name] = i
      names.append(yeast_name)
      
      for alt_name in row[1].split(",") + row[2].split(","):
        if len(alt_name) > 0:
          alt_name = alt_name.lower().strip()
          yeast_name_to_id[alt_name] = i
          names.append(alt_name)
              
      for product_id in row[7].split(",") + row[8].split(","):
        if len(product_id) > 0:
          product_id = product_id.lower().strip()
          yeast_name_to_id[product_id] = i
          names.append(product_id)
      assert len(names) > 0
      id_to_yeast_names[i] = names
      
      for brand in row[5].split(",") + row[6].split(","):
        if len(brand) > 0:
          if brand not in brand_to_ids:
            brand_to_ids[brand] = []
          brand_to_ids[brand].append(i)

  return yeast_name_to_id, brand_to_ids, id_to_yeast_names

def build_style_to_common_yeast_dict(
  yeast_name_to_id,
  style_to_common_yeasts_db_filepath="./data/_db/style_to_common_yeasts.csv"):
  style_to_yeast_ids = {}
  with open(style_to_common_yeasts_db_filepath, "r", encoding="utf-8") as f:
    csv_reader = csv.reader(f, delimiter=";")
    for i, row in enumerate(csv_reader):
      if i == 0: continue
      style_name = row[0].lower().strip()
      #style_id = int(row[1])
      yeast_ids = []
      style_to_yeast_ids[style_name] = yeast_ids
      for yeast in row[2].split(","):
        yeast_name = yeast.lower().strip()
        if len(yeast_name) == 0: continue
        if yeast_name in yeast_name_to_id:
          yeast_ids.append(yeast_name_to_id[yeast_name])
        else:
          print(f"Failed to find yeast '{yeast_name}' in database, skipping.")
  return style_to_yeast_ids

# Given a recipe and yeast id dictionaries, return any matching ID for the yeast of the given recipe
def match_recipe_to_yeast_id(recipe, yeast_name_to_id, brand_to_ids, id_to_yeast_names):

  def clean_replace(s, target):
      return s.replace(target, '').replace("  ", " ").strip()

  for y in recipe.yeasts:
    if not isinstance(y.name, str):
      # Try turning the name into a code (likely a wyeast strain)...
      yeast_name = str(int(y.name))
    else:
      yeast_name = y.name.lower()
      # Some basic clean-up to start...
      if "conan" in yeast_name: yeast_name = "vermont ale"
      elif "super high gravity" in yeast_name or "wlp 099" in yeast_name: yeast_name = "super high gravity ale"
      elif "super yeast" in yeast_name: yeast_name = "san diego super"
      elif "chico" in yeast_name: yeast_name = "chico ale"
      elif "orval" in yeast_name: yeast_name = "brettanomyces bruxellensis"
      elif "duvel" in yeast_name: yeast_name = "belgian golden ale"
      elif "dupont" in yeast_name: yeast_name = "french saison ale"
      else:
          # Clean up any spelling mistakes, ordering
          yeast_name = yeast_name.replace("kรถlsch","kolsch").replace("kรถlsh","kolsch").replace("kölsh","kolsch")
          yeast_name = yeast_name.replace("monastery","monastary").replace("monestary","monastary")
          yeast_name = yeast_name.replace("california v ale", "california ale v")
          yeast_name = yeast_name.replace("mã©lange","melange").replace("brettâ€™","brett") \
              .replace("cã´te","cote").replace("munuch","munich").replace("lellemand","lallemand") \
              .replace("champagene", "champagne").replace("vemont","vermont")

    if yeast_name not in yeast_name_to_id:
      # First attempt: Try to find a yeast code to match
      s = re.search(r"(inis\-|wlp|us\-|([kwst]|oyl|bry)\-|[mg]|\d+/)\s?\-?(\d+(/\d+)?)", yeast_name, flags=re.IGNORECASE)
      if s != None and len(s.group()) > 0:
          # We have a potential yeast product id...
          product_id = s.group().replace("--","-").replace(" ","")
          if product_id in yeast_name_to_id:
              return yeast_name_to_id[product_id]

      # Try removing the word "yeast" (and watch out for any double spaces that may ensue)
      yeast_name_temp = clean_replace(yeast_name, "yeast")
      if yeast_name_temp in yeast_name_to_id:
          return yeast_name_to_id[yeast_name_temp]
      # Try replacing "yeast" with "ale" 
      yeast_name_temp = yeast_name.replace("yeast", "ale")
      if yeast_name.replace("yeast", "ale") in yeast_name_to_id:
        return yeast_name_to_id[yeast_name_temp]
      # ... or "lager"
      yeast_name_temp = yeast_name.replace("yeast", "lager")
      if yeast_name_temp in yeast_name_to_id:
        return yeast_name_to_id[yeast_name_temp]

      # Try adding "ale" or "blend" or "lager" to the end
      yeast_name_temp = yeast_name+" ale"
      if yeast_name_temp in yeast_name_to_id:
        return yeast_name_to_id[yeast_name_temp]
      yeast_name_temp = yeast_name+" lager"
      if yeast_name_temp in yeast_name_to_id:
        return yeast_name_to_id[yeast_name_temp]
      yeast_name_temp = yeast_name+" blend"
      if yeast_name_temp in yeast_name_to_id:
        return yeast_name_to_id[yeast_name_temp]

      # Check for a wyeast code (4 digit code)
      s = re.search(r"(\d{4})(\D+|$)", yeast_name, flags=re.IGNORECASE)
      if s != None and len(s.group(1)) > 0:
          product_id = s.group(1)
          if product_id in yeast_name_to_id:
              return yeast_name_to_id[product_id]

      # Try to match the brand name first and then try to figure out
      # which yeast we're dealing with within that brand 
      for brand_name, ids in brand_to_ids.items():
          s = re.search(brand_name, yeast_name, flags=re.IGNORECASE)
          if s != None and len(s.group()) > 0:
              # Find the best possible match
              best_match_count = 0
              best_match_id = -1
              for id in ids:
                  count = 0
                  potential_yeast_names = id_to_yeast_names[id]
                  for name in potential_yeast_names:
                      s = re.search(name, yeast_name, flags=re.IGNORECASE)
                      if s != None: count += 1
                  if count > best_match_count:
                      best_match_count = count
                      best_match_id = id
              if best_match_id == -1:
                  # Just choose the first id...
                  assert len(ids) > 0
                  best_match_id = ids[0]
                  yeast_name = id_to_yeast_names[best_match_id][0]
              break

      # Last attempt - try to match any of the ids directly with the yeast name string
      best_len = 0
      best_id = -1
      for id, dict_name in yeast_name_to_id.items():
          name_opts = f"({dict_name}|{clean_replace(dict_name, 'ale')}|{clean_replace(dict_name,'lager')})"
          s = re.search(name_opts, yeast_name)
          if s != None and len(s.group()) > 0:
              group_len = len(s.group())
              if best_len < group_len:
                  best_len = group_len
                  best_id = id

      if best_id != -1: return best_id

  return None