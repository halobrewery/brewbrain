import csv
import re

def build_hop_name_dicts():
  # Create a hop name-to-id system
  hop_name_to_id  = {}
  id_to_hop_names = {}
  with open("./data/_db/hops.csv", "r", encoding="utf-8") as f:
      csv_reader = csv.reader(f, delimiter=";")
      for i,row in enumerate(csv_reader):
          if i == 0: continue
          names = []
          hop_name = row[0].lower().strip()
          hop_name_to_id[hop_name] = i
          names.append(hop_name)
          
          for alt_name in row[5].split(",") + row[6].split(","):
              alt_name = alt_name.strip()
              if len(alt_name) > 0:
                  alt_name = alt_name.lower()
                  hop_name_to_id[alt_name] = i
                  names.append(alt_name)
          assert len(names) > 0
          id_to_hop_names[i] = names

  return hop_name_to_id, id_to_hop_names


def match_hop_id(hop_name, hop_name_to_id):
  h_name = str(hop_name).lower().strip()
  init_name = h_name

  h_name = re.sub(r"(\,|us|u\.s\.)", "", h_name)
  h_name = re.sub(r"(uk kent(\s*goldings?)?|kent holdings)", "east kent goldings", h_name)
  h_name = re.sub(r"(\(?(\d+\.?\d*%\s*aa|™|â„¢|t\-?90|\s*organic|yakima(\s*valley\s?\-?)?|ychhops|,|bmw|lambic|frech)\)?)", "", h_name)
  h_name = re.sub(r"(\s{2,})", " ", h_name) # Clean up extra spacing
  h_name = re.sub(r"(mittelfr端h|mittelfrã¼h|mittelfruh|mittlefruh)", r"mittlefrüh", h_name, flags=re.IGNORECASE)
  h_name = re.sub(r"\(\d\d\d\d\)", "", h_name) # Remove 4 digit years in brackets e.g, "(2015)"
  h_name = h_name.strip()
  
  if h_name in hop_name_to_id:
    return hop_name_to_id[h_name]
  
  best_len_s1 = 0
  best_id_s1  = -1
  best_len_s2 = 0
  best_id_s2  = -1
  for dict_name, dict_id in hop_name_to_id.items():
    name_opts = "(" + dict_name + ")"
    s1 = re.search(name_opts, h_name)
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
    if best_len_s2 == 0: return None
    else: return best_id_s2
  else: return best_id_s1