import csv
import re

def build_style_dicts():
  # Grab a listing of all styles
  style_name_to_id = {}
  id_to_style_names = {}
  with open("./data/_db/styles.csv", "r", encoding="utf-8") as f:
    style_csv_reader = csv.reader(f, delimiter=",")
    for i, row in enumerate(style_csv_reader):
      if i == 0: continue
      style_name = row[0].lower().strip()
      id = int(row[1])
      style_name_to_id[style_name] = id
      id_to_style_names[id] = [style_name]
  with open("./data/_db/alt_style_names.csv", "r", encoding="utf-8") as f:
    alt_name_csv_reader = csv.reader(f, delimiter=";")
    for row in alt_name_csv_reader:
      base_name = row[0].lower().strip() # Base name for the style as defined in styles.csv
      assert base_name in style_name_to_id
      id = style_name_to_id[base_name]
      for row_idx in range(1,len(row)):
        style_name = row[row_idx].lower().strip()
        style_name_to_id[style_name] = id
        id_to_style_names[id].append(style_name)

  return style_name_to_id, id_to_style_names


def match_style_id(style_name, style_name_to_id):
  s_name = str(style_name).lower().strip()
  
  if re.search("wee\s*heavy", s_name) != None:
    s_name = "wee heavy"
  
  if s_name in style_name_to_id: return style_name_to_id[s_name]

  best_len_s1 = 0
  best_id_s1  = -1
  for dict_name, dict_id in style_name_to_id.items():
    name_opts = "(" + dict_name + ")"
    s1 = re.search(name_opts, s_name)
    if s1 != None and len(s1.group()) > 0:
      group_len = len(s1.group())
      if best_len_s1 < group_len:
        best_len_s1 = group_len
        best_id_s1  = dict_id

  if best_len_s1 != 0: best_id_s1
  return None