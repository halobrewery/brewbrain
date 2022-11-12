import csv

def create_style_dict():
  # Grab a listing of all styles
  style_name_to_id = {}
  with open("./data/_db/styles.csv", "r", encoding="utf-8") as f:
    style_csv_reader = csv.reader(f, delimiter=",")
    for i, row in enumerate(style_csv_reader):
      if i == 0: continue
      style_name_to_id[row[0].lower().strip()] = int(row[1])
  with open("./data/_db/alt_style_names.csv", "r", encoding="utf-8") as f:
    alt_name_csv_reader = csv.reader(f, delimiter=",")
    for row in alt_name_csv_reader:
      base_name = row[0].lower() # Base name for the style as defined in styles.csv
      assert base_name in style_name_to_id
      id = style_name_to_id[base_name]
      for row_idx in range(1,len(row)):
        style_name_to_id[row[row_idx].lower()] = id
  
  return style_name_to_id
