import os
import shutil
import csv
from pybeerxml.parser import Parser
from style_db import create_style_dict

def clean_data():
  parser = Parser()
  for dirpath, dirnames, files in os.walk('./data'):
    #print(dirpath)
    for file_name in files:
      file_path = os.path.join(dirpath, file_name)
      if not file_path.endswith((".xml", ".beerxml")):
        continue
      try:
        recipes = parser.parse(file_path)
        
        # Make sure we're dealing with an all-grain recipe
        recipe_type = recipes[0].type.lower()
        if recipe_type == "extract" or "partial" in recipe_type:
          print("Removing extract/partial mash recipe: " + file_path)
          os.remove(file_path)
        
      except:
        print("Erroneous file found, removing: " + file_path)
        os.remove(file_path)
      #print(file_path)
  
def copy_xml_files(base_dirpath, rename_files=False, move_files=False):
  parser = Parser()
  style_name_to_id = create_style_dict()

  #style_names = set()
  id = 0
  def gen_filename(id):
    return f"bt_{id}.xml"

  for dirpath, dirnames, files in os.walk(base_dirpath):
    for file_name in files:
      file_path = os.path.join(dirpath, file_name)
      if not file_path.endswith((".xml", ".beerxml")): continue

      try:
        recipes = parser.parse(file_path)
      except Exception as e:
        print("Erroneous file found: " + str(e))
        print("Removing: " + file_path)
        os.remove(file_path)
        continue
      
      for recipe in recipes:
        recipe_type = recipe.type.lower()
        if recipe_type == "extract" or "partial" in recipe_type:
          print("Removing extract/partial mash recipe: " + file_path)
          os.remove(file_path)
          continue
        
        # Attempt to match the style
        style_name = recipe.style.name.lower().replace(',', '')
        if style_name in style_name_to_id:
          # Found our match, copy the file to the correct directory
          cp_dirpath = os.path.join("./data", style_name_to_id[style_name])
        else:
          # Couldn't find a matching style, copy the file to a temp directory and deal with it later, lol.
          cp_dirpath = os.path.join("./data", "_not_matched")

        if cp_dirpath == dirpath:
          print(f"No change for file {file_path}, skipping.")
          continue

        if not os.path.exists(cp_dirpath):
          os.makedirs(cp_dirpath, exist_ok=False)
        cp_filepath = os.path.join(cp_dirpath, gen_filename(id)) if rename_files else os.path.join(cp_dirpath, file_name)
        if move_files:
          shutil.move(file_path, cp_filepath)
        else:
          shutil.copy(file_path, cp_filepath)
        id += 1

    mv_or_cp_str = "moved" if move_files else "copied"
    print(f"Total files {mv_or_cp_str}: {id}")


if __name__ == "__main__":
  copy_xml_files('./data/_not_matched', move_files=True)