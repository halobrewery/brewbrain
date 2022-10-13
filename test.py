import os
from pybeerxml.parser import Parser

path_to_beerxml_file = "./data/SierraNevadaPaleAleClone.xml"

def main():
  parser = Parser()
  recipes = parser.parse(path_to_beerxml_file)

  for recipe in recipes:
    # some general recipe properties
    print(recipe.name)
    print(recipe.brewer)

    # calculated properties
    print(recipe.og)
    print(recipe.fg)
    print(recipe.ibu)
    print(recipe.abv)

    # iterate over the ingredients
    for hop in recipe.hops:
      print(hop.name)

    for fermentable in recipe.fermentables:
      print(fermentable.name)
      

    for yeast in recipe.yeasts:
      print(yeast.name)
        
    for misc in recipe.miscs:
      print(misc.name)

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
  
if __name__ == "__main__":
  clean_data()