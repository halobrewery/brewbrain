import os
import argparse
import csv
import re
from urllib.request import Request, urlopen

def main(args):
  
  with open(args.csv, "r", encoding="ISO-8859-1") as f:
    reader = csv.reader(f, delimiter=",")
    
    # Read through the csv and find the identifiers for each of the recipes
    for i, line_array in enumerate(reader):
      if i == 0: continue
      
      style_id = line_array[4]
      url_path = line_array[2]
      recipe_id = re.match(r"/homebrew/recipe/view/(\d+)/", url_path).group(1)
      
      # Create a directory for the style id
      xml_dirpath = os.path.join('data', style_id)
      os.makedirs(xml_dirpath, exist_ok=True)
      
      #recipe_name = line_array[1]
      xml_filepath = os.path.join(xml_dirpath, recipe_id + ".xml")
      if os.path.exists(xml_filepath): continue
      
      # Open the webpage for d/l of the beerxml for each recipe
      # via https://www.brewersfriend.com/homebrew/recipe/beerxml1.0/<recipe_id>
      req = Request(
        url="https://www.brewersfriend.com/homebrew/recipe/beerxml1.0/" + recipe_id, 
        headers={'User-Agent': 'Mozilla/5.0'}
      )
      xml_file = urlopen(req)
      with open(xml_filepath, 'wb') as output:
        output.write(xml_file.read())

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Auto-download beerxml files off brewersfriend sites.")
  parser.add_argument('-csv', '--csv', type=str, default="data/recipeData.csv", help="The csv filepath containing all recipe data to scour.")
  args = parser.parse_args()
  main(args)