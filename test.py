import os
from pybeerxml.parser import Parser

path_to_beerxml_file = "./data/1/231868.xml"

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

if __name__ == "__main__":
  main()