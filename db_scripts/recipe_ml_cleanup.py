from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy import select
from sqlalchemy import or_, and_

from brewbrain_db import BREWBRAIN_DB_ENGINE_STR, Base, RecipeML

def remove_zero_mash_or_ferment_step_recipes(session):
  bad_recipes = session.query(RecipeML).filter(or_(RecipeML.num_mash_steps == 0, RecipeML.num_ferment_stages == 0)).all()
  for recipe in bad_recipes:
    session.delete(recipe)

if __name__ == "__main__":
  engine = create_engine(BREWBRAIN_DB_ENGINE_STR, echo=True, future=True)
  Base.metadata.create_all(engine)

  with Session(engine) as session:
    remove_zero_mash_or_ferment_step_recipes(session)
    session.commit()