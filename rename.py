from glob import glob
import os

maps = glob("./maps/*.svg")
for map in maps:
    os.rename(map, map.lower())
