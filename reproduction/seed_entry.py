"""Seed model initialization as well as dataset sampling before upstream code."""
import os
import runpy
import sys
from pathlib import Path
import lightning as pl

pl.seed_everything(int(os.environ['LEWM_SEED']), workers=True)
script = Path(sys.argv[1]).resolve()
sys.argv = [str(script), *sys.argv[2:]]
sys.path.insert(0, str(script.parent))
runpy.run_path(str(script), run_name='__main__')
