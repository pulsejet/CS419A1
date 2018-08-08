"""Generate model"""

# Add parent directory to PATH
# https://stackoverflow.com/a/11158224
import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

import getopt
from helpers import train_tree
from helpers import train_forest

# Get command line arguments
try:
    opts, args = getopt.getopt(sys.argv[1:], 'd:fma', ['data_file=', 'forest', 'mean_squared', 'absolute'])
except getopt.GetoptError as err:
    print(err)
    sys.exit(2)

# Initialize Defaults
FILE = 'toy_dataset.csv'
FOREST = False
LOSS = 'mse'

# Set arguments
for o, a in opts:
    if o in ("-d", "--data_file"):
        FILE = a
    elif o in ("-f", "--forest"):
        FOREST = True
    elif o in ("-a", "--absolute"):
        LOSS = 'mae'
    elif o in ("-m", "--mean_squared"):
        LOSS = 'mse'
    else:
        assert False, "Unhandled option " + o

if not FOREST:
    train_tree(FILE, 'model', output='output', numvalid=2,
               loss=LOSS, min_leaf=2, max_depth=15, min_depth=3)
else:
    train_forest(FILE, 'model', output='output', numvalid=2,
                 loss=LOSS, min_leaf=2, num_trees=128, dropout=0.2,
                 max_depth=15, min_depth=3)
