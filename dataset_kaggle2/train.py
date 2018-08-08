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
    opts, args = getopt.getopt(sys.argv[1:], 'd:fmav', ['data_file=', 'forest', 'mean_squared', 'absolute', 'verbose'])
except getopt.GetoptError as err:
    print(err)
    sys.exit(2)

# Initialize Defaults
FILE = 'train.csv'
FOREST = False
VERBOSE = not FOREST
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
    elif o in ("-v", "--verbose"):
        VERBOSE = True
    else:
        assert False, "Unhandled option " + o

# Train
if not FOREST:
    train_tree(FILE, 'model', output='quality', numvalid=200,
               loss=LOSS, min_leaf=2, max_depth=15, min_depth=3, loss_prob=True,
               verbose=VERBOSE)
else:
    train_forest(FILE, 'model', output='quality', numvalid=200,
                 loss=LOSS, min_leaf=2, num_trees=32, dropout=0.2,
                 max_depth=15, min_depth=3, loss_prob=True, verbose=VERBOSE)
