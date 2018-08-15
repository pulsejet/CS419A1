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
    opts, args = getopt.getopt(sys.argv[1:], 'd:t:fmavl:s', ['train_data=', 'test_data=', 'forest', 'mean_squared', 'absolute', 'verbose', 'min_leaf_size=', 'std'])
except getopt.GetoptError as err:
    print(err)
    sys.exit(2)

# Initialize Defaults
FILE = 'toy_dataset.csv'
TEST_FILE = 'test.csv'
FOREST = False
VERBOSE = False
LOSS = 'mse'
MIN_LEAF = 2

# Set arguments
for o, a in opts:
    if o in ("-d", "--train_data"):
        FILE = a
    elif o in ("-t", "--test_data"):
        TEST_FILE = a
    elif o in ("-f", "--forest"):
        FOREST = True
    elif o in ("-a", "--absolute"):
        LOSS = 'mae'
    elif o in ("-m", "--mean_squared"):
        LOSS = 'mse'
    elif o in ("-s", "--std"):
        LOSS = 'std'
    elif o in ("-v", "--verbose"):
        VERBOSE = not VERBOSE
    elif o in ("-l", "--min_leaf_size"):
        MIN_LEAF = int(a)
    else:
        assert False, "Unhandled option " + o

if not FOREST:
    train_tree(FILE, 'model', output='Power(output)', numvalid=2,
               loss=LOSS, min_leaf=MIN_LEAF, max_depth=15, min_depth=3,
               verbose=VERBOSE)
else:
    train_forest(FILE, 'model', output='Power(output)', numvalid=2,
                 loss=LOSS, min_leaf=MIN_LEAF, num_trees=128, dropout=0.2,
                 max_depth=15, min_depth=3, verbose=VERBOSE)

print("No test data supplied, so not predicting")
