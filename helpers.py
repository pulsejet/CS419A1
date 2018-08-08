import csv
import time
import math
import operator
import pickle
from random import randint
from collections import Counter

# Make this true to dump graph data
MAKE_GRAPH = False
GRAPH = []

class Node:
    """A single node in a tree."""
    def __init__(self):
        self.splitter = None
        self.splitter_value = None
        self.lowerchild = None
        self.upperchild = None
        self.lowerpred = None
        self.upperpred = None
        self.depth = 0
        self.pruned = False

    def print(self):
        """Print as readable."""
        print(" "  * (self.depth  * 2), self.splitter, self.splitter_value, self.lowerpred, self.upperpred)
        if self.lowerchild:
            self.lowerchild.print()
        if self.upperchild:
            self.upperchild.print()

    def forward_propagate(self, data):
        """Predict from a dict of features."""
        if not self.splitter or self.pruned:
            return None

        if data[self.splitter] > self.splitter_value:
            if not self.upperchild:
                return self.upperpred
            return self.upperchild.forward_propagate(data) or self.upperpred
        else:
            if not self.lowerchild:
                return self.lowerpred
            return self.lowerchild.forward_propagate(data) or self.lowerpred

    def count_nodes(self):
        """Count total number of children."""
        count = 0
        if self.lowerchild:
            count += self.lowerchild.count_nodes()
        else:
            count += 1
        if self.upperchild:
            count += self.upperchild.count_nodes()
        else:
            count += 1
        return count

def read_data(file):
    """Read a CSV file as list of dicts with float values."""
    return [dict([a, float(x)] for a, x in data.items()) for data in csv.DictReader(open(file, 'r'))]

def most_common(lst):
    """Find the most common element in list"""
    return Counter(lst).most_common(1)[0][0]

def get_sorted_data(data, feature):
    """Sort data with a feature."""
    return sorted(data, key=lambda k: k[feature])

def get_split_loss(split, pred):
    """Get loss from a split and prediction."""
    if LOSS == 'mse':
        return sum([(x[OUTPUT] - pred) ** 2 for x in split])
    elif LOSS == 'mae':
        return sum([abs(x[OUTPUT] - pred) for x in split])
    else:
        raise Exception("Unknown loss " + LOSS)

def get_total_split_loss(split_1, split_2, split_1_pred, split_2_pred):
    """Get total loss of a proposed split."""
    if LOSS_PROB:
        total_nos = len(split_1) + len(split_2)
        s1 = (len(split_1) / total_nos) * get_split_loss(split_1, split_1_pred)
        s2 = (len(split_2) / total_nos) * get_split_loss(split_2, split_2_pred)
    else:
        s1 = get_split_loss(split_1, split_1_pred)
        s2 = get_split_loss(split_2, split_2_pred)
    return s1 + s2

def lastIndex(lst, val):
    """Get last index in list."""
    return len(lst) - 1 - lst[::-1].index(val)

def get_split(sorted_data, feature, value):
    """Split given feature and value."""
    i = lastIndex([d[feature] for d in sorted_data], value)
    split_1 = sorted_data[:i + 1]
    split_2 = sorted_data[i + 1:]
    return split_1, split_2

def get_best_splitter(sorted_data, feature):
    """Get split with minimum loss for a given feature."""
    split_points = {}
    highest = sorted_data[-1][feature]
    for i in reversed(range(0, len(sorted_data))):
        # Get only for all split
        if sorted_data[i][feature] in split_points or sorted_data[i][feature] == highest:
            continue

        # Make splits
        split_1 = sorted_data[:i + 1]
        split_2 = sorted_data[i + 1:]

        # Get best prediction
        split_1_pred = most_common([d[OUTPUT] for d in split_1])
        split_2_pred = most_common([d[OUTPUT] for d in split_2])

        # Get loss
        loss = get_total_split_loss(split_1, split_2, split_1_pred, split_2_pred)
        split_points[sorted_data[i][feature]] = loss

    if not split_points:
        return None

    best_point = min(split_points, key=split_points.get)
    best_points = [x for x in split_points if split_points[x] == split_points[best_point]]
    best_point = median(best_points)

    return best_point, split_points[best_point]

def median(lst):
    """Get a rough median from a list of numbers."""
    sortedLst = sorted(lst)
    return sortedLst[(len(lst) - 1) // 2]

def train(data, node):
    """Train a node recursively."""
    # Check for excessively long branches
    if node.depth >= MAX_DEPTH:
        return

    # Get best splitter feature
    splitters = {}
    splitters_losses = {}

    # Iterate all keys in first row, assuming same keys
    for key in data[0]:
        # Skip OUTPUT
        if key == OUTPUT or randint(0, 10) < DROPOUT * 10:
            continue

        # Sort and get best splittere
        f = get_sorted_data(data, key)
        splitter = get_best_splitter(f, key)

        # Skip if split not possible
        if not splitter:
            continue

        # Get the splitter
        splitters[key], splitters_losses[key] = splitter

    # End of recursion tree
    if not splitters:
        return node

    # Get best splitter and splits
    best_splitter = min(splitters_losses.items(), key=operator.itemgetter(1))[0]
    f = get_sorted_data(data, best_splitter)
    split_1, split_2 = get_split(f, best_splitter, splitters[best_splitter])

    # Get most common classes for splits
    split_1_pred = most_common([d[OUTPUT] for d in split_1])
    split_2_pred = most_common([d[OUTPUT] for d in split_2])

    # Set node attributes
    node.splitter = best_splitter
    node.splitter_value = splitters[best_splitter]
    node.lowerpred = split_1_pred
    node.upperpred = split_2_pred

    # Validation
    if VERBOSE:
        loss = validation_loss(valid_data, root)
        train_loss = validation_loss(train_data[:100], root)
        print("VALID LOSS", loss, "\tTRAIN LOSS", train_loss, "\tNODES", root.count_nodes(), "\tSPLIT", best_splitter, node.splitter_value, len(split_1), len(split_2))

        # Write Graph Data
        if MAKE_GRAPH:
            GRAPH.append({
                "valid": loss,
                "train": train_loss,
                "nodes": root.count_nodes()
            })

    # Create children if have elements
    if len(split_1) >= MIN_LEAF:
        node.lowerchild = Node()
        node.lowerchild.depth = node.depth  + 1
        train(split_1, node.lowerchild)
    if len(split_2) >= MIN_LEAF:
        node.upperchild = Node()
        node.upperchild.depth = node.depth  + 1
        train(split_2, node.upperchild)

    return node

def prune(data, node):
    """Prune a node recursively."""
    if node.lowerchild and not node.lowerchild.pruned:
        prune(data, node.lowerchild)
    if node.upperchild and not node.upperchild.pruned:
        prune(data, node.upperchild)

    # Remove dead nodes
    if node.lowerchild and node.lowerchild.pruned:
        node.lowerchild = None
    if node.upperchild and node.upperchild.pruned:
        node.upperchild = None

    # Do not over-prune
    if node.depth <= MIN_DEPTH:
        return

    # Try pruning current node
    loss_pre_prune = validation_loss(data, root)
    node.pruned = True
    loss_post_prune = validation_loss(data, root)
    if loss_post_prune > loss_pre_prune:
        node.pruned = False

    if VERBOSE:
        print('DEPTH', node.depth, '\tPRUNED', node.pruned, "\tNODES", root.count_nodes(), '\tVALID LOSS', validation_loss(valid_data, root))

def validation_loss(data, tree):
    """Calculate loss over given data."""
    loss = sum([(row[OUTPUT] - (tree.forward_propagate(row) or 0)) ** 2 for row in data])
    return loss / len(data)

def forest_validation_loss(data, forest):
    """Get loss from a random forest (list of roots)."""
    loss = sum([(row[OUTPUT] - forest_propagate(row, forest)) ** 2 for row in data])
    return loss / len(data)

def forest_propagate(data, forest):
    """Get output from a random forest (list of roots)."""
    return sum([tree.forward_propagate(data) or 0 for tree in forest]) / len(forest)

MAX_DEPTH = None
MIN_DEPTH = None
MIN_LEAF = None
DROPOUT = None
NUM_TREES = None
VERBOSE = True
PRED_ID = 'Id'

LOSS = None
LOSS_PROB = None
num_valid = None
OUTPUT = None
train_data = None
valid_data = None
root = None

def train_tree(train_file, out_model, output='predicted.csv',
               max_depth=15, min_depth=1, numvalid=200, dropout=0, min_leaf=2,
               loss_prob=True, loss='mse', verbose=False):
    """Front facing API to train a single tree."""
    global MAX_DEPTH
    global MIN_DEPTH
    global num_valid
    global OUTPUT
    global train_data
    global valid_data
    global root
    global VERBOSE
    global DROPOUT
    global MIN_LEAF
    global LOSS_PROB
    global LOSS

    # Never show output
    VERBOSE = verbose

    # Setup variables
    MIN_DEPTH = min_depth
    MAX_DEPTH = max_depth
    num_valid = numvalid
    OUTPUT = output
    fulldata = read_data(train_file)
    DROPOUT = dropout
    MIN_LEAF = min_leaf
    LOSS_PROB = loss_prob
    LOSS = loss

    # Setup training data
    train_data = fulldata[num_valid:]
    valid_data = fulldata[:num_valid]

    # Train and prune
    start = time.time()
    root = Node()
    train(train_data, root)
    prune(valid_data, root)
    end = time.time()

    # Print time elapsed
    print("Grew and pruned tree in", str(end - start) + "s")
    print("Final validation loss", validation_loss(valid_data, root))

    # Dump graph
    if MAKE_GRAPH:
        with open('graph.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, GRAPH[0].keys())
            w.writeheader()
            w.writerows(GRAPH)

    # Pickle model
    pickle.dump(root, open(out_model, 'wb'))

def train_forest(train_file, out_model, output='predicted.csv',
                 max_depth=15, min_depth=1, numvalid=200, dropout=0.2,
                 num_trees=4, min_leaf=2, loss_prob=False, loss='mse',
                 verbose=False):
    """Front facing API to train a random forest."""

    global MAX_DEPTH
    global MIN_DEPTH
    global num_valid
    global OUTPUT
    global train_data
    global valid_data
    global root
    global roots
    global VERBOSE
    global DROPOUT
    global NUM_TREES
    global MIN_LEAF
    global LOSS_PROB
    global LOSS

    # Don't Show output for forest
    VERBOSE = verbose

    # Setup variables
    MIN_DEPTH = min_depth
    MAX_DEPTH = max_depth
    num_valid = numvalid
    OUTPUT = output
    fulldata = read_data(train_file)
    DROPOUT = dropout
    NUM_TREES = num_trees
    MIN_LEAF = min_leaf
    LOSS_PROB = loss_prob
    LOSS = loss

    # Setup training data
    full_train_data = fulldata[num_valid:] * int(NUM_TREES * 0.7 + 1)
    valid_data = fulldata[:num_valid]
    train_data = []

    # Start with barren land
    roots = []

    # Make sections in train data
    sect = len(full_train_data) // NUM_TREES

    # Grow all trees
    for i in range(0, NUM_TREES):
        root = Node()
        roots.append(root)
        train_data = full_train_data[i * sect : (i+1) * sect]
        start = time.time()
        train(train_data, root)
        prune(valid_data, root)
        end = time.time()
        print("Trained tree #" + str(i + 1) + " in " + str(end-start) + "s", "\tVALID LOSS: ", forest_validation_loss(valid_data, roots))

    # Pickle model
    pickle.dump(roots, open(out_model, 'wb'))

def predict(file_model, test_file, out_file, output):
    """Predict on test data from a trained model."""
    # Load model
    model = pickle.load(open(file_model, 'rb'))

    # Read input test data
    test_data = read_data(test_file)

    # Predict
    start = time.time()
    with open(out_file, 'w') as file:
        file.write(PRED_ID + ',' + output + '\n')
        for i, row in enumerate(test_data):
            if type(model) == Node:
                file.write(str(i + 1) + ',' + str(model.forward_propagate(row)) + '\n')
            else:
                file.write(str(i + 1) + ',' + str(forest_propagate(row, model)) + '\n')

    end = time.time()

    # Print inference time
    print("Inferred predicted values in", str(end - start) + "s")
    print("Average inference time per sample", str((end - start) / len(test_data)) + "s")
