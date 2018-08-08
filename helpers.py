import csv
import math
import operator
import pickle
from random import randint
from random import shuffle
from collections import Counter

class Node:
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
        print(" "  * (self.depth  * 2), self.splitter, self.splitter_value, self.lowerpred, self.upperpred)
        if self.lowerchild:
            self.lowerchild.print()
        if self.upperchild:
            self.upperchild.print()

    def forward_propagate(self, data):
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
    return sum([(x[OUTPUT] - pred) ** 2 for x in split])

def get_total_split_loss(split_1, split_2, split_1_pred, split_2_pred):
    """Get total loss of a proposed split."""
    total_nos = len(split_1) + len(split_2)
    s1 = (len(split_1) / total_nos) * get_split_loss(split_1, split_1_pred)
    s2 = (len(split_2) / total_nos) * get_split_loss(split_2, split_2_pred)
    return s1 + s2

def lastIndex(lst, val):
    return len(lst) - 1 - lst[::-1].index(val)

def get_split(sorted_data, feature, value):
    i = lastIndex([d[feature] for d in sorted_data], value)
    split_1 = sorted_data[:i + 1]
    split_2 = sorted_data[i + 1:]
    return split_1, split_2

def get_best_splitter(sorted_data, feature):
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
    sortedLst = sorted(lst)
    return sortedLst[(len(lst) - 1) // 2]

def train(data, node):
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
    loss = validation_loss(valid_data, root)
    train_loss = validation_loss(train_data[:100], root)
    print("VALID", loss, " -- TRAIN ", train_loss, " -- ", best_splitter, node.splitter_value, len(split_1), len(split_2))

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

    print(node.depth, node.pruned, validation_loss(valid_data, root))

def validation_loss(data, tree):
    """Calculate loss over given data."""
    loss = sum([(row[OUTPUT] - (tree.forward_propagate(row) or 0)) ** 2 for row in data])
    return loss / len(data)

def forest_validation_loss(data, forest):
    """Calculate loss over given data from multiple roots."""
    loss = sum([(row[OUTPUT] - forest_propagate(row, forest)) ** 2 for row in data])
    return loss / len(data)

def forest_propagate(data, forest):
    return sum([tree.forward_propagate(data) or 0 for tree in forest]) / len(forest)

MODEL = 1
MIN_DEPTH = 3
MIN_LEAF = 2
DROPOUT = 0.3
NUM_TREES = 120
TRAIN = True
TEST = True
PRED_ID = 'Id'

if MODEL == 0:
    MAX_DEPTH = 15
    num_valid = 200
    OUTPUT = 'quality'
    fulldata = read_data('train.csv')
    TEST_CSV = 'test.csv'
elif MODEL == 1:
    MAX_DEPTH = 10
    num_valid = 150
    OUTPUT = 'output'
    fulldata = read_data('train1.csv')
    TEST_CSV = 'test1.csv'
else:
    num_valid = 2
    OUTPUT = 'Power(output)'
    fulldata = read_data('toy_dataset.csv')
    TEST_CSV = 'toytest.csv'

full_train_data = fulldata[num_valid:] * int(NUM_TREES * 0.7 + 1)
valid_data = fulldata[:num_valid]

train_data = []

root = Node()
roots = []

sect = len(full_train_data) // NUM_TREES

for i in range(0, NUM_TREES):
    print("Training tree #" + str(i + 1))
    root = Node()
    train_data = full_train_data[i * sect : (i+1) * sect]
    train(train_data, root)
    prune(valid_data, root)
    roots.append(root)

print(forest_validation_loss(valid_data, roots))

if TEST:
    test_data = read_data(TEST_CSV)
    with open('pred.csv', 'w') as file:
        file.write(PRED_ID + ',' + OUTPUT + '\n')
        for i, row in enumerate(test_data):
            file.write(str(i + 1) + ',' + str(forest_propagate(row, roots)) + '\n')

exit()

if TRAIN:
    train(train_data, root)
    prune(valid_data, root)
    pickle.dump(root, open('model.p', 'wb'))

if TEST:
    root = pickle.load(open('model.p', 'rb'))
    test_data = read_data(TEST_CSV)
    with open('pred.csv', 'w') as file:
        file.write(PRED_ID + ',' + OUTPUT + '\n')
        for i, row in enumerate(test_data):
            file.write(str(i + 1) + ',' + str(root.forward_propagate(row)) + '\n')
