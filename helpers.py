import numpy as np
import csv
import pprint
import math
import operator
import pickle

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
    """Find the most common element in list
    Adapted from https://stackoverflow.com/a/1518632"""
    return max(set(lst), key=lst.count)

def get_sorted_data(data, feature):
    """Sort data with a feature."""
    return sorted(data, key=lambda k: k[feature])

def get_split_loss(split, pred):
    loss = 0
    for x in split:
        loss += (x[OUTPUT] - pred) ** 2
    return loss

def get_total_split_loss(split_1, split_2, split_1_pred, split_2_pred):
    s1 = get_split_loss(split_1, split_1_pred)
    s2 = get_split_loss(split_2, split_2_pred)
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
    best_point = min(split_points.items(), key=operator.itemgetter(1))[0]
    return best_point, split_points[best_point]


def train(data, node):
    # Get best splitter feature
    splitters = {}
    splitters_losses = {}

    # Iterate all keys in first row, assuming same keys
    for key in data[0]:
        # Skip OUTPUT
        if key == OUTPUT:
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
    if len(split_1) > 0:
        node.lowerchild = Node()
        node.lowerchild.depth = node.depth  + 1
        train(split_1, node.lowerchild)
    if len(split_2) > 0:
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

    # Try pruning current node
    loss_pre_prune = validation_loss(data, root)
    node.pruned = True
    loss_post_prune = validation_loss(data, root)
    if loss_post_prune > loss_pre_prune:
        node.pruned = False
    print(node.depth, node.pruned, loss_pre_prune, loss_post_prune)

def validation_loss(data, node):
    """Calculate loss over given data."""
    loss = sum([(row[OUTPUT] - (node.forward_propagate(row) or 0)) ** 2 for row in data])
    return math.sqrt(loss)

MODEL = 0
TRAIN = True
PRUNE = True
CHECK_TRAIN_SANITY = False

if MODEL == 0:
    num_valid = 400
    OUTPUT = 'quality'
    fulldata = read_data('train.csv')
else:
    num_valid = 2
    OUTPUT = 'Power(output)'
    fulldata = read_data('toy_dataset.csv')

train_data = fulldata[num_valid:]
valid_data = fulldata[:num_valid]

root = Node()

if TRAIN:
    train(train_data, root)
    pickle.dump(root, open('model.p', 'wb'))

if PRUNE:
    root = pickle.load(open('model.p', 'rb'))
    prune(valid_data, root)
    pickle.dump(root, open('model_pruned.p', 'wb'))

if CHECK_TRAIN_SANITY:
    m = pickle.load(open('model.p', 'rb'))
    i = 0
    for d in train_data:
        if d[OUTPUT] != m.forward_propagate(d):
            print(d, m.forward_propagate(d))
            for i, f in enumerate(train_data):
                match = True
                for key in train_data[0]:
                    if key== OUTPUT:
                        continue
                    if f[key] != d[key]:
                        match = False
                if match:
                    print(i)
