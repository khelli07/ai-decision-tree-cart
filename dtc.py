import numpy as np
import pandas as pd

def unique_vals(array):
    return set(array)

def class_count(y_data):
    count = {}
    # Initialize table
    for class_ in unique_vals(y_data):
        count[class_] = 0
    # Fill table
    for class_ in y_data:
        count[class_] += 1

    return count

def class_probability(y_data):
    probability = class_count(y_data)
    # Total labels
    total = sum([ctr for _, ctr in probability.items()])

    # Overwrite counter with probability
    for class_, ctr in probability.items():
        probability[class_] = ctr / total

    return probability

def is_col_numeric(types, col):
    type_ = str(types[col])
    return ('int' in type_) or ('float' in type_)

def partition(data, question):
    true_data = []
    false_data =[]

    for row in data:
        if (question.match(row)):
            true_data.append(row)
        else:
            false_data.append(row)

    return np.array(true_data), np.array(false_data)

def gini_impurity(y_data):
    count_db = class_count(y_data)
    total_count = float(len(y_data))
    gini = 1 
    
    for _, ctr in count_db.items():
        gini -= (ctr / total_count)**2

    return gini

def info_gain(true, false, current_uncertainty):
    lt = float(len(true)); lf = float(len(false))
    frac = lt / (lt + lf)
    avg_gini = frac * gini_impurity(true) + (1 - frac) * gini_impurity(false)
    
    return current_uncertainty - avg_gini

def get_best_split(header, data, types):
    best_gain = 0
    best_question = None 
    current_uncertainty = gini_impurity(data[:, -1])

    n_features = len(data[0, :-1])
    for col in range(n_features):
        values = unique_vals(data[:, col])

        for val in values:
            question = Question(header, col, val, types)
            true_data, false_data = partition(data, question)

            # If no information is gain, skip
            if len(true_data) == 0 or len(false_data) == 0:
                continue

            gain = info_gain(true_data[:, -1], false_data[:, -1],
            current_uncertainty)

            if gain >= best_gain:
                best_gain, best_question = gain, question

    return best_gain, best_question

def get_best_label(prob_dict):
    label = None
    probability = -1

    for key, val in prob_dict.items():
        if (val > probability):
            probability = val
            label = key

    return label

class Question:
    def __init__(self, header, column, value, types):
        self.name = header[column]
        self.column = column
        self.value = value
        self.isnumeric = is_col_numeric(types, column)

    def match(self, row):
        val = row[self.column]
        if self.isnumeric:
            return val >= self.value
        else:
            return val == self.value
        
    def __repr__(self):
        if self.isnumeric:
            return f"Is {self.name} >= {self.value}?"
        else:
            return f"Is {self.name} == {self.value}?"

class Leaf:
    def __init__(self, data):
        self.probability = class_probability(data[:, -1])

    def __repr__(self):
        txt = "{"
        for key, val in self.probability.items():
            txt += f"{key}: {val}, "
        txt = txt[:-2] + "}"
        return txt

class DecisionNode:
    def __init__(self, question, gain, true_branch, false_branch, types):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch
        self.types = types
        self.info_gain = gain

    def classify(self, node, X_data):
        features = np.array(X_data)
        if isinstance(node, Leaf):
            return node.probability
        
        else:
            if node.question.match(features):
                return self.classify(node.true_branch, features)
            else:
                return self.classify(node.false_branch, features)
                
    def print_node(self, indent):
        spacing = ' ' * indent
        print(spacing + str(self.question))
        print(spacing + f"Information gained: {self.info_gain}")
        print(spacing + "|=== TRUE ===|")
        if not(isinstance(self.true_branch, Leaf)):
            self.true_branch.print_node(indent + 4)
        else:
            print(spacing, end="")
            print(self.true_branch)

        print(' ' * indent + "|=== FALSE ===|")
        if not(isinstance(self.false_branch, Leaf)):
            self.false_branch.print_node(indent + 4)
        else:
            print(spacing, end="")
            print(self.false_branch)

class DecisionTreeClassifier:
    def __init__(self, max_depth=None, ig_toll=0.0):
        self.root = None
        self.depth = None
        self.col_types = None

        # Pre-pruning feature
        self.ig_toll = ig_toll
        self.max_depth = max_depth
                
    def grow_tree(self, data, header, current_depth):
        gain, question = get_best_split(header, data, self.col_types)
        self.depth = current_depth

        if gain <= self.ig_toll or ((self.max_depth) and current_depth >= self.max_depth):
            return Leaf(data)

        true_data, false_data = partition(data, question)

        true_branch = self.grow_tree(true_data, header, current_depth + 1)
        false_branch = self.grow_tree(false_data, header, current_depth + 1)

        return DecisionNode(question, gain, true_branch, false_branch, self.col_types)

    def fit(self, X_data, y_data):
        features = np.array(X_data)
        target = np.array([y_data]).T
        data = np.concatenate((features, target), axis=1)

        header = [name for name in X_data.columns]
        header.append(y_data.name)
        
        self.col_types = [val for _, val in X_data.dtypes.items()][:-1]
        self.col_types.append(y_data.dtype)
        
        self.root = self.grow_tree(data, header, 0)

    def predict(self, X_data, key=None):
        y_pred = []
        rows = len(X_data)

        for i in range(rows):
            prob_dict = self.root.classify(self.root, X_data.iloc[i])
            if key:
                if prob_dict.get(key):
                    y_pred.append(prob_dict[key])
                else:
                    y_pred.append(0)
            else:
                y_pred.append(get_best_label(prob_dict))

        if key:
            return np.array(y_pred, dtype='float32')
        else:
            return np.array(y_pred, dtype='int32')

    def print_tree(self):
        self.root.print_node(0)