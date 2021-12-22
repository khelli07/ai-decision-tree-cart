import pandas as pd
import numpy as np

class DecisionTreeClassifier:
    def __init__(self, X_data, y_data, max_depth=15):
        X_data = np.array(X_data)
        y_data = np.array(y_data)
        # To name the child and the root
        # If used will be popped with X_data
        self.original_col = [i for i in range(len(X_data[0, :]))]
        self.ft_types = [val for _, val in X_data.dtypes.items()][:-1]

        self.max_depth = max_depth
        self.tree = None

    def fit(self, X_data, y_data):
        newRoot = TreeNode(None, None, None, None, None, self.original_col, 0)
        self.tree = newRoot.grow_tree(X_data, y_data, self.max_depth, True)

    def plot_tree(self):
        pass

    def predict(y_input):
        pass

class StopNode:
    def __init__(self):
        self.info = 'STOP'

class TreeNode:
    def __init__(self, 
        ft_types, 
        parent_ft_col, 
        parent_gini, 
        parent_threshold, 
        parent_category,
        remaining_col,
        depth):
    
        self.ft_col = parent_ft_col
        self.remaining_col = remaining_col
        self.types = ft_types[parent_ft_col]

        # Node info
        self.threshold = parent_threshold
        self.category = parent_category
        self.w_gini = parent_gini
        self.leaf = False
        self.depth = depth
        
        self.children = None

    def get_best_split(self, X_data, y_data, parent_gini): # nggak mengubah node kalo udah bagus
        best_col = None
        best_wgini = parent_gini; gini_list = None
        col_threshold = None; col_categories = None
        changeRoot = False

        for i in self.remaining_col:
            feature = X_data[:, i]
            target = y_data
            col_type = self.types[i]
            
            if (col_type != 'O'):
                w_gini, threshold, gl = self.get_weighed_gini(feature, target, col_type)
            else:
                w_gini, cats, gl = self.get_weighed_gini(feature, target, col_type)

            if (w_gini < best_wgini):
                best_col = i; changeRoot = True
                best_wgini = w_gini; gini_list = gl
                if (col_type != 'O'):
                    col_threshold = threshold
                else:
                    col_categories = cats
        
        if (changeRoot) or (len(self.remaining_col) > 0):
            self.ft_col = best_col
            self.types = self.types[best_col]
            self.w_gini = best_wgini
            pass_col = [k for k in self.remaining_col if k != i]

            if (self.types != 'O'):
                self.threshold = col_threshold
                left_child = TreeNode(self.types, self.ft_col, gini_list[0], self.threshold, 
                                None, pass_col, self.depth + 1)
                right_child = TreeNode(self.types, self.ft_col, gini_list[1], self.threshold, 
                                None, pass_col, self.depth + 1)
                
                self.children = [left_child, right_child]

            elif (self.type == 'O'):
                self.children = []
                for i in range(len(col_categories)):
                    child = TreeNode(self.types, self.ft_col, gini_list[i], None,
                                col_categories[i], pass_col, self.depth + 1)
                    self.children.append(child)
        else:
            self.leaf = True # kalau mentok
                    
    def grow_tree(self, X_data, y_data, max_depth, first_root=False):
        if not(self.leaf) and (self.depth < max_depth):
            if (first_root):
                self.get_best_split(X_data, y_data, 1.1)
            else:
                self.get_best_split(X_data, y_data, self.w_gini)

            for child in self.children:
                child.grow_tree(X_data, y_data, max_depth)

    @staticmethod
    def get_gini_impurity(y0_count, y1_count):
        ycount = y0_count + y1_count
        gini = 1 - (y0_count/ycount)**2 - (y1_count/ycount)**2
        
        return ycount, gini

    def get_wgini_numeric(self, feature, target):
        sorted_idx = np.argsort(feature)
        feature = feature[sorted_idx]
        target = target[sorted_idx]

        iteration = len(sorted_idx) - 1
        best_wgini = 1.1
        best_threshold = None
        for i in range(iteration):
            threshold = (feature[i] + feature[i + 1]) / 2
            lower_than = target[:i + 1]
            greater_than = target[i + 1:]

            # Left gini impurity
            y0_left = len(np.where(lower_than == 0)[0])
            y1_left = len(np.where(lower_than == 1)[0])
            sum_left, left_gini = self.get_gini_impurity(y0_left, y1_left)

            # Right gini impurity
            y0_right = len(np.where(greater_than == 0)[0])
            y1_right = len(np.where(greater_than == 1)[0])
            sum_right, right_gini = self.get_gini_impurity(y0_right, y1_right)

            # Weighed gini
            total = sum_left + sum_right
            w_gini = (sum_left/total)*left_gini + (right_gini/total)*right_gini

            # Pick the least weighed gini
            if (w_gini < best_wgini):
                best_wgini = w_gini
                best_threshold = threshold

        return best_wgini, best_threshold, [left_gini, right_gini]

    def get_wgini_cat(self, feature, target):
        cats = set(feature)
        count_cat = []
        gini_cat = []

        for i in cats:
            target_tmp = target[np.where(feature == i)[0]]
            y0_count = len(np.where(target_tmp == 0)[0])
            y1_count = len(np.where(target_tmp == 1)[0])
            ycount, gini = self.get_gini_impurity(y0_count, y1_count)

            count_cat.append(ycount)
            gini_cat.append(gini)

        # Weighed gini is alone, must be the best
        total = sum(count_cat)
        best_wgini = 0
        for i in range(len(gini_cat)):
            best_wgini += (count_cat[i]/total) * gini_cat[i]

        return best_wgini, cats, gini_cat

    def get_weighed_gini(self, feature, target, type):
        ft_type = str(type)

        if ('int' in ft_type) or ('float' in ft_type):
            return self.get_wgini_numeric(feature, target)

        elif (ft_type == 'O'):
            return self.get_wgini_cat(feature, target)
        
        else:
            raise TypeError("Only integers, floats, and objects are allowed.")

    def plot_node(self, indent): # print out
        pass