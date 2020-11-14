import copy
import numpy as np
import pandas as pd
from collections import Counter
from utils import getEnt, getInfoGainRatio, getGini
import plotTree

class Node():
    def __init__(self, is_leaf, classification, attr_split, attr_split_value,
                 parent, depth, children=None, samples=None,gt=None):
        self.is_leaf = is_leaf                      # 是否为叶节点
        self.classification = classification        # 多数类
        self.attr_split = attr_split                # 分类属性
        self.attr_split_value = attr_split_value    # 分类准则的值
        self.parent = parent                        # 父节点
        self.children = copy.deepcopy(children)     # 子节点
        self.depth = depth                          # 节点深度
        self.samples = samples                      # 包含样本数量
        self.gt = gt                                # 用于cart树剪枝

class DecisionTreeClassifier:
    def __init__(self, type, epsilon=1e-6):
        self.type = type            # C4.5, CART
        self.epsilon = epsilon      # C4.5: info_gain_ratio threshold
        self.tree = None            # root
        self.dict = {}              # dict form

    def get_num_leaves(self, node):
        if node.is_leaf:
            return 1
        n_leaves = 0
        for attr, child_node in node.children.items():
            n_leaves += self.get_num_leaves(child_node)
        return n_leaves

    def get_emp_ent(self, node, data):
        split_list = []
        while node.parent != None:
            parent_node = node.parent
            for k, v in parent_node.children.items():
                if v == node:
                    split_list.append([parent_node.attr_split, k])
            node = parent_node

        while len(split_list) > 0:
            attr, val = split_list[-1]
            data = data[data[attr] == val]
            split_list.pop()
        return getEnt(data['class'])

    def get_C45_best_feature(self, data):
        n_features = data.shape[1] - 1
        best_feature = -1
        best_info_gain_ratio = 0.0
        for i in range(n_features):
            feature = list(data)[i]
            if feature == 'class':
                continue
            if len(set(data[feature])) == 1:
                info_gain_ratio = 0
            else:
                info_gain_ratio = getInfoGainRatio(data, feature)
            if info_gain_ratio > best_info_gain_ratio:
                best_info_gain_ratio = info_gain_ratio
                best_feature = feature
        if best_info_gain_ratio > self.epsilon:
            return best_feature, best_info_gain_ratio
        else:
            return -1, None

    def get_CART_best_feature(self, data):
        n_features = data.shape[1] - 1
        best_feature = -1
        best_feature_val = None
        best_gini = 1e6
        for i in range(n_features):
            feature = list(data)[i]
            if feature == 'class':
                continue
            feature_prob = data[feature].value_counts(normalize=True)
            feature_vals = feature_prob.index.sort_values()
            if len(feature_vals) == 1:
                continue
            for feature_val in feature_vals:
                data1 = data[data[feature] == feature_val]['class']
                data2 = data[data[feature] != feature_val]['class']
                gini = feature_prob[feature_val] * getGini(data1) + \
                       (1 - feature_prob[feature_val]) * getGini(data2)
                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature
                    best_feature_val = feature_val
        return best_feature, best_feature_val, best_gini

    def fit(self, data, parent=None):
        node = Node(is_leaf=True,
                    classification=None,
                    attr_split=None,
                    attr_split_value=None,
                    parent=parent,
                    depth=0)
        if node.parent != None:
            node.depth = node.parent.depth + 1
        node.samples = data.shape[0]
        node.classification = Counter(data['class']).most_common(1)[0][0]

        n_features = data.shape[1] - 1
        n_classes = len(set(data['class']))
        if n_classes == 1:
            node.is_leaf = True
            return node
        if n_features == 1:
            node.is_leaf = True
            return node
        feature_list = list(data)
        feature_list.pop()

        if self.type == 'C4.5':
            node.attr_split, node.attr_split_value = self.get_C45_best_feature(data)
            if node.attr_split == -1:
                node.is_leaf = 1
                return node
            node.is_leaf = False
            best_feature_vals = data[node.attr_split].unique()
            node.children = {}
            for val in best_feature_vals:
                child_node = self.fit(
                    data[data[node.attr_split] == val].drop([node.attr_split, ], axis=1), parent=node)
                node.children[val] = child_node

        elif self.type == 'CART':
            best_feature, best_feature_val, best_gini = self.get_CART_best_feature(data)
            data1 = data[data[best_feature] == best_feature_val].drop([best_feature, ], axis=1)
            if len(set(data[best_feature])) == 1:
                data2 = data[data[best_feature] != best_feature_val].drop([best_feature, ], axis=1)
            else:
                data2 = data[data[best_feature] != best_feature_val]

            left_child = self.fit(data1, node)
            right_child = self.fit(data2, node)

            node.is_leaf = False
            node.attr_split = '{}:{}'.format(best_feature, best_feature_val)
            node.attr_split_value = best_gini
            node.children = {}
            node.children['yes'] = left_child
            node.children['no'] = right_child

        if node.parent == None:
            self.tree = node
        return node

    def predict(self, X, root=None):
        if root == None:
            root = self.tree
        id_list = []
        y = []
        for id, x in X.iterrows():
            node = root
            while node.is_leaf == False:
                if self.type == 'C4.5':
                    attr_split = node.attr_split
                    val = x[attr_split]
                    if not val in node.children.keys():
                        break
                    node = node.children[val]
                elif self.type == 'CART':
                    attr = node.attr_split.split(':')[0]
                    attr_val = node.attr_split.split(':')[1]
                    if x[attr] == attr_val:
                        node = node.children['yes']
                    else:
                        node = node.children['no']
            id_list.append(int(id))
            y.append(node.classification)
        df = pd.DataFrame({'predict': y, 'index': id_list}).set_index(['index',], drop=True)
        df.index.name = None
        return df

    def validate(self, data, root=None):
        if root == None:
            root = self.tree
        X = data.drop(columns=['class',])
        y = data[['class']]
        y_pred = self.predict(X, root=root)
        y_cmp = pd.concat([y_pred, y], axis=1, join='inner')
        y_cmp['cmp'] = y_cmp.apply(lambda x: x['class'] == x['predict'], axis=1)
        acc = y_cmp['cmp'].mean()
        return acc

    def copy_tree(self, node=None, parent=None):
        if node == None:
            node = self.tree
        new_node = Node(
            is_leaf=node.is_leaf,
            classification=node.classification,
            attr_split=node.attr_split,
            attr_split_value=node.attr_split_value,
            parent=parent,
            children =node.children,
            depth=node.depth,
            samples=node.samples,
            gt=node.gt
        )
        if node.is_leaf:
            return new_node
        else:
            for val, child_node in node.children.items():
                new_child_node = self.copy_tree(child_node, new_node)
                new_node.children[val] = new_child_node
            return new_node

    def update_gt(self, data, root=None):
        if root == None:
            root = self.tree
        alpha = 1
        if root.is_leaf == False:
            c1 = 1 - data[data['class'] == root.classification].shape[0] / data.shape[0]
            c2 = 1 - self.validate(data, root)
            t = self.get_num_leaves(root)
            root.gt = (c1 - c2) / t
            # print(root.attr_split)
            # print(root.gt)
            # print()

            if root.gt < alpha:
                alpha = root.gt
                # print('smaller: {}'.format(alpha))
            attr_split = root.attr_split.split(':')[0]
            val_split = root.attr_split.split(':')[1]
            child1 = root.children['yes']
            child2 = root.children['no']
            data1 = data[data[attr_split] == val_split].drop([attr_split, ], axis=1)
            data2 = data[data[attr_split] != val_split]

            alpha1 = self.update_gt(data1, child1)
            alpha2 = self.update_gt(data2, child2)
            alpha = min(alpha, alpha1)
            alpha = min(alpha, alpha2)
            return alpha
        else:
            return 1

    def prune_cart(self, train, valid=None):
        best_tree = self.copy_tree(self.tree)
        best_tree_validation = self.validate(valid, self.tree)
        root = self.copy_tree(self.tree)

        flag = True
        while flag:
            alpha = self.update_gt(train, root)
            node_stack = []
            node_stack.append(root)

            while len(node_stack) > 0:
                node = node_stack[-1]
                node_stack.pop()
                if not node.is_leaf:
                    if node.gt == alpha:
                        node.is_leaf = True
                    else:
                        for _, child_node in node.children.items():
                            node_stack.append(child_node)
            validation = self.validate(valid, root)

            if validation > best_tree_validation:
                best_tree = self.copy_tree(root)
                best_tree_validation = validation
                print('update, validation: {}, T: {}'.format(validation, self.get_num_leaves(best_tree)))
                plotTree.CART_Tree(self.to_dict(best_tree), 'valid{}.png'.format(best_tree_validation))
            if root.is_leaf:
                flag = False
        return best_tree

    def prune_algo4_5(self, data, alpha=5, node=None):
        if node == None:
            node = self.tree
        # print("entering node {}, depth={}, num_children={}".format(node.attr_split, node.depth, len(node.children)))
        new_node = Node(
            is_leaf=node.is_leaf,
            classification=node.classification,
            attr_split=node.attr_split,
            attr_split_value=node.attr_split_value,
            parent=node.parent,
            depth=node.depth,
        )
        new_node.samples = node.samples
        new_node.children = node.children
        if node.is_leaf:
            return new_node

        for val, child_node in node.children.items():
            new_child = self.prune_algo4_5(data, alpha, child_node)
            new_node.children[val] = new_child

        prune_flag = True
        for child_node in node.children.values():
            if child_node.is_leaf == False:
                prune_flag = False
        if prune_flag:
            loss1 = 0
            for child_node in node.children.values():
                loss1 += child_node.samples * self.get_emp_ent(child_node, data)
            loss2 = node.samples * self.get_emp_ent(node, data)
            if loss1 + len(node.children.values()) * alpha > loss2 + alpha:
                new_node.is_leaf = True
        return new_node

    def prune_depth(self, node, depth):
        if node.depth == depth:
            # print("PRUNE: node {}, depth={}".format(node.attr_split, node.depth))
            node.is_leaf = True
        elif node.depth < depth:
            for val, child_node in node.children.items():
                self.prune_depth(child_node, depth)

    def prune_samples(self, node, samples):
        if node.samples < samples:
            # print("PRUNE: node {}, samples={}".format(node.attr_split, node.samples))
            node.is_leaf = True
        else:
            # print("entering node samples={}".format(node.samples))
            for val, child_node in node.children.items():
                self.prune_samples(child_node, samples)

    def to_dict(self, node=None):
        if node == None:
            node = self.tree
        dict = {node.attr_split: {}}
        for val, child_node in node.children.items():
            if child_node.is_leaf == True:
                dict[node.attr_split][val] = child_node.classification
            else:
                dict[node.attr_split][val] = self.to_dict(child_node)
        return dict
