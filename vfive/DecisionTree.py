class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = []

    def fit(self, data, labels):
        """
        Fit the decision tree given some training data
        :param data:
        :param labels:
        :return:
        """
        self.tree = self.fit_tree(data, labels, depth=0)

    def fit_tree(self, data, labels, depth=0):
        """
        Recursive function to fit the decision tree
        :param data:
        :param labels:
        :param depth:
        :return:
        """
        not_done = False
        if self.max_depth is None or depth < self.max_depth:
            # not yet reached maximum depth
            not_done = True
        if len(data) > self.min_samples_split:
            # not yet reached minimum samples
            not_done = True

        if not_done:
            print('not done')
            col_vals = {col: data[col].unique() for col in data}

            split, gini = self.get_best_split(data, labels, col_vals)
            print(split, gini)
            if gini == 0:
                # no more splits possible
                return self.predict_leaf_label(labels)

            mask = data[split[0]] == split[1]
            left_data, right_data = data[mask], data[~mask]
            left_labels, right_labels = labels[mask], labels[~mask]

            print(len(left_data))
            print(len(right_data))

            sub_tree = [split]

            left_tree = self.fit_tree(left_data, left_labels, depth=depth + 1)
            right_tree = self.fit_tree(right_data, right_labels, depth=depth + 1)

            sub_tree.append(left_tree)
            sub_tree.append(right_tree)

            return sub_tree

        else:
            # reached end condition, return the prediction
            return self.predict_leaf_label(labels)

    def predict_leaf_label(self, labels):
        print('predict_leaf_label')
        return labels.mode()[0]

    def get_best_split(self, data, labels, candidate_splits):
        print('get_best_split')
        n = len(labels)
        print(n)
        print(len(data))
        best_split = None
        best_gini = 0
        for candidate_split_col, candidate_split_vals in candidate_splits.items():
            if len(candidate_split_vals) <= 1:
                continue
            for candidate_split_val in candidate_split_vals:
                #print('candidadate', candidate_split_col, candidate_split_val)
                mask = data[candidate_split_col] == candidate_split_val
                gini = (1/n) * (weighted_gini(labels[mask]) + weighted_gini(labels[~mask]))
                #print(gini)
                if gini > best_gini:
                    best_gini = gini
                    best_split = (candidate_split_col, candidate_split_val)
        print('--')
        print(f'best split: {best_split}, gini={best_gini}')
        return best_split, best_gini

    def predict(self, data):
        if len(self.tree) < 1:
            raise Exception("tree not trained!")
        return self.predict_tree(self.tree, data)

    def predict_tree(self, tree, data):
        if data[tree[0][0]] == tree[0][1]:
            branch = 1
        else:
            branch = 2
        if isinstance(tree[branch], str):
            return tree[branch]
        else:
            return self.predict_tree(tree[branch], data)

def weighted_gini(l):
    # I = 1 - sum_i p_i^2
    p = l.value_counts() / len(l)
    return (1 - sum(p ** 2)) * len(l)
