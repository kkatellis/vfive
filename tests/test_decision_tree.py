import unittest
import pandas as pd

from vfive.DecisionTree import DecisionTree, weighted_gini


class TestDecisionTree(unittest.TestCase):

    def test_init(self):

        dt = DecisionTree(max_depth=5, min_samples_split=2)
        self.assertEqual(dt.max_depth, 5)
        self.assertEqual(dt.min_samples_split, 2)

    def test_gini(self):
        labels = pd.Series(["Lion", "Lion", "Lion", "Panda"])
        g = weighted_gini(labels)
        print(g)
        self.assertEqual(g, 1.5, "(1 - ((3/4)^2 + (1/4)^2)) * 4")

    def test_tree(self):

        data = pd.read_csv('./data/animals.csv')
        print(data)
        X = data[['Number of legs', 'Color']]
        y = data['Name']
        print(X)
        print(y)

        dt = DecisionTree(max_depth=5)
        dt.fit(data=X, labels=y)
        print('trained decision tree', dt.tree)

        print('prediction for ', X.iloc[0], dt.predict(data=X.iloc[0]))

        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()
