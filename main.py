def run_tests():
    from tests.test_decision_tree import TestDecisionTree

    tdt = TestDecisionTree()
    tdt.test_tree()


if __name__ == '__main__':
    run_tests()
