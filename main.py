import pandas as pd
from tree import DecisionTreeClassifier

import plotTree

if __name__ == '__main__':
    #### test
    # data = pd.read_csv("data/test.data", header=None, index_col=None, sep=',')
    # data.columns = ['age', 'work', 'house', 'credit', 'class']

    #### Lenses
    # data = pd.read_csv("data/lenses/lenses1.data", header=None, dtype=str)
    # data.columns = ['age_of_patient', 'spectacle_prescription', 'astigmatic', 'tear_production_rate', 'class']

    #### car
    data = pd.read_csv("data/car/car.data", header=None, dtype=str)
    data.columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']

    #### DATASET: mushroom
    # data = pd.read_csv("data/mushroom/mushroom.data", header=None)
    # data.columns = ['cap-shape', 'cap-surface', 'cap-color', 'bruises?',
    #                 'odor', 'gill-attachment', 'gill-spacing', 'gill-size',
    #                 'gill-color', 'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
    #                 'stalk-surface-below-ring', 'stalk-color-above-ring',
    #                 'stalk-color-below-ring', 'veil-type', 'veil-color',
    #                 'ring-number', 'ring-type', 'spore-print-color',
    #                 'population', 'habitat', 'class']
    #
    # data.drop_duplicates(['cap-shape', 'cap-surface', 'cap-color', 'bruises?',
    #                       'odor', 'gill-attachment', 'gill-spacing', 'gill-size',
    #                       'gill-color', 'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
    #                       'stalk-surface-below-ring', 'stalk-color-above-ring',
    #                       'stalk-color-below-ring', 'veil-type', 'veil-color',
    #                       'ring-number', 'ring-type', 'spore-print-color',
    #                       'population', 'habitat'], inplace=True)

    data = data.sample(frac=1)

    n_train = int(len(data) * 0.7)
    train = data[:n_train]
    valid = data[n_train:]

    print('\ngenerate CART...')
    clf = DecisionTreeClassifier(type='CART')
    clf.fit(train, parent=None)
    plotTree.CART_Tree(clf.to_dict(), 'test/cart.png')
    print('num_leaves={}'.format(clf.get_num_leaves(clf.tree)))
    print('[train acc]\t{}'.format(clf.validate(train)))
    print('[valid acc][cart]\t{}'.format(clf.validate(valid)))

    print('\nprune CART...')
    pruned = clf.prune_cart(train, valid)
    print('num_leaves={}'.format(clf.get_num_leaves(pruned)))
    pruned_dict = clf.to_dict(pruned)
    plotTree.CART_Tree(pruned_dict, 'test/cart_p.png')
    print('[valid acc][cart_pruned]\t{}'.format(clf.validate(valid, pruned)))

    print('\ngenerate C4.5...')
    clf_C45 = DecisionTreeClassifier(type='C4.5', epsilon=1e-6)
    clf_C45.fit(train, parent=None)
    print('num_leaves={}'.format(clf.get_num_leaves(clf_C45.tree)))
    plotTree.C45_Tree(clf_C45.to_dict(), 'test/c45.png')
    print('[train acc]\t{}'.format(clf_C45.validate(train)))
    print('[valid acc][C4.5]\t{}'.format(clf_C45.validate(valid)))

    print('\nprune C4.5...')
    pruned1 = clf_C45.prune_algo4_5(train)
    print('num_leaves={}'.format(clf.get_num_leaves(pruned1)))
    pruned_dict1 = clf_C45.to_dict(pruned1)
    plotTree.CART_Tree(pruned_dict1, 'test/c45_p.png')
    print('[valid acc][C4.5_pruned]\t{}'.format(clf_C45.validate(valid, pruned1)))


    #
    # y_pred = clf_C45.predict(test_X).reset_index(drop=True)
    # y = pd.concat([y_pred, test_y], axis=1)
    # y['cmp'] = y.apply(lambda x: x['class'] == x['predict'], axis=1)
    # acc = y['cmp'].mean()
    # print('[C4.5] acc: {}'.format(acc))



    ########## EXP_1

    #### DATASET: lenses
    # data = pd.read_csv("data/lenses/lenses1.data", header=None)
    # data.columns = ['age_of_patient', 'spectacle_prescription', 'astigmatic', 'tear_production_rate', 'class']

    #### DATASET: car
    # data = pd.read_csv("data/car/car.data", header=None)
    # data.columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']

    # data = data.sample(frac=1)
    # print("training samples: {}".format(len(data)))
    # n_train = int(len(data) * 0.7)
    # train = data[:n_train]
    # test= data[n_train:]
    # test_X, test_y = test, test['class'].reset_index(drop=True)
    # del(test_X['class'])
    #
    # clf_C45 = DecisionTreeClassifier(type='C4.5', epsilon=1e-6)
    # clf_C45.fit(data, parent=None)
    # dict = clf_C45.to_dict(clf_C45.tree)
    # plotTree.C45_Tree(dict, 'c45.png')
    #
    # y_pred = clf_C45.predict(test_X).reset_index(drop=True)
    # y = pd.concat([y_pred, test_y], axis=1)
    # y['cmp'] = y.apply(lambda x: x['class'] == x['predict'], axis=1)
    # acc = y['cmp'].mean()
    # print('[C4.5] acc: {}'.format(acc))
    #
    # clf_CART = DecisionTreeClassifier(type='CART', epsilon=1e-6)
    # clf_CART.fit(data, parent=None)
    # CART_DICT = clf_CART.to_dict(clf_CART.tree)
    # plotTree.CART_Tree(CART_DICT, 'cart.png')
    #
    # y_pred = clf_CART.predict(test_X).reset_index(drop=True)
    # y = pd.concat([y_pred, test_y], axis=1)
    # y['cmp'] = y.apply(lambda x: x['class'] == x['predict'], axis=1)
    # acc = y['cmp'].mean()
    # print('[CART] acc: {}'.format(acc))




    #### DATASET: mushroom
    # data = pd.read_csv("data/mushroom/mushroom.data", header=None)
    # data.columns = ['cap-shape', 'cap-surface', 'cap-color', 'bruises?',
    #                 'odor', 'gill-attachment', 'gill-spacing', 'gill-size',
    #                 'gill-color', 'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
    #                 'stalk-surface-below-ring', 'stalk-color-above-ring',
    #                 'stalk-color-below-ring', 'veil-type', 'veil-color',
    #                 'ring-number', 'ring-type', 'spore-print-color',
    #                 'population', 'habitat', 'class']
    #
    # ## shuffle
    # data = data.sample(frac=0.6)
    # print("total: {}".format(len(data)))
    # n_train = int(len(data) * 0.7)
    # train = data[:n_train]
    # test= data[n_train:]
    # test_X, test_y = test, test['class'].reset_index(drop=True)
    # del(test_X['class'])

    # # no prune
    # clf = DecisionTreeClassifier(type='C4.5', epsilon=1e-6)
    # clf.fit(data, parent=None)
    # dict = clf.to_dict()
    # print(dict)
    # plotTree.C45_Tree(dict, "no_prune.png")
    #
    # y_pred = clf.predict(test_X).reset_index(drop=True)
    # y = pd.concat([y_pred, test_y], axis=1)
    # y['cmp'] = y.apply(lambda x: x['class'] == x['predict'], axis=1)
    # acc = y['cmp'].mean()
    # print('[no prune] acc: {}'.format(acc))


    # # post prune
    # clf_p = DecisionTreeClassifier(type='C4.5', epsilon=1e-6)
    # clf_p.fit(data, parent=None, prune='post_prune', alpha=1)
    # dict = clf_p.to_dict()
    # print(dict)
    # plotTree.C45_Tree(dict, "post_prune.png")
    #
    # y_pred = clf_p.predict(test_X).reset_index(drop=True)
    # y = pd.concat([y_pred, test_y], axis=1)
    # y['cmp'] = y.apply(lambda x: x['class'] == x['predict'], axis=1)
    # acc = y['cmp'].mean()
    # print('[post-prune] acc: {}'.format(acc))
    #
    # # depth prune
    # clf_d = DecisionTreeClassifier(type='C4.5', epsilon=1e-6)
    # clf_d.fit(data, parent=None, prune='depth_prune', alpha=1)
    # dict = clf_d.to_dict()
    # print(dict)
    # plotTree.C45_Tree(dict, "depth_prune.png")
    #
    # y_pred = clf_d.predict(test_X).reset_index(drop=True)
    # y = pd.concat([y_pred, test_y], axis=1)
    # y['cmp'] = y.apply(lambda x: x['class'] == x['predict'], axis=1)
    # acc = y['cmp'].mean()
    # print('[depth-prune] acc: {}'.format(acc))
    #
    # # sample prune
    # clf_s = DecisionTreeClassifier(type='C4.5', epsilon=1e-6)
    # clf_s.fit(data, parent=None, prune='sample_prune', alpha=1)
    # dict = clf_s.to_dict()
    # print(dict)
    # plotTree.C45_Tree(dict, "sample_prune.png")
    #
    # y_pred = clf_s.predict(test_X).reset_index(drop=True)
    # y = pd.concat([y_pred, test_y], axis=1)
    # y['cmp'] = y.apply(lambda x: x['class'] == x['predict'], axis=1)
    # acc = y['cmp'].mean()
    # print('[sample-prune] acc: {}'.format(acc))

    # ###### param test
    # for alpha in [1, 5, 10, 20]:
    #     clf_p = DecisionTreeClassifier(type='C4.5', epsilon=1e-6)
    #     clf_p.fit(data, parent=None, prune='post_prune', alpha=1)
    #     # dict = clf_p.to_dict()
    #     # print(dict)
    #     # plotTree.C45_Tree(dict, "post_prune.png")
    #
    #     y_pred = clf_p.predict(test_X).reset_index(drop=True)
    #     y = pd.concat([y_pred, test_y], axis=1)
    #     y['cmp'] = y.apply(lambda x: x['class'] == x['predict'], axis=1)
    #     acc = y['cmp'].mean()
    #     print('[alpha={}] acc: {}'.format(alpha, acc))

















