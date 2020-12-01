import pandas as pd

from Tree import Tree


if __name__ == '__main__':
    df = pd.read_csv('data/zoo.csv')
    X = df[['hair', 'feathers', 'eggs', 'milk', 'airborne', 'aquatic', 'predator']].values
    y = df.type
    print('-=-=- X =-=-=-=\n{}\n-=-=- y -=-=-{}'.format(X, y))

    print('-=-=-=-= Custom =-=-=-=-=-')
    model = Tree(7)
    print("Fit:")
    model.fit(X, y)

    print('-=-=-=-=-=-')
    print(model.get_node_list())

    handle_graph = open('graph.dot', 'w')
    model.export_graph(file=handle_graph)
    print(model.predict(X))
