import os
import neat
import numpy as np
import visualize
import pandas as pd
import multiprocessing
import pickle
from model import create_model
from sklearn.metrics import accuracy_score, confusion_matrix


def eval_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    real = []
    pred = []
    for xi, xo in zip(x_train, y_train):
        output = net.activate(xi)
        if output[0] >= 0.5:
            output[0] = 1.0
        else:
            output[0] = 0.0
        pred.append(output[0])
        real.append(xo)

    acc = accuracy_score(real, pred)
    return acc


def run(config_file, epochs, graph=True):
    p, config, stats = create_model(config_file)

    pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genome)
    winner = p.run(pe.evaluate, epochs)

    print('\nBest genome:\n{!s}'.format(winner))
    print('\nOutput:')

    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)

    # Evaluate model
    real = []
    pred = []

    for xi, xo in zip(x_test, y_test):
        output = winner_net.activate(xi)
        if output[0] >= 0.5:
            output[0] = 1.0
        else:
            output[0] = 0.0

        pred.append(output[0])
        real.append(xo)

    acc = accuracy_score(real, pred)
    confusion_mat = confusion_matrix(real, pred)
    confusion_mat = pd.DataFrame(confusion_mat)
    confusion_mat.to_csv('models/neat_model/csv_logs/confusion_matrix.csv', index=False)
    print(acc)


    # Save metrics
    stats.save_genome_fitness(delimiter=',', filename='models/neat_model/csv_logs/fitness_history.csv')
    stats.save_species_count(delimiter=',', filename='models/neat_model/csv_logs/speciation.csv')
    stats.save_species_fitness(delimiter=',', null_value='NA',
                               filename='models/neat_model/csv_logs/species_fitness.csv')
    # Save model
    with open('models/neat_model/checkpoints/winner.pkl', 'wb') as f:
        pickle.dump(winner_net, f)
        f.close()

    if graph:
        node_names = {k: v for (k, v) in enumerate(features_name, -n_features + 1)}
        visualize.draw_net(config, genome=winner, filename='./models/neat_model/figures/Diagraph.svg', view=False,
                           node_names=node_names)
        visualize.plot_stats(stats, filename='./models/neat_model/figures/avg_fitness.svg', ylog=False, view=False)
        visualize.plot_species(stats, filename='./models/neat_model/figures/speciation.svg', view=False)


x_train = np.load('data/train/x_train.npy').tolist()
y_train = np.load('data/train/y_train.npy').tolist()
x_test = np.load('data/test/x_test.npy').tolist()
y_test = np.load('data/test/y_test.npy').tolist()

features_name = pd.read_csv('dataset_eval.csv')
features_name = list(features_name.columns)
n_features = len(features_name)


if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config.txt')
    run(config_path, epochs=10, graph=True)