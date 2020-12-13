import json
from utils.market import Market
from utils.portfolio import Portfolio
import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_neat', type=bool, default=False)
    parser.add_argument('--train_tf', type=bool, default=False)

    return parser.parse_known_args()


if __name__ == '__main__':
    args, _ = parse_args()

    with open('config.json', 'r') as f:
        config = json.load(f)

    if not os.path.exists('NASDAQ100.pkl'):
        market = Market(config).run()

    if args.train_tf:
        os.system('python models/tf_model/train.py')

    if args.train_neat:
        os.system('python models/neat_model/train.py')

    portfolio = Portfolio(config)
