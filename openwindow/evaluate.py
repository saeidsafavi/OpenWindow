import argparse
import sys
from pathlib import Path

import cli


def evaluate_all(args):
    import numpy as np

    from core.openwindow import OpenWindow

    def _statistics(scores):
        sem = np.std(scores, ddof=1) / np.sqrt(len(scores))
        return np.mean(scores), sem

    dataset = OpenWindow(args.dataset_dir)
    pred_dir = args.prediction_dir

    if len(args.training_id) == 1:
        evaluate(dataset, args.training_id[0], pred_dir)
        return

    scores = []
    for training_id in args.training_id:
        scores.append(evaluate(dataset, training_id, pred_dir, verbose=False))

    for metric in ['Accuracy', 'Average Precision']:
        for average in ['Macro Average', 'Micro Average']:
            mean, sem = _statistics([s[metric][average] for s in scores])
            print(f'{metric} {average}: {mean:.3} \u00b1 {sem:.3}')


def evaluate(dataset, training_id, prediction_dir, verbose=True):
    import pandas as pd

    import core.evaluation as evaluation

    # Load grouth truth data and predictions
    path = prediction_dir / training_id / 'test.csv'
    y_pred = pd.read_csv(path, index_col=0)
    y_true = dataset.label(dataset['test'])
    y_true = y_true.loc[y_pred.index]

    y_true = y_true.values
    y_pred = y_pred.values

    # Evaluate audio tagging performance
    scores = evaluation.evaluate(y_true, y_pred, dataset.label_set)
    C = evaluation.confusion_matrix(y_true, y_pred)

    # Print results (optional)
    if verbose:
        print('Confusion Matrix:\n', C, '\n')

        pd.options.display.float_format = '{:,.3f}'.format
        print(str(scores))

    return scores


def parse_args():
    config, conf_parser, remaining_args = cli.parse_config_args()
    parser = argparse.ArgumentParser(parents=[conf_parser])

    args_default = dict(config.items('Default'))
    training_id = config['Training']['training_id']
    parser.set_defaults(**args_default, training_id=training_id)

    parser.add_argument('--dataset_dir', type=Path, metavar='DIR')
    parser.add_argument('--prediction_dir', type=Path, metavar='DIR')
    parser.add_argument('--training_id', type=cli.array, metavar='LIST')

    return parser.parse_args(remaining_args)


if __name__ == '__main__':
    sys.exit(evaluate_all(parse_args()))
