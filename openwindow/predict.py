import argparse
import sys
from pathlib import Path

import cli


def predict(args):
    import pandas as pd

    import pytorch.training as training
    from core.openwindow import OpenWindow

    dataset = OpenWindow(args.dataset_dir)
    subset = dataset['test']

    # Mask out data based on user specification
    if args.mask:
        subset = subset[args.mask]

    # Compute predictions for each model and ensemble using mean
    log_dir = args.log_dir / args.training_id
    model_dir = args.model_dir / args.training_id
    epochs = _determine_epochs(args.epochs, log_dir)
    preds = [training.predict(subset, epoch, model_dir) for epoch in epochs]

    y_pred = pd.concat(preds).groupby(level=0).mean()

    # Ensure output directory exists
    prediction_dir = args.prediction_dir / args.training_id
    prediction_dir.mkdir(parents=True, exist_ok=True)

    # Write predictions to output directory
    output_path = prediction_dir / f'{subset.name}.csv'
    print(f'Output path: {output_path}')
    y_pred.to_csv(output_path)

    # Remove model files that were not used for prediction
    if args.clean:
        count = 0
        for path in model_dir.glob('model.[0-9][0-9].pth'):
            if int(str(path)[-6:-4]) not in epochs:
                path.unlink()
                count += 1
        print(f'Removed {count} unused model files')


def parse_args():
    config, conf_parser, remaining_args = cli.parse_config_args()
    parser = argparse.ArgumentParser(parents=[conf_parser])

    args_default = dict(config.items('Default'))
    args_prediction = dict(config.items('Prediction'))
    training_id = config['Training']['training_id']
    parser.set_defaults(**args_default,
                        **args_prediction,
                        training_id=training_id)

    parser.add_argument('--dataset_dir', type=Path, metavar='DIR')
    parser.add_argument('--model_dir', type=Path, metavar='DIR')
    parser.add_argument('--log_dir', type=Path, metavar='DIR')
    parser.add_argument('--prediction_dir', type=Path, metavar='DIR')
    parser.add_argument('--training_id', metavar='ID')
    parser.add_argument('--mask', type=cli.mask, metavar='MASK')
    parser.add_argument('--epochs', type=_epochs, metavar='EPOCHS')
    parser.add_argument('--clean', type=cli.boolean, metavar='BOOL')

    return parser.parse_args(remaining_args)


def _epochs(arg):
    split = arg.split(':')

    if len(split) == 1:
        return list(map(int, arg.split(',')))

    metric, n_epochs = split
    if metric in ['val_loss',
                  'val_acc',
                  'val_map',
                  ]:
        return metric, int(n_epochs)
    raise argparse.ArgumentTypeError(f'unrecognized metric: {metric}')


def _determine_epochs(spec, log_dir):
    import pandas as pd

    if type(spec) is list:
        return spec

    metric, n_epochs = spec
    df = pd.read_csv(log_dir / 'history.csv', index_col=0).iloc[10:]
    df.sort_values(by=metric, ascending=metric in ['val_loss'], inplace=True)
    return df.index.values[:n_epochs]


if __name__ == '__main__':
    sys.exit(predict(parse_args()))
