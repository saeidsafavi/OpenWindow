import argparse
import sys
from pathlib import Path

import cli


def train(args):
    import pytorch.training as training
    from core.openwindow import OpenWindow

    # Create training and validation sets
    dataset = OpenWindow(args.dataset_dir)
    if args.training_mask:
        train_set = dataset['root'].subset('training', args.training_mask)
    else:
        train_set = dataset['training']
    if args.validation_mask:
        val_set = dataset['root'].subset('validation', args.validation_mask)
    else:
        val_set = dataset['validation']

    # Ensure output directories exist
    log_dir = args.log_dir / args.training_id
    log_dir.mkdir(parents=True, exist_ok=True)
    model_dir = args.model_dir / args.training_id
    model_dir.mkdir(parents=True, exist_ok=True)

    params = {
        'seed': args.seed,
        'batch_size': args.batch_size,
        'n_epochs': args.n_epochs,
        'lr': args.lr,
        'lr_decay': args.lr_decay,
        'lr_decay_rate': args.lr_decay_rate,
    }
    print(params, '\n')
    print(f'log_dir: {log_dir}')
    print(f'model_dir: {model_dir}')
    training.train(train_set, val_set, log_dir, model_dir, **params)


def parse_args():
    config, conf_parser, remaining_args = cli.parse_config_args()
    parser = argparse.ArgumentParser(parents=[conf_parser])

    args_default = dict(config.items('Default'))
    args_training = dict(config.items('Training'))
    parser.set_defaults(**args_default, **args_training)

    parser.add_argument('--dataset_dir', type=Path, metavar='DIR')
    parser.add_argument('--model_dir', type=Path, metavar='DIR')
    parser.add_argument('--log_dir', type=Path, metavar='DIR')
    parser.add_argument('--training_id', metavar='ID')
    parser.add_argument('--training_mask', type=cli.mask, metavar='MASK')
    parser.add_argument('--validation_mask', type=cli.mask, metavar='MASK')
    parser.add_argument('--seed', type=int, metavar='N')
    parser.add_argument('--batch_size', type=int, metavar='N')
    parser.add_argument('--n_epochs', type=int, metavar='N')
    parser.add_argument('--lr', type=float, metavar='NUM')
    parser.add_argument('--lr_decay', type=float, metavar='NUM')
    parser.add_argument('--lr_decay_rate', type=int, metavar='N')

    return parser.parse_args(remaining_args)


if __name__ == '__main__':
    sys.exit(train(parse_args()))
