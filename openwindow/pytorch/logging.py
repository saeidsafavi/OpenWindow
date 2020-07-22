from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd

import torch
from torch.utils.tensorboard import SummaryWriter


class Logger:
    def __init__(self, log_dir, model_dir, overwrite=True):
        self.log_path = Path(log_dir) / 'history.csv'
        self.model_dir = Path(model_dir)
        self.results = OrderedDict()
        self.tb_writer = None

        self.checkpoint_args = {}

        # Remove previous TensorBoard log files if applicable
        if overwrite:
            for path in log_dir.glob('*tfevents*'):
                print(f'Deleting {path}')
                path.unlink()

        # Read from existing log file if applicable
        if overwrite or not self.log_path.exists():
            self.results_df = pd.DataFrame()
        else:
            self.results_df = pd.read_csv(self.log_path, index_col=0)

    def log(self, key, value):
        if self.results.get(key) is None:
            self.results[key] = []
        self.results[key].append(value)

    def step(self, model, optimizer, scheduler):
        # Write results to CSV file
        results = OrderedDict((k, np.mean(v)) for k, v in self.results.items())
        self.results_df = self.results_df.append(results, ignore_index=True)
        self.results_df.to_csv(self.log_path)

        # Write results to TensorBoard log file
        if self.tb_writer is None:
            self.tb_writer = SummaryWriter(self.log_path.parent)
        epoch = self.results_df.index[-1]
        for key, value in results.items():
            self.tb_writer.add_scalar(key, value, epoch)
        self.tb_writer.file_writer.flush()

        # Save model state to disk
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'rng_state': torch.get_rng_state(),
            **self.checkpoint_args,
        }
        if torch.cuda.is_available():
            checkpoint['cuda_rng_state'] = torch.cuda.get_rng_state()
        torch.save(checkpoint, self.model_dir / f'model.{epoch:02d}.pth')

        # Print results to stdout
        print(', '.join(['{}: {:.4f}'.format(k, v)
                         for k, v in results.items()]))

        self.results.clear()

    def close(self):
        self.tb_writer.close()
