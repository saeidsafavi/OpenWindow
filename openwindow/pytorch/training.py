import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from pytorch.data import SegmentedDataset, MappedDataLoader
from pytorch.logging import Logger
from pytorch.models import CNN6, LogmelExtractor


def train(train_set, val_set, log_dir, model_dir, **params):
    if params['seed'] >= 0:
        _ensure_reproducibility(params['seed'])

    # Determine which device (GPU or CPU) to use
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Instantiate neural network
    model_args = {
        'n_classes': len(train_set.dataset.label_set),
        'extractor': LogmelExtractor(
            sample_rate=train_set.dataset.sample_rate,
            n_fft=2048,
            hop_length=1024,
            n_mels=64,
        )
    }
    model = CNN6(**model_args)
    model.to(device)

    # Use cross-entropy loss and Adam optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=params['lr'])
    # Use StepLR scheduler to decay learning rate regularly
    scheduler = StepLR(optimizer, params['lr_decay_rate'], params['lr_decay'])

    # Use helper classes to iterate over data in batches
    batch_size = params['batch_size']
    loader_train = MappedDataLoader(
        SegmentedDataset(train_set),
        device=device,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=True,
    )
    loader_val = MappedDataLoader(
        SegmentedDataset(val_set),
        device=device,
        batch_size=batch_size,
        pin_memory=True,
    )

    # Instantiate Logger to record training/validation performance and
    # save checkpoint to disk after every epoch.
    logger = Logger(log_dir, model_dir)
    logger.checkpoint_args['model_args'] = model_args

    print('\n', model, '\n', sep='')

    for epoch in range(params['n_epochs']):
        # Train model using training set
        pbar = tqdm(loader_train)
        pbar.set_description(f'Epoch {epoch}')
        _train(model.train(), pbar, criterion, optimizer, logger)

        # Evaluate model using validation set
        _validate(model.eval(), loader_val, criterion, logger)

        # Invoke learning rate scheduler
        scheduler.step()

        # Log results and save model to disk
        logger.step(model, optimizer, scheduler)

    logger.close()


def predict(subset, epoch, model_dir, batch_size=128):
    # Determine which device (GPU or CPU) to use
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Load model from disk
    model_dir = model_dir / f'model.{epoch:02d}.pth'
    checkpoint = torch.load(model_dir, map_location=device)
    model = CNN6(**checkpoint['model_args'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device).eval()

    # Generate predictions
    dataset = SegmentedDataset(subset)
    loader = MappedDataLoader(
        dataset,
        device=device,
        batch_size=batch_size,
        pin_memory=True,
    )
    with torch.no_grad():
        output = torch.cat([model(batch_x).data for batch_x, _ in loader])
        y = output.softmax(dim=1).cpu().numpy()

    # Return as DataFrame object
    y = pd.DataFrame(y, index=dataset.tags.index,
                     columns=subset.dataset.label_set)
    return y


def _train(model, loader, criterion, optimizer, logger):
    for batch_x, batch_y in loader:
        optimizer.zero_grad()
        output = model(batch_x)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()

        logger.log('loss', loss.item())


def _validate(model, loader, criterion, logger):
    with torch.no_grad():
        y = [(model(batch_x).data, batch_y) for batch_x, batch_y in loader]
        y_pred, y_true = tuple(zip(*y))
        y_pred = torch.cat(y_pred)
        y_true = torch.cat(y_true)

    loss = criterion(y_pred, y_true)
    logger.log('val_loss', loss.item())

    y_pred = y_pred.softmax(dim=1).argmax(dim=1)
    acc = (y_pred == y_true).to(torch.float).mean()
    logger.log('val_acc', acc.item())


def _ensure_reproducibility(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
