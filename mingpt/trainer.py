"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""

import time
from collections import defaultdict

import torch
from torch.utils.data.dataloader import DataLoader
from mingpt.utils import CfgNode as CN
from IPython.display import clear_output
from tqdm import tqdm

class Trainer:

    @staticmethod
    def get_default_config():
        C = CN()
        # device to train on
        C.device = 'auto'
        # dataloader parameters
        C.num_workers = 4
        # optimizer parameters
        C.max_iters = None
        C.batch_size = 64
        C.learning_rate = 3e-4
        C.betas = (0.9, 0.95)
        C.weight_decay = 0.1 # only applied on matmul weights
        C.grad_norm_clip = 1.0
        C.collate_fn = None
        # validation parameters
        C.eval_interval = 1000  # Run validation every 1000 iterations
        return C

    def __init__(self, config, model, train_dataset, val_dataset=None):
        self.config = config
        self.model = model
        self.optimizer = None
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset  # Added validation dataset
        self.callbacks = defaultdict(list)

        # determine the device we'll train on
        if config.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = config.device
        self.model = self.model.to(self.device)
        print("Running on device", self.device)

        # variables that will be assigned to trainer class later for logging and etc
        self.iter_num = 0
        self.iter_time = 0.0
        self.iter_dt = 0.0

    def add_callback(self, onevent: str, callback):
        self.callbacks[onevent].append(callback)

    def set_callback(self, onevent: str, callback):
        self.callbacks[onevent] = [callback]

    def trigger_callbacks(self, onevent: str):
        for callback in self.callbacks.get(onevent, []):
            callback(self)

    @torch.no_grad()
    def validate(self):
        """Runs validation and computes average loss."""
        if self.val_dataset is None:
            return  # No validation dataset provided

        self.model.eval()
        val_loader = DataLoader(
            self.val_dataset,
            shuffle=False,
            pin_memory=True,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            collate_fn=self.config.collate_fn
        )

        total_loss = 0.0
        num_batches = 0

        for batch in val_loader:
            batch = [t.to(self.device) for t in batch]
            x, y = batch

            logits, loss = self.model(x, y)
            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        print(f"Validation Loss: {avg_loss:.4f}")

        self.model.train()  # Switch back to training mode

    def run(self):
        model, config = self.model, self.config

        # setup the optimizer
        self.optimizer = model.configure_optimizers(config)

        # setup the dataloader
        train_loader = DataLoader(
            self.train_dataset,
            sampler=torch.utils.data.RandomSampler(self.train_dataset, replacement=True, num_samples=int(1e10)),
            shuffle=False,
            pin_memory=True,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            collate_fn=config.collate_fn
        )

        model.train()
        self.iter_num = 0
        self.iter_time = time.time()
        data_iter = iter(train_loader)

        print("Training started...")

        progress_bar = tqdm(total=config.max_iters, desc="Training Progress", position=0, leave=True) if config.max_iters else None

        while True:

            # fetch the next batch (x, y) and re-init iterator if needed
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)
            batch = [t.to(self.device) for t in batch]
            x, y = batch

            # forward the model
            logits, self.loss = model(x, y)

            # Print loss every 1000 iterations
            if self.iter_num % 1000 == 0:
                print(f"Iteration {self.iter_num}: loss {self.loss.item()}")
                clear_output(wait=True)

            # Run validation every `eval_interval` iterations
            if self.val_dataset and self.iter_num % config.eval_interval == 0:
                self.validate()

            # Update progress bar
            if progress_bar:
                progress_bar.update(1)

            # backprop and update the parameters
            model.zero_grad(set_to_none=True)
            self.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
            self.optimizer.step()

            # Callbacks
            self.trigger_callbacks('on_batch_end')
            self.iter_num += 1

            # termination conditions
            if config.max_iters is not None and self.iter_num >= config.max_iters:
                break

        if progress_bar:
            progress_bar.close()

        print("Training finished!")
