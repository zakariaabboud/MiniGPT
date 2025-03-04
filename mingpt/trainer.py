"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""

import time
from collections import defaultdict

import torch
from torch.utils.data.dataloader import DataLoader
from mingpt.utils import CfgNode as CN
from tqdm import tqdm
from IPython.display import clear_output

def collate_fn(batch):
    # Find the max length in the batch
    max_len = max(len(x) for x, y in batch)

    # Pad each sequence to the max length
    x_batch = [torch.cat([x, torch.zeros(max_len - len(x), dtype=torch.long)]) for x, y in batch]
    y_batch = [torch.cat([y, torch.zeros(max_len - len(y), dtype=torch.long)]) for x, y in batch]
    
    # Stack the sequences into tensors
    x_batch = torch.stack(x_batch)
    y_batch = torch.stack(y_batch)
    
    return x_batch, y_batch
class Trainer:

    @staticmethod
    def get_default_config():
        C = CN()
        # device to train on
        C.device = 'auto'
        # dataloder parameters
        C.num_workers = 4
        # optimizer parameters
        C.max_iters = None
        C.batch_size = 64
        C.learning_rate = 3e-4
        C.betas = (0.9, 0.95)
        C.weight_decay = 0.1 # only applied on matmul weights
        C.grad_norm_clip = 1.0
        C.collate_fn=collate_fn
        return C

    def __init__(self, config, model, train_dataset):
        self.config = config
        self.model = model
        self.optimizer = None
        self.train_dataset = train_dataset
        self.callbacks = defaultdict(list)

        # determine the device we'll train on
        if config.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = config.device
        self.model = self.model.to(self.device)
        print("running on device", self.device)

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
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)

            batch = [t.to(self.device) for t in batch]
            x, y = batch

            # forward the model
            logits, self.loss = model(x, y)

            # Print loss and step dynamically
            clear_output(wait=True)  # Clears previous output for real-time updates
            print(f"Step: {self.iter_num} / {config.max_iters if config.max_iters else '∞'}")
            print(f"Loss: {self.loss.item():.6f}")

            # Update progress bar
            if progress_bar:
                progress_bar.update(1)

            # backprop and update the parameters
            model.zero_grad(set_to_none=True)
            self.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
            self.optimizer.step()

            # Print learning rate
            for param_group in self.optimizer.param_groups:
                print(f"Learning Rate: {param_group['lr']:.6e}")

            # Callbacks
            self.trigger_callbacks('on_batch_end')
            self.iter_num += 1

            # termination conditions
            if config.max_iters is not None and self.iter_num >= config.max_iters:
                break

        if progress_bar:
            progress_bar.close()

        print("Training finished!")
