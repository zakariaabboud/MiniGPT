import time
from collections import defaultdict

import torch
from torch.utils.data.dataloader import DataLoader
from mingpt.utils import CfgNode as CN
from tqdm import tqdm
from IPython.display import clear_output

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
        
        def test_dataloader(train_loader):
            print("Testing DataLoader...")

            # Test iterating over the DataLoader
            try:
                # Fetch a single batch
                batch = next(iter(train_loader))
                print(f"Batch Type: {type(batch)}")
                
                if isinstance(batch, tuple) and len(batch) == 2:
                    x, y = batch
                    print(f"Input shape: {x.shape}, Target shape: {y.shape}")
                elif isinstance(batch, dict):
                    print(f"Batch Keys: {batch.keys()}")
                    # Check if the batch has the expected keys
                    print(f"Input IDs sample: {batch['input_ids'][:2] if 'input_ids' in batch else 'Not found'}")
                    print(f"Labels sample: {batch['labels'][:2] if 'labels' in batch else 'Not found'}")
                else:
                    print(f"Batch structure: {batch}")
                    
            except Exception as e:
                print(f"Error while testing DataLoader: {e}")
        
        test_dataloader(train_loader)

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

            # Process batch to handle both tuple and dictionary formats
            if isinstance(batch, tuple) and len(batch) == 2:
                x, y = batch
                x = x.to(self.device)
                y = y.to(self.device)
            else:
                # If it's a dictionary or other format, convert to device first
                batch = {k: v.to(self.device) for k, v in batch.items()} if isinstance(batch, dict) else batch
                # In your specific case, you'll need to handle how to get x and y from batch

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