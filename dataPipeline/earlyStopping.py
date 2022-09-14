import numpy as np
import torch

# class inspired by https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py"

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0            
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.previous_value = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta        
        self.trace_func = trace_func

    def __call__(self, val_loss):
        # Initialization 
        if self.previous_value is None:            
            self.previous_value = val_loss
        # if the loss is lower, we continue with the training
        elif val_loss < self.previous_value + self.delta:                                   
            self.counter = 0
        else: #If the loss is not lower, it could be either greater or equals (no change reported here)
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience} | {val_loss}>{self.previous_value}')
            if self.counter >= self.patience:
                self.early_stop = True
        self.previous_value = val_loss # update previous value

    def save_checkpoint(self, val_loss):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f})')        
        self.val_loss_min = val_loss