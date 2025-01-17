import os
import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler, DataLoader
import random

class SequenceDataset(Dataset):
    def __init__(self, directory, sequence_length=None):
        """
        Args:
            directory (str): Path to the directory containing .npy files.
            sequence_length (int, optional): Fixed length for sequences. If specified, sequences will be
                                            truncated or padded to this length. Default is None.
        """
        self.directory = directory
        self.sequence_length = sequence_length
        self.file_paths = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.npy')]

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # Load the sequence from the .npy file
        file_path = self.file_paths[idx]
        sequence = np.load(file_path)

        if self.sequence_length:
            if self.sequence_length > len(sequence):
                padding = np.zeros(self.sequence_length - len(sequence), dtype=np.int64)
                sequence = np.concatenate([sequence, padding])
            # Adjust sequence length (truncate or pad)
            elif len(sequence) > self.sequence_length:
                ix = random.randint(0, len(sequence) - self.sequence_length - 1)
                sequence = sequence[ix : ix + self.sequence_length]

        # Convert to tensor
        return torch.tensor(sequence, dtype=torch.long)
    
    def file_prob(self):
        file_prob = [len(np.load(path)) for path in self.file_paths]
        file_prob /= np.sum(file_prob)
        return file_prob

# Function to create DataLoader with WeightedRandomSampler
def get_dataloader(directory, batch_size=32, sequence_length=None, shuffle=False):
    dataset = SequenceDataset(directory, sequence_length)
    file_prob = dataset.file_prob()

    # Create WeightedRandomSampler using file probabilities
    sampler = WeightedRandomSampler(
        weights=file_prob,  # Weights for each file
        num_samples=len(dataset),  # Number of samples to draw in an epoch
        replacement=True  # Allow replacement to sample with given probabilities
    )

    # Pass sampler to DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,  # Use the sampler instead of shuffle
        shuffle=False  # Must be False when using a sampler
    )
    return dataloader