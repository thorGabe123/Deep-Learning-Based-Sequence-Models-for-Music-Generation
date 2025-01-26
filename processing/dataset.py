import os
import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler, DataLoader, random_split
import random
import json
from processing import *
from config import *


def shift_sequence(sequence, rand_int, lower_bound, upper_bound):
    shifted_sequence = sequence.clone()
    mask = (sequence >= lower_bound) & (sequence < upper_bound)
    shifted_sequence[mask] = torch.clamp(sequence[mask] + rand_int, min=lower_bound, max=upper_bound - 1)
    return shifted_sequence

def multiply_sequence(sequence, rand_ints, lower_bound, upper_bound):
    multiplied_sequence = sequence.clone()
    mask = (sequence >= lower_bound) & (sequence < upper_bound)
    multiplied_sequence[mask] = torch.clamp((sequence[mask] - lower_bound) * rand_ints + lower_bound, min=lower_bound, max=upper_bound - 1)
    return multiplied_sequence

def get_metadata_json():
    with open('F:\\GitHub\\dataset\\midi_dataset\\metadata.json', 'r') as f:
        metadata = json.load(f)
    return metadata

def floor_to_nearest_10(number):
    return (number // 10) * 10

class SequenceDataset(Dataset):
    def __init__(self, directory):
        """
        Args:
            directory (str): Path to the directory containing .npy files.
            sequence_length (int, optional): Fixed length for sequences. If specified, sequences will be
                                            truncated or padded to this length. Default is None.
        """
        self.directory = directory
        self.sequence_length = BLOCK_SIZE
        self.file_paths = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.npy')]
        self.metadata = self.get_metadata()

    def get_metadata(self):
        # Load the metadata
        metadata = get_metadata_json()

        genre_list = []
        min_time, max_time = 1e9, 0
        final_metadata = {}
        for data in metadata['artists']:
            band = data['name']
            decade = floor_to_nearest_10(data['year_started'])
            min_time = min(min_time, decade)
            max_time = max(max_time, decade)
            genres = data['genres']
            for genre in genres:
                if genre not in genre_list:
                    genre_list.append(genre)
            final_metadata[band] = {'decade' : decade,
                                    'genres' : genres}

        band_tokenized = {band : idx + 1 for idx, band in enumerate(final_metadata.keys())}
        time_tokenized = {time : idx + 1 for idx, time in enumerate(range(min_time, max_time + 1, 10))}
        genre_tokenized = {genre : idx + 1 for idx, genre in enumerate(genre_list)}

        for band in final_metadata.keys():
            final_metadata[band]['band'] = band_tokenized[band]
            final_metadata[band]['decade'] = time_tokenized[final_metadata[band]['decade']]
            final_metadata[band]['genres'] = [genre_tokenized[genre] for genre in final_metadata[band]['genres']]

        return final_metadata
    
    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # Load the sequence from the .npy file
        file_path = self.file_paths[idx]
        sequence = np.load(file_path)
        seq_len_extra = self.sequence_length + 1

        if self.sequence_length:
            if seq_len_extra > len(sequence):
                padding = np.zeros(seq_len_extra - len(sequence), dtype=np.int64)
                sequence = np.concatenate([sequence, padding])
            # Adjust sequence length (truncate or pad)
            elif len(sequence) > seq_len_extra:
                ix = random.randint(0, len(sequence) - seq_len_extra - 1)
                sequence = sequence[ix : ix + seq_len_extra]

        sequence = torch.tensor(sequence)

        # Pitch shifting
        note_r_ints = random.randint(-12, 12)
        note_lb = START_IDX['PITCH_RES']
        note_ub = START_IDX['PITCH_RES'] + PITCH_RES - 1
        sequence = shift_sequence(sequence, note_r_ints, note_lb, note_ub)
        
        # Velocity shifting
        vel_r_ints = random.randint(-20, 20)
        vel_lb = START_IDX['DYN_RES']
        vel_ub = START_IDX['DYN_RES'] + DYN_RES  - 1
        sequence = shift_sequence(sequence, vel_r_ints, vel_lb, vel_ub)
        
        # Time multiplication
        time_r_ints = random.randint(1, 8) // 2
        time_lb = START_IDX['TIME_RES']
        time_ub = START_IDX['TIME_RES'] + TIME_RES  - 1
        sequence = multiply_sequence(sequence, time_r_ints, time_lb, time_ub)
        
        # Length multiplication
        len_lb = START_IDX['LENGTH_RES']
        len_ub = START_IDX['LENGTH_RES'] + LENGTH_RES  - 1
        sequence = multiply_sequence(sequence, time_r_ints, len_lb, len_ub)

        # tempo multiplication
        temp_lb = START_IDX['TEMPO_RES']
        temp_ub = START_IDX['TEMPO_RES'] + TEMPO_RES - 1
        sequence = multiply_sequence(sequence, time_r_ints, temp_lb, temp_ub)

        # Convert to tensor
        return sequence[:-1].to(DEVICE), sequence[1:].to(DEVICE)
    
    def file_prob(self):
        file_prob = [len(np.load(path)) for path in self.file_paths]
        file_prob /= np.sum(file_prob)
        return file_prob

def get_train_test_dataloaders(directory, batch_size=BATCH_SIZE, test_ratio=TEST_RATIO):
    """
    Create training and testing DataLoaders with WeightedRandomSampler.

    Args:
        directory (str): Path to the dataset directory.
        batch_size (int): Batch size for the DataLoader.
        sequence_length (int or None): Sequence length for the dataset.
        test_ratio (float): Ratio of the dataset to use for testing (default: 0.2).

    Returns:
        tuple: (train_dataloader, test_dataloader)
    """

    # Load the dataset
    dataset = SequenceDataset(directory)

    #TODO Add metadata to dataloader
    file_prob = dataset.file_prob()  # Get file probabilities

    # Split the dataset into training and testing subsets
    test_size = int(len(dataset) * test_ratio)
    train_size = len(dataset) - test_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Create WeightedRandomSamplers for train and test datasets
    train_sampler = WeightedRandomSampler(
        weights=[file_prob[i] for i in train_dataset.indices],  # Train subset probabilities
        num_samples=len(train_dataset),  # Samples to draw per epoch
        replacement=True
    )
    test_sampler = WeightedRandomSampler(
        weights=[file_prob[i] for i in test_dataset.indices],  # Test subset probabilities
        num_samples=len(test_dataset),  # Samples to draw per epoch
        replacement=True
    )

    # Create DataLoaders for train and test datasets
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=False
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        sampler=test_sampler,
        shuffle=False
    )

    return train_dataloader, test_dataloader