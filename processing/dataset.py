import os
import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler, DataLoader, random_split
import random
from pathlib import Path
import json
from processing import *
import configs.common as cc
import configs.paths as paths

current_dir = Path(__file__).parent

def shift_sequence(sequence, rand_int, lower_bound, upper_bound):
    shifted_sequence = sequence.clone()
    mask = (sequence >= lower_bound) & (sequence < upper_bound)
    shifted_sequence[mask] = torch.clamp(sequence[mask] + rand_int, min=lower_bound, max=upper_bound - 1)
    return shifted_sequence

def shift_sequence_drums(sequence, rand_int, lower_bound, upper_bound):
    shifted_sequence = sequence.clone()
    mask = (sequence >= lower_bound) & (sequence < upper_bound)
    prev = torch.roll(sequence, shifts=1)
    prev_mask = (prev < cc.start_idx['channel'] + 128) | (prev >= cc.start_idx['tempo'])
    combined_mask = mask & prev_mask
    shifted_sequence[combined_mask] = torch.clamp(sequence[combined_mask] + rand_int, min=lower_bound, max=upper_bound - 1)
    return shifted_sequence

def multiply_sequence(sequence, rand_ints, lower_bound, upper_bound):
    multiplied_sequence = sequence.clone()
    mask = (sequence >= lower_bound) & (sequence < upper_bound)
    multiplied_sequence[mask] = torch.clamp((sequence[mask] - lower_bound) * rand_ints + lower_bound, min=lower_bound, max=upper_bound - 1).to(multiplied_sequence.dtype)
    return multiplied_sequence

def get_metadata_json():
    metadata_path = paths.config.paths.metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    return metadata

def save_metadata_tokenizations(tokenizations):
    meta_vocab_size = sum([len(x) for x in tokenizations.values()])
    tokenizations['VOCAB_SIZE'] = meta_vocab_size
    tokenizations_path = paths.config.paths.tokenizations
    with open(tokenizations_path, 'w') as f:
            json.dump(tokenizations, f, indent=4)

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
        self.sequence_length = cc.config.values.block_len
        self.file_paths = []
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith('.npy'):
                    self.file_paths.append(os.path.join(root, file))
        self.metadata_dict = self.get_metadata_dict()

    def get_metadata_dict(self):
        # Load the metadata
        metadata = get_metadata_json()

        # Initialize genre list and time range
        genre_list = []
        min_time, max_time = 1e9, 0

        # Process metadata to extract bands, decades, and genres
        metadata_json = {}
        for data in metadata['artists']:
            band = data['name']
            decade = floor_to_nearest_10(data['year_started'])
            min_time = min(min_time, decade)
            max_time = max(max_time, decade)
            genres = data['genres']
            for genre in genres:
                if genre not in genre_list:
                    genre_list.append(genre)
            metadata_json[band] = {'decade': decade, 'genres': genres}

        # Calculate metadata tokenization ranges
        num_decades = (max_time - min_time) // 10 + 1
        num_genres = len(genre_list)
        num_bands = len(metadata_json)

        # Define START_IDX_META for tokenization
        START_IDX_META = {}
        START_IDX_META['DECADE'] = 1
        START_IDX_META['GENRE'] = START_IDX_META['DECADE'] + num_decades + 1
        START_IDX_META['BAND'] = START_IDX_META['GENRE'] + num_genres + 1

        # Tokenize bands, decades, and genres
        band_tokenized = {band: idx + START_IDX_META['BAND'] for idx, band in enumerate(metadata_json)}
        time_tokenized = {time: idx + START_IDX_META['DECADE'] for idx, time in enumerate(range(min_time, max_time + 1, 10))}
        genre_tokenized = {genre: idx + START_IDX_META['GENRE'] for idx, genre in enumerate(genre_list)}

        # Save tokenizations
        tokenizations = {
            'time_tokenized': time_tokenized,
            'genre_tokenized': genre_tokenized,
            'band_tokenized': band_tokenized
        }
        tokenizations['time_tokenized'][None] = START_IDX_META['DECADE'] - 1
        tokenizations['genre_tokenized'][None] = START_IDX_META['GENRE'] - 1
        tokenizations['band_tokenized'][None] = START_IDX_META['BAND'] - 1
        save_metadata_tokenizations(tokenizations)

        # Create final metadata dictionary with tokenized values
        test_meta = {}
        for band, elem in metadata_json.items():
            # Tokenize and pad genres
            genres = [genre_tokenized[genre] for genre in elem['genres']]
            if len(genres) < 4:
                genres += [START_IDX_META['GENRE'] - 1] * (4 - len(genres))  # Pad with 0 tokens

            # Combine band, genres, and decade into a list
            test_meta[band] = torch.tensor([band_tokenized[band]] + genres + [time_tokenized[elem['decade']]])

        return test_meta

    def __len__(self):
        return len(self.file_paths)
    
    def data_augementation(self, sequence):
        # Pitch shifting
        note_r_ints = random.randint(-12, 12)
        note_lb = cc.start_idx['pitch']
        note_ub = cc.start_idx['pitch'] + 128 - 1
        sequence = shift_sequence_drums(sequence, note_r_ints, note_lb, note_ub)

        # Velocity shifting
        vel_r_ints = random.randint(-20, 20)
        vel_lb = cc.start_idx['dyn']
        vel_ub = cc.start_idx['dyn'] + cc.config.discretization.dyn - 1
        sequence = shift_sequence(sequence, vel_r_ints, vel_lb, vel_ub)

        # Time multiplication
        time_r_ints = random.randint(1, 8) / 2
        time_lb = cc.start_idx['time']
        time_ub = cc.start_idx['time'] + cc.config.discretization.time - 1
        sequence = multiply_sequence(sequence, time_r_ints, time_lb, time_ub)

        # Length multiplication
        len_lb = cc.start_idx['length']
        len_ub = cc.start_idx['length'] + cc.config.discretization.length - 1
        sequence = multiply_sequence(sequence, time_r_ints, len_lb, len_ub)

        # tempo multiplication
        temp_lb = cc.start_idx['tempo']
        temp_ub = cc.start_idx['tempo'] + cc.config.discretization.tempo - 1
        sequence = multiply_sequence(sequence, time_r_ints, temp_lb, temp_ub)
        return sequence

    def __getitem__(self, idx):
        # Load the sequence from the .npy file
        file_path = self.file_paths[idx]
        sequence = np.load(file_path)
        seq_len_extra = self.sequence_length + 1

        if seq_len_extra > len(sequence):
            padding = np.zeros(seq_len_extra - len(sequence), dtype=np.int64)
            sequence = np.concatenate([sequence, padding])
        # Adjust sequence length (truncate or pad)
        elif len(sequence) >= seq_len_extra:
            ix = random.randint(0, (len(sequence) - seq_len_extra - 1))
            sequence = sequence[ix: ix + seq_len_extra]

        sequence = torch.tensor(sequence, device=cc.config.values.device, dtype=torch.long)
        sequence = self.data_augementation(sequence)

        # Fetch metadata for the band
        path_parts = Path(file_path).parts
        band_name = path_parts[-2]
        band_metadata = self.metadata_dict[band_name].to(cc.config.values.device)

        # Return sequence and metadata
        return sequence[:-1], sequence[1:], band_metadata

    def file_prob(self):
        file_prob = [len(np.load(path)) for path in self.file_paths]
        file_prob /= np.sum(file_prob)
        return file_prob

class DatasetLoader:
    def __init__(self, directory, batch_size=cc.config.values.batch_size, test_ratio=cc.config.values.test_ratio):
        """
        Initialize the DatasetLoader with the dataset directory, batch size, and test ratio.
        """
        self.directory = directory
        self.batch_size = batch_size
        self.test_ratio = test_ratio

        # Load and split the dataset
        self.dataset = SequenceDataset(directory)
        self.file_prob = self.dataset.file_prob()
        self.train_dataset, self.test_dataset = self._split_dataset()

        # Create samplers
        self.train_sampler = self._create_sampler(self.train_dataset)
        self.test_sampler = self._create_sampler(self.test_dataset)

    def _split_dataset(self):
        """
        Split the dataset into training and testing subsets.
        """
        test_size = int(len(self.dataset) * self.test_ratio)
        train_size = len(self.dataset) - test_size
        return random_split(self.dataset, [train_size, test_size])

    def _create_sampler(self, dataset):
        """
        Create a WeightedRandomSampler for a given dataset.
        """
        return WeightedRandomSampler(
            weights=[self.file_prob[i] for i in dataset.indices],
            num_samples=len(dataset),
            replacement=True
        )

    def get_dataloaders(self):
        """
        Get train and test DataLoaders.
        """
        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=self.train_sampler,
            shuffle=False
        )
        test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            sampler=self.test_sampler,
            shuffle=False
        )
        return train_dataloader, test_dataloader

    def get_random_sample(self, sampler_type='train'):
        """
        Get a random sample from chosen sampler: 'train' or 'test'.
        """
        sampler = self.train_sampler if sampler_type == 'train' else self.test_sampler
        indices = list(sampler)
        random_index = random.choice(indices)
        return self.dataset[random_index]

    def get_metadata_vocab_size():
        tokenizations_path = paths.config.paths.tokenizations
        with open(tokenizations_path, 'r') as f:
            tokenizations = json.load(f)
        return tokenizations['VOCAB_SIZE']