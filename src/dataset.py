import os
import lief
import lief.logging
import numpy
from torch.utils import data
from tqdm import tqdm

# Set LIEF logging level to suppress unnecessary output
lief.logging.set_level(lief.logging.LOGGING_LEVEL.CRITICAL)

# Constants for padding
FILE_START = 256
FILE_END = 257

class FunctionIdentificationDataset(data.Dataset):
    """Generates the dataset for processing by the CNN"""
    def __init__(self, root, block_size, padding_amount):
        """
        Initialize the dataset generator.

        Args:
            root (str): Root directory containing binary files.
            block_size (int): Size of each data block.
            padding_amount (int): Amount of padding to add to each block.
        """
        data, features = self._preprocess(root)
        self.data_blocks, self.feature_blocks = self._data_to_blocks(data, features, block_size, padding_amount)

    def __len__(self):
        """
        Get the number of data blocks in the dataset.

        Returns:
            int: Number of data blocks.
        """
        return len(self.data_blocks)
    
    def __getitem__(self, idx):
        """
        Get a data block and its corresponding features at the specified index.

        Args:
            idx (int): Index of the data block.

        Returns:
            tuple: Tuple containing data block and feature block.
        """
        return self.data_blocks[idx], self.feature_blocks[idx]
        
    def _preprocess(self, root: str):
        """
        Preprocess binary files and extract data and features.

        Args:
            root (str): Root directory containing binary files.

        Returns:
            tuple: Tuple containing lists of data and features.
        """
        bins_data = []
        bins_features = []

        all_files = [os.path.join(dirpath, file) 
            for dirpath, _, filenames in os.walk(root) 
            for file in filenames]

        for file_path in tqdm(all_files):
            bin = lief.parse(file_path)
            if not bin:
                continue
            data = self._get_code(bin)
            features = self._generate_features(bin)

            bins_data.append(data)
            bins_features.append(features)
        
        return bins_data, bins_features

    def _get_code(self, bin: lief._lief.ELF.Binary):
        """
        Extract the code (.text) section content from the binary.

        Args:
            bin (lief._lief.ELF.Binary): LIEF ELF binary object.

        Returns:
            numpy.ndarray: Numpy array containing code section content.
        """
        return numpy.array(bin.get_section(".text").content, dtype=int)
    
    def _generate_features(self, bin: lief._lief.ELF.Binary):
        """
        Generate features indicating function start addresses.

        Args:
            bin (lief._lief.ELF.Binary): LIEF ELF binary object.

        Returns:
            numpy.ndarray: Numpy array containing generated features.
        """
        text_section = bin.get_section(".text")
        
        excluded_symbols = ["_fini", "_init"] 

        start_addrs = []

        for symbol in bin.symbols:
            if symbol.name in excluded_symbols:
                continue
            if symbol.type == lief.ELF.SYMBOL_TYPES.FUNC and symbol.value != 0:
                relative_address = symbol.value - text_section.virtual_address
                start_addrs.append((relative_address, symbol.name))
        
        features = numpy.zeros(text_section.size, dtype=int)
        
        for addr, symbol_name in start_addrs:
            if addr >= text_section.size:
                print(f"[!] Warning: Address {addr} (from symbol {symbol_name}) is out of bounds for the .text section.")
            else:
                features[addr] = 1

        return features

    def _data_to_blocks(self, data: list, features: list, block_size: int, padding_amount: int):
        """
        Convert data and features to blocks with specified size and padding.

        Args:
            data (list): List of data arrays.
            features (list): List of feature arrays.
            block_size (int): Size of each block.
            padding_amount (int): Amount of padding for each block.

        Returns:
            tuple: Tuple containing lists of data blocks and feature blocks.
        """
        data_blocks = []
        feature_blocks = []

        for bin_data, bin_features in zip(data, features):
            for start_idx in range(0, len(bin_data), block_size):
                block_end_offset = start_idx + block_size
                data_blocks.append(self._get_padded_data(bin_data, start_idx, block_size, padding_amount))
                feature_blocks.append(bin_features[start_idx:block_end_offset])
        
        return data_blocks, feature_blocks

    def _get_padded_data(self, data, idx, block_size, padding_amount):
        """
        Apply padding to a data block.

        Args:
            data: Data array.
            idx (int): Index of the current block.
            block_size (int): Size of the block.
            padding_amount (int): Amount of padding to add.

        Returns:
            numpy.ndarray: Numpy array containing the padded data block.
        """
        left_padding_amount = int(padding_amount / 2)
        right_padding_amount = padding_amount - left_padding_amount

        left_padding = numpy.array([FILE_START] * (left_padding_amount - idx), dtype=int)
        right_padding = numpy.array([FILE_END] * (right_padding_amount - max(data.size - idx - block_size, 0)), dtype=int)

        block = data[max(idx - left_padding_amount, 0) : idx + block_size + right_padding_amount]

        return numpy.concatenate((left_padding, block, right_padding))