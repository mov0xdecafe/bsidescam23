import os

import lief
import lief.logging
import numpy
from torch.utils import data
from tqdm import tqdm

# LIEF will log everything, e.g. when a binary format is not recognised, we don't want that
lief.logging.set_level(lief.logging.LOGGING_LEVEL.CRITICAL)

class FunctionIdentificationDataset(data.Dataset):
    """Generates the dataset for processing by the CNN"""
    def __init__(self, root):
        self._preprocess(root)
        
        
    def _preprocess(self, root: str):
        bins_paths = []
        bins_data = []
        bins_tags = []

        # recursively walk from the root directory and create a list of every file encountered
        all_files = [os.path.join(dirpath, file) 
            for dirpath, _, filenames in os.walk(root) 
            for file in filenames]

        for file_path in tqdm(all_files):
            # attempt to parse the given file, we do not care if it is not a valid binary as that is handled next
            bin = lief.parse(file_path)
            if not bin:
                # parse will not return a valid object on failure, so we can use that to check.
                continue
            data = self._get_code(bin)
            features = self._generate_features(bin)

    def _get_code(self, bin: lief._lief.ELF.Binary):
        """Return the code (.text) section from the indicated binary"""
        return numpy.array(bin.get_section(".text").content, dtype=int)
    
    def _generate_features(self, bin: lief._lief.ELF.Binary):
        """Generate a list of features for the dataset"""
        text_section = bin.get_section(".text")

        # list of function relative start addresses 
        start_addrs = []
        for symbol in bin.symbols:
            if symbol.type == lief.ELF.SYMBOL_TYPES.FUNC and symbol.value != 0:
                relative_address = symbol.value - text_section.virtual_address
                start_addrs.append(relative_address)
                print("Function start @ " + str(relative_address))

